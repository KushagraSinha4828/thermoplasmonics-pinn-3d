
import sys
import os

# 1. Give nanothermal.py the map to the custom D: drive installation FIRST
custom_lib_path = r"F:\python_libs"
if custom_lib_path not in sys.path:
    sys.path.insert(0, custom_lib_path)

# 2. Now it is safe to import the heavy libraries
import numpy as np
import scipy.linalg
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, ifftshift, fftshift
from scipy.special import erfc, spherical_jn, spherical_yn
from nptdms import TdmsFile
import tempfile
import os

# --- REDIRECT ALL TEMP FILES TO D: DRIVE ---
os.environ['TMP'] = r'D:\Temp'
os.environ['TEMP'] = r'D:\Temp'
tempfile.tempdir = r'D:\Temp'
os.makedirs(r'D:\Temp', exist_ok=True)

import matplotlib.pyplot as plt
import gc
from scipy.special import erfc
from scipy.ndimage import median_filter, maximum_filter
from sklearn.cluster import DBSCAN
import torch
import torch.fft as tfft
class NanoThermalSolver:
    def __init__(self, coordinates, a, h_water=10.0, T_inf=25.0, k_medium=0.6, k_substrate=1.38):
        """
        Master Nano-Thermal Field Solver unifying Forward Simulation and Non-Linear Inverse Retrieval.
        NOTE: All spatial inputs (coordinates, a, h_water, pixel_size) MUST be in micrometers (μm).
        """
        self.coords_um = np.array(coordinates)
        if self.coords_um.ndim != 2 or self.coords_um.shape[1] != 2:
            raise ValueError("Coordinates must be a list or array of 2D tuples/points in micrometers.")
            
        self.x_p = self.coords_um[:, 0]
        self.y_p = self.coords_um[:, 1]
        
        # Store raw inputs in micrometers (μm)
        self.a_um = a
        self.h_water_um = h_water
        self.T_inf = T_inf
        self.pixel_size_um = None
        
        # Convert to absolute meters for the internal physics engine
        self.a_m = self.a_um * 1e-6
        self.h_water_m = self.h_water_um * 1e-6
        
        self.k_medium = k_medium
        self.k_substrate = k_substrate
        self.k_eff = (k_medium + k_substrate) / 2.0
        
        # --- State Caches & Outputs ---
        self.T_grid = None             # Temp map (K)
        self.Phase_grid = None         # Phase map (nm)
        self.Q_grid = None             # Power array
        self.x_grid = None
        self.y_grid = None
        
        self.temperature_map = None    
        self.power_density = None      
        self.T_max = 0.0               
        self.P_max = 0.0               
        self.Phase_max = 0.0           
        
        # --- Footprint Metrics (in μm) ---
        x_span = self.x_p.max() - self.x_p.min() if len(self.x_p) > 1 else 0
        y_span = self.y_p.max() - self.y_p.min() if len(self.y_p) > 1 else 0
        self.array_span = max(x_span, y_span)
        
        unique_x = np.unique(self.x_p)
        self.spacing_um = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 4.0 * self.a_um
        self.spacing_m = self.spacing_um * 1e-6
        
        self.N_particles = len(self.coords_um)
        self.N_side = int(np.ceil(np.sqrt(self.N_particles)))
        self.max_lim = (self.array_span / 2.0) + (1.2 * self.spacing_um)

        # --- Non-Linear Thermo-Optic Coefficients (Baffou et al. SI) ---
        b1, b2, b3, b4 = -1.0514e-4, -1.5692e-6, 5.7538e-9, -1.2873e-11
        
        # Taylor development at 4th order
        self.beta1 = b1 + 2*self.T_inf*b2 + 3*(self.T_inf**2)*b3 + 4*(self.T_inf**3)*b4
        self.beta2 = b2 + 3*self.T_inf*b3 + 6*(self.T_inf**2)*b4
        self.beta3 = b3 + 4*self.T_inf*b4
        self.beta4 = b4
    # =========================================================
    # OPTICAL ENGINE (Mie Theory for Absorbed Power)
    # =========================================================
    @staticmethod
    def _get_riccati(z, N_max):
        """Constructs Riccati-Bessel psi_n and its derivative for complex arguments."""
        n = np.arange(0, N_max + 1)
        jn = spherical_jn(n, z)
        jnp = spherical_jn(n, z, derivative=True)
        psi = z * jn
        dpsi = jn + z * jnp
        return psi, dpsi

    @staticmethod
    def _get_riccati_xi(z, N_max):
        """Constructs Riccati-Bessel xi_n and its derivative for complex arguments."""
        n = np.arange(0, N_max + 1)
        jn = spherical_jn(n, z)
        yn = spherical_yn(n, z)
        jnp = spherical_jn(n, z, derivative=True)
        ynp = spherical_yn(n, z, derivative=True)
        
        hn = jn + 1j * yn
        hnp = jnp + 1j * ynp
        
        xi = z * hn
        dxi = hn + z * hnp
        return xi, dxi

    def compute_optical_properties(self, lambda_pump_nm, n_s, n_m, theta_array=None):
        """
        Computes exact Mie scattering parameters for the nanoparticle defined in __init__.
        Automatically bridges optical absorption to the thermal solver's expected units.
        
        Parameters:
        - lambda_pump_nm: Excitation wavelength in nm.
        - n_s: Complex refractive index of the nanoparticle (e.g., np.sqrt(epsilon)).
        - n_m: Complex or real refractive index of the surrounding medium.
        - theta_array: Optional array of scattering angles (radians) for phase functions.
        """
        if theta_array is None:
            theta_array = np.array([0.0]) # Dummy array if angular distribution isn't needed

        # The class stores radius 'a' in micrometers. Convert to nm for optical math.
        radius_nm = self.a_um * 1000.0

        # Size parameters
        x = (2 * np.pi * np.real(n_m) * radius_nm) / lambda_pump_nm
        x_c = (2 * np.pi * n_m * radius_nm) / lambda_pump_nm
        m = n_s / n_m
        mx = m * x_c
        
        # Rigorous truncation limit
        x_real = np.abs(x_c)
        N_max = int(np.round(x_real + 4 * (x_real**(1/3)) + 2))
        
        # Compute Riccati-Bessel functions
        psi_x, dpsi_x = self._get_riccati(x_c, N_max)
        psi_mx, dpsi_mx = self._get_riccati(mx, N_max)
        xi_x, dxi_x = self._get_riccati_xi(x_c, N_max)
        
        a = np.zeros(N_max + 1, dtype=complex)
        b = np.zeros(N_max + 1, dtype=complex)
        
        # Calculate a_n and b_n coefficients
        for n in range(1, N_max + 1):
            numerator_a = m * psi_mx[n] * dpsi_x[n] - psi_x[n] * dpsi_mx[n]
            denominator_a = m * psi_mx[n] * dxi_x[n] - xi_x[n] * dpsi_mx[n]
            a[n] = numerator_a / denominator_a
            
            numerator_b = psi_mx[n] * dpsi_x[n] - m * psi_x[n] * dpsi_mx[n]
            denominator_b = psi_mx[n] * dxi_x[n] - m * xi_x[n] * dpsi_mx[n]
            b[n] = numerator_b / denominator_b

        # Efficiencies
        n_array = np.arange(1, N_max + 1)
        prefactor = (2 * n_array + 1)
        
        Qe = (2 / x_real**2) * np.sum(prefactor * np.real(a[1:] + b[1:]))
        Qs = (2 / x_real**2) * np.sum(prefactor * (np.abs(a[1:])**2 + np.abs(b[1:])**2))
        Qa = Qe - Qs
        
        # Physical Cross-Sections (in nm^2)
        geometric_area_nm2 = np.pi * (radius_nm**2)
        sigma_abs_nm2 = Qa * geometric_area_nm2
        sigma_sca_nm2 = Qs * geometric_area_nm2
        
        # Bridge to Thermal Engine: Convert nm^2 to um^2 for self.solve()
        self.sigma_abs_um2 = sigma_abs_nm2 * 1e-6 

        # Angular Phase Functions
        S1 = np.zeros_like(theta_array, dtype=complex)
        S2 = np.zeros_like(theta_array, dtype=complex)
        
        for i, theta in enumerate(theta_array):
            mu = np.cos(theta)
            pi_n = np.zeros(N_max + 1)
            tau_n = np.zeros(N_max + 1)
            
            pi_n[0] = 0.0
            pi_n[1] = 1.0
            
            for n in range(1, N_max + 1):
                if n > 1:
                    pi_n[n] = ((2*n - 1) * mu * pi_n[n-1] - n * pi_n[n-2]) / (n - 1)
                tau_n[n] = n * mu * pi_n[n] - (n + 1) * pi_n[n-1]
                
                angular_factor = (2*n + 1) / (n * (n + 1))
                S1[i] += angular_factor * (a[n] * pi_n[n] + b[n] * tau_n[n])
                S2[i] += angular_factor * (a[n] * tau_n[n] + b[n] * pi_n[n])

        beta_tilde = (np.abs(S1)**2 + np.abs(S2)**2) / (2 * x_real**2 * Qs)

        # Output dictionary
        self.optical_results = {
            "Qa": Qa, "Qs": Qs, "Qe": Qe, 
            "sigma_abs_nm2": sigma_abs_nm2,
            "sigma_sca_nm2": sigma_sca_nm2,
            "sigma_abs_um2": self.sigma_abs_um2,
            "S1": S1, "S2": S2, "beta_tilde": beta_tilde
        }
        
        return self.optical_results
    # =========================================================
    # VOLUMETRIC INTEGRATION ENGINE (UPGRADED)
    # =========================================================
    # =========================================================
    # VOLUMETRIC INTEGRATION ENGINE (NATIVE BASELINE FIX)
    # =========================================================
    def _compute_nonlinear_phase_residual_padded(self, Q_W, dummy_shape, pixel_size_m):
        # Derive shape directly from the input matrix to prevent tuple/int unpacking crashes
        Ny, Nx = Q_W.shape 
        pad_y, pad_x = Ny, Nx
        # SYMMETRIC PADDING: Guarantees origin alignment
        N2_y, N2_x = Ny + pad_y * 2, Nx + pad_x * 2
        
        Q_padded = np.zeros((N2_y, N2_x))
        Q_padded[pad_y : pad_y + Ny, pad_x : pad_x + Nx] = Q_W
        
        Y, X = np.ogrid[:N2_y, :N2_x]
        cy, cx = N2_y // 2, N2_x // 2
        
        z_slices = 15
        dz = self.h_water_m / z_slices
        z_array = np.linspace(dz/2, self.h_water_m - dz/2, z_slices)
        
        phase_residual_padded = np.zeros((N2_y, N2_x))
        Q_k = fft2(Q_padded, workers=-1)
        
        for z in z_array:
            R_z = np.sqrt(((X - cx) * pixel_size_m)**2 + ((Y - cy) * pixel_size_m)**2 + z**2)
            R_z = np.maximum(R_z, self.a_m)  
            
            G_T_z = 1.0 / (4 * np.pi * self.k_eff * R_z)
            G_T_k = fft2(ifftshift(G_T_z), workers=-1)
            
            T_z_padded = np.real(ifft2(Q_k * G_T_k, workers=-1))
            
            dn_z_residual = self.beta2*(T_z_padded**2) + self.beta3*(T_z_padded**3) + self.beta4*(T_z_padded**4)
            phase_residual_padded += dn_z_residual * dz
            
        return phase_residual_padded[pad_y : pad_y + Ny, pad_x : pad_x + Nx]

    # =========================================================
    # FORWARD PROBLEM (Simulation)
    # =========================================================
    def solve(self, intensity, sigma_um2, res_factor=3.0, explicit_pixel_size=None):
        if explicit_pixel_size is not None:
            target_pixel_size = explicit_pixel_size
        else:
            target_pixel_size = self.a_um / res_factor
            
        total_width = 2.0 * self.max_lim
        N_pixels = int(np.ceil(total_width / target_pixel_size))
        if N_pixels % 2 == 0: N_pixels += 1
            
        self.x_grid = np.linspace(-self.max_lim, self.max_lim, N_pixels)
        self.y_grid = np.linspace(-self.max_lim, self.max_lim, N_pixels)
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        self.pixel_size_um = self.x_grid[1] - self.x_grid[0]
        pixel_size_m = self.pixel_size_um * 1e-6
        
        Q_math_W = np.zeros_like(X)
        is_callable = callable(intensity)
        
        for xp, yp in zip(self.x_p, self.y_p):
            x_idx = np.argmin(np.abs(self.x_grid - xp))
            y_idx = np.argmin(np.abs(self.y_grid - yp))
            local_I = intensity(xp, yp) if is_callable else intensity
            Q_math_W[y_idx, x_idx] += (sigma_um2 * local_I) * 1e-6
            
        # --- Rigorous Symmetrical Padded Domain Generation ---
        Ny, Nx = N_pixels, N_pixels
        pad_y, pad_x = Ny, Nx
        N2_y, N2_x = Ny + pad_y * 2, Nx + pad_x * 2
        
        Q_padded = np.zeros((N2_y, N2_x))
        Q_padded[pad_y : pad_y + Ny, pad_x : pad_x + Nx] = Q_math_W
        
        Y_ogrid, X_ogrid = np.ogrid[:N2_y, :N2_x]
        cy, cx = N2_y // 2, N2_x // 2
        R_m = np.maximum(np.sqrt(((X_ogrid - cx) * pixel_size_m)**2 + ((Y_ogrid - cy) * pixel_size_m)**2), self.a_m)
        
        G_T_0 = 1.0 / (4 * np.pi * self.k_eff * R_m)
        G_phase_linear = (self.beta1 / (4 * np.pi * self.k_eff)) * np.arcsinh(self.h_water_m / R_m)
        
        Q_k = fft2(Q_padded, workers=-1)
        G_T_k = fft2(ifftshift(G_T_0), workers=-1)
        G_phase_k = fft2(ifftshift(G_phase_linear), workers=-1)
        
        T_padded = np.real(ifft2(Q_k * G_T_k, workers=-1))
        Phase_linear_padded = np.real(ifft2(Q_k * G_phase_k, workers=-1))
        
        # --- Crop to ROI & Combine ---
        self.Q_grid = Q_math_W
        self.T_grid = T_padded[pad_y : pad_y + Ny, pad_x : pad_x + Nx]
        phase_linear_m = Phase_linear_padded[pad_y : pad_y + Ny, pad_x : pad_x + Nx]
        
        phase_residual_m = self._compute_nonlinear_phase_residual_padded(Q_math_W, N_pixels, pixel_size_m)
        
        phase_total_m = phase_linear_m + phase_residual_m
        self.Phase_grid = -1.0 * phase_total_m * 1e9  
        
        self.T_max = float(np.max(self.T_grid))
        self.Phase_max = float(np.max(self.Phase_grid))
        self.P_max = float(np.max(self.Q_grid)) / (pixel_size_m**2)
        return self
    # =========================================================
    # INVERSE PROBLEM (Experimental Data -> Temperature)
    # =========================================================
    @staticmethod
    def apply_svd_filter(phase_map, keep_modes=12):
        baseline = np.mean(phase_map)
        centered_map = phase_map - baseline
        U, S, Vt = scipy.linalg.svd(centered_map, full_matrices=False)
        S_truncated = np.zeros_like(S)
        S_truncated[:keep_modes] = S[:keep_modes]
        return np.dot(U * S_truncated, Vt) + baseline

    def solve_inverse_linear(self, phase_map, alpha=1e-3):
        """Fast Linear Tikhonov Inversion. Best for max temperatures < 20K.
        
        Parameters
        ----------
        phase_map : ndarray
            The measured Optical Path Difference (OPD) in nanometers. 
            Because heating reduces the refractive index of water, this map 
            must contain physically negative values where heating occurs.
        alpha : float
            Tikhonov regularization parameter.
        """
        if self.pixel_size_um is None:
            raise ValueError("self.pixel_size_um must be set before running inverse solver.")

        pixel_size_m = self.pixel_size_um * 1e-6
        
        # Convert the input OPD (in nanometers) directly to meters.
        # We REMOVED the np.abs() hack here so the physics engine solves 
        # using the true physical phase values.
        raw_phase_meters = phase_map * 1e-9
        
        N_y, N_x = raw_phase_meters.shape
        Y, X = np.ogrid[:N_y, :N_x]
        cy, cx = N_y // 2, N_x // 2
        R_m = np.maximum(np.sqrt(((X - cx) * pixel_size_m)**2 + ((Y - cy) * pixel_size_m)**2), self.a_m)  
        
        G_l = (self.beta1 / (4 * np.pi * self.k_eff)) * np.arcsinh(self.h_water_m / R_m)
        G_T = 1.0 / (4 * np.pi * self.k_eff * R_m)
        
        # Circular convolution is standard for Fourier inversion of periodic/windowed images
        Phase_k = fft2(raw_phase_meters, workers=-1)
        G_l_k = fft2(ifftshift(G_l), workers=-1)
        G_T_k = fft2(ifftshift(G_T), workers=-1)
        
        P_k = Phase_k * np.conj(G_l_k) / (np.abs(G_l_k)**2 + alpha)
        
        # P_k gives power in Watts per pixel. Divide by area for W/m^2.
        p_watts = np.real(ifft2(P_k, workers=-1))
        self.power_density = p_watts / (pixel_size_m**2) 
        self.temperature_map = np.real(ifft2(P_k * G_T_k, workers=-1))
        
        self.T_max = float(np.max(self.temperature_map))
        self.P_max = float(np.max(self.power_density))
        self.Phase_max = float(np.max(phase_map))
        return self.power_density, self.temperature_map

    def solve_inverse_nonlinear(self, phase_map, alpha=1e-3, iterations=10, g=0.3):
        """
        Iterative Non-Linear Inversion. Rigorously computes 3D phase per iteration.
        """
        if self.pixel_size_um is None:
            raise ValueError("self.pixel_size_um must be set before running inverse solver.")

        print(f"Executing Non-Linear Inversion ({iterations} iterations)...")
        pixel_size_m = self.pixel_size_um * 1e-6
        d_l0_m = -1.0 * np.abs(phase_map) * 1e-9
        
        N_y, N_x = d_l0_m.shape
        Y, X = np.ogrid[:N_y, :N_x]
        cy, cx = N_y // 2, N_x // 2
        R_m = np.maximum(np.sqrt(((X - cx) * pixel_size_m)**2 + ((Y - cy) * pixel_size_m)**2), self.a_m)

        # Base kernels
        G_l_m = (self.beta1 / (4*np.pi*self.k_eff)) * np.arcsinh(self.h_water_m / R_m)
        G_T_K = 1.0 / (4*np.pi*self.k_eff * R_m)
        
        G_l_k = fft2(ifftshift(G_l_m), workers=-1)
        G_T_k = fft2(ifftshift(G_T_K), workers=-1)

        p_i_watts = np.zeros_like(d_l0_m)
        d_li_m = np.zeros_like(d_l0_m)
        T_i = np.zeros_like(d_l0_m)
        
        for i in range(iterations):
            error_m = d_l0_m - d_li_m
            delta_p_k = fft2(error_m, workers=-1) * np.conj(G_l_k) / (np.abs(G_l_k)**2 + alpha)
            p_i_watts += g * np.real(ifft2(delta_p_k, workers=-1))
            
            P_i_k = fft2(p_i_watts, workers=-1)
            T_i = np.real(ifft2(P_i_k * G_T_k, workers=-1))
            
            # Compute phase dynamically
            d_li_linear = np.real(ifft2(P_i_k * G_l_k, workers=-1))
            d_li_residual = self._compute_nonlinear_phase_residual_padded(p_i_watts, d_l0_m.shape, pixel_size_m)
            d_li_m = d_li_linear + d_li_residual

        # Convert Watts to W/m^2
        self.power_density = p_i_watts / (pixel_size_m**2)
        self.temperature_map = T_i
        
        self.T_max = float(np.max(self.temperature_map))
        self.P_max = float(np.max(self.power_density))
        self.Phase_max = float(np.max(phase_map)) 
        
        return self.power_density, self.temperature_map

    # =========================================================
    # DATA PROCESSING PIPELINE
    # =========================================================

    @staticmethod
    def _compute_radial_profile(log_mag, center, bin_width=1.0):
        """
        Computes the azimuthally-averaged radial profile of a 2D Fourier magnitude map.
        
        Parameters
        ----------
        log_mag : ndarray (Ny, Nx)
            Log-compressed Fourier magnitude: log10(|F| + 1).
        center : tuple (cx, cy)
            DC center coordinates (column, row).
        bin_width : float
            Radial bin width in pixels. Default 1.0.
            
        Returns
        -------
        radial_mean : 1D array of azimuthally-averaged values per radial bin.
        radial_std  : 1D array of standard deviations per radial bin.
        background_2d : 2D smooth background reconstructed from radial profile.
        dist_map : 2D distance of each pixel from center.
        """
        cx, cy = center
        Ny, Nx = log_mag.shape
        Y, X = np.ogrid[:Ny, :Nx]
        dist_map = np.sqrt(((X - cx).astype(np.float64))**2 + 
                           ((Y - cy).astype(np.float64))**2)
        
        max_radius = np.max(dist_map)
        n_bins = int(np.ceil(max_radius / bin_width))
        bin_indices = np.clip((dist_map / bin_width).astype(int), 0, n_bins - 1)
        
        radial_mean = np.zeros(n_bins)
        radial_std = np.zeros(n_bins)
        
        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                vals = log_mag[mask]
                radial_mean[b] = np.mean(vals)
                radial_std[b] = np.std(vals) if len(vals) > 1 else 0.0
        
        background_2d = radial_mean[bin_indices]
        return radial_mean, radial_std, background_2d, dist_map

    @staticmethod
    def _detect_camera_harmonics(F_mag, dc_center, sideband_center,
                                  sigma_threshold=4.0, dc_exclusion_radius=8,
                                  sb_exclusion_radius=45, spot_radius=3,
                                  dc_cross_width=2, bin_width=1.0):
        """
        Detects camera harmonic noise peaks in Fourier space using radial 
        profile subtraction and adaptive sigma-thresholding in annular rings.
        
        Algorithm:
            1. Compute azimuthal radial profile B(r) of log-magnitude.
            2. Subtract to get residual R(x,y) = log(|F|) - B(r(x,y)).
            3. Per annular ring, compute robust sigma via MAD estimator.
            4. Flag pixels where R > sigma_threshold * local_sigma.
            5. Non-maximum suppression + DBSCAN clustering.
            6. Exclude DC zone, sideband zone, and DC cross lines.
        
        Parameters
        ----------
        F_mag : ndarray
            Fourier magnitude (absolute value of fftshift'd FFT).
        dc_center : tuple (cx, cy)
            DC component center (column, row).
        sideband_center : tuple (cx, cy)
            Primary sideband peak center (column, row).
        sigma_threshold : float
            Detection sensitivity in units of robust sigma. Default 4.0.
        dc_exclusion_radius : float
            Protection radius around DC core only. Default 8px (tight).
            Keep small to allow detection of near-DC camera harmonics.
        sb_exclusion_radius : float
            Protection radius around sideband peak. Default 45px.
        spot_radius : float
            Masking radius for each detected noise spot. Default 6px.
        dc_cross_width : int
            Half-width of the DC cross-line mask (vertical/horizontal). Default 2.
        bin_width : float
            Radial bin width for profile computation. Default 1.0.
            
        Returns
        -------
        notch_mask : ndarray (bool) — True where noise should be removed.
        detected_specs : list of (x, y, radius) tuples.
        diagnostics : dict with 'residual', 'background_2d', 'radial_mean', 
                      'radial_std', 'protected' for visualization.
        """
        Ny, Nx = F_mag.shape
        log_mag = np.log10(F_mag + 1.0)
        
        # 1. Radial profile & background
        radial_mean, radial_std, background_2d, dist_map = \
            NanoThermalSolver._compute_radial_profile(log_mag, dc_center, bin_width)
        
        # 2. Residual = observed - expected background
        residual = log_mag - background_2d
        
        # 3. Build exclusion (protection) zones
        cx_dc, cy_dc = dc_center
        cx_sb, cy_sb = sideband_center
        Y, X = np.ogrid[:Ny, :Nx]
        
        dc_zone = (X - cx_dc)**2 + (Y - cy_dc)**2 <= dc_exclusion_radius**2
        sb_zone = (X - cx_sb)**2 + (Y - cy_sb)**2 <= sb_exclusion_radius**2
        # DC cross lines: vertical & horizontal bright streaks through DC
        dc_cross = ((np.abs(X - cx_dc) <= dc_cross_width) | 
                    (np.abs(Y - cy_dc) <= dc_cross_width))
        protected = dc_zone | sb_zone | dc_cross
        
        # 4. Adaptive sigma per annular ring (using robust MAD estimator)
        bin_indices = np.clip((dist_map / bin_width).astype(int), 0, len(radial_mean) - 1)
        local_sigma_2d = np.full_like(residual, 1e10)
        
        for b in range(len(radial_mean)):
            ring_all = bin_indices == b
            ring_clean = ring_all & ~protected
            if np.any(ring_clean):
                ring_vals = residual[ring_clean]
                med = np.median(ring_vals)
                mad = np.median(np.abs(ring_vals - med))
                robust_sigma = max(1.4826 * mad, 1e-10)  # MAD -> sigma
                local_sigma_2d[ring_all] = robust_sigma
        
        # 5. Threshold: positive spikes exceeding sigma_threshold * local sigma
        candidate_mask = ((residual > sigma_threshold * local_sigma_2d) & 
                          (local_sigma_2d < 1e9) & ~protected)
        
        # 6. Non-maximum suppression: keep only local maxima (5x5 neighborhood)
        local_max = maximum_filter(residual, size=5) == residual
        peaks = candidate_mask & local_max
        
        y_peaks, x_peaks = np.where(peaks)
        
        if len(x_peaks) == 0:
            print("[*] No camera harmonic noise spots detected.")
            diagnostics = {
                'residual': residual, 'background_2d': background_2d,
                'radial_mean': radial_mean, 'radial_std': radial_std,
                'protected': protected
            }
            return np.zeros(F_mag.shape, dtype=bool), [], diagnostics
        
        # 7. DBSCAN clustering to merge multi-pixel detections
        coords_arr = np.column_stack([x_peaks, y_peaks])
        clustering = DBSCAN(eps=spot_radius * 1.5, min_samples=1).fit(coords_arr)
        
        notch_mask = np.zeros(F_mag.shape, dtype=bool)
        detected_specs = []
        
        for label in set(clustering.labels_):
            if label == -1:
                continue
            cluster_pts = coords_arr[clustering.labels_ == label]
            # Pick the brightest pixel in the cluster as center
            brightnesses = [residual[int(pt[1]), int(pt[0])] for pt in cluster_pts]
            best_idx = np.argmax(brightnesses)
            cx_spot, cy_spot = cluster_pts[best_idx]
            
            spot_mask = (X - cx_spot)**2 + (Y - cy_spot)**2 <= spot_radius**2
            notch_mask |= spot_mask
            detected_specs.append((int(cx_spot), int(cy_spot), spot_radius))
        
        print(f"[*] Detected {len(detected_specs)} camera harmonic noise spots.")
        
        diagnostics = {
            'residual': residual, 'background_2d': background_2d,
            'radial_mean': radial_mean, 'radial_std': radial_std,
            'protected': protected
        }
        return notch_mask, detected_specs, diagnostics

    @staticmethod
    def auto_detect_noise_spots(F_mag, sideband_center, dc_center,
                                dc_exclusion_radius=8, sb_exclusion_radius=45,
                                sigma_threshold=4.0, spot_radius=6):
        """
        Public interface for automated camera harmonic noise detection.
        Uses radial profile subtraction + adaptive sigma-thresholding.
        
        Backward compatible: returns (notch_mask, detected_specs).
        """
        notch_mask, detected_specs, _ = NanoThermalSolver._detect_camera_harmonics(
            F_mag, dc_center, sideband_center,
            sigma_threshold=sigma_threshold,
            dc_exclusion_radius=dc_exclusion_radius,
            sb_exclusion_radius=sb_exclusion_radius,
            spot_radius=spot_radius
        )
        return notch_mask, detected_specs

    @staticmethod
    def process_and_clean_tdms(tdms_path, noise_specs=None, sensor_size_px=1024, spot_radius_um=6, 
                               show_pipeline=False, sigma_threshold=4.0, 
                               dc_exclusion_radius_px=8, sb_exclusion_radius_px=45,
                               probe_lambda_nm=457.0):
        """
        Loads TDMS data, extracts complex sidebands, averages frames, and applies 
        Fourier notch filtering to remove camera harmonics.
        
        Parameters
        ----------
        tdms_path : str
            Path to the .tdms file.
        noise_specs : list of tuples, optional
            If None (default), triggers the AUTOMATED radial noise detection.
            If provided, triggers MANUAL mode using the provided (x, y, r) tuples.
        sensor_size : int
            Default masking radius for auto-detected spots. Default 6px.
        show_pipeline : bool
            If True, displays full diagnostic visualization grid.
        sigma_threshold : float
            Auto-detection sensitivity. Default 4.0.
        dc_exclusion_radius : float
            DC core protection radius. Default 8px (tight — only protects
            the bright DC center, allowing near-DC camera spots to be found).
        sb_exclusion_radius : float
            Sideband protection radius. Default 45px.
            
        Returns
        -------
        cleaned_image : ndarray (sensor_size x sensor_size)
            Clean phase map ready for thermodynamic inversion.
        """

        # --- Helper: Off-axis sideband isolation via super-Gaussian window ---
        def isolate_sideband(frame, window_radius=150):
            F_raw = fftshift(fft2(frame, workers=-1))
            cy, cx = sensor_size_px // 2, sensor_size_px // 2
            Y, X = np.ogrid[:sensor_size_px, :sensor_size_px]
            
            # Mask DC to find sideband peak
            dc_mask = (X - cx)**2 + (Y - cy)**2 <= 60**2
            mag = np.abs(F_raw)
            mag[dc_mask] = 0
            mag[:, :cx] = 0  # Search right half only (off-axis geometry)
            
            peak_y, peak_x = np.unravel_index(np.argmax(mag), mag.shape)
            dist = np.sqrt((X - peak_x)**2 + (Y - peak_y)**2)
            window = np.exp(-(dist / window_radius)**4)  # Super-Gaussian
            
            F_isolated = F_raw * window
            return (ifft2(ifftshift(F_isolated), workers=-1), 
                    F_raw, window, F_isolated, (peak_x, peak_y))

        # ============================
        # 1. LOAD TDMS DATA
        # ============================
        print("Loading TDMS Data...")
        temp_dir = r"D:\IISERB\Research\ThermoPlasmonics\Simulation\Temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        tdms_file = TdmsFile.read(tdms_path, memmap_dir=temp_dir)
        raw_1d_data = tdms_file['Rawdata']['Untitled'].data
        
        n_pixels = sensor_size_px * sensor_size_px
        n_frames = raw_1d_data.size // n_pixels
        if n_frames % 2 != 0:
            n_frames -= 1
        
        phase_stack = raw_1d_data[:n_frames * n_pixels].reshape(
            (n_frames, sensor_size_px, sensor_size_px))
        n_pairs = n_frames // 2

        # ============================
        # 2. COMPLEX FIELD ACCUMULATION
        # ============================
        # pump-on x conj(pump-off) — pixel-aligned complex conjugate product
        print(f"Extracting and averaging {n_pairs} frame pairs...")
        complex_accumulator = np.zeros((sensor_size_px, sensor_size_px), dtype=np.complex128)
        
        E_on, F_raw_0, window_0, F_iso_0, sb_center = isolate_sideband(phase_stack[0])
        E_off, _, _, _, _ = isolate_sideband(phase_stack[1])
        complex_accumulator += E_on * np.conj(E_off)
        
        for p in range(1, n_pairs):
            E_on_next, _, _, _, _ = isolate_sideband(phase_stack[2*p])
            E_off_next, _, _, _, _ = isolate_sideband(phase_stack[2*p + 1])
            complex_accumulator += E_on_next * np.conj(E_off_next)
        
        # The holographic sideband extraction isolates the conjugate order, which inverts the phase.
        # Heating water physically reduces the refractive index, so the phase shift must be negative.
        # We multiply by -1.0 here to return the true physical phase shift (in radians).
        raw_average = -1.0 * np.angle(complex_accumulator)
        
        # --- MEMORY OPTIMIZATION ---
        # The TDMS raw data takes >400MB. Save the first frame for plotting,
        # then delete the massive arrays to free RAM for the rest of the pipeline.
        first_frame_plot = phase_stack[0].copy()
        del phase_stack
        del raw_1d_data
        del tdms_file
        gc.collect()

        # ============================
        # 3. FOURIER OF AVERAGED PHASE
        # ============================
        F_avg = fftshift(fft2(raw_average, workers=-1))
        mag_avg = np.abs(F_avg)
        dc_center = (sensor_size_px // 2, sensor_size_px // 2)

        # ============================
        # 4. NOISE DETECTION
        # ============================
        if noise_specs is None:
            # AUTOMATIC: Radial profile subtraction + sigma thresholding
            notch_mask, detected_specs, diagnostics = \
                NanoThermalSolver._detect_camera_harmonics(
                    F_mag=mag_avg,
                    dc_center=dc_center,
                    sideband_center=sb_center,
                    sigma_threshold=sigma_threshold,
                    dc_exclusion_radius=dc_exclusion_radius_px,
                    sb_exclusion_radius=sb_exclusion_radius_px,
                    spot_radius=spot_radius_um
                )
        else:
            # MANUAL: User-specified noise spots (backward compatible)
            Y, X = np.ogrid[:sensor_size_px, :sensor_size_px]
            notch_mask = np.zeros(F_avg.shape, dtype=bool)
            detected_specs = []
            for spec in noise_specs:
                nx, ny = spec[0], spec[1]
                r = spec[2] if len(spec) == 3 else spot_radius_um
                spot_mask = (X - nx)**2 + (Y - ny)**2 <= r**2
                notch_mask |= spot_mask
                detected_specs.append((nx, ny, r))
            diagnostics = None
            print(f"[*] Applied {len(noise_specs)} manual noise spot masks.")

        # ============================
        # 5. APPLY NOTCH FILTER & IFFT
        # ============================
        F_cleaned = F_avg.copy()
        F_cleaned[notch_mask] = 0
        cleaned_image = np.real(ifft2(ifftshift(F_cleaned), workers=-1))

        # ============================
        # 6. DIAGNOSTIC VISUALIZATION
        # ============================
        if show_pipeline:
            cy_plot, cx_plot = dc_center[1], dc_center[0]
            
            if diagnostics is not None:
                # ---- Full 4x3 diagnostic grid for AUTO mode ----
                fig, axes = plt.subplots(4, 3, figsize=(20, 24))
                fig.suptitle("Automated Fourier Cleaning Pipeline "
                             "(Radial Profile Method)", fontsize=20, y=0.95)
                
                # Row 1: Raw data overview
                axes[0,0].imshow(first_frame_plot, cmap='twilight')
                axes[0,0].set_title("1. Raw Frame")
                
                axes[0,1].imshow(np.log10(np.abs(F_raw_0) + 1), cmap='viridis')
                axes[0,1].set_title("2. Full Fourier Magnitude")
                
                axes[0,2].imshow(window_0, cmap='gray')
                axes[0,2].set_title("3. Sideband Window")
                
                # Row 2: Sideband + Radial analysis
                axes[1,0].imshow(np.log10(np.abs(F_iso_0) + 1), cmap='viridis')
                axes[1,0].set_title("4. Isolated Sideband")
                
                im_res = axes[1,1].imshow(diagnostics['residual'], cmap='RdBu_r',
                                          vmin=-0.5, vmax=1.5)
                axes[1,1].set_xlim(cx_plot-200, cx_plot+200)
                axes[1,1].set_ylim(cy_plot+200, cy_plot-200)
                axes[1,1].set_title("5. Radial Residual R(x,y)")
                fig.colorbar(im_res, ax=axes[1,1], fraction=0.046, pad=0.04)
                
                # 1D radial profile plot
                rm = diagnostics['radial_mean']
                rs = diagnostics['radial_std']
                axes[1,2].plot(rm, 'b-', lw=1, alpha=0.8, label='Azimuthal Mean')
                axes[1,2].fill_between(range(len(rm)), rm - rs, rm + rs,
                                       alpha=0.2, color='blue')
                axes[1,2].set_xlabel("Radius (pixels)")
                axes[1,2].set_ylabel(r"$\log_{10}(|F| + 1)$")
                axes[1,2].set_title("6. Radial Profile B(r)")
                axes[1,2].legend()
                axes[1,2].set_xlim(0, 350)
                axes[1,2].grid(True, ls=':', alpha=0.5)
                
                # Row 3: Detection results
                axes[2,0].imshow(np.log10(mag_avg + 1), cmap='viridis')
                for x_s, y_s, r_s in detected_specs:
                    circ = plt.Circle((x_s, y_s), r_s + 2, color='yellow',
                                     fill=False, lw=1.5)
                    axes[2,0].add_patch(circ)
                axes[2,0].contour(diagnostics['protected'], colors='cyan',
                                  linewidths=0.8, alpha=0.5)
                axes[2,0].set_xlim(cx_plot-200, cx_plot+200)
                axes[2,0].set_ylim(cy_plot+200, cy_plot-200)
                axes[2,0].set_title(
                    f"7. Detected Noise ({len(detected_specs)} spots)")
                
                axes[2,1].imshow(np.log10(np.abs(F_cleaned) + 1), cmap='viridis')
                axes[2,1].set_xlim(cx_plot-200, cx_plot+200)
                axes[2,1].set_ylim(cy_plot+200, cy_plot-200)
                axes[2,1].set_title("8. Cleaned Fourier Core")
                
                axes[2,2].imshow(notch_mask.astype(float), cmap='Reds', alpha=0.7)
                axes[2,2].set_xlim(cx_plot-200, cx_plot+200)
                axes[2,2].set_ylim(cy_plot+200, cy_plot-200)
                axes[2,2].set_title("9. Notch Mask")
                
                # Row 4: Phase maps
                im7 = axes[3,0].imshow(raw_average, cmap='turbo')
                axes[3,0].set_title("10. Raw Averaged Phase")
                fig.colorbar(im7, ax=axes[3,0])
                
                im8 = axes[3,1].imshow(cleaned_image, cmap='turbo')
                axes[3,1].set_title("11. Final Cleaned Phase Map")
                fig.colorbar(im8, ax=axes[3,1])
                
                diff = raw_average - cleaned_image
                im_diff = axes[3,2].imshow(diff, cmap='seismic',
                    vmin=-np.std(diff)*3, vmax=np.std(diff)*3)
                axes[3,2].set_title("12. Removed Noise (Raw - Clean)")
                fig.colorbar(im_diff, ax=axes[3,2])
                
            else:
                # ---- Compact 3x3 grid for MANUAL mode ----
                fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                fig.suptitle("Fourier Cleaning Pipeline (Manual Mode)",
                             fontsize=20, y=0.95)
                
                axes[0,0].imshow(first_frame_plot, cmap='twilight')
                axes[0,0].set_title("1. Raw Frame")
                axes[0,1].imshow(np.log10(np.abs(F_raw_0) + 1), cmap='viridis')
                axes[0,1].set_title("2. Full Fourier Magnitude")
                axes[0,2].imshow(window_0, cmap='gray')
                axes[0,2].set_title("3. Sideband Window")
                
                axes[1,0].imshow(np.log10(np.abs(F_iso_0) + 1), cmap='viridis')
                axes[1,0].set_title("4. Isolated Sideband")
                
                axes[1,1].imshow(np.log10(mag_avg + 1), cmap='viridis')
                axes[1,1].contour(notch_mask, colors='magenta', linewidths=1.5)
                axes[1,1].set_xlim(cx_plot-200, cx_plot+200)
                axes[1,1].set_ylim(cy_plot+200, cy_plot-200)
                axes[1,1].set_title(f"5. Noise Spots ({len(detected_specs)})")
                
                axes[1,2].imshow(np.log10(np.abs(F_cleaned) + 1), cmap='viridis')
                axes[1,2].set_xlim(cx_plot-200, cx_plot+200)
                axes[1,2].set_ylim(cy_plot+200, cy_plot-200)
                axes[1,2].set_title("6. Cleaned Fourier Core")
                
                im7 = axes[2,0].imshow(raw_average, cmap='turbo')
                axes[2,0].set_title("7. Raw Averaged Phase")
                fig.colorbar(im7, ax=axes[2,0])
                im8 = axes[2,1].imshow(cleaned_image, cmap='turbo')
                axes[2,1].set_title("8. Final Cleaned Phase Map")
                fig.colorbar(im8, ax=axes[2,1])
                axes[2,2].axis('off')
            
            plt.tight_layout()
            plt.show()

                # Final Conversion: Phase (radians) -> OPD (nanometers)
                # The phase is already negative for heating (enforced above).
        # We convert radians directly to Optical Path Difference in nm.
        cleaned_opd_nm = cleaned_image * (probe_lambda_nm / (2 * np.pi))
        
        return cleaned_opd_nm

    @staticmethod
    def process_and_clean_tdms_auto(tdms_path, sensor_size_px=1024, spot_radius_um=6,
                                     show_pipeline=False, sigma_threshold=4.0):
        """Backward compatibility wrapper. Delegates to process_and_clean_tdms."""
        return NanoThermalSolver.process_and_clean_tdms(
            tdms_path, noise_specs=None, sensor_size_px=sensor_size_px,
            spot_radius_um=spot_radius_um, show_pipeline=show_pipeline,
            sigma_threshold=sigma_threshold
        )
    # =========================================================
    # Transient Evolution T(t)
    # =========================================================
    def plot_transient_evolution(self, q0_watts, alpha_m2_s=1.43e-7, t_max_s=1e-6):
        """
        Tracks and visualizes the time evolution of the nanoparticle temperature.
        """
        t_array = np.linspace(1e-12, t_max_s, 500) 
        r_array = np.linspace(self.a_m, self.a_m * 10, 500) 
        
        def calc_T_finite(r, t):
            sqrt_alpha_t = np.sqrt(alpha_m2_s * t)
            
            if np.isscalar(r) and r <= self.a_m:
                T_max = q0_watts / (4 * np.pi * self.k_eff * self.a_m)
                return T_max * (1.0 - np.exp((alpha_m2_s * t) / self.a_m**2) * erfc(sqrt_alpha_t / self.a_m))
            
            r_safe = np.maximum(r, self.a_m)
            steady = q0_watts / (4 * np.pi * self.k_eff * r_safe)
            
            term1 = erfc((r_safe - self.a_m) / (2 * sqrt_alpha_t))
            term2 = np.exp((r_safe - self.a_m) / self.a_m + (alpha_m2_s * t) / self.a_m**2)
            term3 = erfc((r_safe - self.a_m) / (2 * sqrt_alpha_t) + sqrt_alpha_t / self.a_m)
            
            T_out = steady * (term1 - term2 * term3)
            
            if not np.isscalar(r):
                T_surface = steady[r_safe == self.a_m][0] * (1.0 - np.exp((alpha_m2_s * t) / self.a_m**2) * erfc(sqrt_alpha_t / self.a_m))
                T_out[r <= self.a_m] = T_surface
                
            return T_out
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Nanoparticle Transient Thermal Dynamics", fontsize=16)

        time_snapshots = [t_max_s * 0.01, t_max_s * 0.1, t_max_s * 0.5, t_max_s]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(time_snapshots)))
        
        for t_snap, color in zip(time_snapshots, colors):
            T_r = calc_T_finite(r_array, t_snap)
            axes[0].plot(r_array * 1e9, T_r, color=color, lw=2, 
                         label=f't = {t_snap*1e6:.2f} $\mu$s')
            
        axes[0].set_xlabel("Distance from center, $r$ (nm)", fontsize=12)
        axes[0].set_ylabel("Temperature rise, $\Delta T$ (K)", fontsize=12)
        axes[0].set_title("Spatial Heat Diffusion", fontsize=14)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        T_surface_t = calc_T_finite(self.a_m, t_array)
        
        axes[1].plot(t_array * 1e6, T_surface_t, 'r-', lw=2)
        axes[1].axhline(y=T_surface_t[-1], color='k', linestyle=':', 
                        label=f'Steady State ({T_surface_t[-1]:.2f} K)')
        
        axes[1].set_xlabel("Time, $t$ ($\mu$s)", fontsize=12)
        axes[1].set_ylabel("Surface Temperature rise, $\Delta T(a)$ (K)", fontsize=12)
        axes[1].set_title(f"Temporal Evolution at Surface ($r = {self.a_um * 1e3:.0f}$ nm)", fontsize=14)
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    def plot_2d_density_maps(self, q0_watts, alpha_m2_s=1.43e-7, window_um=1.0):
        """
        Generates 2D spatiotemporal temperature density maps using imshow.
        """
        N_pixels = 200
        limit_m = (window_um / 2.0) * 1e-6
        grid_1d = np.linspace(-limit_m, limit_m, N_pixels)
        X, Y = np.meshgrid(grid_1d, grid_1d)
        
        R_m = np.sqrt(X**2 + Y**2)
        R_m = np.maximum(R_m, self.a_m) 
        
        times_s = [1e-9, 20e-9, 1e-6] 
        titles = ["t = 1 ns", "t = 20 ns", "Steady State (1 $\mu$s)"]
        
        T_max_global = q0_watts / (4 * np.pi * self.k_eff * self.a_m)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Spatiotemporal Evolution of the Thermal Field", fontsize=16, y=1.02)
        
        for i, (t, title) in enumerate(zip(times_s, titles)):
            steady_state = q0_watts / (4 * np.pi * self.k_eff * R_m)
            diffusion_term = erfc(R_m / (2 * np.sqrt(alpha_m2_s * t)))
            T_map = steady_state * diffusion_term
            
            im = axes[i].imshow(T_map, 
                                extent=[-window_um/2*1e3, window_um/2*1e3, 
                                        window_um/2*1e3, -window_um/2*1e3], 
                                cmap='hot', 
                                vmin=0, vmax=T_max_global)
            
            axes[i].set_title(title, fontsize=14)
            axes[i].set_xlabel("x (nm)")
            if i == 0:
                axes[i].set_ylabel("y (nm)")
            
            particle = plt.Circle((0, 0), self.a_um * 1e3, color='white', fill=False, ls=':', lw=1.5)
            axes[i].add_patch(particle)
            
            fig.colorbar(im, ax=axes[i], label="$\Delta T$ (K)", fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def compute_collective_transient(self, intensity, sigma_um2, times_s, alpha_m2_s=1.43e-7, window_um=2.0, N_pixels=400):
        """
        Computes the collective transient temperature evolution for N finite-sized nanoparticles.
        Returns the spatial grid and a list of T(r,t) density maps for each requested time.
        """
        limit_m = (window_um / 2.0) * 1e-6
        grid_1d = np.linspace(-limit_m, limit_m, N_pixels)
        X, Y = np.meshgrid(grid_1d, grid_1d)
        
        S_watts = np.zeros((N_pixels, N_pixels))
        is_callable = callable(intensity)
        
        for (xp_um, yp_um) in self.coords_um:
            xp_m, yp_m = xp_um * 1e-6, yp_um * 1e-6
            if abs(xp_m) <= limit_m and abs(yp_m) <= limit_m:
                col = np.argmin(np.abs(grid_1d - xp_m))
                row = np.argmin(np.abs(grid_1d - yp_m))
                
                local_I = intensity(xp_um, yp_um) if is_callable else intensity
                q_p_watts = (local_I * sigma_um2) * 1e-6  
                S_watts[row, col] += q_p_watts

        R_raw = np.sqrt(X**2 + Y**2)
        R_safe = np.maximum(R_raw, self.a_m)
        T_maps_over_time = []
        
        for t in times_s:
            if t <= 0:
                T_maps_over_time.append(np.zeros_like(X))
                continue
                
            sqrt_alpha_t = np.sqrt(alpha_m2_s * t)
            
            steady_unit = 1.0 / (4 * np.pi * self.k_eff * R_safe)
            term1 = erfc((R_safe - self.a_m) / (2 * sqrt_alpha_t))
            term2 = np.exp((R_safe - self.a_m) / self.a_m + (alpha_m2_s * t) / self.a_m**2)
            term3 = erfc((R_safe - self.a_m) / (2 * sqrt_alpha_t) + sqrt_alpha_t / self.a_m)
            
            K_unit = steady_unit * (term1 - term2 * term3)
            
            T_surface_unit = (1.0 / (4 * np.pi * self.k_eff * self.a_m)) * \
                             (1.0 - np.exp((alpha_m2_s * t) / self.a_m**2) * erfc(sqrt_alpha_t / self.a_m))
            K_unit[R_raw <= self.a_m] = T_surface_unit
            
            T_total = fftconvolve(S_watts, K_unit, mode='same')
            T_maps_over_time.append(T_total)
            
        return grid_1d, T_maps_over_time

    def plot_collective_transient(self, intensity, sigma_um2, times_s, alpha_m2_s=1.43e-7, window_um=2.0, N_pixels=400):
        """
        Executes the collective transient computation and plots the resulting density maps.
        """
        print(f"Computing transient convolution for {len(self.coords_um)} finite particles...")
        grid_1d, T_maps = self.compute_collective_transient(
            intensity, sigma_um2, times_s, alpha_m2_s, window_um, N_pixels
        )
        
        T_max_global = np.max(T_maps[-1])
        
        fig, axes = plt.subplots(1, len(times_s), figsize=(6 * len(times_s), 5))
        if len(times_s) == 1: axes = [axes] # Handle single-time case
            
        fig.suptitle(f"Collective Spatiotemporal Evolution ({len(self.coords_um)} Particles)", fontsize=16, y=1.02)
        
        for i, (t, T_map) in enumerate(zip(times_s, T_maps)):
            title = f"t = {t*1e9:.1f} ns" if t < 1e-6 else f"t = {t*1e6:.2f} $\mu$s"
            
            im = axes[i].imshow(T_map, 
                                extent=[-window_um/2*1e3, window_um/2*1e3, 
                                        window_um/2*1e3, -window_um/2*1e3], 
                                cmap='turbo', vmin=0, vmax=T_max_global)
            
            axes[i].set_title(title, fontsize=14)
            axes[i].set_xlabel("x (nm)")
            if i == 0: axes[i].set_ylabel("y (nm)")
            
            for (xp_um, yp_um) in self.coords_um:
                if abs(xp_um) <= window_um/2 and abs(yp_um) <= window_um/2:
                    particle = plt.Circle((xp_um * 1e3, yp_um * 1e3), self.a_um * 1e3, 
                                          color='white', fill=False, ls=':', lw=1.0, alpha=0.8)
                    axes[i].add_patch(particle)
                    
            fig.colorbar(im, ax=axes[i], label="$\Delta T$ (K)", fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    def print_report(self):
        """Prints a comprehensive summary of all physics parameters with exact units."""
        print("\n" + "="*45)
        print(" NANO-THERMAL SOLVER: PHYSICS REPORT")
        print("="*45)
        print("Setup Parameters:")
        print(f"  - Particle Radius (a):   {self.a_um * 1e3:.1f} nm")
        print(f"  - Array Spacing:         {self.spacing_um * 1e3:.1f} nm")
        if self.pixel_size_um is not None:
            print(f"  - Pixel Size:            {self.pixel_size_um * 1e3:.1f} nm")
        print(f"  - Chamber Height (h):    {self.h_water_um:.1f} μm")
        print(f"  - Ambient Temp (T_inf):  {self.T_inf:.1f} °C")
        print(f"  - Effective k (k_eff):   {self.k_eff:.3f} W/(m·K)")
        print("-" * 45)
        print("Extracted Maximums:")
        print(f"  - Peak Temperature (ΔT): {self.T_max:.2f} K")
        print(f"  - Peak Power Density:    {self.P_max:.2e} W/m²")
        print(f"                           ({self.P_max * 1e-6:.2e} μW/μm²)")
        if self.Phase_max != 0:
            print(f"  - Peak Phase OPD:        {self.Phase_max:.2f} nm")
        print("="*45 + "\n")


    # =========================================================
    # 3D VOLUMETRIC EXTENSIONS (PyTorch GPU Integrated)
    # =========================================================
    @staticmethod
    def calculate_error_map(true_matrix, retrieved_matrix):
        """
        Computes the relative error percentage map between two matrices.
        Divides by the maximum absolute value of the true matrix to avoid 
        asymptotes near absolute zero limits.
        """
        epsilon = 1e-15
        max_val = np.max(np.abs(true_matrix)) + epsilon
        error_map = (np.abs(true_matrix - retrieved_matrix) / max_val) * 100
        mean_error = np.mean(error_map)
        
        return error_map, mean_error
    
    @staticmethod
    def _apply_cauchy_dispersion(coefficient, lambda_um, lambda_ref_um=0.532):
        """
        Applies a first-order Cauchy dispersion scaling to the thermo-optic coefficients.
        Assumes the baseline beta coefficients are calibrated for a 532 nm (0.532 um) probe.
        """
        # Empirical Cauchy parameters for liquid water in the visible regime
        A = 1.311
        B = 0.0053
        
        scale_ref = A + (B / lambda_ref_um**2)
        scale_target = A + (B / lambda_um**2)
        
        return coefficient * (scale_target / scale_ref)
    
    @staticmethod
    def _crop_1x(array):
        """Crops a 2x padded array back to its original 1x dimensions."""
        Ny2, Nx2 = array.shape
        Ny, Nx = Ny2 // 2, Nx2 // 2
        return array[Ny//2 : Ny//2 + Ny, Nx//2 : Nx//2 + Nx]

    def _compute_3d_kernels_padded(self, Ny, Nx, pixel_size_m, dz_m, lambda_probe_nm, device):
        """
        Calculates the accumulated optical path length kernel G_l over the z-volume.
        """
        Ny2, Nx2 = Ny * 2, Nx * 2
        y = torch.arange(Ny2, device=device)
        x = torch.arange(Nx2, device=device)
        Y_gpu, X_gpu = torch.meshgrid(y, x, indexing='ij')
        cy, cx = Ny2 // 2, Nx2 // 2

        z_array = torch.arange(dz_m / 2, self.h_water_m, dz_m, device=device)
        beta1_adj = self._apply_cauchy_dispersion(self.beta1, lambda_probe_nm * 1e-3)
        
        G_l_gpu = torch.zeros((Ny2, Nx2), dtype=torch.float64, device=device)

        for z in z_array:
            R_z = torch.sqrt(((X_gpu - cx) * pixel_size_m)**2 + ((Y_gpu - cy) * pixel_size_m)**2 + z**2)
            R_z = torch.maximum(R_z, torch.tensor(self.a_m, device=device))
            G_T_z = 1.0 / (4 * torch.pi * self.k_eff * R_z)
            G_l_gpu += beta1_adj * G_T_z * dz_m
            
        return G_l_gpu
    # =========================================================
    # 3D VOLUMETRIC EXTENSIONS (PyTorch GPU Integrated)
    # =========================================================
    @torch.no_grad()
    def solve_forward_3d(self, p_watts, pixel_size_um, dz_um=0.25, lambda_probe_nm=532):
        pixel_size_m = pixel_size_um * 1e-6
        dz_m = dz_um * 1e-6
        lambda_probe_m = lambda_probe_nm * 1e-9
        
        Ny, Nx = p_watts.shape
        pad_y, pad_x = Ny, Nx
        
        # FORWARD PADDING: Constant 0 is physically correct because power is strictly localized.
        p_padded_cpu = np.pad(p_watts, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        p_gpu = torch.tensor(p_padded_cpu, dtype=torch.float64, device=device)
        P_k_gpu = tfft.fft2(p_gpu)
        
        Ny2, Nx2 = p_padded_cpu.shape
        y = torch.arange(Ny2, device=device)
        x = torch.arange(Nx2, device=device)
        Y_gpu, X_gpu = torch.meshgrid(y, x, indexing='ij')
        cy, cx = Ny2 // 2, Nx2 // 2
        
        R_m = torch.sqrt(((X_gpu - cx) * pixel_size_m)**2 + ((Y_gpu - cy) * pixel_size_m)**2)
        R_m = torch.maximum(R_m, torch.tensor(self.a_m, device=device))
        
        beta1_adj = self._apply_cauchy_dispersion(self.beta1, lambda_probe_nm * 1e-3)
        beta2_adj = self._apply_cauchy_dispersion(self.beta2, lambda_probe_nm * 1e-3)
        beta3_adj = self._apply_cauchy_dispersion(self.beta3, lambda_probe_nm * 1e-3)
        beta4_adj = self._apply_cauchy_dispersion(self.beta4, lambda_probe_nm * 1e-3)
        
        # 1. Exact Linear Phase
        G_l_m = (beta1_adj / (4 * torch.pi * self.k_eff)) * torch.arcsinh(torch.tensor(self.h_water_m, device=device) / R_m)
        G_l_k = tfft.fft2(tfft.ifftshift(G_l_m))
        opd_linear_padded = torch.real(tfft.ifft2(P_k_gpu * G_l_k))
        
        # 2. Non-Linear Residuals
        opd_residual_padded = torch.zeros((Ny2, Nx2), dtype=torch.float64, device=device)
        z_array = torch.arange(dz_m / 2, self.h_water_m, dz_m, device=device)
        
        for z in z_array:
            R_z = torch.sqrt(((X_gpu - cx) * pixel_size_m)**2 + ((Y_gpu - cy) * pixel_size_m)**2 + z**2)
            R_z = torch.maximum(R_z, torch.tensor(self.a_m, device=device))
            
            G_T_z = 1.0 / (4 * torch.pi * self.k_eff * R_z)
            G_T_k = tfft.fft2(tfft.ifftshift(G_T_z))
            
            T_z_padded = torch.real(tfft.ifft2(P_k_gpu * G_T_k))
            
            dn_z_res = beta2_adj * (T_z_padded**2) + beta3_adj * (T_z_padded**3) + beta4_adj * (T_z_padded**4)
            opd_residual_padded += dn_z_res * dz_m
            
        opd_padded_final = (opd_linear_padded + opd_residual_padded).cpu().numpy()
        opd_final = opd_padded_final[pad_y : pad_y + Ny, pad_x : pad_x + Nx]
        phase_map_rad = (2 * np.pi / lambda_probe_m) * opd_final
        
        return phase_map_rad, opd_final

    @torch.no_grad()
    def solve_inverse_3d(self, phase_map_rad, pixel_size_um, z0_um=0.0, dz_um=0.25, lambda_probe_nm=532, alpha=1e-12, iterations=3, fp32=True, dc_tune=3.0):
        """
        PINN-Optimized Inverse Engine.
        Includes `dc_tune` (default 0.5) to act as a Y-axis translation joystick,
        perfectly balancing the Fourier DC-energy volume to align phase baselines.
        """
        import torch.nn.functional as F
        dtype = torch.float32 if fp32 else torch.float64
        pixel_size_m = pixel_size_um * 1e-6
        dz_m = dz_um * 1e-6
        lambda_probe_m = lambda_probe_nm * 1e-9
        
        Ny, Nx = phase_map_rad.shape
        pad_y, pad_x = Ny, Nx
        Ny2, Nx2 = Ny + 2*pad_y, Nx + 2*pad_x
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # =====================================================================
        # TUNABLE BASELINE ALIGNMENT
        # =====================================================================
        opd_m = -1.0 * np.abs(phase_map_rad) * (lambda_probe_m / (2 * np.pi))
        edge_val = (opd_m[0,:].mean() + opd_m[-1,:].mean() + opd_m[:,0].mean() + opd_m[:,-1].mean()) / 4.0
        
        # dc_tune=1.0 shifts curve down. dc_tune=0.0 shifts curve up.
        opd_target_gpu = torch.tensor(opd_m - (edge_val * dc_tune), dtype=dtype, device=device)
        
        # Precompute kernels
        beta1, beta2, beta3, beta4 = [self._apply_cauchy_dispersion(b, lambda_probe_nm * 1e-3) for b in [self.beta1, self.beta2, self.beta3, self.beta4]]
        
        y_pad, x_pad = torch.arange(Ny2, device=device), torch.arange(Nx2, device=device)
        Y_p, X_p = torch.meshgrid(y_pad, x_pad, indexing='ij')
        R_p = torch.maximum(torch.sqrt(((X_p - Ny2/2)*pixel_size_m)**2 + ((Y_p - Ny2/2)*pixel_size_m)**2), torch.tensor(self.a_m, device=device))
        
        G_l_k = tfft.fft2(tfft.ifftshift((beta1 / (4 * torch.pi * self.k_eff)) * torch.arcsinh(self.h_water_m / R_p).to(dtype)))
        G_power = torch.abs(G_l_k)**2 + (alpha * torch.max(torch.abs(G_l_k)**2))
        
        # Rectangular Mask
        y_r, x_r = torch.arange(Ny, device=device), torch.arange(Nx, device=device)
        Y_r, X_r = torch.meshgrid(y_r, x_r, indexing='ij')
        my, mx = Ny * 0.15, Nx * 0.15
        mask = (0.5*(1+torch.cos(torch.pi*torch.clamp(torch.abs(Y_r - Ny/2) - (Ny/2 - my), 0.0, my)/my))) * \
               (0.5*(1+torch.cos(torch.pi*torch.clamp(torch.abs(X_r - Nx/2) - (Nx/2 - mx), 0.0, mx)/mx)))

        # Solver Loop
        opd_nl = torch.zeros((Ny, Nx), dtype=dtype, device=device)
        p_clean = torch.zeros((Ny, Nx), dtype=dtype, device=device)
        
        for i in range(iterations):
            padded = F.pad( (opd_target_gpu - opd_nl).unsqueeze(0).unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode='replicate').squeeze()
            p_clean = torch.real(tfft.ifft2(tfft.fft2(padded) * torch.conj(G_l_k) / G_power))[pad_y:pad_y+Ny, pad_x:pad_x+Nx] * mask
            
            if i == iterations - 1: break
            
            # Streaming Z-loop
            p_padded = torch.zeros((Ny2, Nx2), dtype=dtype, device=device)
            p_padded[pad_y:pad_y+Ny, pad_x:pad_x+Nx] = p_clean
            P_k = tfft.fft2(p_padded)
            opd_nl = torch.zeros((Ny, Nx), dtype=dtype, device=device)
            for z in torch.arange(dz_m/2, self.h_water_m, dz_m, device=device):
                T_z = torch.real(tfft.ifft2(P_k * tfft.fft2(tfft.ifftshift((1.0/(4*torch.pi*self.k_eff*torch.maximum(torch.sqrt(((X_p - Ny2/2)*pixel_size_m)**2 + ((Y_p - Ny2/2)*pixel_size_m)**2 + z**2), torch.tensor(self.a_m, device=device)))).to(dtype)))))
                opd_nl += (beta2*T_z**2 + beta3*T_z**3 + beta4*T_z**4)[pad_y:pad_y+Ny, pad_x:pad_x+Nx] * dz_m
                
        return p_clean.cpu().numpy() / (pixel_size_m**2), torch.real(tfft.ifft2(tfft.fft2(p_padded) * tfft.fft2(tfft.ifftshift((1.0/(4*torch.pi*self.k_eff*torch.maximum(torch.sqrt(((X_p - Ny2/2)*pixel_size_m)**2 + ((Y_p - Ny2/2)*pixel_size_m)**2 + z0_um**2), torch.tensor(self.a_m, device=device)))).to(dtype)))))[pad_y:pad_y+Ny, pad_x:pad_x+Nx].cpu().numpy()
 