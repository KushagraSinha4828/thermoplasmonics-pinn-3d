import torch
import torch.nn as nn
import torch.fft as tfft
import torch.nn.functional as F
import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

# =========================================================================
# 1. MATERIALS DATABASE
# =========================================================================
MATERIALS_DB = {
    'water': {'k': 0.6, 'beta': [-1.0514e-4, -1.5692e-6, 5.7538e-9, -1.2873e-11], 'n': 1.33 + 0.0j},
    'glass': {'k': 1.0, 'beta': [1.0e-5], 'n': 1.45 + 0.0j},
    'silica glass': {'k': 1.4, 'beta': [1.0e-5], 'n': 1.45 + 0.0j},
    'ethanol': {'k': 0.17, 'beta': [-4.0e-4], 'n': 1.36 + 0.0j},
    'pdms': {'k': 0.15, 'beta': [-4.5e-4], 'n': 1.40 + 0.0j},
    'ipa': {'k': 0.135, 'beta': [-4.5e-4], 'n': 1.37 + 0.0j},
    'acetone': {'k': 0.16, 'beta': [-5.0e-4], 'n': 1.36 + 0.0j},
    'gold': {'k': 314.0, 'beta': [0.0], 'n': 0.54 + 2.23j},
    'silver': {'k': 429.0, 'beta': [0.0], 'n': 0.05 + 3.42j} 
}

# =========================================================================
# 1. MODULAR INTERACTIVE DASHBOARD
# =========================================================================
def plot_interactive_dashboard(phase_2d_np, T_true_3d_np, T_pred_3d_np, 
                               z_array_um, pixel_size_um, z0_true_um, z0_pred_um, 
                               layers, boundaries_um, scale_val_K, true_max_K,
                               source_3d_np=None):
    Nz, Ny, Nx = T_true_3d_np.shape
    error_map = np.abs(T_pred_3d_np - T_true_3d_np) / (T_true_3d_np.max() + 1e-5) * 100.0

    y = np.linspace(-Ny//2, Ny//2 - 1, Ny) * pixel_size_um
    x = np.linspace(-Nx//2, Nx//2 - 1, Nx) * pixel_size_um
    
    z_init = np.argmin(np.abs(z_array_um - z0_true_um))
    y_init = Ny // 2
    x_init = int(Nx * 0.8)
    pinn_max = np.nanmax(T_pred_3d_np)

    fig = plt.figure(figsize=(24, 11))
    
    # Dynamically adjusts layout if you don't pass the 5th source_3d parameter
    if source_3d_np is not None:
        gs = GridSpec(2, 5, figure=fig)
    else:
        gs = GridSpec(2, 4, figure=fig)
        
    fig.suptitle(f"3D PINN Dynamic Dashboard | Z-Locator: {z0_pred_um:.2f} µm (True: {z0_true_um:.2f} µm)\n"
                 f"Predicted Scale: {scale_val_K:.2f} K (True T_max: {true_max_K:.2f} K)", 
                 fontsize=14, weight='bold')

    ax_phase = fig.add_subplot(gs[0, 0])
    im_phase = ax_phase.imshow(phase_2d_np, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
    ax_phase.set_title("Input 2D Phase Map (QPI)")
    plt.colorbar(im_phase, ax=ax_phase)

    ax_true = fig.add_subplot(gs[0, 1])
    im_true = ax_true.imshow(T_true_3d_np[z_init], extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='hot', vmin=0, vmax=true_max_K)
    ax_true.set_title("Ground Truth Temp")
    plt.colorbar(im_true, ax=ax_true)
    hl_true = ax_true.axhline(y[y_init], color='cyan', linestyle='--')
    vl_true = ax_true.axvline(x[x_init], color='lime', linestyle='--')

    ax_pred = fig.add_subplot(gs[0, 2])
    im_pred = ax_pred.imshow(T_pred_3d_np[z_init], extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='hot', vmin=0, vmax=max(true_max_K, pinn_max))
    ax_pred.set_title(f"PINN Predicted Temp (Max: {pinn_max:.1f} K)")
    plt.colorbar(im_pred, ax=ax_pred)
    hl_pred = ax_pred.axhline(y[y_init], color='cyan', linestyle='--')
    vl_pred = ax_pred.axvline(x[x_init], color='lime', linestyle='--')

    ax_err = fig.add_subplot(gs[0, 3])
    im_err = ax_err.imshow(error_map[z_init], extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='Reds', vmin=0, vmax=10)
    ax_err.set_title("Absolute Error (%)")
    plt.colorbar(im_err, ax=ax_err)

    if source_3d_np is not None:
        ax_src = fig.add_subplot(gs[0, 4])
        src_vmax = np.max(source_3d_np) + 1e-8
        im_src = ax_src.imshow(source_3d_np[z_init], extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='magma', vmin=0, vmax=src_vmax)
        ax_src.set_title("PINN Source Generator", weight='bold', color='purple')
        plt.colorbar(im_src, ax=ax_src)

    ax_x_cut = fig.add_subplot(gs[1, 0])
    line_x_true, = ax_x_cut.plot(x, T_true_3d_np[z_init, y_init, :], 'k-', label='True', linewidth=2)
    line_x_pred, = ax_x_cut.plot(x, T_pred_3d_np[z_init, y_init, :], 'r--', label='PINN', linewidth=2)
    ax_x_cut.set_title("X-Cut Profile")
    ax_x_cut.grid(True)
    ax_x_cut.legend()

    ax_y_cut = fig.add_subplot(gs[1, 1])
    line_y_true, = ax_y_cut.plot(y, T_true_3d_np[z_init, :, x_init], 'k-', label='True')
    line_y_pred, = ax_y_cut.plot(y, T_pred_3d_np[z_init, :, x_init], 'r--', label='PINN')
    ax_y_cut.set_title("Y-Cut Profile")
    ax_y_cut.grid(True)

    ax_z_cut = fig.add_subplot(gs[1, 2:])
    z_min, z_max, b1, b2 = z_array_um[0], z_array_um[-1], boundaries_um[0], boundaries_um[1]
    if b1 > z_min: ax_z_cut.axvspan(z_min, min(b1, z_max), color='lightgray', alpha=0.3, label=layers[0].upper())
    if min(b2, z_max) > max(b1, z_min): ax_z_cut.axvspan(max(b1, z_min), min(b2, z_max), color='lightblue', alpha=0.3, label=layers[1].upper())
    if b2 < z_max: ax_z_cut.axvspan(max(b2, z_min), z_max, color='lightgray', alpha=0.5, label=f"{layers[2].upper()} (Top)")

    line_z_true, = ax_z_cut.plot(z_array_um, T_true_3d_np[:, y_init, x_init], 'k-', label='True', linewidth=2)
    line_z_pred, = ax_z_cut.plot(z_array_um, T_pred_3d_np[:, y_init, x_init], 'r--', label='PINN', linewidth=2)
    ax_z_cut.set_title("Z-Cut Profile (Expanded)")
    ax_z_cut.grid(True)
    
    vl_z = ax_z_cut.axvline(z_array_um[z_init], color='red', linestyle=':', label='Current View Z')
    ax_z_cut.axvline(z0_true_um, color='orange', linestyle='-', linewidth=3.0, alpha=0.6, label='True Source Plane')
    ax_z_cut.axvline(z0_pred_um, color='magenta', linestyle='-.', linewidth=2.5, label='Z-Locator Output')
    ax_z_cut.legend(loc='upper right')

    plt.subplots_adjust(bottom=0.2, hspace=0.3, wspace=0.3)
    axcolor = 'lightgoldenrodyellow'
    ax_z = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_x = plt.axes([0.15, 0.0, 0.65, 0.03], facecolor=axcolor)

    slider_z = Slider(ax_z, 'Z', 0, Nz-1, valinit=z_init, valstep=1)
    slider_y = Slider(ax_y, 'Y', 0, Ny-1, valinit=y_init, valstep=1)
    slider_x = Slider(ax_x, 'X', 0, Nx-1, valinit=x_init, valstep=1)

    def update(val):
        z_idx, y_idx, x_idx = int(slider_z.val), int(slider_y.val), int(slider_x.val)
        z_val = z_array_um[z_idx]
        
        im_true.set_data(T_true_3d_np[z_idx])
        im_pred.set_data(T_pred_3d_np[z_idx])
        im_err.set_data(error_map[z_idx])
        
        if source_3d_np is not None:
            im_src.set_data(source_3d_np[z_idx])
        
        hl_true.set_ydata([y[y_idx], y[y_idx]])
        vl_true.set_xdata([x[x_idx], x[x_idx]])
        hl_pred.set_ydata([y[y_idx], y[y_idx]])
        vl_pred.set_xdata([x[x_idx], x[x_idx]])
        
        line_x_true.set_ydata(T_true_3d_np[z_idx, y_idx, :])
        line_x_pred.set_ydata(T_pred_3d_np[z_idx, y_idx, :])
        line_y_true.set_ydata(T_true_3d_np[z_idx, :, x_idx])
        line_y_pred.set_ydata(T_pred_3d_np[z_idx, :, x_idx])
        line_z_true.set_ydata(T_true_3d_np[:, y_idx, x_idx])
        line_z_pred.set_ydata(T_pred_3d_np[:, y_idx, x_idx])
        vl_z.set_xdata([z_val, z_val])
        
        current_max_x = max(T_true_3d_np[z_idx, y_idx, :].max(), T_pred_3d_np[z_idx, y_idx, :].max())
        current_max_y = max(T_true_3d_np[z_idx, :, x_idx].max(), T_pred_3d_np[z_idx, :, x_idx].max())
        
        ax_x_cut.set_ylim(-0.1, max(current_max_x * 1.1, 1.0))
        ax_y_cut.set_ylim(-0.1, max(current_max_y * 1.1, 1.0))
        
        fig.canvas.draw_idle()

    slider_z.on_changed(update)
    slider_y.on_changed(update)
    slider_x.on_changed(update)
    
    plt.show()

# =========================================================================
# 5. ORIGINAL TQSI SOLVER CODE (UNCHANGED)
# =========================================================================
class Native3DTQSISolver:
    def __init__(self, pixel_size_um, z_array_um, 
                 layer_materials=['glass', 'water', 'glass'],
                 boundaries_um=[0.0, 3.0],
                 a_m=50e-9, probe_nm=400):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        self.dx = pixel_size_um * 1e-6
        self.z_array = torch.tensor(z_array_um, dtype=self.dtype, device=self.device) * 1e-6
        self.dz = (self.z_array[1] - self.z_array[0]) if len(self.z_array) > 1 else 1e-6
        
        self.b1 = boundaries_um[0] * 1e-6
        self.b2 = boundaries_um[1] * 1e-6
        self.a_m = a_m
        self.lambda_probe = probe_nm * 1e-9
        
        self.layer_materials = layer_materials
        mat_bot = MATERIALS_DB[layer_materials[0].lower()]
        mat_mid = MATERIALS_DB[layer_materials[1].lower()]
        mat_top = MATERIALS_DB[layer_materials[2].lower()]
        
        self.k_g = mat_bot['k']
        self.k_w = mat_mid['k']
        self.k_top = mat_top['k']
        
        self.beta_bot = mat_bot['beta']
        self.beta_mid = mat_mid['beta']
        self.beta_top = mat_top['beta']

    @staticmethod
    def _get_riccati(z, N_max):
        n = np.arange(0, N_max + 1)
        jn = spherical_jn(n, z)
        jnp = spherical_jn(n, z, derivative=True)
        return z * jn, jn + z * jnp

    @staticmethod
    def _get_riccati_xi(z, N_max):
        n = np.arange(0, N_max + 1)
        jn = spherical_jn(n, z)
        yn = spherical_yn(n, z)
        jnp = spherical_jn(n, z, derivative=True)
        ynp = spherical_yn(n, z, derivative=True)
        hn = jn + 1j * yn
        hnp = jnp + 1j * ynp
        return z * hn, hn + z * hnp

    @classmethod
    def compute_optical_properties(cls, radius_nm, lambda_pump_nm, n_s, n_m):
        x = (2 * np.pi * np.real(n_m) * radius_nm) / lambda_pump_nm
        x_c = (2 * np.pi * n_m * radius_nm) / lambda_pump_nm
        m = n_s / n_m
        mx = m * x_c
        
        x_real = np.abs(x_c)
        N_max = int(np.round(x_real + 4 * (x_real**(1/3)) + 2))
        
        psi_x, dpsi_x = cls._get_riccati(x_c, N_max)
        psi_mx, dpsi_mx = cls._get_riccati(mx, N_max)
        xi_x, dxi_x = cls._get_riccati_xi(x_c, N_max)
        
        a = np.zeros(N_max + 1, dtype=complex)
        b = np.zeros(N_max + 1, dtype=complex)
        
        for n in range(1, N_max + 1):
            numerator_a = m * psi_mx[n] * dpsi_x[n] - psi_x[n] * dpsi_mx[n]
            denominator_a = m * psi_mx[n] * dxi_x[n] - xi_x[n] * dpsi_mx[n]
            a[n] = numerator_a / denominator_a
            
            numerator_b = psi_mx[n] * dpsi_x[n] - m * psi_x[n] * dpsi_mx[n]
            denominator_b = psi_mx[n] * dxi_x[n] - m * xi_x[n] * dpsi_mx[n]
            b[n] = numerator_b / denominator_b

        n_array = np.arange(1, N_max + 1)
        prefactor = (2 * n_array + 1)
        
        Qe = (2 / x_real**2) * np.sum(prefactor * np.real(a[1:] + b[1:]))
        Qs = (2 / x_real**2) * np.sum(prefactor * (np.abs(a[1:])**2 + np.abs(b[1:])**2))
        Qa = Qe - Qs
        
        geometric_area_nm2 = np.pi * (radius_nm**2)
        sigma_abs_nm2 = Qa * geometric_area_nm2
        return sigma_abs_nm2 * 1e-6 

    def _build_3d_greens_tensor(self, Ny, Nx, z0):
        Ny2, Nx2 = Ny * 2, Nx * 2
        y_1d = (torch.arange(Ny2, device=self.device) - Ny2//2) * self.dx
        x_1d = (torch.arange(Nx2, device=self.device) - Nx2//2) * self.dx
        
        Y, X = torch.meshgrid(y_1d, x_1d, indexing='ij')
        X_3d, Y_3d = X.unsqueeze(0), Y.unsqueeze(0)
        Z_3d = self.z_array.view(-1, 1, 1)
        
        R2_xy = X_3d**2 + Y_3d**2
        def _invR(dz):
            return 1.0 / torch.clamp_min(torch.sqrt(R2_xy + dz**2), self.a_m)

        G_3d = torch.zeros((len(self.z_array), Ny2, Nx2), dtype=self.dtype, device=self.device)
        
        R_top = (self.k_w - self.k_top) / (self.k_w + self.k_top)
        R_bot = (self.k_w - self.k_g) / (self.k_w + self.k_g)
        W_refl = R_top * R_bot
        d = self.b2 - self.b1
        T21 = 2 * self.k_w / (self.k_w + self.k_g)
        T23 = 2 * self.k_w / (self.k_w + self.k_top)

        m_bot = (Z_3d < self.b1).type(self.dtype)
        m_mid = ((Z_3d >= self.b1) & (Z_3d <= self.b2)).type(self.dtype)
        m_top = (Z_3d > self.b2).type(self.dtype)

        G_3d += (1.0 / (4*np.pi*self.k_w)) * _invR(Z_3d - z0) * m_mid
        
        for n in range(3):
            W_n = W_refl**n
            dz_m1 = 2*(n+1)*d + z0 - Z_3d
            dz_m2 = 2*n*d + 2*self.b2 - z0 - Z_3d
            dz_m3 = 2*n*d - 2*self.b1 + z0 + Z_3d
            dz_m4 = 2*(n+1)*d + Z_3d - z0
            G_3d += (1.0 / (4*np.pi*self.k_w)) * W_n * (W_refl * _invR(dz_m1) + R_top * _invR(dz_m2) + R_bot * _invR(dz_m3) + W_refl * _invR(dz_m4)) * m_mid
            
            dz_b1 = 2*n*d + z0 - Z_3d
            dz_b2 = 2*n*d + 2*self.b2 - z0 - Z_3d
            G_3d += (1.0 / (4*np.pi*self.k_w)) * T21 * W_n * (_invR(dz_b1) + R_top * _invR(dz_b2)) * m_bot
            
            dz_t1 = 2*n*d + Z_3d - z0
            dz_t2 = 2*n*d + z0 + Z_3d - 2*self.b1
            G_3d += (1.0 / (4*np.pi*self.k_w)) * T23 * W_n * (_invR(dz_t1) + R_bot * _invR(dz_t2)) * m_top

        return G_3d

    def _eval_dn(self, T_3d, beta_poly):
        dn = torch.zeros_like(T_3d)
        for i, b in enumerate(beta_poly):
            dn += b * (T_3d ** (i+1))
        return dn

    @torch.no_grad()
    def generate_synthetic_phase(self, Ny, Nx, particles, intensity_W_um2, lambda_pump_nm=532.0, z0_um=1.5):
        y = np.linspace(-Ny//2, Ny//2 - 1, Ny) * (self.dx * 1e6)
        x = np.linspace(-Nx//2, Nx//2 - 1, Nx) * (self.dx * 1e6)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        P_true = np.zeros((Ny, Nx))
        n_m = MATERIALS_DB[self.layer_materials[1].lower()]['n']
        
        for p in particles:
            if 'sigma_abs' not in p:
                r_nm = p.get('radius_nm', 50.0)
                mat = p.get('material', 'gold').lower()
                n_s = p.get('n_s', MATERIALS_DB[mat]['n'])
                sigma_abs = self.compute_optical_properties(r_nm, lambda_pump_nm, n_s, n_m)
                p['sigma_abs'] = sigma_abs
            else:
                sigma_abs = p['sigma_abs']
            
            power = sigma_abs * intensity_W_um2
            p['power'] = power
            idx_x = np.argmin(np.abs(x - p['x0']))
            idx_y = np.argmin(np.abs(y - p['y0']))
            P_true[idx_y, idx_x] += power
            
        pad_y, pad_x = Ny // 2, Nx // 2
        P_padded = np.pad(P_true, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
        P_gpu = torch.tensor(P_padded, dtype=self.dtype, device=self.device)
        
        G_3d_spatial = self._build_3d_greens_tensor(Ny, Nx, z0_um * 1e-6)
        G_3d_k = tfft.fft2(tfft.ifftshift(G_3d_spatial, dim=(-2, -1)), dim=(-2, -1))
        
        P_k = tfft.fft2(P_gpu).unsqueeze(0)
        T_3d_true = torch.real(tfft.ifft2(P_k * G_3d_k, dim=(-2, -1)))
        T_3d_clamped = torch.clamp(T_3d_true, min=0.0)
        
        dn_3d = torch.zeros_like(T_3d_clamped)
        
        m_bot = (self.z_array < self.b1).view(-1,1,1).expand_as(dn_3d)
        m_mid = ((self.z_array >= self.b1) & (self.z_array <= self.b2)).view(-1,1,1).expand_as(dn_3d)
        m_top = (self.z_array > self.b2).view(-1,1,1).expand_as(dn_3d)
        
        dn_3d[m_bot] = self._eval_dn(T_3d_clamped, self.beta_bot)[m_bot]
        dn_3d[m_mid] = self._eval_dn(T_3d_clamped, self.beta_mid)[m_mid]
        dn_3d[m_top] = self._eval_dn(T_3d_clamped, self.beta_top)[m_top]
        
        opd_3d = torch.cumsum(dn_3d * self.dz, dim=0)
        phase_3d = opd_3d * (2 * np.pi / self.lambda_probe)
        phase_3d_np = phase_3d[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx].cpu().numpy()
        
        return phase_3d_np, T_3d_true[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx].cpu().numpy(), P_true

    @torch.no_grad()
    def invert_tqsi_3d(self, phase_map_rad, z0_um, alpha=1e-4, iterations=5, damping_g=0.3):
        Ny, Nx = phase_map_rad.shape
        Ny2, Nx2 = Ny * 2, Nx * 2
        
        opd_target = phase_map_rad * (self.lambda_probe / (2 * np.pi))
        opd_target_gpu = torch.tensor(opd_target, dtype=self.dtype, device=self.device)
        
        bg = (opd_target_gpu[0,:].mean() + opd_target_gpu[-1,:].mean() + opd_target_gpu[:,0].mean() + opd_target_gpu[:,-1].mean()) / 4.0
        opd_target_gpu -= bg
        
        z0 = z0_um * 1e-6
        G_3d_spatial = self._build_3d_greens_tensor(Ny, Nx, z0)
        
        beta_vol = torch.zeros_like(self.z_array).view(-1, 1, 1)
        beta_vol[self.z_array < self.b1] = self.beta_bot[0]
        beta_vol[(self.z_array >= self.b1) & (self.z_array <= self.b2)] = self.beta_mid[0]
        beta_vol[self.z_array > self.b2] = self.beta_top[0]
        
        G_2d_opt = torch.sum(G_3d_spatial * beta_vol * self.dz, dim=0) 
        
        G_3d_k = tfft.fft2(tfft.ifftshift(G_3d_spatial, dim=(-2, -1)), dim=(-2, -1))
        G_2d_k = tfft.fft2(tfft.ifftshift(G_2d_opt))
        G_2d_power = torch.abs(G_2d_k)**2 + (alpha * torch.max(torch.abs(G_2d_k)**2))
        
        pad_y, pad_x = Ny // 2, Nx // 2
        opd_padded = F.pad(opd_target_gpu.unsqueeze(0).unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0).squeeze()

        P_current = torch.zeros((Ny2, Nx2), dtype=self.dtype, device=self.device)
        opd_calc = torch.zeros_like(opd_padded)
        
        mask = torch.zeros_like(opd_padded)
        mask[pad_y:pad_y+Ny, pad_x:pad_x+Nx] = 1.0
        
        m_bot = (self.z_array < self.b1).view(-1,1,1)
        m_mid = ((self.z_array >= self.b1) & (self.z_array <= self.b2)).view(-1,1,1)
        m_top = (self.z_array > self.b2).view(-1,1,1)
        
        for i in range(iterations):
            opd_error = mask * (opd_padded - opd_calc)
            P_correction = torch.real(tfft.ifft2(tfft.fft2(opd_error) * torch.conj(G_2d_k) / G_2d_power))
            
            if i == 0:
                P_current = P_correction
            else:
                P_current = P_current + (damping_g * P_correction)
            
            P_current = torch.clamp(P_current, min=0.0)
            P_current = P_current * mask
            
            P_k = tfft.fft2(P_current).unsqueeze(0)
            T_3d = torch.real(tfft.ifft2(P_k * G_3d_k, dim=(-2, -1)))
            
            # [FIX]: Remove max=100.0! Only clamp the floor to 0.0.
            T_3d_clamped = torch.clamp(T_3d, min=0.0) 
            
            dn_3d = torch.zeros_like(T_3d_clamped)
            dn_3d += self._eval_dn(T_3d_clamped, self.beta_bot) * m_bot
            dn_3d += self._eval_dn(T_3d_clamped, self.beta_mid) * m_mid
            dn_3d += self._eval_dn(T_3d_clamped, self.beta_top) * m_top
            
            opd_calc = torch.sum(dn_3d * self.dz, dim=0)

        P_final = P_current[pad_y:pad_y+Ny, pad_x:pad_x+Nx].cpu().numpy()
        T_3d_final = T_3d[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx].cpu().numpy()
        
        return P_final, T_3d_final