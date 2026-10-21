import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as tfft
import os

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=16, modes2=16):
        """
        Learns mapping in the frequency domain. 
        modes1, modes2: The number of low-frequency modes to keep.
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        # Complex learnable weights for the frequency modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. Compute 2D FFT
        x_ft = tfft.rfft2(x)

        # 2. Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Upper frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # Lower frequencies
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. IFFT back to spatial domain
        x = tfft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FourierBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Expand channels in both frequency and spatial domains
        self.spectral_conv = SpectralConv2d(in_channels, out_channels)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.spatial_conv(x)
        return F.gelu(x1 + x2)
    
class NanoThermalPINN(nn.Module):
    def __init__(self, k_eff=0.99, h_water_m=10e-6, a_m=20e-9):
        super(NanoThermalPINN, self).__init__()
        
        self.k_eff = k_eff
        self.h_water_m = h_water_m
        self.a_m = a_m
        
        self.beta1 = -1.0514e-4
        self.beta2 = -1.5692e-6
        self.beta3 = 5.7538e-9
        self.beta4 = -1.2873e-11

        # U-Net Encoder
        self.enc1 = self._conv_block(1, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        # U-Net Bottleneck
        self.bottleneck = FourierBottleneck(64, 128)

        # U-Net Decoder (align_corners=False prevents macroscopic spatial drift)
        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = self._conv_block(128 + 64, 64)
        
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(64 + 32, 32)
        
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._conv_block(32 + 16, 16)
        
                self.input_skip = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        
        # Final Output Layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, phase_grid):
        e1 = self.enc1(phase_grid)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.upconv3(b)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))
        
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        
        d1_enhanced = d1 + self.input_skip(phase_grid)
        T_pred = torch.abs(self.final_conv(d1_enhanced)) 
        return T_pred

    def physics_informed_loss(self, T_pred, phase_true, pixel_size_m, lambda_probe_m=532e-9):
        # 1. Phase Reconstruction (Optics)
        dn_pred = (self.beta1 * T_pred + self.beta2 * (T_pred**2) + 
                   self.beta3 * (T_pred**3) + self.beta4 * (T_pred**4))
        
        effective_depth = self.a_m * 2.0  
        phase_reconstructed = (2 * torch.pi / lambda_probe_m) * (dn_pred * effective_depth)
        
        # Primary data fidelity in frequency domain (focuses on peaks)
        phase_ft_true = tfft.fft2(phase_true)
        phase_ft_pred = tfft.fft2(phase_reconstructed)
        loss_fourier = torch.mean(torch.abs(phase_ft_true - phase_ft_pred)**2)
        
        loss_phase_l1 = F.l1_loss(phase_reconstructed, phase_true)

        # 2. Isotropic Total Variation (Smooths the ringing, ignores the sharp peak)
        # Calculate gradients
        grad_x = torch.abs(T_pred[:, :, :, 1:] - T_pred[:, :, :, :-1])
        grad_y = torch.abs(T_pred[:, :, 1:, :] - T_pred[:, :, :-1, :])
        
        # Create a dynamic mask: Only smooth the background, allow the peak to remain sharp
        T_norm = T_pred / (torch.max(T_pred) + 1e-8)
        background_mask = (T_norm < 0.2).float() # Only penalize variations in the thermal tail
        
        tv_loss = torch.mean(background_mask[:, :, :, :-1] * grad_x) + \
                  torch.mean(background_mask[:, :, :-1, :] * grad_y)

        # 3. Strict Thermodynamic Positivity Penalty
        # Heavily penalize any values dropping below 0 (the trench artifact)
        loss_positivity = torch.mean(F.relu(-T_pred)**2)

        # 4. Laplacian Physics Constraint
        laplace_kernel = torch.tensor([[[[0., 1., 0.],
                                         [1., -4., 1.],
                                         [0., 1., 0.]]]], device=T_pred.device, dtype=T_pred.dtype)
        laplacian_T = F.conv2d(T_pred, laplace_kernel, padding=1)
        # Heat equation dictates Laplacian <= 0 in a source-free region. 
        loss_laplace = torch.mean(F.relu(laplacian_T)**2)

        # Combined Strategy
        total_loss = (1.0 * loss_fourier +        # Ensures peak amplitudes match exactly
                      1e3 * loss_phase_l1 +       # Spatial alignment
                      50.0 * loss_positivity +    # Kills the negative trench
                      0.5 * tv_loss +             # Smooths the ringing halos
                      0.1 * loss_laplace)         # Enforces physical diffusion
                      
        return total_loss, -laplacian_T
        
    def save_checkpoint(self, file_path="pinn_thermal_optimized.pth"):
        torch.save(self.state_dict(), file_path)
        print(f"[*] PINN model parameters successfully saved to {file_path}")

    def load_checkpoint(self, file_path="pinn_thermal_optimized.pth", device='cpu'):
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=device))
            self.eval() 
            print(f"[*] PINN model parameters successfully loaded from {file_path}")