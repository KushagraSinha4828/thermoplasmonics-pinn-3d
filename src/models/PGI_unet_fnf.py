import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. MAGNITUDE-AWARE DUAL-BRANCH UNET 
# ==========================================
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.gn1 = nn.GroupNorm(4, out_channels) 
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.gn2 = nn.GroupNorm(4, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)

class SpatialExtrusionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, f_2d, f_3d):
        B, C, Z, Y, X = f_3d.shape
        f_2d_extruded = f_2d.unsqueeze(2).expand(-1, -1, Z, -1, -1)
        
        gate = self.gate_conv(torch.cat([f_2d_extruded, f_3d], dim=1))
        f_2d_modulated = f_2d_extruded * gate
        
        fused = torch.cat([f_2d_modulated, f_3d], dim=1)
        return self.fuse_conv(fused)

class PhysicsGuidedInversionUNet(nn.Module):
    def __init__(self, bf=8): 
        super().__init__()
        self.enc_2d = nn.Sequential(
            nn.Conv2d(1, bf, 3, padding=1, padding_mode='replicate'), nn.GroupNorm(4, bf), nn.SiLU(inplace=True),
            nn.Conv2d(bf, bf*2, 3, padding=1, padding_mode='replicate'), nn.GroupNorm(4, bf*2), nn.SiLU(inplace=True)
        )
        self.enc_3d_prior = nn.Sequential(
            nn.Conv3d(3, bf*2, 3, padding=1, padding_mode='replicate'), nn.GroupNorm(4, bf*2), nn.SiLU(inplace=True)
        )
        self.fusion = SpatialExtrusionFusion(channels=bf*2)
        
        self.init_3d = nn.Sequential(
            nn.Conv3d(bf*2, bf*2, 3, padding=1, padding_mode='replicate'), 
            nn.GroupNorm(4, bf*2), 
            nn.SiLU(inplace=True)
        )
        
        self.down1_pool = nn.MaxPool3d(2)
        self.down1_conv = ResBlock3D(bf*2, bf*4)
        
        self.down2_pool = nn.MaxPool3d(2)
        self.down2_conv = ResBlock3D(bf*4, bf*8)
        
        self.mag_pool = nn.AdaptiveAvgPool3d(1)
        
        self.mag_fc = nn.Sequential(
            nn.Linear((bf*8) + 2, bf*4),
            nn.GroupNorm(4, bf*4), 
            nn.SiLU(inplace=True),
            nn.Linear(bf*4, 1)
        )
        nn.init.zeros_(self.mag_fc[-1].weight)
        nn.init.zeros_(self.mag_fc[-1].bias)
        
        self.dec1 = ResBlock3D(bf*12, bf*4) 
        self.dec2 = ResBlock3D(bf*6, bf*2)  
        
        self.final_conv = nn.Conv3d(bf*2, 1, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.constant_(self.final_conv.bias, -4.0) 

    def forward(self, phase_2d, T_inv_3d, k_3d, beta_3d):
        B = phase_2d.size(0)
        
        phase_max = torch.amax(torch.abs(phase_2d), dim=(2,3), keepdim=True).clamp(min=1e-5)
        t_inv_max = torch.amax(torch.abs(T_inv_3d), dim=(2,3,4), keepdim=True).clamp(min=1e-5)
        
        f_2d = self.enc_2d(phase_2d / phase_max)
        f_3d_prior = self.enc_3d_prior(torch.cat([T_inv_3d / t_inv_max, k_3d, beta_3d * 1e4], dim=1))
        f_vol = self.fusion(f_2d, f_3d_prior)
        
        e1 = self.init_3d(f_vol)
        e2 = self.down1_conv(self.down1_pool(e1))
        b = self.down2_conv(self.down2_pool(e2))
        
        b_flat = self.mag_pool(b).view(B, -1)
        log_p_max = torch.log(phase_max.view(B, 1))
        log_t_max = torch.log(t_inv_max.view(B, 1))
        
        mag_input = torch.cat([b_flat, log_p_max, log_t_max], dim=1)
        log_scale = self.mag_fc(mag_input).view(B, 1, 1, 1, 1)
        
        dynamic_peak = torch.exp(log_scale + 4.605).clamp(min=1.0, max=5000.0)
        
        u1 = F.interpolate(b, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        
        u2 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        
        logits = self.final_conv(d2)
        spatial_map_raw = torch.sigmoid(logits) 
        
        spatial_max = spatial_map_raw.amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-5)
        spatial_map_norm = spatial_map_raw / spatial_max
        
        return (spatial_map_norm * dynamic_peak), dynamic_peak


# ==========================================
# 2. STRICTLY ISOLATED LOSS FUNCTION 
# ==========================================
def gaussian_window_3d(window_size, sigma):
    """Creates a 3D Gaussian kernel for SSIM."""
    coords = torch.arange(window_size, dtype=torch.float) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    g1d = g.view(-1, 1, 1)
    g3d = g1d * g1d.view(1, -1, 1) * g1d.view(1, 1, -1)
    return g3d.view(1, 1, window_size, window_size, window_size)

class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=5, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        window = gaussian_window_3d(window_size, sigma)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        # Assuming inputs are [B, 1, Z, Y, X] and normalized to [0, 1]
        mu1 = F.conv3d(img1, self.window, padding=self.window_size//2, groups=1)
        mu2 = F.conv3d(img2, self.window, padding=self.window_size//2, groups=1)
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
        
        sigma1_sq = F.conv3d(img1*img1, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv3d(img2*img2, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv3d(img1*img2, self.window, padding=self.window_size//2) - mu1_mu2
        
        C1, C2 = 0.01**2, 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()

class AsyncLagrangianLoss(nn.Module):
    def __init__(self, dx_um, dz_um):
        super().__init__()
        self.dz_ratio = (dx_um / dz_um)
        self.dz_ratio_sq = self.dz_ratio ** 2
        
        weight = torch.zeros(1, 1, 3, 3, 3)
        weight[0, 0, 1, 1, 1] = -2.0 * (2.0 + self.dz_ratio_sq)
        weight[0, 0, 1, 1, 0] = 1.0; weight[0, 0, 1, 1, 2] = 1.0
        weight[0, 0, 1, 0, 1] = 1.0; weight[0, 0, 1, 2, 1] = 1.0
        weight[0, 0, 0, 1, 1] = self.dz_ratio_sq; weight[0, 0, 2, 1, 1] = self.dz_ratio_sq
        self.register_buffer('lap_kernel', weight)
        
        # Instantiate Structural Loss
        self.ssim_loss = SSIM3DLoss(window_size=5, sigma=1.5)

    def forward(self, T_pred, dynamic_peak, T_true, k_3d, source_mask):
        T_max = T_true.amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
        P_max = T_pred.amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
        
        # LOG-PEAK LOSS
        loss_peak = F.mse_loss(torch.log(dynamic_peak), torch.log(T_max)) * 30.0
        
        t_norm = (T_true / T_max).clamp(min=0.0)
        p_norm = (T_pred / P_max.detach()).clamp(min=0.0)
        
        src_mask_f = (source_mask > 0).float()
        bg_mask_f = (source_mask == 0).float()
        
        # 1. Base Intensity Losses
        mse_diff = F.mse_loss(p_norm, t_norm, reduction='none')
        log_diff = F.l1_loss(torch.log1p(10.0 * p_norm), torch.log1p(10.0 * t_norm), reduction='none')
        shape_diff = mse_diff + (0.5 * log_diff)
        
        src_sum = torch.sum(src_mask_f).clamp(min=1.0)
        bg_sum = torch.sum(bg_mask_f).clamp(min=1.0)
        loss_src_shape = (torch.sum(shape_diff * src_mask_f) / src_sum) * 10.0
        loss_bg_shape = (torch.sum(shape_diff * bg_mask_f) / bg_sum) * 10.0

        # 2. Structural Similarity Loss (Anti-Blur)
        # We apply this directly to the normalized shapes to focus on structure, independent of magnitudes
        loss_structure = self.ssim_loss(p_norm, t_norm) * 20.0

        p_z_profile = p_norm.amax(dim=(3, 4)) 
        t_z_profile = t_norm.amax(dim=(3, 4))
        loss_z_prof = F.l1_loss(p_z_profile, t_z_profile) * 5.0
        
        loss_bc = (F.l1_loss(p_norm[:, :, 0, :, :], t_norm[:, :, 0, :, :]) +
                   F.l1_loss(p_norm[:, :, -1, :, :], t_norm[:, :, -1, :, :]) +
                   F.l1_loss(p_norm[:, :, :, 0, :], t_norm[:, :, :, 0, :]) +
                   F.l1_loss(p_norm[:, :, :, -1, :], t_norm[:, :, :, -1, :]) +
                   F.l1_loss(p_norm[:, :, :, :, 0], t_norm[:, :, :, :, 0]) +
                   F.l1_loss(p_norm[:, :, :, :, -1], t_norm[:, :, :, :, -1])) * 2.0
        
        # 3. High-Frequency Gradient Loss (Edge penalty)
        p_dx, t_dx = p_norm[..., 1:] - p_norm[..., :-1], t_norm[..., 1:] - t_norm[..., :-1]
        p_dy, t_dy = p_norm[..., 1:, :] - p_norm[..., :-1, :], t_norm[..., 1:, :] - t_norm[..., :-1, :]
        loss_grad_xy = (F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)) * 5.0 # Amplified from 2.0 to 5.0
        
        laplacian_field = F.conv3d(p_norm, self.lap_kernel)
        k_inner = k_3d[:, :, 1:-1, 1:-1, 1:-1]
        lap_loss_raw = torch.abs(k_inner * laplacian_field)
        src_inner = source_mask[:, :, 1:-1, 1:-1, 1:-1]
        lap_enforcement_mask = (src_inner == 0).float()
        
        lap_sum = torch.sum(lap_enforcement_mask).clamp(min=1.0)
        loss_laplacian = torch.sum(lap_loss_raw * lap_enforcement_mask) / lap_sum
        
        # Integrate SSIM into the final loss
        total_loss = loss_bg_shape + loss_src_shape + loss_peak + loss_z_prof + loss_grad_xy + loss_bc + loss_structure + (loss_laplacian * 0.1)
        
        with torch.no_grad():
            metrics = {
                'shape': (loss_bg_shape + loss_src_shape).detach(),
                'peak': loss_peak.detach(),
                'z_prof': loss_z_prof.detach(),
                'lap': loss_laplacian.detach(),
                'grad': loss_grad_xy.detach(),
                'ssim': loss_structure.detach()
            }
            
        return total_loss, metrics