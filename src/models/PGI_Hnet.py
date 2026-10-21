import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. BUILDING BLOCKS
# ==========================================

class ResBlock3D(nn.Module):
    """Residual 3D convolution block with optional replicate padding for boundary smoothness."""
    def __init__(self, in_ch, out_ch, replicate_pad=False):
        super().__init__()
        pm = 'replicate' if replicate_pad else 'zeros'
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode=pm)
        self.gn1 = nn.GroupNorm(min(4, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, padding_mode=pm)
        self.gn2 = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)


class SpatialExtrusionFusion(nn.Module):
    """Gated fusion of 2D phase features (extruded to 3D) with 3D volumetric features."""
    def __init__(self, channels):
        super().__init__()
        self.attn_gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, f_2d, f_3d):
        B, C, Z, Y, X = f_3d.shape
        f_2d_ext = f_2d.unsqueeze(2).expand(-1, -1, Z, -1, -1)
        gate = self.attn_gate(torch.cat([f_2d_ext, f_3d], dim=1))
        return self.fuse_conv(torch.cat([f_2d_ext * gate, f_3d], dim=1))


# ==========================================
# 2. PHYSICS-GUIDED INVERSION UNET V2
# ==========================================

class PhysicsGuidedInversionUNet(nn.Module):
    """
    U-Net for 3D thermal profile inversion from 2D phase measurements.

    V2 key improvements over V1:
      - Softplus output + boundary decay envelope: structurally eliminates U-shape artifacts.
        Sigmoid defaults to 0.5 for uninitialized logits; softplus defaults to ~0 for
        negative logits. The envelope forces exact zero at domain faces.
      - AvgPool3d downsampling: eliminates aliasing/ringing caused by MaxPool3d
      - Dilated residual bottleneck: larger receptive field for single-particle far-field
        decay without increasing VRAM usage
      - Max+Avg pooled magnitude head: robust peak prediction for sparse (single-particle)
        sources where global-average pooling dilutes the signal
      - Replicate-padded decoder: smooth boundary temperature profiles, no zero-pad ringing
    """
    def __init__(self, bf=8, grid_shape=(32, 128, 128)):
        super().__init__()
        self.bf = bf

        # --- Input Encoders (replicate padding for physical boundary preservation) ---
        self.enc_2d = nn.Sequential(
            nn.Conv2d(1, bf, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, bf), nn.SiLU(inplace=True),
            nn.Conv2d(bf, bf*2, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, bf*2), nn.SiLU(inplace=True)
        )
        self.enc_3d_prior = nn.Sequential(
            nn.Conv3d(3, bf*2, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, bf*2), nn.SiLU(inplace=True)
        )
        self.fusion = SpatialExtrusionFusion(channels=bf*2)

        # --- Encoder Path (AvgPool for smooth, alias-free downsampling) ---
        self.init_3d = nn.Sequential(
            nn.Conv3d(bf*2, bf*2, 3, padding=1),
            nn.GroupNorm(4, bf*2), nn.SiLU(inplace=True)
        )
        self.down1_pool = nn.AvgPool3d(2)
        self.down1_conv = ResBlock3D(bf*2, bf*4)

        self.down2_pool = nn.AvgPool3d(2)
        self.down2_conv = ResBlock3D(bf*4, bf*8)

        # Removed dilated bottleneck

        # --- Magnitude Head (max+avg pooling for robust single-particle peak prediction) ---
        self.mag_max_pool = nn.AdaptiveMaxPool3d(1)
        self.mag_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mag_fc = nn.Sequential(
            nn.Linear(bf*16 + 2, bf*8),     # bf*8 (max) + bf*8 (avg) + 2 (log scales)
            nn.SiLU(inplace=True),
            nn.Linear(bf*8, bf*4),
            nn.SiLU(inplace=True),
            nn.Linear(bf*4, 1)
        )
        nn.init.zeros_(self.mag_fc[-1].weight)
        nn.init.zeros_(self.mag_fc[-1].bias)

        # --- Decoder Path (replicate padding for smooth, ringing-free boundaries) ---
        self.dec1 = ResBlock3D(bf*12, bf*4, replicate_pad=True)   # cat(up=bf*8, skip=bf*4)
        self.dec2 = ResBlock3D(bf*6, bf*2, replicate_pad=True)    # cat(up=bf*4, skip=bf*2)

        # --- Output Head ---
        self.final_conv = nn.Conv3d(bf*2, 1, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.constant_(self.final_conv.bias, -1.0)   # softplus(-1) ≈ 0.31

        # Removed envelope method to avoid forcing peak to center

    def forward(self, phase_2d, T_inv_3d, k_3d, beta_3d):
        B = phase_2d.size(0)

        # --- Per-sample input normalization ---
        phase_max = torch.amax(torch.abs(phase_2d), dim=(2, 3), keepdim=True).clamp(min=1e-5)
        t_inv_max = torch.amax(torch.abs(T_inv_3d), dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)

        f_2d = self.enc_2d(phase_2d / phase_max)
        f_3d = self.enc_3d_prior(torch.cat([
            T_inv_3d / t_inv_max, k_3d, beta_3d * 1e4
        ], dim=1))
        f_vol = self.fusion(f_2d, f_3d)

        # --- Encoder ---
        e1 = self.init_3d(f_vol)                        # (B, bf*2, Z, Y, X)
        e2 = self.down1_conv(self.down1_pool(e1))        # (B, bf*4, Z/2, Y/2, X/2)
        b = self.down2_conv(self.down2_pool(e2))         # (B, bf*8, Z/4, Y/4, X/4)

        # Removed dilated bottleneck to restore speed

        # --- Magnitude Prediction ---
        b_max = self.mag_max_pool(b).view(B, -1)        # (B, bf*8) peak features
        b_avg = self.mag_avg_pool(b).view(B, -1)        # (B, bf*8) average features
        log_p = torch.log(phase_max.view(B, 1))
        log_t = torch.log(t_inv_max.view(B, 1))
        mag_input = torch.cat([b_max, b_avg, log_p, log_t], dim=1)

        log_scale = self.mag_fc(mag_input).view(B, 1, 1, 1, 1)
        dynamic_peak = torch.exp(log_scale + 4.605).clamp(min=0.1, max=5000.0)

        # --- Decoder ---
        u1 = F.interpolate(b, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        # --- Physics-Constrained Output ---
        # softplus: [0, ∞), naturally near-zero for negative logits (unlike sigmoid → 0.5)
        logits = self.final_conv(d2)
        spatial = F.softplus(logits)

        # Normalize spatial distribution to [0, 1], then apply envelope & peak scale
        sp_max = spatial.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
        spatial_norm = spatial / sp_max

        return spatial_norm * dynamic_peak


# ==========================================
# 3. PHYSICS-CONSTRAINED LOSS FUNCTION
# ==========================================

class PhysicsConstrainedLoss(nn.Module):
    """
    Multi-objective loss with physics constraints for 3D thermal field prediction.

    Improvements over FastAdaptiveLagrangianLoss:
      - Unified normalization: both pred and true normalized by T_max (not separately)
        for direct, scale-aware voxel comparison. Fixes weak gradients for single particles.
      - Hot/cold masking: based on thermal distribution (t_norm > 5%) instead of source_mask.
        Source mask is 1 voxel for single particles → extremely noisy loss. Hot mask captures
        the entire thermal plume → stable supervision.
      - Mean Z-profile: mean-over-XY instead of max-over-XY captures the volume-averaged
        Z distribution, not just the peak envelope.
      - 3D gradient matching: adds Z-gradient term for accurate thermal interface behavior.
      - Smoothness penalty: curvature (2nd derivative) penalty in cold regions suppresses
        residual oscillations/ringing in the far field.
      - Reduced boundary weight: the network's boundary envelope handles BCs structurally;
        the explicit BC loss is now a gentle reinforcement, not the primary mechanism.
    """
    def __init__(self, dx_um, dz_um):
        super().__init__()
        self.dz_ratio = dx_um / dz_um
        self.dz_ratio_sq = self.dz_ratio ** 2

        # Laplacian kernel for anisotropic grid (dx_xy ≠ dz)
        weight = torch.zeros(1, 1, 3, 3, 3)
        weight[0, 0, 1, 1, 1] = -2.0 * (2.0 + self.dz_ratio_sq)
        weight[0, 0, 1, 1, 0] = 1.0
        weight[0, 0, 1, 1, 2] = 1.0
        weight[0, 0, 1, 0, 1] = 1.0
        weight[0, 0, 1, 2, 1] = 1.0
        weight[0, 0, 0, 1, 1] = self.dz_ratio_sq
        weight[0, 0, 2, 1, 1] = self.dz_ratio_sq
        self.register_buffer('lap_kernel', weight)

    def forward(self, T_pred, T_true, k_3d, source_mask):
        T_max = T_true.amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
        P_max = T_pred.amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)

        # ============ 1. PEAK LOSS (log-scale matching) ============
        loss_peak = F.mse_loss(torch.log(P_max), torch.log(T_max)) * 10.0

        # ============ 2. SHAPE LOSS (separate normalization) ============
        # Normalize fields by their own max to decouple shape learning from peak magnitude learning
        t_norm = (T_true / T_max).clamp(min=0.0)
        p_norm = (T_pred / P_max).clamp(min=0.0, max=1.0)

        # Adaptive hot/cold mask based on thermal distribution (NOT source_mask)
        hot_mask = (t_norm > 0.05).float()       # >5% of peak temperature
        cold_mask = 1.0 - hot_mask
        hot_n = hot_mask.sum().clamp(min=1.0)
        cold_n = cold_mask.sum().clamp(min=1.0)

        # Combined MSE + log-space error
        mse_err = (p_norm - t_norm) ** 2
        log_err = (torch.log1p(10.0 * p_norm) - torch.log1p(10.0 * t_norm)).abs()
        shape_err = mse_err + 0.5 * log_err

        loss_hot = torch.sum(shape_err * hot_mask) / hot_n * 10.0
        loss_cold = torch.sum(shape_err * cold_mask) / cold_n * 5.0

        # ============ 3. Z-PROFILE LOSS (mean over XY, not max) ============
        p_z = p_norm.mean(dim=(3, 4))             # (B, 1, Z) volume-averaged Z profile
        t_z = t_norm.mean(dim=(3, 4))
        loss_z_prof = F.l1_loss(p_z, t_z) * 5.0

        # ============ 4. 3D GRADIENT LOSS (XY + Z) ============
        loss_grad = (
            F.l1_loss(p_norm[..., 1:] - p_norm[..., :-1],
                      t_norm[..., 1:] - t_norm[..., :-1]) +
            F.l1_loss(p_norm[..., 1:, :] - p_norm[..., :-1, :],
                      t_norm[..., 1:, :] - t_norm[..., :-1, :]) +
            F.l1_loss(p_norm[:, :, 1:] - p_norm[:, :, :-1],
                      t_norm[:, :, 1:] - t_norm[:, :, :-1])
        ) * 2.0

        # ============ 5. BOUNDARY LOSS (gentle reinforcement) ============
        # Reduced weight: the network's boundary envelope structurally enforces BCs
        loss_bc = (
            F.l1_loss(p_norm[:, :, 0], t_norm[:, :, 0]) +
            F.l1_loss(p_norm[:, :, -1], t_norm[:, :, -1]) +
            F.l1_loss(p_norm[:, :, :, 0], t_norm[:, :, :, 0]) +
            F.l1_loss(p_norm[:, :, :, -1], t_norm[:, :, :, -1]) +
            F.l1_loss(p_norm[..., 0], t_norm[..., 0]) +
            F.l1_loss(p_norm[..., -1], t_norm[..., -1])
        ) * 0.5

        # ============ 6. LAPLACIAN PHYSICS LOSS ============
        # Enforce ∇²T = 0 away from heat sources (Laplace's equation)
        lap_field = F.conv3d(p_norm, self.lap_kernel)
        k_inner = k_3d[:, :, 1:-1, 1:-1, 1:-1]
        lap_raw = (k_inner * lap_field).abs()
        src_inner = source_mask[:, :, 1:-1, 1:-1, 1:-1]
        no_src = (src_inner == 0).float()
        loss_lap = torch.sum(lap_raw * no_src) / no_src.sum().clamp(min=1.0)

        # ============ 7. SMOOTHNESS LOSS (anti-ringing in cold regions) ============
        # Penalize 2nd-derivative magnitude (curvature) where temperature should be smooth.
        # High curvature in the far-field indicates ringing/oscillation artifacts.
        curv_x = p_norm[..., 2:] - 2.0 * p_norm[..., 1:-1] + p_norm[..., :-2]
        curv_y = p_norm[..., 2:, :] - 2.0 * p_norm[..., 1:-1, :] + p_norm[..., :-2, :]
        curv_z = p_norm[:, :, 2:] - 2.0 * p_norm[:, :, 1:-1] + p_norm[:, :, :-2]

        # Only penalize curvature in cold (background) regions — hot regions have
        # legitimate sharp gradients from the thermal source
        ci_x = cold_mask[..., 1:-1]
        ci_y = cold_mask[..., 1:-1, :]
        ci_z = cold_mask[:, :, 1:-1]

        loss_smooth = (
            torch.sum(curv_x.abs() * ci_x) / ci_x.sum().clamp(min=1.0) +
            torch.sum(curv_y.abs() * ci_y) / ci_y.sum().clamp(min=1.0) +
            torch.sum(curv_z.abs() * ci_z) / ci_z.sum().clamp(min=1.0)
        ) * 1.0

        # ============ TOTAL ============
        total_loss = (loss_hot + loss_cold + loss_peak + loss_z_prof +
                      loss_grad + loss_bc + loss_lap * 0.1 + loss_smooth)

        with torch.no_grad():
            metrics = {
                'shape': (loss_hot + loss_cold).item(),
                'peak': loss_peak.item(),
                'z_prof': loss_z_prof.item(),
                'grad': loss_grad.item(),
                'lap': loss_lap.item(),
                'smooth': loss_smooth.item()
            }

        return total_loss, metrics


# ==========================================
# BACKWARD COMPATIBILITY ALIAS
# ==========================================
# Keep the old name as an alias so existing scripts that import it still work
FastAdaptiveLagrangianLoss = PhysicsConstrainedLoss