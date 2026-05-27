import torch
import torch.nn as nn
import numpy as np

class VideoAutoencoderPhysics(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder_conv = nn.Sequential(
            # Input: (B, 1, 101, 48, 96)
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (B, 16, 51, 24, 48)
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (B, 32, 26, 12, 24)
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (B, 64, 13, 6, 12)
            nn.Conv3d(64, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 64, 7, 3, 6)
        )
        
        self._flat_size = 64 * 7 * 3 * 6
        
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self._flat_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Multi-Task Features Head ─────────────────────────────────────────
        # Predicts 11 engineered features
        self.feat_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 11)
        )

        # ── Multi-Channel Physics Decoder ────────────────────────────────────
        # Upsample to exact target sizes to avoid any padding arithmetic
        self.decoder_conv = nn.Sequential(
            # (B, 64, 7, 3, 6) → (B, 64, 13, 6, 12)
            nn.Upsample(size=(13, 6, 12), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 13, 6, 12) → (B, 32, 26, 12, 24)
            nn.Upsample(size=(26, 12, 24), mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 32, 26, 12, 24) → (B, 16, 51, 24, 48)
            nn.Upsample(size=(51, 24, 48), mode='trilinear', align_corners=False),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 16, 51, 24, 48) → (B, 3, 101, 48, 96)  <-- 3 Channels!
            nn.Upsample(size=(101, 48, 96), mode='trilinear', align_corners=False),
            nn.Conv3d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # ── Physics Trajectory Scaling ───────────────────────────────────────
        self.pol1_scale = nn.Linear(1, 1)
        self.pol2_scale = nn.Linear(1, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 7, 3, 6)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        
        # Decoder outputs 3 channels: x_hat, mask_pol1, mask_pol2
        multi_out = self.decode(z)
        x_hat     = multi_out[:, 0:1, :, :, :]  # (B, 1, T, H, W)
        mask_pol1 = multi_out[:, 1:2, :, :, :]  # (B, 1, T, H, W)
        mask_pol2 = multi_out[:, 2:3, :, :, :]  # (B, 1, T, H, W)
        
        # ── Differentiable Property Extraction ──
        # Compute intensity of the generated image weighted by the generated masks
        # We sum over H and W (dims 3 and 4) to get a (B, 1, T) vector
        eps = 1e-6
        pol1_intensity = torch.sum(x_hat * mask_pol1, dim=(3, 4)) / (torch.sum(mask_pol1, dim=(3, 4)) + eps)
        pol2_intensity = torch.sum(x_hat * mask_pol2, dim=(3, 4)) / (torch.sum(mask_pol2, dim=(3, 4)) + eps)
        
        # (B, 1, T) -> (B, T)
        pol1_intensity = pol1_intensity.squeeze(1)
        pol2_intensity = pol2_intensity.squeeze(1)
        
        # Scale to match the StandardScaled trajectory ground truth
        pol1_traj_hat = self.pol1_scale(pol1_intensity.unsqueeze(-1)).squeeze(-1)
        pol2_traj_hat = self.pol2_scale(pol2_intensity.unsqueeze(-1)).squeeze(-1)
        
        # Stack to (B, T, 2)
        traj_hat = torch.stack([pol1_traj_hat, pol2_traj_hat], dim=2)
        
        # Standard MLP for the 11 global features
        feat_hat = self.feat_head(z)
        
        return x_hat, traj_hat, feat_hat, z, mask_pol1, mask_pol2

if __name__ == '__main__':
    model = VideoAutoencoderPhysics()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    dummy = torch.zeros(2, 1, 101, 48, 96)
    x_hat, traj, feat, z, m1, m2 = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Latent: {z.shape}")
    print(f"Traj:   {traj.shape}")
    print(f"Feat:   {feat.shape}")
    print(f"Masks:  {m1.shape}, {m2.shape}")
    print(f"Output: {x_hat.shape}")
    assert x_hat.shape == dummy.shape, "Output shape mismatch!"
    print("✅ Physics Architecture check passed.")
