#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_model.py

3D Convolutional Video Autoencoder.

Input:  (B, 1, 101, 48, 96) — batch of single-cell time-lapse GFP movies
                               B=batch, 1 channel, 101 frames, 48×96 pixels

Architecture:
  Encoder — 4× strided Conv3d layers compress T×H×W progressively:
    (1, 101, 48, 96) → (16, 51, 24, 48) → (32, 26, 12, 24) → (64, 13, 6, 12) → (64, 7, 3, 6)
  Bottleneck — Flatten + Linear → latent_dim

  Decoder — FC unpacks to spatial volume, then 4× Upsample+Conv3d to exact input size:
    (64, 7, 3, 6) → (64, 13, 6, 12) → (64, 26, 12, 24) → (32, 51, 24, 48) → (1, 101, 48, 96)

Upsample + Conv3d is used instead of ConvTranspose3d to:
  - Avoid output_padding headaches with asymmetric dimensions
  - Avoid checkerboard artifacts
"""

import torch
import torch.nn as nn


class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        # Input: (B, 1, 101, 48, 96)
        self.encoder_conv = nn.Sequential(
            # Layer 1  → (B, 16, 51, 24, 48)
            nn.Conv3d(1, 16, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2  → (B, 32, 26, 12, 24)
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3  → (B, 64, 13, 6, 12)
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4  → (B, 64, 7, 3, 6)
            nn.Conv3d(64, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flat size after 4 encoder layers: 64 × 7 × 3 × 6 = 8064
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

        # ── Multi-Task Heads ──────────────────────────────────────────────────
        # Predicts 101 frames * 2 polarities = 202
        self.traj_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 101 * 2)
        )
        
        # Predicts 11 engineered features
        self.feat_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 11)
        )

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

            # (B, 16, 51, 24, 48) → (B, 1, 101, 48, 96)
            nn.Upsample(size=(101, 48, 96), mode='trilinear', align_corners=False),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 101, 48, 96) → z: (B, latent_dim)"""
        h = self.encoder_conv(x)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → x_hat: (B, 1, 101, 48, 96)"""
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 7, 3, 6)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        
        traj_hat = self.traj_head(z)
        traj_hat = traj_hat.view(-1, 101, 2)
        
        feat_hat = self.feat_head(z)
        
        return x_hat, traj_hat, feat_hat, z


if __name__ == '__main__':
    model = VideoAutoencoder(latent_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    dummy = torch.zeros(2, 1, 101, 48, 96)
    out, traj, feat, z = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Latent: {z.shape}")
    print(f"Traj:   {traj.shape}")
    print(f"Feat:   {feat.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == dummy.shape, "Output shape mismatch!"
    print("✅ Architecture check passed.")
