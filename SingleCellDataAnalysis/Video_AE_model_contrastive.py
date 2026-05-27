#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_model_contrastive.py  — Strategy C

Contrastive video encoder. Reuses the 3D-CNN encoder backbone from
VideoAutoencoderSegmenter and adds a projection head for NT-Xent loss.

The decoder is discarded — we only need the encoder for representation learning.

Architecture:
  Encoder: (B,1,101,32,112) → 16D latent z   (shared with segmenter)
  Projector: 16D → 64D → 32D  (used only during training)

The 16D latent z is what we visualise in UMAP after training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')


class ContrastiveVideoEncoder(nn.Module):
    """
    3D-CNN Encoder + Projection Head for contrastive (SimCLR-style) training.

    The encoder backbone is identical to VideoAutoencoderSegmenter.encoder_conv
    + encoder_fc, so weights can optionally be warm-started from a segmenter
    checkpoint.

    Args:
        latent_dim: size of the latent representation z (default 16)
        proj_dim:   size of the projection head output (default 32)
    """

    def __init__(self, latent_dim: int = 16, proj_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj_dim   = proj_dim

        # ── Encoder (identical to segmenter) ─────────────────────────────────
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._flat_size = 64 * 7 * 2 * 7  # = 6272 for 32x112 input

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )

        # ── Projection Head (used only during training, discarded for UMAP) ──
        # 2-layer MLP, final output is L2-normalised
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,101,32,112) → z: (B, latent_dim)"""
        h = self.encoder_conv(x)
        return self.encoder_fc(h)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → p: (B, proj_dim), L2-normalised"""
        return F.normalize(self.projector(z), dim=-1)

    def forward(self, x: torch.Tensor):
        """Returns (z, p) — latent and normalised projection."""
        z = self.encode(x)
        p = self.project(z)
        return z, p

    def load_encoder_from_segmenter(self, segmenter_path: str, device='cpu'):
        """
        Warm-start the encoder backbone from a saved VideoAutoencoderSegmenter
        checkpoint. Only transfers encoder_conv and encoder_fc weights.
        """
        ckpt = torch.load(segmenter_path, map_location=device)
        own_state = self.state_dict()
        loaded = 0
        for name, param in ckpt.items():
            if name.startswith('encoder_conv.') or name.startswith('encoder_fc.'):
                if name in own_state:
                    own_state[name].copy_(param)
                    loaded += 1
        self.load_state_dict(own_state)
        print(f"   Warm-started {loaded} encoder tensors from {segmenter_path}")


def nt_xent_loss(p_i: torch.Tensor, p_j: torch.Tensor, temperature: float = 0.1):
    """
    NT-Xent (Normalised Temperature-scaled Cross Entropy) loss.

    p_i, p_j: L2-normalised projection vectors, shape (B, proj_dim).
    Row k of p_i is the positive pair of row k of p_j, and vice versa.

    The loss encourages p_i[k] ≈ p_j[k] while pushing away all other pairs.
    """
    B = p_i.shape[0]
    # Concatenate: rows 0..B-1 are from view i, rows B..2B-1 from view j
    z = torch.cat([p_i, p_j], dim=0)           # (2B, D)
    sim = torch.mm(z, z.T) / temperature        # (2B, 2B) cosine similarity / T

    # Mask self-similarities to -inf so they don't contribute to softmax
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (k, k+B) and (k+B, k)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),   # for rows 0..B-1, positives are B..2B-1
        torch.arange(0, B,     device=z.device),    # for rows B..2B-1, positives are 0..B-1
    ])

    return F.cross_entropy(sim, labels)


if __name__ == '__main__':
    model = ContrastiveVideoEncoder(latent_dim=16, proj_dim=32)
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n:,}")
    dummy = torch.zeros(4, 1, 101, 32, 112)
    z, p = model(dummy)
    print(f"Latent z: {z.shape}")
    print(f"Projection p: {p.shape}")

    # Test loss
    p_i = F.normalize(torch.randn(4, 32), dim=-1)
    p_j = F.normalize(torch.randn(4, 32), dim=-1)
    loss = nt_xent_loss(p_i, p_j)
    print(f"NT-Xent loss (random): {loss.item():.3f}")
    print("✅ Architecture check passed.")
