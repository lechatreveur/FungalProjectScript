#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FC_AE_3d_train_M156.py

Trains MultimodalAutoencoder3D exclusively on M156 data.
Saves checkpoint to /Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d/fc_ae_3d_M156_final.pth
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

EXPERIMENTS = {
    "M156": "/Volumes/X10 Pro/Movies/2026_07_16_M156"
}

OUTPUT_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d/"
MODEL_PATH = os.path.join(OUTPUT_DIR, "fc_ae_3d_M156_final.pth")

EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ALPHA = 1.0  # Weight for feature reconstruction


class MultimodalAutoencoder3D(nn.Module):
    def __init__(self, seq_len=101, in_channels=2, num_features=11, latent_dim=3):
        super(MultimodalAutoencoder3D, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.num_features = num_features
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.flatten_dim = 64 * 13

        self.encoder_feat = nn.Sequential(
            nn.Linear(num_features, 32), nn.ReLU()
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flatten_dim + 32, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # --- Decoder ---
        self.decoder_traj_fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, self.flatten_dim), nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=0), nn.ReLU(),
            nn.ConvTranspose1d(16, in_channels, kernel_size=5, stride=2, padding=2, output_padding=0),
        )

        self.decoder_feat = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, num_features)
        )

    def forward(self, traj, feat):
        traj = traj.transpose(1, 2)  # (B, C, L)

        z_conv = self.encoder_conv(traj).view(traj.size(0), -1)
        z_feat = self.encoder_feat(feat)

        z = self.encoder_fc(torch.cat([z_conv, z_feat], dim=1))

        recon_traj_flat = self.decoder_traj_fc(z)
        recon_traj = self.decoder_conv(recon_traj_flat.view(traj.size(0), 64, 13))
        recon_traj = recon_traj.transpose(1, 2)  # (B, L, C)

        recon_feat = self.decoder_feat(z)
        return recon_traj, recon_feat, z


def evaluate(model, dataloader, criterion, alpha, device):
    model.eval()
    total_loss, total_recon, total_feat = 0.0, 0.0, 0.0
    total_samples = 0
    with torch.no_grad():
        for b_traj, b_feat in dataloader:
            b_traj, b_feat = b_traj.to(device), b_feat.to(device)
            recon_traj, recon_feat, _ = model(b_traj, b_feat)
            l_traj = criterion(recon_traj, b_traj)
            l_feat = criterion(recon_feat, b_feat)
            loss = l_traj + alpha * l_feat

            bs = b_traj.size(0)
            total_loss += loss.item() * bs
            total_recon += l_traj.item() * bs
            total_feat += l_feat.item() * bs
            total_samples += bs

    return (total_loss / total_samples, total_recon / total_samples, total_feat / total_samples)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("📥 Loading M156 constrained dataset...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)

    N = len(X_traj)
    print(f"Total valid M156 observations loaded: {N}")

    # Fixed seed train/val split (80% train, 20% val)
    np.random.seed(42)
    indices = np.random.permutation(N)
    val_size = int(0.2 * N)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    X_traj_train, X_feat_train = X_traj[train_idx], X_feat[train_idx]
    X_traj_val, X_feat_val = X_traj[val_idx], X_feat[val_idx]

    print(f"Train set: {len(X_traj_train)} samples | Val set: {len(X_traj_val)} samples")

    train_dataset = TensorDataset(
        torch.tensor(X_traj_train, dtype=torch.float32),
        torch.tensor(X_feat_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_traj_val, dtype=torch.float32),
        torch.tensor(X_feat_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    model = MultimodalAutoencoder3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("🚀 Starting AE 3D Training for M156...")
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_recon, total_feat = 0.0, 0.0, 0.0
        train_samples = 0

        for b_traj, b_feat in train_loader:
            b_traj, b_feat = b_traj.to(device), b_feat.to(device)
            optimizer.zero_grad()

            recon_traj, recon_feat, z = model(b_traj, b_feat)

            l_traj = criterion(recon_traj, b_traj)
            l_feat = criterion(recon_feat, b_feat)
            loss = l_traj + ALPHA * l_feat

            loss.backward()
            optimizer.step()

            bs = b_traj.size(0)
            total_loss += loss.item() * bs
            total_recon += l_traj.item() * bs
            total_feat += l_feat.item() * bs
            train_samples += bs

        train_loss = total_loss / train_samples
        train_recon_loss = total_recon / train_samples
        train_feat_loss = total_feat / train_samples

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            val_loss, val_recon_loss, val_feat_loss = evaluate(model, val_loader, criterion, ALPHA, device)
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}], "
                  f"Train Loss: {train_loss:.4f} (Traj: {train_recon_loss:.4f}, Feat: {train_feat_loss:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (Traj: {val_recon_loss:.4f}, Feat: {val_feat_loss:.4f})")

    elapsed = time.time() - t0
    val_loss, val_recon_loss, val_feat_loss = evaluate(model, val_loader, criterion, ALPHA, device)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Training complete in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
    print(f"💾 Checkpoint saved to: {MODEL_PATH}")
    print(f"Final Train Loss: {train_loss:.4f} (Traj: {train_recon_loss:.4f}, Feat: {train_feat_loss:.4f})")
    print(f"Final Val Loss:   {val_loss:.4f} (Traj: {val_recon_loss:.4f}, Feat: {val_feat_loss:.4f})")


if __name__ == "__main__":
    main()
