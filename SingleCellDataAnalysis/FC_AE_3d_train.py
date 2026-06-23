#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

OUTPUT_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "fc_ae_3d_final.pth")

EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ALPHA = 1.0 # Weight for feature reconstruction

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
        traj = traj.transpose(1, 2) # (B, C, L)
        
        z_conv = self.encoder_conv(traj).view(traj.size(0), -1)
        z_feat = self.encoder_feat(feat)
        
        z = self.encoder_fc(torch.cat([z_conv, z_feat], dim=1))
        
        recon_traj_flat = self.decoder_traj_fc(z)
        recon_traj = self.decoder_conv(recon_traj_flat.view(traj.size(0), 64, 13))
        recon_traj = recon_traj.transpose(1, 2) # (B, L, C)
        
        recon_feat = self.decoder_feat(z)
        return recon_traj, recon_feat, z

def main():
    print("📥 Loading constrained dataset...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    
    X_traj_t = torch.tensor(X_traj, dtype=torch.float32)
    X_feat_t = torch.tensor(X_feat, dtype=torch.float32)
    
    dataset = TensorDataset(X_traj_t, X_feat_t)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    model = MultimodalAutoencoder3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("🚀 Starting AE 3D Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_recon, total_feat = 0, 0, 0
        
        for b_traj, b_feat in dataloader:
            b_traj, b_feat = b_traj.to(device), b_feat.to(device)
            optimizer.zero_grad()
            
            recon_traj, recon_feat, z = model(b_traj, b_feat)
            
            l_traj = criterion(recon_traj, b_traj)
            l_feat = criterion(recon_feat, b_feat)
            loss = l_traj + ALPHA * l_feat
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * b_traj.size(0)
            total_recon += l_traj.item() * b_traj.size(0)
            total_feat += l_feat.item() * b_traj.size(0)
            
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataset):.4f} | Traj: {total_recon/len(dataset):.4f} | Feat: {total_feat/len(dataset):.4f}")
            
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model weights saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
