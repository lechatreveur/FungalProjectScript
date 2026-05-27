#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_sweep_model import FeatureConstrainedAutoencoder
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

LATENT_DIM = 4
OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/fc_final_4d/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "fc_ae_4d_model.pth")

EPOCHS = 600
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ALPHA = 1.0

def train_final_4d():
    # 1. Load Data
    print("📥 Loading data...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    X_traj_tensor = torch.tensor(X_traj, dtype=torch.float32)
    X_feat_tensor = torch.tensor(X_feat, dtype=torch.float32)
    dataset = TensorDataset(X_traj_tensor, X_feat_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FeatureConstrainedAutoencoder(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 3. Train
    print(f"🚀 Training final 4D Feature-Constrained model...")
    for epoch in range(EPOCHS):
        model.train()
        for b_traj, b_feat in dataloader:
            b_traj, b_feat = b_traj.to(device), b_feat.to(device)
            optimizer.zero_grad()
            recon, _, pred_f = model(b_traj)
            loss = criterion(recon, b_traj) + ALPHA * criterion(pred_f, b_feat)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] complete.")
            
    # 4. Save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

from torch.utils.data import TensorDataset, DataLoader
if __name__ == "__main__":
    train_final_4d()
