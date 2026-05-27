#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
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

OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/fc_sweep/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIMS = [2, 3, 4, 5, 6, 8, 10]
REPEATS = 3 # Reducing to 3 for speed, but sufficient for trend
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ALPHA = 1.0 # Feature loss weight

def run_sweep():
    # 1. Load Data
    print("📥 Loading constrained dataset...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    X_traj_tensor = torch.tensor(X_traj, dtype=torch.float32)
    X_feat_tensor = torch.tensor(X_feat, dtype=torch.float32)
    
    dataset = TensorDataset(X_traj_tensor, X_feat_tensor)
    
    # 2. Sweep
    results = {dim: [] for dim in LATENT_DIMS}
    recon_results = {dim: [] for dim in LATENT_DIMS}
    feat_results = {dim: [] for dim in LATENT_DIMS}
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    for dim in LATENT_DIMS:
        for r in range(REPEATS):
            print(f"🔄 Training Dim {dim} (Repeat {r+1}/{REPEATS})...")
            
            # Split into 80/20 train/val
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_db, val_db = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False)
            
            model = FeatureConstrainedAutoencoder(latent_dim=dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            best_recon = float('inf')
            best_feat = float('inf')
            
            for epoch in range(EPOCHS):
                model.train()
                for b_traj, b_feat in train_loader:
                    b_traj, b_feat = b_traj.to(device), b_feat.to(device)
                    optimizer.zero_grad()
                    recon, _, pred_f = model(b_traj)
                    l_recon = criterion(recon, b_traj)
                    l_feat = criterion(pred_f, b_feat)
                    loss = l_recon + ALPHA * l_feat
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                val_total = 0
                val_recon = 0
                val_feat = 0
                with torch.no_grad():
                    for v_traj, v_feat in val_loader:
                        v_traj, v_feat = v_traj.to(device), v_feat.to(device)
                        v_recon, _, v_pred_f = model(v_traj)
                        l_r = criterion(v_recon, v_traj)
                        l_f = criterion(v_pred_f, v_feat)
                        val_total += (l_r + ALPHA * l_f).item() * v_traj.size(0)
                        val_recon += l_r.item() * v_traj.size(0)
                        val_feat += l_f.item() * v_traj.size(0)
                
                avg_val = val_total / val_size
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_recon = val_recon / val_size
                    best_feat = val_feat / val_size
            
            results[dim].append(best_val_loss)
            recon_results[dim].append(best_recon)
            feat_results[dim].append(best_feat)
            
    # 3. Plot Results
    dims = sorted(results.keys())
    means = [np.mean(results[d]) for d in dims]
    stds = [np.std(results[d]) for d in dims]
    
    recon_means = [np.mean(recon_results[d]) for d in dims]
    feat_means = [np.mean(feat_results[d]) for d in dims]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(dims, means, yerr=stds, marker='o', label='Total Val Loss', color='black', capsize=5)
    plt.plot(dims, recon_means, marker='s', linestyle='--', label='Reconstruction Loss', color='blue')
    plt.plot(dims, feat_means, marker='^', linestyle='-.', label='Feature Prediction Loss', color='red')
    
    plt.xlabel('Latent Dimensions (N)')
    plt.ylabel('Validation MSE')
    plt.title('Feature-Constrained AE Dimension Sweep')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "fc_latent_dim_sweep_plot.png"), dpi=150)
    print(f"💾 Sweep plot saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_sweep()
