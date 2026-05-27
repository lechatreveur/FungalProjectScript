#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
import numpy as np

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.AE_model import TrajectoryAutoencoder
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/sufficiency_test/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "fc_ae_model.pth")

EPOCHS = 600
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ALPHA = 1.0 # Weight for the feature matching loss

def train_fc_ae():
    # 1. Load Data
    print("📥 Loading constrained dataset...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    
    # Convert to tensors
    # PyTorch Conv1d expects (Batch, Channels, Length), but our model's forward pass handles the transpose internally.
    X_traj_tensor = torch.tensor(X_traj, dtype=torch.float32)
    X_feat_tensor = torch.tensor(X_feat, dtype=torch.float32)
    
    dataset = TensorDataset(X_traj_tensor, X_feat_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model (Bottleneck is exactly 11)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    model = TrajectoryAutoencoder(latent_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print("🚀 Starting Feature Sufficiency Training...")
    loss_history = []
    recon_history = []
    feat_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_recon = 0
        total_feat = 0
        
        for batch_traj, batch_feat in dataloader:
            batch_traj = batch_traj.to(device)
            batch_feat = batch_feat.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_traj, latent_z = model(batch_traj)
            
            # Losses
            loss_recon = criterion(recon_traj, batch_traj)
            loss_feat = criterion(latent_z, batch_feat)
            
            loss = loss_recon + ALPHA * loss_feat
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_traj.size(0)
            total_recon += loss_recon.item() * batch_traj.size(0)
            total_feat += loss_feat.item() * batch_traj.size(0)
            
        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon / len(dataset)
        avg_feat = total_feat / len(dataset)
        
        loss_history.append(avg_loss)
        recon_history.append(avg_recon)
        feat_history.append(avg_feat)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Total: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Feat: {avg_feat:.4f}")
            
    # 4. Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Model weights saved.")
    
    # 5. Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss', color='black', linewidth=2)
    plt.plot(recon_history, label='Reconstruction Loss', color='blue', linestyle='--')
    plt.plot(feat_history, label='Feature Prediction Loss', color='red', linestyle='-.')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Feature Sufficiency Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "fc_training_loss.png"), dpi=150)
    
    # 6. Basic Evaluation Plot
    model.eval()
    with torch.no_grad():
        # Get 3 random indices or just first 3
        test_idx = [0, 1, 2]
        test_traj = X_traj_tensor[test_idx].to(device)
        test_feat = X_feat_tensor[test_idx].to(device)
        recon_test, latent_test = model(test_traj)
        
        # Plot 3 examples
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i in range(3):
            ax = axes[i]
            # Convert back to (101, 2)
            orig = test_traj[i].cpu().numpy() # shape (101, 2)
            rec = recon_test[i].cpu().numpy()
            
            # Inverse transform
            orig_inv = s_traj.inverse_transform(orig)
            rec_inv = s_traj.inverse_transform(rec)
            
            ax.plot(orig_inv[:, 0], color='blue', alpha=0.5, label='Pol1 Orig')
            ax.plot(orig_inv[:, 1], color='red', alpha=0.5, label='Pol2 Orig')
            ax.plot(rec_inv[:, 0], color='blue', linestyle='--', label='Pol1 Recon')
            ax.plot(rec_inv[:, 1], color='red', linestyle='--', label='Pol2 Recon')
            
            # Compute R2 for the features
            f_true = test_feat[i].cpu().numpy()
            f_pred = latent_test[i].cpu().numpy()
            r2 = 1 - (np.sum((f_true - f_pred)**2) / (np.sum((f_true - np.mean(f_true))**2) + 1e-8))
            
            ax.set_title(f"Cell {gids[test_idx[i]]}\nFeat R2: {r2:.2f}")
            if i == 0: ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fc_reconstructions.png"), dpi=150)
        print("💾 Evaluation plots saved.")
    
if __name__ == "__main__":
    train_fc_ae()
