#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.AE_data_loader import load_and_preprocess_trajectories
from SingleCellDataAnalysis.AE_model import TrajectoryAutoencoder

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
}
OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/sweep/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIMS = [2, 3, 4, 5, 6, 8, 10]
NUM_REPS = 5
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3

# ==== 2. Load and Split Data ====
print("📥 Loading and preprocessing trajectories...")
X_np, _, _, _ = load_and_preprocess_trajectories(EXPERIMENTS)

# Train/Val Split (80/20)
X_train_np, X_val_np = train_test_split(X_np, test_size=0.20, random_state=42)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_val = torch.tensor(X_val_np, dtype=torch.float32)

train_dataset = TensorDataset(X_train)
val_dataset = TensorDataset(X_val)

print(f"✅ Training set: {X_train.shape[0]} cells")
print(f"✅ Validation set: {X_val.shape[0]} cells")

# ==== 3. Training Function ====
def train_trial(latent_dim, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = TrajectoryAutoencoder(seq_len=101, in_channels=2, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, _ = model(batch_x)
            loss = criterion(recon_x, batch_x)
            loss.backward()
            optimizer.step()
            
    # Final Validation Loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch_x = batch[0].to(device)
            recon_x, _ = model(batch_x)
            loss = criterion(recon_x, batch_x)
            val_loss += loss.item() * batch_x.size(0)
    
    return val_loss / len(val_dataset)

# ==== 4. Run Sweep ====
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Using device: {device}")

results = []

for ld in LATENT_DIMS:
    print(f"🔍 Testing Latent Dimension: {ld}")
    ld_losses = []
    for rep in range(NUM_REPS):
        seed = 42 + rep
        loss = train_trial(ld, seed, device)
        ld_losses.append(loss)
        print(f"  - Rep {rep+1}/{NUM_REPS}: Val Loss = {loss:.4f}")
    
    results.append({
        'latent_dim': ld,
        'mean_val_loss': np.mean(ld_losses),
        'std_val_loss': np.std(ld_losses)
    })

# ==== 5. Summary and Plotting ====
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "latent_dim_sweep_results.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.errorbar(df_results['latent_dim'], df_results['mean_val_loss'], yerr=df_results['std_val_loss'], 
             marker='o', linestyle='-', capsize=5, color='teal')
plt.title("Autoencoder Latent Dimension Sweep")
plt.xlabel("Latent Dimension Size")
plt.ylabel("Mean Validation MSE Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(LATENT_DIMS)

# Annotate Elbow
plt.annotate('Elbow?', xy=(df_results['latent_dim'].iloc[2], df_results['mean_val_loss'].iloc[2]), 
             xytext=(5, 0.05), arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "latent_dim_sweep_plot.png"), dpi=150)
plt.show()

print("\n🚀 Sweep Complete. Results saved to:", OUTPUT_DIR)
print(df_results)
