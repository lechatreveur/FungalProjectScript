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
import seaborn as sns
import umap
from sklearn.manifold import TSNE

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.AE_data_loader import load_and_preprocess_trajectories
from SingleCellDataAnalysis.AE_model import TrajectoryAutoencoder

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
}
OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 500
BATCH_SIZE = 32
LR = 1e-3
LATENT_DIM = 8

# ==== 2. Load Data ====
print("📥 Loading and preprocessing trajectories...")
X_np, global_ids, labels, scaler = load_and_preprocess_trajectories(EXPERIMENTS)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_np, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"✅ Data shape: {X_tensor.shape}")

# ==== 3. Initialize Model & Optimizer ====
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Using device: {device}")

model = TrajectoryAutoencoder(seq_len=101, in_channels=2, latent_dim=LATENT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# ==== 4. Training Loop ====
print("🚀 Starting training...")
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        batch_x = batch[0].to(device)
        
        optimizer.zero_grad()
        recon_x, _ = model(batch_x)
        loss = criterion(recon_x, batch_x)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_x.size(0)
        
    epoch_loss /= len(dataloader.dataset)
    history.append(epoch_loss)
    
    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss:.4f}")

# Save the trained model weights
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "ae_model.pth"))
print("💾 Model weights saved.")

# Plot Training Loss
plt.figure(figsize=(6, 4))
plt.plot(history)
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_training_loss.png"), dpi=150)
plt.close()

# ==== 5. Inference & Evaluation ====
model.eval()
with torch.no_grad():
    X_dev = X_tensor.to(device)
    X_recon_dev, Z_dev = model(X_dev)
    X_recon_np = X_recon_dev.cpu().numpy()
    Z_np = Z_dev.cpu().numpy()

# Reverse scaling for plotting reconstructions
# X_np shape: (N, T, C). We need (N*T, C) for scaler.inverse_transform
def inverse_scale(tensor_3d):
    N, T, C = tensor_3d.shape
    flat = tensor_3d.reshape(-1, C)
    flat_inv = scaler.inverse_transform(flat)
    return flat_inv.reshape(N, T, C)

X_orig_inv = inverse_scale(X_np)
X_recon_inv = inverse_scale(X_recon_np)

# Plot reconstructions for 5 random cells
np.random.seed(42)
sample_indices = np.random.choice(len(global_ids), 5, replace=False)

fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
for i, idx in enumerate(sample_indices):
    ax = axes[i]
    cell_id = global_ids[idx]
    exp_lbl = labels[idx]
    
    # Original
    ax.plot(X_orig_inv[idx, :, 0], label="Pol1 (Orig)", color="blue", alpha=0.7)
    ax.plot(X_orig_inv[idx, :, 1], label="Pol2 (Orig)", color="red", alpha=0.7)
    
    # Reconstructed
    ax.plot(X_recon_inv[idx, :, 0], label="Pol1 (Recon)", color="cyan", linestyle="--")
    ax.plot(X_recon_inv[idx, :, 1], label="Pol2 (Recon)", color="orange", linestyle="--")
    
    ax.set_title(f"{exp_lbl} - Cell: {cell_id}")
    ax.set_ylabel("Intensity")
    if i == 0:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

axes[-1].set_xlabel("Time Point (Aligned Frame)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_reconstructions.png"), dpi=150)
plt.close()

# ==== 6. Latent Space Manifold Learning (UMAP/t-SNE) ====
print("🌌 Generating UMAP and t-SNE embeddings from latent space...")

# Save Latent Representations
df_latent = pd.DataFrame(Z_np, columns=[f"Latent_{i+1}" for i in range(LATENT_DIM)])
df_latent['global_cell_id'] = global_ids
df_latent['experiment'] = labels
df_latent.set_index('global_cell_id', inplace=True)
df_latent.to_csv(os.path.join(OUTPUT_DIR, "ae_latent_features.csv"))

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_embedding = reducer.fit_transform(Z_np)
df_latent['UMAP1'] = umap_embedding[:, 0]
df_latent['UMAP2'] = umap_embedding[:, 1]

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embedding = tsne.fit_transform(Z_np)
df_latent['tSNE1'] = tsne_embedding[:, 0]
df_latent['tSNE2'] = tsne_embedding[:, 1]

sns.set_theme(style="whitegrid")

# Plot UMAP
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_latent, x='UMAP1', y='UMAP2', 
    hue='experiment', palette='Set2', alpha=0.8, s=50
)
plt.title(f"UMAP of Autoencoder Latent Space (dim={LATENT_DIM})")
plt.legend(title='Experiment')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_umap.png"), dpi=150)
plt.close()

# Plot t-SNE
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_latent, x='tSNE1', y='tSNE2', 
    hue='experiment', palette='Set2', alpha=0.8, s=50
)
plt.title(f"t-SNE of Autoencoder Latent Space (dim={LATENT_DIM})")
plt.legend(title='Experiment')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_tsne.png"), dpi=150)
plt.close()

print("🚀 Autoencoder Analysis Complete.")
