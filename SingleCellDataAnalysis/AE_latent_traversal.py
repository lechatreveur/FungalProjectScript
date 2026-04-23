#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
MODEL_PATH = os.path.join(OUTPUT_DIR, "ae_model.pth")
LATENT_CSV = os.path.join(OUTPUT_DIR, "ae_latent_features.csv")
LATENT_DIM = 8

# ==== 2. Initialization ====
print("📥 Loading scaler and dataset (required for inverse transform)...")
# We just need the scaler to inverse-transform the decoded trajectories
_, _, _, scaler = load_and_preprocess_trajectories(EXPERIMENTS)

print("🧠 Loading trained Autoencoder model...")
device = torch.device("cpu")
model = TrajectoryAutoencoder(seq_len=101, in_channels=2, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

print("📊 Loading latent embeddings to determine bounds...")
df_latent = pd.read_csv(LATENT_CSV, index_col=0)
z_cols = [f"Latent_{i+1}" for i in range(LATENT_DIM)]

# Calculate mean and standard deviation for each latent dimension
z_means = df_latent[z_cols].mean().values
z_stds = df_latent[z_cols].std().values

# ==== 3. Latent Traversal ====
print("🚶‍♂️ Performing latent traversal...")

# 5 steps from -2 std to +2 std
num_steps = 5
multipliers = np.linspace(-2, 2, num_steps)
colors = plt.cm.coolwarm(np.linspace(0, 1, num_steps))

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

# We need a helper to inverse transform a batch of trajectories
def inverse_scale(tensor_3d):
    N, T, C = tensor_3d.shape
    flat = tensor_3d.reshape(-1, C)
    flat_inv = scaler.inverse_transform(flat)
    return flat_inv.reshape(N, T, C)

with torch.no_grad():
    for dim in range(LATENT_DIM):
        ax = axes[dim]
        
        # We will plot pol1 and pol2 for each step
        for step_idx, mult in enumerate(multipliers):
            # Start with the mean vector
            z_sweep = np.copy(z_means)
            # Vary only the current dimension
            z_sweep[dim] = z_means[dim] + mult * z_stds[dim]
            
            # Decode
            z_tensor = torch.tensor(z_sweep, dtype=torch.float32).unsqueeze(0) # (1, 8)
            x_recon = model.decode(z_tensor)
            x_recon = x_recon.transpose(1, 2) # (1, 101, 2)
            
            # Inverse scale
            x_recon_inv = inverse_scale(x_recon.numpy())[0] # (101, 2)
            
            # Plot Pol1 (solid) and Pol2 (dashed)
            # Alpha scales so extremes are more visible
            ax.plot(x_recon_inv[:, 0], color=colors[step_idx], linestyle='-', linewidth=2, alpha=0.8)
            ax.plot(x_recon_inv[:, 1], color=colors[step_idx], linestyle='--', linewidth=1.5, alpha=0.6)
            
        ax.set_title(f"Dimension {dim+1} Traversal\n(-2σ to +2σ)")
        if dim >= 4:
            ax.set_xlabel("Time Point")
        if dim % 4 == 0:
            ax.set_ylabel("Intensity")

# Custom Legend
import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], color=colors[0], lw=2, label='-2σ'),
    mlines.Line2D([0], [0], color=colors[len(colors)//2], lw=2, label='Mean'),
    mlines.Line2D([0], [0], color=colors[-1], lw=2, label='+2σ'),
    mlines.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Pol1'),
    mlines.Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Pol2')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize='large')

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "ae_latent_traversal.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"💾 Latent traversal plot saved to {save_path}")

plt.show()
