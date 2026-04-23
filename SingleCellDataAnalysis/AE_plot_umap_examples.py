#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.AE_data_loader import load_and_preprocess_trajectories

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
}
OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
LATENT_CSV = os.path.join(OUTPUT_DIR, "ae_latent_features.csv")

# ==== 2. Load Data ====
print("📥 Loading original trajectories...")
X_np, global_ids, labels, scaler = load_and_preprocess_trajectories(EXPERIMENTS)

# Reverse scale to get actual biological intensity values
def inverse_scale(tensor_3d):
    N, T, C = tensor_3d.shape
    flat = tensor_3d.reshape(-1, C)
    flat_inv = scaler.inverse_transform(flat)
    return flat_inv.reshape(N, T, C)

X_orig = inverse_scale(X_np)

print("📥 Loading UMAP coordinates from latent space...")
df_latent = pd.read_csv(LATENT_CSV, index_col=0)
z_cols = [f"Latent_{i+1}" for i in range(8)]
z_np = df_latent.loc[global_ids][z_cols].values

import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_coords = reducer.fit_transform(z_np)

# ==== 3. Select 20 Representative Cells ====
# To evenly sample the structures mentioned (the curves and groups), 
# we'll use KMeans on the 2D UMAP space to find 20 spread-out clusters,
# and select the cell closest to each cluster center.

print("🔍 Selecting 20 representative cells using KMeans on UMAP space...")
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
kmeans.fit(umap_coords)
centers = kmeans.cluster_centers_

selected_indices = []
for center in centers:
    # Find the point closest to this center
    dists = np.sum((umap_coords - center)**2, axis=1)
    closest_idx = np.argmin(dists)
    selected_indices.append(closest_idx)

# ==== 4. Plot ====
print("📈 Plotting trajectories...")
fig, axes = plt.subplots(4, 5, figsize=(25, 16), sharex=True, sharey=True)
axes = axes.flatten()

# We want to color them based on their general position to easily cross-reference
# Let's sort the selected indices by their UMAP X coordinate to group them somewhat
selected_indices = sorted(selected_indices, key=lambda idx: umap_coords[idx][0])

for i, idx in enumerate(selected_indices):
    ax = axes[i]
    cell_id = global_ids[idx]
    u1, u2 = umap_coords[idx]
    
    # Plot Pol1 and Pol2
    ax.plot(X_orig[idx, :, 0], label="Pol1", color="blue", lw=2)
    ax.plot(X_orig[idx, :, 1], label="Pol2", color="red", lw=2, linestyle="--")
    
    # Clean title with UMAP coords
    ax.set_title(f"UMAP: ({u1:.1f}, {u2:.1f})\nID: {cell_id}", fontsize=12)
    
    if i >= 15:
        ax.set_xlabel("Time Point (Frames)", fontsize=10)
    if i % 5 == 0:
        ax.set_ylabel("Intensity", fontsize=10)
        
    if i == 0:
        ax.legend(loc="upper left")
        
    # Add a subtle grid
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "ae_umap_examples.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"💾 Plot saved to {save_path}")

# Additionally, generate a scatter plot of the UMAP highlighting these 20 points
plt.figure(figsize=(10, 8))
plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgray', alpha=0.5, label='All Cells')
plt.scatter(umap_coords[selected_indices, 0], umap_coords[selected_indices, 1], c='red', s=50, edgecolor='black', label='Selected 20')

# Label the 20 points with their subplot number (1 to 20)
for i, idx in enumerate(selected_indices):
    plt.annotate(str(i+1), (umap_coords[idx, 0], umap_coords[idx, 1]), 
                 textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, weight='bold')

plt.title("UMAP Highlighting the 20 Selected Examples")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend()
plt.tight_layout()
save_path_scatter = os.path.join(OUTPUT_DIR, "ae_umap_highlighted.png")
plt.savefig(save_path_scatter, dpi=150)
print(f"💾 Highlighted scatter saved to {save_path_scatter}")
