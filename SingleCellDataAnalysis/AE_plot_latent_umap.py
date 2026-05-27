#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== 1. Configuration ====
AE_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
LATENT_CSV = os.path.join(AE_DIR, "ae_latent_features.csv")
OUTPUT_PLOT = os.path.join(AE_DIR, "ae_umap_colored_by_latent.png")

# ==== 2. Load Data ====
print("📥 Loading AE latent features and UMAP coordinates...")
df = pd.read_csv(LATENT_CSV, index_col=0)

# UMAP coordinates might not be in the CSV if I didn't save them in the latest train_evaluate run?
# Let's check columns.
print("Columns:", df.columns.tolist())

# If UMAP1/UMAP2 are missing, we'll re-calculate them here deterministically.
if 'UMAP1' not in df.columns:
    print("🌌 Re-calculating UMAP coordinates...")
    import umap
    latent_dims = ['Latent_1', 'Latent_2', 'Latent_3']
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(df[latent_dims].values)
    df['UMAP1'] = embedding[:, 0]
    df['UMAP2'] = embedding[:, 1]

# ==== 3. Plotting ====
latent_dims = ['Latent_1', 'Latent_2', 'Latent_3']
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

# Define titles based on our previous interpretation
titles = {
    'Latent_1': 'Latent 1 (Dominant Pole Strength)',
    'Latent_2': 'Latent 2 (Global Intensity)',
    'Latent_3': 'Latent 3 (Signal Variance/Noise)'
}

for i, ld in enumerate(latent_dims):
    ax = axes[i]
    # We use 'viridis' or 'magma' for continuous values
    scatter = ax.scatter(df['UMAP1'], df['UMAP2'], c=df[ld], cmap='viridis', s=30, alpha=0.8, edgecolor='none')
    
    # Add a colorbar for each subplot
    plt.colorbar(scatter, ax=ax, label='Latent Value')
    
    ax.set_title(titles.get(ld, ld))
    ax.set_xlabel("UMAP 1")
    if i == 0:
        ax.set_ylabel("UMAP 2")
    
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"💾 Plot saved to {OUTPUT_PLOT}")
plt.show()
