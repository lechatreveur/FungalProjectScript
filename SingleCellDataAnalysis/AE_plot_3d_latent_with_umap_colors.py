#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==== 1. Configuration ====
AE_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
LATENT_CSV = os.path.join(AE_DIR, "ae_latent_features.csv")
OUTPUT_PLOT = os.path.join(AE_DIR, "ae_3d_latent_with_umap_colors.png")

# ==== 2. Load Data ====
print("📥 Loading AE latent features and UMAP coordinates...")
df = pd.read_csv(LATENT_CSV, index_col=0)

# Ensure UMAP coordinates are present
if 'UMAP1' not in df.columns:
    import umap
    latent_dims = ['Latent_1', 'Latent_2', 'Latent_3']
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(df[latent_dims].values)
    df['UMAP1'] = embedding[:, 0]
    df['UMAP2'] = embedding[:, 1]

# ==== 3. Create Bivariate Color Mapping ====
# Map UMAP1 to Red and UMAP2 to Green to visualize the 2D manifold in 3D space
def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

u1_norm = normalize(df['UMAP1'])
u2_norm = normalize(df['UMAP2'])

# Create RGB colors: Red=UMAP1, Green=UMAP2, Blue=Fixed
colors = np.stack([u1_norm, u2_norm, np.ones_like(u1_norm)*0.5], axis=1)

# ==== 4. Plotting ====
fig = plt.figure(figsize=(15, 7))

# Panel A: 2D UMAP
ax1 = fig.add_subplot(121)
ax1.scatter(df['UMAP1'], df['UMAP2'], c=colors, s=30, alpha=0.8)
ax1.set_title("2D UMAP (Bivariate Color Map)")
ax1.set_xlabel("UMAP 1 (Red Gradient)")
ax1.set_ylabel("UMAP 2 (Green Gradient)")

# Panel B: 3D Latent Space with same colors
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df['Latent_1'], df['Latent_2'], df['Latent_3'], c=colors, s=30, alpha=0.6)
ax2.set_xlabel('Latent 1')
ax2.set_ylabel('Latent 2')
ax2.set_zlabel('Latent 3')
ax2.set_title("3D Latent Space (Colored by UMAP Position)")
ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"💾 Bivariate plot saved to {OUTPUT_PLOT}")
plt.show()
