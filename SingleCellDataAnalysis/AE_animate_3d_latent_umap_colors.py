#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# ==== 1. Configuration ====
AE_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
LATENT_CSV = os.path.join(AE_DIR, "ae_latent_features.csv")
OUTPUT_GIF = os.path.join(AE_DIR, "ae_3d_latent_umap_colors_rotation.gif")

# ==== 2. Load Data ====
print("📥 Loading AE latent features and UMAP coordinates...")
df = pd.read_csv(LATENT_CSV, index_col=0)

if 'UMAP1' not in df.columns:
    import umap
    latent_dims = ['Latent_1', 'Latent_2', 'Latent_3']
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(df[latent_dims].values)
    df['UMAP1'] = embedding[:, 0]
    df['UMAP2'] = embedding[:, 1]

# ==== 3. Bivariate Color Mapping ====
def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

u1_norm = normalize(df['UMAP1'])
u2_norm = normalize(df['UMAP2'])
colors = np.stack([u1_norm, u2_norm, np.ones_like(u1_norm)*0.5], axis=1)

# ==== 4. Animation Setup ====
fig = plt.figure(figsize=(15, 7))

# Panel A: Static 2D UMAP (Reference)
ax1 = fig.add_subplot(121)
ax1.scatter(df['UMAP1'], df['UMAP2'], c=colors, s=30, alpha=0.8)
ax1.set_title("2D UMAP Reference")
ax1.set_xlabel("UMAP 1")
ax1.set_ylabel("UMAP 2")

# Panel B: Rotating 3D Latent Space
ax2 = fig.add_subplot(122, projection='3d')

def update(frame):
    ax2.clear()
    ax2.scatter(df['Latent_1'], df['Latent_2'], df['Latent_3'], c=colors, s=30, alpha=0.6)
    ax2.set_xlabel('Latent 1 (Strength)')
    ax2.set_ylabel('Latent 2 (Intensity)')
    ax2.set_zlabel('Latent 3 (Variance)')
    ax2.set_title("3D Latent Space (UMAP-Colored)")
    
    # Complex rotation: Azimuth goes 0-360, Elevation oscillates
    azim = frame
    elev = 20 + 15 * np.sin(np.radians(frame * 2)) # Oscillate elevation
    
    ax2.view_init(elev=elev, azim=azim)
    return fig,

# 120 frames = 360 degrees
print("🎬 Generating animation (120 frames)... This may take a minute.")
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 3), interval=50)

# Save as GIF
writer = PillowWriter(fps=20)
ani.save(OUTPUT_GIF, writer=writer)
print(f"✅ Animation saved to {OUTPUT_GIF}")
