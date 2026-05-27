#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ==== 1. Configuration ====
AE_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
LATENT_CSV = os.path.join(AE_DIR, "ae_latent_features.csv")
OUTPUT_PLOT = os.path.join(AE_DIR, "ae_latent_3d.png")

# ==== 2. Load Data ====
print("📥 Loading AE latent features...")
df = pd.read_csv(LATENT_CSV, index_col=0)
latent_dims = ['Latent_1', 'Latent_2', 'Latent_3']

# ==== 3. 3D Plotting ====
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Labels and interpretation
# Latent 1: Dominant Pole Strength
# Latent 2: Global Intensity
# Latent 3: Signal Variance/Noise

# Use seaborn palette for consistent colors
palette = sns.color_palette("Set2", n_colors=df['experiment'].nunique())
color_map = {exp: palette[i] for i, exp in enumerate(df['experiment'].unique())}

for exp in df['experiment'].unique():
    sub = df[df['experiment'] == exp]
    ax.scatter(sub['Latent_1'], sub['Latent_2'], sub['Latent_3'], 
               label=exp, alpha=0.6, s=30, color=color_map[exp])

ax.set_xlabel('Latent 1 (Strength)')
ax.set_ylabel('Latent 2 (Intensity)')
ax.set_zlabel('Latent 3 (Variance)')
ax.set_title('3D Latent Space Manifold (431 Cells)', fontsize=14)
ax.legend(title='Experiment')

# Adjust view angle for best visibility
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"💾 3D Plot saved to {OUTPUT_PLOT}")
plt.show()
