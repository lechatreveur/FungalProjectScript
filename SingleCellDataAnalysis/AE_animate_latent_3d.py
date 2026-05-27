#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import numpy as np

# ==== 1. Configuration ====
AE_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
LATENT_CSV = os.path.join(AE_DIR, "ae_latent_features.csv")
OUTPUT_GIF = os.path.join(AE_DIR, "ae_latent_3d_rotation.gif")

# ==== 2. Load Data ====
print("📥 Loading AE latent features...")
df = pd.read_csv(LATENT_CSV, index_col=0)

# ==== 3. Animation Setup ====
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

palette = sns.color_palette("Set2", n_colors=df['experiment'].nunique())
color_map = {exp: palette[i] for i, exp in enumerate(df['experiment'].unique())}

def update(frame):
    ax.clear()
    for exp in df['experiment'].unique():
        sub = df[df['experiment'] == exp]
        ax.scatter(sub['Latent_1'], sub['Latent_2'], sub['Latent_3'], 
                   label=exp, alpha=0.6, s=30, color=color_map[exp])
    
    ax.set_xlabel('Latent 1 (Strength)')
    ax.set_ylabel('Latent 2 (Intensity)')
    ax.set_zlabel('Latent 3 (Variance)')
    ax.set_title('3D Latent Space Manifold Rotation', fontsize=14)
    ax.legend(title='Experiment')
    
    # Rotate the view
    ax.view_init(elev=20, azim=frame)
    return fig,

# Create animation: 360 degrees in 120 frames (3 degrees per frame)
print("🎬 Generating animation (120 frames)... This may take a minute.")
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 3), interval=50)

# Save as GIF
try:
    import numpy as np # Ensure numpy is available in the script context
    writer = PillowWriter(fps=20)
    ani.save(OUTPUT_GIF, writer=writer)
    print(f"✅ Animation saved to {OUTPUT_GIF}")
except Exception as e:
    print(f"❌ Error saving animation: {e}")
