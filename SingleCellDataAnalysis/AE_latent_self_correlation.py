#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==== 1. Configuration ====
AE_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
LATENT_CSV = os.path.join(AE_DIR, "ae_latent_features.csv")
OUTPUT_PLOT = os.path.join(AE_DIR, "ae_latent_self_correlation.png")

# ==== 2. Load Data ====
print("📥 Loading AE latent features...")
df = pd.read_csv(LATENT_CSV, index_col=0)
latent_dims = ['Latent_1', 'Latent_2', 'Latent_3']

# ==== 3. Correlation Matrix ====
corr_matrix = df[latent_dims].corr()
print("\n📊 Pearson Correlation Matrix:")
print(corr_matrix)

# ==== 4. Visualization (Pairplot) ====
sns.set_theme(style="ticks")
# Add experiment info for coloring in pairplot
plt.figure(figsize=(10, 8))
g = sns.pairplot(df, vars=latent_dims, hue='experiment', palette='Set2', 
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'})
g.fig.suptitle("Self-Correlation & Distributions of 3D Latent Space", y=1.02, fontsize=16)

plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"💾 Plot saved to {OUTPUT_PLOT}")
plt.show()
