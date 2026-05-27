#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==== 1. Configuration ====
AE_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/autoencoder/"
PCA_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/"

AE_CSV = os.path.join(AE_DIR, "ae_latent_features.csv")
PCA_CSV = os.path.join(PCA_DIR, "pca_combined_features.csv")
OUTPUT_PLOT = os.path.join(AE_DIR, "ae_latent_correlation_heatmap.png")

import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

# ==== 2. Load and Combine Data ====
from SingleCellDataAnalysis.PCA_utils import load_experiment_features

EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
}

print("📥 Loading AE latent features...")
df_ae = pd.read_csv(AE_CSV, index_col=0)

print("📥 Re-loading engineered features from experiments...")
df_list = []
for exp_name, exp_dir in EXPERIMENTS.items():
    df_exp = load_experiment_features(exp_dir)
    df_exp['global_cell_id'] = exp_name + "_" + df_exp.index.astype(str)
    df_exp.set_index('global_cell_id', inplace=True)
    df_list.append(df_exp)

df_pca = pd.concat(df_list)

# Merge on global_cell_id
df_combined = df_ae[['Latent_1', 'Latent_2', 'Latent_3']].join(df_pca, how='inner')

print(f"✅ Merged {len(df_combined)} cells for analysis.")

# Engineered feature names to focus on (excluding temporary ID columns)
engineered_features = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
latent_dims = ['Latent_1', 'Latent_2', 'Latent_3']

# ==== 3. Correlation Analysis ====
print("📊 Calculating Pearson correlations...")
corr_matrix = df_combined.corr()

# Extract only the correlations between Latent Dims and Engineered Features
# We want rows=LatentDims, cols=EngineeredFeatures
sub_corr = corr_matrix.loc[latent_dims, engineered_features]

# ==== 4. Visualization ====
plt.figure(figsize=(12, 6))
sns.heatmap(sub_corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title("Correlation: 3D Latent Space vs. 11 Engineered Features")
plt.xlabel("Engineered Features")
plt.ylabel("Learned Latent Dimensions")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"💾 Heatmap saved to {OUTPUT_PLOT}")

# ==== 5. Automated Interpretation ====
print("\n📝 Automated Interpretation:")
for ld in latent_dims:
    top_corr = sub_corr.loc[ld].abs().sort_values(ascending=False).head(3)
    interpretation = []
    for feat, val in top_corr.items():
        sign = "+" if sub_corr.loc[ld, feat] > 0 else "-"
        interpretation.append(f"{feat} ({sign}{abs(val):.2f})")
    print(f"  - {ld} is most represented by: {', '.join(interpretation)}")
