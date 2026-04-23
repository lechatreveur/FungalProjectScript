#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA_latest_features_2025_09_17.py
Perform PCA on the latest biological features for Sept 17 experiment.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.PCA_utils import load_sept17_latest_features, run_pca_workflow

# ==== 1. Configuration ====
EXP_DIR = "/Volumes/X10 Pro/Movies/2025_09_17/"
OUTPUT_CSV = os.path.join(EXP_DIR, "pca_latest_feature_scores_Sept17.csv")

# ==== 2. Load and Preprocess Data ====
print(f"📥 Loading features from {EXP_DIR}...")
try:
    df_features = load_sept17_latest_features(EXP_DIR)
    print(f"✅ Loaded features for {len(df_features)} cells.")
except Exception as e:
    print(f"❌ Failed to load features: {e}")
    sys.exit(1)

# Inspect features
print("\n📊 Feature Summary:")
print(df_features.describe().loc[['mean', 'std']])

# ==== 3. Run PCA ====
print("\n🔍 Running PCA...")
X = df_features.values
X_pca, pca, scaler = run_pca_workflow(X, n_components=5)

# Create a DataFrame for PCA results
pc_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pc_cols, index=df_features.index)

# ==== 4. Save Results ====
print(f"💾 Saving PCA results to {OUTPUT_CSV}")
df_pca.to_csv(OUTPUT_CSV)

# ==== 5. Visualizations ====

# A. Scree Plot
explained_var = pca.explained_variance_ratio_
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(explained_var) + 1), explained_var, 'o-', linewidth=2)
plt.title("Scree Plot: Variance Explained by Principal Components")
plt.xlabel("Component")
plt.ylabel("Variance Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

# B. Loading Plot (Heatmap)
# Show which features contribute to which PC
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=pc_cols, 
    index=df_features.columns
)

plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='vlag', center=0)
plt.title("PCA Loadings: Feature contribution to each PC")
plt.tight_layout()
plt.show()

# C. PCA Score Plot (PC1 vs PC2)
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.6, s=20, c='darkgreen')
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
plt.title("PCA of Sept 17 Latest Features")
plt.grid(True, linestyle='--', alpha=0.5)
# D. Pairplot (PC1 through PC5)
print("🧬 Generating Pairplot...")
sns.pairplot(df_pca, corner=True, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10})
plt.suptitle("PCA Pairplot: All PC Relationships", y=1.02)
pairplot_save = os.path.join(EXP_DIR, "pca_latest_feature_pairplot_Sept17.png")
plt.savefig(pairplot_save, dpi=150, bbox_inches='tight')
plt.show()

print("\n🚀 PCA Analysis Complete.")
