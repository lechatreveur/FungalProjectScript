#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA_combined.py
Perform PCA on the 11 latest biological features for combined datasets:
Sept 17, M92, and M93. Generates UMAP and t-SNE embeddings to check for
batch effects.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
import umap

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.PCA_utils import load_experiment_features, run_pca_workflow

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
}

# Output will be saved in the first experiment's directory for now, 
# or a new combined directory. Let's use the SingleCellDataAnalysis folder.
OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pca_combined_features.csv")

# ==== 2. Load and Combine Data ====
df_list = []

for exp_name, exp_dir in EXPERIMENTS.items():
    print(f"📥 Loading features for {exp_name} from {exp_dir}...")
    try:
        df_exp = load_experiment_features(exp_dir)
        df_exp['experiment'] = exp_name
        
        # In Sept 17, the cell IDs might overlap with M92/M93, so we create a global ID
        df_exp['global_cell_id'] = exp_name + "_" + df_exp.index.astype(str)
        
        # Set index
        df_exp.set_index('global_cell_id', inplace=True)
        
        df_list.append(df_exp)
        print(f"✅ Loaded {len(df_exp)} cells from {exp_name}.")
    except Exception as e:
        print(f"❌ Failed to load features for {exp_name}: {e}")

if not df_list:
    print("No data loaded. Exiting.")
    sys.exit(1)

df_combined = pd.concat(df_list)

# Drop non-feature columns for PCA
feature_cols = [c for c in df_combined.columns if c not in ['cell_id', 'experiment']]

print("\n📊 Combined Feature Summary:")
print(f"Total Cells: {len(df_combined)}")
print(df_combined[feature_cols].describe().loc[['mean', 'std']])

# ==== 3. Run PCA ====
print("\n🔍 Running PCA...")
X = df_combined[feature_cols].values
X_pca, pca, scaler = run_pca_workflow(X, n_components=5)

# Create a DataFrame for PCA results
pc_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pc_cols, index=df_combined.index)
df_pca['experiment'] = df_combined['experiment']

# ==== 4. Run UMAP & t-SNE ====
print("\n🌌 Running UMAP and t-SNE on Principal Components...")
# We use the top PCs (or scaled X) for UMAP/t-SNE. Usually, using the scaled original features or top PCs is fine.
# We'll use the scaled features X_scaled (from scaler), or just X_pca to denoise.
X_scaled = scaler.transform(X)

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_embedding = reducer.fit_transform(X_scaled)
df_pca['UMAP1'] = umap_embedding[:, 0]
df_pca['UMAP2'] = umap_embedding[:, 1]

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embedding = tsne.fit_transform(X_scaled)
df_pca['tSNE1'] = tsne_embedding[:, 0]
df_pca['tSNE2'] = tsne_embedding[:, 1]


# ==== 5. Save Results ====
print(f"💾 Saving PCA/UMAP results to {OUTPUT_CSV}")
df_pca.to_csv(OUTPUT_CSV)


# ==== 6. Visualizations ====
sns.set_theme(style="whitegrid")

# A. Scree Plot
explained_var = pca.explained_variance_ratio_
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(explained_var) + 1), explained_var, 'o-', linewidth=2)
plt.title("Scree Plot: Variance Explained by Principal Components (Combined)")
plt.xlabel("Component")
plt.ylabel("Variance Ratio")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_scree_plot.png"), dpi=150)
plt.show()

# B. Loading Plot (Heatmap)
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=pc_cols, 
    index=feature_cols
)

plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='vlag', center=0)
plt.title("PCA Loadings: Feature contribution to each PC (Combined)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_pca_loadings.png"), dpi=150)
plt.show()

# C. PCA Score Plot (PC1 vs PC2)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca, x='PC1', y='PC2', 
    hue='experiment', palette='Set2', alpha=0.8, s=40
)
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
plt.title("PCA of Combined Datasets (Sept17, M92, M93)")
plt.legend(title='Experiment')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_pca_scatter.png"), dpi=150)
plt.show()

# D. UMAP
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca, x='UMAP1', y='UMAP2', 
    hue='experiment', palette='Set2', alpha=0.8, s=40
)
plt.title("UMAP of Combined Datasets")
plt.legend(title='Experiment')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_umap.png"), dpi=150)
plt.show()

# E. t-SNE
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca, x='tSNE1', y='tSNE2', 
    hue='experiment', palette='Set2', alpha=0.8, s=40
)
plt.title("t-SNE of Combined Datasets")
plt.legend(title='Experiment')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_tsne.png"), dpi=150)
plt.show()

print("\n🚀 Combined Analysis Complete.")
