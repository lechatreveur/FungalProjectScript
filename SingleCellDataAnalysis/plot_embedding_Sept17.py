#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_embedding_Sept17.py

Projects the PCA feature space of the Sept 17 experiment into 2D
using both UMAP and t-SNE, then colours cells by PC1, PC2, and PC3.

Layout: 2 rows (UMAP / t-SNE) × 3 cols (PC1 / PC2 / PC3 colouring).

Input : pca_latest_feature_scores_Sept17.csv
         (cells × PC1…PC5 — only PC1-3 used here for colouring)
         The raw PCA feature matrix is rebuilt via load_sept17_latest_features
         so that UMAP / t-SNE run on the full feature space, not just PCA scores.
Output: PCA_Extremes/embedding_umap_tsne.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

sys.path.insert(0, "/Users/user/Documents/Python_Scripts/FungalProjectScript/")
from SingleCellDataAnalysis.PCA_utils import load_sept17_latest_features

# ── Config ────────────────────────────────────────────────────────────────────
EXP_DIR    = "/Volumes/X10 Pro/Movies/2025_09_17/"
PCA_CSV    = os.path.join(EXP_DIR, "pca_latest_feature_scores_Sept17.csv")
OUTPUT_DIR = os.path.join(EXP_DIR, "PCA_Extremes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.25, random_state=42)
TSNE_PARAMS = dict(perplexity=20, max_iter=2000, random_state=42, learning_rate="auto",
                   init="pca")

# ── Load data ─────────────────────────────────────────────────────────────────
print("📥  Loading PCA scores ...")
df_pca = pd.read_csv(PCA_CSV).set_index("cell_id")

print("📥  Rebuilding raw feature matrix for embeddings ...")
df_features = load_sept17_latest_features(EXP_DIR)   # 157 × 9 features
# Align indices (inner join keeps only cells present in both)
shared_idx = df_features.index.intersection(df_pca.index)
df_features = df_features.loc[shared_idx]
df_pca      = df_pca.loc[shared_idx]
print(f"   {len(shared_idx)} cells used.\n")

# Standardise feature matrix (same as PCA pipeline)
X = df_features.values
X_scaled = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))

# ── Compute embeddings ────────────────────────────────────────────────────────
print("🔄  Running UMAP …")
reducer = umap.UMAP(**UMAP_PARAMS)
Z_umap  = reducer.fit_transform(X_scaled)

print("🔄  Running t-SNE …")
Z_tsne  = TSNE(**TSNE_PARAMS).fit_transform(X_scaled)

# ── Colour columns: PC1, PC2, PC3 ─────────────────────────────────────────────
pc_labels = ["PC1", "PC2", "PC3"]
pc_data   = {pc: df_pca[pc].values for pc in pc_labels}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("UMAP & t-SNE of Sept 17 cell features\n"
             "(colour = PC score)", fontsize=14, fontweight="bold", y=0.98)

CMAP   = "coolwarm"
MSIZE  = 40

for col, pc in enumerate(pc_labels):
    vals  = pc_data[pc]
    vmin, vmax = vals.min(), vals.max()

    for row, (Z, method) in enumerate([(Z_umap, "UMAP"), (Z_tsne, "t-SNE")]):
        ax = axes[row, col]
        sc = ax.scatter(Z[:, 0], Z[:, 1],
                        c=vals, cmap=CMAP, vmin=vmin, vmax=vmax,
                        s=MSIZE, alpha=0.85, linewidths=0.3, edgecolors="white")
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label(pc, fontsize=9)
        cb.ax.tick_params(labelsize=8)

        ax.set_title(f"{method} — coloured by {pc}", fontsize=10)
        ax.set_xlabel(f"{method}1", fontsize=8)
        ax.set_ylabel(f"{method}2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", "datalim")

        # Annotate the 3 extreme cells per end so they can be traced back
        sorted_idx = np.argsort(vals)
        for i in list(sorted_idx[:2]) + list(sorted_idx[-2:]):
            cell_id = shared_idx[i]
            ax.annotate(str(cell_id),
                        (Z[i, 0], Z[i, 1]),
                        fontsize=6, color="black", alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(OUTPUT_DIR, "embedding_umap_tsne.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n✅  Saved → {out_path}")

# ── Also save a quick PC1 vs PC2 scatter for reference ────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 5))
sc = ax2.scatter(df_pca["PC1"], df_pca["PC2"],
                 c=df_pca["PC3"], cmap=CMAP, s=50, alpha=0.85,
                 linewidths=0.3, edgecolors="white")
cb = fig2.colorbar(sc, ax=ax2, pad=0.02)
cb.set_label("PC3", fontsize=9)
ax2.set_xlabel("PC1", fontsize=10)
ax2.set_ylabel("PC2", fontsize=10)
ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax2.axvline(0, color="grey", linewidth=0.5, linestyle="--")
ax2.set_title("PC1 vs PC2 (colour = PC3)", fontsize=11, fontweight="bold")
fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "PC1_vs_PC2_scatter.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"✅  Saved → {out2}")
