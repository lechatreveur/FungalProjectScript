#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pca_loadings_Sept17.py

Produces two panels:
  1. Scree plot  — variance explained per PC + cumulative
  2. Loadings heatmap — features × first 5 PCs
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, "/Users/user/Documents/Python_Scripts/FungalProjectScript/")
from SingleCellDataAnalysis.PCA_utils import load_sept17_latest_features, run_pca_workflow

EXP_DIR    = "/Volumes/X10 Pro/Movies/2025_09_17/"
OUTPUT_DIR = os.path.join(EXP_DIR, "PCA_Extremes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load & run PCA ─────────────────────────────────────────────────────────────
df_features = load_sept17_latest_features(EXP_DIR)
X           = df_features.values
_, pca, _   = run_pca_workflow(X, n_components=len(df_features.columns))

n_pcs       = pca.n_components_
pc_labels   = [f"PC{i+1}" for i in range(n_pcs)]
feat_labels = list(df_features.columns)
loadings    = pca.components_.T          # shape: (n_features, n_pcs)
evr         = pca.explained_variance_ratio_
evr_cum     = np.cumsum(evr)

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.7], wspace=0.35)

# ────────────────────────────────────────────────────────────────────────── (1)
ax_scree = fig.add_subplot(gs[0])

bar_color = "#4C72B0"
line_color = "#DD8452"

bars = ax_scree.bar(pc_labels, evr * 100, color=bar_color, alpha=0.85,
                    edgecolor="white", linewidth=0.6)
ax_scree.set_ylabel("Variance Explained (%)", color=bar_color, fontsize=11)
ax_scree.tick_params(axis="y", labelcolor=bar_color)
ax_scree.set_ylim(0, max(evr * 100) * 1.25)

# Annotate bars
for bar, v in zip(bars, evr):
    ax_scree.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.4,
                  f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

# Cumulative line on twin axis
ax2 = ax_scree.twinx()
ax2.plot(pc_labels, evr_cum * 100, "o-", color=line_color,
         linewidth=2.0, markersize=6)
ax2.set_ylabel("Cumulative Variance (%)", color=line_color, fontsize=11)
ax2.tick_params(axis="y", labelcolor=line_color)
ax2.set_ylim(0, 110)
ax2.axhline(80, linestyle="--", color="grey", linewidth=0.8, alpha=0.6)
ax2.axhline(95, linestyle=":", color="grey", linewidth=0.8, alpha=0.6)
ax2.text(n_pcs - 0.5, 81, "80%", fontsize=7, color="grey", va="bottom")
ax2.text(n_pcs - 0.5, 96, "95%", fontsize=7, color="grey", va="bottom")

ax_scree.set_title("Scree Plot — Variance Explained per PC",
                   fontsize=12, fontweight="bold", pad=10)
ax_scree.set_xlabel("Principal Component", fontsize=10)

# ────────────────────────────────────────────────────────────────────────── (2)
ax_heat = fig.add_subplot(gs[1])

# Show only first 5 PCs in the heatmap (most interpretable)
n_show = min(5, n_pcs)
L      = loadings[:, :n_show]

cmap  = plt.cm.RdBu_r
norm  = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im    = ax_heat.imshow(L, aspect="auto", cmap=cmap, norm=norm)

# Axes labels
ax_heat.set_xticks(range(n_show))
ax_heat.set_xticklabels(pc_labels[:n_show], fontsize=10, fontweight="bold")
ax_heat.set_yticks(range(len(feat_labels)))
ax_heat.set_yticklabels(feat_labels, fontsize=10)
ax_heat.set_xlabel("Principal Component", fontsize=11)
ax_heat.set_title("PCA Loadings (Features × PCs 1–5)",
                  fontsize=12, fontweight="bold", pad=10)

# Annotate each cell with the loading value
for i in range(len(feat_labels)):
    for j in range(n_show):
        val     = L[i, j]
        txt_col = "white" if abs(val) > 0.4 else "black"
        ax_heat.text(j, i, f"{val:+.2f}", ha="center", va="center",
                     fontsize=8, color=txt_col, fontweight="bold")

# Colour bar
cb = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
cb.set_label("Loading", fontsize=10)
cb.ax.tick_params(labelsize=8)

# Add thin horizontal separators between features
for row in range(1, len(feat_labels)):
    ax_heat.axhline(row - 0.5, color="white", linewidth=0.5)

fig.suptitle("PCA: Variance Explained & Feature Loadings — Sept 17 (11 features)",
             fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "pca_loadings_scree.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅  Saved → {out_path}")
