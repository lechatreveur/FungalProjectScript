#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==== 1. Configuration ====
FINAL_4D_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/fc_final_4d/"
CLUSTERED_CSV = os.path.join(FINAL_4D_DIR, "fc_ae_4d_clustered.csv")
OUTPUT_PDF = os.path.join(FINAL_4D_DIR, "fc_all_traces_gallery.pdf")
OUTPUT_SVG_UMAP = os.path.join(FINAL_4D_DIR, "fc_umap_all_ids.svg")

EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

def generate_static_deliverables():
    # 1. Load Data
    print("📥 Loading data...")
    df = pd.read_csv(CLUSTERED_CSV, index_col=0)
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    
    # 2. Generate High-Res SVG UMAP with ALL IDs
    print("🎨 Generating High-Res SVG UMAP with all IDs...")
    plt.figure(figsize=(25, 25))
    scatter = plt.scatter(df['UMAP1'], df['UMAP2'], c=df['phenotype_cluster'], cmap='tab10', s=20, alpha=0.6)
    
    for i, gid in enumerate(gids):
        plt.text(df.loc[gid, 'UMAP1'], df.loc[gid, 'UMAP2'], gid, fontsize=5, alpha=0.7, verticalalignment='center')
        
    plt.title("4D Phenotypic Manifold (Searchable IDs)", fontsize=20)
    plt.savefig(OUTPUT_SVG_UMAP, format='svg')
    print(f"💾 SVG UMAP saved to {OUTPUT_SVG_UMAP}")

    # 3. Generate Multi-page PDF Gallery (20 traces per page)
    print("📑 Generating Multi-page PDF Trace Gallery...")
    n_per_page = 20
    n_pages = int(np.ceil(len(gids) / n_per_page))
    
    with PdfPages(OUTPUT_PDF) as pdf:
        for p in range(n_pages):
            fig, axes = plt.subplots(5, 4, figsize=(15, 18))
            axes = axes.flatten()
            
            start_idx = p * n_per_page
            for i in range(n_per_page):
                curr_idx = start_idx + i
                ax = axes[i]
                if curr_idx < len(gids):
                    gid = gids[curr_idx]
                    traj_raw = s_traj.inverse_transform(X_traj[curr_idx])
                    
                    ax.plot(traj_raw[:, 0], color='blue', alpha=0.7)
                    ax.plot(traj_raw[:, 1], color='red', alpha=0.7)
                    ax.set_title(f"{gid}", fontsize=8)
                    ax.set_ylim(0, 110)
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            if (p+1) % 5 == 0: print(f"   Processed {p+1}/{n_pages} pages...")

    print(f"💾 Trace gallery PDF saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_static_deliverables()
