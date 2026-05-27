#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.AE_model import TrajectoryAutoencoder

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

SUFF_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/sufficiency_test/"
LATENT_CSV = os.path.join(SUFF_DIR, "fc_ae_latent_features.csv")
MODEL_PATH = os.path.join(SUFF_DIR, "fc_ae_model.pth")
OUTPUT_EXAMPLES = os.path.join(SUFF_DIR, "fc_ae_umap_examples.png")
OUTPUT_HIGHLIGHT = os.path.join(SUFF_DIR, "fc_ae_umap_highlighted.png")

def plot_fc_umap_examples():
    # 1. Load Data
    print("📥 Loading data...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    df_latent = pd.read_csv(LATENT_CSV, index_col=0)
    
    # Generate UMAP coordinates if not in CSV (they aren't)
    import umap
    print("🌌 Generating UMAP for selection...")
    feature_cols = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(df_latent[feature_cols].values)
    df_latent['UMAP1'] = embedding[:, 0]
    df_latent['UMAP2'] = embedding[:, 1]

    # 2. Select 20 Diverse Examples
    print("🎯 Selecting 20 representative examples...")
    # Use a simple grid or random sampling across UMAP space
    df_latent['u1_bin'] = pd.cut(df_latent['UMAP1'], bins=4)
    df_latent['u2_bin'] = pd.cut(df_latent['UMAP2'], bins=5)
    
    selected_gids = []
    for (b1, b2), group in df_latent.groupby(['u1_bin', 'u2_bin'], observed=True):
        if not group.empty:
            selected_gids.append(group.index[0])
            if len(selected_gids) >= 20:
                break
    
    # Fill up if less than 20
    if len(selected_gids) < 20:
        remaining = df_latent[~df_latent.index.isin(selected_gids)].index.tolist()
        selected_gids.extend(remaining[:20-len(selected_gids)])

    # 3. Load Model for Reconstructions
    print("🧠 Loading model for reconstructions...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TrajectoryAutoencoder(latent_dim=11).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 4. Plot Highlight Map
    plt.figure(figsize=(8, 8))
    plt.scatter(df_latent['UMAP1'], df_latent['UMAP2'], c='lightgrey', alpha=0.3, s=10)
    for i, gid in enumerate(selected_gids):
        row = df_latent.loc[gid]
        plt.scatter(row['UMAP1'], row['UMAP2'], label=f"#{i+1}", s=50)
        plt.text(row['UMAP1'], row['UMAP2'], str(i+1), fontsize=12, fontweight='bold')
    plt.title("20 Selected Examples on 11D Constrained UMAP")
    plt.savefig(OUTPUT_HIGHLIGHT, dpi=150)
    print(f"💾 Highlight map saved to {OUTPUT_HIGHLIGHT}")

    # 5. Plot Trajectories and Features
    print("📈 Plotting trajectories and feature values...")
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, gid in enumerate(selected_gids):
            ax = axes[i]
            
            # Get original trajectory
            idx = gids.index(gid)
            traj_tensor = torch.tensor(X_traj[idx:idx+1], dtype=torch.float32).to(device)
            traj_raw = s_traj.inverse_transform(X_traj[idx])
            
            # Get reconstructed trajectory
            recon_traj_tensor, _ = model(traj_tensor)
            recon_raw = s_traj.inverse_transform(recon_traj_tensor.cpu().numpy()[0])
            
            # Plot Original
            ax.plot(traj_raw[:, 0], color='blue', label='Pol1 Orig', alpha=0.4)
            ax.plot(traj_raw[:, 1], color='red', label='Pol2 Orig', alpha=0.4)
            
            # Plot Reconstructed (Dashed)
            ax.plot(recon_raw[:, 0], color='blue', linestyle='--', label='Pol1 Recon')
            ax.plot(recon_raw[:, 1], color='red', linestyle='--', label='Pol2 Recon')
            
            # Display 11 features
            f_row = df_latent.loc[gid, feature_cols]
            feat_str = "\n".join([f"{k}: {v:.2f}" for k, v in f_row.items()])
            
            ax.set_title(f"Example #{i+1} | {labels[idx]}\n{gid}", fontsize=10)
            ax.text(1.05, 0.5, feat_str, transform=ax.transAxes, fontsize=8, 
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if i == 0: ax.legend()
        
        # Make Y-axis flexible
        y_min = traj_raw.min()
        y_max = traj_raw.max()
        padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    plt.savefig(OUTPUT_EXAMPLES, dpi=150, bbox_inches='tight')
    print(f"💾 Examples plot saved to {OUTPUT_EXAMPLES}")

if __name__ == "__main__":
    plot_fc_umap_examples()
