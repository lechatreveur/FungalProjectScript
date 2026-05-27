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
from SingleCellDataAnalysis.FC_AE_sweep_model import FeatureConstrainedAutoencoder

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

FINAL_4D_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/fc_final_4d/"
LATENT_CSV = os.path.join(FINAL_4D_DIR, "fc_ae_4d_features.csv")
MODEL_PATH = os.path.join(FINAL_4D_DIR, "fc_ae_4d_model.pth")
OUTPUT_EXAMPLES = os.path.join(FINAL_4D_DIR, "fc_4d_umap_examples.png")
OUTPUT_HIGHLIGHT = os.path.join(FINAL_4D_DIR, "fc_4d_umap_highlighted.png")

def plot_fc_4d_examples():
    # 1. Load Data
    print("📥 Loading data...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    df_latent = pd.read_csv(LATENT_CSV, index_col=0)
    feature_cols = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']

    # 2. Select 20 Diverse Examples
    print("🎯 Selecting 20 representative examples...")
    df_latent['u1_bin'] = pd.cut(df_latent['UMAP1'], bins=4)
    df_latent['u2_bin'] = pd.cut(df_latent['UMAP2'], bins=5)
    
    selected_gids = []
    for (b1, b2), group in df_latent.groupby(['u1_bin', 'u2_bin'], observed=True):
        if not group.empty:
            selected_gids.append(group.index[0])
            if len(selected_gids) >= 20:
                break
    
    if len(selected_gids) < 20:
        remaining = df_latent[~df_latent.index.isin(selected_gids)].index.tolist()
        selected_gids.extend(remaining[:20-len(selected_gids)])

    # 3. Load Model
    print("🧠 Loading 4D model for reconstructions...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FeatureConstrainedAutoencoder(latent_dim=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 4. Plot Highlight Map
    plt.figure(figsize=(8, 8))
    plt.scatter(df_latent['UMAP1'], df_latent['UMAP2'], c='lightgrey', alpha=0.3, s=10)
    for i, gid in enumerate(selected_gids):
        row = df_latent.loc[gid]
        plt.scatter(row['UMAP1'], row['UMAP2'], label=f"#{i+1}", s=50)
        plt.text(row['UMAP1'], row['UMAP2'], str(i+1), fontsize=12, fontweight='bold')
    plt.title("20 Selected Examples on 4D Constrained UMAP")
    plt.savefig(OUTPUT_HIGHLIGHT, dpi=150)
    print(f"💾 Highlight map saved to {OUTPUT_HIGHLIGHT}")

    # 5. Plot Trajectories
    print("📈 Plotting 4D reconstructions...")
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, gid in enumerate(selected_gids):
            ax = axes[i]
            idx = gids.index(gid)
            
            # Original
            traj_tensor = torch.tensor(X_traj[idx:idx+1], dtype=torch.float32).to(device)
            orig_raw = s_traj.inverse_transform(X_traj[idx])
            
            # Reconstructed from 4D
            recon_traj_tensor, _, pred_f_tensor = model(traj_tensor)
            recon_raw = s_traj.inverse_transform(recon_traj_tensor.cpu().numpy()[0])
            
            ax.plot(orig_raw[:, 0], color='blue', alpha=0.3, label='Pol1 Orig')
            ax.plot(orig_raw[:, 1], color='red', alpha=0.3, label='Pol2 Orig')
            ax.plot(recon_raw[:, 0], color='blue', linestyle='--', label='Pol1 4D-Recon')
            ax.plot(recon_raw[:, 1], color='red', linestyle='--', label='Pol2 4D-Recon')
            
            # Show original features for reference
            # f_row = df_features_raw.loc[gid] # Not easily available here, just show Latent values or skip
            
            ax.set_title(f"#{i+1} | {labels[idx]}\n{gid}", fontsize=10)
            
            # Flexible Y axis
            y_min, y_max = orig_raw.min(), orig_raw.max()
            padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
            
            if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_EXAMPLES, dpi=150, bbox_inches='tight')
    print(f"💾 Examples gallery saved to {OUTPUT_EXAMPLES}")

if __name__ == "__main__":
    plot_fc_4d_examples()
