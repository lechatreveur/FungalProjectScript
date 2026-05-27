#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import sys

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.AE_model import TrajectoryAutoencoder
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

SUFF_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/sufficiency_test/"
MODEL_PATH = os.path.join(SUFF_DIR, "fc_ae_model.pth")
OUTPUT_UMAP = os.path.join(SUFF_DIR, "fc_ae_umap.png")

def plot_fc_manifold():
    # 1. Load Data
    print("📥 Loading data...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    X_traj_tensor = torch.tensor(X_traj, dtype=torch.float32) # (N, 101, 2)
    
    # 2. Load Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TrajectoryAutoencoder(latent_dim=11).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Extract Latent Features
    print("🧠 Extracting 11-dimensional latent features...")
    with torch.no_grad():
        # Process in one batch as it's only 431 cells
        _, latent_z = model(X_traj_tensor.to(device))
        Z_np = latent_z.cpu().numpy()
        
    # 4. Generate UMAP
    print("🌌 Generating UMAP embedding...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(Z_np)
    
    # 5. Plotting
    df_plot = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df_plot['experiment'] = labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_plot, x='UMAP1', y='UMAP2', hue='experiment', palette='Set2', alpha=0.7, s=40)
    plt.title('UMAP Manifold of 11-Dimensional Feature-Constrained Latent Space', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_UMAP, dpi=150, bbox_inches='tight')
    print(f"💾 Manifold plot saved to {OUTPUT_UMAP}")
    
    # Optional: Save latent features to CSV for reference
    latent_cols = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
    df_latent = pd.DataFrame(Z_np, columns=latent_cols, index=gids)
    df_latent['experiment'] = labels
    df_latent.to_csv(os.path.join(SUFF_DIR, "fc_ae_latent_features.csv"))
    print("💾 Latent features saved to CSV.")

if __name__ == "__main__":
    plot_fc_manifold()
