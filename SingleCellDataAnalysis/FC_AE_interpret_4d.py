#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import umap

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_sweep_model import FeatureConstrainedAutoencoder
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==== 1. Configuration ====
EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

OUTPUT_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/fc_final_4d/"
MODEL_PATH = os.path.join(OUTPUT_DIR, "fc_ae_4d_model.pth")

def interpret_4d():
    # 1. Load Data
    print("📥 Loading data...")
    X_traj, X_feat_scaled, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    X_traj_tensor = torch.tensor(X_traj, dtype=torch.float32)
    
    # We also want the raw (unscaled) 11 features for better interpretation
    # load_feature_constrained_data doesn't return raw features easily, so we re-load them
    from SingleCellDataAnalysis.PCA_utils import load_experiment_features
    df_list = []
    for exp_name, exp_dir in EXPERIMENTS.items():
        df_exp = load_experiment_features(exp_dir)
        df_exp['global_cell_id'] = exp_name + "_" + df_exp.index.astype(str)
        df_exp.set_index('global_cell_id', inplace=True)
        df_list.append(df_exp)
    df_features_raw = pd.concat(df_list)
    feature_cols = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
    df_features_raw = df_features_raw.loc[gids, feature_cols]

    # 2. Load Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FeatureConstrainedAutoencoder(latent_dim=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 3. Extract 4D Latent Features
    print("🧠 Extracting 4D latent features...")
    with torch.no_grad():
        _, latent_z, _ = model(X_traj_tensor.to(device))
        Z_np = latent_z.cpu().numpy()
        
    df_latent = pd.DataFrame(Z_np, columns=['Latent_1', 'Latent_2', 'Latent_3', 'Latent_4'], index=gids)
    df_latent['experiment'] = labels

    # 4. Correlation Analysis
    print("📊 Computing correlations...")
    # Correlate 4 Latents with 11 Raw Features
    corr_matrix = pd.concat([df_latent.iloc[:, :4], df_features_raw], axis=1).corr()
    # Extract only the 4x11 submatrix
    subset_corr = corr_matrix.loc[['Latent_1', 'Latent_2', 'Latent_3', 'Latent_4'], feature_cols]

    plt.figure(figsize=(12, 6))
    sns.heatmap(subset_corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title("Correlation: 4D Latent Dimensions vs 11 Biological Features")
    plt.savefig(os.path.join(OUTPUT_DIR, "fc_4d_correlation_heatmap.png"), dpi=150, bbox_inches='tight')

    # 5. UMAP of 4D space
    print("🌌 Generating UMAP of 4D space...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(Z_np)
    df_latent['UMAP1'] = embedding[:, 0]
    df_latent['UMAP2'] = embedding[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_latent, x='UMAP1', y='UMAP2', hue='experiment', palette='Set2', alpha=0.7)
    plt.title("4D Feature-Constrained UMAP")
    plt.savefig(os.path.join(OUTPUT_DIR, "fc_4d_umap.png"), dpi=150)
    
    # 6. Save data
    df_latent.to_csv(os.path.join(OUTPUT_DIR, "fc_ae_4d_features.csv"))
    print("✅ Analysis complete.")

if __name__ == "__main__":
    interpret_4d()
