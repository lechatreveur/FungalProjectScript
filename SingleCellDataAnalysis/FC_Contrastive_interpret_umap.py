#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import torch
import numpy as np
import pandas as pd
import umap
from scipy.stats import spearmanr

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_Contrastive_train import FCTrajectoryEncoder, EXPERIMENTS, LATENT_DIM

# --- Paths ---
MODEL_PATH = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_contrastive_final.pth"
DEVICE = torch.device("cpu") 

def main():
    # 1. Load Data
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    
    # 2. Extract Latents
    model = FCTrajectoryEncoder(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    t_tensor = torch.from_numpy(X_traj).float().to(DEVICE)
    f_tensor = torch.from_numpy(X_feat).float().to(DEVICE)
    with torch.no_grad():
        latents = model(t_tensor, f_tensor).cpu().numpy()
    
    # 3. Run UMAP (3D)
    reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(latents)
    
    # 4. Correlation Analysis
    feature_names = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
    
    results = []
    for dim in range(3):
        u_dim = embedding[:, dim]
        dim_corrs = {}
        for i, feat_name in enumerate(feature_names):
            corr, _ = spearmanr(u_dim, X_feat[:, i])
            dim_corrs[feat_name] = corr
        results.append(dim_corrs)
    
    # 5. Print Interpretation
    print("\n=== UMAP Axis Interpretation (Spearman Correlation) ===")
    for dim in range(3):
        print(f"\nDimension {dim+1}:")
        # Sort by absolute correlation
        sorted_feats = sorted(results[dim].items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, corr in sorted_feats[:5]:
            print(f"  {feat:15s}: {corr:6.3f}")

if __name__ == "__main__":
    main()
