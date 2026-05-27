#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

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
MODEL_PATH = os.path.join(FINAL_4D_DIR, "fc_ae_4d_model.pth")
OUTPUT_TRAVERSAL = os.path.join(FINAL_4D_DIR, "fc_4d_latent_traversal.png")

def run_4d_traversal():
    # 1. Load Model and Scalers
    print("📥 Loading data and model...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FeatureConstrainedAutoencoder(latent_dim=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Get population latent mean
    X_traj_tensor = torch.tensor(X_traj, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, latent_all, _ = model(X_traj_tensor)
        z_mean = latent_all.mean(dim=0).cpu().numpy()
        z_std = latent_all.std(dim=0).cpu().numpy()

    # 2. Setup Traversal
    n_steps = 7
    # Use -2 to +2 standard deviations
    steps = np.linspace(-2, 2, n_steps)
    
    feature_cols = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    with torch.no_grad():
        for latent_idx in range(4):
            ax = axes[latent_idx]
            colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
            
            for i, val in enumerate(steps):
                # Create latent vector: [mean, mean, mean, mean] but vary current dimension
                z_test = z_mean.copy()
                z_test[latent_idx] = z_mean[latent_idx] + val * z_std[latent_idx]
                z_tensor = torch.tensor([z_test], dtype=torch.float32).to(device)
                
                # Decode to trajectory
                # We need to reach the decoder_linear directly or call model.decode
                x_dec = model.decoder_linear(z_tensor)
                x_dec = x_dec.view(x_dec.size(0), 64, 13)
                recon_traj = model.decoder_conv(x_dec).transpose(1, 2).cpu().numpy()[0]
                
                # Inverse transform curve
                recon_raw = s_traj.inverse_transform(recon_traj)
                
                # Predict features for this latent state
                pred_f_scaled = model.feature_predictor(z_tensor).cpu().numpy()[0]
                pred_f_raw = s_feat.inverse_transform([pred_f_scaled])[0]
                
                lbl = f"SD {val:+.1f}"
                ax.plot(recon_raw[:, 0], color=colors[i], label=lbl if latent_idx==0 else "")
                ax.plot(recon_raw[:, 1], color=colors[i], linestyle='--')

            ax.set_title(f"Traversal of Feature-Constrained Latent {latent_idx+1}")
            ax.set_ylabel("Intensity")
            # Set y-axis to min and max of the data in this traversal
            ax.autoscale(enable=True, axis='y', tight=True)
            if latent_idx == 0: ax.legend(title="Latent Value", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_TRAVERSAL, dpi=150, bbox_inches='tight')
    print(f"💾 Traversal plot saved to {OUTPUT_TRAVERSAL}")

if __name__ == "__main__":
    run_4d_traversal()
