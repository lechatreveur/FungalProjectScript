#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_multitask_eval.py

Evaluates the trained Multi-Task Video Autoencoder by comparing
predicted trajectories and features against ground truth.
"""

import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_model import VideoAutoencoder
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==============================================================================
OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT   = os.path.join(OUTPUT_DIR, "video_multitask_final.pth")
CACHE_VIDEOS = os.path.join(OUTPUT_DIR, "video_cache.npy")
CACHE_GIDS   = os.path.join(OUTPUT_DIR, "video_gids.txt")

EXPERIMENTS = {
    'Sept17': '/Volumes/X10 Pro/Movies/2025_09_17/',
    'M92':    '/Volumes/X10 Pro/Movies/2025_12_31_M92/',
    'M93':    '/Volumes/X10 Pro/Movies/2026_01_08_M93/',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/'
}
LATENT_DIM = 16

FEATURE_NAMES = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
# ==============================================================================

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    if not os.path.exists(CHECKPOINT):
        print(f"❌ Checkpoint not found: {CHECKPOINT}")
        return

    model = VideoAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    # Load original cache
    videos = np.load(CACHE_VIDEOS)
    with open(CACHE_GIDS) as f:
        video_gids = [l.strip() for l in f]

    # Load targets
    Xt, Xf, fc_gids, labels, s_t, s_f = load_feature_constrained_data(EXPERIMENTS)
    traj_dict = {gid: traj for gid, traj in zip(fc_gids, Xt)}
    feat_dict = {gid: feat for gid, feat in zip(fc_gids, Xf)}

    aligned_videos, aligned_traj, aligned_feat = [], [], []
    for i, gid in enumerate(video_gids):
        if gid in traj_dict and gid in feat_dict:
            aligned_videos.append(videos[i])
            aligned_traj.append(traj_dict[gid])
            aligned_feat.append(feat_dict[gid])

    V_np = np.array(aligned_videos)
    T_true = np.array(aligned_traj)
    F_true = np.array(aligned_feat)

    # Predict
    T_pred = []
    F_pred = []
    
    print("⏳ Running predictions on all cells...")
    with torch.no_grad():
        for i in range(len(V_np)):
            v = torch.from_numpy(V_np[i]).float().unsqueeze(0).to(device)
            v = v.permute(0, 2, 1, 3, 4)
            _, t_hat, f_hat, _ = model(v)
            T_pred.append(t_hat.cpu().numpy()[0])
            F_pred.append(f_hat.cpu().numpy()[0])
            
    T_pred = np.array(T_pred)
    F_pred = np.array(F_pred)

    # 1. Feature Scatter Plots
    print("📊 Generating Feature Scatter Plots...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Multi-Task Autoencoder: True vs Predicted Features (Scaled)', fontsize=18)

    for i in range(11):
        ax = axes[i]
        t = F_true[:, i]
        p = F_pred[:, i]
        
        # Calculate R^2
        r, _ = pearsonr(t, p)
        
        ax.scatter(t, p, alpha=0.5, s=15)
        
        # Plot y=x line
        min_val = min(t.min(), p.min())
        max_val = max(t.max(), p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f"{FEATURE_NAMES[i]} ($R^2$ = {r**2:.2f})")
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.grid(alpha=0.3)

    axes[11].axis('off') # Hide 12th plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "multitask_eval_features.png"), dpi=150)

    # 2. Trajectory Plots
    print("📈 Generating Trajectory Plots...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Multi-Task Autoencoder: Trajectory Predictions', fontsize=18)
    
    # Pick 3 random cells
    np.random.seed(42)
    sample_indices = np.random.choice(len(V_np), 3, replace=False)
    
    for row, idx in enumerate(sample_indices):
        for col in range(2): # pol1, pol2
            ax = axes[row, col]
            t = T_true[idx, :, col]
            p = T_pred[idx, :, col]
            
            ax.plot(t, label='True', color='black', lw=2)
            ax.plot(p, label='Predicted', color='red', linestyle='--')
            
            pol_name = 'Pol1' if col == 0 else 'Pol2'
            ax.set_title(f"Cell Index {idx} - {pol_name}")
            if row == 0 and col == 0: ax.legend()
            ax.grid(alpha=0.3)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "multitask_eval_trajectories.png"), dpi=150)
    print("✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate()
