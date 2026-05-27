#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_physics_eval.py

Evaluates the Physics-Informed Autoencoder.
Generates:
1. Feature & Trajectory accuracy plots.
2. Visual proof that the predicted spatial masks successfully learned the location of the poles in a completely unsupervised way!
"""

import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_model_physics import VideoAutoencoderPhysics
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==============================================================================
OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT   = os.path.join(OUTPUT_DIR, "video_physics_final.pth")
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

    model = VideoAutoencoderPhysics(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    videos = np.load(CACHE_VIDEOS)
    with open(CACHE_GIDS) as f:
        video_gids = [l.strip() for l in f]

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

    T_pred = []
    F_pred = []
    
    print("⏳ Running predictions on all cells...")
    # Pick a random cell for mask visualization
    np.random.seed(42)
    vis_idx = np.random.randint(0, len(V_np))
    vis_x_hat, vis_m1, vis_m2 = None, None, None

    with torch.no_grad():
        for i in range(len(V_np)):
            v = torch.from_numpy(V_np[i]).float().unsqueeze(0).to(device)
            v = v.permute(0, 2, 1, 3, 4)
            x_hat, t_hat, f_hat, _, m1, m2 = model(v)
            T_pred.append(t_hat.cpu().numpy()[0])
            F_pred.append(f_hat.cpu().numpy()[0])
            
            if i == vis_idx:
                vis_x_hat = x_hat.cpu().numpy()[0, 0] # (T, H, W)
                vis_m1 = m1.cpu().numpy()[0, 0]
                vis_m2 = m2.cpu().numpy()[0, 0]
            
    T_pred = np.array(T_pred)
    F_pred = np.array(F_pred)

    # 1. Feature Scatter Plots
    print("📊 Generating Feature Scatter Plots...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Physics Autoencoder: True vs Predicted Features', fontsize=18)

    for i in range(11):
        ax = axes[i]
        t = F_true[:, i]
        p = F_pred[:, i]
        r, _ = pearsonr(t, p)
        ax.scatter(t, p, alpha=0.5, s=15, color='purple')
        min_val = min(t.min(), p.min())
        max_val = max(t.max(), p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        ax.set_title(f"{FEATURE_NAMES[i]} ($R^2$ = {r**2:.2f})")
        ax.grid(alpha=0.3)

    axes[11].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "physics_eval_features.png"), dpi=150)

    # 2. Trajectory Plots
    print("📈 Generating Trajectory Plots...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Physics Autoencoder: Trajectory Predictions', fontsize=18)
    
    sample_indices = np.random.choice(len(V_np), 3, replace=False)
    for row, idx in enumerate(sample_indices):
        for col in range(2):
            ax = axes[row, col]
            t = T_true[idx, :, col]
            p = T_pred[idx, :, col]
            ax.plot(t, label='True', color='black', lw=2)
            ax.plot(p, label='Predicted (from image masks)', color='magenta', linestyle='--')
            pol_name = 'Pol1' if col == 0 else 'Pol2'
            ax.set_title(f"Cell Index {idx} - {pol_name}")
            if row == 0 and col == 0: ax.legend()
            ax.grid(alpha=0.3)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "physics_eval_trajectories.png"), dpi=150)

    # 3. Mask Visualizations
    print("🎨 Generating Spatial Mask Visualization...")
    timepoints = [0, 25, 50, 75, 100]
    fig, axes = plt.subplots(3, 5, figsize=(18, 9))
    fig.suptitle(f'Unsupervised Learned Spatial Masks (Cell {vis_idx})', fontsize=20)
    
    for c, t in enumerate(timepoints):
        # Original Reconstructed Image
        axes[0, c].imshow(vis_x_hat[t], cmap='gray')
        axes[0, c].set_title(f"t={t} Reconstructed")
        axes[0, c].axis('off')
        
        # Mask 1 Overlay
        axes[1, c].imshow(vis_x_hat[t], cmap='gray')
        axes[1, c].imshow(vis_m1[t], cmap='Reds', alpha=0.5)
        axes[1, c].set_title("Pol1 Learned Mask")
        axes[1, c].axis('off')
        
        # Mask 2 Overlay
        axes[2, c].imshow(vis_x_hat[t], cmap='gray')
        axes[2, c].imshow(vis_m2[t], cmap='Blues', alpha=0.5)
        axes[2, c].set_title("Pol2 Learned Mask")
        axes[2, c].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "physics_learned_masks.png"), dpi=150)
    print("✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate()
