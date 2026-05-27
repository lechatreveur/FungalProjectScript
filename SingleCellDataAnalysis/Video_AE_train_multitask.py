#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_train_multitask.py

Trains the Video Autoencoder with supervised regression heads for:
1. Video reconstruction (x_hat)
2. Polarity trajectories (traj_hat, shape: 101x2)
3. Engineered features (feat_hat, shape: 11)
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_model import VideoAutoencoder
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==============================================================================
OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS = os.path.join(OUTPUT_DIR, "video_cache.npy")
CACHE_GIDS   = os.path.join(OUTPUT_DIR, "video_gids.txt")
LOG_FILE     = os.path.join(OUTPUT_DIR, "train_multitask.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = {
    'Sept17': '/Volumes/X10 Pro/Movies/2025_09_17/',
    'M92':    '/Volumes/X10 Pro/Movies/2025_12_31_M92/',
    'M93':    '/Volumes/X10 Pro/Movies/2026_01_08_M93/',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/'
}

LATENT_DIM    = 16
EPOCHS        = 200
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
LR_DECAY_STEP = 20
SAVE_EVERY    = 20

ALPHA_TRAJ = 1.0
BETA_FEAT  = 1.0
# ==============================================================================

def train():
    with open(LOG_FILE, 'w') as log_f:
        def log_print(msg):
            print(msg)
            log_f.write(msg + '\n')
            log_f.flush()
            
        if not os.path.exists(CACHE_VIDEOS):
            log_print(f"❌ Cache not found: {CACHE_VIDEOS}")
            return

        # 1. Load video cache
        log_print("📥 Loading video cache from SSD...")
        videos = np.load(CACHE_VIDEOS)
        with open(CACHE_GIDS) as f:
            video_gids = [l.strip() for l in f]
        log_print(f"   Loaded {len(video_gids)} video cells.")

        # 2. Load trajectories and features
        log_print("📥 Loading trajectories and 11 features...")
        Xt, Xf, fc_gids, labels, s_t, s_f = load_feature_constrained_data(EXPERIMENTS)
        
        # Build lookup dictionaries
        traj_dict = {gid: traj for gid, traj in zip(fc_gids, Xt)}
        feat_dict = {gid: feat for gid, feat in zip(fc_gids, Xf)}

        # 3. Align datasets
        aligned_videos = []
        aligned_traj = []
        aligned_feat = []
        final_gids = []

        for i, gid in enumerate(video_gids):
            if gid in traj_dict and gid in feat_dict:
                aligned_videos.append(videos[i])
                aligned_traj.append(traj_dict[gid])
                aligned_feat.append(feat_dict[gid])
                final_gids.append(gid)

        log_print(f"✅ Aligned {len(final_gids)} cells with all modalities.")
        
        V_np = np.array(aligned_videos)
        T_np = np.array(aligned_traj)
        F_np = np.array(aligned_feat)

        V = torch.from_numpy(V_np).float()    # (N, 101, 1, 48, 96) -> (N, T, C, H, W)
        V = V.permute(0, 2, 1, 3, 4)          # (N, C, T, H, W)
        T = torch.from_numpy(T_np).float()    # (N, 101, 2)
        F = torch.from_numpy(F_np).float()    # (N, 11)
        
        dataset = TensorDataset(V, T, F)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # 4. Model setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        log_print(f"🖥️  Device: {device}")
        model = VideoAutoencoder(latent_dim=LATENT_DIM).to(device)
        log_print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=0.5)
        criterion = nn.MSELoss()

        # 5. Train
        log_print(f"\n🚀 Multi-Task Training (epochs={EPOCHS}, batch={BATCH_SIZE})")
        history = {'total': [], 'recon': [], 'traj': [], 'feat': []}
        t_start = time.time()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            e_total, e_recon, e_traj, e_feat = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for v_batch, t_batch, f_batch in dataloader:
                v_batch = v_batch.to(device)
                t_batch = t_batch.to(device)
                f_batch = f_batch.to(device)
                
                optimizer.zero_grad()
                v_hat, t_hat, f_hat, _ = model(v_batch)
                
                loss_recon = criterion(v_hat, v_batch)
                loss_traj  = criterion(t_hat, t_batch)
                loss_feat  = criterion(f_hat, f_batch)
                
                loss = loss_recon + ALPHA_TRAJ * loss_traj + BETA_FEAT * loss_feat
                loss.backward()
                optimizer.step()
                
                e_total += loss.item()
                e_recon += loss_recon.item()
                e_traj  += loss_traj.item()
                e_feat  += loss_feat.item()
                n_batches += 1

            scheduler.step()
            history['total'].append(e_total / n_batches)
            history['recon'].append(e_recon / n_batches)
            history['traj'].append(e_traj / n_batches)
            history['feat'].append(e_feat / n_batches)

            elapsed = time.time() - t_start
            eta = elapsed / epoch * (EPOCHS - epoch)
            log_print(f"Epoch [{epoch:3d}/{EPOCHS}] "
                      f"Total: {history['total'][-1]:.4f} "
                      f"(Recon: {history['recon'][-1]:.4f}, Traj: {history['traj'][-1]:.4f}, Feat: {history['feat'][-1]:.4f}) "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}  ETA: {eta/60:.1f}m")

            if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
                ckpt = os.path.join(OUTPUT_DIR, f"video_multitask_epoch_{epoch:03d}.pth")
                torch.save(model.state_dict(), ckpt)
                log_print(f"  💾 Checkpoint: {ckpt}")

        # 6. Save final model
        final = os.path.join(OUTPUT_DIR, "video_multitask_final.pth")
        torch.save(model.state_dict(), final)
        log_print(f"\n✅ Done. Model: {final}")

        # 7. Loss Curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['total'], label='Total Loss', color='black')
        plt.plot(history['recon'], label='Recon Loss', color='blue')
        plt.title('Video Reconstruction Loss')
        plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['traj'], label='Traj Loss', color='red')
        plt.plot(history['feat'], label='Feat Loss', color='green')
        plt.title('Supervised Feature/Trajectory Loss')
        plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "multitask_training_loss.png"), dpi=150)
        log_print("📊 Loss curves saved.")

if __name__ == "__main__":
    train()
