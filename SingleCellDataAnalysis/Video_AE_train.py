#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_train.py  --  Cache-based training for the Video Autoencoder.

Expects the cache built by Video_AE_build_cache.py:
  video_cache.npy  — (N, 1, 101, 48, 96) float32
  video_gids.txt   — corresponding cell IDs
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

# ==============================================================================
OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS = os.path.join(OUTPUT_DIR, "video_cache.npy")
CACHE_GIDS   = os.path.join(OUTPUT_DIR, "video_gids.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM    = 16
EPOCHS        = 200
BATCH_SIZE    = 8
LEARNING_RATE = 1e-3
LR_DECAY_STEP = 60    # halve LR every N epochs
SAVE_EVERY    = 50
# ==============================================================================


def train():
    if not os.path.exists(CACHE_VIDEOS):
        print(f"❌ Cache not found: {CACHE_VIDEOS}")
        print("   Run Video_AE_build_cache.py first.")
        return

    # 1. Load cache
    print("📥 Loading video cache from SSD...")
    videos = np.load(CACHE_VIDEOS)          # (N, 101, 1, 48, 96)
    with open(CACHE_GIDS) as f:
        valid_gids = [l.strip() for l in f]
    print(f"   Loaded {videos.shape[0]} cells. Shape: {videos.shape}")

    X = torch.from_numpy(videos).float()    # (N, 101, 1, 48, 96) -> (N, T, C, H, W)
    X = X.permute(0, 2, 1, 3, 4)           # (N, C, T, H, W)
    
    dataset    = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 2. Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    model = VideoAutoencoder(latent_dim=LATENT_DIM).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=0.5)
    criterion = nn.MSELoss()

    # 3. Train
    print(f"\n🚀 Training  (latent_dim={LATENT_DIM}, epochs={EPOCHS}, batch={BATCH_SIZE})")
    losses  = []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)

        elapsed = time.time() - t_start
        eta     = elapsed / epoch * (EPOCHS - epoch)
        print(f"  Epoch [{epoch:3d}/{EPOCHS}]  Loss: {avg:.5f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}  "
              f"Elapsed: {elapsed/60:.1f}m  ETA: {eta/60:.1f}m")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            ckpt = os.path.join(OUTPUT_DIR, f"video_ae_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  💾 Checkpoint: {ckpt}")

    # 4. Final model
    final = os.path.join(OUTPUT_DIR, "video_ae_final.pth")
    torch.save(model.state_dict(), final)
    print(f"\n✅ Done. Model: {final}")

    # 5. Loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, EPOCHS+1), losses, 'b-', lw=1.5)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Video AE Training Loss"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "video_ae_training_loss.png"), dpi=150)
    print("📊 Loss curve saved.")
    return model, losses


if __name__ == "__main__":
    train()
