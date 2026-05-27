#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_train_segmenter.py  — Strategy B

Key change from previous run:
  - Replaced flat BCEWithLogitsLoss with a WEIGHTED FOCAL LOSS.
  - Polarity channels (Pol1, Pol2) are up-weighted 50x.
  - Septum channels (Sep1, Sep2) are up-weighted 30x.
  - Focal loss exponent gamma=2 suppresses easy (background) gradients.

This forces the encoder to learn the spatial location and temporal dynamics
of polarity sites, rather than ignoring them in favour of the bulk cytoplasm.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_model_segmenter import VideoAutoencoderSegmenter

# ==============================================================================
OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS = os.path.join(OUTPUT_DIR, "video_cache_32x112_padded.npy")
CACHE_GAMMA  = os.path.join(OUTPUT_DIR, "gamma_cache_32x112_padded.npy")
CACHE_GIDS   = os.path.join(OUTPUT_DIR, "video_gids.txt")
LOG_FILE     = os.path.join(OUTPUT_DIR, "train_segmenter_strategyB.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM    = 16
EPOCHS        = 200
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
LR_DECAY_STEP = 20
SAVE_EVERY    = 20

# ==============================================================================
# Weighted Focal Loss for class-imbalanced spatial segmentation
# ==============================================================================
# Per-channel weights: [Cyto, Nuc, Pol1, Pol2, Sep1, Sep2, Bg]
# Polarity sites are ~0.7% of pixels → need 50x up-weighting to compete.
GAMMA_CHANNEL_WEIGHTS = torch.tensor([1.0, 2.0, 50.0, 50.0, 30.0, 30.0, 1.0],
                                      dtype=torch.float32)
FOCAL_GAMMA = 2.0   # Focus on hard examples (standard value)
ALPHA_GAMMA = 5.0   # Overall gamma loss scale factor

def focal_loss_with_channel_weights(logits, targets, channel_weights, focal_gamma=2.0):
    """
    Computes a per-channel weighted focal loss.

    Args:
        logits:          (B, 7, T, H, W) raw predictions (before sigmoid)
        targets:         (B, 7, T, H, W) ground truth probabilities in [0,1]
        channel_weights: (7,) tensor of per-channel loss multipliers
        focal_gamma:     focusing parameter (2.0 is standard)

    Returns:
        Scalar loss (mean over all elements, with channel weighting applied).
    """
    # Standard BCE per element — shape (B, 7, T, H, W)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # Focal factor: (1 - p_t)^gamma
    p_t = torch.sigmoid(logits)
    # For positive pixels: p_t = p; for negative: p_t = 1-p
    p_t_correct = p_t * targets + (1 - p_t) * (1 - targets)
    focal_factor = (1.0 - p_t_correct).pow(focal_gamma)

    focal_bce = focal_factor * bce  # (B, 7, T, H, W)

    # Apply per-channel weights — broadcast over (B, T, H, W)
    w = channel_weights.to(logits.device)                   # (7,)
    w = w.view(1, 7, 1, 1, 1)                               # broadcast shape
    weighted = focal_bce * w                                 # (B, 7, T, H, W)

    return weighted.mean()


# ==============================================================================
class MemmapDataset(Dataset):
    def __init__(self, videos_mmap, gammas_mmap):
        self.videos = videos_mmap
        self.gammas = gammas_mmap
        self.n = len(videos_mmap)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        v = torch.from_numpy(self.videos[idx].copy()).float()
        g = torch.from_numpy(self.gammas[idx].copy()).float()
        # (101, 1, H, W) → (1, 101, H, W) for 3D conv
        v = v.permute(1, 0, 2, 3)
        # (101, 7, H, W) → (7, 101, H, W)
        g = g.permute(1, 0, 2, 3)
        return v, g


# ==============================================================================
def train():
    with open(LOG_FILE, 'w') as log_f:
        def log_print(msg):
            print(msg)
            log_f.write(msg + '\n')
            log_f.flush()

        if not os.path.exists(CACHE_GAMMA):
            log_print(f"❌ Gamma cache not found: {CACHE_GAMMA}")
            return

        log_print("📥 Loading caches (Memory-Mapped)...")
        videos = np.load(CACHE_VIDEOS, mmap_mode='r')
        gammas = np.load(CACHE_GAMMA,  mmap_mode='r')

        with open(CACHE_GIDS) as f:
            video_gids = [l.strip() for l in f]

        log_print(f"   Loaded {len(video_gids)} cells.")
        log_print(f"   Videos: {videos.shape}")
        log_print(f"   Gammas: {gammas.shape}")

        dataset    = MemmapDataset(videos, gammas)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2, pin_memory=False)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        log_print(f"🖥️  Device: {device}")
        model = VideoAutoencoderSegmenter(latent_dim=LATENT_DIM).to(device)
        log_print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        log_print(f"   Loss: Focal (γ={FOCAL_GAMMA}) + per-channel weights {GAMMA_CHANNEL_WEIGHTS.tolist()}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=0.5)
        criterion_recon = nn.MSELoss()

        log_print(f"\n🚀 Strategy B Training (epochs={EPOCHS}, batch={BATCH_SIZE})")
        history = {'total': [], 'recon': [], 'gamma': [], 'pol': []}
        t_start = time.time()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            e_total, e_recon, e_gamma, e_pol = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for v_batch, g_batch in dataloader:
                v_batch = v_batch.to(device)
                g_batch = g_batch.to(device)

                optimizer.zero_grad()
                out_hat, _ = model(v_batch)

                v_hat = out_hat[:, 0:1, :, :, :]   # reconstruction
                g_hat = out_hat[:, 1:8, :, :, :]   # 7 gamma channels

                loss_recon = criterion_recon(v_hat, v_batch)
                loss_gamma = focal_loss_with_channel_weights(
                    g_hat, g_batch, GAMMA_CHANNEL_WEIGHTS, FOCAL_GAMMA
                )

                # Separate polarity-only loss for monitoring (no gradient)
                with torch.no_grad():
                    pol_hat = g_hat[:, 2:4, :, :, :]   # Pol1, Pol2 only
                    pol_gt  = g_batch[:, 2:4, :, :, :]
                    e_pol_batch = F.binary_cross_entropy_with_logits(pol_hat, pol_gt).item()

                loss = loss_recon + ALPHA_GAMMA * loss_gamma
                loss.backward()
                optimizer.step()

                e_total += loss.item()
                e_recon += loss_recon.item()
                e_gamma += loss_gamma.item()
                e_pol   += e_pol_batch
                n_batches += 1

            scheduler.step()
            history['total'].append(e_total / n_batches)
            history['recon'].append(e_recon / n_batches)
            history['gamma'].append(e_gamma / n_batches)
            history['pol'].append(e_pol   / n_batches)

            elapsed = time.time() - t_start
            eta = elapsed / epoch * (EPOCHS - epoch)
            log_print(
                f"Epoch [{epoch:3d}/{EPOCHS}] "
                f"Total: {history['total'][-1]:.4f} "
                f"(Recon: {history['recon'][-1]:.4f}, "
                f"Gamma: {history['gamma'][-1]:.4f}, "
                f"Pol: {history['pol'][-1]:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.2e}  ETA: {eta/60:.1f}m"
            )

            if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
                ckpt = os.path.join(OUTPUT_DIR, f"video_segmenter_stratB_epoch_{epoch:03d}.pth")
                torch.save(model.state_dict(), ckpt)
                log_print(f"  💾 Checkpoint: {ckpt}")

        # Save final model
        final = os.path.join(OUTPUT_DIR, "video_segmenter_stratB_final.pth")
        torch.save(model.state_dict(), final)
        log_print(f"\n✅ Done. Model: {final}")

        # Loss Curves
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(history['total'], color='black', label='Total')
        axes[0].plot(history['recon'], color='steelblue', label='Recon')
        axes[0].set_title('Total & Recon Loss'); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(history['gamma'], color='purple', label='Focal Gamma Loss')
        axes[1].set_title('Focal Gamma Loss (weighted)'); axes[1].legend(); axes[1].grid(True)

        axes[2].plot(history['pol'], color='crimson', label='Pol1+Pol2 BCE (monitor)')
        axes[2].set_title('Polarity BCE (unweighted, monitoring)'); axes[2].legend(); axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "segmenter_stratB_loss.png"), dpi=150)
        log_print("📊 Loss curves saved.")


if __name__ == "__main__":
    train()
