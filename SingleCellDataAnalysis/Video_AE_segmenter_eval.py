#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_segmenter_eval.py

Evaluates the Video Autoencoder Segmenter.
Produces side-by-side comparisons of True Gammas vs Predicted Gammas.
"""

import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_model_segmenter import VideoAutoencoderSegmenter

OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS = os.path.join(OUTPUT_DIR, "video_cache_32x112_padded.npy")
CACHE_GAMMA  = os.path.join(OUTPUT_DIR, "gamma_cache_32x112_padded.npy")
CACHE_GIDS   = os.path.join(OUTPUT_DIR, "video_gids.txt")

# Use the latest checkpoint or final model
MODEL_PATH   = os.path.join(OUTPUT_DIR, "video_segmenter_stratB_final.pth")

LATENT_DIM   = 16

GAMMA_LABELS = ['Z0: Cyto', 'Z1: Nuc', 'Z2: Pol1', 'Z3: Pol2', 'Z4: Sep1', 'Z5: Sep2', 'Z6: Bg']

def eval_model():
    if not os.path.exists(MODEL_PATH):
        # Fallback to the latest available if 200 is not done yet
        import glob
        ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "video_segmenter_epoch_*.pth")))
        if not ckpts:
            print(f"❌ Model not found: {MODEL_PATH}")
            return
        model_path = ckpts[-1]
    else:
        model_path = MODEL_PATH

    print(f"📥 Loading {model_path}...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VideoAutoencoderSegmenter(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    print("📥 Loading caches...")
    videos = np.load(CACHE_VIDEOS)
    gammas = np.load(CACHE_GAMMA)

    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]

    # Pick a few random cells to visualize
    np.random.seed(42)
    sample_idxs = np.random.choice(len(videos), size=3, replace=False)

    fig, axes = plt.subplots(len(sample_idxs) * 2, 8, figsize=(20, len(sample_idxs)*3))
    
    # Timepoint to visualize
    t_plot = 50

    with torch.no_grad():
        for i, idx in enumerate(sample_idxs):
            v_np = videos[idx:idx+1] # (1, 101, 1, 48, 96)
            g_np = gammas[idx]       # (101, 7, 48, 96)
            
            v_tensor = torch.from_numpy(v_np).float()
            v_tensor = v_tensor.permute(0, 2, 1, 3, 4).to(device) # (1, 1, 101, 32, 112)
            
            out_hat, _ = model(v_tensor)
            out_hat = out_hat.cpu().numpy()[0] # (8, 101, 48, 96)
            
            v_hat = out_hat[0] # (101, 48, 96)
            
            # Apply sigmoid to gamma hat to get probabilities
            g_hat = 1 / (1 + np.exp(-out_hat[1:8])) # (7, 101, 48, 96)
            
            # Ground truth row
            row_true = i * 2
            axes[row_true, 0].imshow(v_np[0, t_plot, 0], cmap='gray')
            axes[row_true, 0].set_title(f"True Vid ({gids[idx]})")
            axes[row_true, 0].axis('off')
            
            for c in range(7):
                axes[row_true, c+1].imshow(g_np[t_plot, c], cmap='inferno', vmin=0, vmax=1)
                axes[row_true, c+1].set_title(f"True {GAMMA_LABELS[c]}")
                axes[row_true, c+1].axis('off')

            # Prediction row
            row_pred = i * 2 + 1
            axes[row_pred, 0].imshow(v_hat[t_plot], cmap='gray')
            axes[row_pred, 0].set_title("Pred Vid")
            axes[row_pred, 0].axis('off')
            
            for c in range(7):
                axes[row_pred, c+1].imshow(g_hat[c, t_plot], cmap='inferno', vmin=0, vmax=1)
                axes[row_pred, c+1].set_title(f"Pred {GAMMA_LABELS[c]}")
                axes[row_pred, c+1].axis('off')
                
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "segmenter_learned_masks_32x112.png")
    plt.savefig(out_png, dpi=150)
    print(f"✅ Saved visualization to {out_png}")

if __name__ == "__main__":
    eval_model()
