import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SingleCellQuantificationHPC.tracker_model import load_tracker
from SingleCellQuantificationHPC.tracker_dataset import TrackerDataset
import SingleCellQuantificationHPC.tracker_dataset as td

# Mock discover_corrected_films to ONLY return the held-out film
old_discover = td.discover_corrected_films
def fast_discover(movie_root):
    films = old_discover(movie_root)
    # Filter only the test film
    return [f for f in films if f["film"] == "A14-YES-1t-FBFBF-4_F2"]

td.discover_corrected_films = fast_discover

def visualize_tracking(ckpt_path, movie_root, out_path, num_samples=3):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model from {ckpt_path} on {device}")
    model = load_tracker(ckpt_path, device=device)
    
    print("Loading dataset (fast mode)...")
    ds = TrackerDataset(movie_root=movie_root, hold_out_film=None, augment=False, topk_neg=3)
    held = ds.samples
    print(f"Found {len(held)} samples in held-out film.")
    if len(held) == 0:
        print("No samples found!")
        return
        
    np.random.seed(42)
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
    
    with torch.no_grad():
        for row_idx, i in enumerate(indices):
            sample = ds[i]
            img3ch = sample["img3ch"].unsqueeze(0).to(device)
            candidates = sample["candidates"].unsqueeze(0).to(device)
            sep_t = sample["sep_t"].unsqueeze(0).to(device)
            sep_t1 = sample["sep_t1"].unsqueeze(0).to(device)
            area_ratio = sample["area_ratio"].unsqueeze(0).to(device)
            adjacency = sample["adjacency"].unsqueeze(0).to(device)
            gt_idx = sample["gt_idx"].item()
            
            scores, div_logit, mrg_logit = model(img3ch, candidates, sep_t, area_ratio, sep_t1, adjacency)
            scores = scores.squeeze(0).cpu().numpy()
            
            img_np = img3ch.squeeze(0).cpu().numpy()
            bf_t = img_np[0]
            bf_t1 = img_np[1]
            mask_t = img_np[2]
            
            ax = axes[row_idx, 0]
            ax.imshow(bf_t, cmap="gray")
            ax.set_title("BF at t")
            ax.axis("off")
            
            ax = axes[row_idx, 1]
            ax.imshow(bf_t1, cmap="gray")
            ax.set_title("BF at t+1")
            ax.axis("off")
            
            ax = axes[row_idx, 2]
            ax.imshow(mask_t, cmap="hot")
            ax.set_title("Mask at t")
            ax.axis("off")
            
            cands_np = candidates.squeeze(0).cpu().numpy() # (K, 1, H, W)
            K = cands_np.shape[0]
            for k in range(3):
                ax = axes[row_idx, 3 + k]
                if k < K:
                    mask_cand = cands_np[k, 0]
                    ax.imshow(mask_cand, cmap="Blues")
                    is_gt = " (GT)" if k == gt_idx else ""
                    ax.set_title(f"Cand {k}{is_gt}\nScore: {scores[k]:.2f}")
                else:
                    ax.axis("off")
                ax.axis("off")
                
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    visualize_tracking(
        ckpt_path="./tracker_checkpoints/model_best.pt",
        movie_root="/Volumes/X10 Pro/Movies",
        out_path="tracking_viz.png",
        num_samples=5
    )
