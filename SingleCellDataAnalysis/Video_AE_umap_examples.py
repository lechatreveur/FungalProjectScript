import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Video_AE_model import VideoAutoencoder

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT = os.path.join(BASE_DIR, "video_ae_final.pth")
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache.npy")
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")
OUTPUT_EXAMPLES = os.path.join(BASE_DIR, "umap_examples.png")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load UMAP data
    # (We re-run UMAP briefly to get coordinates if we didn't save them, 
    # but I'll just load the latents and re-run to be safe and consistent)
    latents = np.load(os.path.join(BASE_DIR, "video_latents.npy"))
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]
    
    # 2. Pick 4 diverse points
    # (Min/Max of UMAP1 and UMAP2)
    indices = [
        np.argmin(embedding[:, 0]), # Far Left
        np.argmax(embedding[:, 0]), # Far Right
        np.argmin(embedding[:, 1]), # Bottom
        np.argmax(embedding[:, 1]), # Top
    ]
    
    # 3. Load model and generate reconstructions
    model = VideoAutoencoder(latent_dim=16).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    
    videos = np.load(CACHE_VIDEOS)
    
    fig = plt.figure(figsize=(18, 10))
    # Grid: UMAP on left, 4 examples on right
    gs = fig.add_gridspec(2, 3)
    ax_umap = fig.add_subplot(gs[:, 0])
    
    # Plot UMAP
    ax_umap.scatter(embedding[:, 0], embedding[:, 1], c='lightgrey', alpha=0.5, s=10)
    colors = ['red', 'blue', 'green', 'purple']
    labels = ['Point A (Left)', 'Point B (Right)', 'Point C (Bottom)', 'Point D (Top)']
    
    for i, idx in enumerate(indices):
        ax_umap.scatter(embedding[idx, 0], embedding[idx, 1], c=colors[i], s=100, label=labels[i], edgecolors='black')
        
        # Generate reconstruction
        with torch.no_grad():
            orig = videos[idx]
            x = torch.from_numpy(orig).float().unsqueeze(0).to(device)
            x = x.permute(0, 2, 1, 3, 4)
            recon, _ = model(x)
            recon = recon.permute(0, 2, 1, 3, 4).cpu().numpy()[0]
        
        # Plot montage (6 timepoints)
        ax_ex = fig.add_subplot(gs[i//2, 1 + i%2])
        t_indices = np.linspace(0, 100, 6, dtype=int)
        montage = np.hstack([recon[t, 0] for t in t_indices])
        ax_ex.imshow(montage, cmap='viridis')
        ax_ex.set_title(f"{labels[i]}: {gids[idx]}", color=colors[i], fontweight='bold')
        ax_ex.axis('off')

    ax_umap.set_title('UMAP with Selected Points', fontsize=16)
    ax_umap.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_EXAMPLES, dpi=150)
    print(f"Saved examples to {OUTPUT_EXAMPLES}")

if __name__ == "__main__":
    main()
