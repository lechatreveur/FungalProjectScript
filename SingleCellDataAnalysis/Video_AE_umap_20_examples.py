import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Video_AE_model import VideoAutoencoder
import umap

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT = os.path.join(BASE_DIR, "video_ae_final.pth")
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache.npy")
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")
OUTPUT_GRID = os.path.join(BASE_DIR, "umap_20_examples.png")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load UMAP and pick 20 diverse points
    latents = np.load(os.path.join(BASE_DIR, "video_latents.npy"))
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=20, random_state=42)
    kmeans.fit(embedding)
    
    # Find points closest to centroids
    indices = []
    for center in kmeans.cluster_centers_:
        dist = np.linalg.norm(embedding - center, axis=1)
        indices.append(np.argmin(dist))
    
    # 2. Load model
    model = VideoAutoencoder(latent_dim=16).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    
    videos = np.load(CACHE_VIDEOS)
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]

    # 3. Plot Grid (4 rows x 5 columns)
    fig, axes = plt.subplots(4, 5, figsize=(25, 12))
    axes = axes.flatten()
    plt.suptitle("Phenotypic Gallery: 20 Examples from Across the Latent Space", fontsize=24)

    for i, idx in enumerate(indices):
        # Generate reconstruction
        with torch.no_grad():
            orig = videos[idx]
            x = torch.from_numpy(orig).float().unsqueeze(0).to(device)
            x = x.permute(0, 2, 1, 3, 4)
            recon, _ = model(x)
            recon = recon.permute(0, 2, 1, 3, 4).cpu().numpy()[0]
        
        # Show montage of 5 frames
        t_indices = np.linspace(0, 100, 5, dtype=int)
        montage = np.hstack([recon[t, 0] for t in t_indices])
        
        axes[i].imshow(montage, cmap='viridis')
        axes[i].set_title(f"{gids[idx]}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_GRID, dpi=150)
    print(f"Saved 20 examples to {OUTPUT_GRID}")

if __name__ == "__main__":
    main()
