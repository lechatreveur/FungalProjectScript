import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_model_segmenter import VideoAutoencoderSegmenter
import umap
from sklearn.cluster import KMeans

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT = os.path.join(BASE_DIR, "video_segmenter_stratB_final.pth")
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache_32x112_padded.npy")
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")

OUTPUT_LATENTS = os.path.join(BASE_DIR, "segmenter_latents.npy")
OUTPUT_UMAP_LABELED = os.path.join(BASE_DIR, "segmenter_umap_labeled.png")
OUTPUT_GRID_LABELED = os.path.join(BASE_DIR, "segmenter_umap_20_examples.png")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load model and compute latents
    model = VideoAutoencoderSegmenter(latent_dim=16).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    
    videos = np.load(CACHE_VIDEOS)
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]

    print("Computing latents...")
    latents = []
    # Batch process to avoid memory issues
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(videos), batch_size):
            v_batch = videos[i:i+batch_size]
            v_tensor = torch.from_numpy(v_batch).float().to(device)
            v_tensor = v_tensor.permute(0, 2, 1, 3, 4) # (B, 1, 101, 32, 112)
            
            _, z = model(v_tensor)
            latents.append(z.cpu().numpy())
    
    latents = np.vstack(latents)
    np.save(OUTPUT_LATENTS, latents)
    print(f"Saved latents of shape {latents.shape}")

    # 2. UMAP
    print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    kmeans = KMeans(n_clusters=20, random_state=42)
    kmeans.fit(embedding)
    indices = []
    for center in kmeans.cluster_centers_:
        dist = np.linalg.norm(embedding - center, axis=1)
        indices.append(np.argmin(dist))
    
    # 3. Plot UMAP with labels
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c='lightgrey', alpha=0.5, s=20)
    for i, idx in enumerate(indices):
        plt.scatter(embedding[idx, 0], embedding[idx, 1], c='purple', s=80, edgecolors='black')
        plt.annotate(f"{i+1}", (embedding[idx, 0], embedding[idx, 1]), 
                     fontsize=12, fontweight='bold', xytext=(5, 5), textcoords='offset points')
    
    plt.title('Segmenter Latent UMAP with 20 Sampled Examples', fontsize=18)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(OUTPUT_UMAP_LABELED, dpi=150)
    print(f"Saved labeled UMAP to {OUTPUT_UMAP_LABELED}")

    # 4. Generate Grid with labels
    fig, axes = plt.subplots(4, 5, figsize=(25, 12))
    axes = axes.flatten()
    plt.suptitle("Segmenter Phenotypic Gallery (Numbered 1-20)", fontsize=24)

    for i, idx in enumerate(indices):
        with torch.no_grad():
            orig = videos[idx]
            x = torch.from_numpy(orig).float().unsqueeze(0).to(device)
            x = x.permute(0, 2, 1, 3, 4)
            out_hat, _ = model(x)
            v_hat = out_hat[:, 0:1, :, :, :]
            v_hat = v_hat.permute(0, 2, 1, 3, 4).cpu().numpy()[0]
        
        t_indices = np.linspace(0, 100, 5, dtype=int)
        montage = np.hstack([v_hat[t, 0] for t in t_indices])
        
        axes[i].imshow(montage, cmap='viridis')
        axes[i].set_title(f"#{i+1}: {gids[idx]}", fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_GRID_LABELED, dpi=150)
    print(f"Saved labeled grid to {OUTPUT_GRID_LABELED}")

if __name__ == "__main__":
    main()
