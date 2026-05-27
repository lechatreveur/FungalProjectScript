import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from Video_AE_model import VideoAutoencoder

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT = os.path.join(BASE_DIR, "video_ae_final.pth")
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache.npy")
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")
OUTPUT_UMAP = os.path.join(BASE_DIR, "video_ae_umap.png")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    model = VideoAutoencoder(latent_dim=16).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    # 2. Load data
    videos = np.load(CACHE_VIDEOS) # (N, 101, 1, 48, 96)
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]
    
    # 3. Extract Latent Space
    latents = []
    print(f"Extracting latents for {len(videos)} cells...")
    with torch.no_grad():
        for i in range(len(videos)):
            x = torch.from_numpy(videos[i]).float().unsqueeze(0).to(device) # (1, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4) # (1, C, T, H, W)
            z = model.encode(x) # (1, 16)
            latents.append(z.cpu().numpy()[0])
    
    latents = np.array(latents)
    np.save(os.path.join(BASE_DIR, "video_latents.npy"), latents)
    print(f"Saved latents: {latents.shape}")

    # 4. UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    # 5. Visualize
    # Parse experiment labels for coloring
    exp_labels = []
    for gid in gids:
        if gid.startswith('June25'): exp_labels.append('June25')
        elif gid.startswith('Sept17'): exp_labels.append('Sept17')
        elif gid.startswith('M92'): exp_labels.append('M92')
        elif gid.startswith('M93'): exp_labels.append('M93')
        else: exp_labels.append('Other')
    
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Experiment': exp_labels
    })

    plt.figure(figsize=(10, 8))
    for exp in df['Experiment'].unique():
        sub = df[df['Experiment'] == exp]
        plt.scatter(sub['UMAP1'], sub['UMAP2'], label=exp, alpha=0.7, s=30)
    
    plt.title('UMAP Projection of 16D Cellular Dynamics (Video AE)', fontsize=16)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(OUTPUT_UMAP, dpi=150)
    print(f"Saved UMAP to {OUTPUT_UMAP}")

if __name__ == "__main__":
    main()
