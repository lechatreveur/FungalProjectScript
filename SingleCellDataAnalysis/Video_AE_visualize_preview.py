import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Video_AE_model import VideoAutoencoder

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT = os.path.join(BASE_DIR, "video_ae_final.pth")
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache.npy")
OUTPUT_IMG = os.path.join(BASE_DIR, "reconstruction_preview_final.png")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    model = VideoAutoencoder(latent_dim=16).to(device)
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        return
    
    # Load state dict directly (as saved in training script)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")

    # 2. Load cache
    videos = np.load(CACHE_VIDEOS) # (N, 101, 1, 48, 96)
    print(f"Cache loaded: {videos.shape}")

    # 3. Pick 4 random cells
    indices = np.random.choice(len(videos), 4, replace=False)
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    plt.suptitle(f"Video AE Reconstruction Preview (Epoch 100)\nCells: {indices.tolist()}", fontsize=16)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Original
            orig = videos[idx] # (101, 1, 48, 96)
            
            # Reconstruction
            x = torch.from_numpy(orig).float().unsqueeze(0).to(device) # (1, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4) # (1, C, T, H, W)
            recon, _ = model(x)
            recon = recon.permute(0, 2, 1, 3, 4).cpu().numpy()[0] # (T, C, H, W)

            # Show a representative frame (e.g. middle of the movie)
            frame_idx = 50
            
            axes[i, 0].imshow(orig[frame_idx, 0], cmap='viridis')
            axes[i, 0].set_title(f"Cell {idx} - Ground Truth (t={frame_idx})")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(recon[frame_idx, 0], cmap='viridis')
            axes[i, 1].set_title(f"Cell {idx} - Reconstruction (t={frame_idx})")
            axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"Saved preview to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
