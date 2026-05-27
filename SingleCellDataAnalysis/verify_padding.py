import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

CACHE_VIDEOS = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/video_cache_32x112_padded.npy"
ARTIFACT_DIR = "/Users/user/.gemini/antigravity/brain/8e3e2fd2-945e-4dcf-a62c-5ef369b9b1d7/"

def check_padding():
    videos = np.load(CACHE_VIDEOS, mmap_mode='r')
    n_cells = videos.shape[0]
    
    # Pick a few indices
    indices = [0, n_cells//2, n_cells-1]
    
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, len(indices)*2))
    for i, idx in enumerate(indices):
        # Take middle frame
        img = videos[idx, 50, 0]
        axes[i].imshow(img, cmap='viridis')
        axes[i].set_title(f"Cell {idx} (Padded 32x112)")
        axes[i].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, "verify_padding_32x112.png")
    plt.savefig(out_path)
    print(f"Saved verification to {out_path}")

if __name__ == "__main__":
    check_padding()
