import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache.npy")
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")
ARTIFACT_DIR = "/Users/user/.gemini/antigravity/brain/8e3e2fd2-945e-4dcf-a62c-5ef369b9b1d7/"

def debug_strip():
    videos = np.load(CACHE_VIDEOS, mmap_mode='r')
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]
    
    # Pick a cell that was likely cropped before (a long one)
    # Let's just pick one from the middle
    idx = 200
    v = videos[idx, :, 0, :, :] # (101, 64, 224)
    
    # Take every 5th frame for a shorter strip
    v_sampled = v[::5]
    strip = np.vstack(v_sampled)
    
    strip_norm = np.clip(strip / 1.5, 0, 1) * 255
    strip_uint8 = strip_norm.astype(np.uint8)
    
    cm = plt.get_cmap('viridis')
    strip_rgb = (cm(strip_uint8 / 255.0)[:, :, :3] * 255).astype(np.uint8)
    
    img = Image.fromarray(strip_rgb)
    img_path = os.path.join(ARTIFACT_DIR, "debug_strip_64x224.png")
    img.save(img_path)
    print(f"Saved debug strip to {img_path}")
    print(f"Strip shape: {strip_rgb.shape}")

if __name__ == "__main__":
    debug_strip()
