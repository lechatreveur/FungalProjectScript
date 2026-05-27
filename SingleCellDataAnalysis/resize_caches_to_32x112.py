import numpy as np
import os
from skimage.transform import resize
import torch

BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS_64 = os.path.join(BASE_DIR, "video_cache.npy")
CACHE_GAMMA_64  = os.path.join(BASE_DIR, "gamma_cache.npy")

CACHE_VIDEOS_32 = os.path.join(BASE_DIR, "video_cache_32x112.npy")
CACHE_GAMMA_32  = os.path.join(BASE_DIR, "gamma_cache_32x112.npy")

def resize_caches():
    print("Resizing Video Cache...")
    # Load 64x224
    videos = np.load(CACHE_VIDEOS_64, mmap_mode='r')
    n, t, c, h, w = videos.shape
    
    # Initialize 32x112
    v32 = np.lib.format.open_memmap(CACHE_VIDEOS_32, mode='w+', dtype='float32', shape=(n, t, c, 32, 112))
    
    for i in range(n):
        for j in range(t):
            v32[i, j, 0] = resize(videos[i, j, 0], (32, 112), order=1, anti_aliasing=True, preserve_range=True)
        if (i+1) % 50 == 0: print(f"  Videos {i+1}/{n}...")
    v32.flush()
    
    print("Resizing Gamma Cache...")
    gammas = np.load(CACHE_GAMMA_64, mmap_mode='r')
    g32 = np.lib.format.open_memmap(CACHE_GAMMA_32, mode='w+', dtype='float32', shape=(n, t, 7, 32, 112))
    
    for i in range(n):
        for j in range(t):
            for k in range(7):
                g32[i, j, k] = resize(gammas[i, j, k], (32, 112), order=1, anti_aliasing=True, preserve_range=True)
        if (i+1) % 20 == 0: print(f"  Gammas {i+1}/{n}...")
    g32.flush()
    print("Done!")

if __name__ == "__main__":
    resize_caches()
