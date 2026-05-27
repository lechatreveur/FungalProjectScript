import numpy as np
import os
from skimage.transform import resize

BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_GAMMA_64 = os.path.join(BASE_DIR, "gamma_cache.npy") # This is the 16GB PADDED one
CACHE_GAMMA_32 = os.path.join(BASE_DIR, "gamma_cache_32x112_padded.npy")

def resize_gamma():
    print("Resizing Padded Gamma Cache from 64x224 to 32x112...")
    gammas = np.load(CACHE_GAMMA_64, mmap_mode='r')
    n, t, c, h, w = gammas.shape
    
    g32 = np.lib.format.open_memmap(CACHE_GAMMA_32, mode='w+', dtype='float32', shape=(n, t, c, 32, 112))
    
    for i in range(n):
        for j in range(t):
            for k in range(c):
                # order=1 is bilinear, fast and good for probabilities
                g32[i, j, k] = resize(gammas[i, j, k], (32, 112), order=1, anti_aliasing=True, preserve_range=True)
        if (i+1) % 20 == 0: print(f"  Processed {i+1}/{n} cells...")
    
    g32.flush()
    print("Done!")

if __name__ == "__main__":
    resize_gamma()
