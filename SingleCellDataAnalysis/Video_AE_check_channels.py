import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread

# Import loader functions
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_data_loader import (
    load_tif_frame, FILM_FOLDER_MAP, EXPERIMENT_BASES, resolve_cell_info_sept17, resolve_cell_info_generic
)

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")

def main():
    # 1. Load GIDs
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]
    
    # 2. Pick 4 specific indices (e.g. 10, 100, 200, 300)
    indices = [10, 100, 200, 300]
    
    # Load all stacked CSVs and ID maps for resolution
    from SingleCellDataAnalysis.Video_AE_data_loader import STACKED_CSV_PATHS, ID_MAP_CSV_PATHS
    stacked_dfs = {exp: pd.read_csv(path) for exp, path in STACKED_CSV_PATHS.items() if os.path.exists(path)}
    id_map_dfs = {exp: pd.read_csv(path) for exp, path in ID_MAP_CSV_PATHS.items() if path and os.path.exists(path)}

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    plt.suptitle("Raw GFP Frame Verification", fontsize=16)

    for i, idx in enumerate(indices):
        gid = gids[idx]
        
        # Parse GID
        exp_label = None
        local_id = None
        for lbl in ['June25_20m', 'Sept17', 'M92', 'M93']:
            if gid.startswith(lbl + '_'):
                exp_label = lbl
                local_id = int(gid[len(lbl)+1:])
                break
        
        # Resolve
        film_name, orig_id = None, None
        if exp_label == 'Sept17':
            film_name, orig_id = resolve_cell_info_sept17(local_id, stacked_dfs['Sept17'])
        elif exp_label == 'June25_20m':
            film_name = FILM_FOLDER_MAP.get(('June25_20m', 'GFP1', 'F0'))
            orig_id = local_id
        else:
            film_name, orig_id = resolve_cell_info_generic(local_id, id_map_dfs.get(exp_label), exp_label)
            
        base_dir = EXPERIMENT_BASES[exp_label]
        
        # Check channel loading logic
        from SingleCellDataAnalysis.Video_AE_data_loader import build_cell_video, FRAME_H, FRAME_W
        
        # Load one frame only (middle frame t=50)
        try:
            # Re-implement build_cell_video slightly to get only one frame
            film_folder = os.path.join(base_dir, film_name)
            frames_dir = os.path.join(film_folder, f"Frames_{film_name}")
            
            # This will use the actual build_cell_video logic
            video = build_cell_video(film_folder, film_name, orig_id, frames_dir, n_frames=1)
            crop = video[0] # (H, W)
            
            axes[i].imshow(crop, cmap='viridis')
            axes[i].set_title(f"{gid}\n{film_name}\nMax={crop.max():.2f}")
            print(f"Index {idx}: GID={gid} Crop Max={crop.max():.2f}")
        except Exception as e:
            axes[i].set_title(f"{gid}\nERROR: {str(e)[:20]}")
            print(f"Index {idx}: GID={gid} Error: {e}")
        
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "channel_verification.png"), dpi=150)
    print(f"Saved verification plot to {os.path.join(BASE_DIR, 'channel_verification.png')}")

if __name__ == "__main__":
    main()
