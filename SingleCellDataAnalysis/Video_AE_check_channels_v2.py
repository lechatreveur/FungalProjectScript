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
    load_tif_frame, FILM_FOLDER_MAP, EXPERIMENT_BASES, resolve_cell_info_sept17, resolve_cell_info_generic,
    STACKED_CSV_PATHS, ID_MAP_CSV_PATHS, build_cell_video
)

def main():
    # Load support files
    stacked_dfs = {exp: pd.read_csv(path) for exp, path in STACKED_CSV_PATHS.items() if os.path.exists(path)}
    id_map_dfs = {exp: pd.read_csv(path) for exp, path in ID_MAP_CSV_PATHS.items() if path and os.path.exists(path)}

    # Sample cells to check
    samples = [
        ('June25_20m', 110),
        ('Sept17', 50),
        ('M92', 10),
        ('M93', 5)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    plt.suptitle('Verification: Actual Cells from Correct Channels', fontsize=16)

    for i, (exp_label, local_id) in enumerate(samples):
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
        
        try:
            film_folder = os.path.join(base_dir, film_name)
            frames_dir = os.path.join(film_folder, f"Frames_{film_name}")
            
            # Load one frame only (middle frame t=50)
            video = build_cell_video(film_folder, film_name, orig_id, frames_dir, n_frames=1, exp_label=exp_label)
            crop = video[0]
            
            axes[i].imshow(crop, cmap='viridis')
            axes[i].set_title(f"{exp_label} ID {local_id}\n{film_name}\nMax={crop.max():.2f}")
            print(f"Index {i}: {exp_label} ID {local_id} -> {film_name} OK")
        except Exception as e:
            axes[i].set_title(f"{exp_label} ERROR\n{str(e)[:20]}")
            print(f"Index {i}: {exp_label} ID {local_id} Error: {e}")
        
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('/Volumes/X10 Pro/FungalProject_Outputs/video_ae/c0_channel_verification_v2.png', dpi=150)
    print(f"Saved verification plot to c0_channel_verification_v2.png")

if __name__ == "__main__":
    main()
