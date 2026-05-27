import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_data_loader import (
    FILM_FOLDER_MAP, EXPERIMENT_BASES, resolve_cell_info_sept17, resolve_cell_info_generic,
    STACKED_CSV_PATHS, ID_MAP_CSV_PATHS
)
from Cell_tracking_functions import rle_decode

def main():
    # Load support files
    stacked_dfs = {exp: pd.read_csv(path) for exp, path in STACKED_CSV_PATHS.items() if os.path.exists(path)}
    id_map_dfs = {exp: pd.read_csv(path) for exp, path in ID_MAP_CSV_PATHS.items() if path and os.path.exists(path)}

    samples = [
        ('June25_20m', 110),
        ('Sept17', 50),
        ('M92', 10),
        ('M93', 5)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    plt.suptitle('Gallery: Selected Cells Boxed in Raw GFP (c_0) Frames', fontsize=20)

    for i, (exp_label, local_id) in enumerate(samples):
        # Resolve film and orig ID
        film_name, orig_id = None, None
        if exp_label == 'Sept17':
            film_name, orig_id = resolve_cell_info_sept17(local_id, stacked_dfs['Sept17'])
        elif exp_label == 'June25_20m':
            film_name = FILM_FOLDER_MAP.get(('June25_20m', 'GFP1', 'F0'))
            orig_id = local_id
        else:
            film_name, orig_id = resolve_cell_info_generic(local_id, id_map_dfs.get(exp_label), exp_label)
            
        base_dir = EXPERIMENT_BASES[exp_label]
        film_folder = os.path.join(base_dir, film_name)
        frames_dir = os.path.join(film_folder, f"Frames_{film_name}")
        tracked_dir = os.path.join(film_folder, f"TrackedCells_{film_name}")
        
        # Load mask to get box
        mask_csv = os.path.join(tracked_dir, f"cell_{orig_id}_masks.csv")
        df_masks = pd.read_csv(mask_csv)
        
        # Priority for RLE column
        rle_col = 'rle'
        if 'rle_gfp' in df_masks.columns: rle_col = 'rle_gfp'
        
        rle = df_masks.iloc[0][rle_col]
        H, W = int(df_masks.iloc[0]['height']), int(df_masks.iloc[0]['width'])
        mask = rle_decode(rle, (H, W))
        
        # Bbox
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        rmin, rmax, cmin, cmax = rows[0], rows[-1], cols[0], cols[-1]
        
        # Load Raw Image (c_0)
        img_path = os.path.join(frames_dir, f"{film_name}_t_{0:03d}_c_0.tif")
        if not os.path.exists(img_path):
            # Try 2-digit for June25
            img_path = os.path.join(frames_dir, f"{film_name}_t_{0:02d}_c_0.tif")
            
        if os.path.exists(img_path):
            img = imread(img_path)
            # Zoom in slightly around the cell to make it visible
            pad = 200
            z_rmin, z_rmax = max(0, rmin-pad), min(H, rmax+pad)
            z_cmin, z_cmax = max(0, cmin-pad), min(W, cmax+pad)
            
            axes[i].imshow(img[z_rmin:z_rmax, z_cmin:z_cmax], cmap='viridis', 
                           extent=[z_cmin, z_cmax, z_rmax, z_rmin])
            rect = patches.Rectangle((cmin, rmin), cmax-cmin, rmax-rmin, 
                                     linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].set_title(f"{exp_label} Cell {local_id}\n{os.path.basename(img_path)}", fontsize=12)
        else:
            axes[i].set_title(f"{exp_label} - NOT FOUND")
        
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('/Volumes/X10 Pro/FungalProject_Outputs/video_ae/boxed_gallery.png', dpi=150)
    print("Gallery saved.")

if __name__ == "__main__":
    main()
