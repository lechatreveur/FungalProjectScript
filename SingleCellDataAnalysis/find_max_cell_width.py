import os
import pandas as pd
import numpy as np
import sys

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_data_loader import EXPERIMENT_BASES, resolve_cell_info_sept17, resolve_cell_info_generic, ID_MAP_CSV_PATHS, FILM_FOLDER_MAP

OUTPUT_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_GIDS = os.path.join(OUTPUT_DIR, "video_gids.txt")

def find_max_width():
    with open(CACHE_GIDS, 'r') as f:
        gids = [line.strip() for line in f]
    
    EXPERIMENTS = {
        'Sept17':     '/Volumes/X10 Pro/Movies/2025_09_17/',
        'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/',
        'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/',
        'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/',
    }

    df_map_m92 = pd.read_csv(ID_MAP_CSV_PATHS['M92']) if os.path.exists(ID_MAP_CSV_PATHS['M92']) else None
    df_map_m93 = pd.read_csv(ID_MAP_CSV_PATHS['M93']) if os.path.exists(ID_MAP_CSV_PATHS['M93']) else None
    df_map_june25 = pd.read_csv(ID_MAP_CSV_PATHS['June25_20m']) if os.path.exists(ID_MAP_CSV_PATHS['June25_20m']) else None
    df_stack_sept17 = pd.read_csv(os.path.join(EXPERIMENTS['Sept17'], "unaligned_pairs_quant", "stacked_gfp1_gfp2_for_unaligned_pairs.csv"))

    known_labels = sorted(EXPERIMENTS.keys(), key=lambda x: -len(x))

    max_widths = []

    for gid in gids:
        exp_label = None
        cell_id = None
        for lbl in known_labels:
            if gid.startswith(lbl + '_'):
                exp_label = lbl
                cell_id = int(gid[len(lbl)+1:])
                break
        
        if exp_label == 'Sept17':
            film_name, orig_cell_id = resolve_cell_info_sept17(cell_id, df_stack_sept17)
        elif exp_label == 'M92':
            film_name, orig_cell_id = resolve_cell_info_generic(cell_id, df_map_m92, exp_label)
        elif exp_label == 'M93':
            film_name, orig_cell_id = resolve_cell_info_generic(cell_id, df_map_m93, exp_label)
        elif exp_label == 'June25_20m':
            film_name = FILM_FOLDER_MAP.get(('June25_20m', 'GFP1', 'F0'))
            orig_cell_id = cell_id
            
        film_folder = EXPERIMENTS[exp_label] + film_name + '/'
        tracked_cells_dir = film_folder + f"TrackedCells_{film_name}/"
        mask_csv_path = tracked_cells_dir + f"cell_{orig_cell_id}_masks.csv"
        
        if not os.path.exists(mask_csv_path): continue
        
        df_masks = pd.read_csv(mask_csv_path)
        from Cell_tracking_functions import rle_decode
        from skimage.measure import label, regionprops
        
        for t in [0, 50, 100]: # Sample 3 timepoints per cell to be faster
            rle = df_masks.iloc[t]['rle'] if 'rle' in df_masks.columns else df_masks.iloc[t]['rle_gfp']
            if not isinstance(rle, str): continue
            h, w = int(df_masks.iloc[t]['height']), int(df_masks.iloc[t]['width'])
            mask = rle_decode(rle, (h, w))
            props = regionprops(label(mask))
            if props:
                max_widths.append(max(p.major_axis_length for p in props))
                
    if max_widths:
        print(f"Max observed major axis length: {max(max_widths):.1f}")
        print(f"99th percentile: {np.percentile(max_widths, 99):.1f}")
    else:
        print("No widths found.")


if __name__ == "__main__":
    find_max_width()
