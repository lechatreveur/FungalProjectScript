import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
import SingleCellDataAnalysis.Video_AE_data_loader as loader

# --- 1. Inject M133 into the loader configurations ---
loader.FILM_FOLDER_MAP.update({
    ('M133', 'GFP1', 'F0'): 'YES_Scd1_D_2_F0',
    ('M133', 'GFP1', 'F1'): 'YES_Scd1_D_2_F1',
    ('M133', 'GFP1', 'F2'): 'YES_Scd1_D_2_F2',
    ('M133', 'GFP2', 'F0'): 'YES_Scd1_D_4_F0',
    ('M133', 'GFP2', 'F1'): 'YES_Scd1_D_4_F1',
    ('M133', 'GFP2', 'F2'): 'YES_Scd1_D_4_F2',
})

loader.EXPERIMENT_BASES['M133'] = '/Volumes/X10 Pro/Movies/2026_04_29_M133/'
loader.STACKED_CSV_PATHS['M133'] = '/Volumes/X10 Pro/Movies/2026_04_29_M133/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv'
loader.ID_MAP_CSV_PATHS['M133'] = '/Volumes/X10 Pro/Movies/2026_04_29_M133/unaligned_pairs_quant/id_map_unaligned.csv'

# --- 2. Find all M133 global cell IDs that have 101 frames ---
csv_path = loader.STACKED_CSV_PATHS['M133']
df = pd.read_csv(csv_path)
df['global_cell_id'] = "M133_" + df['cell_id'].astype(str)

target_gids = []
for gid, grp in df.groupby('global_cell_id'):
    if len(grp) == 101:
        target_gids.append(gid)

print(f"Total M133 valid cells: {len(target_gids)}")

STRIPS_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/vertical_strips/"
os.makedirs(STRIPS_DIR, exist_ok=True)

print(f"Processing all {len(target_gids)} cells at once...")
videos, valid_gids = loader.load_video_dataset(target_gids, loader.EXPERIMENT_BASES, frame_h=32, frame_w=112)

for j, gid in enumerate(valid_gids):
    v = videos[j, :, 0, :, :] # (101, 32, 112)
    strip = np.vstack(v) # (101*32, 112)
    
    strip_norm = np.clip(strip / 1.5, 0, 1) * 255
    strip_uint8 = strip_norm.astype(np.uint8)
    
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('viridis')
    strip_rgb = (cm(strip_uint8 / 255.0)[:, :, :3] * 255).astype(np.uint8)
    
    img = Image.fromarray(strip_rgb)
    img.save(os.path.join(STRIPS_DIR, f"{gid}.png"))

print("Done generating all M133 vertical strips!")
