import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
import SingleCellDataAnalysis.Video_AE_data_loader as loader

EXP = "2025_09_17"
MOVIE_ROOT = "/Volumes/X10 Pro/Movies"
STRIPS_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/vertical_strips/"

def main():
    os.makedirs(STRIPS_DIR, exist_ok=True)
    
    # Load sequence linkages
    linkage_path = os.path.join(MOVIE_ROOT, EXP, "sequence_linkage.json")
    with open(linkage_path) as f:
        linkage = json.load(f)
        
    # Identify curated film cell mappings
    curated_film_cells = set() # elements: (film_name, orig_gfp_id)
    
    for seq in ['F0', 'F1']:
        qc_path = os.path.join(MOVIE_ROOT, EXP, seq, f"qc_{seq}.json")
        if not os.path.exists(qc_path):
            continue
        with open(qc_path) as f:
            qc = json.load(f)
            
        seq_link = linkage.get(seq)
        if not seq_link:
            continue
        f_cells = seq_link['global_cells']
        films = seq_link['films']
        
        for gid, status in qc.items():
            if status in ['good', 'corrected'] and gid in f_cells:
                local_ids = f_cells[gid]
                # GFP films are index 0 (GFP1) and index 2 (GFP2)
                for idx in [0, 2]:
                    lid = local_ids[idx]
                    if lid != -1:
                        film = films[idx]
                        curated_film_cells.add((film, lid))
                        
    print(f"Total curated film cells identified: {len(curated_film_cells)}")
    
    # Load the stacked CSV to map film cells to stable global cell IDs
    stacked_csv = loader.STACKED_CSV_PATHS['Sept17']
    df = pd.read_csv(stacked_csv)
    
    target_gids = []
    skipped_count = 0
    
    # df has columns: cell_id, source, tp, orig_gfp_id, field, global_cell_id
    for (cell_id, source, tp, orig_id, field, gcid), grp in df.groupby(['cell_id', 'source', 'tp', 'orig_gfp_id', 'field', 'global_cell_id']):
        # Map source/tp to film name
        if source == 'GFP1' and tp == 1:
            film = f'A14_1TP1_{field}'
        elif source == 'GFP2' and tp == 2:
            film = f'A14_1TP2_{field}'
        else:
            continue
            
        if (film, int(orig_id)) in curated_film_cells:
            row = grp.iloc[0]
            gcid = row['global_cell_id']
            src = row['source']
            gid_stable = f"Sept17_{gcid}_{src}"
            
            strip_path = os.path.join(STRIPS_DIR, f"{gid_stable}.png")
            if os.path.exists(strip_path):
                skipped_count += 1
                continue
            target_gids.append(gid_stable)
            
    # Deduplicate and sort
    target_gids = sorted(list(set(target_gids)))
    print(f"Skipped {skipped_count} already existing strips.")
    print(f"Total new global IDs to generate strips for: {len(target_gids)}")
    
    if not target_gids:
        print("All strips are already generated!")
        return
        
    # Generate strips
    print(f"Generating cell strips and saving to {STRIPS_DIR}...")
    videos, valid_gids = loader.load_video_dataset(target_gids, loader.EXPERIMENT_BASES, frame_h=32, frame_w=112)
    
    for j, gid in enumerate(valid_gids):
        v = videos[j, :, 0, :, :] # (101, 32, 112)
        strip = np.vstack(v) # (101*32, 112)
        
        # Max-normalize and scale appropriately for display
        strip_norm = np.clip(strip / 1.5, 0, 1) * 255
        strip_uint8 = strip_norm.astype(np.uint8)
        
        import matplotlib.pyplot as plt
        cm = plt.get_cmap('viridis')
        strip_rgb = (cm(strip_uint8 / 255.0)[:, :, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(strip_rgb)
        img.save(os.path.join(STRIPS_DIR, f"{gid}.png"))
        
    print("Done generating curated Sept17 vertical strips!")

if __name__ == '__main__':
    main()
