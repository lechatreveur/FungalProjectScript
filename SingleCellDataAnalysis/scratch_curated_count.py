import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_AE_3d_train import EXPERIMENTS

def test_curated():
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    print(f"Total loaded gids: {len(gids)}")
    
    sept17_df = pd.read_csv('/Volumes/X10 Pro/Movies/2025_09_17/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv')
    
    with open('/Volumes/X10 Pro/Movies/2025_09_17/sequence_linkage.json') as f:
        seq_linkage = json.load(f)
        
    qc_data = {}
    for seq in ['F0', 'F1', 'F2']:
        qc_path = f'/Volumes/X10 Pro/Movies/2025_09_17/{seq}/qc_{seq}.json'
        if os.path.exists(qc_path):
            with open(qc_path) as f:
                qc_data.update(json.load(f))
                
    film_cell_to_seq = {}
    for seq_name, seq_info in seq_linkage.items():
        films = seq_info.get('films', [])
        for seq_cell_id, cell_array in seq_info.get('global_cells', {}).items():
            for film_idx, film_id in enumerate(cell_array):
                if film_id != -1:
                    film_name = films[film_idx]
                    film_cell_to_seq[(film_name, film_id)] = seq_cell_id
                    
    keep_indices = []
    
    for i, gid in enumerate(gids):
        if not gid.startswith("Sept17_"):
            continue
            
        local_id = int(gid.split("_")[1])
        row = sept17_df[sept17_df.cell_id == local_id]
        if row.empty: continue
        row = row.iloc[0]
        
        source = row.source
        tp = int(row.tp)
        orig_id = int(row.orig_gfp_id)
        
        if source == 'GFP1' and tp == 1:
            film_name = 'A14_1TP1_F1'
        elif source == 'GFP2' and tp == 2:
            film_name = 'A14_1TP2_F1'
        else:
            film_name = None
            
        if not film_name: continue
        
        seq_cell = film_cell_to_seq.get((film_name, orig_id))
        
        status = None
        if seq_cell and seq_cell in qc_data:
            status = qc_data[seq_cell]
            
        if status in ['good', 'corrected']:
            keep_indices.append(i)
            
    print(f"Kept cells from Sept17: {len(keep_indices)}")

if __name__ == '__main__':
    test_curated()
