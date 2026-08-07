#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# Set up project paths similar to one_cell_quantification_1CH.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SingleCellDataAnalysis.signal_cor import quantify_all_cells_acor

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies/2025_09_17")
OUT_DIR = MOVIE_ROOT / "unaligned_pairs_quant"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # 1. Load sequence linkages
    linkage_path = MOVIE_ROOT / "sequence_linkage.json"
    with open(linkage_path) as f:
        linkage = json.load(f)

    # 2. Load QC data
    qc_data = {}
    for seq in ['F0', 'F1', 'F2']:
        qc_path = MOVIE_ROOT / seq / f"qc_{seq}.json"
        if qc_path.exists():
            with open(qc_path) as f:
                qc_data.update(json.load(f))

    # 3. Gather all curated tracks (GFP1 and GFP2)
    curated_tracks = []
    global_id_counter = 1

    for seq in ['F0', 'F1']:
        seq_link = linkage.get(seq)
        if not seq_link:
            continue
        global_cells = seq_link['global_cells']
        films = seq_link['films']
        
        for gcid, local_ids in sorted(global_cells.items()):
            status = qc_data.get(gcid)
            if status in ['good', 'corrected']:
                g1_id = local_ids[0]
                g2_id = local_ids[2]
                bf1_id = local_ids[1]
                
                if g1_id != -1:
                    curated_tracks.append({
                        'seq': seq,
                        'film': films[0],
                        'local_id': g1_id,
                        'source': 'GFP1',
                        'tp': 1,
                        'pair_bf1_id': bf1_id,
                        'pair_index': global_id_counter,
                        'global_cell_id': gcid
                    })
                if g2_id != -1:
                    curated_tracks.append({
                        'seq': seq,
                        'film': films[2],
                        'local_id': g2_id,
                        'source': 'GFP2',
                        'tp': 2,
                        'pair_bf1_id': bf1_id,
                        'pair_index': global_id_counter,
                        'global_cell_id': gcid
                    })
                global_id_counter += 1

    print(f"Total curated tracks identified: {len(curated_tracks)}")

    stack_rows = []
    map_rows = []
    new_id = 1

    for track in curated_tracks:
        seq = track['seq']
        film = track['film']
        lid = track['local_id']
        source = track['source']
        tp = track['tp']
        bf = track['pair_bf1_id']
        pair_idx = track['pair_index']
        gcid = track['global_cell_id']
        
        csv_p = MOVIE_ROOT / film / f"TrackedCells_{film}" / f"cell_{lid}_data.csv"
        
        # Self-healing: if file doesn't exist or is empty, run quantification
        # Ensure the process is the same as in one_cell_quantification_1CH.py
        if not csv_p.exists() or os.path.getsize(csv_p) == 0:
            cmd = f"KMP_DUPLICATE_LIB_OK=TRUE python one_cell_quantification_1CH.py --cell_id {lid} --experiment_path \"/Volumes/X10 Pro/Movies/2025_09_17\" --file_name \"{film}\" --track_channel gfp"
            print(f"Quantifying missing cell {lid} in {film}...")
            subprocess.run(cmd, shell=True, cwd="/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        if not csv_p.exists() or os.path.getsize(csv_p) == 0:
            print(f"Warning: could not generate data for {film} cell {lid}")
            continue
            
        try:
            df = pd.read_csv(csv_p)
            df_parent = df[df['cell_id'].astype(str) == str(lid)].copy()
            if df_parent.empty:
                print(f"Warning: no parent cell row found in {csv_p}")
                continue
        except Exception as e:
            print(f"Error reading {csv_p}: {e}")
            continue
            
        # Ensure correct columns
        if 'pol1_int' in df_parent.columns and 'cyt_int' in df_parent.columns:
            df_parent['pol1_int_corr'] = df_parent['pol1_int'] - df_parent['cyt_int']
        else:
            df_parent['pol1_int_corr'] = 0.0
            
        if 'pol2_int' in df_parent.columns and 'cyt_int' in df_parent.columns:
            df_parent['pol2_int_corr'] = df_parent['pol2_int'] - df_parent['cyt_int']
        else:
            df_parent['pol2_int_corr'] = 0.0
            
        # Reindex to ensure exactly 101 frames (0 to 100)
        df_parent = df_parent.set_index('time_point')
        df_parent = df_parent.reindex(range(101))
        df_parent['pol1_int_corr'] = df_parent['pol1_int_corr'].ffill().bfill().fillna(0.0)
        df_parent['pol2_int_corr'] = df_parent['pol2_int_corr'].ffill().bfill().fillna(0.0)
        df_parent = df_parent.reset_index()
        
        # Assign new_cell_id
        df_parent['cell_id'] = new_id
        df_parent['source'] = source
        df_parent['tp'] = tp
        df_parent['pair_index'] = pair_idx
        df_parent['pair_bf1_id'] = bf
        df_parent['orig_gfp_id'] = lid
        df_parent['field'] = seq
        df_parent['global_cell_id'] = gcid
        
        stack_rows.append(df_parent[["time_point","cell_id","pol1_int_corr","pol2_int_corr","source","tp","pair_index","pair_bf1_id","orig_gfp_id","field","global_cell_id"]])
        
        map_rows.append({
            "new_cell_id": new_id,
            "orig_gfp_id": lid,
            "field": seq,
            "source": source,
            "global_cell_id": gcid
        })
        new_id += 1

    df_stacked = pd.concat(stack_rows, ignore_index=True)
    df_map = pd.DataFrame(map_rows)

    # Save compiled outputs
    data_csv = OUT_DIR / "stacked_gfp1_gfp2_for_unaligned_pairs.csv"
    map_csv = OUT_DIR / "id_map_unaligned.csv"

    df_stacked.to_csv(data_csv, index=False)
    df_map.to_csv(map_csv, index=False)

    print(f"Saved stacked data to {data_csv}")
    print(f"Total traces stacked: {len(df_map)}")

    # Update periodicity
    cell_ids = df_map["new_cell_id"].unique().tolist()
    acor_csv = OUT_DIR / "acor_detrended_results.csv"

    print("Running quantify_all_cells_acor to update periodicity...")
    quantify_all_cells_acor(df_stacked, cell_ids, delta_threshold=10, visualize=False, filename=str(acor_csv))

    print(f"Periodicity successfully updated and saved to {acor_csv}")

if __name__ == '__main__':
    main()
