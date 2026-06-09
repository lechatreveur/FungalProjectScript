#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.multi_field import run_field_sequence, make_field_sequence
from SingleCellDataAnalysis.population_movie_gui import build_global_id_maps_from_pairings
from SingleCellDataAnalysis.visualization import load_good_gfp1_gfp2_for_field
from SingleCellDataAnalysis.signal_analysis import quantify_all_cells
from SingleCellDataAnalysis.signal_cor import quantify_all_cells_acor

# ==========================================
# Process M133 Experiment
# ==========================================
def process_m133(working_dir, fields, film_names, use_bf1_only=False):
    print(f"\n{'='*60}")
    print(f"Processing M133 in {working_dir}")
    print(f"{'='*60}")
    
    # 1. Re-run or load field sequences to get all_res and all_global_maps
    all_res = {}
    all_global_maps = {}
    
    for field in fields:
        out_dir = os.path.join(working_dir, "pipeline_outputs", f"{field}_maps_and_aligned")
        
        # Determine sequence based on exp naming structure
        # Use only D_2 (GFP), D_3 (BF), D_4 (GFP) because D was processed as BF and lacks GFP intensity data.
        seq = [
            ("gfp", f"{film_names[2]}_{field}"),
            ("bf",  f"{film_names[3]}_{field}"),
            ("gfp", f"{film_names[4]}_{field}")
        ]
        anchor = f"{film_names[2]}_{field}"
            
        res = run_field_sequence(
            working_dir, seq, out_dir, iou_min=0.01,
            manifest_relpath="training_dataset/pipeline_manifest.csv"
        )
        all_res[field] = res
        
        gmaps = build_global_id_maps_from_pairings(
            field_seq=seq, pair_mappings=res, anchor_film=anchor
        )
        all_global_maps[field] = gmaps
        
    # 2. Extract good gfp traces and stack them
    stack_rows = []
    map_rows = []
    new_id = 1
    
    for field in fields:
        # We use a dummy film names list so that visualization.py extracts D_2 as gfp1 and D_4 as gfp2
        dummy_film_names = [film_names[2], film_names[3], film_names[4], "DUMMY_1", "DUMMY_2"]
        d1, d2 = load_good_gfp1_gfp2_for_field(
            field, dummy_film_names, working_dir, all_global_maps, all_res,
            only_has_septum_bf1=use_bf1_only
        )
        
        # d1 and d2 have cell_id as "F0:123". Let's convert to unique integers
        for which, df in [("GFP1", d1), ("GFP2", d2)]:
            if df.empty: continue
            
            # Group by the original string cell_id
            for orig_cid, grp in df.groupby("cell_id"):
                grp = grp.copy()
                grp["cell_id"] = new_id
                
                # Keep necessary columns
                grp["source"] = which
                stack_rows.append(grp[["time_point", "cell_id", "pol1_int_corr", "pol2_int_corr", "source", "field"]])
                
                map_rows.append({
                    "new_cell_id": new_id,
                    "orig_str_id": orig_cid,
                    "field": field,
                    "source": which
                })
                new_id += 1
                
    if not stack_rows:
        print(f"No valid traces found for M133")
        return
        
    df_stacked = pd.concat(stack_rows, ignore_index=True)
    df_map = pd.DataFrame(map_rows)
    
    # 3. Save
    out_dir = os.path.join(working_dir, "unaligned_pairs_quant")
    os.makedirs(out_dir, exist_ok=True)
    
    data_csv = os.path.join(out_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
    map_csv = os.path.join(out_dir, "id_map_unaligned.csv")
    
    df_stacked.to_csv(data_csv, index=False)
    df_map.to_csv(map_csv, index=False)
    
    print(f"Saved stacked data to: {data_csv}")
    print(f"Total traces (cells): {df_map.shape[0]}")
    
    # 4. Quantify
    cell_ids = df_map["new_cell_id"].unique().tolist()
    
    print(f"Running standard quantification (model fits)...")
    fits_csv = os.path.join(out_dir, "model_fits_by_cell.csv")
    _ = quantify_all_cells(
        df_stacked, cell_ids, 
        feature1='pol1_int_corr', feature2='pol2_int_corr', 
        delta_threshold=10, filename=fits_csv
    )
    
    print(f"Running autocorrelation quantification...")
    acor_csv = os.path.join(out_dir, "acor_detrended_results.csv")
    _ = quantify_all_cells_acor(
        df_stacked, cell_ids, 
        delta_threshold=10, visualize=False, filename=acor_csv
    )

# ==========================================
# Run
# ==========================================
if __name__ == "__main__":
    
    M133_WORKING_DIR = "/Volumes/X10 Pro/Movies/2026_04_29_M133/"
    M133_FILMS = [
        "YES_Scd1_D", "YES_Scd1_D_1", "YES_Scd1_D_2",
        "YES_Scd1_D_3", "YES_Scd1_D_4", "YES_Scd1_D_5"
    ]
    
    FIELDS = ["F0", "F1", "F2"]
    
    process_m133(M133_WORKING_DIR, FIELDS, M133_FILMS)
