#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.signal_analysis import quantify_all_cells
from SingleCellDataAnalysis.signal_cor import quantify_all_cells_acor

# ==== 1. Configuration ====
EXP_DIR = "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
TRACKED_DIR = os.path.join(EXP_DIR, "TrackedCells_A14_10_20min")
OUT_DIR = os.path.join(EXP_DIR, "unaligned_pairs_quant")
OUT_CSV = os.path.join(OUT_DIR, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")

TARGET_LEN = 101

# ==== 2. Processing ====
def prepare_20min_data():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    search_pattern = os.path.join(TRACKED_DIR, "cell_*_data.csv")
    csv_files = glob.glob(search_pattern)
    print(f"Found {len(csv_files)} cell data files.")
    
    stack_rows = []
    
    for f in csv_files:
        basename = os.path.basename(f)
        # File format: cell_ID_data.csv
        cell_id_str = basename.replace("cell_", "").replace("_data.csv", "")
        
        # 1. Filter out daughter cells
        if "_1" in cell_id_str or "_2" in cell_id_str:
            continue
            
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading {basename}: {e}")
            continue
            
        # 2. Keep only length 101 trajectories
        if len(df) != TARGET_LEN:
            continue
            
        # Ensure it has the right columns
        if not {'pol1_int', 'pol2_int', 'cyt_int', 'time_point'}.issubset(df.columns):
            continue
            
        # 3. Calculate corrected intensities
        df['pol1_int_corr'] = df['pol1_int'] - df['cyt_int']
        df['pol2_int_corr'] = df['pol2_int'] - df['cyt_int']
        
        # Format for stacked CSV
        # The autoencoder expects: time_point, cell_id, pol1_int_corr, pol2_int_corr, source, field
        # We can put dummy values for source and field if they aren't needed, but it's best to match exactly.
        df['cell_id'] = cell_id_str
        df['source'] = "GFP1" # Doesn't matter for autoencoder, just to match format
        df['field'] = "F0" # Doesn't matter for autoencoder
        
        stack_rows.append(df[['time_point', 'cell_id', 'pol1_int_corr', 'pol2_int_corr', 'source', 'field']])
        
    if not stack_rows:
        print("No valid trajectories found!")
        return
        
    df_stacked = pd.concat(stack_rows, ignore_index=True)
    df_stacked.to_csv(OUT_CSV, index=False)
    
    num_cells = df_stacked['cell_id'].nunique()
    print(f"✅ Saved {num_cells} valid cells to {OUT_CSV}")
    
    # ==== 4. Quantify ====
    print("🔬 Running quantification to extract 11 features...")
    cell_ids = df_stacked['cell_id'].unique().tolist()
    
    fits_csv = os.path.join(OUT_DIR, "model_fits_by_cell.csv")
    _ = quantify_all_cells(
        df_stacked, cell_ids, 
        feature1='pol1_int_corr', feature2='pol2_int_corr', 
        delta_threshold=10, filename=fits_csv
    )
    
    acor_csv = os.path.join(OUT_DIR, "acor_detrended_results.csv")
    _ = quantify_all_cells_acor(
        df_stacked, cell_ids, 
        delta_threshold=10, visualize=False, filename=acor_csv
    )
    
    print("✅ Quantification complete!")

if __name__ == "__main__":
    prepare_20min_data()
