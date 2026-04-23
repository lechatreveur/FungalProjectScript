#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_trajectories(experiments_dict):
    """
    Loads raw pol1 and pol2 trajectories from the stacked CSVs.
    Returns:
        X (np.ndarray): Tensor of shape (num_cells, 101, 2)
        global_ids (list): List of unique global cell IDs
        labels (list): List of experiment names for each cell
        scaler (StandardScaler): The fitted scaler
    """
    df_list = []
    
    # 1. Load Data
    for exp_name, exp_dir in experiments_dict.items():
        csv_path = os.path.join(exp_dir, "unaligned_pairs_quant", "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
        if not os.path.exists(csv_path):
            # Fallback for Sept 17 which might be at root
            csv_path = os.path.join(exp_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
            
        if not os.path.exists(csv_path):
            print(f"Warning: Could not find stacked CSV for {exp_name}")
            continue
            
        df = pd.read_csv(csv_path)
        df['experiment'] = exp_name
        df['global_cell_id'] = exp_name + "_" + df['cell_id'].astype(str)
        df_list.append(df)
        
    if not df_list:
        raise ValueError("No data could be loaded.")
        
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # 2. Pivot into 3D Tensor
    # Ensure sorted by time_point
    df_combined.sort_values(by=['global_cell_id', 'time_point'], inplace=True)
    
    global_ids = []
    labels = []
    trajectories = []
    
    grouped = df_combined.groupby('global_cell_id')
    for gid, grp in grouped:
        if len(grp) != 101:
            print(f"Skipping {gid} due to incorrect length: {len(grp)}")
            continue
            
        pol1 = grp['pol1_int_corr'].values
        pol2 = grp['pol2_int_corr'].values
        
        # 3. Polarity Alignment (Pol1 is always the pole with the higher integral)
        if np.sum(pol2) > np.sum(pol1):
            pol1, pol2 = pol2, pol1  # Swap
            
        # Shape: (101, 2)
        trace = np.column_stack((pol1, pol2))
        
        trajectories.append(trace)
        global_ids.append(gid)
        labels.append(grp['experiment'].iloc[0])
        
    X = np.array(trajectories) # (N, 101, 2)
    
    # 4. Global Scaling
    # We fit a single StandardScaler on all values across time and channels to preserve relative amplitudes
    N, T, C = X.shape
    X_flat = X.reshape(-1, C) # (N*T, C)
    
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    
    X_scaled = X_flat_scaled.reshape(N, T, C)
    
    return X_scaled, global_ids, labels, scaler

if __name__ == "__main__":
    EXPERIMENTS = {
        "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
        "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
        "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
    }
    
    X, gids, labels, scaler = load_and_preprocess_trajectories(EXPERIMENTS)
    print(f"Loaded Trajectories Tensor Shape: {X.shape}")
    print(f"Total Cells: {len(gids)}")
    print("Mean:", np.mean(X, axis=(0, 1)))
    print("Std:", np.std(X, axis=(0, 1)))
