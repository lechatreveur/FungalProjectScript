#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler # Removed dependency
import sys

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.AE_data_loader import load_and_preprocess_trajectories
from SingleCellDataAnalysis.PCA_utils import load_experiment_features

def load_feature_constrained_data(experiments_dict):
    """
    Loads both the 101-frame trajectories and the 11 engineered features.
    Ensures that they match cell-by-cell.
    Returns:
        X_traj (np.ndarray): Shape (N, 101, 2)
        X_feat (np.ndarray): Shape (N, 11)
        gids (list): Global cell IDs
        labels (list): Experiment labels
        scaler_traj (StandardScaler): Scaler for trajectories
        scaler_feat (object): Manual scaler for the 11 features
    """
    # 1. Load Trajectories (Using existing loader)
    X_traj, gids, labels, scaler_traj = load_and_preprocess_trajectories(experiments_dict)
    
    # 2. Load Features
    df_list = []
    for exp_name, exp_dir in experiments_dict.items():
        df_exp = load_experiment_features(exp_dir)
        
        csv_path = os.path.join(exp_dir, "unaligned_pairs_quant", "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(exp_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
            
        if os.path.exists(csv_path):
            df_stacked = pd.read_csv(csv_path)
            if 'global_cell_id' in df_stacked.columns:
                df_map = df_stacked.drop_duplicates('cell_id')
                mapping = {
                    r['cell_id']: exp_name + "_" + str(r['global_cell_id']) + "_" + str(r['source'])
                    for _, r in df_map.iterrows()
                }
            else:
                mapping = {cid: exp_name + "_" + str(cid) for cid in df_stacked['cell_id'].unique()}
        else:
            mapping = {cid: exp_name + "_" + str(cid) for cid in df_exp.index}
            
        df_exp['global_cell_id'] = df_exp.index.map(mapping)
        df_exp.set_index('global_cell_id', inplace=True)
        df_list.append(df_exp)
        
    df_features_all = pd.concat(df_list)
    
    # Define the exact 11 features
    feature_cols = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
    
    # 3. Align Data
    # Some trajectories might have been dropped or some features might not have been calculated successfully
    aligned_traj = []
    aligned_feat = []
    aligned_gids = []
    aligned_labels = []
    
    for i, gid in enumerate(gids):
        if gid in df_features_all.index:
            feat_row = df_features_all.loc[gid, feature_cols].values
            
            # Check for NaNs in the features
            if pd.isna(feat_row).any():
                continue
                
            aligned_traj.append(X_traj[i])
            aligned_feat.append(feat_row)
            aligned_gids.append(gid)
            aligned_labels.append(labels[i])
            
    X_traj_final = np.array(aligned_traj)
    X_feat_raw = np.array(aligned_feat)
    
    # 4. Scale Features
    X_feat_raw = np.nan_to_num(X_feat_raw, posinf=np.nan, neginf=np.nan)
    mean_f = np.nanmean(X_feat_raw, axis=0)
    std_f = np.nanstd(X_feat_raw, axis=0)
    
    # Fill any remaining NaNs in mean/std with 0 and 1
    mean_f = np.nan_to_num(mean_f, nan=0.0)
    std_f = np.nan_to_num(std_f, nan=1.0)
    
    class ManualScalerFeat:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std
        def transform(self, x):
            return (x - self.mean) / (self.std + 1e-8)
            
    scaler_feat = ManualScalerFeat(mean_f, std_f)
    X_feat_scaled = scaler_feat.transform(X_feat_raw)
    X_feat_scaled = np.nan_to_num(X_feat_scaled, nan=0.0)
    
    print(f"✅ Loaded {len(aligned_gids)} cells successfully with BOTH trajectories and 11 features.")
    
    return X_traj_final, X_feat_scaled, aligned_gids, aligned_labels, scaler_traj, scaler_feat

if __name__ == "__main__":
    EXPERIMENTS = {
        "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
        "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
        "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
        "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
    }
    
    Xt, Xf, gids, labels, s_t, s_f = load_feature_constrained_data(EXPERIMENTS)
    print(f"Trajectories shape: {Xt.shape}")
    print(f"Features shape: {Xf.shape}")
