#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:46:14 2025

@author: user
"""

# load_data.py

import os
import pandas as pd

def load_and_merge_csv(file_names, working_dir):
    df_all = pd.DataFrame()
    max_cell_id = 0

    for fname in file_names:
        csv_path = os.path.join(working_dir, f"{fname}/TrackedCells_{fname}/all_cells_time_series.csv")
        df = pd.read_csv(csv_path)
        df['original_cell_id'] = df['cell_id']
        df['cell_id'] += max_cell_id
        df['source_file'] = fname
        max_cell_id = df['cell_id'].max() + 1
        df_all = pd.concat([df_all, df], ignore_index=True)

    return df_all


def load_preprocessed_data(work_dir, filename="combined_all_cells_with_aligned_time.csv"):
    """
    Load preprocessed aligned single-cell data and reconstruct best_shifts.

    Returns:
        df_all (pd.DataFrame): The full dataframe.
        time_points (List[int]): Unique time_point values.
        global_time (List[float]): Unique aligned_time values.
        cell_ids (np.ndarray): Array of cell_ids.
        best_shifts (dict): Mapping of cell_id to shift.
    """
    aligned_csv = os.path.join(work_dir, filename)

    if not os.path.exists(aligned_csv):
        raise FileNotFoundError(f"❌ Processed file not found: {aligned_csv}\nRun full pipeline first.")

    print("⚡ Loading preprocessed aligned data...")
    df_all = pd.read_csv(aligned_csv)

    time_points = sorted(df_all['time_point'].unique())
    global_time = sorted(df_all['aligned_time'].unique())
    cell_ids = df_all['cell_id'].unique()

    print("🔁 Reconstructing best_shifts from aligned_time...")
    best_shifts = (
        df_all.groupby("cell_id")[["time_point", "aligned_time"]]
        .min()
        .eval("aligned_time - time_point")
        .astype(int)
        .to_dict()
    )

    return df_all, time_points, global_time, cell_ids, best_shifts

def offset_cell_ids_globally(df):
    """
    Ensures globally unique cell_id values across different source_file entries
    by offsetting each movie's cell IDs by the maximum cell_id seen so far.
    Assumes `source_file`, `original_cell_id`, and `cell_id` columns exist.
    """
    df_fixed = pd.DataFrame()
    max_cell_id = 0

    # Process each source_file separately
    for source, group in df.groupby('source_file', sort=True):
        group = group.copy()
        group['original_cell_id'] = group['cell_id']
        group['cell_id'] += max_cell_id
        max_cell_id = group['cell_id'].max() + 1
        df_fixed = pd.concat([df_fixed, group], ignore_index=True)
        
    # List of columns you want to ensure are numeric
    cols_to_convert = ['cyt_int', 'septum_int', 'pol1_int', 'pol2_int', 'nu_dis', 'nu_int']

    # Strip whitespace and convert to numeric
    for col in cols_to_convert:
        df_fixed[col] = pd.to_numeric(df_fixed[col].astype(str).str.strip(), errors='coerce')
    
    df_fixed.loc[df_fixed["time_point"] > 50, "time_point"] -= 50


    return df_fixed
