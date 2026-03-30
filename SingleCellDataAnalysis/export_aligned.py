#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:54:05 2025

@author: user
"""

# export_aligned.py

import os
import pandas as pd

# def generate_aligned_time_column(df_all, best_shifts, time_points):
#     """
#     Adds a new column 'aligned_time' to df_all using best_shifts.
#     """
#     aligned_time_records = []

#     for cell_id in df_all['cell_id'].unique():
#         shift = best_shifts[cell_id]

#         df_cell = df_all[df_all["cell_id"] == cell_id][["cell_id", "time_point"]].copy()
#         df_cell["aligned_time"] = df_cell["time_point"] + shift

#         aligned_time_records.append(df_cell)

#     # Merge all
#     df_aligned_time = pd.concat(aligned_time_records, ignore_index=True)
#     df_all = df_all.merge(df_aligned_time, on=["cell_id", "time_point"])
#     return df_all
def generate_aligned_time_column(df_all, best_shifts, time_points):
    """
    Adds a new column 'aligned_time' to df_all using best_shifts.
    
    Parameters:
        df_all: pandas DataFrame
        best_shifts: dict {cell_id: shift}
        time_points: list or array of time_point values
    
    Returns:
        df_all with an added 'aligned_time' column.
    """
    aligned_time_records = []

    for cell_id in df_all['cell_id'].unique():
        if cell_id not in best_shifts:
            continue  # Safety check

        shift = best_shifts[cell_id]
        df_cell = df_all[df_all["cell_id"] == cell_id][["cell_id", "time_point"]].copy()
        df_cell["aligned_time"] = df_cell["time_point"] + shift
        aligned_time_records.append(df_cell)

    # Merge all aligned time data
    df_aligned_time = pd.concat(aligned_time_records, ignore_index=True)

    # Drop old 'aligned_time' column if it already exists
    if 'aligned_time' in df_all.columns:
        df_all = df_all.drop(columns=['aligned_time'])

    # Merge new aligned times
    df_all = df_all.merge(df_aligned_time, on=["cell_id", "time_point"])
    return df_all


def export_aligned_dataframe(df_all, output_dir, filename="combined_all_cells_with_aligned_time.csv"):
    """
    Saves the aligned DataFrame to a CSV file.
    """
    output_path = os.path.join(output_dir, filename)
    df_all.to_csv(output_path, index=False)
    print(f"✅ Saved aligned data to: {output_path}")
    return output_path


def trace_cell_metadata(df_all, cell_ids, columns=['cell_id', 'original_cell_id', 'source_file']):
    """
    Return a summary DataFrame for specified cell_ids, including traceable metadata.
    """
    traced = df_all[df_all['cell_id'].isin(cell_ids)][columns].drop_duplicates().reset_index(drop=True)
    print("🔍 Traced metadata for selected cells:")
    print(traced)
    return traced



from SingleCellDataAnalysis.config import WORKING_DIR
from SingleCellDataAnalysis.load_data import load_preprocessed_data


def trace_cells_by_id(cell_id_list, work_dir=WORKING_DIR):
    """
    Load preprocessed data and trace metadata for given cell_id list.
    
    Parameters:
        cell_id_list (list of int): The cell IDs to investigate.
        work_dir (str): Path to working directory (default: from config).
    
    Returns:
        pd.DataFrame: Traced metadata.
    """
    df_all, *_ = load_preprocessed_data(work_dir)
    return trace_cell_metadata(df_all, cell_id_list)
