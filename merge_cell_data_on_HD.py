#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:50:05 2025

@author: user
"""


import os
import glob
import pandas as pd

# Root directory containing all movie folders
root_dir = "/Volumes/Ian's microscopy/2025_06_25/"

# Loop through each item in the root directory
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    # Check if it's a directory (e.g., A14_1, A14_2, etc.)
    if os.path.isdir(subfolder_path):
        # Construct TrackedCells path (e.g., A14_1/TrackedCells_A14_1)
        tracked_folder = os.path.join(subfolder_path, f"TrackedCells_{subfolder}")

        # Skip if TrackedCells folder doesn't exist
        if not os.path.isdir(tracked_folder):
            print(f"Skipped (no TrackedCells folder): {tracked_folder}")
            continue

        # Find all cell data CSVs in this folder
        csv_files = glob.glob(os.path.join(tracked_folder, "cell_*_data.csv"))

        # Skip if no files found
        if not csv_files:
            print(f"No CSV files found in: {tracked_folder}")
            continue

        # Merge all CSVs
        df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # Save to merged CSV
        merged_path = os.path.join(tracked_folder, "all_cells_time_series.csv")
        df_all.to_csv(merged_path, index=False)

        print(f"Merged {len(csv_files)} files into: {merged_path}")


# Initialize tracking variables
merged_global = []
cell_id_offset = 0

# Sort for consistent processing order
for subfolder in sorted(os.listdir(root_dir)):
    subfolder_path = os.path.join(root_dir, subfolder)
    
    if not os.path.isdir(subfolder_path):
        continue

    tracked_folder = os.path.join(subfolder_path, f"TrackedCells_{subfolder}")
    csv_path = os.path.join(tracked_folder, "all_cells_time_series.csv")

    if not os.path.isfile(csv_path):
        print(f"Skipped (no merged CSV): {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Adjust cell_id
    if 'cell_id' not in df.columns:
        print(f"Skipped (no 'cell_id' column): {csv_path}")
        continue

    df['cell_id'] += cell_id_offset
    cell_id_offset = df['cell_id'].max() + 1  # Prepare offset for next movie

    # Optional: Add a movie identifier column
    df['movie'] = subfolder

    merged_global.append(df)
    print(f"Appended data from: {csv_path}")

# Combine and save
if merged_global:
    df_merged = pd.concat(merged_global, ignore_index=True)
    out_path = os.path.join(root_dir, "complete_cells_time_series.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"Saved merged data to: {out_path}")
else:
    print("No CSVs found to merge.")
