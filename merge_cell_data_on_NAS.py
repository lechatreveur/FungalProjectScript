#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust merger for tracked cell CSVs:
- Skips empty/bad CSVs and logs them
- Builds per-movie merged CSVs
- Builds a global merged CSV with adjusted cell_id offsets
"""

import os
import glob
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

# ---------- config ----------
#root_dir = "/Volumes/Movies/2025_09_17/"
root_dir = "/Volumes/Movies/2026_01_08_M93/"
pattern = "cell_*_data.csv"
per_movie_out = "all_cells_time_series.csv"
global_out = "complete_cells_time_series.csv"
# ----------------------------

def read_csv_safe(path):
    """Read a CSV, returning (df, reason_if_skipped_or_None)."""
    try:
        # Fast check: empty file on disk
        if os.path.getsize(path) == 0:
            return None, "empty file (0 bytes)"
        df = pd.read_csv(path, on_bad_lines='skip')  # skip malformed lines if any
        # Skip frames with no columns or no rows
        if df is None or df.empty or len(df.columns) == 0:
            return None, "no columns/rows after parse"
        return df, None
    except EmptyDataError:
        return None, "EmptyDataError (no columns to parse)"
    except ParserError as e:
        return None, f"ParserError: {e}"
    except UnicodeDecodeError as e:
        return None, f"UnicodeDecodeError: {e}"
    except Exception as e:
        return None, f"Unexpected read error: {type(e).__name__}: {e}"

# First pass: build per-movie merged CSVs
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    tracked_folder = os.path.join(subfolder_path, f"TrackedCells_{subfolder}")
    if not os.path.isdir(tracked_folder):
        print(f"Skipped (no TrackedCells folder): {tracked_folder}")
        continue

    csv_files = sorted(glob.glob(os.path.join(tracked_folder, pattern)))
    if not csv_files:
        print(f"No CSV files found in: {tracked_folder}")
        continue

    dfs, skipped = [], []
    for f in csv_files:
        df, err = read_csv_safe(f)
        if err is not None:
            skipped.append((f, err))
        else:
            dfs.append(df)

    if not dfs:
        print(f"All CSVs skipped in: {tracked_folder}")
        for f, reason in skipped:
            print(f"  - Skipped {os.path.basename(f)}: {reason}")
        continue

    df_all = pd.concat(dfs, ignore_index=True)

    merged_path = os.path.join(tracked_folder, per_movie_out)
    df_all.to_csv(merged_path, index=False)
    print(f"Merged {len(dfs)} / {len(csv_files)} files into: {merged_path}")
    if skipped:
        print("  Files skipped:")
        for f, reason in skipped:
            print(f"    - {os.path.basename(f)}: {reason}")

# Second pass: build global merged CSV with cell_id offsetting
merged_global = []
cell_id_offset = 0

for subfolder in sorted(os.listdir(root_dir)):
    subfolder_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    tracked_folder = os.path.join(subfolder_path, f"TrackedCells_{subfolder}")
    csv_path = os.path.join(tracked_folder, per_movie_out)

    if not os.path.isfile(csv_path):
        print(f"Skipped (no merged CSV): {csv_path}")
        continue

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Skipped (failed to read merged CSV): {csv_path} -> {e}")
        continue

    if 'cell_id' not in df.columns:
        print(f"Skipped (no 'cell_id' column): {csv_path}")
        continue

    # Ensure numeric cell_id
    try:
        df['cell_id'] = pd.to_numeric(df['cell_id'], errors='raise')
    except Exception as e:
        print(f"Skipped (non-numeric 'cell_id' values): {csv_path} -> {e}")
        continue

    df['cell_id'] = df['cell_id'] + cell_id_offset
    # Prepare offset for next movie (handle empty just in case)
    if not df.empty:
        cell_id_offset = int(df['cell_id'].max()) + 1

    df['movie'] = subfolder
    merged_global.append(df)
    print(f"Appended data from: {csv_path} (new offset -> {cell_id_offset})")

if merged_global:
    df_merged = pd.concat(merged_global, ignore_index=True)
    out_path = os.path.join(root_dir, global_out)
    df_merged.to_csv(out_path, index=False)
    print(f"Saved merged data to: {out_path}")
else:
    print("No CSVs found to merge globally.")
