#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:08:10 2025

@author: user
"""
import os, glob, pandas as pd
from pandas.errors import EmptyDataError, ParserError

#root_dir = "/Users/user/Documents/FungalProject/TimeLapse/2025_07_23_M77_pc/"
root_dir = "/RAID1/working/R402/hsushen/FungalProject/Movies/2025_07_23_M77/"

def first_nonempty_line(path, max_lines=5):
    # Return first non-empty decoded line or ""
    with open(path, "rb") as fh:
        for _ in range(max_lines):
            line = fh.readline()
            if not line:
                break
            try:
                s = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                s = ""
            if s:
                return s
    return ""

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    tracked_folder = os.path.join(subfolder_path, f"TrackedCells_{subfolder}")
    if not os.path.isdir(tracked_folder):
        print(f"Skipped (no TrackedCells folder): {tracked_folder}")
        continue

    csv_files = glob.glob(os.path.join(tracked_folder, "cell_*_data.csv"))
    if not csv_files:
        print(f"No CSV files found in: {tracked_folder}")
        continue

    valid_dfs, skipped = [], []

    for f in csv_files:
        # 1) Skip files that are only BOM/newline/whitespace (often 1 byte)
        head = first_nonempty_line(f)
        if head == "":
            skipped.append((f, "whitespace/BOM-only"))
            continue

        # 2) If the first non-empty line has no delimiter, it's just a lone 'title'
        #    (e.g., 'time' without comma/tab); treat as header-only -> skip
        if ("," not in head) and ("\t" not in head):
            skipped.append((f, "no delimiter (single token header / title-only)"))
            continue

        try:
            # Let pandas sniff delimiter (comma vs tab)
            df = pd.read_csv(f, sep=None, engine="python")

            # 3) Skip true header-only files (columns exist, but zero data rows)
            if df.shape[0] == 0:
                skipped.append((f, "header-only (no data rows)"))
                continue

            valid_dfs.append(df)

        except (EmptyDataError, ParserError, UnicodeDecodeError) as e:
            skipped.append((f, f"{type(e).__name__}: {e}"))
        except Exception as e:
            skipped.append((f, f"Unexpected: {type(e).__name__}: {e}"))

    if not valid_dfs:
        print(f"Nothing to merge in: {tracked_folder}")
        if skipped:
            for f,e in skipped[:10]:
                print(f"  skipped: {f} -> {e}")
        continue

    df_all = pd.concat(valid_dfs, ignore_index=True)
    merged_path = os.path.join(tracked_folder, "all_cells_time_series.csv")
    df_all.to_csv(merged_path, index=False)

    print(f"Merged {len(valid_dfs)}/{len(csv_files)} files into: {merged_path}")
    if skipped:
        print(f"  Skipped {len(skipped)} files (examples):")
        for f,e in skipped[:5]:
            print(f"    {f} -> {e}")
