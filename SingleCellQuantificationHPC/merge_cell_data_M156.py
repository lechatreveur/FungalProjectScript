#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_cell_data_M156.py

Merges per-cell cell_*_data.csv files into all_cells_time_series.csv for each
M156 film. Adapted from merge_cell_data.py (same logic, different root_dir).
Needed as a prerequisite for the M156 unaligned-pairs feature pipeline
(quantify_M156.py), which reads all_cells_time_series.csv via
multi_field.find_timeseries_csv().
"""
import os, glob, pandas as pd
from pandas.errors import EmptyDataError, ParserError

root_dir = "/Volumes/X10 Pro/Movies/2026_07_16_M156/"


def first_nonempty_line(path, max_lines=5):
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


def main():
    for subfolder in sorted(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        tracked_folder = os.path.join(subfolder_path, f"TrackedCells_{subfolder}")
        if not os.path.isdir(tracked_folder):
            continue

        csv_files = glob.glob(os.path.join(tracked_folder, "cell_*_data.csv"))
        # Exclude any accidental matches on masks/backup files
        csv_files = [f for f in csv_files if f.endswith("_data.csv")]
        if not csv_files:
            print(f"No CSV files found in: {tracked_folder}")
            continue

        valid_dfs, skipped = [], []

        for f in csv_files:
            head = first_nonempty_line(f)
            if head == "":
                skipped.append((f, "whitespace/BOM-only"))
                continue

            if ("," not in head) and ("\t" not in head):
                skipped.append((f, "no delimiter (single token header / title-only)"))
                continue

            try:
                df = pd.read_csv(f, sep=None, engine="python")
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
                for f, e in skipped[:10]:
                    print(f"  skipped: {f} -> {e}")
            continue

        df_all = pd.concat(valid_dfs, ignore_index=True)
        merged_path = os.path.join(tracked_folder, "all_cells_time_series.csv")
        df_all.to_csv(merged_path, index=False)

        print(f"Merged {len(valid_dfs)}/{len(csv_files)} files into: {merged_path}")
        if skipped:
            print(f"  Skipped {len(skipped)} files (examples):")
            for f, e in skipped[:5]:
                print(f"    {f} -> {e}")


if __name__ == "__main__":
    main()
