#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_cell_data_M156.py

Merges per-cell cell_*_data.csv files into all_cells_time_series.csv for each
M156 film. Adapted from merge_cell_data.py (same logic, different root_dir).
Needed as a prerequisite for the M156 unaligned-pairs feature pipeline
(quantify_M156.py), which reads all_cells_time_series.csv via
multi_field.find_timeseries_csv().

Prints a progress heartbeat while listing/reading per-cell CSVs in each
TrackedCells_ folder, since X10 Pro has shown very high per-file I/O latency
(observed ~7-8 files/sec even from native Mac Terminal) -- without this, a
folder with a few hundred/thousand per-cell CSVs can look hung for minutes
between the original script's print() calls, which only fired once per whole
film.
"""
import os, glob, time, pandas as pd
from pandas.errors import EmptyDataError, ParserError

root_dir = "/Volumes/X10 Pro/Movies/2026_07_16_M156/"

HEARTBEAT_EVERY_FILES = 25
HEARTBEAT_EVERY_SECS = 3.0


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
    print(f"Listing subfolders under: {root_dir}", flush=True)
    subfolders = sorted(os.listdir(root_dir))
    print(f"Found {len(subfolders)} entries. Scanning for TrackedCells_ folders...\n", flush=True)

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        tracked_folder = os.path.join(subfolder_path, f"TrackedCells_{subfolder}")
        if not os.path.isdir(tracked_folder):
            continue

        print(f"=== {subfolder} ===", flush=True)
        print(f"  Globbing cell_*_data.csv in {tracked_folder} ...", flush=True)
        csv_files = glob.glob(os.path.join(tracked_folder, "cell_*_data.csv"))
        # Exclude any accidental matches on masks/backup files
        csv_files = [f for f in csv_files if f.endswith("_data.csv")]
        if not csv_files:
            print(f"  No CSV files found in: {tracked_folder}\n", flush=True)
            continue

        print(f"  Found {len(csv_files)} per-cell CSVs. Reading...", flush=True)
        valid_dfs, skipped = [], []
        start = time.monotonic()
        last_print = start

        for i, f in enumerate(csv_files, 1):
            head = first_nonempty_line(f)
            if head == "":
                skipped.append((f, "whitespace/BOM-only"))
            elif ("," not in head) and ("\t" not in head):
                skipped.append((f, "no delimiter (single token header / title-only)"))
            else:
                try:
                    df = pd.read_csv(f, sep=None, engine="python")
                    if df.shape[0] == 0:
                        skipped.append((f, "header-only (no data rows)"))
                    else:
                        valid_dfs.append(df)
                except (EmptyDataError, ParserError, UnicodeDecodeError) as e:
                    skipped.append((f, f"{type(e).__name__}: {e}"))
                except Exception as e:
                    skipped.append((f, f"Unexpected: {type(e).__name__}: {e}"))

            now = time.monotonic()
            if i % HEARTBEAT_EVERY_FILES == 0 or (now - last_print) >= HEARTBEAT_EVERY_SECS:
                print(f"    ...read {i}/{len(csv_files)} files, {now - start:.0f}s elapsed", flush=True)
                last_print = now

        print(f"  Done reading: {len(valid_dfs)} valid, {len(skipped)} skipped, "
              f"{time.monotonic() - start:.1f}s", flush=True)

        if not valid_dfs:
            print(f"  Nothing to merge in: {tracked_folder}", flush=True)
            if skipped:
                for f, e in skipped[:10]:
                    print(f"    skipped: {f} -> {e}", flush=True)
            print(flush=True)
            continue

        print(f"  Concatenating and writing merged CSV...", flush=True)
        df_all = pd.concat(valid_dfs, ignore_index=True)
        merged_path = os.path.join(tracked_folder, "all_cells_time_series.csv")
        df_all.to_csv(merged_path, index=False)

        print(f"  Merged {len(valid_dfs)}/{len(csv_files)} files into: {merged_path}", flush=True)
        if skipped:
            print(f"  Skipped {len(skipped)} files (examples):", flush=True)
            for f, e in skipped[:5]:
                print(f"    {f} -> {e}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
