#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantify_M156.py

M156 analog of quantify_M133.py, generalized from 2 GFP sessions (GFP1/GFP2)
to M156's 6 GFP (FL) sessions per field, interleaved with 5 short BF
relinking snapshots: FL1-BF1-FL2-BF2-FL3-BF3-FL4-BF4-FL5-BF5-FL6.

Unlike M133/Sept17, M156 does NOT need run_field_sequence /
build_global_id_maps_from_pairings run from scratch: the cross-film cell
identity chain has already been computed and stored in
<M156_ROOT>/sequence_linkage.json (same format used by Sept17), with one
entry per field:
    linkage[field]["films"]        -> ordered list of 11 film names
    linkage[field]["global_cells"] -> {global_cell_id: [local_id_per_film, ...]}
                                       (-1 where that film has no match)

Each (global_cell, FL_k) pair with a valid local id is treated as its own
independent ~101-frame observation -- exactly how M133/Sept17 treat GFP1 and
GFP2 as separate observations of the same lineage, not one merged trace.
This also matches the biology: each FL burst is its own short continuous GFP
movie, and BF sessions exist only to re-anchor identity in between, not to
carry fluorescence signal (BF per-cell CSVs have no pol/cyt/septum columns).

Outputs (into <M156_ROOT>/unaligned_pairs_quant/, matching the M133 schema
so the existing downstream tooling -- PCA_utils.load_experiment_features,
FC_AE_data_loader, manifold_explorer -- can consume it unmodified):
    stacked_gfp1_gfp2_for_unaligned_pairs.csv   (legacy name kept for compat)
    id_map_unaligned.csv
    model_fits_by_cell.csv
    acor_detrended_results.csv
"""
import os
import sys
import json
import re
import time
import numpy as np
import pandas as pd

for _base in (
    '/Users/user/Documents/Python_Scripts/FungalProjectScript/',
    '/sessions/sweet-magical-cray/mnt/FungalProjectScript/',
):
    if os.path.isdir(_base):
        sys.path.append(_base)
from SingleCellDataAnalysis.signal_analysis import quantify_all_cells
from SingleCellDataAnalysis.signal_cor import quantify_all_cells_acor

M156_ROOT = "/Volumes/X10 Pro/Movies/2026_07_16_M156/"
SANDBOX_ROOT = "/sessions/sweet-magical-cray/mnt/X10 Pro/Movies/2026_07_16_M156/"
ROOT = SANDBOX_ROOT if os.path.isdir(SANDBOX_ROOT) else M156_ROOT

FIELDS = ["3_F0", "3_F1", "3_F2"]
FL_CHAIN_INDICES = [0, 2, 4, 6, 8, 10]   # positions of FL1..FL6 in the 11-film chain
FL_SOURCE_LABELS = ["FL1", "FL2", "FL3", "FL4", "FL5", "FL6"]
MIN_VALID_FRAMES = 5   # quantify_all_cells/_acor both require >= 5


def load_linkage(root):
    path = os.path.join(root, "sequence_linkage.json")
    with open(path) as f:
        return json.load(f)


def load_cell_trace(root, film, local_id):
    """Load one cell's per-timepoint trace from a single FL film, restricted
    to base rows only (drops derived sub-pattern rows like '100_1')."""
    csv_path = os.path.join(root, film, f"TrackedCells_{film}", f"cell_{local_id}_data.csv")
    if not os.path.isfile(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if df.empty or "cell_id" not in df.columns or "time_point" not in df.columns:
        return None
    df = df[df["cell_id"].astype(str) == str(local_id)].copy()
    needed = {"pol1_int", "pol2_int", "cyt_int", "septum_int"}
    if df.empty or not needed.issubset(df.columns):
        return None
    return df


def main():
    print(f"Using root: {ROOT}")
    linkage = load_linkage(ROOT)

    stack_rows = []
    map_rows = []
    new_id = 1
    n_skipped_short = 0
    n_skipped_missing = 0

    for field in FIELDS:
        seq = linkage.get(field)
        if not seq:
            print(f"  [skip] no linkage entry for field {field}")
            continue

        films = seq["films"]
        global_cells = seq["global_cells"]
        print(f"=== {field}: {len(global_cells)} global cells, chain: {films} ===")

        for gcid, local_ids in global_cells.items():
            for chain_idx, source in zip(FL_CHAIN_INDICES, FL_SOURCE_LABELS):
                local_id = local_ids[chain_idx]
                if local_id is None or local_id == -1:
                    continue
                film = films[chain_idx]

                df = load_cell_trace(ROOT, film, local_id)
                if df is None:
                    n_skipped_missing += 1
                    continue

                df = df.sort_values("time_point").copy()
                df["pol1_int_corr"] = df["pol1_int"] - df["cyt_int"]
                df["pol2_int_corr"] = df["pol2_int"] - df["cyt_int"]
                df["septum_int_corr"] = df["septum_int"] - df["cyt_int"]

                valid = df[["pol1_int_corr", "pol2_int_corr"]].notna().all(axis=1)
                if valid.sum() < MIN_VALID_FRAMES:
                    n_skipped_short += 1
                    continue

                out = df[["time_point", "pol1_int_corr", "pol2_int_corr", "septum_int_corr"]].copy()
                out["cell_id"] = new_id
                out["source"] = source
                out["field"] = field

                stack_rows.append(out)
                map_rows.append({
                    "new_cell_id": new_id,
                    "orig_str_id": f"{film}:{local_id}",
                    "field": field.replace("3_", ""),   # -> "F0"/"F1"/"F2", matches GenericAdapter convention
                    "source": source,
                    "global_cell_id": gcid,
                    "local_fl_id": local_id,
                    "film": film,
                })
                new_id += 1

        print(f"  ...running total observations so far: {new_id - 1} "
              f"(skipped {n_skipped_missing} missing/empty, {n_skipped_short} too-short)")

    if not stack_rows:
        print("No valid traces found for M156 -- nothing to quantify.")
        return

    df_stacked = pd.concat(stack_rows, ignore_index=True)
    df_map = pd.DataFrame(map_rows)

    out_dir = os.path.join(ROOT, "unaligned_pairs_quant")
    os.makedirs(out_dir, exist_ok=True)

    data_csv = os.path.join(out_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
    map_csv = os.path.join(out_dir, "id_map_unaligned.csv")

    df_stacked.to_csv(data_csv, index=False)
    df_map.to_csv(map_csv, index=False)

    print(f"\nSaved stacked data to: {data_csv} ({len(df_stacked)} rows)")
    print(f"Saved id map to: {map_csv} ({len(df_map)} observations)")
    print(f"Total skipped: {n_skipped_missing} missing/empty, {n_skipped_short} too-short (<{MIN_VALID_FRAMES} valid frames)")

    cell_ids = df_map["new_cell_id"].unique().tolist()

    CHUNK_SIZE = 150
    chunks = [cell_ids[i:i + CHUNK_SIZE] for i in range(0, len(cell_ids), CHUNK_SIZE)]

    def run_chunked(func, label, chunk_subdir, final_csv, **kwargs):
        """Runs `func` over cell_ids in chunks, one temp CSV per chunk. Chunks
        already on disk from a prior (interrupted) run are skipped, so this is
        safe to just re-invoke after a timeout/Ctrl+C -- it picks up where it
        left off instead of redoing completed work."""
        chunk_dir = os.path.join(out_dir, chunk_subdir)
        os.makedirs(chunk_dir, exist_ok=True)
        t0 = time.time()
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.csv")
            if os.path.exists(chunk_path):
                print(f"  [{label}] chunk {i+1}/{len(chunks)} already done, skipping")
                continue
            func(df_stacked, chunk, filename=chunk_path, **kwargs)
            print(f"  [{label}] chunk {i+1}/{len(chunks)} done "
                  f"({(i+1)*CHUNK_SIZE} / {len(cell_ids)} cells, {time.time()-t0:.0f}s elapsed)",
                  flush=True)

        parts = [pd.read_csv(os.path.join(chunk_dir, f"chunk_{i:04d}.csv")) for i in range(len(chunks))]
        df_final = pd.concat(parts, ignore_index=True)
        df_final.to_csv(final_csv, index=False)
        print(f"  [{label}] wrote combined result: {final_csv} ({len(df_final)} rows)")
        return df_final

    print("\nRunning standard quantification (model fits)...")
    fits_csv = os.path.join(out_dir, "model_fits_by_cell.csv")
    run_chunked(
        lambda df, ids, filename: quantify_all_cells(
            df, ids, feature1="pol1_int_corr", feature2="pol2_int_corr",
            delta_threshold=10, filename=filename,
        ),
        "fits", "_fits_chunks", fits_csv,
    )

    print("\nRunning autocorrelation quantification...")
    acor_csv = os.path.join(out_dir, "acor_detrended_results.csv")
    run_chunked(
        lambda df, ids, filename: quantify_all_cells_acor(
            df, ids, delta_threshold=10, visualize=False, filename=filename,
        ),
        "acor", "_acor_chunks", acor_csv,
    )

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
