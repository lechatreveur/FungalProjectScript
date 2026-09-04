#!/usr/bin/env python3
"""
Retrack M160 Sequences.
Tracks cells within each film across all frames and links tracks across consecutive films in the sequence.
"""

import os
import re
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment

HPC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from ground_truth_corrector.schemas import encode_mask_to_rle, validate_and_decode_rle

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_08_28_M160"

def track_film(exp_dir: Path, film: str, min_area: int = 2000, min_iou: float = 0.15) -> Dict[int, Dict[int, str]]:
    """
    Track cells in a single film across all frames using precomputed instance masks.
    Returns {cell_id: {time_point: rle_str}}.
    """
    film_dir = exp_dir / film
    masks_dir = film_dir / f"Masks_{film}"
    tracked_dir = film_dir / f"TrackedCells_{film}"
    tracked_dir.mkdir(parents=True, exist_ok=True)

    seg_files = sorted([f for f in masks_dir.glob("*_seg.tif") if not f.name.startswith(".")])
    if not seg_files:
        print(f"  [Warning] No seg files in {masks_dir}")
        return {}

    # Map timepoint to file
    time_map = {}
    for f in seg_files:
        m = re.search(r"_t_(\d+)_", f.name) or re.search(r"_t(\d+)_", f.name)
        if m:
            time_map[int(m.group(1))] = f

    sorted_times = sorted(time_map.keys())
    if not sorted_times:
        return {}

    # 1. Initialize from t=0
    t0 = sorted_times[0]
    seg0 = tifffile.imread(str(time_map[t0]))
    H, W = seg0.shape[:2]

    props0 = regionprops(seg0)
    initial_labels = [r.label for r in props0 if r.area >= min_area]

    # Map tracking state: cell_id -> {t: rle}
    cell_tracks: Dict[int, Dict[int, str]] = {}
    current_label_map: Dict[int, int] = {} # cell_id -> label in current seg

    for cid in initial_labels:
        cell_mask = (seg0 == cid).astype(np.uint8)
        rle = encode_mask_to_rle(cell_mask)
        cell_tracks[cid] = {t0: rle}
        current_label_map[cid] = cid

    prev_seg = seg0

    # 2. Track forward
    for t in sorted_times[1:]:
        seg_cur = tifffile.imread(str(time_map[t]))
        
        # Fast overlap computation
        mask_overlap = (prev_seg > 0) & (seg_cur > 0)
        if mask_overlap.any():
            pairs, counts = np.unique(np.column_stack((prev_seg[mask_overlap], seg_cur[mask_overlap])), axis=0, return_counts=True)
            # Find best match for each active cell
            match_candidates = {} # prev_lbl -> (cur_lbl, overlap_count)
            for (p_lbl, c_lbl), count in zip(pairs, counts):
                if p_lbl not in match_candidates or count > match_candidates[p_lbl][1]:
                    match_candidates[p_lbl] = (c_lbl, count)
        else:
            match_candidates = {}

        # Areas in cur seg
        cur_props = {r.label: r.area for r in regionprops(seg_cur)}
        prev_props = {r.label: r.area for r in regionprops(prev_seg)}

        new_label_map = {}
        for cid, prev_lbl in current_label_map.items():
            if prev_lbl in match_candidates:
                c_lbl, inter_count = match_candidates[prev_lbl]
                a_prev = prev_props.get(prev_lbl, 1)
                a_cur = cur_props.get(c_lbl, 1)
                iou = inter_count / float(a_prev + a_cur - inter_count)
                if iou >= min_iou:
                    cell_mask = (seg_cur == c_lbl).astype(np.uint8)
                    cell_tracks[cid][t] = encode_mask_to_rle(cell_mask)
                    new_label_map[cid] = c_lbl
                else:
                    cell_tracks[cid][t] = ""
            else:
                cell_tracks[cid][t] = ""

        current_label_map = new_label_map
        prev_seg = seg_cur

    # 3. Write CSVs
    is_fl = "FL" in film
    rle_col = "rle_gfp" if is_fl else "rle_bf"
    alt_col = "rle_bf" if is_fl else "rle_gfp"

    for cid, t_dict in cell_tracks.items():
        rows = []
        for t in sorted_times:
            rle_val = t_dict.get(t, "")
            row = {
                "time_point": t,
                "width": W,
                "height": H,
                rle_col: rle_val,
                alt_col: ""
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = tracked_dir / f"cell_{cid}_masks.csv"
        df.to_csv(csv_path, index=False)

    print(f"  Film {film}: tracked {len(cell_tracks)} cells across {len(sorted_times)} frames.")
    return cell_tracks


def link_sequence_films(exp_dir: Path, sequence: str, films: List[str]) -> Dict[str, Any]:
    """
    Compute global tracks across films in a sequence using IoU matching between consecutive films.
    """
    print(f"\n--- Linking sequence {sequence} ({len(films)} films) ---")
    
    # Load cell tables for each film
    film_cells: Dict[str, Dict[int, pd.DataFrame]] = {}
    for f in films:
        t_dir = exp_dir / f / f"TrackedCells_{f}"
        film_cells[f] = {}
        for csv_f in t_dir.glob("cell_*_masks.csv"):
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_f.name)
            if m:
                cid = int(m.group(1))
                try:
                    df = pd.read_csv(csv_f)
                    film_cells[f][cid] = df
                except Exception:
                    pass

    # Initialize global tracks from film 0
    f0 = films[0]
    c0_ids = sorted(film_cells[f0].keys())
    global_cells: Dict[str, List[int]] = {}
    for cid in c0_ids:
        gid = f"{sequence}_cell_{cid}"
        global_cells[gid] = [cid]

    # Map across pairs
    for i in range(len(films) - 1):
        fA = films[i]
        fB = films[i + 1]
        
        cellsA = film_cells[fA]
        cellsB = film_cells[fB]
        cB_ids = sorted(cellsB.keys())

        # Build mask for each cell at t_last of fA
        rle_col_A = "rle_gfp" if "FL" in fA else "rle_bf"
        rle_col_B = "rle_gfp" if "FL" in fB else "rle_bf"

        masksA = {}
        for cid, df in cellsA.items():
            last_row = df.iloc[-1]
            rle = str(last_row.get(rle_col_A, ""))
            if rle and rle != "nan":
                m = validate_and_decode_rle(rle, int(last_row["height"]), int(last_row["width"]))
                if m.any():
                    masksA[cid] = m

        masksB = {}
        for cid, df in cellsB.items():
            first_row = df.iloc[0]
            rle = str(first_row.get(rle_col_B, ""))
            if rle and rle != "nan":
                m = validate_and_decode_rle(rle, int(first_row["height"]), int(first_row["width"]))
                if m.any():
                    masksB[cid] = m

        # Match using IoU
        mapping: Dict[int, int] = {}
        listA = list(masksA.keys())
        listB = list(masksB.keys())

        if listA and listB:
            cost_matrix = np.ones((len(listA), len(listB)), dtype=np.float32)
            for idxA, cA in enumerate(listA):
                mA = masksA[cA]
                areaA = mA.sum()
                for idxB, cB in enumerate(listB):
                    mB = masksB[cB]
                    areaB = mB.sum()
                    inter = (mA & mB).sum()
                    if inter > 0:
                        iou = inter / float(areaA + areaB - inter)
                        cost_matrix[idxA, idxB] = 1.0 - iou

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                iou = 1.0 - cost_matrix[r, c]
                if iou >= 0.05:  # Valid link
                    mapping[listA[r]] = listB[c]

        print(f"  Linked {fA} -> {fB}: {len(mapping)}/{len(listA)} cells matched.")

        # Update existing tracks
        mapped_B = set()
        for gid, track in list(global_cells.items()):
            last_cid = track[-1]
            if last_cid != -1 and last_cid in mapping:
                next_cid = mapping[last_cid]
                global_cells[gid].append(next_cid)
                mapped_B.add(next_cid)
            else:
                global_cells[gid].append(-1)

        # Add new cells appearing in fB
        for cB in cB_ids:
            if cB not in mapped_B:
                gid = f"{sequence}_{fB}_cell_{cB}"
                track = [-1] * (i + 1) + [cB]
                global_cells[gid] = track

    print(f"  Sequence {sequence} created {len(global_cells)} global cell tracks.")
    return {
        "films": films,
        "global_cells": global_cells,
        "lineage": {}
    }


def main():
    exp_dir = MOVIE_ROOT / EXP
    seq_file = exp_dir / "sequence_linkage.json"
    
    with open(seq_file) as f:
        seq_config = json.load(f)

    # 1. Track all films in the sequences
    all_films = set()
    for seq_name, seq_data in seq_config.items():
        for film in seq_data.get("films", []):
            all_films.add(film)

    print(f"Starting intra-film tracking for {len(all_films)} films in {EXP}...")
    t_start = time.time()
    
    # Process sequentially or with threads
    for idx, film in enumerate(sorted(all_films), 1):
        print(f"[{idx}/{len(all_films)}] Tracking film {film}...")
        track_film(exp_dir, film)

    # 2. Compute inter-film sequence linkages
    updated_seq_data = {}
    for seq_name, seq_data in seq_config.items():
        films = seq_data.get("films", [])
        updated_seq_data[seq_name] = link_sequence_films(exp_dir, seq_name, films)

    # 3. Save sequence_linkage.json
    with open(seq_file, "w") as f:
        json.dump(updated_seq_data, f, indent=2)

    print(f"\n✅ All sequence cells retracked and linked in {time.time() - t_start:.1f}s.")
    print(f"Updated {seq_file}")

if __name__ == "__main__":
    main()
