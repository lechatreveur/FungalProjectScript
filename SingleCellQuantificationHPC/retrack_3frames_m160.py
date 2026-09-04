#!/usr/bin/env python3
"""
Fast 3-Keyframe Tracker and Sequence Linker for M160.
Tracks cells across only the 3 keyframes (first, middle, last) for each film,
then links global cell tracks across consecutive films in each sequence.
"""

import os
import re
import sys
import time
import json
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

def get_film_keyframe_files(film_dir: Path, film: str) -> Tuple[List[int], Dict[int, Path]]:
    """Return (sorted_keyframes, {t: path}) for a film."""
    masks_dir = film_dir / f"Masks_{film}"
    seg_files = sorted([f for f in masks_dir.glob("*_seg.tif") if not f.name.startswith(".")])
    
    time_map = {}
    for f in seg_files:
        m = re.search(r"_t_(\d+)_", f.name) or re.search(r"_t(\d+)_", f.name)
        if m:
            time_map[int(m.group(1))] = f

    all_times = sorted(time_map.keys())
    if not all_times:
        return [], {}

    n = len(all_times)
    if n <= 3:
        k_times = all_times
    else:
        k_times = [all_times[0], all_times[(n - 1) // 2], all_times[-1]]
    
    return k_times, time_map


def track_film_3keyframes(exp_dir: Path, film: str, min_area: int = 2000, min_iou: float = 0.15) -> Dict[int, Dict[int, str]]:
    """
    Track cells in a film across only its 3 keyframes (first, middle, last).
    Writes cell_*_masks.csv into TrackedCells_<film>/.
    Returns {cell_id: {t: rle_str}}.
    """
    film_dir = exp_dir / film
    tracked_dir = film_dir / f"TrackedCells_{film}"
    tracked_dir.mkdir(parents=True, exist_ok=True)

    k_times, time_map = get_film_keyframe_files(film_dir, film)
    if not k_times:
        print(f"  [Warning] No keyframes found for {film}")
        return {}

    all_times = sorted(time_map.keys())
    t0 = k_times[0]
    seg0 = tifffile.imread(str(time_map[t0]))
    H, W = seg0.shape[:2]

    # 1. Seed cell labels at t0
    props0 = regionprops(seg0)
    initial_labels = [r.label for r in props0 if r.area >= min_area]

    cell_tracks: Dict[int, Dict[int, str]] = {}
    current_label_map: Dict[int, int] = {}

    for cid in initial_labels:
        cell_mask = (seg0 == cid).astype(np.uint8)
        rle = encode_mask_to_rle(cell_mask)
        cell_tracks[cid] = {t0: rle}
        current_label_map[cid] = cid

    prev_seg = seg0

    # 2. Track across the remaining keyframes
    for t in k_times[1:]:
        seg_cur = tifffile.imread(str(time_map[t]))
        
        # Overlap computation
        mask_overlap = (prev_seg > 0) & (seg_cur > 0)
        match_candidates = {}
        if mask_overlap.any():
            pairs, counts = np.unique(np.column_stack((prev_seg[mask_overlap], seg_cur[mask_overlap])), axis=0, return_counts=True)
            for (p_lbl, c_lbl), count in zip(pairs, counts):
                if p_lbl not in match_candidates or count > match_candidates[p_lbl][1]:
                    match_candidates[p_lbl] = (c_lbl, count)

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

    # 3. Save cell_*_masks.csv
    is_fl = "FL" in film
    rle_col = "rle_gfp" if is_fl else "rle_bf"
    alt_col = "rle_bf" if is_fl else "rle_gfp"

    for cid, t_dict in cell_tracks.items():
        rows = []
        for t in all_times:
            # Populate keyframe masks; other timepoints remain empty until tracked on HPC
            rle_val = t_dict.get(t, "")
            rows.append({
                "time_point": t,
                "width": W,
                "height": H,
                rle_col: rle_val,
                alt_col: ""
            })
        df = pd.DataFrame(rows)
        csv_path = tracked_dir / f"cell_{cid}_masks.csv"
        df.to_csv(csv_path, index=False)

    print(f"  Film {film}: tracked {len(cell_tracks)} cells across keyframes {k_times}.")
    return cell_tracks


def link_sequence_3keyframes(exp_dir: Path, sequence: str, films: List[str]) -> Dict[str, Any]:
    """
    Link cells across consecutive films in sequence using keyframe matching.
    """
    print(f"\n--- Linking sequence {sequence} ({len(films)} films) ---")

    # Load masks for all cells in each film
    film_cells: Dict[str, Dict[int, pd.DataFrame]] = {}
    film_last_masks: Dict[str, Dict[int, np.ndarray]] = {}
    film_first_masks: Dict[str, Dict[int, np.ndarray]] = {}

    for f in films:
        t_dir = exp_dir / f / f"TrackedCells_{f}"
        film_cells[f] = {}
        film_last_masks[f] = {}
        film_first_masks[f] = {}

        k_times, _ = get_film_keyframe_files(exp_dir / f, f)
        t_first = k_times[0] if k_times else 0
        t_last = k_times[-1] if k_times else 0
        rle_col = "rle_gfp" if "FL" in f else "rle_bf"

        for csv_f in t_dir.glob("cell_*_masks.csv"):
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_f.name)
            if not m: continue
            cid = int(m.group(1))
            try:
                df = pd.read_csv(csv_f)
                film_cells[f][cid] = df
                
                # First keyframe mask
                r_first = df[df["time_point"] == t_first]
                if not r_first.empty:
                    rle1 = str(r_first.iloc[0].get(rle_col, ""))
                    if rle1 and rle1 != "nan":
                        mask1 = validate_and_decode_rle(rle1, int(r_first.iloc[0]["height"]), int(r_first.iloc[0]["width"]))
                        if mask1.any():
                            film_first_masks[f][cid] = mask1

                # Last keyframe mask
                r_last = df[df["time_point"] == t_last]
                if not r_last.empty:
                    rle2 = str(r_last.iloc[0].get(rle_col, ""))
                    if rle2 and rle2 != "nan":
                        mask2 = validate_and_decode_rle(rle2, int(r_last.iloc[0]["height"]), int(r_last.iloc[0]["width"]))
                        if mask2.any():
                            film_last_masks[f][cid] = mask2
            except Exception:
                pass

    # Initialize global tracks from film 0
    f0 = films[0]
    c0_ids = sorted(film_cells[f0].keys())
    global_cells: Dict[str, List[int]] = {}
    for cid in c0_ids:
        gid = f"{sequence}_cell_{cid}"
        global_cells[gid] = [cid]

    # Map across film pairs: fA (t_last) -> fB (t_first)
    for i in range(len(films) - 1):
        fA = films[i]
        fB = films[i + 1]

        masksA = film_last_masks.get(fA, {})
        masksB = film_first_masks.get(fB, {})
        cB_ids = sorted(film_cells[fB].keys())

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
                if iou >= 0.05:  # Overlap between end of fA and start of fB
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

        # Add new cells starting at fB
        for cB in cB_ids:
            if cB not in mapped_B:
                gid = f"{sequence}_{fB}_cell_{cB}"
                track = [-1] * (i + 1) + [cB]
                global_cells[gid] = track

    print(f"  Sequence {sequence}: generated {len(global_cells)} global cell tracks.")
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

    all_films = []
    for seq_name, seq_data in seq_config.items():
        for film in seq_data.get("films", []):
            if film not in all_films:
                all_films.append(film)

    print(f"Starting 3-keyframe tracking for {len(all_films)} films in {EXP}...")
    t0 = time.time()

    for idx, film in enumerate(all_films, 1):
        print(f"[{idx}/{len(all_films)}] Tracking film {film} (3 keyframes)...")
        track_film_3keyframes(exp_dir, film)

    # Compute inter-film sequence linkages
    updated_seq_data = {}
    for seq_name, seq_data in seq_config.items():
        films = seq_data.get("films", [])
        updated_seq_data[seq_name] = link_sequence_3keyframes(exp_dir, seq_name, films)

    # Save sequence_linkage.json
    with open(seq_file, "w") as f:
        json.dump(updated_seq_data, f, indent=2)

    print(f"\n🎉 Done in {time.time() - t0:.1f}s! Updated {seq_file}")

if __name__ == "__main__":
    main()
