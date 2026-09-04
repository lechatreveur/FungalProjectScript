#!/usr/bin/env python3
"""
Fast sequence linkage generator for M160.
Links pre-tracked cells across consecutive films using vectorized IoU matching.
"""

import os
import re
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HPC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from ground_truth_corrector.schemas import validate_and_decode_rle

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_08_28_M160"

def get_film_keyframe_files(film_dir: Path, film: str) -> Tuple[List[int], Dict[int, Path]]:
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


def fast_link_sequence(exp_dir: Path, sequence: str, films: List[str]) -> Dict[str, Any]:
    print(f"\n--- Fast Linking sequence {sequence} ({len(films)} films) ---")

    # 1. Collect first and last keyframe masks for each cell in each film
    film_first_masks = {} # film -> {cid: mask}
    film_last_masks = {}  # film -> {cid: mask}
    film_cids = {}        # film -> sorted [cids]

    for f in films:
        t_dir = exp_dir / f / f"TrackedCells_{f}"
        film_first_masks[f] = {}
        film_last_masks[f] = {}
        
        k_times, _ = get_film_keyframe_files(exp_dir / f, f)
        t_first = k_times[0] if k_times else 0
        t_last = k_times[-1] if k_times else 0
        rle_col = "rle_gfp" if "FL" in f else "rle_bf"

        cids = []
        for csv_f in t_dir.glob("cell_*_masks.csv"):
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_f.name)
            if not m: continue
            cid = int(m.group(1))
            cids.append(cid)
            try:
                df = pd.read_csv(csv_f)
                # First keyframe
                r1 = df[df["time_point"] == t_first]
                if not r1.empty:
                    s1 = str(r1.iloc[0].get(rle_col, ""))
                    if s1 and s1 != "nan":
                        mask1 = validate_and_decode_rle(s1, int(r1.iloc[0]["height"]), int(r1.iloc[0]["width"]))
                        if mask1.any():
                            film_first_masks[f][cid] = mask1

                # Last keyframe
                r2 = df[df["time_point"] == t_last]
                if not r2.empty:
                    s2 = str(r2.iloc[0].get(rle_col, ""))
                    if s2 and s2 != "nan":
                        mask2 = validate_and_decode_rle(s2, int(r2.iloc[0]["height"]), int(r2.iloc[0]["width"]))
                        if mask2.any():
                            film_last_masks[f][cid] = mask2
            except Exception:
                pass
        film_cids[f] = sorted(cids)
        print(f"  Loaded {len(cids)} cell masks from {f}")

    # 2. Build global tracks
    f0 = films[0]
    c0_ids = film_cids[f0]
    global_cells: Dict[str, List[int]] = {}
    for cid in c0_ids:
        gid = f"{sequence}_cell_{cid}"
        global_cells[gid] = [cid]

    # Map consecutively: fA (t_last) -> fB (t_first)
    for i in range(len(films) - 1):
        fA = films[i]
        fB = films[i + 1]

        masksA = film_last_masks.get(fA, {})
        masksB = film_first_masks.get(fB, {})
        cB_ids = film_cids[fB]

        mapping: Dict[int, int] = {}
        
        if masksA and masksB:
            # Vectorized fast pairwise overlap:
            # Build combined label image A and B
            H, W = 2000, 2000
            lblA = np.zeros((H, W), dtype=np.int32)
            areaA = {}
            for cid, m in masksA.items():
                lblA[m] = cid
                areaA[cid] = int(m.sum())

            lblB = np.zeros((H, W), dtype=np.int32)
            areaB = {}
            for cid, m in masksB.items():
                lblB[m] = cid
                areaB[cid] = int(m.sum())

            mask_ov = (lblA > 0) & (lblB > 0)
            if mask_ov.any():
                pairs, counts = np.unique(
                    np.column_stack((lblA[mask_ov], lblB[mask_ov])), 
                    axis=0, 
                    return_counts=True
                )
                
                # Filter by IoU >= 0.05
                # For each cidA, find best candidate in cidB
                best_matches = {} # cidA -> (cidB, iou)
                for (cA, cB), count in zip(pairs, counts):
                    a_A = areaA.get(cA, 1)
                    a_B = areaB.get(cB, 1)
                    iou = count / float(a_A + a_B - count)
                    if iou >= 0.05:
                        if cA not in best_matches or iou > best_matches[cA][1]:
                            best_matches[cA] = (cB, iou)

                # Ensure 1-to-1 assignment (pick highest IoU if conflict)
                b_used = {}
                for cA, (cB, iou) in sorted(best_matches.items(), key=lambda x: x[1][1], reverse=True):
                    if cB not in b_used:
                        mapping[cA] = cB
                        b_used[cB] = cA

        print(f"  Linked {fA} -> {fB}: {len(mapping)}/{len(masksA)} cells linked.")

        # Update existing global tracks
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

    t0 = time.time()
    updated_seq_data = {}
    for seq_name, seq_data in seq_config.items():
        films = seq_data.get("films", [])
        updated_seq_data[seq_name] = fast_link_sequence(exp_dir, seq_name, films)

    with open(seq_file, "w") as f:
        json.dump(updated_seq_data, f, indent=2)

    print(f"\n🎉 All sequence linkages created and saved in {time.time() - t0:.2f}s!")
    print(f"Saved to {seq_file}")

if __name__ == "__main__":
    main()
