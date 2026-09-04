#!/usr/bin/env python3
"""
Backward Sequence Tracker with Shared-Mother Cell Architecture for M160.
Tracks cells backward from the terminal film (Film 12) to the initial film (Film 0).
Merges daughter cell tracks onto shared mother cells when Area(Mother) >= 1.4x Area(Daughter)
(up to 2 daughters per mother), preserving all existing 13/13 complete and curated tracks.
"""

import os
import re
import sys
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HPC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from ground_truth_corrector.schemas import validate_and_decode_rle, encode_mask_to_rle

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_08_28_M160"


def fast_rle_area_and_centroid(rle_str: str, H: int = 2000, W: int = 2000) -> Tuple[int, Tuple[float, float]]:
    """Fast analytical calculation of mask area and centroid from 1-indexed Fortran RLE."""
    if not isinstance(rle_str, str) or not rle_str.strip() or rle_str == "nan":
        return 0, (0.0, 0.0)
    parts = rle_str.strip().split()
    if len(parts) < 2:
        return 0, (0.0, 0.0)
    
    total_area = 0
    sum_x, sum_y = 0.0, 0.0
    for i in range(0, len(parts), 2):
        s = int(parts[i]) - 1
        L = int(parts[i + 1])
        total_area += L
        start_col = s // H
        end_col = (s + L - 1) // H
        if start_col == end_col:
            c = start_col
            start_row = s % H
            sum_x += L * c
            sum_y += L * start_row + (L * (L - 1)) / 2.0
        else:
            for p in range(s, s + L):
                sum_x += p // H
                sum_y += p % H
                
    if total_area == 0:
        return 0, (0.0, 0.0)
    return total_area, (sum_x / total_area, sum_y / total_area)


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


def load_film_cell_data(exp_dir: Path, film: str) -> Dict[str, Any]:
    """Load cell masks and precompute keyframe decoded masks and areas for a film."""
    t_dir = exp_dir / film / f"TrackedCells_{film}"
    k_times, _ = get_film_keyframe_files(exp_dir / film, film)
    t_first = k_times[0] if k_times else 0
    t_last = k_times[-1] if k_times else 0
    rle_col = "rle_gfp" if "FL" in film else "rle_bf"

    cells_data = {}
    first_masks = {}
    last_masks = {}
    first_props = {}
    last_props = {}

    csv_files = [p for p in t_dir.glob("cell_*_masks.csv") if not p.name.startswith(".")]
    for csv_f in csv_files:
        m = re.match(r"^cell_(\d+)_masks\.csv$", csv_f.name)
        if not m: continue
        cid = int(m.group(1))
        try:
            df = pd.read_csv(csv_f)
            cells_data[cid] = df

            # First keyframe
            r_first = df[df["time_point"] == t_first]
            if not r_first.empty:
                rle1 = str(r_first.iloc[0].get(rle_col, ""))
                if rle1 and rle1 != "nan":
                    a1, c1 = fast_rle_area_and_centroid(rle1)
                    if a1 > 0:
                        first_props[cid] = {"area": a1, "centroid": c1, "rle": rle1}
                        m1 = validate_and_decode_rle(rle1, 2000, 2000)
                        if m1.any():
                            first_masks[cid] = m1

            # Last keyframe
            r_last = df[df["time_point"] == t_last]
            if not r_last.empty:
                rle2 = str(r_last.iloc[0].get(rle_col, ""))
                if rle2 and rle2 != "nan":
                    a2, c2 = fast_rle_area_and_centroid(rle2)
                    if a2 > 0:
                        last_props[cid] = {"area": a2, "centroid": c2, "rle": rle2}
                        m2 = validate_and_decode_rle(rle2, 2000, 2000)
                        if m2.any():
                            last_masks[cid] = m2
        except Exception:
            pass

    return {
        "film": film,
        "cells": cells_data,
        "first_masks": first_masks,
        "last_masks": last_masks,
        "first_props": first_props,
        "last_props": last_props,
        "t_first": t_first,
        "t_last": t_last
    }


def retrack_sequence_backward(
    exp_dir: Path,
    sequence: str,
    films: List[str],
    current_global_cells: Dict[str, List[int]],
    qc_data: Dict[str, Any],
    mother_area_ratio_thresh: float = 1.4,
    max_centroid_dist: float = 65.0,
    min_iou: float = 0.05
) -> Dict[str, Any]:
    """
    Executes backward tracking from Film N-1 to Film 0 with shared-mother merging.
    Preserves all existing 13/13 complete and curated tracks.
    """
    num_films = len(films)
    print(f"\n======================================================================")
    print(f"RETRACKING SEQUENCE BACKWARD: {sequence} ({num_films} Films)")
    print(f"======================================================================")

    # 1. Load data for all films
    print("Loading keyframe masks for all films...")
    film_cache = {}
    for f in films:
        film_cache[f] = load_film_cell_data(exp_dir, f)
        print(f"  Loaded {f:14s}: {len(film_cache[f]['cells'])} local cells.")

    # 2. Separate Protected Full (13/13) & Curated Tracks from Partial Tracks
    protected_globals: Dict[str, List[int]] = {}
    partial_globals: Dict[str, List[int]] = {}

    for gid, tr in current_global_cells.items():
        st = qc_data.get(gid, {}).get("status", "unreviewed")
        is_full_13 = (len(tr) == num_films and all(x > 0 for x in tr))
        is_curated = (st in ["corrected", "good"])
        
        if is_full_13 or is_curated:
            protected_globals[gid] = list(tr)
        else:
            partial_globals[gid] = list(tr)

    print(f"\nProtected Complete Tracks: {len(protected_globals)}")
    print(f"Partial/Candidate Tracks to Resolve/Consolidate: {len(partial_globals)}")

    # 3. Determine Claimed Local Cells across Films
    claimed_by_film: List[Dict[int, List[str]]] = [{} for _ in range(num_films)]
    for gid, tr in protected_globals.items():
        for f_idx, loc_id in enumerate(tr):
            if loc_id > 0:
                claimed_by_film[f_idx].setdefault(loc_id, []).append(gid)

    # Count how many local cells in each film are claimed
    last_f_idx = num_films - 1
    last_film_name = films[last_f_idx]
    all_last_cids = sorted(film_cache[last_film_name]["cells"].keys())
    unclaimed_last_cids = [c for c in all_last_cids if c not in claimed_by_film[last_f_idx]]

    print(f"\nTerminal Film ({last_film_name}):")
    print(f"  Total Local Cells       : {len(all_last_cids)}")
    print(f"  Already Claimed (13/13) : {len(all_last_cids) - len(unclaimed_last_cids)}")
    print(f"  Unclaimed Final Cells   : {len(unclaimed_last_cids)} (To trace backward)")

    # 4. Initialize New Backward Tracks for Unclaimed Final Cells
    new_backward_tracks: Dict[str, List[int]] = {}
    for cid in unclaimed_last_cids:
        # Check if there was an existing global cell named for this last local cell
        cand_name = None
        for old_gid, tr in partial_globals.items():
            if len(tr) == num_films and tr[last_f_idx] == cid:
                cand_name = old_gid
                break
        if not cand_name:
            cand_name = f"{sequence}_{last_film_name}_cell_{cid}"

        # Initialize track with -1 up to last film, then [cid]
        tr = [-1] * num_films
        tr[last_f_idx] = cid
        new_backward_tracks[cand_name] = tr

    # 5. Backward Propagation Step by Step: From Film N-1 down to Film 0
    mother_daughter_links = []

    for f_idx in range(last_f_idx - 1, -1, -1):
        f_cur = films[f_idx]       # Film i (earlier)
        f_next = films[f_idx + 1]  # Film i+1 (later)

        masks_cur_last = film_cache[f_cur]["last_masks"]     # End of Film i
        props_cur_last = film_cache[f_cur]["last_props"]
        
        masks_next_first = film_cache[f_next]["first_masks"] # Start of Film i+1
        props_next_first = film_cache[f_next]["first_props"]

        # Local cell IDs in Film i+1 that currently need a backward link in our active tracks
        active_next_cids = []
        active_next_gids = []
        for gid, tr in new_backward_tracks.items():
            if tr[f_idx + 1] > 0 and tr[f_idx] == -1:
                active_next_cids.append(tr[f_idx + 1])
                active_next_gids.append(gid)

        if not active_next_cids:
            continue

        print(f"\nStepping Backward: {f_next} -> {f_cur} (Resolving {len(active_next_cids)} daughter tracks)")

        # Candidate local cells in Film i
        cand_cur_cids = list(props_cur_last.keys())

        # Build Cost Matrix for spatial matching
        cost_matrix = np.ones((len(active_next_cids), len(cand_cur_cids)), dtype=np.float32)

        for r_idx, c_next in enumerate(active_next_cids):
            m_next = masks_next_first.get(c_next)
            p_next = props_next_first.get(c_next)
            if not p_next: continue
            
            a_next = p_next["area"]
            cx_next, cy_next = p_next["centroid"]

            for c_idx, c_cur in enumerate(cand_cur_cids):
                m_cur = masks_cur_last.get(c_cur)
                p_cur = props_cur_last.get(c_cur)
                if not p_cur: continue
                
                a_cur = p_cur["area"]
                cx_cur, cy_cur = p_cur["centroid"]

                # Centroid distance check
                dist = np.sqrt((cx_next - cx_cur)**2 + (cy_next - cy_cur)**2)
                if dist > max_centroid_dist:
                    continue

                # Overlap IoU
                iou = 0.0
                if m_next is not None and m_cur is not None:
                    inter = np.logical_and(m_next > 0, m_cur > 0).sum()
                    if inter > 0:
                        iou = inter / float(a_next + a_cur - inter)

                if iou >= min_iou or dist <= 25.0:
                    cost = (1.0 - iou) * 0.7 + (dist / max_centroid_dist) * 0.3
                    cost_matrix[r_idx, c_idx] = cost

        # Solve matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.85:
                c_next = active_next_cids[r]
                gid = active_next_gids[r]
                c_cur = cand_cur_cids[c]
                matched_pairs.append((gid, c_next, c_cur, cost_matrix[r, c]))

        print(f"  Matched {len(matched_pairs)}/{len(active_next_cids)} tracks backward to {f_cur}.")

        # Assign matches and handle mother-sharing
        for gid, c_next, c_cur, cost in matched_pairs:
            p_cur = props_cur_last[c_cur]
            p_next = props_next_first[c_next]
            a_cur = p_cur["area"]
            a_next = p_next["area"]
            area_ratio = a_cur / float(max(1, a_next))

            existing_holders = claimed_by_film[f_idx].get(c_cur, [])

            if not existing_holders:
                new_backward_tracks[gid][f_idx] = c_cur
                claimed_by_film[f_idx].setdefault(c_cur, []).append(gid)
            else:
                # MOTHER-SHARING CASE:
                if len(existing_holders) < 2 and area_ratio >= mother_area_ratio_thresh:
                    holder_gid = existing_holders[0]
                    holder_track = protected_globals.get(holder_gid) or new_backward_tracks.get(holder_gid) or partial_globals.get(holder_gid)
                    
                    if holder_track:
                        for k in range(0, f_idx + 1):
                            new_backward_tracks[gid][k] = holder_track[k]
                                
                        claimed_by_film[f_idx].setdefault(c_cur, []).append(gid)
                        mother_daughter_links.append({
                            "mother_film": f_cur,
                            "film_idx": f_idx,
                            "mother_loc_id": c_cur,
                            "mother_area": a_cur,
                            "daughter_film": f_next,
                            "daughter_gid": gid,
                            "daughter_loc_id": c_next,
                            "daughter_area": a_next,
                            "ratio": round(area_ratio, 2),
                            "sister_gid": holder_gid
                        })
                        print(f"    [SHARED MOTHER] {f_cur} #{c_cur} (Area {a_cur}px) shared by {holder_gid} and {gid} (Daughter #{c_next}, Area {a_next}px, {area_ratio:.2f}x)")
                else:
                    if len(existing_holders) < 2:
                        new_backward_tracks[gid][f_idx] = c_cur
                        claimed_by_film[f_idx].setdefault(c_cur, []).append(gid)

    # 6. Combine Protected Tracks and Newly Resolved Backward Tracks
    final_global_cells: Dict[str, List[int]] = {}
    
    for gid, tr in protected_globals.items():
        final_global_cells[gid] = tr

    for gid, tr in new_backward_tracks.items():
        valid_cnt = sum(1 for x in tr if x > 0)
        if valid_cnt > 0:
            final_global_cells[gid] = tr

    print(f"\n======================================================================")
    print(f"RETRACKING COMPLETE: {len(final_global_cells)} Final Global Cells")
    print(f"Shared Mother Merges Formed: {len(mother_daughter_links)}")
    print(f"======================================================================")

    return {
        "sequence": sequence,
        "films": films,
        "global_cells": final_global_cells,
        "mother_daughter_links": mother_daughter_links,
        "protected_count": len(protected_globals),
        "new_tracks_count": len(new_backward_tracks)
    }


def main():
    exp_dir = MOVIE_ROOT / EXP
    seq_file = exp_dir / "sequence_linkage.json"

    with open(seq_file) as f:
        seq_config = json.load(f)

    target_seq = "5_1_N1_F1"
    films = seq_config[target_seq]["films"]
    cur_globals = seq_config[target_seq]["global_cells"]

    qc_file = exp_dir / f"qc_{target_seq}.json"
    qc_data = {}
    if qc_file.exists():
        with open(qc_file) as f:
            qc_data = json.load(f)

    res = retrack_sequence_backward(
        exp_dir=exp_dir,
        sequence=target_seq,
        films=films,
        current_global_cells=cur_globals,
        qc_data=qc_data
    )

    final_cells = res["global_cells"]
    complete_13 = sum(1 for gid, tr in final_cells.items() if len(tr) == 13 and all(x > 0 for x in tr))
    print(f"\nSummary for {target_seq}:")
    print(f"  Total Global Cells before: {len(cur_globals)}")
    print(f"  Total Global Cells after : {len(final_cells)}")
    print(f"  Complete 13/13 Tracks    : {complete_13} ({complete_13/len(final_cells)*100:.1f}%)")


if __name__ == "__main__":
    main()
