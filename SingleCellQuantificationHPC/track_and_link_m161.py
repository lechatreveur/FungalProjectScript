#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_and_link_m161.py

Stage 3 keyframe tracking and cross-film linkage for M161 (2026_09_03,
`NeonG_YES_1`). Copy-to-modify variant of track_and_link_m162.py (P15): M161
is the replicate control for M162, so the tracking logic is held identical and
only the film naming and the sequence list differ.

M161 has FL1-FL6 + BF1-BF5 over 4 fields, against M162's FL1-FL4 + BF1-BF3.

Adds `--films`, because the FL1-only comparison needs 4 films rather than all
44. When the selection contains a single FL film per field, each sequence is
that one film and the cross-film linkage step is a no-op — which is correct,
since a datapoint is one cell in ONE film over 101 frames (P12) and the FL1
comparison never needs identities carried between films.

Keyframe scheme:
- FL: t = 0, 50, 100
- BF: t = 0, 20, 40

Outputs:
- <film>/TrackedCells_<film>/cell_<cid>_masks.csv
- sequence_linkage.json
- qc_<sequence>.json
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
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

# Data stays on the external SSD, never the system disk (P4).
DEFAULT_EXP_DIR = Path("/Volumes/X10 Pro/Movies/2026_09_03_M161")

FL_FILMS = [f"FL{i}" for i in range(1, 7)]   # M161 has FL1-FL6
BF_FILMS = [f"BF{i}" for i in range(1, 6)]   # and BF1-BF5
N_FIELDS = 4


def get_keyframes_for_film(film_name: str) -> list[int]:
    if "FL" in film_name:
        return [0, 50, 100]
    elif "BF" in film_name:
        return [0, 20, 40]
    else:
        return [0]


def track_film_3keyframes(
    exp_dir: Path,
    film: str,
    min_area: int = 1500,
    min_iou: float = 0.15
) -> Dict[int, Dict[int, str]]:
    film_dir = exp_dir / film
    tracked_dir = film_dir / f"TrackedCells_{film}"
    tracked_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = film_dir / f"Masks_{film}"

    k_times = get_keyframes_for_film(film)
    t0 = k_times[0]
    seg0_path = masks_dir / f"{film}_t_{t0:03d}_c_0_seg.tif"
    if not seg0_path.exists():
        print(f"  [Warning] First keyframe mask missing for {film}: {seg0_path}")
        return {}

    seg0 = tifffile.imread(str(seg0_path))
    H, W = seg0.shape[:2]

    props0 = regionprops(seg0)
    initial_labels = [r.label for r in props0 if r.area >= min_area]

    cell_tracks: Dict[int, Dict[int, str]] = {}
    current_label_map: Dict[int, int] = {}

    for cid in initial_labels:
        mask = (seg0 == cid).astype(np.uint8)
        cell_tracks[cid] = {t0: encode_mask_to_rle(mask)}
        current_label_map[cid] = cid

    prev_seg = seg0

    for t in k_times[1:]:
        seg_path = masks_dir / f"{film}_t_{t:03d}_c_0_seg.tif"
        if not seg_path.exists():
            for cid in current_label_map.keys():
                cell_tracks[cid][t] = ""
            continue

        seg_cur = tifffile.imread(str(seg_path))
        mask_overlap = (prev_seg > 0) & (seg_cur > 0)
        match_candidates = {}
        if mask_overlap.any():
            pairs, counts = np.unique(
                np.column_stack((prev_seg[mask_overlap], seg_cur[mask_overlap])),
                axis=0,
                return_counts=True
            )
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
                    mask = (seg_cur == c_lbl).astype(np.uint8)
                    cell_tracks[cid][t] = encode_mask_to_rle(mask)
                    new_label_map[cid] = c_lbl
                else:
                    cell_tracks[cid][t] = ""
            else:
                cell_tracks[cid][t] = ""

        current_label_map = new_label_map
        prev_seg = seg_cur

    # Determine total timepoints in film to create complete timeline table
    is_fl = "FL" in film
    total_t = 101 if is_fl else 41
    rle_col = "rle_gfp" if is_fl else "rle_bf"
    alt_col = "rle_bf" if is_fl else "rle_gfp"

    for cid, t_dict in cell_tracks.items():
        rows = []
        for t in range(total_t):
            rows.append({
                "time_point": t,
                "width": W,
                "height": H,
                rle_col: t_dict.get(t, ""),
                alt_col: ""
            })
        df = pd.DataFrame(rows)
        csv_path = tracked_dir / f"cell_{cid}_masks.csv"
        df.to_csv(csv_path, index=False)

    print(f"  Film {film}: tracked {len(cell_tracks)} cells across keyframes {k_times}.")
    return cell_tracks


def link_sequence_3keyframes(
    exp_dir: Path,
    sequence: str,
    films: List[str]
) -> Dict[str, Any]:
    print(f"\n--- Linking sequence {sequence} ({len(films)} films) ---")

    film_cells: Dict[str, Dict[int, pd.DataFrame]] = {}
    film_last_masks: Dict[str, Dict[int, np.ndarray]] = {}
    film_first_masks: Dict[str, Dict[int, np.ndarray]] = {}

    for f in films:
        t_dir = exp_dir / f / f"TrackedCells_{f}"
        film_cells[f] = {}
        film_last_masks[f] = {}
        film_first_masks[f] = {}

        k_times = get_keyframes_for_film(f)
        t_first = k_times[0]
        t_last = k_times[-1]
        rle_col = "rle_gfp" if "FL" in f else "rle_bf"

        for csv_f in t_dir.glob("cell_*_masks.csv"):
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_f.name)
            if not m:
                continue
            cid = int(m.group(1))
            try:
                df = pd.read_csv(csv_f)
                film_cells[f][cid] = df

                # First keyframe mask
                r_first = df[df["time_point"] == t_first]
                if not r_first.empty:
                    rle1 = str(r_first.iloc[0].get(rle_col, ""))
                    if rle1 and rle1 != "nan":
                        mask1 = validate_and_decode_rle(
                            rle1,
                            int(r_first.iloc[0]["height"]),
                            int(r_first.iloc[0]["width"])
                        )
                        if mask1.any():
                            film_first_masks[f][cid] = mask1

                # Last keyframe mask
                r_last = df[df["time_point"] == t_last]
                if not r_last.empty:
                    rle2 = str(r_last.iloc[0].get(rle_col, ""))
                    if rle2 and rle2 != "nan":
                        mask2 = validate_and_decode_rle(
                            rle2,
                            int(r_last.iloc[0]["height"]),
                            int(r_last.iloc[0]["width"])
                        )
                        if mask2.any():
                            film_last_masks[f][cid] = mask2
            except Exception as e:
                pass

    # Initialize tracks from first film
    f0 = films[0]
    c0_ids = sorted(film_cells[f0].keys())
    global_cells: Dict[str, List[int]] = {}
    for cid in c0_ids:
        gid = f"{sequence}_cell_{cid}"
        global_cells[gid] = [cid]

    # Map across film pairs
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
                if iou >= 0.05:
                    mapping[listA[r]] = listB[c]

        print(f"  Linked {fA} -> {fB}: {len(mapping)}/{len(listA)} cells matched.")

        mapped_B = set()
        for gid, track in list(global_cells.items()):
            last_cid = track[-1]
            if last_cid != -1 and last_cid in mapping:
                next_cid = mapping[last_cid]
                global_cells[gid].append(next_cid)
                mapped_B.add(next_cid)
            else:
                global_cells[gid].append(-1)

        # Add newly appearing cells starting at fB
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
    parser = argparse.ArgumentParser(description="Track and link M161 films on keyframes")
    parser.add_argument("--exp_dir", type=str, default=str(DEFAULT_EXP_DIR))
    parser.add_argument("--films", nargs="+", default=None,
                        help="restrict to these films (default: all of M161)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    # Full chronological sequence per field: FL1, BF1, FL2, BF2, ...
    def full_sequence(f_idx: int) -> list:
        films = []
        for i in range(len(FL_FILMS)):
            films.append(f"NeonG_YES_1_{FL_FILMS[i]}_F{f_idx}")
            if i < len(BF_FILMS):
                films.append(f"NeonG_YES_1_{BF_FILMS[i]}_F{f_idx}")
        return films

    sequences = {f"NeonG_YES_1_F{f_idx}": full_sequence(f_idx)
                 for f_idx in range(N_FIELDS)}

    if args.films:
        want = set(args.films)
        sequences = {seq: [f for f in films if f in want]
                     for seq, films in sequences.items()}
        sequences = {seq: films for seq, films in sequences.items() if films}
        missing = want - {f for films in sequences.values() for f in films}
        if missing:
            raise SystemExit(f"films not in M161's layout: {sorted(missing)}")
        print(f"restricted to {sum(len(v) for v in sequences.values())} film(s) "
              f"in {len(sequences)} sequence(s)")

    all_films = []
    for seq_films in sequences.values():
        all_films.extend(seq_films)

    print(f"Tracking {len(all_films)} film(s) across 3 keyframes each...")
    t0 = time.time()

    for idx, film in enumerate(all_films, 1):
        print(f"[{idx}/{len(all_films)}] Tracking film {film}...")
        track_film_3keyframes(exp_dir, film)

    # Link sequences
    seq_linkage = {}
    for seq_name, seq_films in sequences.items():
        seq_linkage[seq_name] = link_sequence_3keyframes(exp_dir, seq_name, seq_films)

        # Initialize QC file for sequence
        qc_file = exp_dir / f"qc_{seq_name}.json"
        qc_data = {}
        for gid in seq_linkage[seq_name]["global_cells"].keys():
            qc_data[gid] = {
                "status": "unreviewed",
                "notes": "initial_keyframe_tracking"
            }
        with open(qc_file, "w") as f:
            json.dump(qc_data, f, indent=2)
        print(f"  ✓ Initialized {qc_file.name} with {len(qc_data)} cells")

    # Save sequence_linkage.json
    seq_file = exp_dir / "sequence_linkage.json"
    with open(seq_file, "w") as f:
        json.dump(seq_linkage, f, indent=2)

    print(f"\n🎉 Keyframe tracking and linkage complete in {time.time() - t0:.1f}s -> {seq_file}")


if __name__ == "__main__":
    main()
