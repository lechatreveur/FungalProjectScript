#!/usr/bin/env python3
"""
Ultra-fast Sequence Linker for M160.
Directly uses TIFF label overlaps to link cell tracks across consecutive films in seconds.
"""

import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_08_28_M160"

def get_film_keyframes(film_dir: Path, film: str) -> Tuple[int, int, Path, Path]:
    masks_dir = film_dir / f"Masks_{film}"
    seg_files = sorted([f for f in masks_dir.glob("*_seg.tif") if not f.name.startswith(".")])
    time_map = {}
    for f in seg_files:
        m = re.search(r"_t_(\d+)_", f.name) or re.search(r"_t(\d+)_", f.name)
        if m:
            time_map[int(m.group(1))] = f

    sorted_times = sorted(time_map.keys())
    t_first = sorted_times[0]
    t_last = sorted_times[-1]
    return t_first, t_last, time_map[t_first], time_map[t_last]


def link_sequence(exp_dir: Path, sequence: str, films: List[str]) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f"Linking sequence: {sequence} ({len(films)} films)")
    print(f"==========================================")

    # 1. Discover cells in Film 0
    f0 = films[0]
    t_dir_0 = exp_dir / f0 / f"TrackedCells_{f0}"
    cids_0 = sorted([
        int(m.group(1)) for f in t_dir_0.glob("cell_*_masks.csv")
        if (m := re.match(r"^cell_(\d+)_masks\.csv$", f.name))
    ])
    
    global_cells: Dict[str, List[int]] = {}
    for cid in cids_0:
        gid = f"{sequence}_cell_{cid}"
        global_cells[gid] = [cid]

    # 2. Iterate through consecutive film pairs
    for i in range(len(films) - 1):
        fA = films[i]
        fB = films[i + 1]

        t_dir_A = exp_dir / fA / f"TrackedCells_{fA}"
        t_dir_B = exp_dir / fB / f"TrackedCells_{fB}"

        cids_B = sorted([
            int(m.group(1)) for f in t_dir_B.glob("cell_*_masks.csv")
            if (m := re.match(r"^cell_(\d+)_masks\.csv$", f.name))
        ])

        _, t_last_A, _, segA_path = get_film_keyframes(exp_dir / fA, fA)
        t_first_B, _, segB_path, _ = get_film_keyframes(exp_dir / fB, fB)

        # Load segmentations
        segA = tifffile.imread(str(segA_path))
        segB = tifffile.imread(str(segB_path))

        # Build label map for Film A at t_last: cidA -> label in segA
        rle_col_A = "rle_gfp" if "FL" in fA else "rle_bf"
        cid_to_labelA = {}
        for csv_file in t_dir_A.glob("cell_*_masks.csv"):
            cid = int(re.match(r"^cell_(\d+)_masks\.csv$", csv_file.name).group(1))
            try:
                df = pd.read_csv(csv_file)
                r_last = df[df["time_point"] == t_last_A]
                if not r_last.empty:
                    rle = str(r_last.iloc[0].get(rle_col_A, ""))
                    if rle and rle != "nan":
                        first_idx = int(rle.split()[0]) - 1
                        lbl = int(segA[first_idx % 2000, first_idx // 2000])
                        if lbl > 0:
                            cid_to_labelA[cid] = lbl
            except Exception:
                pass

        labelA_to_cid = {lbl: cid for cid, lbl in cid_to_labelA.items()}

        # Measure areas
        areaA = {r.label: r.area for r in regionprops(segA)}
        areaB = {r.label: r.area for r in regionprops(segB)}

        # Direct overlap
        mask_ov = (segA > 0) & (segB > 0)
        mapping: Dict[int, int] = {} # cidA -> cidB

        if mask_ov.any():
            pairs, counts = np.unique(
                np.column_stack((segA[mask_ov], segB[mask_ov])),
                axis=0,
                return_counts=True
            )

            # Match candidates
            candidates = []
            for (lblA, lblB), count in zip(pairs, counts):
                if lblA in labelA_to_cid and lblB in cids_B:
                    aA = areaA.get(lblA, 1)
                    aB = areaB.get(lblB, 1)
                    iou = count / float(aA + aB - count)
                    if iou >= 0.05:
                        candidates.append((labelA_to_cid[lblA], lblB, iou))

            # 1-to-1 greedy matching by highest IoU
            candidates.sort(key=lambda x: x[2], reverse=True)
            used_A = set()
            used_B = set()
            for cA, cB, iou in candidates:
                if cA not in used_A and cB not in used_B:
                    mapping[cA] = cB
                    used_A.add(cA)
                    used_B.add(cB)

        print(f"  Step {i+1}/12: {fA} -> {fB} matched {len(mapping)} cells.")

        # Update existing global tracks
        mapped_B = set()
        for gid, track in list(global_cells.items()):
            last_cid = track[-1]
            if last_cid != -1 and last_cid in mapping:
                next_cid = int(mapping[last_cid])
                global_cells[gid].append(next_cid)
                mapped_B.add(next_cid)
            else:
                global_cells[gid].append(-1)

        # Add new tracks for cells appearing in fB
        for cB in cids_B:
            cB_int = int(cB)
            if cB_int not in mapped_B:
                gid = f"{sequence}_{fB}_cell_{cB_int}"
                track = [-1] * (i + 1) + [cB_int]
                global_cells[gid] = track

    clean_global_cells = {str(k): [int(x) for x in v] for k, v in global_cells.items()}
    print(f"Sequence {sequence} created {len(clean_global_cells)} global cell tracks across {len(films)} films.")
    return {
        "films": films,
        "global_cells": clean_global_cells,
        "lineage": {}
    }


def main():
    exp_dir = MOVIE_ROOT / EXP
    seq_file = exp_dir / "sequence_linkage.json"

    # Define M160 sequences directly
    seq_config = {}
    for field in ["F0", "F1", "F2"]:
        ordered = []
        for i in range(1, 8):
            ordered.append(f"5_1_N1_FL{i}_{field}")
            if i <= 6:
                ordered.append(f"5_1_N1_BF{i}_{field}")
        seq_config[f"5_1_N1_{field}"] = {"films": ordered}

    t0 = time.time()
    updated_seq_data = {}
    for seq_name, seq_data in seq_config.items():
        films = seq_data.get("films", [])
        updated_seq_data[seq_name] = link_sequence(exp_dir, seq_name, films)

    # Write atomically
    tmp_seq_file = seq_file.with_suffix(".tmp")
    with open(tmp_seq_file, "w") as f:
        json.dump(updated_seq_data, f, indent=2)
    tmp_seq_file.replace(seq_file)

    print(f"\n🎉 Successfully linked all sequences in {time.time() - t0:.2f}s!")
    print(f"Saved to {seq_file}")

if __name__ == "__main__":
    main()
