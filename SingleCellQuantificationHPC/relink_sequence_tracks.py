#!/usr/bin/env python3
"""
Robust Spatial Sequence Track Relinker for Ground Truth Corrector.

Rebuilds sequence_linkage.json tracks across film transitions using:
- Dynamic detection of first and last active timepoints per film.
- Distance-gated Hungarian bipartite assignment (IoU + centroid displacement).
- Strict preservation of user-curated ('good', 'corrected') tracks from qc_<sequence>.json.
- Automated timestamped backups before updating canonical linkage.
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HPC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(ROOT_DIR))

DEFAULT_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
DEFAULT_EXP = "2026_08_28_M160"


def rle_to_intervals_and_props(rle_str: str, H: int = 2000, W: int = 2000) -> Tuple[List[Tuple[int, int]], int, Tuple[float, float]]:
    """Decode RLE string to list of intervals, total area, and centroid (cx, cy)."""
    if not rle_str or str(rle_str) == "nan" or not isinstance(rle_str, str):
        return [], 0, (np.nan, np.nan)
    parts = list(map(int, rle_str.strip().split()))
    if not parts:
        return [], 0, (np.nan, np.nan)
    starts = parts[0::2]
    lengths = parts[1::2]
    intervals = []
    area = 0
    total_x, total_y = 0, 0
    for s_1, l in zip(starts, lengths):
        s = s_1 - 1
        e = s + l
        intervals.append((s, e))
        area += l
        curr_s, rem = s, l
        while rem > 0:
            c = curr_s // H
            r = curr_s % H
            can_take = min(rem, H - r)
            total_x += c * can_take
            total_y += can_take * r + (can_take * (can_take - 1)) // 2
            curr_s += can_take
            rem -= can_take
    return intervals, area, (total_x / area, total_y / area)


def interval_intersection(intA: List[Tuple[int, int]], intB: List[Tuple[int, int]]) -> int:
    """Compute intersection count between two sorted RLE interval sets in O(N+M)."""
    i, j = 0, 0
    inter = 0
    nA, nB = len(intA), len(intB)
    while i < nA and j < nB:
        sA, eA = intA[i]
        sB, eB = intB[j]
        st = max(sA, sB)
        en = min(eA, eB)
        if st < en:
            inter += (en - st)
        if eA < eB:
            i += 1
        else:
            j += 1
    return inter


def load_film_cell_boundaries(exp_dir: Path, film: str) -> Dict[int, Dict[str, Any]]:
    """
    Load first and last active mask properties for each cell in film.
    Returns {cid: {'first': (t, area, (cx, cy), intervals), 'last': (t, area, (cx, cy), intervals)}}.
    """
    t_dir = exp_dir / film / f"TrackedCells_{film}"
    if not t_dir.exists():
        print(f"  [Warning] Tracked directory missing: {t_dir}")
        return {}

    film_data = {}
    rle_col = "rle_gfp" if "FL" in film else "rle_bf"

    for p in t_dir.glob("cell_*_masks.csv"):
        m = re.match(r"^cell_(\d+)_masks\.csv$", p.name)
        if not m:
            continue
        cid = int(m.group(1))
        try:
            df = pd.read_csv(p)
            col = rle_col if rle_col in df.columns else ("rle_bf" if rle_col == "rle_gfp" else "rle_gfp")
            valid_rows = []
            for _, row in df.iterrows():
                val = str(row.get(col, "")).strip()
                if val and val != "nan":
                    valid_rows.append((int(row["time_point"]), val))
            if valid_rows:
                t_first, rle_first = valid_rows[0]
                t_last, rle_last = valid_rows[-1]
                int_first, a_first, c_first = rle_to_intervals_and_props(rle_first)
                int_last, a_last, c_last = rle_to_intervals_and_props(rle_last)
                if a_first > 0 or a_last > 0:
                    film_data[cid] = {
                        "first": (t_first, a_first, c_first, int_first),
                        "last": (t_last, a_last, c_last, int_last),
                    }
        except Exception as e:
            pass

    return film_data


def match_film_transition(
    cellsA: Dict[int, Dict[str, Any]],
    cellsB: Dict[int, Dict[str, Any]],
    max_dist: float = 30.0,
    max_area_ratio: float = 2.2,
) -> Dict[int, int]:
    """Solve optimal distance-gated Hungarian bipartite assignment between Film A (last) and Film B (first)."""
    listA = [cid for cid, d in cellsA.items() if d["last"][1] > 0]
    listB = [cid for cid, d in cellsB.items() if d["first"][1] > 0]

    mapping: Dict[int, int] = {}
    if not listA or not listB:
        return mapping

    cost_matrix = np.full((len(listA), len(listB)), 1e6, dtype=np.float32)
    for iA, cA in enumerate(listA):
        _, aA, (xA, yA), intA = cellsA[cA]["last"]
        for iB, cB in enumerate(listB):
            _, aB, (xB, yB), intB = cellsB[cB]["first"]
            dist = np.hypot(xB - xA, yB - yA)
            ratio = max(aA, aB) / max(min(aA, aB), 1)
            if dist <= max_dist and ratio <= max_area_ratio:
                inter = interval_intersection(intA, intB)
                union = aA + aB - inter
                iou = inter / float(union) if union > 0 else 0.0
                if iou > 0.0 or dist <= 15.0:
                    cost_matrix[iA, iB] = (1.0 - iou) + (dist / 50.0)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 100.0:
            mapping[listA[r]] = listB[c]

    return mapping


def relink_sequence(
    exp_dir: Path,
    sequence: str,
    films: List[str],
    preserved_tracks: Optional[Dict[str, List[int]]] = None,
    max_dist: float = 30.0,
    max_area_ratio: float = 2.2,
) -> Dict[str, List[int]]:
    """Relink tracks for a single sequence, locking in any preserved_tracks."""
    print(f"\n==========================================")
    print(f"Relinking sequence: {sequence} ({len(films)} films)")
    print(f"==========================================")

    # 1. Preload boundaries for each film
    film_data = {}
    for f in films:
        film_data[f] = load_film_cell_boundaries(exp_dir, f)
        print(f"  Loaded {len(film_data[f])} cells from {f}")

    # 2. Compute pairwise transition links
    transitions = []
    for i in range(len(films) - 1):
        fA = films[i]
        fB = films[i + 1]
        m = match_film_transition(film_data[fA], film_data[fB], max_dist, max_area_ratio)
        transitions.append(m)
        print(f"  Transition {fA} -> {fB}: {len(m)} / {len(film_data[fA])} cells linked.")

    # 3. Build global tracks
    global_tracks: Dict[str, List[int]] = {}

    # Seed tracks from film 0
    f0 = films[0]
    for cid in sorted(film_data[f0].keys()):
        gid = f"{sequence}_cell_{cid}"
        track = [cid]
        curr_cid = cid
        for i in range(len(films) - 1):
            if curr_cid != -1 and curr_cid in transitions[i]:
                next_cid = transitions[i][curr_cid]
                track.append(next_cid)
                curr_cid = next_cid
            else:
                track.append(-1)
                curr_cid = -1
        global_tracks[gid] = track

    # Add tracks for cells appearing in later films
    for f_idx in range(1, len(films)):
        f_curr = films[f_idx]
        reached = {t[f_idx] for t in global_tracks.values() if t[f_idx] != -1}
        for cid in sorted(film_data[f_curr].keys()):
            if cid not in reached:
                gid = f"{sequence}_{f_curr}_cell_{cid}"
                track = [-1] * f_idx + [cid]
                curr_cid = cid
                for i in range(f_idx, len(films) - 1):
                    if curr_cid != -1 and curr_cid in transitions[i]:
                        next_cid = transitions[i][curr_cid]
                        track.append(next_cid)
                        curr_cid = next_cid
                    else:
                        track.append(-1)
                        curr_cid = -1
                global_tracks[gid] = track

    # 4. Strictly preserve user-curated tracks
    if preserved_tracks:
        preserved_count = 0
        for gid, track in preserved_tracks.items():
            if len(track) == len(films):
                global_tracks[gid] = [int(x) for x in track]
                preserved_count += 1
        print(f"  Preserved {preserved_count} user-curated tracks from QC.")

    # Convert all track IDs to standard Python integers
    clean_tracks = {str(k): [int(x) for x in v] for k, v in global_tracks.items()}
    print(f"  Generated {len(clean_tracks)} total global tracks for {sequence}.")
    return clean_tracks


def process_relinking(
    movie_root: Path = DEFAULT_MOVIE_ROOT,
    exp: str = DEFAULT_EXP,
    sequences_to_run: Optional[List[str]] = None,
    max_dist: float = 30.0,
    max_area_ratio: float = 2.2,
):
    exp_dir = movie_root / exp
    link_file = exp_dir / "sequence_linkage.json"

    if not link_file.exists():
        print(f"❌ sequence_linkage.json not found at {link_file}")
        sys.exit(1)

    with open(link_file, "r") as f:
        link_data = json.load(f)

    if not sequences_to_run:
        target_seqs = sorted(link_data.keys())
    else:
        target_seqs = [s for s in sequences_to_run if s in link_data]

    print(f"Starting spatial relinking for experiment: {exp}")
    print(f"Target sequences: {target_seqs}")

    for seq_name in target_seqs:
        films = link_data[seq_name].get("films", [])
        if not films:
            continue

        # Check for preserved user QC tracks ('good' or 'corrected')
        qc_file = exp_dir / f"qc_{seq_name}.json"
        preserved_tracks: Dict[str, List[int]] = {}
        if qc_file.exists():
            try:
                with open(qc_file, "r") as f:
                    qc_data = json.load(f)
                old_tracks = link_data[seq_name].get("global_cells", {})
                for gid, q in qc_data.items():
                    if q.get("status") in ("good", "corrected") and gid in old_tracks:
                        preserved_tracks[gid] = old_tracks[gid]
                print(f"\nFound {len(preserved_tracks)} curated tracks to preserve in {qc_file.name}")
            except Exception as e:
                print(f"  [Warning] Failed loading QC for preservation: {e}")

        new_global_cells = relink_sequence(
            exp_dir=exp_dir,
            sequence=seq_name,
            films=films,
            preserved_tracks=preserved_tracks,
            max_dist=max_dist,
            max_area_ratio=max_area_ratio,
        )

        link_data[seq_name]["global_cells"] = new_global_cells

    # Backup existing linkage
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = exp_dir / f"sequence_linkage.json.bak_{timestamp}"
    with open(backup_path, "w") as f:
        with open(link_file, "r") as orig:
            f.write(orig.read())
    print(f"\nCreated backup: {backup_path}")

    # Write updated linkage
    with open(link_file, "w") as f:
        json.dump(link_data, f, indent=2)
    print(f"Successfully saved updated linkage to {link_file}")


def main():
    parser = argparse.ArgumentParser(description="Robust Spatial Sequence Track Relinker for Ground Truth Corrector.")
    parser.add_argument("--movie_root", type=Path, default=DEFAULT_MOVIE_ROOT, help="Base root folder for movies")
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP, help="Experiment folder name")
    parser.add_argument("--sequence", type=str, help="Specific sequence name to relink (e.g. 5_1_N1_F0)")
    parser.add_argument("--all_sequences", action="store_true", help="Relink all sequences in sequence_linkage.json")
    parser.add_argument("--max_dist", type=float, default=30.0, help="Maximum centroid distance in px for matching")
    parser.add_argument("--max_area_ratio", type=float, default=2.2, help="Maximum area ratio for matching")

    args = parser.parse_args()

    seqs = None
    if args.sequence:
        seqs = [args.sequence]

    process_relinking(
        movie_root=args.movie_root,
        exp=args.exp,
        sequences_to_run=seqs,
        max_dist=args.max_dist,
        max_area_ratio=args.max_area_ratio,
    )


if __name__ == "__main__":
    main()
