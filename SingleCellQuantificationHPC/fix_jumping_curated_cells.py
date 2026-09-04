#!/usr/bin/env python3
"""
Repair jumping curated ('good' or 'corrected') cells.

Finds all curated cells that contain unphysical transitions (displacement > 30px)
and rebuilds their tracks using optimal spatial Hungarian matching anchored at their
primary valid film positions.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

import sys
HPC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from relink_sequence_tracks import (
    DEFAULT_MOVIE_ROOT,
    DEFAULT_EXP,
    load_film_cell_boundaries,
    match_film_transition,
)



def rle_to_centroid(rle_str: str, H: int = 2000, W: int = 2000) -> Optional[Tuple[float, float]]:
    if not rle_str or str(rle_str) == "nan" or not isinstance(rle_str, str):
        return None
    parts = list(map(int, rle_str.strip().split()))
    if not parts:
        return None
    starts = parts[0::2]
    lengths = parts[1::2]
    area = sum(lengths)
    if area == 0:
        return None
    total_x, total_y = 0, 0
    for s_1, l in zip(starts, lengths):
        s = s_1 - 1
        curr_s, rem = s, l
        while rem > 0:
            c = curr_s // H
            r = curr_s % H
            can_take = min(rem, H - r)
            total_x += c * can_take
            total_y += can_take * r + (can_take * (can_take - 1)) // 2
            curr_s += can_take
            rem -= can_take
    return (total_x / area, total_y / area)


def repair_sequence_curated_tracks(
    base_root: Path,
    exp: str,
    sequence: str,
    max_jump_thresh: float = 30.0,
) -> Tuple[int, int]:
    exp_dir = base_root / exp
    link_file = exp_dir / "sequence_linkage.json"
    qc_file = exp_dir / f"qc_{sequence}.json"

    if not link_file.exists() or not qc_file.exists():
        print(f"Skipping {sequence}: link_file or qc_file missing.")
        return 0, 0

    with open(link_file, "r") as f:
        link_data = json.load(f)
    with open(qc_file, "r") as f:
        qc_data = json.load(f)

    films = link_data[sequence].get("films", [])
    global_cells = link_data[sequence].get("global_cells", {})

    print(f"\n==========================================")
    print(f"Scanning & Repairing: {sequence} ({len(films)} films)")
    print(f"==========================================")

    # 1. Preload boundaries and compute forward + backward spatial transitions
    film_data = {}
    for f in films:
        film_data[f] = load_film_cell_boundaries(exp_dir, f)
        print(f"  Loaded {len(film_data[f])} cells from {f}")

    fwd_transitions: List[Dict[int, int]] = []
    bwd_transitions: List[Dict[int, int]] = []
    for i in range(len(films) - 1):
        fA = films[i]
        fB = films[i + 1]
        m_fwd = match_film_transition(film_data[fA], film_data[fB], max_dist=30.0, max_area_ratio=2.2)
        fwd_transitions.append(m_fwd)
        # Reverse mapping
        m_bwd = {v: k for k, v in m_fwd.items()}
        bwd_transitions.append(m_bwd)

    # 2. Identify curated cells that jump
    repaired_count = 0
    total_curated = 0

    for gid, q in qc_data.items():
        st = q.get("status")
        if st not in ("good", "corrected"):
            continue
        total_curated += 1

        curr_track = global_cells.get(gid)
        if not curr_track or len(curr_track) != len(films):
            continue

        # Check for transitions > max_jump_thresh
        has_jump = False
        anchor_idx = None
        for i in range(len(films)):
            cid = curr_track[i]
            if cid != -1 and cid in film_data[films[i]]:
                if anchor_idx is None:
                    anchor_idx = i
            if i < len(films) - 1:
                cA = curr_track[i]
                cB = curr_track[i + 1]
                if cA != -1 and cB != -1:
                    posA = film_data[films[i]][cA]["last"][2]
                    posB = film_data[films[i + 1]][cB]["first"][2]
                    d = np.hypot(posB[0] - posA[0], posB[1] - posA[1])
                    if d > max_jump_thresh:
                        has_jump = True

        if not has_jump or anchor_idx is None:
            continue

        # 3. Rebuild continuous track from anchor
        anchor_cid = curr_track[anchor_idx]
        new_track = [-1] * len(films)
        new_track[anchor_idx] = anchor_cid

        # Propagate forward
        curr_c = anchor_cid
        for i in range(anchor_idx, len(films) - 1):
            if curr_c != -1 and curr_c in fwd_transitions[i]:
                next_c = fwd_transitions[i][curr_c]
                new_track[i + 1] = next_c
                curr_c = next_c
            else:
                curr_c = -1

        # Propagate backward
        curr_c = anchor_cid
        for i in range(anchor_idx - 1, -1, -1):
            if curr_c != -1 and curr_c in bwd_transitions[i]:
                prev_c = bwd_transitions[i][curr_c]
                new_track[i] = prev_c
                curr_c = prev_c
            else:
                curr_c = -1

        # Replace track in linkage
        global_cells[gid] = new_track
        repaired_count += 1
        print(f"  Fixed {st} cell {gid}:")
        print(f"    Old: {curr_track}")
        print(f"    New: {new_track}")

    print(f"\n{sequence} Summary:")
    print(f"  Total Curated Cells Checked: {total_curated}")
    print(f"  Total Jumping Tracks Repaired: {repaired_count}")

    # Save backup & update
    if repaired_count > 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = exp_dir / f"sequence_linkage.json.bak_repair_{timestamp}"
        with open(backup_file, "w") as f:
            with open(link_file, "r") as orig:
                f.write(orig.read())
        print(f"  Created backup: {backup_file.name}")

        with open(link_file, "w") as f:
            json.dump(link_data, f, indent=2)
        print(f"  Saved repaired linkages to {link_file.name}")

    return total_curated, repaired_count


def main():
    parser = argparse.ArgumentParser(description="Repair jumping curated tracks.")
    parser.add_argument("--movie_root", type=Path, default=DEFAULT_MOVIE_ROOT)
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP)
    parser.add_argument("--all_sequences", action="store_true")
    parser.add_argument("--sequence", type=str)
    args = parser.parse_args()

    link_file = args.movie_root / args.exp / "sequence_linkage.json"
    with open(link_file, "r") as f:
        link_data = json.load(f)

    seqs = [args.sequence] if args.sequence else sorted(link_data.keys())

    total_rep = 0
    for seq in seqs:
        if seq in link_data:
            _, rep = repair_sequence_curated_tracks(args.movie_root, args.exp, seq)
            total_rep += rep

    print(f"\n==========================================")
    print(f"ALL DONE! Repaired {total_rep} jumping curated tracks.")
    print(f"==========================================")


if __name__ == "__main__":
    main()
