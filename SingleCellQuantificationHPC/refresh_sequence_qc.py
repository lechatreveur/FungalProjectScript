#!/usr/bin/env python3
"""
Clean QC Refresh Tool for Ground Truth Corrector.

Re-evaluates automated QC flags (tracking jumps, size changes, border contacts)
against the new canonical sequence_linkage.json, while strictly preserving 100%
of all manual human reviews.
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

HPC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(ROOT_DIR))

DEFAULT_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
DEFAULT_EXP = "2026_08_28_M160"


def rle_to_props_and_border(rle_str: str, H: int = 2000, W: int = 2000) -> Tuple[int, Tuple[float, float], bool]:
    """Decode RLE to area, centroid (cx, cy), and whether mask touches the 2000x2000 image border."""
    if not rle_str or str(rle_str) == "nan" or not isinstance(rle_str, str):
        return 0, (np.nan, np.nan), False
    parts = list(map(int, rle_str.strip().split()))
    if not parts:
        return 0, (np.nan, np.nan), False
    starts = parts[0::2]
    lengths = parts[1::2]
    area = sum(lengths)
    if area == 0:
        return 0, (np.nan, np.nan), False

    total_x, total_y = 0, 0
    touches_border = False

    for s_1, l in zip(starts, lengths):
        s = s_1 - 1
        curr_s, rem = s, l
        while rem > 0:
            c = curr_s // H
            r = curr_s % H
            can_take = min(rem, H - r)
            total_x += c * can_take
            total_y += can_take * r + (can_take * (can_take - 1)) // 2

            # Check if run touches rows 0, H-1 or cols 0, W-1
            if c == 0 or c == W - 1:
                touches_border = True
            elif r == 0 or (r + can_take - 1) == H - 1:
                touches_border = True

            curr_s += can_take
            rem -= can_take

    return area, (total_x / area, total_y / area), touches_border


def refresh_qc_for_sequence(
    exp_dir: Path,
    sequence: str,
    films: List[str],
    global_cells: Dict[str, List[int]],
    max_jump: float = 40.0,
    max_size_ratio_same: float = 1.8,
    max_size_ratio_cross: float = 2.2,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Refresh QC records for a sequence, preserving human reviews and re-evaluating automated flags."""
    qc_file = exp_dir / f"qc_{sequence}.json"
    old_qc = {}
    if qc_file.exists():
        with open(qc_file, "r") as f:
            old_qc = json.load(f)

    # 1. Separate human reviews from automated reviews
    human_reviews: Dict[str, Any] = {}
    for gid, entry in old_qc.items():
        reviewer = entry.get("reviewer", "")
        status = entry.get("status", "")
        # Human review: explicitly marked good/corrected or reviewer not starting with 'auto_'
        if status in ("good", "corrected") or (reviewer and not reviewer.startswith("auto_")):
            human_reviews[gid] = entry

    print(f"\n================ Sequence: {sequence} ================")
    print(f"Total global tracks : {len(global_cells)}")
    print(f"Human reviews kept  : {len(human_reviews)} (strictly preserved)")

    # 2. Preload mask properties for all (film, cell) pairs
    film_masks: Dict[str, Dict[int, Dict[int, Tuple[int, Tuple[float, float], bool]]]] = {}
    for f in films:
        film_masks[f] = {}
        t_dir = exp_dir / f / f"TrackedCells_{f}"
        rle_col = "rle_gfp" if "FL" in f else "rle_bf"
        for p in t_dir.glob("cell_*_masks.csv"):
            m = re.match(r"^cell_(\d+)_masks\.csv$", p.name)
            if not m:
                continue
            cid = int(m.group(1))
            try:
                df = pd.read_csv(p)
                col = rle_col if rle_col in df.columns else ("rle_bf" if rle_col == "rle_gfp" else "rle_gfp")
                t_dict = {}
                for _, row in df.iterrows():
                    val = str(row.get(col, "")).strip()
                    if val and val != "nan":
                        t_val = int(row["time_point"])
                        area, centroid, border = rle_to_props_and_border(val)
                        if area > 0:
                            t_dict[t_val] = (area, centroid, border)
                if t_dict:
                    film_masks[f][cid] = t_dict
            except Exception:
                pass

    # 3. Evaluate each non-human-reviewed global track
    new_qc: Dict[str, Any] = dict(human_reviews)

    auto_bad_count = 0
    auto_mistracked_count = 0
    clean_unreviewed_count = 0

    for gid, track in global_cells.items():
        if gid in human_reviews:
            continue

        # Build timeline of keyframes across present films
        timeline = []
        touches_border = False

        for f_idx, f_name in enumerate(films):
            cid = track[f_idx]
            if cid != -1 and cid in film_masks[f_name]:
                t_dict = film_masks[f_name][cid]
                all_t = sorted(t_dict.keys())
                k_times = [all_t[0], all_t[len(all_t) // 2], all_t[-1]] if len(all_t) > 3 else all_t
                for t_val in k_times:
                    if t_val in t_dict:
                        area, centroid, border = t_dict[t_val]
                        timeline.append((f_name, t_val, area, centroid, cid))
                        if border:
                            touches_border = True

        # Check for border contact first
        if touches_border:
            new_qc[gid] = {
                "status": "bad",
                "reasons": ["touches_border"],
                "note": "Auto-flagged bad: touches image border",
                "reviewer": "auto_border_qc",
            }
            auto_bad_count += 1
            continue

        if len(timeline) <= 1:
            clean_unreviewed_count += 1
            continue

        # Check for jumps or sudden size changes
        jumps = []
        size_changes = []
        reasons = []

        for i in range(len(timeline) - 1):
            f1, t1, a1, c1, cid1 = timeline[i]
            f2, t2, a2, c2, cid2 = timeline[i + 1]
            dist = np.hypot(c2[0] - c1[0], c2[1] - c1[1])
            ratio = max(a1, a2) / max(min(a1, a2), 1)

            if dist > max_jump:
                jumps.append(f"jump {dist:.1f}px ({f1}@t{t1}->{f2}@t{t2})")

            is_cross = ("FL" in f1 and "BF" in f2) or ("BF" in f1 and "FL" in f2)
            max_r = max_size_ratio_cross if is_cross else max_size_ratio_same
            if ratio > max_r:
                size_changes.append(f"size change {ratio:.2f}x ({a1}->{a2}px)")

        if jumps:
            reasons.append("jump")
        if size_changes:
            reasons.append("size_change")

        if reasons:
            notes = "; ".join(jumps[:2] + size_changes[:2])
            new_qc[gid] = {
                "status": "mistracked",
                "reasons": reasons,
                "note": f"Auto-flagged mistracked: {notes}",
                "reviewer": "auto_tracking_qc",
            }
            auto_mistracked_count += 1
        else:
            clean_unreviewed_count += 1

    # Status breakdown
    counts = {}
    for entry in new_qc.values():
        st = entry.get("status", "unknown")
        counts[st] = counts.get(st, 0) + 1

    print(f"Auto-flagged bad (border) : {auto_bad_count}")
    print(f"Auto-flagged mistracked   : {auto_mistracked_count}")
    print(f"Clean unreviewed tracks   : {clean_unreviewed_count}")
    print(f"Updated QC breakdown      : {counts}")

    # Backup & write
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_file = exp_dir / f"qc_{sequence}.json.bak_{timestamp}"
    if qc_file.exists():
        with open(backup_file, "w") as f:
            with open(qc_file, "r") as orig:
                f.write(orig.read())
        print(f"Created backup: {backup_file.name}")

    with open(qc_file, "w") as f:
        json.dump(new_qc, f, indent=2)
    print(f"Saved {len(new_qc)} records to {qc_file.name}")

    return new_qc, counts


def main():
    parser = argparse.ArgumentParser(description="Refresh QC flags against relinked sequence tracks.")
    parser.add_argument("--movie_root", type=Path, default=DEFAULT_MOVIE_ROOT)
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP)
    parser.add_argument("--all_sequences", action="store_true", help="Process all sequences in sequence_linkage.json")
    parser.add_argument("--sequence", type=str, help="Specific sequence name to process")
    args = parser.parse_args()

    exp_dir = args.movie_root / args.exp
    link_file = exp_dir / "sequence_linkage.json"
    if not link_file.exists():
        print(f"❌ sequence_linkage.json not found at {link_file}")
        sys.exit(1)

    with open(link_file, "r") as f:
        link_data = json.load(f)

    seqs = [args.sequence] if args.sequence else sorted(link_data.keys())

    for seq_name in seqs:
        if seq_name not in link_data:
            continue
        films = link_data[seq_name].get("films", [])
        global_cells = link_data[seq_name].get("global_cells", {})
        refresh_qc_for_sequence(
            exp_dir=exp_dir,
            sequence=seq_name,
            films=films,
            global_cells=global_cells,
        )


if __name__ == "__main__":
    main()
