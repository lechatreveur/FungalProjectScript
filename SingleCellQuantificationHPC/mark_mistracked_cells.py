#!/usr/bin/env python3
"""
mark_mistracked_cells.py

Automated QC script for ground_truth_corrector:
Identifies unreviewed cells in sequence tracks that exhibit:
1. Centroid jumps between consecutive present keyframes (displacement > jump_threshold, default 40 px)
2. Sudden cell size / area changes (ratio max/min > size_ratio_threshold, default 1.8x)

Labels qualified unreviewed cells as 'mistracked' in sequence QC (qc_<sequence>.json).
Existing reviewed cells ('good', 'bad', 'corrected') are strictly preserved.
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
DEFAULT_EXP = "2026_08_28_M160"
DEFAULT_SEQ = "5_1_N1_F0"


def rle_to_props(rle_str: str, H: int = 2000, W: int = 2000) -> Tuple[int, Tuple[float, float]]:
    """Compute foreground area and centroid (cx, cy) directly from Fortran-order RLE."""
    if not rle_str or rle_str == "nan" or not isinstance(rle_str, str):
        return 0, (np.nan, np.nan)
    parts = list(map(int, rle_str.strip().split()))
    if not parts:
        return 0, (np.nan, np.nan)
    starts = parts[0::2]
    lengths = parts[1::2]
    area = sum(lengths)
    if area == 0:
        return 0, (np.nan, np.nan)

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
    return area, (total_x / area, total_y / area)


def process_sequence(
    exp_dir: Path,
    seq_name: str,
    link_data: Dict[str, Any],
    jump_th: float = 40.0,
    ratio_th: float = 1.8,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if seq_name not in link_data:
        print(f"⚠️ Sequence '{seq_name}' not found in sequence_linkage.json")
        return {}

    films = link_data[seq_name]["films"]
    g_cells = link_data[seq_name]["global_cells"]

    qc_path = exp_dir / f"qc_{seq_name}.json"
    qc_data: Dict[str, Any] = {}
    if qc_path.exists():
        try:
            with open(qc_path, "r", encoding="utf-8") as f:
                qc_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning loading {qc_path}: {e}")

    # Determine unreviewed cells
    unreviewed_cids = [
        gid for gid in g_cells
        if gid not in qc_data or qc_data[gid].get("status") in ("unreviewed", None, "")
    ]
    reviewed_count = len(g_cells) - len(unreviewed_cids)
    print(f"\n================ Sequence: {seq_name} ================")
    print(f"Total global cells : {len(g_cells)}")
    print(f"Already reviewed   : {reviewed_count} (preserved)")
    print(f"Unreviewed to check: {len(unreviewed_cids)}")

    if not unreviewed_cids:
        print("No unreviewed cells to process.")
        return {"total": len(g_cells), "flagged": 0}

    # Preload needed keyframe masks
    needed_files = set()
    for gid in unreviewed_cids:
        track = g_cells[gid]
        for f_idx, cid in enumerate(track):
            if cid != -1 and f_idx < len(films):
                needed_files.add((films[f_idx], cid))

    print(f"Preloading masks for {len(needed_files)} (film, cell) pairs...")
    keyframe_cache: Dict[Tuple[str, int, int], Tuple[int, Tuple[float, float]]] = {}
    for f_name, cid in needed_files:
        csv_p = exp_dir / f_name / f"TrackedCells_{f_name}" / f"cell_{cid}_masks.csv"
        if csv_p.exists():
            try:
                df = pd.read_csv(csv_p)
                for t_val in [0, 50, 100]:
                    rows = df[df["time_point"] == t_val]
                    if not rows.empty:
                        rle_col = "rle_gfp" if "FL" in f_name else "rle_bf"
                        if rle_col not in rows.columns or pd.isna(rows.iloc[0].get(rle_col)):
                            rle_col = "rle_bf" if rle_col == "rle_gfp" else "rle_gfp"
                        rle = str(rows.iloc[0].get(rle_col, ""))
                        a, c = rle_to_props(rle)
                        keyframe_cache[(f_name, cid, t_val)] = (a, c)
            except Exception:
                pass

    # Evaluate each unreviewed cell
    flagged_cells = {}
    clean_cells = []

    for gid in unreviewed_cids:
        track = g_cells[gid]
        series = []
        for f_idx, f_name in enumerate(films):
            cid = track[f_idx]
            for t_val in [0, 50, 100]:
                a, c = keyframe_cache.get((f_name, cid, t_val), (0, (np.nan, np.nan)))
                series.append((f_name, t_val, a, c))

        present = [item for item in series if item[2] > 0]
        if len(present) < 2:
            # Cannot determine jump or size change across keyframes
            clean_cells.append(gid)
            continue

        worst_jump = 0.0
        worst_ratio = 1.0
        reasons = []
        notes = []

        for i in range(len(present) - 1):
            f1, t1, a1, c1 = present[i]
            f2, t2, a2, c2 = present[i + 1]
            d = np.hypot(c2[0] - c1[0], c2[1] - c1[1])
            ratio = max(a1, a2) / min(a1, a2)

            if d > worst_jump:
                worst_jump = d
            if ratio > worst_ratio:
                worst_ratio = ratio

            if d > jump_th:
                notes.append(f"jump {d:.1f}px ({f1}@t{t1}->{f2}@t{t2})")
            if ratio > ratio_th:
                notes.append(f"size change {ratio:.2f}x ({a1}->{a2}px)")

        if worst_jump > jump_th:
            reasons.append("jump")
        if worst_ratio > ratio_th:
            reasons.append("size_change")

        if reasons:
            flagged_cells[gid] = {
                "status": "mistracked",
                "reasons": reasons,
                "note": f"Auto-flagged mistracked: {'; '.join(notes[:2])}",
                "reviewer": "auto_tracking_qc",
            }
        else:
            clean_cells.append(gid)

    print(f"Flagged as mistracked: {len(flagged_cells)} cells ({len(flagged_cells)/len(unreviewed_cids)*100:.1f}%)")
    print(f"Clean unreviewed cells: {len(clean_cells)} cells ({len(clean_cells)/len(unreviewed_cids)*100:.1f}%)")

    if flagged_cells:
        sample_keys = list(flagged_cells.keys())[:3]
        print("\nSample flagged entries:")
        for sk in sample_keys:
            print(f"  {sk}: {flagged_cells[sk]}")

    if not dry_run and flagged_cells:
        # Create timestamped backup first
        if qc_path.exists():
            bak_path = qc_path.with_suffix(".json.bak")
            shutil.copy2(qc_path, bak_path)
            print(f"\nCreated backup: {bak_path}")

        # Update qc_data
        for gid, entry in flagged_cells.items():
            qc_data[gid] = entry

        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump(qc_data, f, indent=2)
        print(f"Saved {len(qc_data)} total QC records to {qc_path}")

    return {
        "sequence": seq_name,
        "total_cells": len(g_cells),
        "unreviewed_checked": len(unreviewed_cids),
        "flagged_mistracked": len(flagged_cells),
        "clean_unreviewed": len(clean_cells),
    }


def main():
    parser = argparse.ArgumentParser(description="Flag mistracked unreviewed cells based on jumps and size changes.")
    parser.add_argument("--movie_root", type=Path, default=MOVIE_ROOT, help="Base movie root directory")
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP, help="Experiment folder name")
    parser.add_argument("--sequence", type=str, default=DEFAULT_SEQ, help="Specific sequence name to process")
    parser.add_argument("--all_sequences", action="store_true", help="Process all sequences in sequence_linkage.json")
    parser.add_argument("--jump_threshold", type=float, default=40.0, help="Centroid jump threshold in pixels (default: 40)")
    parser.add_argument("--ratio_threshold", type=float, default=1.8, help="Area ratio change threshold (default: 1.8)")
    parser.add_argument("--dry_run", action="store_true", help="Preview results without modifying files")
    args = parser.parse_args()

    exp_dir = args.movie_root / args.exp
    link_file = exp_dir / "sequence_linkage.json"
    if not link_file.exists():
        print(f"❌ sequence_linkage.json not found at {link_file}")
        sys.exit(1)

    with open(link_file, "r", encoding="utf-8") as f:
        link_data = json.load(f)

    if args.all_sequences:
        targets = list(link_data.keys())
    else:
        targets = [args.sequence]

    summaries = []
    for seq in targets:
        s = process_sequence(
            exp_dir=exp_dir,
            seq_name=seq,
            link_data=link_data,
            jump_th=args.jump_threshold,
            ratio_th=args.ratio_threshold,
            dry_run=args.dry_run,
        )
        summaries.append(s)

    print("\n================ FINAL SUMMARY ================")
    for s in summaries:
        if s:
            print(f"Sequence {s.get('sequence')}: {s.get('flagged_mistracked')}/{s.get('unreviewed_checked')} unreviewed flagged as mistracked.")


if __name__ == "__main__":
    main()
