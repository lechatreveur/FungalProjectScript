#!/usr/bin/env python3
"""
mark_mistracked_border_cells_bad.py

Inspects cells labeled as 'mistracked' in sequence QC.
If a cell's segmentation mask touches the 2000x2000 image border in any frame,
marks its status as 'bad' and adds 'touches_border' to its reasons.
"""

import json
import shutil
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
DEFAULT_EXP = "2026_08_28_M160"
DEFAULT_SEQ = "5_1_N1_F0"


def rle_touches_border(rle_str: str, H: int = 2000, W: int = 2000) -> bool:
    """Check if Fortran-order RLE mask touches boundary rows (0, H-1) or cols (0, W-1)."""
    if not rle_str or str(rle_str) == "nan" or not isinstance(rle_str, str):
        return False
    try:
        nums = list(map(int, rle_str.strip().split()))
    except Exception:
        return False
    for i in range(0, len(nums), 2):
        s = nums[i] - 1
        l = nums[i + 1]
        e = s + l - 1
        # Left boundary (col 0)
        if s < H:
            return True
        # Right boundary (col W-1)
        if e >= (W - 1) * H:
            return True
        c_start = s // H
        c_end = e // H
        r_start = s % H
        r_end = e % H
        # Top boundary (row 0), bottom boundary (row H-1), or column wrap
        if c_start != c_end or r_start == 0 or r_end == H - 1:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Mark mistracked cells that touch image border as bad.")
    parser.add_argument("--movie_root", type=Path, default=MOVIE_ROOT, help="Base movie root directory")
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP, help="Experiment folder name")
    parser.add_argument("--sequence", type=str, default=DEFAULT_SEQ, help="Sequence name")
    parser.add_argument("--dry_run", action="store_true", help="Preview changes without modifying files")
    args = parser.parse_args()

    exp_dir = args.movie_root / args.exp
    link_file = exp_dir / "sequence_linkage.json"
    qc_file = exp_dir / f"qc_{args.sequence}.json"

    if not link_file.exists():
        print(f"❌ sequence_linkage.json not found at {link_file}")
        return
    if not qc_file.exists():
        print(f"❌ QC file not found at {qc_file}")
        return

    with open(link_file, "r", encoding="utf-8") as f:
        link_data = json.load(f)

    if args.sequence not in link_data:
        print(f"❌ Sequence '{args.sequence}' not found in linkage")
        return

    films = link_data[args.sequence]["films"]
    g_cells = link_data[args.sequence]["global_cells"]

    with open(qc_file, "r", encoding="utf-8") as f:
        qc_data: Dict[str, Any] = json.load(f)

    mistracked_gids = [gid for gid, data in qc_data.items() if data.get("status") == "mistracked"]
    print(f"Total cells currently labeled 'mistracked': {len(mistracked_gids)}")

    updated_cells = {}
    for gid in mistracked_gids:
        track = g_cells.get(gid, [])
        touches = False
        touch_detail = []

        for f_idx, f_name in enumerate(films):
            if f_idx >= len(track):
                continue
            cid = track[f_idx]
            if cid == -1:
                continue
            csv_p = exp_dir / f_name / f"TrackedCells_{f_name}" / f"cell_{cid}_masks.csv"
            if not csv_p.exists():
                continue
            try:
                df = pd.read_csv(csv_p)
                rle_col = "rle_gfp" if "FL" in f_name else "rle_bf"
                if rle_col not in df.columns or df[rle_col].dropna().empty:
                    rle_col = "rle_bf" if rle_col == "rle_gfp" else "rle_gfp"

                for _, row in df.iterrows():
                    rle = str(row.get(rle_col, ""))
                    if rle_touches_border(rle):
                        touches = True
                        touch_detail.append(f"{f_name}@t{row['time_point']}")
                        break
            except Exception:
                pass
            if touches:
                break

        if touches:
            cur_entry = qc_data[gid]
            reasons = cur_entry.get("reasons", [])
            if isinstance(reasons, str):
                reasons = [r.strip() for r in reasons.split(";") if r.strip()]
            if "touches_border" not in reasons:
                reasons.append("touches_border")

            old_note = cur_entry.get("note", "")
            new_note = f"Auto-flagged bad: touches image border ({touch_detail[0]})"
            if old_note and "touches image border" not in old_note:
                new_note = f"{new_note}; {old_note}"

            updated_cells[gid] = {
                "status": "bad",
                "reasons": reasons,
                "note": new_note,
                "reviewer": "auto_border_qc",
            }

    print(f"Mistracked cells touching border: {len(updated_cells)} / {len(mistracked_gids)}")

    if updated_cells:
        sample_keys = list(updated_cells.keys())[:5]
        print("\nSample updated cells:")
        for sk in sample_keys:
            print(f"  {sk}: {updated_cells[sk]}")

    if not args.dry_run and updated_cells:
        bak_file = qc_file.with_suffix(".json.bak_border")
        shutil.copy2(qc_file, bak_file)
        print(f"\nCreated backup: {bak_file}")

        for gid, entry in updated_cells.items():
            qc_data[gid] = entry

        with open(qc_file, "w", encoding="utf-8") as f:
            json.dump(qc_data, f, indent=2)

        print(f"Successfully updated {len(updated_cells)} cells to 'bad' in {qc_file}")

        # Summary of new status distribution
        status_counts = {}
        for k, v in qc_data.items():
            st = v.get("status")
            status_counts[st] = status_counts.get(st, 0) + 1
        print(f"New QC status counts in {args.sequence}: {status_counts}")


if __name__ == "__main__":
    main()
