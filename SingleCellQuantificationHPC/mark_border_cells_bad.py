#!/usr/bin/env python3
"""
mark_border_cells_bad.py

Identifies all cells in M160 films and sequences whose segmentation masks
touch the 2000x2000 image border (row 0, row 1999, col 0, or col 1999).
Marks them as 'bad' with reason 'touches_border' in:
1. Local film qc.csv files (TrackedCells_<film>/qc.csv)
2. Sequence-level QC files (qc_<sequence>.json)
"""

import os
import re
import json
import pandas as pd
from pathlib import Path

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_08_28_M160"
exp_dir = MOVIE_ROOT / EXP


def rle_touches_border_fast(rle_str, H=2000, W=2000):
    if not rle_str or rle_str == "nan" or not isinstance(rle_str, str):
        return False
    try:
        nums = list(map(int, rle_str.strip().split()))
    except Exception:
        return False
    for i in range(0, len(nums), 2):
        s = nums[i] - 1
        l = nums[i + 1]
        e = s + l - 1
        if s < H or e >= (W - 1) * H:
            return True
        c_start = s // H
        c_end = e // H
        r_start = s % H
        r_end = e % H
        if c_start != c_end or r_start == 0 or r_end == H - 1:
            return True
    return False


def main():
    print(f"Scanning {EXP} for border-touching cells...")

    # 1. Check all local films
    film_border_cells = {}
    film_total_cells = {}

    for f_dir in sorted(exp_dir.iterdir()):
        if not f_dir.is_dir() or not f_dir.name.startswith("5_1_N1_"):
            continue
        film = f_dir.name
        t_dir = f_dir / f"TrackedCells_{film}"
        if not t_dir.exists():
            continue

        border_cids = set()
        all_cids = set()

        for csv_file in t_dir.glob("cell_*_masks.csv"):
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_file.name)
            if not m:
                continue
            cid = int(m.group(1))
            all_cids.add(cid)

            try:
                df = pd.read_csv(csv_file)
                for _, r in df.iterrows():
                    rle = str(r.get("rle_gfp", r.get("rle_bf", "")))
                    if rle_touches_border_fast(rle):
                        border_cids.add(cid)
                        break
            except Exception:
                pass

        film_border_cells[film] = border_cids
        film_total_cells[film] = all_cids

        # Update film TrackedCells_<film>/qc.csv
        qc_path = t_dir / "qc.csv"
        qc_records = {}
        if qc_path.exists():
            try:
                old_df = pd.read_csv(qc_path)
                for _, row in old_df.iterrows():
                    c = int(row["cell_id"]) if pd.notna(row["cell_id"]) else None
                    if c is not None:
                        qc_records[c] = row.to_dict()
            except Exception:
                pass

        for bcid in border_cids:
            if bcid in qc_records:
                qc_records[bcid]["status"] = "bad"
                reasons = str(qc_records[bcid].get("reasons", "")).split(";")
                if "touches_border" not in reasons:
                    reasons = [r for r in reasons if r and r != "nan"] + ["touches_border"]
                qc_records[bcid]["reasons"] = ";".join(reasons)
                qc_records[bcid]["note"] = "Auto-flagged: touches image border"
            else:
                qc_records[bcid] = {
                    "cell_id": bcid,
                    "status": "bad",
                    "reasons": "touches_border",
                    "note": "Auto-flagged: touches image border",
                    "reviewer": "auto_border_qc"
                }

        out_df = pd.DataFrame(list(qc_records.values()))
        if not out_df.empty:
            out_df.to_csv(qc_path, index=False)

        print(f"  {film}: {len(border_cids)} / {len(all_cids)} cells marked bad (touches border)")

    # 2. Update sequence-level QC
    linkage_file = exp_dir / "sequence_linkage.json"
    if linkage_file.exists():
        with open(linkage_file, "r") as f:
            seq_data = json.load(f)

        for seq_name, s_info in seq_data.items():
            films = s_info.get("films", [])
            g_cells = s_info.get("global_cells", {})
            seq_qc_path = exp_dir / f"qc_{seq_name}.json"
            
            seq_qc = {}
            if seq_qc_path.exists():
                try:
                    with open(seq_qc_path, "r") as f:
                        seq_qc = json.load(f)
                except Exception:
                    pass

            border_count = 0
            for gid, track in g_cells.items():
                touches = False
                for f_idx, cid in enumerate(track):
                    if cid != -1 and f_idx < len(films):
                        film = films[f_idx]
                        if cid in film_border_cells.get(film, set()):
                            touches = True
                            break

                if touches:
                    border_count += 1
                    cur_entry = seq_qc.get(gid, {})
                    cur_reasons = cur_entry.get("reasons", [])
                    if isinstance(cur_reasons, str):
                        cur_reasons = cur_reasons.split(";")
                    if "touches_border" not in cur_reasons:
                        cur_reasons.append("touches_border")
                    
                    seq_qc[gid] = {
                        "status": "bad",
                        "reasons": [r for r in cur_reasons if r],
                        "note": "Auto-flagged: touches image border",
                        "reviewer": "auto_border_qc"
                    }

            with open(seq_qc_path, "w", encoding="utf-8") as f:
                json.dump(seq_qc, f, indent=2)

            print(f"=== Sequence {seq_name}: {border_count} / {len(g_cells)} global cells marked bad ===")

    print("Finished marking border cells as bad successfully!")


if __name__ == "__main__":
    main()
