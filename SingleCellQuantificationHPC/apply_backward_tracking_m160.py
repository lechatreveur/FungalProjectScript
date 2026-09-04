#!/usr/bin/env python3
"""
Canonical Script to Apply Backward Retracking with Shared-Mother Architecture for M160.
Executes backward retracking on 5_1_N1_F0, 5_1_N1_F1, and 5_1_N1_F2.
Saves timestamped backups before updating sequence_linkage.json and qc_<seq>.json files.
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

HPC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(HPC_DIR))
from backward_tracker_m160 import retrack_sequence_backward

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_08_28_M160"
exp_dir = MOVIE_ROOT / EXP


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Backward Retracking Application for {EXP} (Timestamp: {timestamp})")

    # 1. Load canonical sequence_linkage.json
    seq_file = exp_dir / "sequence_linkage.json"
    if not seq_file.exists():
        raise FileNotFoundError(f"Missing {seq_file}")

    # Backup sequence_linkage.json
    backup_seq_file = exp_dir / f"sequence_linkage.json.bak_{timestamp}"
    shutil.copy2(seq_file, backup_seq_file)
    print(f"✓ Backed up sequence_linkage.json -> {backup_seq_file.name}")

    with open(seq_file) as f:
        seq_config = json.load(f)

    report = {
        "timestamp": timestamp,
        "sequences": {}
    }

    # 2. Process sequences
    target_sequences = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]
    
    for seq in target_sequences:
        if seq not in seq_config:
            print(f"Skipping {seq}: not in sequence_linkage.json")
            continue

        films = seq_config[seq]["films"]
        cur_globals = seq_config[seq]["global_cells"]

        # Load QC data
        qc_file = exp_dir / f"qc_{seq}.json"
        qc_data = {}
        if qc_file.exists():
            # Backup QC file
            backup_qc_file = exp_dir / f"qc_{seq}.json.bak_{timestamp}"
            shutil.copy2(qc_file, backup_qc_file)
            print(f"✓ Backed up qc_{seq}.json -> {backup_qc_file.name}")
            with open(qc_file) as f:
                qc_data = json.load(f)

        # Run Backward Tracking
        res = retrack_sequence_backward(
            exp_dir=exp_dir,
            sequence=seq,
            films=films,
            current_global_cells=cur_globals,
            qc_data=qc_data,
            mother_area_ratio_thresh=1.4,
            max_centroid_dist=65.0,
            min_iou=0.05
        )

        new_globals = res["global_cells"]

        # Update sequence_linkage.json data structure
        seq_config[seq]["global_cells"] = new_globals

        # Update QC file: keep preserved statuses, retain unreviewed for new
        new_qc_data = {}
        for gid in new_globals.keys():
            if gid in qc_data:
                new_qc_data[gid] = qc_data[gid]
            else:
                new_qc_data[gid] = {
                    "status": "unreviewed",
                    "notes": "backward_tracked_mother_merge"
                }

        # Save updated QC file
        with open(qc_file, "w") as f:
            json.dump(new_qc_data, f, indent=2)
        print(f"✓ Updated {qc_file.name} ({len(new_qc_data)} entries)")

        # Compile stats
        comp_13 = sum(1 for tr in new_globals.values() if len(tr) == 13 and all(x > 0 for x in tr))
        curated_conserved = sum(1 for gid, d in qc_data.items() if d.get("status") in ["corrected", "good"] and gid in new_globals)
        total_curated = sum(1 for d in qc_data.values() if d.get("status") in ["corrected", "good"])

        report["sequences"][seq] = {
            "orig_globals": len(cur_globals),
            "new_globals": len(new_globals),
            "complete_13_tracks": comp_13,
            "shared_mother_merges": len(res["mother_daughter_links"]),
            "curated_conserved": f"{curated_conserved}/{total_curated}",
            "mother_daughter_links": res["mother_daughter_links"]
        }

    # 3. Save updated sequence_linkage.json
    with open(seq_file, "w") as f:
        json.dump(seq_config, f, indent=2)
    print(f"✓ Updated sequence_linkage.json successfully.")

    # 4. Save audit report
    report_file = exp_dir / f"backward_tracking_report_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved audit report to {report_file.name}")

    print("\n" + "="*70)
    print("BACKWARD RETRACKING COMPLETED SUCCESSFULLY")
    print("="*70)
    for seq, st in report["sequences"].items():
        print(f"{seq:12s}: {st['orig_globals']} -> {st['new_globals']} globals ({st['complete_13_tracks']} 13/13 tracks, {st['shared_mother_merges']} shared-mother merges, {st['curated_conserved']} curated conserved)")


if __name__ == "__main__":
    main()
