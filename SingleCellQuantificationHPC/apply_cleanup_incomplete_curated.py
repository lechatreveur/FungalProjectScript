#!/usr/bin/env python3
"""
Clean up incomplete curated cells:
- Safely remove incomplete curated cells whose local cells are all registered in complete 13/13 cells.
- Reset incomplete curated cells with unique unregistered local cells to 'mistracked'.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

exp_dir = Path("/Volumes/X10 Pro/Movies/2026_08_28_M160")
seq_file = exp_dir / "sequence_linkage.json"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Backup sequence_linkage.json
shutil.copy2(seq_file, exp_dir / f"sequence_linkage.json.bak_curated_clean_{timestamp}")

with open(seq_file) as f:
    seq_config = json.load(f)

summary = {}

for seq in ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]:
    qc_file = exp_dir / f"qc_{seq}.json"
    if not qc_file.exists(): continue

    shutil.copy2(qc_file, exp_dir / f"qc_{seq}.json.bak_curated_clean_{timestamp}")

    with open(qc_file) as f:
        qc_data = json.load(f)

    globals_dict = seq_config.get(seq, {}).get("global_cells", {})
    films = seq_config.get(seq, {}).get("films", [])
    num_films = len(films)

    # Complete 13/13 cells
    complete_cells = {
        gid: tr for gid, tr in globals_dict.items()
        if len(tr) == num_films and all(x > 0 for x in tr)
    }

    complete_registered = set()
    for tr in complete_cells.values():
        for f_idx, loc_cid in enumerate(tr):
            if loc_cid > 0:
                complete_registered.add((f_idx, loc_cid))

    removed_gids = []
    reset_mistracked_gids = []

    # Iterate over QC items
    for gid in list(qc_data.keys()):
        d = qc_data[gid]
        st = d.get("status")
        if st in ["good", "corrected"]:
            tr = globals_dict.get(gid)
            if tr is None:
                # Ghost key in QC
                del qc_data[gid]
                removed_gids.append(gid)
            elif any(x <= 0 for x in tr):
                # Incomplete curated cell
                valid_pairs = [(f_idx, loc_cid) for f_idx, loc_cid in enumerate(tr) if loc_cid > 0]
                unregistered_pairs = [pair for pair in valid_pairs if pair not in complete_registered]

                if len(unregistered_pairs) == 0:
                    # Safely remove
                    if gid in globals_dict:
                        del globals_dict[gid]
                    del qc_data[gid]
                    removed_gids.append(gid)
                else:
                    # Reset to mistracked
                    qc_data[gid]["status"] = "mistracked"
                    qc_data[gid]["notes"] = f"incomplete_curated_reset ({len(valid_pairs)}/13 films)"
                    reset_mistracked_gids.append(gid)

    # Save updated sequence linkage
    seq_config[seq]["global_cells"] = globals_dict

    # Save updated QC data
    with open(qc_file, "w") as f:
        json.dump(qc_data, f, indent=2)

    summary[seq] = {
        "removed_redundant_stubs": len(removed_gids),
        "reset_to_mistracked": len(reset_mistracked_gids),
        "remaining_globals": len(globals_dict)
    }

# Save updated sequence_linkage.json
with open(seq_file, "w") as f:
    json.dump(seq_config, f, indent=2)

print("\n" + "="*70)
print("CLEANUP OF INCOMPLETE CURATED CELLS COMPLETED")
print("="*70)
for seq, stats in summary.items():
    print(f"{seq}: Removed {stats['removed_redundant_stubs']} redundant stubs | Reset {stats['reset_to_mistracked']} to mistracked | Remaining Globals: {stats['remaining_globals']}")

