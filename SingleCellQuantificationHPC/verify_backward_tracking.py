#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import numpy as np

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_08_28_M160"
seq = "5_1_N1_F1"

from backward_tracker_m160 import retrack_sequence_backward

exp_dir = MOVIE_ROOT / EXP
with open(exp_dir / "sequence_linkage.json") as f:
    seq_config = json.load(f)

films = seq_config[seq]["films"]
cur_globals = seq_config[seq]["global_cells"]

with open(exp_dir / f"qc_{seq}.json") as f:
    qc_data = json.load(f)

res = retrack_sequence_backward(
    exp_dir=exp_dir,
    sequence=seq,
    films=films,
    current_global_cells=cur_globals,
    qc_data=qc_data
)

new_globals = res["global_cells"]
links = res["mother_daughter_links"]

print("\n--- DETAILED VERIFICATION ---")

# 1. Curated tracks conservation
curated_gids = [gid for gid, q in qc_data.items() if q.get("status") in ["corrected", "good"]]
print(f"Total Curated Tracks in QC: {len(curated_gids)}")
curated_conserved = 0
curated_mismatches = []
for gid in curated_gids:
    if gid in new_globals:
        old_tr = cur_globals[gid]
        new_tr = new_globals[gid]
        if old_tr == new_tr:
            curated_conserved += 1
        else:
            curated_mismatches.append((gid, old_tr, new_tr))
    else:
        curated_mismatches.append((gid, cur_globals.get(gid), "MISSING"))

print(f"Curated Tracks Exactly Conserved: {curated_conserved} / {len(curated_gids)}")
if curated_mismatches:
    print(f"Curated Mismatches ({len(curated_mismatches)}):")
    for m in curated_mismatches[:10]:
        print(f"  {m[0]}: old={m[1]} -> new={m[2]}")

# 2. Check mother sharing frequency & sister pairs
print(f"\nTotal Mother-Daughter Links: {len(links)}")
mother_counts = {}
for l in links:
    m_key = f"{l['mother_film']}_{l['mother_loc_id']}"
    mother_counts[m_key] = mother_counts.get(m_key, 0) + 1

max_daughters_per_mother = max(mother_counts.values()) if mother_counts else 0
print(f"Max daughters sharing any single mother: {max_daughters_per_mother}")

# 3. Check distribution of track lengths
lengths = [sum(1 for x in tr if x > 0) for tr in new_globals.values()]
len_dist = pd.Series(lengths).value_counts().sort_index(ascending=False)
print("\nDistribution of Track Lengths (Valid Films Linked / 13):")
print(len_dist.to_string())

# 4. Check cells that were previously mistracked or duplicate stubs
# Check cell_364 vs BF4_cell_418
print("\nSpecific Check on cell_364 and daughter stubs:")
for test_gid in ["5_1_N1_F1_cell_364", "5_1_N1_F1_5_1_N1_BF4_F1_cell_418", "5_1_N1_F1_cell_67", "5_1_N1_F1_5_1_N1_BF2_F1_cell_72"]:
    if test_gid in new_globals:
        print(f"  {test_gid} in new_globals: {new_globals[test_gid]}")
    else:
        print(f"  {test_gid} removed as redundant stub")

