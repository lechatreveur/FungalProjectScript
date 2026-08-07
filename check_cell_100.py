import json
import os
import datetime
from pathlib import Path

exp_dir = Path("/Volumes/X10 Pro/Movies/2025_09_17")
seq = "F1"

# Load sequence linkage
linkage_file = exp_dir / "sequence_linkage.json"
with open(linkage_file, 'r') as f:
    linkage = json.load(f)

# Load sequence QC
qc_file = exp_dir / seq / f"qc_{seq}.json"
with open(qc_file, 'r') as f:
    qc = json.load(f)

films = linkage.get(seq, {}).get("films", [])
global_cells = linkage.get(seq, {}).get("global_cells", {})

print(f"Films in sequence {seq}: {films}")

# 1. Check F1_cell_100 mapping and files
g_key = "F1_cell_100"
if g_key in global_cells:
    local_ids = global_cells[g_key]
    status = qc.get(g_key, "No QC Status")
    print(f"\nGlobal ID: {g_key}")
    print(f"  Local Mappings: {local_ids}")
    print(f"  QC Status: {status}")
    
    for idx, film in enumerate(films):
        local_id = local_ids[idx]
        if local_id != -1:
            csv_path = exp_dir / film / f"TrackedCells_{film}" / f"cell_{local_id}_masks.csv"
            if csv_path.exists():
                mtime = csv_path.stat().st_mtime
                dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"    Film: {film} -> Local cell {local_id} -> CSV: {csv_path.name} -> Modified: {dt}")
            else:
                print(f"    Film: {film} -> Local cell {local_id} -> CSV: {csv_path.name} DOES NOT EXIST")
else:
    print(f"\n{g_key} not found in global cells.")

# 2. Check for local cell 100 in each film
print("\nChecking for local cell 100 CSV files across all films:")
for film in films:
    csv_path = exp_dir / film / f"TrackedCells_{film}" / "cell_100_masks.csv"
    if csv_path.exists():
        mtime = csv_path.stat().st_mtime
        dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Film: {film} -> cell_100_masks.csv -> Modified: {dt}")
    else:
        print(f"  Film: {film} -> cell_100_masks.csv DOES NOT EXIST")
