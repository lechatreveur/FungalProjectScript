import json
import os
import datetime
from pathlib import Path

exp_dir = Path("/Volumes/X10 Pro/Movies/2025_09_17")
seq = "F1"

# Load sequence QC
qc_file = exp_dir / seq / f"qc_{seq}.json"
with open(qc_file, 'r') as f:
    qc = json.load(f)

print("Direct matches in qc_F1.json for '_100':")
for key, status in qc.items():
    if key.endswith("_100") or "_cell_100" in key:
        print(f"  {key} -> Status: {status}")

# Find any cell_100_masks.csv files in any film directory and print their paths & mtimes
print("\nChecking all cell_100_masks.csv on disk in 2025_09_17:")
for path in exp_dir.rglob("cell_100_masks.csv"):
    mtime = path.stat().st_mtime
    dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  {path.relative_to(exp_dir)} - Modified: {dt}")
