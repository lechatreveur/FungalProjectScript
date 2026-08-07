import os
import datetime
from pathlib import Path

movie_root = Path("/Volumes/X10 Pro/Movies")

all_files = []
for p in movie_root.rglob("cell_*_masks.csv"):
    if "Finetuned" in str(p) or "SAM2_Finetuned" in p.parts:
        continue
    all_files.append(p)

# Sort all by mtime
all_files_sorted = sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)

print("Top 15 most recently modified cell CSV files across ALL movies:")
for f in all_files_sorted[:15]:
    mtime = f.stat().st_mtime
    dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  {f.relative_to(movie_root)} - Modified: {dt}")

print("\nSpecific checks for any cell_100 or mapping to it:")
# Let's search all qc.json files in all experiments
for qf in movie_root.rglob("qc_*.json"):
    import json
    try:
        with open(qf, 'r') as f:
            qc = json.load(f)
        for key, val in qc.items():
            if "100" in key:
                mtime = qf.stat().st_mtime
                dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  In {qf.relative_to(movie_root)} -> {key}: {val} (File Modified: {dt})")
    except Exception:
        pass
