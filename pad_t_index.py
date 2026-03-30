#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:22:23 2025

@author: user
"""

#!/usr/bin/env python3
from pathlib import Path
import re

# Set to True to actually rename files
DO_RENAME = True

dirs = [
    Path("/Volumes/Movies/2025_07_23_M77/A14_25oC_2hr_1/Masks_A14_25oC_2hr_1/GFP_seg/"),
    Path("/Volumes/Movies/2025_07_23_M77/A14_25oC_2hr_1/Masks_A14_25oC_2hr_1/brightfield_seg/"),
    Path("/Volumes/Movies/2025_07_23_M77/A14_25oC_2hr_1/Frames_A14_25oC_2hr_1/"),
]

# Match '_t_' followed by 1–2 digits when the next character is an underscore
t_index_rx = re.compile(r'(_t_)(\d{1,2})(?=_)')

def pad_t_index(name: str) -> str:
    # Only change if there's a 1–2 digit t-index; 3+ digits are left as-is
    def repl(m):
        return m.group(1) + m.group(2).zfill(3)
    return t_index_rx.sub(repl, name)

changed = 0
skipped = 0
conflicts = 0

for d in dirs:
    if not d.exists():
        print(f"[WARN] Missing directory: {d}")
        continue
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue

        new_name = pad_t_index(p.name)
        if new_name == p.name:
            skipped += 1
            continue

        dest = p.with_name(new_name)
        if dest.exists():
            print(f"[CONFLICT] {p.name} -> {new_name} (target already exists)")
            conflicts += 1
            continue

        if DO_RENAME:
            p.rename(dest)
            print(f"[RENAMED]  {p.name} -> {new_name}")
        else:
            print(f"[DRY-RUN]  {p.name} -> {new_name}")
        changed += 1

print(f"\nSummary: to-change={changed}, skipped={skipped}, conflicts={conflicts}")
if not DO_RENAME:
    print("Dry-run only. Set DO_RENAME = True to apply.")
