#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:56:36 2025

@author: user
"""

#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, shutil

def planned_moves(root: Path, apply: bool):
    mask_dirs = []
    for movie_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for m in movie_dir.glob("Masks_*"):
            if m.is_dir():
                mask_dirs.append(m)

    ops, moved = [], 0
    for mdir in mask_dirs:
        for sub in ("GFP_seg", "brightfield_seg"):
            sdir = mdir / sub
            if not sdir.is_dir():
                continue
            for item in sorted(sdir.iterdir()):
                if not item.is_file():
                    continue
                dest = mdir / item.name
                if dest.exists():
                    stem, suffix = dest.stem, dest.suffix
                    tag = "_GFP" if "gfp" in sub.lower() else "_BF" if "brightfield" in sub.lower() else f"_{sub}"
                    if not stem.endswith(tag):
                        candidate = mdir / f"{stem}{tag}{suffix}"
                    else:
                        candidate = mdir / f"{stem}{suffix}"
                    i = 1
                    while candidate.exists():
                        candidate = mdir / f"{stem}{tag}_{i}{suffix}"
                        i += 1
                    dest = candidate
                ops.append((item, dest))
                if apply:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(dest))
                    moved += 1
            # try to delete emptied subdir
            if apply and sdir.exists():
                try:
                    for p in sorted(sdir.rglob("*"), reverse=True):
                        if p.is_dir():
                            try: p.rmdir()
                            except OSError: pass
                    sdir.rmdir()
                except OSError:
                    pass
    return ops, moved

def main():
    ap = argparse.ArgumentParser(description="Flatten masks from GFP_seg/brightfield_seg into Masks_* folders.")
    ap.add_argument("--root", type=Path, default=Path("/Volumes/Movies/2025_09_17"))
    ap.add_argument("--apply", action="store_true", help="Actually perform moves and deletions.")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"Root not found: {args.root}", file=sys.stderr)
        sys.exit(2)

    ops, n = planned_moves(args.root, apply=args.apply)
    if not args.apply:
        print("DRY RUN (no changes made). Planned moves:")
        for src, dst in ops:
            print(f"{src} -> {dst}")
        print(f"\n{len(ops)} files would be moved. Re-run with --apply to make the changes.")
    else:
        print(f"Done. Moved {n} files and removed empty subfolders where possible.")

if __name__ == "__main__":
    main()
