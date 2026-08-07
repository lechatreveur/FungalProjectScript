#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_stale_experiments.py

Scans every experiment folder directly under a movie root and reports the
most recent "real" activity timestamp for each -- i.e. ignoring the render
caches (CellCrops_*/PopulationFrames_*) and macOS AppleDouble junk (._*),
since those get touched just by browsing an old experiment in the QC tool or
by Finder, not by doing actual analysis work.

Flags any experiment whose real last-touched date is older than --stale-days
(default 60 / ~2 months) as a backup-and-prune candidate. Names in
--keep (comma-separated, case-insensitive substring match against the folder
name) are always reported but never flagged stale, regardless of their
computed date -- use this for experiments you know are still active.

Run this directly on the Mac, not through a sandboxed mount -- it walks a lot
of files and needs to be fast and reliable. Prints a progress heartbeat per
experiment since some of these folders are large.

Usage:
    python3 find_stale_experiments.py
    python3 find_stale_experiments.py --stale-days 60 --keep "2025_09_17,2026_07_16_M156"
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

IGNORED_DIR_PREFIXES = ("CellCrops_", "PopulationFrames_", "._")
IGNORED_FILE_PREFIX = "._"

HEARTBEAT_EVERY_FILES = 5000
HEARTBEAT_EVERY_SECS = 3.0


def latest_real_mtime(exp_dir: Path):
    """Walk exp_dir, return (max_mtime, files_scanned), skipping cache dirs
    and AppleDouble files so they don't fake-freshen the 'last touched' signal."""
    best = 0.0
    scanned = 0
    start = time.monotonic()
    last_print = start
    for dirpath, dirnames, filenames in os.walk(exp_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(IGNORED_DIR_PREFIXES)]
        for fn in filenames:
            if fn.startswith(IGNORED_FILE_PREFIX):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                mtime = os.path.getmtime(fp)
            except OSError:
                continue
            if mtime > best:
                best = mtime
            scanned += 1
            now = time.monotonic()
            if scanned % HEARTBEAT_EVERY_FILES == 0 or (now - last_print) >= HEARTBEAT_EVERY_SECS:
                print(f"    ...scanning {exp_dir.name}: {scanned} files, {now - start:.0f}s elapsed", flush=True)
                last_print = now
    return best, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie-root", default="/Volumes/X10 Pro/Movies")
    ap.add_argument("--stale-days", type=int, default=60)
    ap.add_argument("--keep", default="",
                     help="Comma-separated substrings of experiment folder names to always keep, "
                          "regardless of computed last-touched date (e.g. active experiments)")
    args = ap.parse_args()

    movie_root = Path(args.movie_root)
    keep_substrings = [s.strip().lower() for s in args.keep.split(",") if s.strip()]
    threshold = datetime.now() - timedelta(days=args.stale_days)

    if not movie_root.is_dir():
        print(f"[error] movie root not found: {movie_root}", file=sys.stderr)
        return 1

    exp_dirs = sorted(d for d in movie_root.iterdir() if d.is_dir())
    print(f"Scanning {len(exp_dirs)} experiment folders under {movie_root}")
    print(f"Stale threshold: last real activity before {threshold.date()} ({args.stale_days} days ago)")
    if keep_substrings:
        print(f"Always-keep list: {keep_substrings}")
    print()

    rows = []
    for exp_dir in exp_dirs:
        print(f"=== {exp_dir.name} ===", flush=True)
        best_mtime, n_files = latest_real_mtime(exp_dir)
        last_touched = datetime.fromtimestamp(best_mtime) if best_mtime else None
        age_days = (datetime.now() - last_touched).days if last_touched else None
        kept_by_policy = any(sub in exp_dir.name.lower() for sub in keep_substrings)
        is_stale = (age_days is not None and age_days > args.stale_days) and not kept_by_policy
        rows.append((exp_dir.name, n_files, last_touched, age_days, kept_by_policy, is_stale))
        note = " [KEEP -- policy override]" if kept_by_policy else ""
        lt_str = last_touched.strftime("%Y-%m-%d") if last_touched else "unknown"
        print(f"    -> {n_files} real files, last touched {lt_str} "
              f"({age_days if age_days is not None else '?'} days ago){note}", flush=True)

    print()
    print("=" * 70)
    print(f"{'Experiment':<28} {'Last touched':<14} {'Age (d)':>8}  Status")
    print("-" * 70)
    for name, _n, last_touched, age_days, kept, stale in rows:
        lt_str = last_touched.strftime("%Y-%m-%d") if last_touched else "unknown"
        status = "STALE - backup+prune candidate" if stale else ("ACTIVE (kept by policy)" if kept else "active")
        print(f"{name:<28} {lt_str:<14} {str(age_days) if age_days is not None else '?':>8}  {status}")

    stale_names = [name for name, *_r, stale in rows if stale]
    print()
    if stale_names:
        print(f"{len(stale_names)} candidate(s) for backup-and-prune: {', '.join(stale_names)}")
        print("Next step: for each, run backup_and_prune_experiment.py <name> --backup,")
        print("then --verify, and only then --prune.")
    else:
        print("No stale candidates found under the current threshold/keep list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
