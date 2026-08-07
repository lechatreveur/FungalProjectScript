#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleanup_render_caches.py

Deletes the tracking_corrector web tool's on-disk render caches:
  <exp>/<film>/CellCrops_<film>/       (per cell/frame/channel crop JPEGs)
  <exp>/<film>/PopulationFrames_<film>/ (per-frame population overlay JPEGs)

Both are pure caches written by SingleCellQuantificationHPC/tracking_corrector/
services/frames_service.py. Nothing else reads them, and both are regenerated
automatically (and losslessly, from the real Frames_/Masks_/TrackedCells_ data)
the next time someone opens that cell or frame in the tool. Deleting them is
always safe.

Run this directly on the Mac (not through a sandboxed mount) for speed and to
avoid the external-drive relay issues that make bulk deletes unreliable there.

Usage:
    python3 cleanup_render_caches.py --dry-run     # see what would be removed
    python3 cleanup_render_caches.py                # actually delete

Prints a progress heartbeat every few seconds (and every N files) while
scanning/deleting each cache folder, since some of these folders hold tens of
thousands of files and would otherwise look hung for minutes at a time.

By default scans --movie-root (i.e. the OLD per-film cache locations under
/Volumes/X10 Pro/Movies/<exp>/<film>/{CellCrops_,PopulationFrames_}<film> --
this is where every cache lives up through this fix and is what you want to
run today).

Going forward, new caches are written to a single consolidated folder at
/Volumes/X10 Pro/_tracking_corrector_cache instead (see config.py's
cache_root). If you ever want to clear THAT location later, point
--movie-root at it directly, e.g.:
    python3 cleanup_render_caches.py --movie-root "/Volumes/X10 Pro/_tracking_corrector_cache"

Pass --root to scan a single arbitrary directory instead of the configured
experiment list.
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

HEARTBEAT_EVERY_FILES = 2000
HEARTBEAT_EVERY_SECS = 3.0


def get_configured_experiments(config_yaml_path: Path) -> list[str]:
    if yaml is None or not config_yaml_path.exists():
        return []
    with open(config_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list((data.get("experiments") or {}).keys())


def find_cache_dirs(exp_root: Path):
    """Yield (cache_dir, kind) for every CellCrops_*/PopulationFrames_* dir
    one level under each film folder in exp_root."""
    if not exp_root.is_dir():
        return
    for film_dir in sorted(exp_root.iterdir()):
        if not film_dir.is_dir():
            continue
        film = film_dir.name
        for kind, dirname in (
            ("CellCrops", f"CellCrops_{film}"),
            ("PopulationFrames", f"PopulationFrames_{film}"),
        ):
            cand = film_dir / dirname
            if cand.is_dir():
                yield cand, kind


class Heartbeat:
    """Prints '...N files (X MB), Ys elapsed' periodically so a long scan/delete
    never looks stalled, without spamming a line per file."""
    def __init__(self, label: str):
        self.label = label
        self.start = time.monotonic()
        self.last_print = self.start
        self.n = 0
        self.n_bytes = 0

    def tick(self, size: int = 0):
        self.n += 1
        self.n_bytes += size
        now = time.monotonic()
        if self.n % HEARTBEAT_EVERY_FILES == 0 or (now - self.last_print) >= HEARTBEAT_EVERY_SECS:
            elapsed = now - self.start
            print(f"    ...{self.label}: {self.n} files, {self.n_bytes/1e6:.1f} MB, {elapsed:.0f}s elapsed",
                  flush=True)
            self.last_print = now

    def finish(self):
        elapsed = time.monotonic() - self.start
        print(f"    {self.label} done: {self.n} files, {self.n_bytes/1e6:.1f} MB, {elapsed:.1f}s",
              flush=True)


def scan_cache_dir(cache_dir: Path):
    """Single-pass count + size (one stat() per file, not two)."""
    hb = Heartbeat("scanning")
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            hb.tick(size)
    hb.finish()
    return hb.n, hb.n_bytes


def delete_cache_dir(cache_dir: Path):
    """Bottom-up delete with live progress, instead of an opaque shutil.rmtree()
    that gives no feedback for minutes on a 30k+ file folder."""
    hb = Heartbeat("deleting")
    for dirpath, dirnames, filenames in os.walk(cache_dir, topdown=False):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0
            try:
                os.remove(fp)
                hb.tick(size)
            except OSError as e:
                print(f"    [warn] could not remove {fp}: {e}", flush=True)
        for dn in dirnames:
            dp = os.path.join(dirpath, dn)
            try:
                os.rmdir(dp)
            except OSError:
                pass
    hb.finish()
    try:
        os.rmdir(cache_dir)
    except OSError:
        pass
    return hb.n, hb.n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie-root", default="/Volumes/X10 Pro/Movies",
                     help="Base movie root (default: %(default)s)")
    ap.add_argument("--root", default=None,
                     help="Scan this single directory instead of the configured experiment list")
    ap.add_argument("--dry-run", action="store_true", help="List what would be deleted, delete nothing")
    args = ap.parse_args()

    movie_root = Path(args.movie_root)
    config_yaml = Path(__file__).parent / "config.yaml"

    if args.root:
        exp_roots = [Path(args.root)]
    else:
        exp_ids = get_configured_experiments(config_yaml)
        if not exp_ids:
            print(f"[warn] Could not read experiment list from {config_yaml}; "
                  f"pass --root explicitly. Falling back to nothing.")
            return 1
        exp_roots = [movie_root / exp_id for exp_id in exp_ids]

    total_files = 0
    total_bytes = 0
    total_dirs = 0

    for exp_root in exp_roots:
        if not exp_root.is_dir():
            print(f"[skip] {exp_root} (not found)")
            continue
        print(f"=== {exp_root.name} ===", flush=True)
        for cache_dir, kind in find_cache_dirs(exp_root):
            print(f"  [{kind:16s}] {cache_dir}", flush=True)
            if args.dry_run:
                n_files, n_bytes = scan_cache_dir(cache_dir)
            else:
                n_files, n_bytes = delete_cache_dir(cache_dir)
            total_files += n_files
            total_bytes += n_bytes
            total_dirs += 1

    print()
    print(f"{'Would remove' if args.dry_run else 'Removed'}: "
          f"{total_dirs} cache directories, {total_files} files, {total_bytes/1e9:.2f} GB (content size)")
    if args.dry_run:
        print("Re-run without --dry-run to actually delete.")
    else:
        print("Note: actual space freed on a large-block-size filesystem (X10 Pro measures")
        print("512KB-1MB per file) is likely much higher than the content size shown above,")
        print("since each small file also frees its full allocation block, not just its byte count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
