#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backup_and_prune_experiment.py

Three explicit, separate steps for retiring a stale experiment from the local
X10 Pro drive onto the NAS -- deliberately NOT one combined "backup and
delete" command, because this deletes real, irreplaceable microscopy data
(unlike cleanup_render_caches.py, which only ever touches regenerable
caches). Deletion requires a *fresh, passing* verification, enforced by a
marker file -- you cannot skip straight to --prune.

Usage (run all three, in order, on the Mac):
    python3 backup_and_prune_experiment.py 2025_06_25 --backup
    python3 backup_and_prune_experiment.py 2025_06_25 --verify
    python3 backup_and_prune_experiment.py 2025_06_25 --prune --yes-i-verified

--backup   rsyncs the experiment folder to the NAS, excluding render caches
           and macOS junk (those never need to exist on the NAS copy).
--verify   runs a checksum-comparing dry-run rsync local->NAS. If it reports
           ANY files that would still need transferring, verification FAILS
           and nothing further is written. On success, writes a timestamped
           marker recording the comparison.
--prune    refuses unless a --verify marker for this exact experiment exists
           and is less than 24 hours old, AND --yes-i-verified is passed.
           Deletes the local experiment folder only after that check.

rsync must be on PATH (Homebrew's /opt/homebrew/bin/rsync if present, else
whatever `rsync` resolves to).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

EXCLUDES = ["._*", "CellCrops_*", "PopulationFrames_*"]
MARKER_DIR_NAME = ".backup_verified"
MARKER_MAX_AGE_HOURS = 24


def find_rsync() -> str:
    for cand in ("/opt/homebrew/bin/rsync", "/usr/local/bin/rsync", "rsync"):
        if shutil.which(cand) or cand == "rsync":
            return cand
    return "rsync"


def rsync_excludes(cmd: list[str]) -> list[str]:
    for pat in EXCLUDES:
        cmd += ["--exclude", pat]
    return cmd


def do_backup(rsync: str, local_dir: Path, nas_dir: Path) -> int:
    nas_dir.mkdir(parents=True, exist_ok=True)
    cmd = [rsync, "-av", "--progress", "--partial"]
    cmd = rsync_excludes(cmd)
    cmd += [f"{local_dir}/", f"{nas_dir}/"]
    print(f"Running: {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd)


def do_verify(rsync: str, local_dir: Path, nas_dir: Path, marker_dir: Path, exp_name: str) -> int:
    # --checksum forces byte-content comparison (not just size/mtime), --dry-run
    # + -i (itemize-changes) shows exactly what rsync thinks still differs.
    cmd = [rsync, "-avc", "--dry-run", "-i"]
    cmd = rsync_excludes(cmd)
    cmd += [f"{local_dir}/", f"{nas_dir}/"]
    print(f"Running: {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Any itemized line starting with '>' means rsync would still transfer that
    # file -- i.e. NAS copy differs from or is missing that file.
    diff_lines = [ln for ln in result.stdout.splitlines() if ln.startswith(">")]

    if result.returncode != 0:
        print(f"\n[FAIL] rsync exited {result.returncode} -- verification not passed.")
        return 1

    if diff_lines:
        print(f"\n[FAIL] {len(diff_lines)} file(s) differ between local and NAS copy:")
        for ln in diff_lines[:20]:
            print(f"    {ln}")
        if len(diff_lines) > 20:
            print(f"    ... and {len(diff_lines) - 20} more")
        print("\nRe-run --backup to fix, then --verify again.")
        return 1

    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{exp_name}.json"
    marker_path.write_text(json.dumps({
        "experiment": exp_name,
        "verified_at": datetime.now().isoformat(),
        "local_dir": str(local_dir),
        "nas_dir": str(nas_dir),
    }, indent=2))
    print(f"\n[PASS] NAS copy matches local (checksum-verified, excluding caches/junk).")
    print(f"Marker written: {marker_path}")
    return 0


def do_prune(local_dir: Path, marker_dir: Path, exp_name: str, confirmed: bool) -> int:
    marker_path = marker_dir / f"{exp_name}.json"
    if not marker_path.exists():
        print(f"[refused] No verification marker found at {marker_path}.")
        print("Run --verify successfully first.")
        return 1

    marker = json.loads(marker_path.read_text())
    verified_at = datetime.fromisoformat(marker["verified_at"])
    age = datetime.now() - verified_at
    if age > timedelta(hours=MARKER_MAX_AGE_HOURS):
        print(f"[refused] Verification marker is {age} old (max {MARKER_MAX_AGE_HOURS}h). "
              f"Data may have changed since. Re-run --verify.")
        return 1

    if not confirmed:
        print(f"[refused] Verified {age} ago, looks good, but --yes-i-verified was not passed.")
        print(f"This will permanently delete: {local_dir}")
        print("Re-run with --prune --yes-i-verified to actually delete.")
        return 1

    print(f"Deleting {local_dir} (verified {age} ago)...")
    shutil.rmtree(local_dir)
    marker_path.unlink(missing_ok=True)
    print("Done.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="Experiment folder name, e.g. 2025_06_25")
    ap.add_argument("--local-movie-root", default="/Volumes/X10 Pro/Movies")
    ap.add_argument("--nas-movie-root", default="/Volumes/Movies")
    ap.add_argument("--backup", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--yes-i-verified", action="store_true",
                     help="Required in addition to --prune to actually delete")
    args = ap.parse_args()

    if sum([args.backup, args.verify, args.prune]) != 1:
        print("[error] Pass exactly one of --backup, --verify, --prune", file=sys.stderr)
        return 2

    local_dir = Path(args.local_movie_root) / args.experiment
    nas_dir = Path(args.nas_movie_root) / args.experiment
    marker_dir = Path(args.local_movie_root) / MARKER_DIR_NAME

    if not local_dir.is_dir():
        print(f"[error] Not found: {local_dir}", file=sys.stderr)
        return 2

    rsync = find_rsync()

    if args.backup:
        return do_backup(rsync, local_dir, nas_dir)
    if args.verify:
        return do_verify(rsync, local_dir, nas_dir, marker_dir, args.experiment)
    if args.prune:
        return do_prune(local_dir, marker_dir, args.experiment, args.yes_i_verified)


if __name__ == "__main__":
    sys.exit(main())
