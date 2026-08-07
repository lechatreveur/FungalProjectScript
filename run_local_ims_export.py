#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

# Path to the existing export script
EXPORT_SCRIPT = str(Path(__file__).with_name("batch_ims_export.py"))
PYTHON_EXE = str(Path(__file__).with_name("ims_env") / "bin" / "python3")

def run_export(working_dir, include_patterns, exclude_patterns, list_only=False):
    print(f"\n🚀 Starting export for: {working_dir}")
    command = [PYTHON_EXE, EXPORT_SCRIPT, working_dir]
    for pattern in include_patterns:
        command.extend(["--include", pattern])
    for pattern in exclude_patterns:
        command.extend(["--exclude", pattern])
    if list_only:
        command.append("--list-only")

    subprocess.run(command, check=True)
    print(f"✅ Finished export for: {working_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export selected IMS movies from experiments on X10 Pro."
    )
    parser.add_argument("movie_folders", nargs="+", help="Experiment folder name(s)")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Filename glob to include; repeat for multiple patterns (default: *.ims)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Filename glob to exclude; repeat for multiple patterns",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print selected IMS files without exporting them",
    )
    args = parser.parse_args()

    for movie in args.movie_folders:
        wd = f"/Volumes/X10 Pro/Movies/{movie}"
        if os.path.isdir(wd):
            run_export(wd, args.include, args.exclude, args.list_only)
        else:
            print(f"⚠️ Warning: Directory not found: {wd}")
