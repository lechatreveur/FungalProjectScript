#!/usr/bin/env python3
"""Remove terminal failures from a manifest and write per-movie retry ID files."""

import argparse
import collections
import shutil
import subprocess
from pathlib import Path

FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def chunks(values, size=200):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = [
        line.split("\t")
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    states = {}
    job_ids = [row[2] for row in rows]
    for group in chunks(job_ids):
        result = subprocess.run(
            ["sacct", "-n", "-X", "-j", ",".join(group), "-o", "JobIDRaw,State"],
            check=True,
            capture_output=True,
            text=True,
        )
        requested = set(group)
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] in requested:
                states[fields[0]] = fields[1].split("+", 1)[0]

    missing = [job_id for job_id in job_ids if job_id not in states]
    if missing:
        raise RuntimeError(f"Accounting is missing {len(missing)} job IDs.")
    unfinished = [
        job_id
        for job_id, state in states.items()
        if state != "COMPLETED" and state not in FAILURE_STATES
    ]
    if unfinished:
        raise RuntimeError(f"{len(unfinished)} jobs are not terminal.")

    failed_rows = [row for row in rows if states[row[2]] in FAILURE_STATES]
    kept_rows = [row for row in rows if states[row[2]] not in FAILURE_STATES]
    retry_by_movie = collections.defaultdict(list)
    for movie, cell_id, _job_id in failed_rows:
        retry_by_movie[movie].append(int(cell_id))

    print(
        f"Plan: manifest={len(rows)} completed={len(kept_rows)} "
        f"retry={len(failed_rows)} movies={len(retry_by_movie)}"
    )
    if not args.apply:
        return

    backup = args.manifest.with_name("slurm_job_ids.before_failed_retry.tsv")
    backup_index = 2
    while backup.exists():
        backup = args.manifest.with_name(
            f"slurm_job_ids.before_failed_retry_{backup_index}.tsv"
        )
        backup_index += 1
    shutil.copy2(args.manifest, backup)
    args.manifest.write_text(
        "".join("\t".join(row) + "\n" for row in kept_rows),
        encoding="utf-8",
    )
    for old_retry_file in args.manifest.parent.glob("3_*/retry_failed_cell_ids.txt"):
        old_retry_file.unlink()
    for movie, cell_ids in retry_by_movie.items():
        retry_path = args.manifest.parent / movie / "retry_failed_cell_ids.txt"
        retry_path.write_text(
            "".join(f"{cell_id}\n" for cell_id in sorted(cell_ids)),
            encoding="utf-8",
        )
    print(f"Applied: backup={backup}")


if __name__ == "__main__":
    main()
