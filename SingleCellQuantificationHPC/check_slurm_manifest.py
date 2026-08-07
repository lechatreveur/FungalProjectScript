#!/usr/bin/env python3
import argparse
import collections
import subprocess
import sys
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--by-movie",
        action="store_true",
        help="Print per-movie SLURM state counts.",
    )
    args = parser.parse_args()

    rows = [line.split("\t") for line in args.manifest.read_text().splitlines() if line.strip()]
    job_ids = [row[2] for row in rows]
    states = {}

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
    counts = collections.Counter(states.values())
    print("SLURM states:", " ".join(f"{state}={count}" for state, count in sorted(counts.items())))
    if args.by_movie:
        movie_states = collections.defaultdict(collections.Counter)
        for row in rows:
            movie, _cell_id, job_id = row[:3]
            movie_states[movie][states.get(job_id, "ACCOUNTING_MISSING")] += 1
        print("Per-movie states:")
        for movie in sorted(movie_states):
            state_text = " ".join(
                f"{state}={count}"
                for state, count in sorted(movie_states[movie].items())
            )
            print(f"  {movie}\t{state_text}")
    if missing:
        print(f"Accounting records not ready for {len(missing)} job(s).", file=sys.stderr)
        return 2

    failures = {state: count for state, count in counts.items() if state in FAILURE_STATES}
    unfinished = {
        state: count
        for state, count in counts.items()
        if state != "COMPLETED" and state not in FAILURE_STATES
    }
    if failures:
        print("Failed jobs:", file=sys.stderr)
        for row in rows:
            movie, cell_id, job_id = row[:3]
            state = states.get(job_id)
            if state in FAILURE_STATES:
                print(
                    f"  movie={movie} cell_id={cell_id} job_id={job_id} state={state}",
                    file=sys.stderr,
                )
        print(f"Terminal SLURM failures: {failures}", file=sys.stderr)
        return 1
    if unfinished:
        print(f"Jobs not terminal yet: {unfinished}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
