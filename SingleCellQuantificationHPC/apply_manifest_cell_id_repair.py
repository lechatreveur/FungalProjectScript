#!/usr/bin/env python3
"""Apply a reviewed cell-ID audit to an M156 manifest without deleting data."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    incomplete = [
        movie
        for movie, result in audit["movies"].items()
        if not result["submission_complete"]
    ]
    if incomplete:
        raise RuntimeError(f"Submission is incomplete for: {incomplete}")

    invalid_pairs = {
        (movie, int(cell_id))
        for movie, result in audit["movies"].items()
        for cell_id in result["invalid_submitted_ids"]
    }
    missing_by_movie = {
        movie: [int(cell_id) for cell_id in result["missing_correct_ids"]]
        for movie, result in audit["movies"].items()
        if result["missing_correct_ids"]
    }
    rows = [
        line.split("\t")
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept_rows = [
        row for row in rows if (row[0], int(row[1])) not in invalid_pairs
    ]
    removed_rows = len(rows) - len(kept_rows)
    missing_count = sum(len(ids) for ids in missing_by_movie.values())
    print(
        f"Plan: preserve={len(rows)} remove_invalid={removed_rows} "
        f"submit_missing={missing_count} canonical_final={len(kept_rows) + missing_count}"
    )
    if removed_rows != len(invalid_pairs):
        raise RuntimeError(
            f"Expected one manifest row per invalid pair: rows={removed_rows}, "
            f"pairs={len(invalid_pairs)}"
        )
    if not args.apply:
        print("Dry run only; pass --apply to modify the manifest and archive outputs.")
        return

    original_manifest = args.manifest.with_name("slurm_job_ids.initial.tsv")
    if original_manifest.exists():
        raise FileExistsError(original_manifest)
    shutil.copy2(args.manifest, original_manifest)

    archive_root = args.experiment_root / "invalid_cell_outputs"
    moved_outputs = 0
    for movie, cell_id in sorted(invalid_pairs):
        tracked = (
            args.movie_root / movie / f"TrackedCells_{movie}"
        )
        destination = archive_root / movie
        destination.mkdir(parents=True, exist_ok=True)
        for suffix in ("masks.csv", "data.csv"):
            source = tracked / f"cell_{cell_id}_{suffix}"
            if not source.exists():
                continue
            target = destination / source.name
            if target.exists():
                raise FileExistsError(target)
            shutil.move(str(source), str(target))
            moved_outputs += 1

    args.manifest.write_text(
        "".join("\t".join(row) + "\n" for row in kept_rows),
        encoding="utf-8",
    )
    for movie, cell_ids in missing_by_movie.items():
        repair_ids = args.experiment_root / movie / "repair_cell_ids.txt"
        repair_ids.write_text(
            "".join(f"{cell_id}\n" for cell_id in cell_ids),
            encoding="utf-8",
        )
    print(
        f"Applied repair plan: archived_manifest={original_manifest} "
        f"moved_outputs={moved_outputs} repair_movies={len(missing_by_movie)}"
    )


if __name__ == "__main__":
    main()
