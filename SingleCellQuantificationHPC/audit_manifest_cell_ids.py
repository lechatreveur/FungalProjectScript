#!/usr/bin/env python3
"""Compare submitted M156 cell IDs with IDs in the instance-labeled masks."""

import argparse
import collections
import json
from pathlib import Path

from skimage.io import imread
from skimage.measure import regionprops

from Cell_tracking_functions import to_labeled_current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=2000)
    args = parser.parse_args()

    submitted: dict[str, list[int]] = collections.defaultdict(list)
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        movie, cell_id, _job_id = line.split("\t")[:3]
        submitted[movie].append(int(cell_id))

    report = {}
    totals = collections.Counter()
    for movie_dir in sorted(
        path
        for path in args.movie_root.iterdir()
        if path.is_dir() and (path.name.startswith("3_BF") or path.name.startswith("3_FL"))
    ):
        movie = movie_dir.name
        cell_ids_file = args.experiment_root / movie / "cell_ids.txt"
        expected_submissions = (
            len(cell_ids_file.read_text(encoding="utf-8").splitlines())
            if cell_ids_file.is_file()
            else None
        )
        submitted_ids = submitted.get(movie, [])
        submission_complete = (
            expected_submissions is not None
            and len(submitted_ids) == expected_submissions
        )

        mask_path = movie_dir / f"Masks_{movie}" / f"{movie}_t_000_c_0_seg.tif"
        labels = to_labeled_current(imread(mask_path))
        correct_ids = {
            int(region.label)
            for region in regionprops(labels)
            if region.area >= args.min_area
        }
        submitted_set = set(submitted_ids)
        invalid = sorted(submitted_set - correct_ids)
        missing = sorted(correct_ids - submitted_set) if submission_complete else []
        duplicates = sorted(
            cell_id
            for cell_id, count in collections.Counter(submitted_ids).items()
            if count > 1
        )

        totals["correct_ids"] += len(correct_ids)
        totals["submitted_ids"] += len(submitted_ids)
        totals["invalid_ids"] += len(invalid)
        totals["missing_ids"] += len(missing)
        totals["duplicate_ids"] += len(duplicates)
        report[movie] = {
            "submission_complete": submission_complete,
            "correct_count": len(correct_ids),
            "submitted_count": len(submitted_ids),
            "invalid_submitted_ids": invalid,
            "missing_correct_ids": missing,
            "duplicate_submitted_ids": duplicates,
        }

    print(json.dumps({"totals": dict(totals), "movies": report}, indent=2))
    if totals["invalid_ids"] or totals["missing_ids"] or totals["duplicate_ids"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
