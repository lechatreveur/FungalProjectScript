#!/usr/bin/env python3
"""Summarize tracking and quantification completeness for one movie."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def truthy_sum(series: pd.Series) -> int:
    if series.dtype == bool:
        return int(series.sum())
    return int(
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
        .sum()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-root", type=Path, required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--cell-ids-file", type=Path, required=True)
    parser.add_argument("--channel", choices=("bf", "gfp"), required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()

    tracked = (
        args.movie_root
        / args.file_name
        / f"TrackedCells_{args.file_name}"
    )
    cell_ids = [
        int(line.strip())
        for line in args.cell_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suffix = args.channel
    expected_times = set(range(args.expected_frames))

    missing_masks = []
    bad_frame_coverage = []
    missing_data = []
    empty_data = []
    data_rows = {}
    cells_touching_border = []
    cells_with_rejected_jumps = []
    rejected_jump_count = 0
    overlap_values = []
    area_step_ratios = []
    source_counts: Counter[str] = Counter()
    selector_counts: Counter[str] = Counter()

    for cell_id in cell_ids:
        masks_path = tracked / f"cell_{cell_id}_masks.csv"
        data_path = tracked / f"cell_{cell_id}_data.csv"

        if not masks_path.is_file() or masks_path.stat().st_size == 0:
            missing_masks.append(cell_id)
        else:
            masks = pd.read_csv(masks_path)
            times = set(pd.to_numeric(masks["time_point"], errors="coerce").dropna().astype(int))
            if len(masks) != args.expected_frames or times != expected_times:
                bad_frame_coverage.append(
                    {"cell_id": cell_id, "rows": len(masks), "unique_times": len(times)}
                )

            border_col = f"touches_border_{suffix}"
            if border_col in masks and truthy_sum(masks[border_col]):
                cells_touching_border.append(cell_id)

            jump_col = f"huge_jump_rejected_{suffix}"
            if jump_col in masks:
                jumps = truthy_sum(masks[jump_col])
                rejected_jump_count += jumps
                if jumps:
                    cells_with_rejected_jumps.append(cell_id)

            overlap_col = f"overlap_score_{suffix}"
            if overlap_col in masks:
                overlap_values.extend(
                    pd.to_numeric(masks[overlap_col], errors="coerce").dropna().tolist()
                )

            area_col = f"area_{suffix}"
            if area_col in masks:
                area = pd.to_numeric(masks[area_col], errors="coerce").to_numpy(float)
                valid = np.isfinite(area) & (area > 0)
                for previous, current in zip(area[:-1], area[1:]):
                    if previous > 0 and current > 0 and np.isfinite(previous + current):
                        area_step_ratios.append(max(current / previous, previous / current))

            source_col = f"source_{suffix}"
            if source_col in masks:
                source_counts.update(masks[source_col].dropna().astype(str))

            selector_col = f"selector_mode_{suffix}"
            if selector_col in masks:
                selector_counts.update(masks[selector_col].dropna().astype(str))

        if not data_path.is_file():
            missing_data.append(cell_id)
        elif data_path.stat().st_size <= 1:
            empty_data.append(cell_id)
            data_rows[str(cell_id)] = 0
        else:
            try:
                data_rows[str(cell_id)] = len(pd.read_csv(data_path))
            except pd.errors.EmptyDataError:
                empty_data.append(cell_id)
                data_rows[str(cell_id)] = 0

    overlaps = np.asarray(overlap_values, dtype=float)
    ratios = np.asarray(area_step_ratios, dtype=float)
    nonempty_rows = [rows for rows in data_rows.values() if rows > 0]
    summary = {
        "movie": args.file_name,
        "eligible_cells": len(cell_ids),
        "expected_frames": args.expected_frames,
        "complete_mask_csvs": len(cell_ids) - len(missing_masks),
        "bad_frame_coverage": bad_frame_coverage,
        "missing_mask_cells": missing_masks,
        "nonempty_data_csvs": len(nonempty_rows),
        "empty_data_cells": empty_data,
        "missing_data_cells": missing_data,
        "data_rows_min_nonempty": min(nonempty_rows) if nonempty_rows else 0,
        "data_rows_max": max(nonempty_rows) if nonempty_rows else 0,
        "cells_touching_border": cells_touching_border,
        "cells_with_rejected_jumps": cells_with_rejected_jumps,
        "rejected_jump_count": rejected_jump_count,
        "overlap_score_median": float(np.nanmedian(overlaps)) if overlaps.size else None,
        "overlap_score_p05": float(np.nanpercentile(overlaps, 5)) if overlaps.size else None,
        "overlap_score_min": float(np.nanmin(overlaps)) if overlaps.size else None,
        "area_step_ratio_p95": float(np.nanpercentile(ratios, 95)) if ratios.size else None,
        "area_step_ratio_max": float(np.nanmax(ratios)) if ratios.size else None,
        "source_counts": dict(source_counts),
        "selector_counts": dict(selector_counts),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if missing_masks or bad_frame_coverage or missing_data:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
