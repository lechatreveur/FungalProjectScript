#!/usr/bin/env python3
"""Numerical QC for selected tracked/quantified cells."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_stats(values):
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"finite_fraction": 0.0}
    return {
        "finite_fraction": float(finite.size / max(array.size, 1)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def step_ratio(values):
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    ratios = []
    for previous, current in zip(array[:-1], array[1:]):
        if np.isfinite(previous) and np.isfinite(current) and previous > 0 and current > 0:
            ratios.append(max(previous / current, current / previous))
    if not ratios:
        return {"p95": None, "max": None}
    return {
        "p95": float(np.percentile(ratios, 95)),
        "max": float(np.max(ratios)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-root", type=Path, required=True)
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        help="movie:cell_id:channel, where channel is bf or gfp",
    )
    args = parser.parse_args()

    reports = []
    for specification in args.sample:
        movie, cell_text, channel = specification.split(":")
        cell_id = int(cell_text)
        film = args.movie_root / movie
        tracked = film / f"TrackedCells_{movie}"
        frames = sorted((film / f"Frames_{movie}").glob(f"{movie}_t_???_c_0.tif"))
        masks = pd.read_csv(tracked / f"cell_{cell_id}_masks.csv")
        data_path = tracked / f"cell_{cell_id}_data.csv"
        data = pd.read_csv(data_path) if data_path.stat().st_size > 1 else pd.DataFrame()

        expected_times = set(range(len(frames)))
        mask_times = set(pd.to_numeric(masks["time_point"], errors="coerce").dropna().astype(int))
        data_times = (
            set(pd.to_numeric(data["time_point"], errors="coerce").dropna().astype(int))
            if not data.empty
            else set()
        )
        parent = pd.DataFrame()
        if not data.empty:
            parent = data[data["cell_id"].astype(str) == str(cell_id)].sort_values("time_point")

        suffix = channel
        area_col = f"area_{suffix}"
        overlap_col = f"overlap_score_{suffix}"
        jump_col = f"huge_jump_rejected_{suffix}"
        border_col = f"touches_border_{suffix}"
        selector_col = f"selector_mode_{suffix}"
        source_col = f"source_{suffix}"

        numeric_columns = [
            name
            for name in ("cell_length", "cell_area", "nu_dis", "nu_int", "cyt_int")
            if name in parent
        ]
        reports.append(
            {
                "movie": movie,
                "cell_id": cell_id,
                "channel": channel,
                "expected_frames": len(frames),
                "mask_rows": len(masks),
                "mask_missing_times": sorted(expected_times - mask_times),
                "data_rows": len(data),
                "data_unique_times": len(data_times),
                "data_missing_times": sorted(expected_times - data_times),
                "parent_rows": len(parent),
                "mask_area": safe_stats(masks[area_col]),
                "mask_area_step_ratio": step_ratio(masks[area_col]),
                "overlap": safe_stats(masks[overlap_col]),
                "rejected_jumps": int(
                    masks[jump_col].astype(str).str.lower().isin(("true", "1")).sum()
                ),
                "border_frames": int(
                    masks[border_col].astype(str).str.lower().isin(("true", "1")).sum()
                ),
                "selector_counts": masks[selector_col].value_counts(dropna=False).to_dict(),
                "source_counts": masks[source_col].value_counts(dropna=False).to_dict(),
                "parent_measurements": {
                    column: safe_stats(parent[column])
                    for column in numeric_columns
                },
                "parent_length_step_ratio": (
                    step_ratio(parent["cell_length"]) if "cell_length" in parent else {}
                ),
                "parent_area_step_ratio": (
                    step_ratio(parent["cell_area"]) if "cell_area" in parent else {}
                ),
            }
        )

    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
