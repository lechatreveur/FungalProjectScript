#!/usr/bin/env python3
"""Plot selected BF/GFP quantification time courses."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-root", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True, help="movie:cell:channel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(len(args.sample), 3, figsize=(12, 2.7 * len(args.sample)))
    for row, specification in enumerate(args.sample):
        movie, cell_text, channel = specification.split(":")
        cell_id = int(cell_text)
        data_path = (
            args.movie_root
            / movie
            / f"TrackedCells_{movie}"
            / f"cell_{cell_id}_data.csv"
        )
        data = pd.read_csv(data_path)
        parent = data[data["cell_id"].astype(str) == str(cell_id)].sort_values("time_point")
        time = parent["time_point"]
        label = f"{movie} cell {cell_id} ({channel})"

        if channel == "bf":
            axes[row, 0].plot(time, parent["pattern_score_norm"], marker="o", ms=2)
            axes[row, 0].set_ylabel("pattern score")
            axes[row, 1].plot(time, parent["pattern_score_norm_intensity"], marker="o", ms=2)
            axes[row, 1].set_ylabel("intensity score")
            axes[row, 2].plot(time, parent["pattern_n_length_px"], marker="o", ms=2)
            axes[row, 2].set_ylabel("pattern length")
        else:
            axes[row, 0].plot(time, parent["cell_area"], label="area")
            length_scaled = parent["cell_length"] * (
                parent["cell_area"].median() / parent["cell_length"].median()
            )
            axes[row, 0].plot(time, length_scaled, label="length (scaled)", alpha=0.8)
            axes[row, 0].set_ylabel("area / scaled length")
            axes[row, 0].legend(fontsize=7)
            axes[row, 1].plot(time, parent["nu_int"], label="nucleus")
            axes[row, 1].plot(time, parent["cyt_int"], label="cytosol")
            axes[row, 1].set_ylabel("intensity")
            axes[row, 1].legend(fontsize=7)
            axes[row, 2].plot(time, parent["nu_dis"])
            axes[row, 2].set_ylabel("nucleus distance")

        axes[row, 0].set_title(label, loc="left", fontsize=9)
        for axis in axes[row]:
            axis.grid(alpha=0.25)
            axis.set_xlabel("time")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
