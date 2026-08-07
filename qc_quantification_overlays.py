#!/usr/bin/env python3
"""Create early/middle/late mask-overlay contact sheets for selected cells."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from make_population_movie import _load_csv_time_to_rle, decode_rle_any


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    low, high = np.percentile(frame, (1.0, 99.5))
    if high <= low:
        high = low + 1
    image = np.clip((frame.astype(np.float32) - low) * 255 / (high - low), 0, 255)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def square_crop(image, mask, margin=50):
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return image, mask
    y0, y1 = max(0, rows.min() - margin), min(mask.shape[0], rows.max() + margin + 1)
    x0, x1 = max(0, cols.min() - margin), min(mask.shape[1], cols.max() + margin + 1)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    y1, x1 = min(mask.shape[0], y0 + side), min(mask.shape[1], x0 + side)
    y0, x0 = max(0, y1 - side), max(0, x1 - side)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-root", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True, help="movie:cell:channel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tile_size = 360
    times_fraction = (0.0, 0.5, 1.0)
    canvas = np.full(
        (len(args.sample) * tile_size, len(times_fraction) * tile_size, 3),
        24,
        dtype=np.uint8,
    )

    for row_index, specification in enumerate(args.sample):
        movie, cell_text, channel = specification.split(":")
        cell_id = int(cell_text)
        film = args.movie_root / movie
        masks_path = film / f"TrackedCells_{movie}" / f"cell_{cell_id}_masks.csv"
        table = pd.read_csv(masks_path)
        rle_map = _load_csv_time_to_rle(masks_path, f"rle_{channel}")
        times = sorted(rle_map)
        selected = [times[round(fraction * (len(times) - 1))] for fraction in times_fraction]

        for column_index, time_point in enumerate(selected):
            frame_path = film / f"Frames_{movie}" / f"{movie}_t_{time_point:03d}_c_0.tif"
            frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise FileNotFoundError(frame_path)
            mask = decode_rle_any(rle_map[time_point], frame.shape[0], frame.shape[1])
            image = normalize_frame(frame)
            image, mask = square_crop(image, mask)
            overlay = image.copy()
            overlay[mask] = (
                0.35 * overlay[mask] + 0.65 * np.array([40, 220, 40])
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            tile = cv2.resize(overlay, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

            record = table.loc[table["time_point"] == time_point].iloc[0]
            overlap = record.get(f"overlap_score_{channel}", np.nan)
            area = record.get(f"area_{channel}", np.nan)
            selector = str(record.get(f"selector_mode_{channel}", ""))
            label = f"{movie} cell {cell_id}  t={time_point}"
            metric = f"area={area:.0f} overlap={overlap:.3f} {selector}"
            cv2.rectangle(tile, (0, 0), (tile_size, 48), (0, 0, 0), -1)
            cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            cv2.putText(tile, metric, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

            y0 = row_index * tile_size
            x0 = column_index * tile_size
            canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Could not write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
