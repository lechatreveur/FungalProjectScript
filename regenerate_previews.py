#!/usr/bin/env python3
import argparse
import fnmatch
from pathlib import Path

import cv2
import numpy as np


def read_frame(path: Path) -> np.ndarray:
    """Read TIFFs with OpenCV, which supports the LZW files written by stabilization."""
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Could not read frame: {path}")
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def find_frames_and_range(frame_dir: Path, pattern: str):
    names = sorted(path.name for path in frame_dir.glob(pattern))
    if not names:
        return [], 0.0, 1.0

    samples = []
    for name in names:
        frame = read_frame(frame_dir / name)
        # A regular spatial sample is sufficient for stable preview contrast.
        samples.append(frame[::10, ::10].reshape(-1))
    sampled_pixels = np.concatenate(samples)
    low, high = np.percentile(sampled_pixels, (0.5, 99.5))
    if high <= low:
        high = low + 1.0
    return names, float(low), float(high)


def write_video_from_frames(
    frame_dir: Path,
    names: list[str],
    output: Path,
    fps: int,
    low: float,
    high: float,
):
    first = read_frame(frame_dir / names[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    try:
        scale = 255.0 / (high - low)
        for name in names:
            frame = read_frame(frame_dir / name)
            preview = np.clip((frame.astype(np.float32) - low) * scale, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Regenerate preview MP4s from stabilized TIFF frames.")
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    includes = args.include or ["*"]
    films = sorted(
        path
        for path in args.experiment.iterdir()
        if path.is_dir()
        and any(fnmatch.fnmatch(path.name, pattern) for pattern in includes)
        and not any(fnmatch.fnmatch(path.name, pattern) for pattern in args.exclude)
    )

    completed = 0
    for film_dir in films:
        film = film_dir.name
        frames_dir = film_dir / f"Frames_{film}"
        tracked_dir = film_dir / f"TrackedCells_{film}"
        if not frames_dir.is_dir():
            continue
        tracked_dir.mkdir(exist_ok=True)
        pattern = f"{film}_t_???_c_0.tif"
        names, low, high = find_frames_and_range(frames_dir, pattern)
        if not names:
            raise RuntimeError(f"No frames found for {film}")
        write_video_from_frames(
            frames_dir,
            names,
            tracked_dir / f"{film}_c_0.mp4",
            args.fps,
            low,
            high,
        )
        completed += 1

    if completed != 33:
        raise RuntimeError(f"Expected to regenerate 33 previews; regenerated {completed}.")
    print(f"Regenerated {completed} preview movies.")


if __name__ == "__main__":
    main()
