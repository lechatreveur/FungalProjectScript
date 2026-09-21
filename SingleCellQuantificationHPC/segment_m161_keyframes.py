#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_m161_keyframes.py

Stage 1/2 keyframes for the M161 experiment (2026_09_03, `NeonG_YES_1`).

M161 is the replicate control for M162: same strain, same medium, same
acquisition settings (101 x 12 s, 350/120 ms, laser 2 at intensity 5),
different session. It is therefore compared against M162 directly, so the
**segmentation must be identical between the two** - a difference in masks
would otherwise show up as a difference in features.

For that reason this defaults to the same checkpoint M162 used,
`cpsam_20260909_neongreen_m160_resumed`, and to the same diameter (80.0),
flow_threshold (0.4) and cellprob_threshold (0.0). Do not "upgrade" the model
here without re-running M162 on the same one.

Keyframe scheme per film (as M162):
- FL films (101 frames): t = 0, 50, 100
- BF films (41 frames):  t = 0, 20, 40

Outputs:
    <exp_dir>/<film>/Masks_<film>/<film>_t_{t:03d}_c_0_seg.tif

Unlike the M162 script this takes `--films`, because the FL1-only comparison
needs four films rather than all 44. Default is every film in the experiment.

Requires the cellpose-sam environment:
    KMP_DUPLICATE_LIB_OK=TRUE \\
    /Users/user/miniforge3/envs/cellpose-sam/bin/python3 \\
        SingleCellQuantificationHPC/segment_m161_keyframes.py --films ...
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc
import sys
import argparse
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
import torch

try:
    from cellpose import models
except ImportError:
    print("Cellpose not found. Run this with the cellpose-sam environment:\n"
          "  /Users/user/miniforge3/envs/cellpose-sam/bin/python3")
    sys.exit(1)

# The checkpoint M162 was segmented with. Held fixed deliberately - see docstring.
CUSTOM_MODEL_NAME = "cpsam_20260909_neongreen_m160_resumed"
DEFAULT_MODEL_PATH = Path.home() / ".cellpose" / "models" / CUSTOM_MODEL_NAME
# Data stays on the external SSD, never the system disk (P4).
DEFAULT_EXP_DIR = Path("/Volumes/X10 Pro/Movies/2026_09_03_M161")

FL_FILMS = [f"FL{i}" for i in range(1, 7)]   # M161 has FL1-FL6
BF_FILMS = [f"BF{i}" for i in range(1, 6)]   # and BF1-BF5
FIELDS = [f"F{i}" for i in range(4)]         # 4 fields, unlike M160's 3


def get_keyframes_for_film(film_name: str) -> list[int]:
    if "FL" in film_name:
        return [0, 50, 100]
    if "BF" in film_name:
        return [0, 20, 40]
    return [0]


def all_films() -> list[str]:
    """Chronological order: FL1, BF1, FL2, BF2, ... per field."""
    films = []
    for field in FIELDS:
        for i in range(len(FL_FILMS)):
            films.append(f"NeonG_YES_1_{FL_FILMS[i]}_{field}")
            if i < len(BF_FILMS):
                films.append(f"NeonG_YES_1_{BF_FILMS[i]}_{field}")
    return films


def segment_film_keyframes(film_name, exp_dir, model, diameter=80.0):
    film_dir = exp_dir / film_name
    frames_dir = film_dir / f"Frames_{film_name}"
    masks_dir = film_dir / f"Masks_{film_name}"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (film_dir / f"TrackedCells_{film_name}").mkdir(parents=True, exist_ok=True)

    k_times = get_keyframes_for_film(film_name)
    print(f"\n[{film_name}] Segmenting {len(k_times)} keyframes: {k_times}")

    for t in k_times:
        frame_path = frames_dir / f"{film_name}_t_{t:03d}_c_0.tif"
        seg_path = masks_dir / f"{film_name}_t_{t:03d}_c_0_seg.tif"

        if seg_path.exists():
            print(f"  t={t}: already segmented -> {seg_path.name}")
            continue
        if not frame_path.exists():
            print(f"  [ERROR] Frame file not found: {frame_path}")
            continue

        img = imread(str(frame_path))
        eval_diameter = None if (diameter is None or diameter <= 0) else float(diameter)
        out = model.eval(img, channel_axis=None, diameter=eval_diameter,
                         flow_threshold=0.4, cellprob_threshold=0.0)
        masks = out[0].astype(np.uint16)
        imwrite(str(seg_path), masks)

        n_cells = len(np.unique(masks)) - (1 if 0 in masks else 0)
        print(f"  t={t:03d}: segmented {n_cells} cells -> {seg_path.name}", flush=True)

        del img, masks, out
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description="Segment keyframes for M161 films")
    ap.add_argument("--exp_dir", type=str, default=str(DEFAULT_EXP_DIR))
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--diameter", type=float, default=80.0)
    ap.add_argument("--films", nargs="+", default=None,
                    help="restrict to these films (default: all)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Custom model not found: {model_path}")

    films = args.films if args.films else all_films()
    print(f"Total films to process: {len(films)}")
    use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    print(f"Loading Cellpose model '{model_path.name}' (use_gpu={use_gpu})...")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

    for idx, film in enumerate(films, 1):
        print(f"\n=== Progress: [{idx}/{len(films)}] {film} ===", flush=True)
        segment_film_keyframes(film, exp_dir, model, diameter=args.diameter)

    print(f"\n✅ Keyframes segmented for {len(films)} film(s).")


if __name__ == "__main__":
    main()
