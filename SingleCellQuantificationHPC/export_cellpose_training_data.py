#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_cellpose_training_data.py
---------------------------------
Export brush-corrected cell masks as Cellpose-compatible training pairs.

Cellpose 4.x training format:
    <name>.tif          — raw image (uint16 BF)
    <name>_masks.tif    — instance-labeled mask (uint16, 0 = background,
                          each cell = unique positive integer)

Usage
-----
    python export_cellpose_training_data.py \
        --movie_root "/Volumes/X10 Pro/Movies" \
        --out_dir "/Volumes/X10 Pro/Movies/cellpose_training_data" \
        [--min_cell_px 50] [--dry_run]

Only cells with qc.csv status == "corrected" are exported.
Frames are skipped when the corrected mask is identical to the Cellpose
seg.tif segment (no brush was needed) — those frames provide no new signal.

After running this script, fine-tune Cellpose with:

    KMP_DUPLICATE_LIB_OK=TRUE \\
    /Users/user/miniforge3/envs/cellpose-sam/bin/python -m cellpose \\
        --train \\
        --dir <out_dir>/train \\
        --test_dir <out_dir>/test \\
        --pretrained_model cyto3 \\
        --chan 0 --chan2 0 \\
        --n_epochs 100 \\
        --learning_rate 0.005 \\
        --weight_decay 0.0001 \\
        --batch_size 8 \\
        --model_name fungal_finetuned \\
        --use_gpu
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

# ─────────────────────────────────────────────────────────────
# Lightweight RLE decode (no project dependency needed for CLI)
# ─────────────────────────────────────────────────────────────

def _rle_decode(rle_str: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode run-length encoding to a boolean mask (H×W, row-major)."""
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    tokens = str(rle_str).split()
    i = 0
    while i < len(tokens) - 1:
        start  = int(tokens[i])
        length = int(tokens[i + 1])
        mask[start: start + length] = True
        i += 2
    return mask.reshape(shape)


def _load_seg_tif(path: Path) -> np.ndarray:
    """Load a Cellpose _seg.tif and return an integer-labeled 2-D array."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        # CellPose stores (3, H, W) — first channel is the label map
        arr = arr[0]
    return arr.astype(np.uint16)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / max(float(union), 1)


# ─────────────────────────────────────────────────────────────
# Film discovery
# ─────────────────────────────────────────────────────────────

def discover_corrected_films(movie_root: Path) -> List[Dict]:
    """Return metadata dicts for all films with ≥1 corrected cell."""
    records = []
    for qc_path in sorted(movie_root.rglob("qc.csv")):
        try:
            qc = pd.read_csv(qc_path)
        except Exception:
            continue

        corrected = {
            int(r["cell_id"])
            for _, r in qc.iterrows()
            if str(r.get("status", "")).strip().lower() == "corrected"
        }
        if not corrected:
            continue

        tracked_dir = qc_path.parent
        m = re.match(r"TrackedCells_(.+)$", tracked_dir.name)
        if not m:
            continue
        film      = m.group(1)
        film_dir  = tracked_dir.parent
        exp_dir   = film_dir.parent

        frames_dir = film_dir / f"Frames_{film}"
        masks_dir  = film_dir / f"Masks_{film}"
        if not frames_dir.is_dir() or not masks_dir.is_dir():
            continue

        # Detect channel index from Masks directory
        ch = 0
        for f in sorted(masks_dir.iterdir()):
            mt = re.search(r"_t_\d+_c_(\d+)_seg\.(tif|npy)$", f.name)
            if mt:
                ch = int(mt.group(1))
                break

        records.append({
            "experiment":    exp_dir.name,
            "film":          film,
            "frames_dir":    frames_dir,
            "masks_dir":     masks_dir,
            "tracked_dir":   tracked_dir,
            "corrected_ids": corrected,
            "channel_idx":   ch,
        })
    return records


# ─────────────────────────────────────────────────────────────
# Per-frame label builder
# ─────────────────────────────────────────────────────────────

def build_frame_label(
    frame_masks: Dict[int, np.ndarray],   # cell_id → bool mask (H, W)
) -> np.ndarray:
    """
    Merge per-cell binary masks into a single uint16 instance-label image.
    Background = 0; each cell gets its cell_id as the label value.
    Conflicts (overlapping corrected masks) are resolved by smaller cell_id
    winning (last write wins by sorted order).
    """
    if not frame_masks:
        return np.zeros((1, 1), dtype=np.uint16)

    sample = next(iter(frame_masks.values()))
    label  = np.zeros(sample.shape, dtype=np.uint16)
    for cid in sorted(frame_masks):
        label[frame_masks[cid]] = cid
    return label


# ─────────────────────────────────────────────────────────────
# Correction check: was the brush actually used at this frame?
# ─────────────────────────────────────────────────────────────

def mask_differs_from_cellpose(
    corr_mask: np.ndarray,   # bool (H, W) — corrected
    seg: np.ndarray,         # uint16 (H, W) — Cellpose labels
    iou_threshold: float = 0.92,
) -> bool:
    """
    True if the corrected mask differs meaningfully from the best Cellpose
    segment.  A high IoU means Cellpose already got it right → no new signal.
    """
    labels = np.unique(seg)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return True  # no seg → correction is definitely new

    best_iou = max(
        _iou(corr_mask, seg == lbl)
        for lbl in labels
    )
    return best_iou < iou_threshold


# ─────────────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────────────

def export_training_data(
    movie_root:      str,
    out_dir:         str,
    min_cell_px:     int   = 50,
    test_frac:       float = 0.15,
    iou_new_signal:  float = 0.92,
    dry_run:         bool  = False,
    seed:            int   = 42,
) -> None:
    random.seed(seed)
    movie_root = Path(movie_root)
    out_dir    = Path(out_dir)

    train_dir = out_dir / "train"
    test_dir  = out_dir / "test"
    if not dry_run:
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

    films = discover_corrected_films(movie_root)
    if not films:
        print("[export] No corrected films found — nothing to do.")
        return

    print(f"[export] Found {len(films)} film(s) with corrected cells.")
    print(f"[export] Output → {out_dir}")

    total_written   = 0
    total_skipped   = 0
    skipped_no_change = 0

    for meta in films:
        film        = meta["film"]
        frames_dir  = meta["frames_dir"]
        masks_dir   = meta["masks_dir"]
        tracked_dir = meta["tracked_dir"]
        ch          = meta["channel_idx"]
        corrected   = meta["corrected_ids"]

        print(f"\n[export] Film: {film}  ({len(corrected)} corrected cells)")

        # ── Load all corrected cell mask tables ─────────────────────
        # cell_id → DataFrame with columns [time_point, rle_bf, width, height]
        cell_tables: Dict[int, pd.DataFrame] = {}
        for cid in corrected:
            csv_p = tracked_dir / f"cell_{cid}_masks.csv"
            if not csv_p.exists():
                print(f"  [warn] cell_{cid}_masks.csv not found, skipping")
                continue
            try:
                df = pd.read_csv(csv_p)
            except Exception as e:
                print(f"  [warn] could not read cell_{cid}_masks.csv: {e}")
                continue
            cell_tables[cid] = df

        if not cell_tables:
            continue

        # Infer image dimensions from first table
        first_df = next(iter(cell_tables.values()))
        H = int(first_df.iloc[0]["height"])
        W = int(first_df.iloc[0]["width"])

        # Determine RLE column (use BF for all single-channel films)
        rle_col = "rle_bf"

        # ── Find all time-points present ────────────────────────────
        all_frames = set()
        for df in cell_tables.values():
            all_frames.update(df["time_point"].astype(int).tolist())
        all_frames = sorted(all_frames)

        # ── Assign each frame to train or test ─────────────────────
        random.shuffle(all_frames)
        n_test   = max(1, int(len(all_frames) * test_frac))
        test_set = set(all_frames[:n_test])

        # ── Per-frame export ────────────────────────────────────────
        for t in sorted(all_frames):
            frame_path = frames_dir / f"{film}_t_{t:03d}_c_{ch}.tif"
            seg_path   = masks_dir  / f"{film}_t_{t:03d}_c_{ch}_seg.tif"

            if not frame_path.exists():
                total_skipped += 1
                continue

            # Load BF image
            try:
                img = tifffile.imread(str(frame_path))
            except Exception:
                total_skipped += 1
                continue

            # Load Cellpose seg (for "new signal" check)
            seg = None
            if seg_path.exists():
                try:
                    seg = _load_seg_tif(seg_path)
                except Exception:
                    pass

            # Collect corrected masks for this frame
            frame_masks: Dict[int, np.ndarray] = {}
            frame_has_new_signal = False

            for cid, df in cell_tables.items():
                row = df[df["time_point"] == t]
                if row.empty:
                    continue
                rle_str = str(row.iloc[0].get(rle_col, "")).strip()
                if not rle_str or rle_str == "nan":
                    continue

                try:
                    m = _rle_decode(rle_str, (H, W))
                except Exception:
                    continue

                if m.sum() < min_cell_px:
                    continue

                frame_masks[cid] = m

                # Check if this mask differs from Cellpose
                if seg is not None and mask_differs_from_cellpose(m, seg, iou_new_signal):
                    frame_has_new_signal = True
                elif seg is None:
                    frame_has_new_signal = True  # no seg means we always want it

            if not frame_masks:
                total_skipped += 1
                continue

            # Only export frames where at least one cell was brush-corrected
            if not frame_has_new_signal:
                skipped_no_change += 1
                continue

            # Build label image
            label_img = build_frame_label(frame_masks)

            # Determine output directory
            dest = test_dir if t in test_set else train_dir
            stem = f"{film}_t{t:03d}"
            img_out   = dest / f"{stem}.tif"
            label_out = dest / f"{stem}_masks.tif"

            if dry_run:
                print(f"  [dry_run] {'test' if t in test_set else 'train'}  "
                      f"{stem}  cells={list(frame_masks.keys())}")
                total_written += 1
                continue

            try:
                # Save image as uint16 (preserve original bit depth)
                img_16 = img.astype(np.uint16) if img.dtype != np.uint16 else img
                tifffile.imwrite(str(img_out),   img_16,     photometric="minisblack")
                tifffile.imwrite(str(label_out), label_img,  photometric="minisblack")
                total_written += 1
            except Exception as e:
                print(f"  [error] could not write {stem}: {e}")
                total_skipped += 1

        print(f"  frames written: {total_written}  "
              f"skipped (no frame/seg): {total_skipped}  "
              f"skipped (identical to Cellpose): {skipped_no_change}")

    print(f"\n[export] DONE — {total_written} image/label pairs saved.")
    print(f"  train → {train_dir}")
    print(f"  test  → {test_dir}")

    if total_written > 0 and not dry_run:
        _print_finetune_command(str(train_dir), str(test_dir))


def _print_finetune_command(train_dir: str, test_dir: str) -> None:
    print("""
══════════════════════════════════════════════════════════════════
 Fine-tune Cellpose with the exported data:

  KMP_DUPLICATE_LIB_OK=TRUE \\
  /Users/user/miniforge3/envs/cellpose-sam/bin/python -m cellpose \\
      --train \\
      --dir "{train_dir}" \\
      --test_dir "{test_dir}" \\
      --pretrained_model cyto3 \\
      --chan 0 --chan2 0 \\
      --n_epochs 100 \\
      --learning_rate 0.005 \\
      --weight_decay 0.0001 \\
      --batch_size 8 \\
      --model_name fungal_finetuned \\
      --use_gpu

 The fine-tuned model will be saved in:
   {train_dir}/models/fungal_finetuned
══════════════════════════════════════════════════════════════════
""".format(train_dir=train_dir, test_dir=test_dir))


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export brush-corrected masks as Cellpose training pairs."
    )
    parser.add_argument(
        "--movie_root", default="/Volumes/X10 Pro/Movies",
        help="Root directory containing all experiment folders."
    )
    parser.add_argument(
        "--out_dir",
        default="/Volumes/X10 Pro/Movies/cellpose_training_data",
        help="Destination directory for train/ and test/ subdirectories."
    )
    parser.add_argument(
        "--min_cell_px", type=int, default=50,
        help="Minimum mask area in pixels; smaller masks are skipped."
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.15,
        help="Fraction of frames reserved for the test set (default 15%%)."
    )
    parser.add_argument(
        "--iou_new_signal", type=float, default=0.92,
        help="IoU threshold above which a corrected mask is considered "
             "identical to Cellpose (no new signal → skipped, default 0.92)."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be exported without writing any files."
    )
    args = parser.parse_args()

    export_training_data(
        movie_root     = args.movie_root,
        out_dir        = args.out_dir,
        min_cell_px    = args.min_cell_px,
        test_frac      = args.test_frac,
        iou_new_signal = args.iou_new_signal,
        dry_run        = args.dry_run,
    )


if __name__ == "__main__":
    main()
