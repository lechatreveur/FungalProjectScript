#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:16:37 2026

@author: user
"""

#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import cv2

T_RE = re.compile(r"_t_(\d{3})_c_0", re.IGNORECASE)

def t_index(p: Path) -> int:
    m = T_RE.search(p.stem)
    return int(m.group(1)) if m else 10**9  # put weird names at the end

def read_tiff_any(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read: {path}")
    return img

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    # If somehow multi-channel, convert for alignment only
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def safe_replace(out_tmp: Path, out_final: Path):
    # Replace atomically when possible
    out_tmp.replace(out_final)

def ecc_translation_warp(template_f32: np.ndarray, current_f32: np.ndarray,
                         max_iters: int, eps: float) -> np.ndarray:
    # Translation-only warp
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
    # ECC returns warp that aligns current -> template; use WARP_INVERSE_MAP when warping
    try:
        _, warp = cv2.findTransformECC(template_f32, current_f32, warp, cv2.MOTION_TRANSLATION, criteria)
    except cv2.error:
        # If ECC fails on a frame, fall back to identity (no correction)
        warp = np.eye(2, 3, dtype=np.float32)
    return warp

def warp_image(img: np.ndarray, warp: np.ndarray, size_wh, interp, border_mode, border_value=0):
    w, h = size_wh
    return cv2.warpAffine(
        img, warp, (w, h),
        flags=interp | cv2.WARP_INVERSE_MAP,
        borderMode=border_mode,
        borderValue=border_value
    )

def stabilize_film(film_dir: Path, dry_run: bool, max_iters: int, eps: float):
    # Identify Frames_* and Masks_* subfolders
    frames_dirs = [p for p in film_dir.iterdir() if p.is_dir() and p.name.startswith("Frames_")]
    masks_dirs  = [p for p in film_dir.iterdir() if p.is_dir() and p.name.startswith("Masks_")]

    if not frames_dirs:
        return 0  # nothing to do
    if len(frames_dirs) != 1:
        raise RuntimeError(f"Expected exactly 1 Frames_* dir in {film_dir}, found: {[p.name for p in frames_dirs]}")
    if len(masks_dirs) != 1:
        raise RuntimeError(f"Expected exactly 1 Masks_* dir in {film_dir}, found: {[p.name for p in masks_dirs]}")

    frames_dir = frames_dirs[0]
    masks_dir  = masks_dirs[0]

    frame_files = sorted(frames_dir.glob("*.tif"), key=t_index)
    if not frame_files:
        return 0

    # Template = first frame (grayscale float32)
    first = read_tiff_any(frame_files[0])
    first_gray = to_gray(first).astype(np.float32)
    h, w = first_gray.shape[:2]

    changed = 0

    for fpath in frame_files:
        img = read_tiff_any(fpath)
        gray_f32 = to_gray(img).astype(np.float32)

        if gray_f32.shape[:2] != (h, w):
            raise RuntimeError(f"Shape mismatch in {fpath}: got {gray_f32.shape[:2]}, expected {(h,w)}")

        warp = ecc_translation_warp(first_gray, gray_f32, max_iters=max_iters, eps=eps)

        # Warp frame
        stabilized_frame = warp_image(
            img, warp, (w, h),
            interp=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT
        )

        # Corresponding mask
        mask_name = fpath.stem + "_seg.tif"
        mpath = masks_dir / mask_name
        if not mpath.exists():
            raise RuntimeError(f"Missing mask for {fpath.name}: expected {mpath.name} in {masks_dir}")

        mask = read_tiff_any(mpath)

        # Warp mask with nearest neighbor, pad with 0
        stabilized_mask = warp_image(
            mask, warp, (w, h),
            interp=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0
        )

        if dry_run:
            print(f"[DRY RUN] Would overwrite:\n  {fpath}\n  {mpath}")
            continue

        # Write to temp then replace originals
        tmp_frame = fpath.with_name(fpath.stem + ".tmp" + fpath.suffix)   # ...tmp.tif
        tmp_mask  = mpath.with_name(mpath.stem + ".tmp" + mpath.suffix)   # ...tmp.tif

        if not cv2.imwrite(str(tmp_frame), stabilized_frame):
            raise RuntimeError(f"Failed writing {tmp_frame}")
        if not cv2.imwrite(str(tmp_mask), stabilized_mask):
            raise RuntimeError(f"Failed writing {tmp_mask}")

        safe_replace(tmp_frame, fpath)
        safe_replace(tmp_mask, mpath)
        changed += 1

    return changed

def main():
    ap = argparse.ArgumentParser(description="Stabilize microscope timelapse TIFF frames + segmentation masks in-place (translation-only).")
    ap.add_argument("working_dir", type=str, help="e.g. /Volume/Movie/2025_12_31_M92")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be changed, but do not overwrite files.")
    ap.add_argument("--max-iters", type=int, default=200, help="ECC max iterations.")
    ap.add_argument("--eps", type=float, default=1e-6, help="ECC convergence epsilon.")
    args = ap.parse_args()

    wd = Path(args.working_dir)
    if not wd.exists():
        raise FileNotFoundError(wd)

    # Each film is a subfolder inside working_dir
    film_dirs = sorted([p for p in wd.iterdir() if p.is_dir()])

    total = 0
    for film_dir in film_dirs:
        try:
            n = stabilize_film(film_dir, dry_run=args.dry_run, max_iters=args.max_iters, eps=args.eps)
            if n > 0:
                print(f"{film_dir.name}: stabilized {n} frames (+ masks)")
            total += n
        except Exception as e:
            print(f"[ERROR] {film_dir}: {e}")
            # continue to next film

    print(f"Done. Films processed: {len(film_dirs)}. Frames stabilized: {total}.")
    if args.dry_run:
        print("Dry run only: no files were overwritten.")

if __name__ == "__main__":
    main()