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

def prepare_mask_for_ecc(mask_path: Path) -> np.ndarray:
    mask = read_tiff_any(mask_path)
    # Convert to binary float32 (0.0 or 255.0)
    binary = (mask > 0).astype(np.float32) * 255.0
    # Apply Gaussian blur to create smooth gradients for ECC optimization
    blurred = cv2.GaussianBlur(binary, (9, 9), 2.0)
    return blurred

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
    except cv2.error as e:
        # If ECC fails on a frame, fall back to identity (no correction)
        print(f"  [WARNING] ECC failed, using identity warp. Error: {e}")
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

def stabilize_film(film_dir: Path, dry_run: bool, max_iters: int, eps: float, align_by_masks: bool = False):
    frames_dirs = [p for p in film_dir.iterdir() if p.is_dir() and p.name.startswith("Frames_")]
    masks_dirs  = [p for p in film_dir.iterdir() if p.is_dir() and p.name.startswith("Masks_")]

    if not frames_dirs:
        return 0  # nothing to do
    if len(frames_dirs) != 1:
        raise RuntimeError(f"Expected exactly 1 Frames_* dir in {film_dir}, found: {[p.name for p in frames_dirs]}")
    
    frames_dir = frames_dirs[0]
    masks_dir  = masks_dirs[0] if masks_dirs else None

    frame_files = sorted([p for p in frames_dir.glob("*.tif") if not p.name.startswith("._")], key=t_index)
    if not frame_files:
        return 0

    if align_by_masks and not masks_dir:
        print(f"[WARNING] align_by_masks requested but no Masks_* directory found in {film_dir}. Falling back to BF frames.")
        align_by_masks = False

    # Template for alignment
    if align_by_masks:
        first_mask_name = frame_files[0].stem + "_seg.tif"
        first_mask_path = masks_dir / first_mask_name
        if not first_mask_path.exists():
            raise RuntimeError(f"First mask not found: {first_mask_path}")
        first_align = prepare_mask_for_ecc(first_mask_path)
    else:
        first = read_tiff_any(frame_files[0])
        first_align = to_gray(first).astype(np.float32)

    h, w = first_align.shape[:2]
    changed = 0

    for fpath in frame_files:
        img = read_tiff_any(fpath)
        
        # Determine current alignment image
        mpath = None
        if masks_dir:
            mask_name = fpath.stem + "_seg.tif"
            mpath = masks_dir / mask_name

        if align_by_masks:
            if mpath and mpath.exists():
                curr_align = prepare_mask_for_ecc(mpath)
            else:
                raise RuntimeError(f"Mask not found for {fpath.name}")
        else:
            curr_align = to_gray(img).astype(np.float32)

        if curr_align.shape[:2] != (h, w):
            raise RuntimeError(f"Shape mismatch in alignment image for {fpath}: got {curr_align.shape[:2]}, expected {(h,w)}")

        warp = ecc_translation_warp(first_align, curr_align, max_iters=max_iters, eps=eps)
        tx = warp[0, 2]
        ty = warp[1, 2]
        print(f"  {fpath.name}: dx={tx:.3f}, dy={ty:.3f}")

        # Warp frame
        stabilized_frame = warp_image(
            img, warp, (w, h),
            interp=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT
        )

        # Corresponding mask (Optional)
        stabilized_mask = None
        if mpath and mpath.exists():
            mask = read_tiff_any(mpath)
            # Warp mask with nearest neighbor, pad with 0
            stabilized_mask = warp_image(
                mask, warp, (w, h),
                interp=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0
            )

        if dry_run:
            print(f"[DRY RUN] Would overwrite: {fpath}")
            if mpath and mpath.exists():
                print(f"[DRY RUN] Would overwrite: {mpath}")
            continue

        # Write to temp then replace originals
        tmp_frame = fpath.with_name(fpath.stem + ".tmp" + fpath.suffix)
        if not cv2.imwrite(str(tmp_frame), stabilized_frame):
            raise RuntimeError(f"Failed writing {tmp_frame}")
        safe_replace(tmp_frame, fpath)

        if stabilized_mask is not None:
            tmp_mask  = mpath.with_name(mpath.stem + ".tmp" + mpath.suffix)
            if not cv2.imwrite(str(tmp_mask), stabilized_mask):
                raise RuntimeError(f"Failed writing {tmp_mask}")
            safe_replace(tmp_mask, mpath)
        
        changed += 1

    return changed

def main():
    ap = argparse.ArgumentParser(description="Stabilize microscope timelapse TIFF frames + segmentation masks in-place (translation-only).")
    ap.add_argument("working_dir", type=str, help="e.g. /Volume/Movie/2025_12_31_M92")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be changed, but do not overwrite files.")
    ap.add_argument("--max-iters", type=int, default=200, help="ECC max iterations.")
    ap.add_argument("--eps", type=float, default=1e-6, help="ECC convergence epsilon.")
    ap.add_argument("--align-by-masks", action="store_true", help="Align images based on segmentations/masks instead of BF images.")
    args = ap.parse_args()

    wd = Path(args.working_dir)
    if not wd.exists():
        raise FileNotFoundError(wd)

    # Check if wd itself is a film directory containing a Frames_* folder
    if any(p.name.startswith("Frames_") for p in wd.iterdir() if p.is_dir()):
        film_dirs = [wd]
    else:
        # Each film is a subfolder inside working_dir
        film_dirs = sorted([p for p in wd.iterdir() if p.is_dir()])

    total = 0
    for film_dir in film_dirs:
        # Skip directories like done_movies, population_movies, etc.
        if film_dir.name in ["done_movies", "population_movies"]:
            continue
        try:
            print(f"Processing film directory: {film_dir.name}")
            n = stabilize_film(film_dir, dry_run=args.dry_run, max_iters=args.max_iters, eps=args.eps, align_by_masks=args.align_by_masks)
            if n > 0:
                print(f"Finished {film_dir.name}: stabilized {n} frames (+ masks)")
            total += n
        except Exception as e:
            print(f"[ERROR] {film_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Done. Films processed: {len(film_dirs)}. Frames stabilized: {total}.")
    if args.dry_run:
        print("Dry run only: no files were overwritten.")

if __name__ == "__main__":
    main()