#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_patch_pattern_score.py

For each cell in the video cache whose cell_ID_data.csv is missing
'pattern_score_norm', re-run the touching-circles pattern scorer
(quantify_one_object from quant_helpers) using the already-tracked masks.

This is a lightweight patch — it does NOT re-run cell tracking. It:
  1. Reads the existing cell_<id>_masks.csv (tracked RLE masks per frame)
  2. Loads each GFP frame
  3. Calls quantify_one_object() to get pattern_score_norm per frame
  4. Patches cell_<id>_data.csv with the new columns (in-place update)

Usage:
    python Video_AE_patch_pattern_score.py [--dry-run] [--limit N]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.measure import regionprops, label

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from Cell_tracking_functions import rle_decode, touches_border
from quant_helpers import (
    pattern_score_touching_circles,
    transform_to_mn_space,
)
from SingleCellDataAnalysis.Video_AE_data_loader import (
    EXPERIMENT_BASES, STACKED_CSV_PATHS, ID_MAP_CSV_PATHS,
    FILM_FOLDER_MAP, resolve_cell_info_sept17, resolve_cell_info_generic
)

# ==============================================================================
GIDS_PATH = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/video_gids.txt"
SIDE_PX   = 60           # template radius, same as quantify_one_object default
DO_PLOTS  = False        # set True to save overlay images (slow)

PATTERN_COLS = [
    'pattern_score_raw', 'pattern_score_norm',
    'pattern_center_row', 'pattern_center_col',
    'pattern_center_m', 'pattern_center_n',
    'pattern_side_px', 'pattern_evalN',
    'pattern_filtered', 'pattern_n_length_px', 'pattern_band_half_px',
]
# ==============================================================================


def get_film_paths(gid: str, stacked_dfs: dict, id_map_dfs: dict):
    """Return (base_dir, film_name, orig_id) for a given GID, or (None,None,None)."""
    known_labels = sorted(EXPERIMENT_BASES.keys(), key=lambda x: -len(x))
    exp_label = local_id = None
    for lbl in known_labels:
        if gid.startswith(lbl + '_'):
            exp_label = lbl
            local_id = int(gid[len(lbl) + 1:])
            break
    if exp_label is None:
        return None, None, None

    base_dir = EXPERIMENT_BASES[exp_label]
    if exp_label == 'Sept17':
        df_s = stacked_dfs.get('Sept17')
        if df_s is None:
            return None, None, None
        film_name, orig_id = resolve_cell_info_sept17(local_id, df_s)
    elif exp_label == 'June25_20m':
        film_name = FILM_FOLDER_MAP.get(('June25_20m', 'GFP1', 'F0'))
        orig_id = local_id
    else:
        df_m = id_map_dfs.get(exp_label)
        if df_m is None:
            return None, None, None
        film_name, orig_id = resolve_cell_info_generic(local_id, df_m, exp_label)

    if film_name is None or orig_id is None:
        return None, None, None
    return base_dir, film_name, orig_id


def find_gfp_frame(frames_dir: str, film_name: str, t: int) -> np.ndarray | None:
    """Try multiple naming patterns to load a GFP TIF frame."""
    patterns = [
        f"{film_name}_t_{t:03d}_c_0.tif",
        f"{film_name}_t_{t:02d}_c_0.tif",
        f"{film_name}_t_{t:03d}_c_1.tif",
        f"{film_name}_t_{t:03d}_z_1_c_1.tif",
    ]
    for p in patterns:
        fp = os.path.join(frames_dir, p)
        if os.path.exists(fp):
            return imread(fp).astype(np.float32)
    return None


def compute_gfp_range(frames_dir: str, film_name: str, n_frames: int):
    """Compute movie-wide GFP min/max (P1, P99.5) from a sample of frames."""
    pixels = []
    step = max(1, n_frames // 10)   # sample ~10 frames
    for t in range(0, n_frames, step):
        img = find_gfp_frame(frames_dir, film_name, t)
        if img is not None:
            flat = img.ravel()
            pixels.extend(flat[::4])   # further subsample to keep memory low
    if not pixels:
        return 0.0, 65535.0
    arr = np.asarray(pixels, dtype=np.float32)
    return float(np.percentile(arr, 1.0)), float(np.percentile(arr, 99.5))


def midpoints_from_mask(mask_bool: np.ndarray):
    """Return two minor-axis midpoints from the cell mask (used by pattern scorer)."""
    lab = label(mask_bool.astype(np.uint8), connectivity=2)
    regs = regionprops(lab)
    if not regs:
        ys, xs = np.nonzero(mask_bool)
        if ys.size == 0:
            return (0.0, 0.0), (0.0, 0.0)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        return (cy - 5.0, cx), (cy + 5.0, cx)
    r = regs[0]
    cy, cx = r.centroid
    theta = float(getattr(r, 'orientation', 0.0) or 0.0)
    vy, vx = np.cos(theta), -np.sin(theta)
    a_minor = max(float(getattr(r, 'minor_axis_length', 0.0) or 0.0) / 2.0, 5.0)
    mid1 = (cy - a_minor * vy, cx - a_minor * vx)
    mid2 = (cy + a_minor * vy, cx + a_minor * vx)
    return mid1, mid2


def prob_to_support_from_mask(mask_bool: np.ndarray) -> np.ndarray:
    """
    Lightweight substitute for the full EM-based support mask:
    use the cell mask itself as support (all foreground pixels = support).
    This is appropriate here since we just need the pattern score geometry.
    """
    return mask_bool.copy()


def score_one_frame(img: np.ndarray, mask_bool: np.ndarray) -> dict:
    """Run touching-circles scorer on one frame. Returns pattern score dict."""
    seg = np.asarray(mask_bool, bool)
    if seg.sum() < 20:
        return {k: None for k in PATTERN_COLS}

    # crop to mask bounding box (with small padding)
    rows = np.where(seg.any(axis=1))[0]
    cols = np.where(seg.any(axis=0))[0]
    PAD = 15
    H, W = seg.shape
    r0 = max(0, rows[0]  - PAD)
    r1 = min(H, rows[-1] + PAD + 1)
    c0 = max(0, cols[0]  - PAD)
    c1 = min(W, cols[-1] + PAD + 1)

    crop_mask = seg[r0:r1, c0:c1]
    support   = prob_to_support_from_mask(crop_mask)

    mid1_rc, mid2_rc = midpoints_from_mask(crop_mask)

    pat = pattern_score_touching_circles(
        support, crop_mask, mid1_rc, mid2_rc,
        side_px=SIDE_PX, stride=1
    )

    return {
        'pattern_score_raw':    pat['best_score_raw'],
        'pattern_score_norm':   pat['best_score_norm'],
        'pattern_center_row':   None if pat['best_center_rc'] is None else pat['best_center_rc'][0] + r0,
        'pattern_center_col':   None if pat['best_center_rc'] is None else pat['best_center_rc'][1] + c0,
        'pattern_center_m':     None if pat['best_center_mn'] is None else pat['best_center_mn'][0],
        'pattern_center_n':     None if pat['best_center_mn'] is None else pat['best_center_mn'][1],
        'pattern_side_px':      pat['side_px'],
        'pattern_evalN':        pat['evaluated_centers'],
        'pattern_filtered':     pat.get('filtered_by_center_n', False),
        'pattern_n_length_px':  pat.get('n_length_px', None),
        'pattern_band_half_px': pat.get('band_half_px', None),
    }


def patch_one_cell(gid: str, base_dir: str, film_name: str, orig_id: int,
                   dry_run: bool = False) -> str:
    """
    Patch pattern_score columns into the cell's data CSV.
    Returns a status string.
    """
    tracked_dir  = os.path.join(base_dir, film_name, f"TrackedCells_{film_name}")
    frames_dir   = os.path.join(base_dir, film_name, f"Frames_{film_name}")
    masks_csv    = os.path.join(tracked_dir, f"cell_{orig_id}_masks.csv")
    data_csv     = os.path.join(tracked_dir, f"cell_{orig_id}_data.csv")

    if not os.path.exists(masks_csv):
        return f"SKIP  no masks CSV: {masks_csv}"
    if not os.path.exists(data_csv):
        return f"SKIP  no data CSV: {data_csv}"

    # Check if already has pattern_score_norm
    df_data = pd.read_csv(data_csv)
    if 'pattern_score_norm' in df_data.columns:
        n_valid = pd.to_numeric(df_data['pattern_score_norm'], errors='coerce').notna().sum()
        if n_valid > 0:
            return f"SKIP  already has {n_valid} valid pattern_score_norm rows"

    # Load masks
    df_masks = pd.read_csv(masks_csv)
    n_frames = len(df_masks)
    H = int(df_masks.iloc[0]['height'])
    W = int(df_masks.iloc[0]['width'])

    # Determine RLE column name (could be 'rle', 'rle_gfp', etc.)
    rle_col = None
    for candidate in ['rle', 'rle_gfp']:
        if candidate in df_masks.columns:
            rle_col = candidate
            break
    if rle_col is None:
        return f"SKIP  no RLE column in masks CSV (cols: {list(df_masks.columns)})"

    if dry_run:
        return f"DRY   would patch {n_frames} frames"

    # Compute GFP range from frames
    gfp_min, gfp_max = compute_gfp_range(frames_dir, film_name, n_frames)

    # Score each frame
    pat_rows = []
    for t in range(n_frames):
        row = df_masks.iloc[t]
        rle_str = row.get(rle_col, '')
        if not isinstance(rle_str, str) or not rle_str.strip():
            pat_rows.append({k: None for k in PATTERN_COLS})
            continue

        mask = np.asarray(rle_decode(str(rle_str), (H, W)), bool)
        img  = find_gfp_frame(frames_dir, film_name, t)
        if img is None:
            pat_rows.append({k: None for k in PATTERN_COLS})
            continue

        scores = score_one_frame(img, mask)
        pat_rows.append(scores)

    df_pat = pd.DataFrame(pat_rows)

    # Align by time_point: df_data may have fewer rows if tracking stopped early
    # Match on row index (time_point == t)
    n_data = len(df_data)
    for col in PATTERN_COLS:
        if col in df_data.columns:
            df_data[col] = df_data[col].astype(object)   # allow mixed types
        else:
            df_data[col] = None

    for i in range(min(n_data, len(df_pat))):
        t_val = int(df_data.iloc[i].get('time_point', i))
        if t_val < len(df_pat):
            for col in PATTERN_COLS:
                df_data.at[i, col] = df_pat.iloc[t_val][col]

    df_data.to_csv(data_csv, index=False)
    return f"OK    patched {min(n_data, n_frames)} frames → {data_csv}"


def main():
    parser = argparse.ArgumentParser(description="Patch pattern_score_norm into existing cell data CSVs.")
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without writing.')
    parser.add_argument('--limit', type=int, default=0, help='Process only the first N cells (0 = all).')
    args = parser.parse_args()

    # Load GIDs
    with open(GIDS_PATH) as f:
        gids = [l.strip() for l in f if l.strip()]

    print(f"📋 {len(gids)} cells in video cache.")

    # Load ID maps
    stacked_dfs, id_map_dfs = {}, {}
    for exp, path in STACKED_CSV_PATHS.items():
        if path and os.path.exists(path):
            stacked_dfs[exp] = pd.read_csv(path)
    for exp, path in ID_MAP_CSV_PATHS.items():
        if path and os.path.exists(path):
            id_map_dfs[exp] = pd.read_csv(path)

    counts = {'ok': 0, 'skip': 0, 'dry': 0, 'error': 0}
    limit = args.limit if args.limit > 0 else len(gids)

    for i, gid in enumerate(gids[:limit]):
        base_dir, film_name, orig_id = get_film_paths(gid, stacked_dfs, id_map_dfs)
        if base_dir is None:
            print(f"[{i+1:3d}/{limit}] {gid:30s} → SKIP  cannot resolve film")
            counts['skip'] += 1
            continue

        try:
            status = patch_one_cell(gid, base_dir, film_name, orig_id, dry_run=args.dry_run)
        except Exception as e:
            status = f"ERROR {e}"
            counts['error'] += 1

        tag = status.split()[0]
        counts[tag.lower() if tag in ('OK', 'SKIP', 'DRY') else 'error'] = counts.get(tag.lower() if tag in ('OK', 'SKIP', 'DRY') else 'error', 0) + 1
        if tag == 'OK':
            counts['ok'] += 1
        elif tag == 'SKIP':
            counts['skip'] += 1
        elif tag == 'DRY':
            counts['dry'] += 1

        print(f"[{i+1:3d}/{limit}] {gid:35s} → {status}")

    print(f"\n✅ Done. OK={counts['ok']}  SKIP={counts['skip']}  DRY={counts['dry']}  ERROR={counts['error']}")


if __name__ == '__main__':
    main()
