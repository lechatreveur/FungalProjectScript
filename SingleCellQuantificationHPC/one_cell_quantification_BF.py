#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:17:48 2025

@author: user
"""


# =========================
# Imports & module setup
# =========================
import os
import sys
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops, label

# Project path(s)
# sys.path.append('/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC')
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from Cell_tracking_functions import (
    load_segmentation,
    to_labeled_current,
    compute_overlap,
    get_cell_mask_area_aware,
    touches_border,
    rle_encode, rle_decode,
)

from quant_helpers import (
    quantify_one_object,            # GFP main entry (unchanged)
    pattern_score_touching_circles, # used for BF pattern-only
    save_touching_circles_pattern_overlay,
    transform_to_mn_space,          # used by pattern pipeline
    save_prob_and_support_debug,
    pattern_score_split_rectangles,
    save_split_rectangles_pattern_overlay

)
from Image_quantification_functions import (
    find_midpoints_on_minor_axis
)

from bf_pattern import bf_pattern_only

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(description="Track a single cell in GFP+BF and quantify signals.")
parser.add_argument('--cell_id', type=int, required=True, help='Cell ID to process')
parser.add_argument('--track_channel', choices=['bf', 'gfp'], default='bf',
                    help='[Info only] Which segmentation to print counts for; both are tracked.')
parser.add_argument('--direction', choices=['forward', 'backward', 'both'], default='both',
                    help='Which tracking direction(s) to run.')
parser.add_argument('--experiment_path', type=str, required=True, help='Path to experiment')
parser.add_argument('--file_name', type=str, required=True, help='File name of .ims file')
parser.add_argument('--z_index', type=int, default=1, help="Z-slice index to load for BF segmentation.")
parser.add_argument('--min_area', type=int, default=2500, help="Minimum area threshold for cell filtering (first frame, BF).")
parser.add_argument('--update_existing', action='store_true', help="Only update existing rows (skip tracking if masks CSV is valid).")
args = parser.parse_args()

cell_id        = args.cell_id
working_dir    = args.experiment_path
file_name      = args.file_name
min_area       = args.min_area
z_index        = args.z_index
direction_mode = args.direction
update_existing = args.update_existing

# =========================
# Paths
# =========================
output_frames_folder        = f"{working_dir}{file_name}/Frames_{file_name}"
output_masks_folder         = f"{working_dir}{file_name}/Masks_{file_name}"
output_tracked_cells_folder = f"{working_dir}{file_name}/TrackedCells_{file_name}"

GFP_seg_folder         = os.path.join(output_masks_folder, "GFP_seg")
brightfield_seg_folder = os.path.join(output_masks_folder, "brightfield_seg")
os.makedirs(GFP_seg_folder, exist_ok=True)
os.makedirs(brightfield_seg_folder, exist_ok=True)
os.makedirs(output_tracked_cells_folder, exist_ok=True)

# Output (plots / tables)
plot_output_root  = os.path.join(output_tracked_cells_folder, "cell_plots")
cell_plot_folder_gfp  = os.path.join(plot_output_root, f"cell_{cell_id}")       # same as before
cell_plot_folder_bf   = os.path.join(plot_output_root, f"cell_{cell_id}_BF")    # NEW
os.makedirs(plot_output_root, exist_ok=True)
os.makedirs(cell_plot_folder_gfp, exist_ok=True)
os.makedirs(cell_plot_folder_bf, exist_ok=True)

# =========================
# Path helper functions
# =========================
def bf_seg_path(t: int) -> str:
    return os.path.join(brightfield_seg_folder, f"{file_name}_t_{t:03d}_z_{z_index}_c_1_seg.tif")

def gfp_seg_path(t: int) -> str:
    return os.path.join(GFP_seg_folder, f"{file_name}_t_{t:03d}_c_0_seg.tif")

def gfp_frame_path(t: int) -> str:
    return os.path.join(output_frames_folder, f"{file_name}_t_{t:03d}_c_0.tif")

def bf_frame_path(t: int) -> str:
    return os.path.join(output_frames_folder, f"{file_name}_t_{t:03d}_z_{z_index}_c_1.tif")

def seg_path(channel: str, t: int) -> str:
    return bf_seg_path(t) if channel == 'bf' else gfp_seg_path(t)

# =========================
# Utilities
# =========================


def load_labeled_seg(channel, t):
    import os
    import numpy as np
    from skimage.measure import label as _label

    # Uses your helpers exactly:
    # seg_path -> bf_seg_path/gfp_seg_path depending on `channel`
    p = seg_path(channel, t)
    print(f"[load_labeled_seg] channel={channel} t={t} -> {p}")

    if not os.path.exists(p):
        raise FileNotFoundError(f"Segmentation path does not exist: {p}")

    raw = load_segmentation(p)  # keep the robust loader (see below)

    # Normalize to labeled integers
    if raw.dtype == bool:
        raw = _label(raw.astype('uint8'))
    elif raw.ndim == 3 and raw.shape[-1] in (3, 4):
        raw = _label((raw[..., 0] > 0).astype('uint8'))
    elif not np.issubdtype(raw.dtype, np.integer):
        raw = _label((raw > 0).astype('uint8'))

    return raw


def GetFilteredRegions(min_area: int = 2500):
    """Use **BF first frame** to select valid cells by area (seed comes from BF)."""
    mask_files = sorted([f for f in os.listdir(brightfield_seg_folder)
                         if f.endswith('_seg.npy') or f.endswith('_seg.tif')])
    if len(mask_files) == 0:
        raise FileNotFoundError("No brightfield segmentation files found.")
    first_mask   = load_segmentation(os.path.join(brightfield_seg_folder, mask_files[0]))
    labeled_mask = to_labeled_current(first_mask)
    regions      = regionprops(labeled_mask)
    filtered     = [r for r in regions if r.area >= min_area]
    return first_mask, labeled_mask, filtered

def FindMovieMaxMin(channel: int, *, z_index: int | None = None, frames_dir: str | None = None, sample_step: int = 10):
    """
    Scan Frames_<file_name> for channel and return (p99.5, p1, frame_files).
    Supports filenames both with and without 'z_<index>' and excludes '*_seg.tif'.
    """
    from fnmatch import fnmatch
    folder = frames_dir or output_frames_folder

    # Build flexible patterns
    patterns = [
        f"{file_name}_t_???_c_{channel}.tif",   # old: no z
        f"{file_name}_t_*_c_{channel}.tif",     # relaxed no z
    ]
    if z_index is not None:
        patterns += [
            f"{file_name}_t_???_z_{z_index}_c_{channel}.tif",  # strict with z
            f"{file_name}_t_*_z_{z_index}_c_{channel}.tif",    # relaxed with z
        ]

    # Collect matches; exclude segmentation files
    try:
        entries = os.listdir(folder)
    except FileNotFoundError:
        return None, None, []

    frame_files = sorted([
        f for f in entries
        if f.lower().endswith(".tif")
        and not f.endswith("_seg.tif")
        and any(fnmatch(f, pat) for pat in patterns)
    ])

    if not frame_files:
        return None, None, []

    # Sample pixels for percentiles
    pixels = []
    for fname in frame_files:
        frame = imread(os.path.join(folder, fname))
        flat = frame.ravel()
        pixels.extend(flat[::sample_step] if sample_step and sample_step > 1 else flat)

    if not pixels:
        return None, None, frame_files

    pixels = np.asarray(pixels)
    hi = float(np.percentile(pixels, 99.5))
    lo = float(np.percentile(pixels, 1.0))
    return hi, lo, frame_files


def choose_side_px_for_crop(h, w):
    """Heuristic size for touching-circles square (works for BF pattern)."""
    m = int(2 * min(h, w))
    return int(np.clip(m, 20, 120))

def midpoints_from_mask_rc(mask_crop_bool):
    """
    Compute two midpoints along the **minor axis** (row, col) using regionprops.
    Mirrors the fallback in quant_helpers.extract_midpoints_rc_from_plot_data.
    """
    lab = label(mask_crop_bool.astype(np.uint8), connectivity=2)
    regs = regionprops(lab)
    if not regs:
        # fallback: center of mass and a small vertical segment
        ys, xs = np.nonzero(mask_crop_bool)
        if ys.size == 0:
            return (0.0, 0.0), (0.0, 0.0)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        return (cy - 5.0, cx), (cy + 5.0, cx)

    r = regs[0]
    cy, cx = r.centroid
    theta = getattr(r, "orientation", 0.0) or 0.0
    # minor-axis direction (rows, cols) = rotate major axis by 90°
    vy, vx = np.cos(theta), -np.sin(theta)
    a_minor = max(float(getattr(r, "minor_axis_length", 0.0) or 0.0) / 2.0, 5.0)
    mid1_rc = (cy - a_minor * vy, cx - a_minor * vx)
    mid2_rc = (cy + a_minor * vy, cx + a_minor * vx)
    return mid1_rc, mid2_rc


# =========================
# Tracking
# =========================
def track_one_direction(t_seq, ref_start_mask, channel='bf',
                        first_threshold=0.5, next_threshold=0.7,
                        area_lambda=0.35, ratio_soft=1.3, ratio_hard=1.8, topk=5,
                        lock_first=True):
    results = {}
    prev_mask = None
    prev_area = float(ref_start_mask.sum())

    for i, t in enumerate(t_seq):
        ref = ref_start_mask if i == 0 else prev_mask
        lab_cur = load_labeled_seg(channel, t)

        if i == 0 and lock_first:
            cm  = ref.copy()               # keep the seed exactly
            ov  = compute_overlap(ref, cm) # typically 1.0
            sc  = ov
            pen = 0.0
            rej = False
        else:
            if lab_cur is not None:
                thr = first_threshold if i == 0 else next_threshold
                cm, ov, sc, pen, rej = get_cell_mask_area_aware(
                    lab_cur, ref, prev_area,
                    threshold=thr, max_segments=2, topk=topk,
                    area_lambda=area_lambda, ratio_soft=ratio_soft, ratio_hard=ratio_hard
                )
            else:
                cm  = ref.copy()
                ov  = 1.0
                sc  = ov
                pen = 0.0
                rej = False

        results[t] = {
            "mask": cm,
            "overlap": float(ov),
            "score": float(sc),
            "area": int(cm.sum()),
            "area_penalty": float(pen),
            "huge_jump_rejected": bool(rej),
            "touch": touches_border(cm),
        }
        prev_mask = cm
        prev_area = float(cm.sum())

    return results


# =========================
# Main
# =========================
if __name__ == "__main__":
    from pathlib import Path

    # Discover frames (for both channels)
    gfp_max, gfp_min, gfp_frame_files = FindMovieMaxMin(0)
    _, _, bf_frame_files  = FindMovieMaxMin(1, z_index=z_index)

    n_gfp = len(gfp_frame_files)
    n_bf  = len(bf_frame_files)
    if n_gfp == 0:
        print("[error] No GFP frames found (c_0).")
        sys.exit(1)
    if n_bf == 0:
        print("[warn] No BF frames found (c_1). Proceeding with GFP only.")
    if n_bf and n_bf != n_gfp:
        print(f"[warn] GFP frames={n_gfp}, BF frames={n_bf}. Using min length to stay aligned.")
    frame_number = min(n_gfp, n_bf) if n_bf else n_gfp
    gfp_frame_files = gfp_frame_files[:frame_number]
    if n_bf:
        bf_frame_files = bf_frame_files[:frame_number]

    masks_csv_path = Path(output_tracked_cells_folder) / f"cell_{cell_id}_masks.csv"

    def is_valid_csv(p: Path) -> bool:
        try:
            return p.is_file() and p.stat().st_size > 0
        except Exception:
            return False

    need_to_track = args.update_existing or (not is_valid_csv(masks_csv_path))

    print(
        f"[track] update_existing={args.update_existing}  "
        f"csv_exists={masks_csv_path.is_file()}  "
        f"csv_valid={is_valid_csv(masks_csv_path)}  "
        f"-> need_to_track={need_to_track}"
    )

    if need_to_track:
        masks_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Seed (BF first frame) & basic checks
        first_mask, labeled_mask, filtered_regions = GetFilteredRegions(min_area)
        cell = next((c for c in filtered_regions if c.label == cell_id), None)
        if cell is None:
            print(f"Cell {cell_id} not found in BF first frame.")
            sys.exit(1)
        initial_mask = (labeled_mask == cell_id)

        # 2) Tracking in both channels
        # GFP
        forward_gfp = track_one_direction(
            range(frame_number), initial_mask,
            channel='gfp', first_threshold=0.5, next_threshold=0.7, lock_first=False
        )
        backward_gfp = None
        if direction_mode in ('backward', 'both'):
            seed_mask_backward_gfp = forward_gfp[frame_number - 1]["mask"]
            backward_gfp = track_one_direction(
                range(frame_number - 1, -1, -1), seed_mask_backward_gfp,
                channel='gfp', first_threshold=0.5, next_threshold=0.7
            )

        # BF
        forward_bf = track_one_direction(
            range(frame_number), initial_mask,
            channel='bf', first_threshold=0.5, next_threshold=0.7, lock_first=False
        )
        backward_bf = None
        if direction_mode in ('backward', 'both'):
            seed_mask_backward_bf = forward_bf[frame_number - 1]["mask"]
            backward_bf = track_one_direction(
                range(frame_number - 1, -1, -1), seed_mask_backward_bf,
                channel='bf', first_threshold=0.5, next_threshold=0.7
            )

        # 3) Reconcile per frame & save masks CSV (both channels)
        combined = []

        # (H, W) from initial mask
        H, W = initial_mask.shape

        for t in range(frame_number):
            # optional: label counts per frame for sanity (print one channel of your choice)
            if os.path.exists(bf_seg_path(t)):
                raw = load_segmentation(bf_seg_path(t)); lab = to_labeled_current(raw)
                print(f"[t={t}] BF labels: {np.unique(lab).size - 1}")
            if os.path.exists(gfp_seg_path(t)):
                raw = load_segmentation(gfp_seg_path(t)); lab = to_labeled_current(raw)
                print(f"[t={t}] GFP labels: {np.unique(lab).size - 1}")

            # choose function for channel
            def choose(forward_dict, backward_dict):
                if direction_mode == 'forward':
                    chosen = forward_dict.get(t, None)
                    src = "forward"
                elif direction_mode == 'backward':
                    chosen = (backward_dict or {}).get(t, None)
                    src = "backward"
                else:  # both -> backward wins, else forward, else empty
                    b = (backward_dict or {}).get(t, None)
                    if b is not None:
                        chosen, src = b, "backward"
                    else:
                        f = forward_dict.get(t, None)
                        if f is not None:
                            chosen, src = f, "forward_fallback"
                        else:
                            chosen = None; src = "none"
                if chosen is None:
                    chosen = {
                        "mask": np.zeros((H, W), bool),
                        "touch": False, "overlap": 0.0, "score": -1e9,
                        "area": 0, "area_penalty": 0.0, "huge_jump_rejected": False
                    }
                return chosen, src

            ch_gfp, src_gfp = choose(forward_gfp, backward_gfp)
            ch_bf,  src_bf  = choose(forward_bf,  backward_bf)

            combined.append({
                "time_point": t,
                "width": W, "height": H,

                # GFP
                "rle_gfp": rle_encode(ch_gfp["mask"]),
                "touches_border_gfp": ch_gfp["touch"],
                "source_gfp": src_gfp,
                "overlap_score_gfp": ch_gfp["overlap"],
                "smooth_score_gfp": ch_gfp["score"],
                "area_gfp": ch_gfp["area"],
                "area_penalty_gfp": ch_gfp.get("area_penalty", 0.0),
                "huge_jump_rejected_gfp": ch_gfp.get("huge_jump_rejected", False),

                # BF
                "rle_bf": rle_encode(ch_bf["mask"]),
                "touches_border_bf": ch_bf["touch"],
                "source_bf": src_bf,
                "overlap_score_bf": ch_bf["overlap"],
                "smooth_score_bf": ch_bf["score"],
                "area_bf": ch_bf["area"],
                "area_penalty_bf": ch_bf.get("area_penalty", 0.0),
                "huge_jump_rejected_bf": ch_bf.get("huge_jump_rejected", False),
            })

        pd.DataFrame(combined).to_csv(masks_csv_path, index=False)
        print(f"[Tracking] Saved dual-channel masks to: {masks_csv_path}")
    else:
        print(f"[track] Skipping tracking: {masks_csv_path.name} exists and --update_existing not set.")

    # =========================
    # Quantification pass
    # =========================
    mask_rows = pd.read_csv(masks_csv_path)
    H = int(mask_rows.iloc[0]['height'])
    W = int(mask_rows.iloc[0]['width'])

    time_series_data = []

    # EM refs per GFP track (single / subcells)
    ep_refs = {
        'single': {'ep1': None, 'ep2': None},
        '1': {'ep1': None, 'ep2': None},
        '2': {'ep1': None, 'ep2': None},
    }

    # per-subcell plot folders (GFP)
    cell_plot_folder_1 = os.path.join(plot_output_root, f"cell_{cell_id}_1")
    cell_plot_folder_2 = os.path.join(plot_output_root, f"cell_{cell_id}_2")
    os.makedirs(cell_plot_folder_1, exist_ok=True)
    os.makedirs(cell_plot_folder_2, exist_ok=True)



    
    
    

    # Iterate over timepoints
    for t in range(frame_number):
        print(f"[t={t}]")
        row = mask_rows.iloc[t]

        # Stop early if either channel touches the border (out-of-frame)
        if bool(row.get('touches_border_gfp', False)) or bool(row.get('touches_border_bf', False)):
            print(f"Cell {cell_id} touches border at t={t} (gfp={row.get('touches_border_gfp')}, bf={row.get('touches_border_bf')}). Stopping.")
            break

        # --- GFP full quantification (unchanged)
        img_gfp = imread(gfp_frame_path(t))
        mask_gfp = np.asarray(rle_decode(row['rle_gfp'], (H, W)), bool)
        rows_gfp = quantify_one_object(
            img_gfp, mask_gfp, id_suffix='', t=t,
            plot_dir=cell_plot_folder_gfp, ep_refs=ep_refs,
            gfp_min=gfp_min, gfp_max=gfp_max, cell_id=str(cell_id),
            do_plot=True, touches_border_flag=bool(row.get('touches_border_gfp', False)),
            allow_split=True
        )
        # tag GFP rows
        for r in rows_gfp:
            r['channel'] = 'gfp'
        time_series_data.extend(rows_gfp)

        # --- BF pattern-only (NEW)
        if 'rle_bf' in row and isinstance(row['rle_bf'], str) and len(row['rle_bf']) > 0:
            mask_bf = np.asarray(rle_decode(row['rle_bf'], (H, W)), bool)
        else:
            mask_bf = np.zeros((H, W), bool)

        if mask_bf.any():
            if os.path.exists(bf_frame_path(t)):
                img_bf = imread(bf_frame_path(t))
            else:
                #print('haha')
                # If BF frame missing, create a blank backdrop for overlay
                img_bf = np.zeros_like(img_gfp)

            bf_row = bf_pattern_only(
                img_bf,
                mask_bf,
                t,
                out_dir=cell_plot_folder_bf,
                cell_id=str(cell_id),          # <-- new (explicit) argument
                posterior_cutoff=0.5,          # optional; same default as before
                side_px=80                     # optional; same default as before
            )

            if bf_row is not None:
                time_series_data.append(bf_row)
        else:
            print(f"[info] Empty BF mask at t={t}; skipping BF pattern.")

    # 5) Save quantification table (mixed channels)
    df_cell = pd.DataFrame(time_series_data)
    out_csv = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_data.csv")
    df_cell.to_csv(out_csv, index=False)
    print(f"Saved (GFP + BF) quantification data for cell {cell_id} to {out_csv}")
