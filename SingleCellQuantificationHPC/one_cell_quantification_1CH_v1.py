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
import re
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
)

from bf_pattern import bf_pattern_only
from xcorr_utils import xcorr_best_of_six, save_xcorr_debug_figure


# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(description="Track a single cell in one channel (GFP or BF) and quantify signals.")
parser.add_argument('--cell_id', type=int, required=True, help='Cell ID to process')
parser.add_argument('--track_channel', choices=['bf', 'gfp'], default='bf',
                    help='Which single channel to process (controls quantification labels/paths only).')
parser.add_argument('--direction', choices=['forward', 'backward', 'both'], default='both',
                    help='Which tracking direction(s) to run.')
parser.add_argument('--experiment_path', type=str, required=True, help='Path to experiment')
parser.add_argument('--file_name', type=str, required=True, help='File name of .ims file (movie stem)')
parser.add_argument('--z_index', type=int, default=1, help="(Unused in 1-CH data; kept for compatibility)")
parser.add_argument('--min_area', type=int, default=2500, help="Minimum area threshold for cell filtering (first frame).")
parser.add_argument('--update_existing', action='store_true', help="Only update existing rows (skip tracking if masks CSV is valid).")

parser.add_argument('--xcorr_select',
                    choices=['off', 'fallback', 'primary'],
                    default='fallback',
                    help='Use rotation-aware cross-correlation selector: off | fallback | primary.')

parser.add_argument('--xcorr_fallback_ov',
                    type=float, default=0.35,
                    help='If overlap score < this threshold (or huge jump), switch to XCorr in fallback mode.')

parser.add_argument('--xcorr_angle_pad', type=float, default=15.0,
                    help='Angle sweep half-width in degrees for rotation-aware XCorr.')

parser.add_argument('--xcorr_angle_step', type=float, default=3.0,
                    help='Angle step in degrees for rotation-aware XCorr.')
parser.add_argument('--xcorr_debug', action='store_true',
                    help='Save per-frame debug figures for XCorr selection (PNG).')



args = parser.parse_args()

cell_id         = args.cell_id
working_dir     = args.experiment_path
file_name       = args.file_name
min_area        = args.min_area
z_index         = args.z_index   # kept for compatibility (not used in filenames)
direction_mode  = args.direction
update_existing = args.update_existing
track_channel   = args.track_channel.lower()  # 'bf' or 'gfp'
xcorr_mode       = args.xcorr_select
xcorr_fallback_ov = args.xcorr_fallback_ov
xcorr_angle_pad   = args.xcorr_angle_pad
xcorr_angle_step  = args.xcorr_angle_step
xcorr_debug = args.xcorr_debug




# =========================
# Paths
# =========================
output_frames_folder        = f"{working_dir}{file_name}/Frames_{file_name}"
output_masks_folder         = f"{working_dir}{file_name}/Masks_{file_name}"
output_tracked_cells_folder = f"{working_dir}{file_name}/TrackedCells_{file_name}"

# No subfolders under Masks_<file_name> anymore
os.makedirs(output_masks_folder, exist_ok=True)
os.makedirs(output_tracked_cells_folder, exist_ok=True)

# Output (plots / tables)
plot_output_root  = os.path.join(output_tracked_cells_folder, "cell_plots")
cell_plot_folder_gfp  = os.path.join(plot_output_root, f"cell_{cell_id}")
cell_plot_folder_bf   = os.path.join(plot_output_root, f"cell_{cell_id}_BF")
os.makedirs(plot_output_root, exist_ok=True)

xcorr_debug_root = os.path.join(output_tracked_cells_folder, "xcorr_debug", f"cell_{cell_id}")
xcorr_debug_dir_fwd = os.path.join(xcorr_debug_root, "forward")
xcorr_debug_dir_bwd = os.path.join(xcorr_debug_root, "backward")
if xcorr_debug:
    os.makedirs(xcorr_debug_dir_fwd, exist_ok=True)
    os.makedirs(xcorr_debug_dir_bwd, exist_ok=True)


# =========================
# Single-channel helpers (auto-detect c_<n> from masks)
# =========================
_DETECTED_CHANNEL_IDX: int | None = None

def _first_mask_path_for_channel(channel: str) -> str | None:
    """
    Find any one segmentation file inside Masks_<file_name>.
    New convention (single channel, no z):
      {file_name}_t_000_c_0_seg.tif  (or .npy)
    """
    if not os.path.isdir(output_masks_folder):
        return None
    entries = sorted(os.listdir(output_masks_folder))
    candidates: list[tuple[int, str]] = []
    for e in entries:
        # must belong to this movie, be a seg file, and be tif/npy
        if not e.startswith(f"{file_name}_t_"):
            continue
        if not (e.endswith(".tif") or e.endswith(".npy")):
            continue
        if "_seg." not in e:
            continue
        # expect ..._t_<nnn>_c_<ch>_seg.(tif|npy)
        m = re.search(r"_t_\d+_c_(\d+)_seg\.(tif|npy)$", e)
        if m:
            ch = int(m.group(1))
            candidates.append((ch, e))
        else:
            candidates.append((999, e))
    if not candidates:
        return None
    candidates.sort(key=lambda ce: (ce[0], ce[1]))  # lowest channel then name
    return os.path.join(output_masks_folder, candidates[0][1])

def _detect_channel_index_from_masks() -> int:
    p = _first_mask_path_for_channel("ignored")
    if p is None:
        raise FileNotFoundError(f"No segmentation files found in {output_masks_folder}")
    m = re.search(r"_c_(\d+)_seg\.(tif|npy)$", os.path.basename(p))
    if not m:
        raise RuntimeError(f"Cannot parse channel index from mask file name: {p}")
    return int(m.group(1))

def _get_channel_index() -> int:
    global _DETECTED_CHANNEL_IDX
    if _DETECTED_CHANNEL_IDX is None:
        _DETECTED_CHANNEL_IDX = _detect_channel_index_from_masks()
        print(f"[info] Detected movie channel index: c_{_DETECTED_CHANNEL_IDX}")
    return _DETECTED_CHANNEL_IDX

# =========================
# Path helper functions (use detected channel)
# =========================
def _seg_path_any_ext(t: int) -> str:
    ch = _get_channel_index()
    cand_tif = os.path.join(output_masks_folder, f"{file_name}_t_{t:03d}_c_{ch}_seg.tif")
    if os.path.exists(cand_tif):
        return cand_tif
    cand_npy = os.path.join(output_masks_folder, f"{file_name}_t_{t:03d}_c_{ch}_seg.npy")
    return cand_npy  # loader will raise if missing

def bf_seg_path(t: int) -> str:   # kept for compatibility
    return _seg_path_any_ext(t)

def gfp_seg_path(t: int) -> str:  # kept for compatibility
    return _seg_path_any_ext(t)

def seg_path(channel: str, t: int) -> str:  # channel ignored in 1-CH setup
    return _seg_path_any_ext(t)

def _frame_path(t: int) -> str:
    ch = _get_channel_index()
    return os.path.join(output_frames_folder, f"{file_name}_t_{t:03d}_c_{ch}.tif")

def gfp_frame_path(t: int) -> str:  # kept for compatibility
    return _frame_path(t)

def bf_frame_path(t: int) -> str:   # kept for compatibility
    return _frame_path(t)

# =========================
# Utilities
# =========================
def load_labeled_seg(channel, t):
    from skimage.measure import label as _label

    p = seg_path(channel, t)
    print(f"[load_labeled_seg] channel={channel} t={t} -> {p}")

    if not os.path.exists(p):
        raise FileNotFoundError(f"Segmentation path does not exist: {p}")

    raw = load_segmentation(p)

    # Normalize to labeled integers
    if raw.dtype == bool:
        raw = _label(raw.astype('uint8'))
    elif raw.ndim == 3 and raw.shape[-1] in (3, 4):
        raw = _label((raw[..., 0] > 0).astype('uint8'))
    elif not np.issubdtype(raw.dtype, np.integer):
        raw = _label((raw > 0).astype('uint8'))

    return raw

def GetFilteredRegions(min_area: int = 2500, channel: str = 'bf'):
    """
    Seed from the first available mask (channel argument kept for compatibility).
    """
    first_path = _first_mask_path_for_channel(channel)
    if first_path is None:
        raise FileNotFoundError(f"No segmentation files found in {output_masks_folder}.")
    first_mask   = load_segmentation(first_path)
    labeled_mask = to_labeled_current(first_mask)
    regions      = regionprops(labeled_mask)
    filtered     = [r for r in regions if r.area >= min_area]
    return first_mask, labeled_mask, filtered

def FindMovieMaxMin(channel: int, *, z_index: int | None = None, frames_dir: str | None = None, sample_step: int = 10):
    """
    Scan Frames_<file_name> for channel and return (p99.5, p1, frame_files).
    New 1-CH convention: filenames do NOT include z; pattern: {file_name}_t_###_c_<channel>.tif
    """
    from fnmatch import fnmatch
    folder = frames_dir or output_frames_folder

    # Patterns (no z)
    patterns = [
        f"{file_name}_t_???_c_{channel}.tif",
        f"{file_name}_t_*_c_{channel}.tif",
    ]

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
    m = int(2 * min(h, w))
    return int(np.clip(m, 20, 120))

def midpoints_from_mask_rc(mask_crop_bool):
    lab = label(mask_crop_bool.astype(np.uint8), connectivity=2)
    regs = regionprops(lab)
    if not regs:
        ys, xs = np.nonzero(mask_crop_bool)
        if ys.size == 0:
            return (0.0, 0.0), (0.0, 0.0)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        return (cy - 5.0, cx), (cy + 5.0, cx)

    r = regs[0]
    cy, cx = r.centroid
    theta = getattr(r, "orientation", 0.0) or 0.0
    vy, vx = np.cos(theta), -np.sin(theta)
    a_minor = max(float(getattr(r, "minor_axis_length", 0.0) or 0.0) / 2.0, 5.0)
    mid1_rc = (cy - a_minor * vy, cx - a_minor * vx)
    mid2_rc = (cy + a_minor * vy, cx + a_minor * vx)
    return mid1_rc, mid2_rc


def _union_many_bboxes(bboxes, H, W, pad=0):
    if not bboxes:
        return 0, H, 0, W
    r0 = max(0, min(b[0] for b in bboxes) - pad)
    r1 = min(H, max(b[1] for b in bboxes) + pad)
    c0 = max(0, min(b[2] for b in bboxes) - pad)
    c1 = min(W, max(b[3] for b in bboxes) + pad)
    return int(r0), int(r1), int(c0), int(c1)



# =========================
# Tracking (single-channel)
# =========================
def track_one_direction(t_seq, ref_start_mask, channel='bf',
                        first_threshold=0.5, next_threshold=0.7,
                        area_lambda=0.35, ratio_soft=1.3, ratio_hard=1.8, topk=5,
                        lock_first=True,
                        xcorr_mode: str = 'off',
                        xcorr_cfg: dict | None = None):
    """
    xcorr_mode: 'off' | 'fallback' | 'primary'
    xcorr_cfg:
      {
        'fallback_overlap_thr': float (default 0.35),
        'angle_pad_deg': float (default 15.0),
        'angle_step_deg': float (default 3.0),
        'num_singles': int (default 3),
        'num_pairs': int (default 3),
        'pair_pool_k': int (default 6),
        'pad_px': int (default 24),
        'min_area_abs': int | None
      }
    """
    cfg = {
        'fallback_overlap_thr': 0.35,
        'angle_pad_deg': 15.0,
        'angle_step_deg': 3.0,
        'num_singles': 3,
        'num_pairs': 3,
        'pair_pool_k': 6,
        'pad_px': 24,
        'min_area_abs': None,
    }
    if xcorr_cfg:
        cfg.update(xcorr_cfg)

    results = {}
    prev_mask = None
    prev_area = float(ref_start_mask.sum())

    for i, t in enumerate(t_seq):
        ref = ref_start_mask if i == 0 else prev_mask
        lab_cur = load_labeled_seg(channel, t)

        if i == 0 and lock_first:
            cm  = ref.copy()
            ov  = compute_overlap(ref, cm)
            sc  = ov
            pen = 0.0
            rej = False
            sel_mode = 'locked_first'
            xinfo = {}
        else:
            # ======= BASELINE (area-aware overlap) =======
            if lab_cur is not None and xcorr_mode != 'primary':
                thr = first_threshold if i == 0 else next_threshold
                cm0, ov0, sc0, pen0, rej0 = get_cell_mask_area_aware(
                    lab_cur, ref, prev_area,
                    threshold=thr, max_segments=2, topk=topk,
                    area_lambda=area_lambda, ratio_soft=ratio_soft, ratio_hard=ratio_hard
                )
            else:
                cm0, ov0, sc0, pen0, rej0 = ref.copy(), 1.0, 1.0, 0.0, False

            # ======= XCORR SELECTOR (primary or fallback) =======
            use_xcorr = (xcorr_mode == 'primary')
            if xcorr_mode == 'fallback' and (rej0 or ov0 < cfg['fallback_overlap_thr']):
                use_xcorr = True

            xinfo = {}
            if use_xcorr and lab_cur is not None:
                # t0 -> frame at previous timepoint if available, else same t
                t_prev = t_seq[i-1] if i > 0 else t
                # read frames (single-channel setup; bf_frame_path works for detected c_<idx>)
                if os.path.exists(bf_frame_path(t_prev)):
                    img_t0 = imread(bf_frame_path(t_prev))
                else:
                    img_t0 = imread(bf_frame_path(t))  # fallback
                img_t1 = imread(bf_frame_path(t)) if os.path.exists(bf_frame_path(t)) else img_t0

                min_area_abs = cfg['min_area_abs'] if cfg['min_area_abs'] is not None else max(50, int(0.2 * prev_area))
                xsel = xcorr_best_of_six(
                    ref, img_t0, lab_cur, img_t1,
                    num_singles=cfg['num_singles'],
                    num_pairs=cfg['num_pairs'],
                    pair_pool_k=cfg['pair_pool_k'],
                    min_area=min_area_abs,
                    angle_pad_deg=cfg['angle_pad_deg'],
                    angle_step_deg=cfg['angle_step_deg'],
                    pad_px=cfg['pad_px']
                )
                if xsel["best"] is not None:
                    cm1 = xsel["best_mask"]
                    ov1 = compute_overlap(ref, cm1)
                    sc1 = float(xsel["best"]["xcorr"])
                    pen1 = 0.0
                    rej1 = False
                    xinfo = {
                        "xcorr": sc1,
                        "xcorr_angle": xsel["best"]["angle_deg"],
                        "xcorr_shift_rc": xsel["best"]["shift_rc"],
                        "xcorr_type": xsel["best"]["type"],
                        "xcorr_labels": xsel["best"]["labels"],
                    }
                else:
                    cm1, ov1, sc1, pen1, rej1 = cm0, ov0, sc0, pen0, rej0  # fallback if xcorr failed

                if xcorr_mode == 'primary':
                    cm, ov, sc, pen, rej, sel_mode = cm1, ov1, sc1, pen1, rej1, 'xcorr_primary'
                else:  # fallback mode
                    if (ov0 < cfg['fallback_overlap_thr']) or rej0:
                        cm, ov, sc, pen, rej, sel_mode = cm1, ov1, sc1, pen1, rej1, 'xcorr_fallback'
                    else:
                        cm, ov, sc, pen, rej, sel_mode = cm0, ov0, sc0, pen0, rej0, 'area_primary'
                # After xsel computed and cm1/ov1/sc1 chosen
                if cfg.get('debug') and xsel["best"] is not None:
                    # choose frames (we already read them as img_t0/img_t1)
                    try:
                        outdir = cfg.get('debug_dir', None)
                        if outdir:
                            os.makedirs(outdir, exist_ok=True)
                            out_png = os.path.join(outdir, f"t_{t:03d}.png")
                            save_xcorr_debug_figure(
                                t=t,
                                t0_img=img_t0, t1_img=img_t1,
                                ref_mask=ref, best_mask=cm1,
                                candidates=xsel["candidates"],
                                out_png_path=out_png
                            )
                    except Exception as e:
                        print(f"[xcorr-debug] plot failed at t={t}: {e}")

            else:
                cm, ov, sc, pen, rej, sel_mode = cm0, ov0, sc0, pen0, rej0, 'area_primary'

        results[t] = {
            "mask": cm,
            "overlap": float(ov),
            "score": float(sc),
            "area": int(cm.sum()),
            "area_penalty": float(pen),
            "huge_jump_rejected": bool(rej),
            "touch": touches_border(cm),
            "selector_mode": sel_mode,
            **({} if not xinfo else xinfo),
        }
        prev_mask = cm
        prev_area = float(cm.sum())

    return results

# 
# =========================
# Main
# =========================
if __name__ == "__main__":
    from pathlib import Path

    # Detect single channel index from masks and discover frames
    CH_IDX = _get_channel_index()

    # Discover frames for detected channel (no z in new naming)
    if track_channel == 'gfp':
        gfp_max, gfp_min, gfp_frame_files = FindMovieMaxMin(CH_IDX)
        if len(gfp_frame_files) == 0:
            print(f"[error] No frames found for detected channel c_{CH_IDX}.")
            sys.exit(1)
        frame_number = len(gfp_frame_files)
    else:
        _, _, bf_frame_files  = FindMovieMaxMin(CH_IDX)
        if len(bf_frame_files) == 0:
            print(f"[error] No frames found for detected channel c_{CH_IDX}.")
            sys.exit(1)
        frame_number = len(bf_frame_files)

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
        f"track_channel={track_channel}  "
        f"-> need_to_track={need_to_track}"
    )

    if need_to_track:
        masks_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Seed from first available mask
        first_mask, labeled_mask, filtered_regions = GetFilteredRegions(min_area, channel=track_channel)
        cell = next((c for c in filtered_regions if c.label == cell_id), None)
        if cell is None:
            print(f"Cell {cell_id} not found in first frame masks.")
            sys.exit(1)
        initial_mask = (labeled_mask == cell_id)

        # 2) Tracking in detected channel only
        xcfg_fwd = {
            'fallback_overlap_thr': xcorr_fallback_ov,
            'angle_pad_deg': xcorr_angle_pad,
            'angle_step_deg': xcorr_angle_step,
            'num_singles': 3, 'num_pairs': 3, 'pair_pool_k': 6, 'pad_px': 24,
            'debug': xcorr_debug,
            'debug_dir': xcorr_debug_dir_fwd,
        }
        
        forward_sel = track_one_direction(
            range(frame_number), initial_mask,
            channel=track_channel, first_threshold=0.5, next_threshold=0.7, lock_first=False,
            xcorr_mode=xcorr_mode, xcorr_cfg=xcfg_fwd
        )
        
        backward_sel = None
        
        if direction_mode in ('backward', 'both'):
            seed_mask_backward = forward_sel[frame_number - 1]["mask"]
            xcfg_bwd = dict(xcfg_fwd, debug_dir=xcorr_debug_dir_bwd)
            backward_sel = track_one_direction(
                range(frame_number - 1, -1, -1), seed_mask_backward,
                channel=track_channel, first_threshold=0.5, next_threshold=0.7,
                xcorr_mode=xcorr_mode, xcorr_cfg=xcfg_bwd
            )



        # 3) Reconcile per frame & save masks CSV (populate only selected channel columns)
        combined = []

        H, W = initial_mask.shape

        def choose(forward_dict, backward_dict, t):
            if direction_mode == 'forward':
                chosen = forward_dict.get(t, None); src = "forward"
            elif direction_mode == 'backward':
                chosen = (backward_dict or {}).get(t, None); src = "backward"
            else:
                b = (backward_dict or {}).get(t, None)
                if b is not None:
                    chosen, src = b, "backward"
                else:
                    f = forward_dict.get(t, None)
                    chosen, src = (f, "forward_fallback") if f is not None else (None, "none")
            if chosen is None:
                chosen = {
                    "mask": np.zeros((H, W), bool),
                    "touch": False, "overlap": 0.0, "score": -1e9,
                    "area": 0, "area_penalty": 0.0, "huge_jump_rejected": False
                }
            return chosen, src

        for t in range(frame_number):
            ch_sel, src_sel = choose(forward_sel, backward_sel, t)

            # Initialize both channel fields with empties; fill the selected one.
            row = {
                "time_point": t,
                "width": W, "height": H,

                # GFP columns (empty defaults)
                "rle_gfp": "",
                "touches_border_gfp": False,
                "source_gfp": "",
                "overlap_score_gfp": 0.0,
                "smooth_score_gfp": -1e9,
                "area_gfp": 0,
                "area_penalty_gfp": 0.0,
                "huge_jump_rejected_gfp": False,

                # BF columns (empty defaults)
                "rle_bf": "",
                "touches_border_bf": False,
                "source_bf": "",
                "overlap_score_bf": 0.0,
                "smooth_score_bf": -1e9,
                "area_bf": 0,
                "area_penalty_bf": 0.0,
                "huge_jump_rejected_bf": False,
            }

            if track_channel == 'gfp':
                row.update({
                    "rle_gfp": rle_encode(ch_sel["mask"]),
                    "touches_border_gfp": ch_sel["touch"],
                    "source_gfp": src_sel,
                    "overlap_score_gfp": ch_sel["overlap"],
                    "smooth_score_gfp": ch_sel["score"],
                    "area_gfp": ch_sel["area"],
                    "area_penalty_gfp": ch_sel.get("area_penalty", 0.0),
                    "huge_jump_rejected_gfp": ch_sel.get("huge_jump_rejected", False),
                })
            else:
                row.update({
                    "rle_bf": rle_encode(ch_sel["mask"]),
                    "touches_border_bf": ch_sel["touch"],
                    "source_bf": src_sel,
                    "overlap_score_bf": ch_sel["overlap"],
                    "smooth_score_bf": ch_sel["score"],
                    "area_bf": ch_sel["area"],
                    "area_penalty_bf": ch_sel.get("area_penalty", 0.0),
                    "huge_jump_rejected_bf": ch_sel.get("huge_jump_rejected", False),
                })

            combined.append(row)

        pd.DataFrame(combined).to_csv(masks_csv_path, index=False)
        print(f"[Tracking] Saved {track_channel.upper()} masks to: {masks_csv_path}")
    else:
        print(f"[track] Skipping tracking: {masks_csv_path.name} exists and --update_existing not set.")

    # =========================
    # Quantification pass (single-channel)
    # =========================
    mask_rows = pd.read_csv(masks_csv_path)
    H = int(mask_rows.iloc[0]['height'])
    W = int(mask_rows.iloc[0]['width'])

    time_series_data = []

    if track_channel == 'gfp':
        # Prepare GFP plotting dirs
        os.makedirs(cell_plot_folder_gfp, exist_ok=True)

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

        # Get dynamic range for GFP (computed earlier)
        for t in range(len(mask_rows)):
            print(f"[t={t}] (GFP)")
            row = mask_rows.iloc[t]

            # Stop if GFP touches border
            if bool(row.get('touches_border_gfp', False)):
                print(f"Cell {cell_id} touches border at t={t} (GFP). Stopping.")
                break

            img_gfp = imread(gfp_frame_path(t))
            mask_gfp = np.asarray(rle_decode(row['rle_gfp'], (H, W)), bool) if isinstance(row['rle_gfp'], str) and row['rle_gfp'] else np.zeros((H, W), bool)

            rows_gfp = quantify_one_object(
                img_gfp, mask_gfp, id_suffix='', t=t,
                plot_dir=cell_plot_folder_gfp, ep_refs=ep_refs,
                gfp_min=gfp_min, gfp_max=gfp_max, cell_id=str(cell_id),
                do_plot=True, touches_border_flag=bool(row.get('touches_border_gfp', False)),
                allow_split=True
            )
            for r in rows_gfp:
                r['channel'] = 'gfp'
            time_series_data.extend(rows_gfp)

    else:
        # BF quantification (pattern-only pipeline)
        os.makedirs(cell_plot_folder_bf, exist_ok=True)

        for t in range(len(mask_rows)):
            print(f"[t={t}] (BF)")
            row = mask_rows.iloc[t]

            # Stop if BF touches border
            if bool(row.get('touches_border_bf', False)):
                print(f"Cell {cell_id} touches border at t={t} (BF). Stopping.")
                break

            mask_bf = np.asarray(rle_decode(row['rle_bf'], (H, W)), bool) if isinstance(row['rle_bf'], str) and row['rle_bf'] else np.zeros((H, W), bool)
            if not mask_bf.any():
                print(f"[info] Empty BF mask at t={t}; skipping.")
                continue

            if os.path.exists(bf_frame_path(t)):
                img_bf = imread(bf_frame_path(t))
            else:
                img_bf = np.zeros((H, W), dtype=np.uint16)

            bf_row = bf_pattern_only(
                img_bf,
                mask_bf,
                t,
                out_dir=cell_plot_folder_bf,
                cell_id=str(cell_id),
                posterior_cutoff=0.05,
                side_px=50
            )
            if bf_row is not None:
                bf_row['channel'] = 'bf'
                time_series_data.append(bf_row)

    # Save quantification table
    df_cell = pd.DataFrame(time_series_data)
    out_csv = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_data.csv")
    df_cell.to_csv(out_csv, index=False)
    print(f"Saved {track_channel.upper()} quantification data for cell {cell_id} to {out_csv}")
