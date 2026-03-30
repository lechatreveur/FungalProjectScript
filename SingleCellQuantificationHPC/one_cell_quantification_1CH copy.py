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
parser.add_argument(
    '--update_existing',
    nargs='?',                 # optional value
    const=-1,                  # present with no number → “old behavior” (recompute all)
    type=int,                  # present with a number → resume from that t
    help=("Update existing CSV. "
          "If used without a value, recompute all timepoints. "
          "If followed by an integer N, resume from timepoint N (seed from N-1).")
)


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

# Interpret update_existing
if args.update_existing is None:
    resume_t = None              # full run logic (or skip if CSV valid and not forced)
elif args.update_existing == -1:
    resume_t = 0                 # recompute from t=0 (old behavior)
else:
    resume_t = int(args.update_existing)  # resume from this timepoint



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

def _save_xcorr_debug_figure(
    t: int,
    t0_img: np.ndarray, t1_img: np.ndarray,
    ref_mask: np.ndarray, best_mask: np.ndarray,
    candidates: list[dict],
    out_png_path: str
):
    """
    Night-mode debug plot.

    - Supports best being a 'single' or 'pair_seq'.
    - Bar chart: all single candidates; if best is a pair, append a 'PAIR:La+Lb' bar with summed penalized score.
    - Title shows penalized and raw scores; for pairs these are sums over parts.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- Night mode styling ----------
    plt.style.use('dark_background')
    bg = "#0d1117"         # page background
    axbg = "#111827"       # axes background
    fg = "#e5e7eb"         # primary text
    subfg = "#9ca3af"      # secondary text
    gridc = "#374151"      # grid color

    H, W = t0_img.shape

    # --- collect bboxes (fallbacks included) for a common crop
    try:
        ref_bb = _bbox_from_mask(ref_mask, pad=20)
    except TypeError:
        ref_bb = _bbox_from_mask(ref_mask)
    c_bbs = []
    for c in candidates or []:
        bb = c.get("bbox", None)
        if not (isinstance(bb, (tuple, list)) and len(bb) == 4):
            bb = ref_bb
        c_bbs.append(tuple(bb))

    # union all bboxes
    def _union_bbox_simple(bboxes, H, W, pad=10):
        if not bboxes:
            return 0, H, 0, W
        rs0 = [b[0] for b in bboxes]
        rs1 = [b[1] for b in bboxes]
        cs0 = [b[2] for b in bboxes]
        cs1 = [b[3] for b in bboxes]
        r0 = max(0, min(rs0) - pad)
        r1 = min(H, max(rs1) + pad)
        c0 = max(0, min(cs0) - pad)
        c1 = min(W, max(cs1) + pad)
        return r0, r1, c0, c1

    r0, r1, c0, c1 = _union_bbox_simple([ref_bb] + c_bbs, H, W, pad=10)

    # crops & masks
    R = t0_img[r0:r1, c0:c1]
    I = t1_img[r0:r1, c0:c1]
    Mref = ref_mask[r0:r1, c0:c1].astype(bool)
    Mbest = best_mask[r0:r1, c0:c1].astype(bool)

    # ---------- Build labels + scores ----------
    # candidates are singles: each has .labels [id], .xcorr (penalized), optionally .xcorr_raw, .penalty
    labs, scores, raw_scores, pens = [], [], [], []
    for c in (candidates or []):
        lbls = c.get("labels", [])
        lbl_txt = "+".join([f"L{int(l)}" for l in lbls]) if lbls else "?"
        labs.append(lbl_txt)
        scores.append(float(c.get("xcorr", float('-inf'))))
        raw_scores.append(float(c.get("xcorr_raw", float('nan'))))
        pens.append(float(c.get("penalty", float('nan'))))

    # chosen details; supports single or pair_seq
    best = None
    if candidates:
        # the caller already decided best, but we’ll use the provided dict in title
        pass
    # We use the best fields passed in xsel["best"] at the callsite; infer here from inputs:
    # The function signature only receives best_mask, not best dict; however in your callsite
    # you pass `candidates` and later set title fields from a `best` object.
    # To keep compatibility, expect caller to still pass a 'best-like' dict inside candidates list,
    # BUT we’ll also allow reading from a separate variable by scanning for a 'selected' flag.
    # Fallback: pick the max penalized single as "best_single" for title if nothing else is available.
    best_is_pair = False
    best_idx = None
    if candidates:
        # find index of max penalized among candidates for chart ordering
        best_idx = int(np.nanargmax(np.array(scores)))
        best = candidates[best_idx]

    # If the caller provided a pair winner, they’d pass it separately to title previously.
    # We’ll try to detect it from an attached attribute on Mbest via length of labels in candidates;
    # safer: accept that the title may not show pair raw if not provided. To fully support pairs,
    # we let the bar chart append a synthetic PAIR bar if we can see 'best_pair' in any candidate dict.
    # Here we check whether any dict has 'type' == 'pair_seq' and highest score via 'xcorr'.
    pair_candidates = [c for c in (candidates or []) if c.get("type") == "pair_seq"]
    best_pair = None
    if pair_candidates:
        best_pair = max(pair_candidates, key=lambda d: float(d.get("xcorr", float('-inf'))))
        best_is_pair = True

    # Create chart data (ordered by score desc)
    order = list(np.argsort(scores))[::-1] if scores else []
    labs_sorted = [labs[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    # If a pair winner exists (from your new logic), append a synthetic bar at the top
    pair_label = None
    pair_score = None
    pair_raw = None
    pair_pen = None
    if best_pair is not None:
        pair_lbls = best_pair.get("labels", [])
        pair_label = "PAIR:" + "+".join([f"L{int(l)}" for l in pair_lbls]) if pair_lbls else "PAIR:?+?"
        pair_score = float(best_pair.get("xcorr", float('nan')))        # SUM penalized
        pair_raw   = float(best_pair.get("xcorr_raw", float('nan')))    # SUM raw
        pair_pen   = float(best_pair.get("penalty", float('nan')))
        # Prepend to lists for visibility
        labs_sorted = [pair_label] + labs_sorted
        scores_sorted = [pair_score] + scores_sorted

    # ---------- Figure ----------
    fig = plt.figure(figsize=(12, 8), facecolor=bg)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], wspace=0.15, hspace=0.22)

    # t0 crop
    ax0 = fig.add_subplot(gs[0, 0], facecolor=axbg)
    im0 = ax0.imshow(R, cmap='gray')
    ax0.contour(Mref, levels=[0.5], linewidths=1.5, colors="#60a5fa")
    ax0.set_title(f"t0 (ref) crop  [t={t-1 if t > 0 else t}]", color=fg, fontsize=12, pad=6)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.grid(False)

    # t1 crop with chosen overlay
    ax1 = fig.add_subplot(gs[0, 1], facecolor=axbg)
    im1 = ax1.imshow(I, cmap='gray')
    ax1.contour(Mbest, levels=[0.5], linewidths=1.5, colors="#34d399")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.grid(False)

    # Build title for chosen
    if best_pair is not None:
        title_best = f"{pair_label}"
        penal = pair_score if pair_score is not None else float('nan')
        raw   = pair_raw   if pair_raw   is not None else float('nan')
        penf  = pair_pen   if pair_pen   is not None else float('nan')
    else:
        # fall back to best single among candidates
        title_best = f"S:{labs_sorted[0]}" if labs_sorted else "n/a"
        penal = scores_sorted[0] if scores_sorted else float('nan')
        raw   = float(best.get("xcorr_raw", float('nan'))) if best is not None else float('nan')
        penf  = float(best.get("penalty", float('nan'))) if best is not None else float('nan')

    ax1.set_title(
        f"t1 crop — chosen: {title_best}\n"
        f"penalized={penal:.3f}   raw={raw:.3f}   penalty={penf:.3f}",
        color=fg, fontsize=12, pad=6
    )

    # bar plot (singles + optional pair)
    ax2 = fig.add_subplot(gs[1, :], facecolor=axbg)
    if scores_sorted:
        x = np.arange(len(scores_sorted))
        ax2.bar(x, scores_sorted)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labs_sorted, rotation=35, ha='right', color=fg)
        ax2.set_ylabel("penalized score", color=fg)
        ax2.set_title(
            f"Candidates — singles={len(scores)}"
            + (f", + pair" if best_pair is not None else ""),
            color=fg
        )
        ax2.grid(True, color=gridc, alpha=0.35, linestyle='--', linewidth=0.6)
        for spine in ax2.spines.values():
            spine.set_color(subfg)
        ax2.tick_params(colors=subfg)
    else:
        ax2.text(0.5, 0.5, "No candidates", ha='center', va='center', color=fg, fontsize=12)
        ax2.axis('off')

    # Overall fig styling and save
    for ax in [ax0, ax1]:
        for spine in ax.spines.values():
            spine.set_color(subfg)

    fig.patch.set_facecolor(bg)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=160, facecolor=bg)
    plt.close(fig)




# =========================
# XCorr selector (6 candidates: 3 singles + 3 unions)
# =========================
from itertools import combinations
from typing import Tuple, List, Dict, Any
from skimage.transform import rotate as _rotate
from skimage.measure import regionprops, label as _label
from numpy.fft import rfft2, irfft2
import numpy as np

def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    """
    Inclusive-exclusive bbox (r0, r1, c0, c1) around True pixels.
    Optional 'pad' expands bbox by that many pixels and clamps to mask shape.
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    if pad:
        r0 = max(0, r0 - pad)
        r1 = min(H, r1 + pad)
        c0 = max(0, c0 - pad)
        c1 = min(W, c1 + pad)
    return (r0, r1, c0, c1)


def _union_bbox(b0, b1, H, W, extra_pad=0):
    r0 = max(0, min(b0[0], b1[0]) - extra_pad)
    r1 = min(H, max(b0[1], b1[1]) + extra_pad)
    c0 = max(0, min(b0[2], b1[2]) - extra_pad)
    c1 = min(W, max(b0[3], b1[3]) + extra_pad)
    return r0, r1, c0, c1

def _masked_norm(img: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(img, dtype=np.float32)
    vals = img[m].astype(np.float32)
    mu, sd = float(vals.mean()), float(vals.std() + eps)
    out = (img.astype(np.float32) - mu) * m
    out /= sd
    return out

def _phase_corr_peak(a: np.ndarray, b: np.ndarray, eps: float = 1e-12):
    Fa, Fb = rfft2(a), rfft2(b)
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), eps)
    c = irfft2(R, s=a.shape)
    peak_idx = np.unravel_index(np.argmax(c), c.shape)
    peak_val = float(c[peak_idx])
    sh = np.array(peak_idx, dtype=float)
    for d, n in enumerate(a.shape):
        if sh[d] > n // 2:
            sh[d] -= n
    return peak_val, (float(sh[0]), float(sh[1]))

def _mask_union_from_labels(labeled: np.ndarray, labels: List[int]) -> np.ndarray:
    m = np.zeros_like(labeled, dtype=bool)
    for lb in labels:
        if lb:
            m |= (labeled == lb)
    return m

def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    lab = _label(mask.astype(np.uint8), connectivity=2)
    regs = regionprops(lab)
    if not regs:
        ys, xs = np.nonzero(mask)
        return (float(ys.mean()) if ys.size else 0.0, float(xs.mean()) if xs.size else 0.0)
    cy, cx = regs[0].centroid
    return float(cy), float(cx)

def _orientation_deg(mask: np.ndarray) -> float:
    lab = _label(mask.astype(np.uint8), connectivity=2)
    regs = regionprops(lab)
    if not regs or getattr(regs[0], "orientation", None) is None:
        return 0.0
    return float(np.degrees(regs[0].orientation))

def _max_xcorr_over_angles(ref_img, ref_mask, cand_img, cand_mask, angles_deg, pad_px=20):
    H, W = ref_img.shape
    b_ref = _bbox_from_mask(ref_mask, pad_px)
    b_cnd = _bbox_from_mask(cand_mask, pad_px)
    r0, r1, c0, c1 = _union_bbox(b_ref, b_cnd, H, W, extra_pad=0)

    R = ref_img[r0:r1, c0:c1]
    Mref = ref_mask[r0:r1, c0:c1].astype(bool)
    Rn = _masked_norm(R, Mref)

    best = {
        "xcorr": -1e9,
        "angle_deg": 0.0,
        "shift_rc": (0.0, 0.0),
        "bbox": (r0, r1, c0, c1),
        "best_rot_mask": None,   # NEW: mask in crop coords, before shift
    }

    C_full  = cand_img[r0:r1, c0:c1]
    Mc_full = cand_mask[r0:r1, c0:c1].astype(bool)

    for ang in angles_deg:
        C_rot  = _rotate(C_full,  angle=ang, resize=False, preserve_range=True, order=1)
        Mc_rot = _rotate(Mc_full.astype(float), angle=ang, resize=False, preserve_range=True, order=0) > 0.5
        if Mc_rot.sum() == 0 or Mref.sum() == 0:
            continue
        Cn = _masked_norm(C_rot, Mc_rot)
        val, sh = _phase_corr_peak(Rn, Cn)
        if val > best["xcorr"]:
            best.update({
                "xcorr": float(val),
                "angle_deg": float(ang),
                "shift_rc": (float(sh[0]), float(sh[1])),
                "best_rot_mask": Mc_rot.copy(),  # save unshifted, rotated mask
            })
    return best


def xcorr_best_of_six(
    t0_mask: np.ndarray, t0_bf: np.ndarray,
    t1_labeled: np.ndarray, t1_bf: np.ndarray,
    *,
    # knobs you actually use
    min_area: int = 50,
    angle_pad_deg: float = 15.0,
    angle_step_deg: float = 3.0,
    pad_px: int = 20,
    shift_limit_px: float = 1000.0,
    remaining_min_area_frac: float = 0.05,  # <-- NEW: default 5% of original t0 area
    **_legacy,                               # <-- keeps compatibility with old kwargs
) -> Dict[str, Any]:
    """
    New logic (no singles/pairs presumption):
      - Split t0 into two lobes along its minor axis -> centers c1, c2.
      - Choose best single A vs FULL ref (penalized by area vs M and distance to c1).
      - Hide A's used footprint from ref.
      - If remaining ref area is small -> final is A.
      - Else choose best single B vs REMAINING ref (area vs M, distance to c2).
      - Final score = SUM of penalized scores (A + B). If no B, it's just A.
    """
    from skimage.transform import rotate as _rotate
    from typing import Tuple, Dict, Any, List

    # ---------- helpers ----------
    def _shift_mask_no_wrap(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
        out = np.zeros_like(m, dtype=bool)
        H, W = m.shape
        sy0 = max(0, -dy); sy1 = min(H, H - max(0, dy))
        sx0 = max(0, -dx); sx1 = min(W, W - max(0, dx))
        ty0 = sy0 + dy;    ty1 = sy1 + dy
        tx0 = sx0 + dx;    tx1 = sx1 + dx
        if sy1 > sy0 and sx1 > sx0:
            out[ty0:ty1, tx0:tx1] = m[sy0:sy1, sx0:sx1]
        return out

    def _area_penalty(area_cand: int, area_ref_M: int) -> float:
        M = max(1, int(area_ref_M))
        p = 1.0 - abs(int(area_cand) - M) / float(M)
        return 0.0 if p < 0.0 else (1.0 if p > 1.0 else float(p))

    def _distance_penalty(dist_val: float, limit: float) -> float:
        if limit is None or limit <= 0.0:
            return 1.0
        p = 1.0 - (float(dist_val) / float(limit))
        return 0.0 if p < 0.0 else (1.0 if p > 1.0 else float(p))

    def _angles_for(mask, ref_orient_, angle_pad, angle_step):
        cand_orient = _orientation_deg(mask)
        base = float(ref_orient_ - cand_orient)
        n_steps = int(np.ceil(angle_pad / max(1e-6, angle_step)))
        angs = [base + k * angle_step for k in range(-n_steps, n_steps + 1)]
        if 0.0 not in angs:
            angs.append(0.0)
        return angs

    def _split_ref_into_two_lobes(ref_mask_bool: np.ndarray):
        lab = _label(ref_mask_bool.astype(np.uint8), connectivity=2)
        regs = regionprops(lab)
        if not regs:
            H, W = ref_mask_bool.shape
            c = (H/2, W/2)
            return ref_mask_bool.copy(), np.zeros_like(ref_mask_bool), c, c
        r = regs[0]
        cy, cx = r.centroid
        theta = float(getattr(r, "orientation", 0.0) or 0.0)  # radians
        vy, vx = np.cos(theta), -np.sin(theta)                # minor-axis dir
        ys, xs = np.nonzero(ref_mask_bool)
        signed = (ys - cy) * vy + (xs - cx) * vx
        sideA_idx = signed < 0
        sideB_idx = ~sideA_idx
        M_A = np.zeros_like(ref_mask_bool, bool); M_A[ys[sideA_idx], xs[sideA_idx]] = True
        M_B = np.zeros_like(ref_mask_bool, bool); M_B[ys[sideB_idx], xs[sideB_idx]] = True
        c1 = _centroid(M_A) if M_A.any() else (cy, cx)
        c2 = _centroid(M_B) if M_B.any() else (cy, cx)
        return M_A, M_B, c1, c2

    def _evaluate_single_against_ref(
        label_id: int,
        ref_img: np.ndarray, ref_mask: np.ndarray,
        cand_img: np.ndarray, labeled_t1: np.ndarray,
        ref_orient_deg: float,
        area_baseline_M: int,
        center_for_distance: Tuple[float, float],   # c1 for A-pass, c2 for B-pass
        centroids_map: Dict[int, Tuple[float, float]],
    ) -> Dict[str, Any] | None:
        """Compute raw xcorr and penalized score for one labeled segment vs a ref mask."""
        cand_mask = (labeled_t1 == int(label_id))
        if not cand_mask.any():
            return None

        # raw xcorr by angle sweep + phase correlation
        angs = _angles_for(cand_mask, ref_orient_deg, angle_pad_deg, angle_step_deg)
        xres = _max_xcorr_over_angles(ref_img, ref_mask, cand_img, cand_mask, angs, pad_px=pad_px)
        xcorr_raw = float(xres["xcorr"])

        # penalties:
        #  - area vs ORIGINAL M
        size_pen = _area_penalty(int(cand_mask.sum()), area_baseline_M)
        #  - distance penalty vs selected center (centroid distance)
        cyx = centroids_map.get(int(label_id), None)
        if cyx is None:
            rr = regionprops((cand_mask.astype(np.uint8)))[0]
            cyx = (float(rr.centroid[0]), float(rr.centroid[1]))
        dist_val = float(np.hypot(cyx[0] - center_for_distance[0], cyx[1] - center_for_distance[1]))
        dist_pen = _distance_penalty(dist_val, shift_limit_px)

        xcorr_pen = xcorr_raw * size_pen * dist_pen

        return {
            "label": int(label_id),
            "xcorr_raw": xcorr_raw,
            "xcorr": xcorr_pen,
            "size_penalty": float(size_pen),
            "distance_penalty": float(dist_pen),
            "distance_to_center": float(dist_val),
            "angle_deg": float(xres["angle_deg"]),
            "shift_rc": (float(xres["shift_rc"][0]), float(xres["shift_rc"][1])),  # used for footprint placement
            "bbox": tuple(xres["bbox"]),
            "cand_mask": cand_mask,  # full-res bool mask
        }

    def _used_footprint_in_full(
        cand_mask_full: np.ndarray,
        bbox: Tuple[int,int,int,int],
        angle_deg: float,
        shift_rc: Tuple[float, float],
        H: int, W: int
    ) -> np.ndarray:
        """Rotate cand mask inside bbox, integer-shift, and return placed footprint in full image."""
        r0, r1, c0, c1 = bbox
        Mc = cand_mask_full[r0:r1, c0:c1].astype(bool)
        if Mc.size == 0:
            return np.zeros((H, W), bool)
        Mc_rot = _rotate(Mc.astype(float), angle=angle_deg, resize=False, preserve_range=True, order=0) > 0.5
        idy, idx = int(round(shift_rc[0])), int(round(shift_rc[1]))
        used_crop = _shift_mask_no_wrap(Mc_rot, idy, idx)
        out = np.zeros((H, W), bool)
        out[r0:r1, c0:c1] = used_crop
        return out

    # ---------- main ----------
    H, W = t0_mask.shape
    ref_orient = _orientation_deg(t0_mask)
    M_full = int(t0_mask.sum())
    if M_full == 0:
        return {"best_mask": np.zeros_like(t0_mask, bool), "best": None, "candidates": [], "meta": {"reason": "empty_ref"}}

    # split ref -> c1, c2
    lobeA, lobeB, c1, c2 = _split_ref_into_two_lobes(t0_mask.astype(bool))

    # gather candidate labels + centroids (restrict to within shift_limit_px of c1 or c2)
    regs = regionprops(t1_labeled)
    
    labels: List[int] = []
    centroids_map: Dict[int, Tuple[float, float]] = {}
    
    kept, dropped = 0, 0
    for r in regs:
        if r.label == 0 or r.area < min_area:
            continue
    
        cy, cx = float(r.centroid[0]), float(r.centroid[1])
        d1 = float(np.hypot(cy - c1[0], cx - c1[1]))
        d2 = float(np.hypot(cy - c2[0], cx - c2[1]))
    
        # keep only if near either center
        if min(d1, d2) <= float(shift_limit_px):
            labels.append(int(r.label))
            centroids_map[int(r.label)] = (cy, cx)
            kept += 1
        else:
            dropped += 1
    
    if not labels:
        return {
            "best_mask": np.zeros_like(t0_mask, bool),
            "best": None,
            "candidates": [],
            "meta": {
                "reason": "no_labels_within_shift_limit",
                "M_full": int(M_full),
                "min_area": int(min_area),
                "shift_limit_px": float(shift_limit_px),
                "c1": (float(c1[0]), float(c1[1])),
                "c2": (float(c2[0]), float(c2[1])),
                "kept": kept,
                "dropped": dropped,
            },
        }

    # ---- Pass A: best single vs FULL reference
    singles_A: List[Dict[str, Any]] = []
    for lb in labels:
        res = _evaluate_single_against_ref(
            lb, t0_bf, t0_mask, t1_bf, t1_labeled,
            ref_orient_deg=ref_orient,
            area_baseline_M=M_full,          # area penalty is 1 - |area(A)-M|/M
            center_for_distance=c1,          # distance penalty uses c1
            centroids_map=centroids_map
        )
        if res is not None:
            singles_A.append(res)
    
    if not singles_A:
        return {
            "best_mask": np.zeros_like(t0_mask, bool),
            "best": None,
            "candidates": [],
            "meta": {"reason": "no_valid_single_A", "M_full": int(M_full)}
        }
    
    # pick best A by penalized score (already includes area & distance penalties)
    singles_A.sort(key=lambda d: d["xcorr"], reverse=True)
    A = singles_A[0]
    
    # hide A footprint from ref (rotate->integer shift inside bbox, then stamp in full image)
    usedA_full = _used_footprint_in_full(
        A["cand_mask"], A["bbox"], A["angle_deg"], A["shift_rc"], H, W
    )
    ref_remaining = t0_mask.astype(bool) & (~usedA_full)
    M_rem = int(ref_remaining.sum())
    min_rem = max(1, int(round(remaining_min_area_frac * M_full)))
    
    # If remaining ref is too small -> choose A only
    if M_rem <= min_rem:
        best_mask = (t1_labeled == A["label"])
        best = {
            "type": "single",
            "labels": [int(A["label"])],
            "xcorr_raw": float(A["xcorr_raw"]),
            "xcorr": float(A["xcorr"]),  # final (single) score
            "penalty": float(A["size_penalty"] * A["distance_penalty"]),
            "angle_deg": float(A["angle_deg"]),
            "shift_rc": (float(A["shift_rc"][0]), float(A["shift_rc"][1])),
            "bbox": tuple(A["bbox"]),
            "distance_to_center": float(A["distance_to_center"]),  # distance to c1
        }
        return {
            "best_mask": best_mask.astype(bool),
            "best": best,
            "candidates": singles_A,  # A-pass ranking for debug
            "meta": {
                "mode": "single_only",
                "M_full": int(M_full),
                "M_remaining": int(M_rem),
                "remaining_min_area_frac": float(remaining_min_area_frac),
                "c1": (float(c1[0]), float(c1[1])),
                "c2": (float(c2[0]), float(c2[1])),
                "shift_limit_px": float(shift_limit_px),
            },
        }
    
    # ---- Pass B: best single vs REMAINING reference
    # Check c2 vs the centroid of the remaining ref (for your logging/debug)
    c2_check = _centroid(ref_remaining.astype(np.uint8))
    c2_delta = float(np.hypot(c2_check[0] - c2[0], c2_check[1] - c2[1]))
    
    singles_B: List[Dict[str, Any]] = []
    for lb in labels:
        if lb == A["label"]:
            continue  # do not reuse A
        res = _evaluate_single_against_ref(
            lb, t0_bf, ref_remaining.astype(np.uint8), t1_bf, t1_labeled,
            ref_orient_deg=ref_orient,
            area_baseline_M=M_full,          # area penalty still vs ORIGINAL M
            center_for_distance=c2,          # distance penalty uses c2
            centroids_map=centroids_map
        )
        if res is not None:
            singles_B.append(res)
    
    if not singles_B:
        # fallback to A only if B cannot be found
        best_mask = (t1_labeled == A["label"])
        best = {
            "type": "single",
            "labels": [int(A["label"])],
            "xcorr_raw": float(A["xcorr_raw"]),
            "xcorr": float(A["xcorr"]),  # single score
            "penalty": float(A["size_penalty"] * A["distance_penalty"]),
            "angle_deg": float(A["angle_deg"]),
            "shift_rc": (float(A["shift_rc"][0]), float(A["shift_rc"][1])),
            "bbox": tuple(A["bbox"]),
            "distance_to_center": float(A["distance_to_center"]),  # vs c1
        }
        return {
            "best_mask": best_mask.astype(bool),
            "best": best,
            "candidates": singles_A,
            "meta": {
                "mode": "single_only_fallback",
                "M_full": int(M_full),
                "M_remaining": int(M_rem),
                "c2_vs_rem_centroid_delta": float(c2_delta),
                "shift_limit_px": float(shift_limit_px),
            },
        }
    
    # pick best B by penalized score
    singles_B.sort(key=lambda d: d["xcorr"], reverse=True)
    B = singles_B[0]
    
    # ----- Compose final "pair" result -----
    # final score is **sum** of penalized parts (NOT average)
    xcorr_raw_sum = float(A["xcorr_raw"] + B["xcorr_raw"])
    xcorr_sum     = float(A["xcorr"]     + B["xcorr"])
    combined_pen  = (xcorr_sum / xcorr_raw_sum) if xcorr_raw_sum > 0.0 else 0.0
    
    best_mask = ((t1_labeled == A["label"]) | (t1_labeled == B["label"]))
    best = {
        "type": "pair",
        "labels": [int(A["label"]), int(B["label"])],
        "xcorr_raw": xcorr_raw_sum,      # sum of raw parts
        "xcorr": xcorr_sum,              # sum of penalized parts (FINAL SCORE)
        "penalty": float(combined_pen),  # for plotting/debug
        "angle_deg": float((A["angle_deg"] + B["angle_deg"]) / 2.0),
        "shift_rc": (
            float((A["shift_rc"][0] + B["shift_rc"][0]) / 2.0),
            float((A["shift_rc"][1] + B["shift_rc"][1]) / 2.0),
        ),
        "bbox": (
            min(A["bbox"][0], B["bbox"][0]),
            max(A["bbox"][1], B["bbox"][1]),
            min(A["bbox"][2], B["bbox"][2]),
            max(A["bbox"][3], B["bbox"][3]),
        ),
        "parts": [
            {
                "label": int(A["label"]),
                "xcorr_part_raw": float(A["xcorr_raw"]),
                "xcorr_part": float(A["xcorr"]),
                "size_penalty": float(A["size_penalty"]),
                "distance_penalty": float(A["distance_penalty"]),
                "distance_to_center": float(A["distance_to_center"]),  # vs c1
                "angle_deg": float(A["angle_deg"]),
                "shift_rc": (float(A["shift_rc"][0]), float(A["shift_rc"][1])),
                "bbox": tuple(A["bbox"]),
            },
            {
                "label": int(B["label"]),
                "xcorr_part_raw": float(B["xcorr_raw"]),
                "xcorr_part": float(B["xcorr"]),
                "size_penalty": float(B["size_penalty"]),
                "distance_penalty": float(B["distance_penalty"]),
                "distance_to_center": float(B["distance_to_center"]),  # vs c2
                "angle_deg": float(B["angle_deg"]),
                "shift_rc": (float(B["shift_rc"][0]), float(B["shift_rc"][1])),
                "bbox": tuple(B["bbox"]),
            },
        ],
    }
    
    return {
        "best_mask": best_mask.astype(bool),
        "best": best,
        "candidates": singles_A + singles_B,  # optional: expose both lists for debugging
        "meta": {
            "mode": "pair",
            "M_full": int(M_full),
            "M_remaining_after_A": int(M_rem),
            "remaining_min_area_frac": float(remaining_min_area_frac),
            "c1": (float(c1[0]), float(c1[1])),
            "c2": (float(c2[0]), float(c2[1])),
            "c2_vs_rem_centroid_delta": float(c2_delta),
            "shift_limit_px": float(shift_limit_px),
        },
    }




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
        'min_area_abs': int | None,
        'shift_limit_px': float (default 10.0),
        'debug': bool,
        'debug_dir': str
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
        'shift_limit_px': 10.0,
        'debug': False,
        'debug_dir': None,
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
            img_t0 = None
            img_t1 = None
            xsel = None

            if use_xcorr and lab_cur is not None:
                # Adjacent frame for t0: forward(step>0)->t-1, backward(step<0)->t+1
                step = getattr(t_seq, 'step', 1)
                movie_prev = t - (1 if step > 0 else -1)

                prev_path = bf_frame_path(movie_prev)
                cur_path  = bf_frame_path(t)
                if os.path.exists(prev_path):
                    img_t0 = imread(prev_path)
                else:
                    img_t0 = imread(cur_path)

                if os.path.exists(cur_path):
                    img_t1 = imread(cur_path)
                else:
                    img_t1 = img_t0  # worst-case fallback

                min_area_abs = cfg['min_area_abs'] if cfg['min_area_abs'] is not None else max(50, int(0.2 * prev_area))
                xsel = xcorr_best_of_six(
                    ref, img_t0, lab_cur, img_t1,
                    num_singles=cfg['num_singles'],
                    num_pairs=cfg['num_pairs'],
                    pair_pool_k=cfg['pair_pool_k'],
                    min_area=min_area_abs,
                    angle_pad_deg=cfg['angle_pad_deg'],
                    angle_step_deg=cfg['angle_step_deg'],
                    pad_px=cfg['pad_px'],
                    shift_limit_px=cfg['shift_limit_px'],
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
                    cm1, ov1, sc1, pen1, rej1 = cm0, ov0, sc0, pen0, rej0

                if xcorr_mode == 'primary':
                    cm, ov, sc, pen, rej, sel_mode = cm1, ov1, sc1, pen1, rej1, 'xcorr_primary'
                else:
                    if (ov0 < cfg['fallback_overlap_thr']) or rej0:
                        cm, ov, sc, pen, rej, sel_mode = cm1, ov1, sc1, pen1, rej1, 'xcorr_fallback'
                    else:
                        cm, ov, sc, pen, rej, sel_mode = cm0, ov0, sc0, pen0, rej0, 'area_primary'

            else:
                # Selection stays with area/overlap
                cm, ov, sc, pen, rej, sel_mode = cm0, ov0, sc0, pen0, rej0, 'area_primary'

            # ======= Debug figure (always, if requested) =======
            if cfg.get('debug'):
                try:
                    # Ensure we have frames for plotting even if XCorr wasn't used.
                    if img_t0 is None or img_t1 is None:
                        step = getattr(t_seq, 'step', 1)
                        movie_prev = t - (1 if step > 0 else -1)
                        prev_path = bf_frame_path(movie_prev)
                        cur_path  = bf_frame_path(t)
                        if os.path.exists(prev_path):
                            img_t0 = imread(prev_path)
                        else:
                            img_t0 = imread(cur_path)
                        img_t1 = imread(cur_path) if os.path.exists(cur_path) else img_t0

                    # Run a lightweight XCorr purely for visualization if we didn't already
                    xsel_dbg = xsel
                    if xsel_dbg is None and lab_cur is not None:
                        min_area_abs_dbg = cfg['min_area_abs'] if cfg['min_area_abs'] is not None else max(50, int(0.2 * prev_area))
                        xsel_dbg = xcorr_best_of_six(
                            ref, img_t0, lab_cur, img_t1,
                            num_singles=cfg['num_singles'],
                            num_pairs=cfg['num_pairs'],
                            pair_pool_k=cfg['pair_pool_k'],
                            min_area=min_area_abs_dbg,
                            angle_pad_deg=cfg['angle_pad_deg'],
                            angle_step_deg=cfg['angle_step_deg'],
                            pad_px=cfg['pad_px'],
                            shift_limit_px=cfg['shift_limit_px'],
                        )

                    outdir = cfg.get('debug_dir', None)
                    if outdir:
                        os.makedirs(outdir, exist_ok=True)
                        out_png = os.path.join(outdir, f"t_{t:03d}.png")

                        # For the figure: prefer the xcorr best mask if available; else use the selected mask.
                        if xsel_dbg is not None and xsel_dbg.get("best") is not None:
                            dbg_best_mask = xsel_dbg["best_mask"]
                            cand_list_raw = xsel_dbg.get("candidates") or []
                        else:
                            dbg_best_mask = cm
                            cand_list_raw = []

                        # Make candidate dicts plot-friendly by ensuring a 'labels' list
                        cand_list = []
                        for c in cand_list_raw:
                            cc = dict(c)
                            if "labels" not in cc:
                                if "label" in cc:
                                    cc["labels"] = [cc["label"]]
                                else:
                                    cc["labels"] = []
                            cand_list.append(cc)

                        _save_xcorr_debug_figure(
                            t=t,
                            t0_img=img_t0, t1_img=img_t1,
                            ref_mask=ref, best_mask=dbg_best_mask,
                            candidates=cand_list,
                            out_png_path=out_png
                        )
                except Exception as e:
                    print(f"[xcorr-debug] plot failed at t={t}: {e}")

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
        _, _, bf_frame_files = FindMovieMaxMin(CH_IDX)
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

    # ---- interpret --update_existing (supports optional integer)
    # expected argparse: nargs='?', const=-1, type=int
    if args.update_existing is None:
        resume_t = None                  # default behavior
    elif int(args.update_existing) == -1:
        resume_t = 0                     # recompute from t=0 (old boolean behavior)
    else:
        resume_t = max(0, int(args.update_existing))  # resume from this timepoint

    csv_exists = masks_csv_path.is_file()
    csv_valid  = is_valid_csv(masks_csv_path)

    # If resume_t is specified, we must track; else keep old logic.
    if resume_t is not None:
        need_to_track = True
    else:
        # old behavior: recompute only if user asked (update_existing=-1) or CSV invalid/missing
        need_to_track = (args.update_existing == -1) or (not csv_valid)

    print(
        f"[track] update_existing={args.update_existing}  "
        f"csv_exists={csv_exists}  csv_valid={csv_valid}  "
        f"track_channel={track_channel}  resume_t={resume_t}  "
        f"-> need_to_track={need_to_track}"
    )

    if need_to_track:
        masks_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Load old CSV if resuming beyond t=0
        df_old = None
        if resume_t is not None and resume_t > 0:
            if not csv_valid:
                raise RuntimeError(
                    f"--update_existing {resume_t} requested, but no valid CSV found at {masks_csv_path}"
                )
            df_old = pd.read_csv(masks_csv_path)

        # Build seed and time sequence
        if resume_t is None or resume_t == 0:
            # Fresh start from first available mask (your current logic)
            first_mask, labeled_mask, filtered_regions = GetFilteredRegions(min_area, channel=track_channel)
            cell = next((c for c in filtered_regions if c.label == cell_id), None)
            if cell is None:
                print(f"Cell {cell_id} not found in first frame masks.")
                sys.exit(1)
            initial_mask = (to_labeled_current(first_mask) == cell_id)
            t_seq = range(0, frame_number)
        else:
            # RESUME from resume_t: seed from t = resume_t - 1 in CSV
            rle_col = 'rle_bf' if track_channel == 'bf' else 'rle_gfp'
            H0 = int(df_old.iloc[0]['height'])
            W0 = int(df_old.iloc[0]['width'])
            prev_idx = resume_t - 1
            if prev_idx < 0 or prev_idx >= len(df_old):
                raise RuntimeError(f"Invalid resume range: previous index {prev_idx} out of bounds.")
            prev_row = df_old.iloc[prev_idx]
            prev_rle = prev_row[rle_col]
            if not isinstance(prev_rle, str) or prev_rle == "":
                raise RuntimeError(
                    f"Cannot resume at t={resume_t}: previous row (t={resume_t-1}) has empty {rle_col}."
                )
            initial_mask = np.asarray(rle_decode(prev_rle, (H0, W0)), bool)
            if initial_mask.shape != (H0, W0):
                raise RuntimeError("Decoded previous mask size mismatch.")
            t_seq = range(resume_t, frame_number)

        # XCorr config
        xcfg_fwd = {
            'fallback_overlap_thr': xcorr_fallback_ov,
            'angle_pad_deg': xcorr_angle_pad,
            'angle_step_deg': xcorr_angle_step,
            'num_singles': 30, 'num_pairs': 20, 'pair_pool_k': 100, 'pad_px': 24,
            'shift_limit_px': 10.0,  # adjust as needed
            'debug': xcorr_debug,
            'debug_dir': xcorr_debug_dir_fwd,
        }

        # Run tracker only on t_seq, seeded from initial_mask
        forward_sel = track_one_direction(
            t_seq, initial_mask,
            channel=track_channel, first_threshold=0.5, next_threshold=0.7, lock_first=False,
            xcorr_mode=xcorr_mode, xcorr_cfg=xcfg_fwd
        )

        backward_sel = None
        if direction_mode in ('backward', 'both'):
            # For resume, backward pass still starts from the last frame’s forward mask
            last_t = (frame_number - 1)
            seed_mask_backward = forward_sel.get(last_t, {}).get("mask", None)
            if seed_mask_backward is None:
                # fallback: use mask from initial frame of t_seq if missing
                first_t = t_seq.start if hasattr(t_seq, 'start') else t_seq[0]
                seed_mask_backward = forward_sel[first_t]["mask"]
            xcfg_bwd = dict(xcfg_fwd, debug_dir=xcorr_debug_dir_bwd)
            backward_sel = track_one_direction(
                range(frame_number - 1, -1, -1), seed_mask_backward,
                channel=track_channel, first_threshold=0.5, next_threshold=0.7,
                xcorr_mode=xcorr_mode, xcorr_cfg=xcfg_bwd
            )

        # ---- Prepare output dataframe: preserve earlier rows when resuming
        H, W = initial_mask.shape
        if df_old is None:
            # fresh build of empty rows
            base_rows = []
            for t in range(frame_number):
                base_rows.append({
                    "time_point": t, "width": W, "height": H,
                    "rle_gfp": "", "touches_border_gfp": False, "source_gfp": "",
                    "overlap_score_gfp": 0.0, "smooth_score_gfp": -1e9,
                    "area_gfp": 0, "area_penalty_gfp": 0.0, "huge_jump_rejected_gfp": False,
                    "rle_bf": "", "touches_border_bf": False, "source_bf": "",
                    "overlap_score_bf": 0.0, "smooth_score_bf": -1e9,
                    "area_bf": 0, "area_penalty_bf": 0.0, "huge_jump_rejected_bf": False,
                })
            df_out = pd.DataFrame(base_rows)
        else:
            df_out = df_old.copy()

        # Helper to pick result per t
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

        # Update only t in t_seq, preserving others
        for t in t_seq:
            ch_sel, src_sel = choose(forward_sel, backward_sel, t)
            if track_channel == 'gfp':
                df_out.loc[df_out['time_point'] == t, [
                    "rle_gfp","touches_border_gfp","source_gfp","overlap_score_gfp",
                    "smooth_score_gfp","area_gfp","area_penalty_gfp","huge_jump_rejected_gfp",
                    "width","height"
                ]] = [
                    rle_encode(ch_sel["mask"]),
                    bool(ch_sel["touch"]),
                    src_sel,
                    float(ch_sel["overlap"]),
                    float(ch_sel["score"]),
                    int(ch_sel["area"]),
                    float(ch_sel.get("area_penalty", 0.0)),
                    bool(ch_sel.get("huge_jump_rejected", False)),
                    W, H
                ]
            else:
                df_out.loc[df_out['time_point'] == t, [
                    "rle_bf","touches_border_bf","source_bf","overlap_score_bf",
                    "smooth_score_bf","area_bf","area_penalty_bf","huge_jump_rejected_bf",
                    "width","height"
                ]] = [
                    rle_encode(ch_sel["mask"]),
                    bool(ch_sel["touch"]),
                    src_sel,
                    float(ch_sel["overlap"]),
                    float(ch_sel["score"]),
                    int(ch_sel["area"]),
                    float(ch_sel.get("area_penalty", 0.0)),
                    bool(ch_sel.get("huge_jump_rejected", False)),
                    W, H
                ]

        df_out.to_csv(masks_csv_path, index=False)
        start_msg = t_seq.start if hasattr(t_seq, 'start') else list(t_seq)[0]
        print(f"[Tracking] Saved {track_channel.upper()} masks to: {masks_csv_path} (updated t >= {start_msg})")
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

