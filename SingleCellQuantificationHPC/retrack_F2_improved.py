#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrack_F2_improved.py
----------------------
Retrack all cells in A14-YES-1t-FBFBF-2_F2 with improved parameters.
For each cell:
  1. Run new tracking into a TEMP directory
  2. Compare new track to old track using mean per-frame IoU against seg.tif
  3. If new track has higher mean IoU AND is not worse than the current masks,
     replace the cell_N_masks.csv.
  4. Cells that were manually corrected (QC status = 'corrected') are SKIPPED.

Run in the background:
  python retrack_F2_improved.py > /tmp/retrack_F2.log 2>&1 &
  tail -f /tmp/retrack_F2.log
"""

import os
import sys
import re
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Cell_tracking_functions import (
    load_segmentation, to_labeled_current, compute_overlap,
    get_cell_mask_area_aware, rle_encode, rle_decode,
    mask_to_rle, rle_to_mask, touches_border, area_change_penalty
)
from skimage.measure import regionprops, label
from skimage.io import imread

# ============================================================
# CONFIG
# ============================================================
MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP        = "2025_12_31_M92"
FILM       = "A14-YES-1t-FBFBF-2_F2"

FILM_DIR         = MOVIE_ROOT / EXP / FILM
FRAMES_DIR       = FILM_DIR / f"Frames_{FILM}"
MASKS_DIR        = FILM_DIR / f"Masks_{FILM}"
TRACKED_DIR      = FILM_DIR / f"TrackedCells_{FILM}"
QC_CSV           = TRACKED_DIR / "qc.csv"

# ---- Improved parameters (vs original defaults) ----
FIRST_THRESHOLD  = 0.4   # was 0.5  — less strict seeding from first frame
NEXT_THRESHOLD   = 0.55  # was 0.7  — much less strict, avoids false fallback
TOPK             = 8     # was 5    — consider more candidates
AREA_LAMBDA      = 0.20  # was 0.35 — softer area penalty
XCORR_MODE       = "fallback"
XCORR_FALLBACK_OV= 0.25  # was 0.35 — only use xcorr when truly lost
XCORR_ANGLE_PAD  = 20.0  # was 15.0 — wider angle search
XCORR_ANGLE_STEP = 3.0
MIN_AREA         = 2500
DIRECTION        = "both"

# ---- Improvement threshold to replace existing CSV ----
# New track must have mean IoU at least this much BETTER than old track.
# Set to 0.0 to replace on any improvement, or e.g. 0.02 for 2% minimum gain.
MIN_IMPROVEMENT  = 0.0

# ============================================================
print(f"=== Retracking {FILM} with improved settings ===")
print(f"  next_threshold : 0.7 → {NEXT_THRESHOLD}")
print(f"  topk           : 5   → {TOPK}")
print(f"  area_lambda    : 0.35→ {AREA_LAMBDA}")
print(f"  xcorr_fallback_ov: 0.35→ {XCORR_FALLBACK_OV}")
print(f"  xcorr_angle_pad: 15 → {XCORR_ANGLE_PAD}")
print()

# Load QC status to skip manually corrected cells
manually_corrected = set()
if QC_CSV.exists():
    try:
        qc_df = pd.read_csv(QC_CSV)
        for _, row in qc_df.iterrows():
            if str(row.get("status", "")).lower() == "corrected":
                manually_corrected.add(int(row["cell_id"]))
        print(f"Skipping {len(manually_corrected)} manually corrected cells: {sorted(manually_corrected)}")
    except Exception as e:
        print(f"[warn] Could not read qc.csv: {e}")

# Discover all cell IDs
cell_files = {}
for f in TRACKED_DIR.iterdir():
    if f.name.startswith("."):
        continue
    m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
    if m:
        cell_files[int(m.group(1))] = f
cell_ids = sorted(cell_files.keys())
print(f"Found {len(cell_ids)} cells to process")

# Detect channel index from mask files
def detect_channel_idx():
    for f in sorted(MASKS_DIR.iterdir()):
        if f.name.startswith("."):
            continue
        m = re.search(r"_t_\d+_c_(\d+)_seg\.(tif|npy)$", f.name)
        if m:
            return int(m.group(1))
    return 0

CH_IDX = detect_channel_idx()
print(f"Detected channel index: c_{CH_IDX}")

# Discover frames
frame_files = sorted([
    f.name for f in FRAMES_DIR.iterdir()
    if f.name.lower().endswith(".tif")
    and not f.name.endswith("_seg.tif")
    and f"_c_{CH_IDX}.tif" in f.name
    and not f.name.startswith(".")
])
NUM_FRAMES = len(frame_files)
print(f"Total frames: {NUM_FRAMES}")

def frame_path(t):
    return FRAMES_DIR / f"{FILM}_t_{t:03d}_c_{CH_IDX}.tif"

def seg_path(t):
    p = MASKS_DIR / f"{FILM}_t_{t:03d}_c_{CH_IDX}_seg.tif"
    return p if p.exists() else None

def load_labeled_seg(t):
    p = seg_path(t)
    if p is None:
        return None
    seg = load_segmentation(str(p))
    return to_labeled_current(seg)

# ============================================================
# Tracking helpers (mirrors one_cell_quantification_1CH.py)
# ============================================================
from xcorr_utils import xcorr_best_of_six

def track_one_direction(t_seq, ref_start_mask,
                        first_threshold=FIRST_THRESHOLD,
                        next_threshold=NEXT_THRESHOLD):
    from itertools import combinations as _combs

    xcorr_cfg = {
        'fallback_overlap_thr': XCORR_FALLBACK_OV,
        'angle_pad_deg': XCORR_ANGLE_PAD,
        'angle_step_deg': XCORR_ANGLE_STEP,
        'num_singles': 3, 'num_pairs': 3, 'pair_pool_k': 6, 'pad_px': 24,
    }

    results = {}
    prev_mask = None
    prev_area = float(ref_start_mask.sum())
    prev_meta = {'composition': 'single'}

    for i, t in enumerate(t_seq):
        ref = ref_start_mask if i == 0 else prev_mask
        lab_cur = load_labeled_seg(t)
        thr = first_threshold if i == 0 else next_threshold

        if i == 0:
            cm = ref.copy()
            ov = compute_overlap(ref, cm)
            sc = ov
            pen = 0.0
            rej = False
            sel_mode = 'locked_first'
            meta_out = {'composition': 'single'}
        else:
            if lab_cur is not None:
                cm0, ov0, sc0, pen0, rej0, meta0 = get_cell_mask_area_aware(
                    lab_cur, ref, prev_area,
                    threshold=thr, max_segments=2, topk=TOPK,
                    area_lambda=AREA_LAMBDA, ratio_soft=1.3, ratio_hard=1.8
                )
                sel_mode_area = 'area_primary'
                area_meta_out = meta0
            else:
                cm0, ov0, sc0, pen0, rej0 = ref.copy(), 1.0, 1.0, 0.0, False
                sel_mode_area = 'area_primary'
                area_meta_out = {'composition': 'keep_prev'}

            # XCorr fallback
            use_xcorr = rej0 or ov0 < xcorr_cfg['fallback_overlap_thr']
            xinfo = {}
            xcorr_meta = area_meta_out

            if use_xcorr and lab_cur is not None:
                t_prev = list(t_seq)[i-1] if i > 0 else t
                fp0 = frame_path(t_prev)
                fp1 = frame_path(t)
                if fp0.exists() and fp1.exists():
                    img_t0 = imread(str(fp0))
                    img_t1 = imread(str(fp1))
                    min_area_abs = max(50, int(0.2 * prev_area))
                    xsel = xcorr_best_of_six(
                        ref, img_t0, lab_cur, img_t1,
                        num_singles=xcorr_cfg['num_singles'],
                        num_pairs=xcorr_cfg['num_pairs'],
                        pair_pool_k=xcorr_cfg['pair_pool_k'],
                        min_area=min_area_abs,
                        angle_pad_deg=xcorr_cfg['angle_pad_deg'],
                        angle_step_deg=xcorr_cfg['angle_step_deg'],
                        pad_px=xcorr_cfg['pad_px']
                    )
                    if xsel["best"] is not None:
                        cm1 = xsel["best_mask"]
                        ov1 = compute_overlap(ref, cm1)
                        sc1 = float(xsel["best"]["xcorr"])
                        xcorr_meta = {'composition': 'single'}

                        if ov0 < xcorr_cfg['fallback_overlap_thr'] or rej0:
                            cm0, ov0, sc0, pen0, rej0, sel_mode_area = cm1, ov1, sc1, 0.0, False, 'xcorr_fallback'
                            area_meta_out = xcorr_meta

            cm, ov, sc, pen, rej, sel_mode = cm0, ov0, sc0, pen0, rej0, sel_mode_area
            meta_out = area_meta_out

        results[t] = {
            "mask": cm,
            "overlap": float(ov),
            "score": float(sc),
            "area": int(cm.sum()),
            "area_penalty": float(pen),
            "huge_jump_rejected": bool(rej),
            "touch": touches_border(cm),
            "selector_mode": sel_mode,
            "composition": meta_out.get('composition', 'single'),
        }
        seg_rle = meta_out.get('segments_rle', None)
        if meta_out.get('composition') == 'pair' and seg_rle:
            results[t]['segments_rle'] = seg_rle

        prev_mask = cm
        prev_area = float(cm.sum())
        prev_meta = {"composition": meta_out.get('composition', 'single')}
        if meta_out.get('composition') == 'pair' and seg_rle:
            prev_meta['segments_rle'] = seg_rle

    return results


def compute_mean_iou_vs_seg(rle_list, H, W):
    """Compare a list of per-frame RLE strings against the cellpose seg.tif.
    Returns mean IoU across frames where seg exists (skips missing frames).
    A mask that is exactly one cellpose segment gets IoU=1.0.
    A mask that drifts or merges gets lower IoU.
    Empty masks are treated as IoU=0.
    """
    scores = []
    for t, rle in enumerate(rle_list):
        p = seg_path(t)
        if p is None:
            continue
        rle_str = str(rle).strip() if pd.notna(rle) else ""
        if not rle_str:
            scores.append(0.0)
            continue
        try:
            cell_mask = rle_decode(rle_str, (H, W)).astype(bool)
        except Exception:
            scores.append(0.0)
            continue
        try:
            seg = load_segmentation(str(p))
            seg_lbl = to_labeled_current(seg)
        except Exception:
            continue

        # Find best-matching single segment by IoU
        overlapping = set(np.unique(seg_lbl[cell_mask])) - {0}
        if not overlapping:
            scores.append(0.0)
            continue
        best_iou = 0.0
        for lbl in overlapping:
            seg_mask = (seg_lbl == lbl)
            inter = (cell_mask & seg_mask).sum()
            union = (cell_mask | seg_mask).sum()
            j = inter / float(union) if union > 0 else 0.0
            if j > best_iou:
                best_iou = j
        scores.append(best_iou)

    return float(np.mean(scores)) if scores else 0.0


# ============================================================
# Main loop
# ============================================================
replaced = 0
skipped_corrected = 0
skipped_no_improvement = 0
errors = 0

for cell_id in cell_ids:
    if cell_id in manually_corrected:
        print(f"  [Cell {cell_id}] SKIP — manually corrected")
        skipped_corrected += 1
        continue

    csv_path = cell_files[cell_id]
    try:
        old_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [Cell {cell_id}] ERROR reading CSV: {e}")
        errors += 1
        continue

    H = int(old_df.iloc[0]['height'])
    W = int(old_df.iloc[0]['width'])

    # Determine RLE column
    rle_col = 'rle_bf'
    if 'rle_gfp' in old_df.columns and old_df['rle_gfp'].dropna().any():
        rle_col = 'rle_gfp'
    track_channel = 'gfp' if rle_col == 'rle_gfp' else 'bf'

    old_rles = old_df[rle_col].fillna("").tolist()
    old_iou = compute_mean_iou_vs_seg(old_rles, H, W)

    # Get initial mask from first-frame seg
    t0_seg = load_labeled_seg(0)
    if t0_seg is None:
        print(f"  [Cell {cell_id}] SKIP — no seg at t=0")
        continue

    initial_mask = (t0_seg == cell_id)
    if not initial_mask.any():
        print(f"  [Cell {cell_id}] SKIP — cell_id not in t=0 seg")
        continue

    try:
        # Forward pass
        fwd = track_one_direction(
            range(NUM_FRAMES), initial_mask,
            first_threshold=FIRST_THRESHOLD, next_threshold=NEXT_THRESHOLD
        )
        # Backward pass (seed from last forward frame)
        seed_bwd = fwd[NUM_FRAMES - 1]["mask"]
        bwd = track_one_direction(
            range(NUM_FRAMES - 1, -1, -1), seed_bwd,
            first_threshold=FIRST_THRESHOLD, next_threshold=NEXT_THRESHOLD
        )
    except Exception as e:
        print(f"  [Cell {cell_id}] ERROR during tracking: {e}")
        errors += 1
        continue

    # Build new CSV rows using backward (preferred) then forward fallback
    new_rows = []
    for t in range(NUM_FRAMES):
        ch_sel = bwd.get(t, fwd.get(t))
        if ch_sel is None:
            ch_sel = {"mask": np.zeros((H, W), bool), "touch": False, "overlap": 0.0,
                      "score": -1.0, "area": 0, "area_penalty": 0.0,
                      "huge_jump_rejected": False, "composition": "single", "selector_mode": "none"}
        src = "backward"

        comp = ch_sel.get("composition", "single")
        pair_a, pair_b = "", ""
        if comp == "pair" and ch_sel.get("segments_rle"):
            segs = ch_sel["segments_rle"]
            pair_a = rle_encode(rle_to_mask(segs[0]))
            pair_b = rle_encode(rle_to_mask(segs[1]))

        row = {"time_point": t, "width": W, "height": H}
        if track_channel == 'gfp':
            row.update({
                "rle_gfp": rle_encode(ch_sel["mask"]),
                "touches_border_gfp": ch_sel["touch"],
                "source_gfp": src,
                "overlap_score_gfp": ch_sel["overlap"],
                "smooth_score_gfp": ch_sel["score"],
                "area_gfp": ch_sel["area"],
                "area_penalty_gfp": ch_sel.get("area_penalty", 0.0),
                "huge_jump_rejected_gfp": ch_sel.get("huge_jump_rejected", False),
                "composition_gfp": comp,
                "pair_segA_rle_gfp": pair_a,
                "pair_segB_rle_gfp": pair_b,
                "selector_mode_gfp": ch_sel.get("selector_mode", ""),
            })
        else:
            row.update({
                "rle_bf": rle_encode(ch_sel["mask"]),
                "touches_border_bf": ch_sel["touch"],
                "source_bf": src,
                "overlap_score_bf": ch_sel["overlap"],
                "smooth_score_bf": ch_sel["score"],
                "area_bf": ch_sel["area"],
                "area_penalty_bf": ch_sel.get("area_penalty", 0.0),
                "huge_jump_rejected_bf": ch_sel.get("huge_jump_rejected", False),
                "composition_bf": comp,
                "pair_segA_rle_bf": pair_a,
                "pair_segB_rle_bf": pair_b,
                "selector_mode_bf": ch_sel.get("selector_mode", ""),
            })
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    new_rles = new_df[rle_col].fillna("").tolist()
    new_iou = compute_mean_iou_vs_seg(new_rles, H, W)

    delta = new_iou - old_iou
    print(f"  [Cell {cell_id:3d}] old_IoU={old_iou:.3f}  new_IoU={new_iou:.3f}  Δ={delta:+.3f}", end="")

    if delta >= MIN_IMPROVEMENT:
        # Backup the old CSV with a .bak extension (keep one backup)
        bak = csv_path.with_suffix(".csv.bak")
        shutil.copy2(csv_path, bak)
        new_df.to_csv(csv_path, index=False)
        print("  → REPLACED ✓")
        replaced += 1
    else:
        print("  → kept old (no improvement)")
        skipped_no_improvement += 1

print()
print("=== Summary ===")
print(f"  Replaced:              {replaced}")
print(f"  No improvement:        {skipped_no_improvement}")
print(f"  Skipped (corrected):   {skipped_corrected}")
print(f"  Errors:                {errors}")
print("Done.")
