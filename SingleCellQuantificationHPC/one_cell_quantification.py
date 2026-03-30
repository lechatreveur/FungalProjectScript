#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactor: two-pass pipeline
Pass 1: bidirectional tracking -> save masks CSV (RLE)
Pass 2: image quantification reads masks from CSV
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
#sys.path.append('/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC')
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
    quantify_one_object,  # main entry
)

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(description="Track a single cell (both directions) and quantify signals.")
parser.add_argument('--cell_id', type=int, required=True, help='Cell ID to process')
parser.add_argument('--track_channel', choices=['bf', 'gfp'], default='bf',
                    help='Which segmentation to track as the driver.')
parser.add_argument('--direction', choices=['forward', 'backward', 'both'], default='both',
                    help='Which tracking direction(s) to run.')
parser.add_argument('--experiment_path', type=str, required=True, help='Path to experiment')
parser.add_argument('--file_name', type=str, required=True, help='File name of .ims file')
parser.add_argument('--z_index', type=int, default=1, help="Z-slice index to load for segmentation.")
parser.add_argument('--min_area', type=int, default=2500, help="Minimum area threshold for cell filtering.")
parser.add_argument('--update_existing', action='store_true', help="(Currently unused) Only update existing rows")
args = parser.parse_args()

cell_id        = args.cell_id
working_dir    = args.experiment_path
file_name      = args.file_name
min_area       = args.min_area
z_index        = args.z_index
track_channel  = args.track_channel
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
cell_plot_folder  = os.path.join(plot_output_root, f"cell_{cell_id}")
os.makedirs(plot_output_root, exist_ok=True)
os.makedirs(cell_plot_folder, exist_ok=True)

# =========================
# Path helper functions
# =========================
def bf_seg_path(t: int) -> str:
    return os.path.join(brightfield_seg_folder, f"{file_name}_t_{t:03d}_z_{z_index}_c_1_seg.tif")

def gfp_seg_path(t: int) -> str:
    return os.path.join(GFP_seg_folder, f"{file_name}_t_{t:03d}_c_0_seg.tif")

def gfp_frame_path(t: int) -> str:
    return os.path.join(output_frames_folder, f"{file_name}_t_{t:03d}_c_0.tif")

def seg_path(channel: str, t: int) -> str:
    return bf_seg_path(t) if channel == 'bf' else gfp_seg_path(t)

# =========================
# Utilities
# =========================
def load_labeled_seg(channel: str, t: int):
    """Load segmentation and convert to labeled mask for frame t."""
    p = seg_path(channel, t)
    if os.path.exists(p):
        raw = load_segmentation(p)
        return to_labeled_current(raw)
    return None

def GetFilteredRegions(min_area: int = 2500):
    """Return brightfield first frame + labeled + regions filtered by area."""
    mask_files = sorted([f for f in os.listdir(brightfield_seg_folder)
                         if f.endswith('_seg.npy') or f.endswith('_seg.tif')])
    if len(mask_files) == 0:
        raise FileNotFoundError("No brightfield segmentation files found.")
    first_mask   = load_segmentation(os.path.join(brightfield_seg_folder, mask_files[0]))
    labeled_mask = to_labeled_current(first_mask)
    regions      = regionprops(labeled_mask)
    filtered     = [r for r in regions if r.area >= min_area]
    return first_mask, labeled_mask, filtered

def FindMovieMaxMin(channel: int):
    """Scan frames for channel and return (gfp_max, gfp_min, frame_files)."""
    from fnmatch import fnmatch
    if channel != 0:
        raise ValueError("Only channel 0 supported in this script.")
    pattern     = f"{file_name}_t_???_c_{channel}.tif"
    frame_files = sorted([f for f in os.listdir(output_frames_folder) if fnmatch(f, pattern)])
    pixels = []
    for fname in frame_files:
        frame = imread(os.path.join(output_frames_folder, fname))
        pixels.extend(frame.ravel()[::10])   # sample
    pixels = np.array(pixels)
    return np.percentile(pixels, 99.5), np.percentile(pixels, 1), frame_files

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
            cm  = ref.copy()               # ← keep the seed exactly
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
    #masks_csv_path = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_masks.csv")
    gfp_max, gfp_min, gfp_frame_files = FindMovieMaxMin(0)
    frame_number = len(gfp_frame_files)
    gfp_frame_files = gfp_frame_files[:frame_number]
    
    from pathlib import Path

    masks_csv_path = Path(output_tracked_cells_folder) / f"cell_{cell_id}_masks.csv"
    
    def is_valid_csv(p: Path) -> bool:
        # minimal sanity check; tweak if you want stronger validation
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
        # 1) Discover frames (intensity stats) & initial cell
       
    
        first_mask, labeled_mask, filtered_regions = GetFilteredRegions(min_area)
        cell = next((c for c in filtered_regions if c.label == cell_id), None)
        if cell is None:
            print(f"Cell {cell_id} not found.")
            sys.exit(1)
        initial_mask = (labeled_mask == cell_id)
    
        # 2) Tracking (forward/backward as requested)
        forward = track_one_direction(
            range(frame_number), initial_mask,
            channel=track_channel, first_threshold=0.5, next_threshold=0.7,lock_first=False
        )
    
        backward = None
        if direction_mode in ('backward', 'both'):
            seed_mask_backward = forward[frame_number - 1]["mask"]
            backward = track_one_direction(
                range(frame_number - 1, -1, -1), seed_mask_backward,
                channel=track_channel, first_threshold=0.5, next_threshold=0.7
            )
    
        # 3) Reconcile forward/backward per frame & save masks CSV
        combined = []
        H, W = initial_mask.shape
   
        
    
        for t in range(frame_number):
            # optional: label counts per frame for sanity
            if track_channel == 'bf' and os.path.exists(bf_seg_path(t)):
                raw = load_segmentation(bf_seg_path(t)); lab = to_labeled_current(raw)
                print(f"[t={t}] BF labels: {np.unique(lab).size - 1}")
            elif track_channel == 'gfp' and os.path.exists(gfp_seg_path(t)):
                raw = load_segmentation(gfp_seg_path(t)); lab = to_labeled_current(raw)
                print(f"[t={t}] GFP labels: {np.unique(lab).size - 1}")
        
            if direction_mode == 'forward':
                chosen = forward.get(t, {
                    "mask": np.zeros_like(initial_mask, bool),
                    "touch": False, "overlap": 0.0, "score": -1e9,
                    "area": 0, "area_penalty": 0.0, "huge_jump_rejected": False
                })
                src = "forward"
        
            elif direction_mode == 'backward':
                chosen = (backward or {}).get(t, {
                    "mask": np.zeros_like(initial_mask, bool),
                    "touch": False, "overlap": 0.0, "score": -1e9,
                    "area": 0, "area_penalty": 0.0, "huge_jump_rejected": False
                })
                src = "backward"
        
            else:  # both -> BACKWARD ALWAYS WINS (fallback to forward, else empty)
                b = (backward or {}).get(t, None)
                if b is not None:
                    chosen, src = b, "backward"
                else:
                    f = forward.get(t, None)
                    if f is not None:
                        chosen, src = f, "forward_fallback"
                    else:
                        chosen = {
                            "mask": np.zeros_like(initial_mask, bool),
                            "touch": False, "overlap": 0.0, "score": -1e9,
                            "area": 0, "area_penalty": 0.0, "huge_jump_rejected": False
                        }
                        src = "none"
        
            combined.append({
                "time_point": t,
                "width": W, "height": H,
                "rle": rle_encode(chosen["mask"]),
                "touches_border": chosen["touch"],
                "source": src,
                "overlap_score": chosen["overlap"],
                "smooth_score": chosen["score"],
                "area": chosen["area"],
                "area_penalty": chosen.get("area_penalty", 0.0),
                "huge_jump_rejected": chosen.get("huge_jump_rejected", False),
            })
    
    
        
        pd.DataFrame(combined).to_csv(masks_csv_path, index=False)
        print(f"[Tracking] Saved masks to: {masks_csv_path}")
    else:
        print(f"[track] Skipping: {masks_csv_path.name} exists and --update_existing not set.")
    # 4) Quantification pass
    mask_rows = pd.read_csv(masks_csv_path)
    H = int(mask_rows.iloc[0]['height'])
    W = int(mask_rows.iloc[0]['width'])

    time_series_data = []

    # EM refs per track (single / subcells)
    ep_refs = {
        'single': {'ep1': None, 'ep2': None},
        '1': {'ep1': None, 'ep2': None},
        '2': {'ep1': None, 'ep2': None},
    }

    # Per-subcell plot folders
    cell_plot_folder_single = cell_plot_folder
    cell_plot_folder_1      = os.path.join(plot_output_root, f"cell_{cell_id}_1")
    cell_plot_folder_2      = os.path.join(plot_output_root, f"cell_{cell_id}_2")
    os.makedirs(cell_plot_folder_1, exist_ok=True)
    os.makedirs(cell_plot_folder_2, exist_ok=True)

    for t in range(frame_number):
        print(f"[t={t}]")
        row = mask_rows.iloc[t]
        if bool(row['touches_border']):
            print(f"Cell {cell_id} touches border at t={t}. Skipping onward.")
            break

        img      = imread(gfp_frame_path(t))
        mask_gfp = np.asarray(rle_decode(row['rle'], (H, W)), bool)

        # quantify_one_object returns a list of rows (parent + two children)
        rows = quantify_one_object(
            img, mask_gfp, id_suffix='', t=t,
            plot_dir=cell_plot_folder, ep_refs=ep_refs,
            gfp_min=gfp_min, gfp_max=gfp_max, cell_id=str(cell_id),
            do_plot=True, touches_border_flag=bool(row['touches_border']),
            allow_split=True
        )
        time_series_data.extend(rows)

    # 5) Save quantification table
    df_cell = pd.DataFrame(time_series_data)
    df_cell.to_csv(os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_data.csv"), index=False)
    print(f"Saved quantification data for cell {cell_id}")
