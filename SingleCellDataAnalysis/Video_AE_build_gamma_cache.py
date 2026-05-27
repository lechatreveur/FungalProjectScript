#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_build_gamma_cache.py

Generates the ground truth Gamma spatial probability masks for the 431 curated cells.
Runs the EM algorithm (ImageQuantification) across all 101 frames per cell.
Saves the 7 spatial probability masks back into the canonical (48, 96) view.

Output:
  gamma_cache.npy — shape (N_valid, 101, 7, 48, 96), float32
"""

import os
import sys
import numpy as np
import pandas as pd
from skimage.io import imread
import cv2
from skimage.measure import regionprops, label
from skimage.transform import resize

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.Video_AE_data_loader import EXPERIMENT_BASES, FILM_FOLDER_MAP, resolve_cell_info_sept17, resolve_cell_info_generic, ID_MAP_CSV_PATHS
from quant_helpers import combine_gammas_prob
from Image_quantification_functions import ImageQuantification
from Cell_tracking_functions import rle_decode

EXPERIMENTS = {
    'Sept17':     '/Volumes/X10 Pro/Movies/2025_09_17/',
    'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/',
    'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/',
}

OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_GAMMA  = os.path.join(OUTPUT_DIR, "gamma_cache.npy")
CACHE_GIDS   = os.path.join(OUTPUT_DIR, "video_gids.txt")

# Maps gammas_unlinked keys to an index in our 7-channel tensor
GAMMA_KEYS = ['Y2Z0', 'Y2Z1', 'Y2Z2', 'Y2Z3', 'Y2Z4', 'Y2Z5', 'Y2Z6']

def load_tif_frame(frames_dir, film_name, t, exp_label=None):
    if exp_label == 'Sept17':
        p1 = os.path.join(frames_dir, f"{film_name}_t_{t:03d}_c_0.tif")
        p2 = os.path.join(frames_dir, f"{film_name}_t_{t:03d}_c_1.tif")
        if os.path.exists(p1): return imread(p1)
        if os.path.exists(p2): return imread(p2)
        raise FileNotFoundError(f"Missing Sept17 frame at t={t} in {frames_dir}")
    elif exp_label == 'June25_20m':
        p1 = os.path.join(frames_dir, f"{film_name}_t_{t:02d}_c_1.tif")
        p2 = os.path.join(frames_dir, f"{film_name}_t_{t:02d}_c_0.tif")
        if os.path.exists(p1): return imread(p1)
        if os.path.exists(p2): return imread(p2)
        # Also try 3 digits just in case
        p3 = os.path.join(frames_dir, f"{film_name}_t_{t:03d}_c_1.tif")
        p4 = os.path.join(frames_dir, f"{film_name}_t_{t:03d}_c_0.tif")
        if os.path.exists(p3): return imread(p3)
        if os.path.exists(p4): return imread(p4)
        raise FileNotFoundError(f"Missing June25_20m frame at t={t}")
    else:
        # Default M92 / M93: c_0 usually
        p1 = os.path.join(frames_dir, f"{film_name}_t_{t:03d}_c_0.tif")
        if os.path.exists(p1): return imread(p1)
        raise FileNotFoundError(f"Missing frame at t={t} in {frames_dir}")


def extract_canonical_crop_gamma(gamma_img, mask, frame_h=64, frame_w=224, pad=10):
    """
    Applies the exact same rotation, bounding box, and scaling to the gamma probability image
    as we did to the raw video, guaranteeing perfectly aligned ground truth.
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((frame_h, frame_w), dtype=np.float32)
    
    r0 = max(0, rows[0] - pad)
    r1 = min(mask.shape[0], rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad)
    c1 = min(mask.shape[1], cols[-1] + pad + 1)
    
    crop_img  = gamma_img[r0:r1, c0:c1].copy()
    crop_mask = mask[r0:r1, c0:c1].copy()

    lab = label(crop_mask.astype(np.uint8))
    props = regionprops(lab)
    if not props: return np.zeros((frame_h, frame_w), dtype=np.float32)
    
    p = max(props, key=lambda x: x.area)
    angle_deg = np.degrees(p.orientation)
    
    # Pad significantly to ensure rotation doesn't clip ends
    pad_rot = 100
    padded_img = np.pad(crop_img, pad_rot, mode='constant', constant_values=0)
    padded_mask = np.pad(crop_mask, pad_rot, mode='constant', constant_values=0)
    
    h_p, w_p = padded_img.shape
    cx, cy = w_p / 2.0, h_p / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    rotated_img  = cv2.warpAffine(padded_img,  M, (w_p, h_p), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(padded_mask.astype(np.float32), M, (w_p, h_p), flags=cv2.INTER_NEAREST) > 0.5
    
    rows_r = np.where(rotated_mask.any(axis=1))[0]
    cols_r = np.where(rotated_mask.any(axis=0))[0]
    if rows_r.size == 0: return np.zeros((frame_h, frame_w), dtype=np.float32)
    
    r0r = max(0, rows_r[0] - 5)
    r1r = min(h_p, rows_r[-1] + 6)
    c0r = max(0, cols_r[0] - 5)
    c1r = min(w_p, cols_r[-1] + 6)
    
    tight_img  = rotated_img[r0r:r1r, c0r:c1r]
    tight_mask = rotated_mask[r0r:r1r, c0r:c1r]
    
    if tight_img.shape[0] > tight_img.shape[1]:
        tight_img  = np.rot90(tight_img)
        tight_mask = np.rot90(tight_mask)
    
    tight_img = tight_img * tight_mask.astype(np.float32)
    
    # --- Fixed Scale & Padding (Same as Video Loader) ---
    scale = frame_w / 210.0 
    new_h = int(tight_img.shape[0] * scale)
    new_w = int(tight_img.shape[1] * scale)
    new_h = min(frame_h, max(1, new_h))
    new_w = min(frame_w, max(1, new_w))
    
    resized = resize(tight_img, (new_h, new_w), order=1, 
                     anti_aliasing=True, preserve_range=True)
    
    canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
    y_off = (frame_h - new_h) // 2
    x_off = (frame_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    return np.clip(canvas, 0, 1.0)


def build_gamma_cache():
    # 1. Load GIDs and lookup tables
    with open(CACHE_GIDS, 'r') as f:
        gids = [line.strip() for line in f]
    
    # Determine total cells
    n_cells = len(gids)
    print(f"Loaded {n_cells} cells from video_gids.txt")
    
    # Initialize Memory-Mapped array
    print(f"Initializing Memory-Mapped cache at {CACHE_GAMMA}...")
    gamma_array = np.lib.format.open_memmap(CACHE_GAMMA, mode='w+', dtype='float32', shape=(n_cells, 101, 7, 64, 224))

    df_map_m92 = pd.read_csv(ID_MAP_CSV_PATHS['M92']) if os.path.exists(ID_MAP_CSV_PATHS['M92']) else None
    df_map_m93 = pd.read_csv(ID_MAP_CSV_PATHS['M93']) if os.path.exists(ID_MAP_CSV_PATHS['M93']) else None
    df_map_june25 = pd.read_csv(ID_MAP_CSV_PATHS['June25_20m']) if os.path.exists(ID_MAP_CSV_PATHS['June25_20m']) else None
    df_stack_sept17 = pd.read_csv(os.path.join(EXPERIMENTS['Sept17'], "unaligned_pairs_quant", "stacked_gfp1_gfp2_for_unaligned_pairs.csv"))

    # Determine gfp max min cache (so we don't recompute per cell in same movie)
    gfp_range_cache = {}

    # Build sorted list of known experiment labels (longest first)
    known_labels = sorted(EXPERIMENTS.keys(), key=lambda x: -len(x))

    for idx, gid in enumerate(gids):
        print(f"\nProcessing {idx+1}/{len(gids)}: {gid}")
        
        exp_label = None
        cell_id = None
        for lbl in known_labels:
            if gid.startswith(lbl + '_'):
                exp_label = lbl
                cell_id = int(gid[len(lbl)+1:])
                break
        
        if exp_label is None:
            print(f"Skipping unknown gid format: {gid}")
            continue
        
        try:
            # Resolve path
            if exp_label == 'Sept17':
                film_name, orig_cell_id = resolve_cell_info_sept17(cell_id, df_stack_sept17)
            elif exp_label == 'M92':
                film_name, orig_cell_id = resolve_cell_info_generic(cell_id, df_map_m92, exp_label)
            elif exp_label == 'M93':
                film_name, orig_cell_id = resolve_cell_info_generic(cell_id, df_map_m93, exp_label)
            elif exp_label == 'June25_20m':
                film_name = FILM_FOLDER_MAP.get(('June25_20m', 'GFP1', 'F0'))
                orig_cell_id = cell_id
        except Exception as e:
            print(f"Skipping {gid} due to missing id_map: {e}")
            continue

        film_folder = EXPERIMENT_BASES[exp_label] + film_name + '/'
        frames_dir = film_folder + f"Frames_{film_name}/"
        tracked_cells_dir = film_folder + f"TrackedCells_{film_name}/"
        mask_csv_path = tracked_cells_dir + f"cell_{orig_cell_id}_masks.csv"

        df_masks = pd.read_csv(mask_csv_path)
        H = int(df_masks.iloc[0]['height'])
        W = int(df_masks.iloc[0]['width'])

        rle_col = None
        for candidate in ['rle', 'rle_gfp', 'rle_bf']:
            if candidate in df_masks.columns:
                rle_col = candidate
                break

        # Compute dynamic range if not cached
        if film_name not in gfp_range_cache:
            # We hardcode channel index based on experiment. Usually c_0 except June25 which is c_1
            ch = 0 if exp_label != 'June25_20m' else 1
            if exp_label == 'Sept17' and 'TP2' in film_name: ch = 1 # Rough guess, let's just do min max of first 10 frames
            
            # Simple manual min max of first 5 frames
            min_vals, max_vals = [], []
            for t in range(5):
                img = load_tif_frame(frames_dir, film_name, t, exp_label=exp_label)
                min_vals.append(np.percentile(img, 1.0))
                max_vals.append(np.percentile(img, 99.5))
            gfp_range_cache[film_name] = (np.mean(max_vals), np.mean(min_vals))

        gfp_max, gfp_min = gfp_range_cache[film_name]

        # Tracking state
        ep1_ref, ep2_ref, prev_params = None, None, None

        try:
            for t in range(101):
                row = df_masks.iloc[t]
                mask_bool = np.asarray(rle_decode(row[rle_col], (H, W)), bool) if isinstance(row[rle_col], str) else np.zeros((H, W), bool)
                img = load_tif_frame(frames_dir, film_name, t, exp_label=exp_label)
                
                if mask_bool.sum() < 50:
                    if t > 0: gamma_array[idx, t] = gamma_array[idx, t-1]
                    continue
                
                par, par_fixed, plot_data, ep1_ref, ep2_ref = ImageQuantification(
                    fluorescent_img=img,
                    cell_mask=mask_bool,
                    selected_label=str(orig_cell_id),
                    C1max=gfp_max,
                    C1min=gfp_min,
                    tp=0 if (ep1_ref is None) else t,
                    ref_ep1=ep1_ref,
                    ref_ep2=ep2_ref,
                    skip_em=False,
                    init_params_unlinked=prev_params,
                    init_blend=0.7
                )
                prev_params = par
                
                gammas_unlinked = plot_data[2]
                y_idx = np.asarray(plot_data[4], int)
                x_idx = np.asarray(plot_data[5], int)
                
                y_indices, x_indices = np.where(mask_bool)
                ymin = y_indices.min() if y_indices.size > 0 else 0
                xmin = x_indices.min() if x_indices.size > 0 else 0
                
                for c_idx, key in enumerate(GAMMA_KEYS):
                    if key in gammas_unlinked:
                        gamma_flat = np.asarray(gammas_unlinked[key], dtype=np.float32)
                        gamma_img = np.zeros((H, W), dtype=np.float32)
                        gamma_img[ymin + y_idx, xmin + x_idx] = gamma_flat
                        
                        # Warp to perfectly align with autoencoder (64, 224) video
                        gamma_array[idx, t, c_idx] = extract_canonical_crop_gamma(gamma_img, mask_bool)

        except Exception as e:
            print(f"Error at cell {idx} ({gid}): {e}")
            if idx > 0: gamma_array[idx] = gamma_array[idx-1]
        
        # Flush to disk every cell
        gamma_array.flush()

    print("Done!")

if __name__ == "__main__":
    build_gamma_cache()
