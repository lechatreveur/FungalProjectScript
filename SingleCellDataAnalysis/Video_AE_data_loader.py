#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_data_loader.py

Maps the same 431 manually-curated cells used in the trajectory AE pipeline
to their raw GFP time-lapse frames, and builds orientation-normalized video
tensors of shape (N, 101, 1, H, W) for use with a Video Autoencoder.

Pipeline per cell per frame:
  1. Decode the RLE mask → 2000×2000 bool mask
  2. Load the corresponding raw GFP TIF frame
  3. Crop image to mask bounding box (with padding)
  4. Rotate crop to canonical orientation (major axis = horizontal)
  5. Resize to fixed (FRAME_H × FRAME_W)
  6. Zero background (pixels outside rotated mask → 0)
  7. Normalize (subtract background median, clip at 0)
  
Output: torch.Tensor of shape (N, 101, 1, FRAME_H, FRAME_W), float32
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.transform import resize

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from Cell_tracking_functions import rle_decode

# ==============================================================================
# Configuration
# ==============================================================================

# Output spatial dimensions (width = long axis, height = short axis)
FRAME_H = 32
FRAME_W = 112

# Padding around bounding box before rotation (in pixels, at 2000×2000 scale)
BBOX_PAD = 30

# Film folder mapping: (experiment_label, source, field/tp) → film_folder_name
# This tells us which film subfolder contains the Frames_ and TrackedCells_ data.
FILM_FOLDER_MAP = {
    # Sept17: tp=1,GFP1 → A14_1TP1_F1; tp=2,GFP2 → A14_1TP2_F1
    ('Sept17', 'GFP1', 1): 'A14_1TP1_F1',
    ('Sept17', 'GFP2', 2): 'A14_1TP2_F1',
    # M92: GFP1 → A14-YES-1t-FBFBF_{field}, GFP2 → A14-YES-1t-FBFBF-5_{field}
    ('M92', 'GFP1', 'F0'): 'A14-YES-1t-FBFBF_F0',
    ('M92', 'GFP1', 'F1'): 'A14-YES-1t-FBFBF_F1',
    ('M92', 'GFP1', 'F2'): 'A14-YES-1t-FBFBF_F2',
    ('M92', 'GFP2', 'F0'): 'A14-YES-1t-FBFBF-5_F0',
    ('M92', 'GFP2', 'F1'): 'A14-YES-1t-FBFBF-5_F1',
    ('M92', 'GFP2', 'F2'): 'A14-YES-1t-FBFBF-5_F2',
    # M93: GFP1 → A14_FL_1_{field}, GFP2 → A14_FL_3_{field}
    ('M93', 'GFP1', 'F0'): 'A14_FL_1_F0',
    ('M93', 'GFP1', 'F1'): 'A14_FL_1_F1',
    ('M93', 'GFP1', 'F2'): 'A14_FL_1_F2',
    ('M93', 'GFP2', 'F0'): 'A14_FL_3_F0',
    ('M93', 'GFP2', 'F1'): 'A14_FL_3_F1',
    ('M93', 'GFP2', 'F2'): 'A14_FL_3_F2',
    # June25_20m: single GFP1/F0 → A14_10_20min
    ('June25_20m', 'GFP1', 'F0'): 'A14_10_20min',
    # M133: GFP1 → YES_Scd1_D_2, GFP2 → YES_Scd1_D_4
    ('M133', 'GFP1', 'F0'): 'YES_Scd1_D_2_F0',
    ('M133', 'GFP1', 'F1'): 'YES_Scd1_D_2_F1',
    ('M133', 'GFP1', 'F2'): 'YES_Scd1_D_2_F2',
    ('M133', 'GFP2', 'F0'): 'YES_Scd1_D_4_F0',
    ('M133', 'GFP2', 'F1'): 'YES_Scd1_D_4_F1',
    ('M133', 'GFP2', 'F2'): 'YES_Scd1_D_4_F2',
}

EXPERIMENT_BASES = {
    'Sept17':     '/Volumes/X10 Pro/Movies/2025_09_17/',
    'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/',
    'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/',
    'M133':       '/Volumes/X10 Pro/Movies/2026_04_29_M133/',
}

STACKED_CSV_PATHS = {
    'Sept17':     '/Volumes/X10 Pro/Movies/2025_09_17/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv',
    'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv',
    'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv',
    'M133':       '/Volumes/X10 Pro/Movies/2026_04_29_M133/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv',
}

ID_MAP_CSV_PATHS = {
    # Sept17 uses orig_gfp_id column directly in the stacked CSV
    'Sept17': None,
    'M92':    '/Volumes/X10 Pro/Movies/2025_12_31_M92/unaligned_pairs_quant/id_map_unaligned.csv',
    'M93':    '/Volumes/X10 Pro/Movies/2026_01_08_M93/unaligned_pairs_quant/id_map_unaligned.csv',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/unaligned_pairs_quant/id_map_unaligned.csv',
    'M133':       '/Volumes/X10 Pro/Movies/2026_04_29_M133/unaligned_pairs_quant/id_map_unaligned.csv',
}


# ==============================================================================
# RLE Decoding (wrapper around existing Cell_tracking_functions)
# ==============================================================================

def decode_rle_mask(rle_str, H, W):
    """Decode an RLE string into a (H, W) boolean mask."""
    if pd.isna(rle_str) or str(rle_str).strip() == '':
        return np.zeros((H, W), dtype=bool)
    return np.asarray(rle_decode(str(rle_str), (H, W)), dtype=bool)


# ==============================================================================
# Frame Loading
# ==============================================================================

def load_tif_frame(frames_dir, film_name, t, exp_label=None):
    """Load a single GFP TIF frame from a Frames_ directory."""
    # Priority list of possible GFP channel patterns (c_0 is usually GFP)
    patterns = [
        f"{film_name}_t_{t:03d}_c_0.tif",
        f"{film_name}_t_{t:02d}_c_0.tif",
        f"{film_name}_t_{t:03d}_z_1_c_1.tif", # Fallback for experiments with different naming
        f"{film_name}_t_{t:03d}_c_1.tif",
        f"{film_name}_t_{t:02d}_z_1_c_1.tif"
    ]
    
    for pattern in patterns:
        path = os.path.join(frames_dir, pattern)
        if os.path.exists(path):
            return imread(path).astype(np.float32)
            
    raise FileNotFoundError(f"GFP Frame not found for {film_name} t={t} in {frames_dir}")


# ==============================================================================
# Canonical Crop Extraction
# ==============================================================================

def extract_canonical_crop(img, mask, frame_h=FRAME_H, frame_w=FRAME_W, pad=BBOX_PAD):
    """
    Given a 2000×2000 GFP image and its corresponding boolean mask:
    1. Crop to the bounding box (with padding)
    2. Rotate so the major axis is horizontal (long axis = width)
    3. Re-mask (zero background)
    4. Resize to (frame_h × frame_w)
    5. Normalize: subtract background median, clip to 0, divide by 99th percentile
    
    Returns a float32 array of shape (frame_h, frame_w).
    """
    # --- 1. Bounding Box ---
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((frame_h, frame_w), dtype=np.float32)
    
    r0 = max(0, rows[0] - pad)
    r1 = min(img.shape[0], rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad)
    c1 = min(img.shape[1], cols[-1] + pad + 1)
    
    crop_img  = img[r0:r1, c0:c1].copy()
    crop_mask = mask[r0:r1, c0:c1].copy()

    # --- 2. Major Axis Orientation ---
    lab = label(crop_mask.astype(np.uint8))
    props = regionprops(lab)
    if not props:
        return np.zeros((frame_h, frame_w), dtype=np.float32)
    
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
    rotated_mask = cv2.warpAffine(padded_mask.astype(np.float32), M, (w_p, h_p),
                                   flags=cv2.INTER_NEAREST) > 0.5
    
    # --- 3. Crop tightly around rotated mask ---
    rows_r = np.where(rotated_mask.any(axis=1))[0]
    cols_r = np.where(rotated_mask.any(axis=0))[0]
    if rows_r.size == 0:
        return np.zeros((frame_h, frame_w), dtype=np.float32)
    
    r0r = max(0, rows_r[0] - 5)
    r1r = min(h_p, rows_r[-1] + 6)
    c0r = max(0, cols_r[0] - 5)
    c1r = min(w_p, cols_r[-1] + 6)
    
    tight_img  = rotated_img[r0r:r1r, c0r:c1r]
    tight_mask = rotated_mask[r0r:r1r, c0r:c1r]
    
    # --- 4. Ensure wide > tall (long axis = width) ---
    if tight_img.shape[0] > tight_img.shape[1]:
        tight_img  = np.rot90(tight_img)
        tight_mask = np.rot90(tight_mask)
    
    # --- 5. Zero Background ---
    tight_img = tight_img * tight_mask.astype(np.float32)
    
    # --- 6. Fixed Scale & Padding (Instead of Resize) ---
    # target_w / max_observed_pixels = 112 / 210 approx 0.53
    scale = frame_w / 210.0 
    
    new_h = int(tight_img.shape[0] * scale)
    new_w = int(tight_img.shape[1] * scale)
    
    # Ensure it's at least 1x1
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    # Limit to frame size
    new_h = min(frame_h, new_h)
    new_w = min(frame_w, new_w)
    
    # Resize with constant aspect ratio
    resized = resize(tight_img, (new_h, new_w), order=1, 
                     anti_aliasing=True, preserve_range=True)
    
    # Pad to (frame_h, frame_w)
    canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
    y_off = (frame_h - new_h) // 2
    x_off = (frame_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    # --- 7. Subtract Background ---
    bg_val = np.median(img[~mask]) if (~mask).any() else 0.0
    canvas = canvas - float(bg_val)
    canvas = np.clip(canvas, 0, None)
    
    return canvas.astype(np.float32)


# ==============================================================================
# Per-Cell Video Builder
# ==============================================================================

def build_cell_video(film_folder, film_name, orig_cell_id, frames_dir, n_frames=101, frame_h=FRAME_H, frame_w=FRAME_W):
    """
    Build a (101, FRAME_H, FRAME_W) video for one cell.
    
    Args:
        film_folder: Full path to the film folder (e.g., .../A14_1TP1_F1/)
        film_name:   Film name string (e.g., "A14_1TP1_F1")
        orig_cell_id: The cell ID in the TrackedCells folder
        frames_dir:  Full path to the Frames_ folder containing TIF images
        n_frames:    Number of frames to load (default 101)
    
    Returns:
        np.ndarray of shape (101, FRAME_H, FRAME_W), float32
    """
    tracked_cells_dir = os.path.join(film_folder, f"TrackedCells_{film_name}")
    mask_csv_path = os.path.join(tracked_cells_dir, f"cell_{orig_cell_id}_masks.csv")
    
    if not os.path.exists(mask_csv_path):
        raise FileNotFoundError(f"Mask CSV not found: {mask_csv_path}")
    
    df_masks = pd.read_csv(mask_csv_path)
    H = int(df_masks.iloc[0]['height'])
    W = int(df_masks.iloc[0]['width'])
    
    # Choose correct RLE column (prefer 'rle' then 'rle_gfp' etc)
    rle_col = None
    for candidate in ['rle', 'rle_gfp', 'rle_bf']:
        if candidate in df_masks.columns:
            rle_col = candidate
            break
    if rle_col is None:
        raise ValueError(f"No RLE column found in {mask_csv_path}. Columns: {df_masks.columns.tolist()}")
    
    frames = []
    for t in range(n_frames):
        try:
            row = df_masks.iloc[t]
            mask = decode_rle_mask(row[rle_col], H, W)
            # Find exp_label from film_name to help load_tif_frame
            exp_label = None
            if 'A14_10_20min' in film_name: exp_label = 'June25_20m'
            
            img = load_tif_frame(frames_dir, film_name, t, exp_label=exp_label)
            
            if mask.sum() < 100:
                # Empty or tiny mask — repeat last good frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((frame_h, frame_w), dtype=np.float32))
                continue
            
            crop = extract_canonical_crop(img, mask, frame_h=frame_h, frame_w=frame_w)
            frames.append(crop)
        except (FileNotFoundError, IndexError) as e:
            # If frame is missing, repeat last good frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((frame_h, frame_w), dtype=np.float32))
    
    video_array = np.stack(frames, axis=0)  # (101, H, W)
    
    # --- GLOBAL SEQUENCE NORMALIZATION ---
    # Find the 99th percentile across all pixels in the entire 101-frame sequence
    # that are > 0.
    pos_pixels = video_array[video_array > 0]
    p99_global = np.percentile(pos_pixels, 99) if pos_pixels.size > 0 else 1.0
    if p99_global > 0:
        video_array /= p99_global
        
    return video_array



# ==============================================================================
# ID Resolution Helpers
# ==============================================================================

def resolve_cell_info_sept17(cell_id, df_stacked):
    """
    For Sept17, the stacked CSV contains orig_gfp_id directly.
    Returns (film_name, orig_cell_id) given a new_cell_id.
    """
    rows = df_stacked[df_stacked['cell_id'] == cell_id]
    if rows.empty:
        return None, None
    row1 = rows.iloc[0]
    source = row1['source']  # 'GFP1' or 'GFP2'
    tp = int(row1['tp'])     # 1 or 2
    orig_gfp_id = int(row1['orig_gfp_id'])
    film_folder_name = FILM_FOLDER_MAP.get(('Sept17', source, tp))
    return film_folder_name, orig_gfp_id


def resolve_cell_info_generic(cell_id, df_id_map, exp_label):
    """
    For M92/M93/June25, use the id_map to get source, field, orig_str_id.
    Returns (film_folder_name, orig_cell_id_int).
    """
    rows = df_id_map[df_id_map['new_cell_id'] == cell_id]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    source = row['source']  # 'GFP1' or 'GFP2'
    field  = row['field']   # 'F0', 'F1', 'F2'
    orig_str = row['orig_str_id']  # e.g. 'F0:11'
    orig_cell_id = int(orig_str.split(':')[1])
    film_folder_name = FILM_FOLDER_MAP.get((exp_label, source, field))
    return film_folder_name, orig_cell_id


# ==============================================================================
# Main Loader
# ==============================================================================

def load_video_dataset(target_gids, exp_label_map, frame_h=FRAME_H, frame_w=FRAME_W):
    """
    Build the full video tensor for a list of global_cell_ids.
    
    Args:
        target_gids:    List of global cell IDs e.g. ['Sept17_5', 'M92_12', ...]
        exp_label_map:  Dict mapping experiment label → experiment base directory
                        (defaults to EXPERIMENT_BASES)
    
    Returns:
        videos: np.ndarray of shape (N, 101, 1, FRAME_H, FRAME_W)
        valid_gids: list of global cell IDs that were successfully loaded
    """
    # --- Load all stacked CSVs and ID maps ---
    stacked_dfs = {}
    id_map_dfs  = {}
    for exp, path in STACKED_CSV_PATHS.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            stacked_dfs[exp] = df
    for exp, path in ID_MAP_CSV_PATHS.items():
        if path and os.path.exists(path):
            id_map_dfs[exp] = pd.read_csv(path)
    
    videos = []
    valid_gids = []
    
    # Build sorted list of known experiment labels (longest first to avoid prefix collisions)
    known_labels = sorted(exp_label_map.keys(), key=lambda x: -len(x))
    
    for gid in target_gids:
        try:
            # Parse experiment label and local cell ID
            # Use known experiment labels as prefixes (handles labels with underscores)
            exp_label = None
            local_cell_id = None
            for lbl in known_labels:
                prefix = lbl + '_'
                if gid.startswith(prefix):
                    exp_label = lbl
                    local_cell_id = int(gid[len(prefix):])
                    break
            if exp_label is None:
                print(f"[skip] Cannot parse experiment from: {gid}")
                continue
            
            base_dir = exp_label_map.get(exp_label)
            if base_dir is None:
                print(f"[skip] Unknown experiment: {gid}")
                continue
            
            # Resolve film folder and orig cell ID
            if exp_label == 'Sept17':
                df_stacked = stacked_dfs.get('Sept17')
                if df_stacked is None:
                    print(f"[skip] No stacked CSV for {gid}")
                    continue
                film_name, orig_id = resolve_cell_info_sept17(local_cell_id, df_stacked)
            elif exp_label == 'June25_20m':
                # June25 doesn't have an ID map, local_id IS the orig_id
                # and it only has one film: A14_10_20min
                film_name = FILM_FOLDER_MAP.get(('June25_20m', 'GFP1', 'F0'))
                orig_id = local_cell_id
            else:
                df_id_map = id_map_dfs.get(exp_label)
                if df_id_map is None:
                    print(f"[skip] No ID map for {gid}")
                    continue
                film_name, orig_id = resolve_cell_info_generic(local_cell_id, df_id_map, exp_label)
            
            if film_name is None or orig_id is None:
                print(f"[skip] Could not resolve film for {gid}")
                continue
            
            # Build paths
            film_folder  = os.path.join(base_dir, film_name)
            frames_dir   = os.path.join(film_folder, f"Frames_{film_name}")
            
            if not os.path.isdir(frames_dir):
                print(f"[skip] Frames not found: {frames_dir}")
                continue
            
            # Build video tensor
            video = build_cell_video(film_folder, film_name, orig_id, frames_dir, frame_h=frame_h, frame_w=frame_w)
            videos.append(video[:, np.newaxis, :, :])  # (101, 1, H, W)
            valid_gids.append(gid)
            
            if len(valid_gids) % 50 == 0:
                print(f"  Loaded {len(valid_gids)}/{len(target_gids)} cells...")
                
        except Exception as e:
            print(f"[error] {gid}: {e}")
            continue
    
    if not videos:
        raise ValueError("No videos could be loaded!")
    
    video_array = np.stack(videos, axis=0)  # (N, 101, 1, H, W)
    print(f"✅ Loaded {len(valid_gids)} video tensors. Shape: {video_array.shape}")
    return video_array, valid_gids


# ==============================================================================
# Quick Test / Preview
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
    from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
    
    EXPERIMENTS = {
        'Sept17':     '/Volumes/X10 Pro/Movies/2025_09_17/',
        'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/',
        'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/',
        'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/',
    }
    
    # Load the 431 curated cell IDs from the existing pipeline
    print("Loading curated 431-cell list...")
    _, _, gids, labels, _, _ = load_feature_constrained_data(EXPERIMENTS)
    print(f"Total curated cells: {len(gids)}")
    
    # Test on first 5 cells only
    test_gids = gids[:5]
    print(f"\nBuilding video tensors for test set: {test_gids}")
    
    videos, valid = load_video_dataset(test_gids, EXPERIMENT_BASES)
    print(f"\nTest complete. Shape: {videos.shape}  |  Range: [{videos.min():.3f}, {videos.max():.3f}]")
