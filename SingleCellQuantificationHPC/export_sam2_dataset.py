#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_sam2_dataset.py
----------------------
Exports M93 and 2025_09_17 cell tracking datasets into SAM2 compatible format.
Saves under /Volumes/X10 Pro/Movies/AI/sam2_finetune_dataset/
"""

import os
import sys
import json
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage.io import imread

# Ensure local imports work
HPC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(HPC_DIR.parent))

from Cell_tracking_functions import rle_decode

BASE_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXPORT_ROOT = Path("/Volumes/X10 Pro/Movies/AI/sam2_finetune_dataset")

def check_qc_csv(tracked_dir):
    qc_csv = tracked_dir / "qc.csv"
    if not qc_csv.exists():
        return {}
    try:
        df = pd.read_csv(qc_csv)
        return {int(row["cell_id"]): str(row.get("status", "")).strip().lower() for _, row in df.iterrows()}
    except Exception:
        return {}

def discover_gt_m93():
    exp_dir = BASE_MOVIE_ROOT / "2026_01_08_M93"
    if not exp_dir.exists():
        print(f"Directory {exp_dir} does not exist.")
        return []
    
    linkage_file = exp_dir / "sequence_linkage.json"
    if not linkage_file.exists():
        print(f"Linkage file {linkage_file} does not exist.")
        return []
    with open(linkage_file, 'r') as f:
        linkage = json.load(f)
        
    results = []
    for seq in ["A14_F0", "A14_F1", "A14_F2"]:
        qc_file = exp_dir / seq / f"qc_{seq}.json"
        if not qc_file.exists():
            continue
        with open(qc_file, 'r') as f:
            qc = json.load(f)
        
        films = linkage[seq]["films"]
        global_cells = linkage[seq]["global_cells"]
        
        for key, status in qc.items():
            from tracking_corrector.qc_schema import USABLE_GLOBAL_STATUSES
            st_val = status.get("status") if isinstance(status, dict) else str(status)
            if st_val.lower() not in USABLE_GLOBAL_STATUSES:
                continue
            
            prefix = f"{seq}_"
            suffix = key[len(prefix):]
            if suffix.startswith("cell_"):
                local_ids = global_cells.get(key, [-1]*len(films))
                for film, local_id in zip(films, local_ids):
                    if local_id != -1 and "BF" in film:
                        results.append({
                            "experiment": "2026_01_08_M93",
                            "film": film,
                            "cell_id": local_id,
                            "status": status.lower(),
                            "source_qc": f"qc_{seq}.json (global)"
                        })
            else:
                parts = suffix.split("_cell_")
                if len(parts) == 2:
                    film = parts[0]
                    local_id = int(parts[1])
                    if "BF" in film:
                        results.append({
                            "experiment": "2026_01_08_M93",
                            "film": film,
                            "cell_id": local_id,
                            "status": status.lower(),
                            "source_qc": f"qc_{seq}.json (local)"
                        })

    # Also scan individual film qc.csv files
    for film_dir in exp_dir.iterdir():
        if not film_dir.is_dir() or film_dir.name.startswith("."):
            continue
        tracked_dir = film_dir / f"TrackedCells_{film_dir.name}"
        if not tracked_dir.exists():
            continue
        qc_data = check_qc_csv(tracked_dir)
        for cell_id, status in qc_data.items():
            from tracking_corrector.qc_schema import USABLE_GLOBAL_STATUSES
            if status.lower() in USABLE_GLOBAL_STATUSES:
                results.append({
                    "experiment": "2026_01_08_M93",
                    "film": film_dir.name,
                    "cell_id": cell_id,
                    "status": status,
                    "source_qc": "qc.csv"
                })
                
    return results

def discover_gt_2025_09_17():
    exp_dir = BASE_MOVIE_ROOT / "2025_09_17"
    if not exp_dir.exists():
        print(f"Directory {exp_dir} does not exist.")
        return []
        
    linkage_file = exp_dir / "sequence_linkage.json"
    if not linkage_file.exists():
        print(f"Linkage file {linkage_file} does not exist.")
        return []
    with open(linkage_file, 'r') as f:
        linkage = json.load(f)
        
    results = []
    for seq in ["F0", "F1"]:
        qc_file = exp_dir / seq / f"qc_{seq}.json"
        if not qc_file.exists():
            continue
        with open(qc_file, 'r') as f:
            qc = json.load(f)
            
        films = linkage[seq]["films"]
        global_cells = linkage[seq]["global_cells"]
        
        for key, status in qc.items():
            from tracking_corrector.qc_schema import USABLE_GLOBAL_STATUSES
            st_val = status.get("status") if isinstance(status, dict) else str(status)
            if st_val.lower() not in USABLE_GLOBAL_STATUSES:
                continue
                
            prefix = f"{seq}_"
            suffix = key[len(prefix):]
            if suffix.startswith("cell_"):
                local_ids = global_cells.get(key, [-1]*len(films))
                for film, local_id in zip(films, local_ids):
                    if local_id != -1 and "BF" in film:
                        results.append({
                            "experiment": "2025_09_17",
                            "film": film,
                            "cell_id": local_id,
                            "status": status.lower(),
                            "source_qc": f"qc_{seq}.json (global)"
                        })
            else:
                parts = suffix.split("_cell_")
                if len(parts) == 2:
                    film = parts[0]
                    local_id = int(parts[1])
                    if "BF" in film:
                        results.append({
                            "experiment": "2025_09_17",
                            "film": film,
                            "cell_id": local_id,
                            "status": status.lower(),
                            "source_qc": f"qc_{seq}.json (local)"
                        })

    # Also scan individual film qc.csv files
    for film_dir in exp_dir.iterdir():
        if not film_dir.is_dir() or film_dir.name.startswith("."):
            continue
        tracked_dir = film_dir / f"TrackedCells_{film_dir.name}"
        if not tracked_dir.exists():
            continue
        qc_data = check_qc_csv(tracked_dir)
        for cell_id, status in qc_data.items():
            from tracking_corrector.qc_schema import USABLE_GLOBAL_STATUSES
            if status.lower() in USABLE_GLOBAL_STATUSES:
                results.append({
                    "experiment": "2025_09_17",
                    "film": film_dir.name,
                    "cell_id": cell_id,
                    "status": status,
                    "source_qc": "qc.csv"
                })
                
    return results

def discover_gt_m135():
    exp_dir = BASE_MOVIE_ROOT / "2026_04_30_M135"
    if not exp_dir.exists():
        print(f"Directory {exp_dir} does not exist.")
        return []
    
    linkage_file = exp_dir / "sequence_linkage.json"
    if not linkage_file.exists():
        print(f"Linkage file {linkage_file} does not exist.")
        return []
    with open(linkage_file, 'r') as f:
        linkage = json.load(f)
        
    results = []
    for seq in ["A14_F0", "A14_F1", "A14_F2"]:
        qc_file = exp_dir / seq / f"qc_{seq}.json"
        if not qc_file.exists():
            continue
        with open(qc_file, 'r') as f:
            qc = json.load(f)
        
        films = linkage[seq]["films"]
        global_cells = linkage[seq]["global_cells"]
        
        for key, status in qc.items():
            from tracking_corrector.qc_schema import USABLE_GLOBAL_STATUSES
            st_val = status.get("status") if isinstance(status, dict) else str(status)
            if st_val.lower() not in USABLE_GLOBAL_STATUSES:
                continue
            
            prefix = f"{seq}_"
            suffix = key[len(prefix):]
            if suffix.startswith("cell_"):
                local_ids = global_cells.get(key, [-1]*len(films))
                for film, local_id in zip(films, local_ids):
                    if local_id != -1 and "BF" in film:
                        results.append({
                            "experiment": "2026_04_30_M135",
                            "film": film,
                            "cell_id": local_id,
                            "status": status.lower(),
                            "source_qc": f"qc_{seq}.json (global)"
                        })
            else:
                parts = suffix.split("_cell_")
                if len(parts) == 2:
                    film = parts[0]
                    local_id = int(parts[1])
                    if "BF" in film:
                        results.append({
                            "experiment": "2026_04_30_M135",
                            "film": film,
                            "cell_id": local_id,
                            "status": status.lower(),
                            "source_qc": f"qc_{seq}.json (local)"
                        })

    # Also scan individual film qc.csv files
    for film_dir in exp_dir.iterdir():
        if not film_dir.is_dir() or film_dir.name.startswith("."):
            continue
        tracked_dir = film_dir / f"TrackedCells_{film_dir.name}"
        if not tracked_dir.exists():
            continue
        qc_data = check_qc_csv(tracked_dir)
        for cell_id, status in qc_data.items():
            from tracking_corrector.qc_schema import USABLE_GLOBAL_STATUSES
            if status.lower() in USABLE_GLOBAL_STATUSES:
                results.append({
                    "experiment": "2026_04_30_M135",
                    "film": film_dir.name,
                    "cell_id": cell_id,
                    "status": status,
                    "source_qc": "qc.csv"
                })
                
    return results

def detect_channel(masks_dir):
    for f in sorted(masks_dir.iterdir()):
        if f.name.startswith("."):
            continue
        m = re.search(r"_t_\d+_c_(\d+)_seg\.(tif|npy)$", f.name)
        if m:
            return int(m.group(1))
    return 0

def normalize_frame(img):
    """Normalize image to standard uint8 RGB (expected by SAM2)."""
    img_f = img.astype(np.float32)
    img_min, img_max = img_f.min(), img_f.max()
    if img_max > img_min:
        img_f = (img_f - img_min) / (img_max - img_min) * 255.0
    else:
        img_f = np.zeros_like(img_f)
    img_u8 = img_f.astype(np.uint8)
    
    if img_u8.ndim == 2:
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    elif img_u8.ndim == 3:
        if img_u8.shape[2] == 4:
            img_rgb = img_u8[:, :, :3]
        else:
            img_rgb = img_u8
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    return img_rgb

def main():
    # Set random seed for reproducibility of split
    random.seed(42)
    
    print("Discovering ground truth tracks...")
    m93_gt = discover_gt_m93()
    s0917_gt = discover_gt_2025_09_17()
    m135_gt = discover_gt_m135()
    
    # Convert to DataFrames and deduplicate
    m93_df = pd.DataFrame(m93_gt)
    if not m93_df.empty:
        m93_df = m93_df.drop_duplicates(subset=["experiment", "film", "cell_id"])
    s0917_df = pd.DataFrame(s0917_gt)
    if not s0917_df.empty:
        s0917_df = s0917_df.drop_duplicates(subset=["experiment", "film", "cell_id"])
    m135_df = pd.DataFrame(m135_gt)
    if not m135_df.empty:
        m135_df = m135_df.drop_duplicates(subset=["experiment", "film", "cell_id"])
        
    all_df = pd.concat([df for df in [m93_df, s0917_df, m135_df] if not df.empty]).reset_index(drop=True)
    print(f"Found {len(all_df)} unique verified cell tracking sequences.")
    
    # Create export directories
    images_root = EXPORT_ROOT / "JPEGImages"
    masks_root = EXPORT_ROOT / "Annotations"
    images_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    
    exported_videos = []
    
    for idx, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Exporting tracks"):
        exp = row["experiment"]
        film = row["film"]
        cell_id = int(row["cell_id"])
        
        # Unique directory name for this cell video
        if "M93" in exp:
            exp_short = "m93"
        elif "M135" in exp:
            exp_short = "m135"
        else:
            exp_short = "s0917"
        video_name = f"{exp_short}_{film}_cell_{cell_id}"
        
        film_dir = BASE_MOVIE_ROOT / exp / film
        frames_dir = film_dir / f"Frames_{film}"
        masks_dir = film_dir / f"Masks_{film}"
        tracked_dir = film_dir / f"TrackedCells_{film}"
        csv_path = tracked_dir / f"cell_{cell_id}_masks.csv"
        
        if not csv_path.exists():
            print(f"  [Skip] Cell masks CSV does not exist: {csv_path}")
            continue
            
        # Load ground truth masks
        gt_df = pd.read_csv(csv_path).sort_values("time_point")
        H, W = int(gt_df.iloc[0]["height"]), int(gt_df.iloc[0]["width"])
        
        rle_col = "rle_bf"
        if "rle_gfp" in gt_df.columns and gt_df["rle_gfp"].dropna().any():
            rle_col = "rle_gfp"
            
        gt_masks = {}
        for _, r_item in gt_df.iterrows():
            t = int(r_item["time_point"])
            rle = r_item[rle_col]
            if isinstance(rle, str) and rle:
                gt_masks[t] = rle_decode(rle, (H, W)).astype(bool)
            else:
                gt_masks[t] = np.zeros((H, W), dtype=bool)
                
        # Find active frame range
        valid_ts = [t for t, m in gt_masks.items() if m.any()]
        if not valid_ts:
            print(f"  [Skip] No valid masks for Cell {cell_id} in {film}")
            continue
            
        t_start = min(valid_ts)
        t_end = max(valid_ts)
        
        # Check if already exported and up-to-date
        video_img_dir = images_root / video_name
        video_mask_dir = masks_root / video_name
        last_jpg = video_img_dir / f"{(t_end - t_start):05d}.jpg"
        last_png = video_mask_dir / f"{(t_end - t_start):05d}.png"
        if last_jpg.exists() and last_png.exists():
            csv_mtime = csv_path.stat().st_mtime
            if last_jpg.stat().st_mtime > csv_mtime and last_png.stat().st_mtime > csv_mtime:
                exported_videos.append(video_name)
                continue
        
        # Load TIFF frame paths
        channel_idx = detect_channel(masks_dir)
        frame_files = sorted([
            f for f in frames_dir.iterdir()
            if f.name.lower().endswith(".tif")
            and not f.name.endswith("_seg.tif")
            and f"_c_{channel_idx}.tif" in f.name
            and not f.name.startswith(".")
        ])
        
        if len(frame_files) <= t_end:
            print(f"  [Skip] Frame files count ({len(frame_files)}) is less than t_end ({t_end}) for cell {cell_id}")
            continue
            
        # Create output dirs for this video
        video_img_dir = images_root / video_name
        video_mask_dir = masks_root / video_name
        video_img_dir.mkdir(parents=True, exist_ok=True)
        video_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Export frames
        for t in range(t_start, t_end + 1):
            frame_idx = t - t_start
            
            # 1. Image
            frame_path = frame_files[t]
            img = imread(str(frame_path))
            img_rgb = normalize_frame(img)
            img_pil = Image.fromarray(img_rgb)
            img_pil.save(video_img_dir / f"{frame_idx:05d}.jpg", quality=95)
            
            # 2. Mask
            mask_np = gt_masks.get(t, np.zeros((H, W), dtype=bool)).astype(np.uint8)
            # Create a palettised mask (0 background, 1 cell)
            mask_pil = Image.fromarray(mask_np, mode="L").convert("P")
            palette = [0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254
            mask_pil.putpalette(palette)
            mask_pil.save(video_mask_dir / f"{frame_idx:05d}.png")
            
        exported_videos.append(video_name)
        
    print(f"Exported {len(exported_videos)} cell tracking videos successfully.")
    
    # Train / Val Split
    random.shuffle(exported_videos)
    split_idx = int(0.8 * len(exported_videos))
    train_videos = sorted(exported_videos[:split_idx])
    val_videos = sorted(exported_videos[split_idx:])
    
    train_txt = EXPORT_ROOT / "train_filelist.txt"
    val_txt = EXPORT_ROOT / "val_filelist.txt"
    
    with open(train_txt, "w") as f:
        for v in train_videos:
            f.write(f"{v}\n")
            
    with open(val_txt, "w") as f:
        for v in val_videos:
            f.write(f"{v}\n")
            
    print(f"Saved {len(train_videos)} train videos list to {train_txt}")
    print(f"Saved {len(val_videos)} val videos list to {val_txt}")

if __name__ == "__main__":
    main()
