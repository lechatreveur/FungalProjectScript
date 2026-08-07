#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pregenerate_and_jumps.py
------------------------
Offline script to:
1. Delete old stale PopulationFrames directories & CellCrops directories
2. Run population frame pregeneration & gallery cell crop pregeneration
3. Recalculate suspicious jumps and write the updated JSON cache to disk.
"""

import os
import sys
import re
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from PIL import Image

# Setup paths to import tracking_corrector package
HPC_DIR = Path(__file__).parent.resolve()
if str(HPC_DIR) not in sys.path:
    sys.path.append(str(HPC_DIR))

from tracking_corrector.config import config as default_config
from tracking_corrector.repositories.mask_repository import MaskRepository
from tracking_corrector.repositories.linkage_repository import LinkageRepository
from tracking_corrector.services.frames_service import FramesService
from tracking_corrector.schemas import validate_and_decode_rle

def get_centroid_from_rle(rle_str, H, W):
    if not rle_str or not isinstance(rle_str, str) or rle_str.strip() == "" or rle_str == "nan":
        return None
    try:
        mask = validate_and_decode_rle(rle_str, H, W)
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        return (float(np.mean(ys)), float(np.mean(xs)))
    except Exception:
        return None

def generate_cell_crops_for_film(
    frames_service: FramesService,
    exp: str,
    film: str,
    crop_size=100,
    force=False,
):
    base_root = default_config.local_movie_root
    tracked_dir = base_root / exp / film / f"TrackedCells_{film}"
    if not tracked_dir.is_dir():
        return
        
    cache_dir = base_root / exp / film / f"CellCrops_{film}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cell_files = [f for f in tracked_dir.iterdir() if f.name.endswith("_masks.csv") and not f.name.startswith(".")]
    if not cell_files:
        return
        
    frames_dir = base_root / exp / film / f"Frames_{film}"
    if not frames_dir.is_dir():
        return
        
    print(f"[{film}] Pre-generating cell gallery crops for {len(cell_files)} cells...")
    
    t_to_crops = {}
    for cf in cell_files:
        m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
        if not m:
            continue
        cell_id = int(m.group(1))
        try:
            df = pd.read_csv(cf)
            if df.empty:
                continue
            H = int(df.iloc[0]['height']) if 'height' in df.columns else 0
            W = int(df.iloc[0]['width']) if 'width' in df.columns else 0
            if H == 0 or W == 0:
                continue
                
            for idx, row in df.iterrows():
                t_val = int(row['time_point']) if 'time_point' in row else idx
                for ch in ['bf', 'gfp']:
                    rle_col = f"rle_{ch}"
                    rle = str(row.get(rle_col, "")) if rle_col in row else ""
                    if rle and rle.strip() and rle.lower() != 'nan':
                        cache_file = cache_dir / f"cell_{cell_id}_t_{t_val:03d}_{ch}.jpg"
                        if cache_file.exists() and not force:
                            continue
                        cy, cx = H // 2, W // 2
                        centroid = get_centroid_from_rle(rle, H, W)
                        if centroid is not None:
                            cy, cx = int(centroid[0]), int(centroid[1])
                        if t_val not in t_to_crops:
                            t_to_crops[t_val] = []
                        t_to_crops[t_val].append((cell_id, cy, cx, ch, H, W, cache_file))
        except Exception:
            pass
            
    if not t_to_crops:
        print(f"[{film}] All cell gallery crops already cached.")
        return
        
    generated_crops = 0
    for t_val, crop_requests in sorted(t_to_crops.items()):
        t_files = sorted([f for f in frames_dir.glob(f"{film}_t_{t_val:03d}_c_*.tif") if not f.name.startswith(".")])
        if not t_files:
            t_files = sorted([f for f in frames_dir.glob(f"*_t_{t_val:03d}_c_*.tif") if not f.name.startswith(".")])
        if not t_files:
            continue
            
        for ch in ['bf', 'gfp']:
            ch_requests = [r for r in crop_requests if r[3] == ch]
            if not ch_requests:
                continue
                
            target_c = 1 if ch == 'gfp' else 0
            ch_file = None
            for tf in t_files:
                m_c = re.search(r"_c_(\d+)\.", tf.name)
                c_num = int(m_c.group(1)) if m_c else 0
                if c_num == target_c:
                    ch_file = tf
                    break
            if ch_file is None:
                ch_file = t_files[0]
                
            try:
                img = imread(str(ch_file))
                fH, fW = img.shape[:2]
                for cell_id, cy, cx, _, H, W, cache_file in ch_requests:
                    y0 = max(0, cy - crop_size // 2)
                    y1 = min(fH, cy + crop_size // 2)
                    x0 = max(0, cx - crop_size // 2)
                    x1 = min(fW, cx + crop_size // 2)
                    
                    crop = img[y0:y1, x0:x1]
                    if crop.size == 0:
                        crop = np.zeros((crop_size, crop_size), dtype=np.uint8)
                    af = crop.astype(np.float32)
                    p_lo = np.percentile(af, 1.0) if np.isfinite(af).any() else 0.0
                    p_hi = np.percentile(af, 99.5) if np.isfinite(af).any() else 255.0
                    
                    if p_hi > p_lo:
                        img_scaled = np.clip((af - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        img_scaled = crop.astype(np.uint8)
                        
                    im = Image.fromarray(img_scaled)
                    im.save(cache_file, format="JPEG", quality=85)
                    generated_crops += 1
            except Exception:
                pass

    print(f"[{film}] Pre-generated {generated_crops} gallery cell crops.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="2026_07_16_M156", help="Experiment name")
    parser.add_argument("--sequence", type=str, default="3_F0", help="Sequence name")
    parser.add_argument("--films", type=str, default="", help="Comma-separated film list")
    parser.add_argument("--threshold", type=float, default=15.0, help="Suspicious jump threshold in pixels")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing population-frame and gallery-crop cache files.",
    )
    args = parser.parse_args()

    exp = args.experiment
    sequence = args.sequence
    threshold = args.threshold
    
    base_root = default_config.local_movie_root
    mask_repo = MaskRepository(base_root)
    linkage_repo = LinkageRepository(base_root)
    frames_service = FramesService(default_config, mask_repo)
    
    seq_linkage_data, _ = linkage_repo.load_linkage(exp)
    seq_linkage = seq_linkage_data if isinstance(seq_linkage_data, dict) else {}
    if args.films:
        films = [f.strip() for f in args.films.split(",") if f.strip()]
    elif sequence in seq_linkage:
        films = seq_linkage[sequence].get("films", [])
    else:
        films = []

    print(f"=== Starting Pregeneration & Jump Calculation for {exp} ({sequence}) ===")
    print(f"Films: {films}")

    # 1. Clear suspicious jumps file if sequence specified
    if sequence:
        susp_file = base_root / exp / sequence / f"suspicious_{sequence}.json"
        if susp_file.exists():
            print(f"Deleting old suspicious jumps file: {susp_file}")
            susp_file.unlink()

    # 2. Run frame pre-generation & gallery cell crop generation for each film
    for film in films:
        print(f"\n--- Generating population frames for {film} ---")
        paths = frames_service.get_film_frame_paths(exp, film, "bf")
        cache_dir = base_root / exp / film / f"PopulationFrames_{film}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        gen_count = 0
        for t_val in sorted(paths.keys()):
            cache_file = cache_dir / f"frame_{t_val:03d}.jpg"
            if args.force or not cache_file.exists():
                try:
                    frames_service._generate_population_frame_bytes(exp, film, t_val, cache_file)
                    gen_count += 1
                except Exception as e:
                    print(f"Error generating population frame t={t_val} for {film}: {e}")
        print(f"[{film}] Pre-generated {gen_count} population frames.")
        
        print(f"--- Generating gallery cell crops for {film} ---")
        generate_cell_crops_for_film(frames_service, exp, film, force=args.force)

    # 3. Recalculate suspicious jumps
    if sequence in seq_linkage:
        print(f"\n--- Recalculating suspicious jumps for sequence {sequence} ---")
        cell_mappings = seq_linkage[sequence].get("global_cells", {})
        suspicious_data = {}

        for cell_id, local_ids in cell_mappings.items():
            film_dfs = []
            for i, f_name in enumerate(films):
                local_id = local_ids[i] if i < len(local_ids) else -1
                if local_id == -1:
                    film_dfs.append(None)
                    continue
                try:
                    df, _ = mask_repo.load_cell_masks(exp, f_name, local_id)
                    film_dfs.append(df)
                except Exception:
                    film_dfs.append(None)

            all_rles = []
            H, W = 0, 0
            for i, df in enumerate(film_dfs):
                f_name = films[i]
                if df is not None and len(df) > 0:
                    if H == 0:
                        H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
                    
                    rle_col = 'rle_bf'
                    if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                        rle_col = 'rle_gfp'
                    elif 'rle_bf' not in df.columns and 'rle_gfp' in df.columns:
                        rle_col = 'rle_gfp'
                        
                    masks = df[rle_col].fillna("").tolist()
                    all_rles.extend(masks)

            if H == 0 or W == 0:
                continue

            centroids = []
            for rle in all_rles:
                centroids.append(get_centroid_from_rle(rle, H, W))

            suspicious_frames = []
            for t in range(1, len(centroids)):
                c1 = centroids[t-1]
                c2 = centroids[t]
                if c1 is not None and c2 is not None:
                    dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    if dist > threshold:
                        suspicious_frames.append(t)

            if suspicious_frames:
                suspicious_data[str(cell_id)] = suspicious_frames

        # Save to disk cache
        try:
            susp_file = base_root / exp / sequence / f"suspicious_{sequence}.json"
            susp_file.parent.mkdir(parents=True, exist_ok=True)
            with open(susp_file, "w", encoding="utf-8") as f:
                json.dump(suspicious_data, f)
            print(f"\n🎉 Successfully calculated jumps for {len(suspicious_data)} cells. Saved cache to {susp_file}")
        except Exception as e:
            print(f"Error writing suspicious jumps JSON file: {e}")

if __name__ == "__main__":
    main()
