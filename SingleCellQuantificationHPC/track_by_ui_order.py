#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_by_ui_order.py
---------------------
Tracks pending (uncurated) cells in sequence F1 according to the sorted UI cell order.
Supports tracking either the first 50 cell IDs or the remaining ones.

Writes mask CSV files to:
- TrackedCells_{film} (active curated folder)
- TrackedCells_{film}_SAM2_Finetuned (SAM2 folder)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import re
import argparse
import tempfile
import shutil
import threading
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Ensure local imports work
HPC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = HPC_DIR.parent
SAM2_DIR = ROOT_DIR / "segment-anything-2"
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SAM2_DIR))

from Cell_tracking_functions import rle_decode, rle_encode
from sam2.build_sam import build_sam2_video_predictor

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
SAM2_CFG        = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CKPT       = ROOT_DIR / "segment-anything-2" / "checkpoints" / "sam2.1_hiera_tiny_fungal_finetuned.pt"


def keep_alive_drive(drive_path: Path, interval=10):
    keep_alive_file = drive_path / ".keep_alive"
    while True:
        try:
            with open(keep_alive_file, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        time.sleep(interval)


def normalize_frame(img):
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


def get_curated_cells(experiment, sequence, film, tracked_dir):
    curated_ids = set()
    exp_dir = BASE_MOVIE_ROOT / experiment
    
    # 1. Check sequence linkages & global qc.json
    linkage_file = exp_dir / "sequence_linkage.json"
    qc_file = exp_dir / sequence / f"qc_{sequence}.json"
    
    if linkage_file.exists() and qc_file.exists():
        with open(linkage_file, 'r') as f:
            linkage = json.load(f)
        with open(qc_file, 'r') as f:
            qc = json.load(f)
            
        films = linkage.get(sequence, {}).get("films", [])
        global_cells = linkage.get(sequence, {}).get("global_cells", {})
        
        from tracking_corrector.qc_schema import GlobalCellQC
        if film in films:
            film_idx = films.index(film)
            for g_id, status in qc.items():
                st_val = status.get("status") if isinstance(status, dict) else str(status)
                if st_val.lower() not in GlobalCellQC.valid_statuses():
                    continue
                if g_id in global_cells:
                    local_ids = global_cells[g_id]
                    if film_idx < len(local_ids):
                        local_id = local_ids[film_idx]
                        if local_id != -1:
                            curated_ids.add(local_id)
                else:
                    prefix = f"{sequence}_{film}_cell_"
                    if g_id.startswith(prefix):
                        try:
                            local_id = int(g_id[len(prefix):])
                            curated_ids.add(local_id)
                        except ValueError:
                            pass

    # 2. Check local qc.csv
    qc_csv = tracked_dir / "qc.csv"
    if qc_csv.exists():
        try:
            df = pd.read_csv(qc_csv)
            for _, row in df.iterrows():
                cid = int(row["cell_id"])
                status = str(row.get("status", "")).strip().lower()
                if status in GlobalCellQC.valid_statuses():
                    curated_ids.add(cid)
        except Exception:
            pass

            
    return curated_ids


def get_sorted_global_cells(experiment, sequence):
    exp_dir = BASE_MOVIE_ROOT / experiment
    linkage_file = exp_dir / "sequence_linkage.json"
    with open(linkage_file, 'r') as f:
        linkage = json.load(f)
    
    global_cells = linkage.get(sequence, {}).get("global_cells", {})
    
    def get_sort_key(k):
        s = str(k)
        m = re.search(r"(\d+)$", s)
        if m:
            return (0, int(m.group(1)))
        return (1, s)
        
    return sorted(list(global_cells.keys()), key=get_sort_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="2025_09_17", help="Experiment name")
    parser.add_argument("--sequence", type=str, default="F1", help="Sequence name")
    parser.add_argument("--film", type=str, required=True, help="Film name (e.g. A14_1TP1_BF_F1)")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu, or cuda)")
    parser.add_argument("--range-start", type=int, default=0, help="Start index in the sorted cell list")
    parser.add_argument("--range-end", type=int, default=50, help="End index in the sorted cell list")
    args = parser.parse_args()

    film_dir = BASE_MOVIE_ROOT / args.experiment / args.film
    if not film_dir.exists():
        raise FileNotFoundError(f"Film directory not found: {film_dir}")

    frames_dir  = film_dir / f"Frames_{args.film}"
    masks_dir   = film_dir / f"Masks_{args.film}"
    tracked_dir = film_dir / f"TrackedCells_{args.film}"

    if not tracked_dir.exists():
        raise FileNotFoundError(f"Curated tracked cells directory not found: {tracked_dir}")

    tracked_out_dir = film_dir / f"TrackedCells_{args.film}_SAM2_Finetuned"
    tracked_out_dir.mkdir(parents=True, exist_ok=True)

    # Start SSD keep-alive thread
    print("Starting SSD keep-alive thread...")
    t_alive = threading.Thread(target=keep_alive_drive, args=(BASE_MOVIE_ROOT,), daemon=True)
    t_alive.start()

    # Get sorted global cells list (UI order)
    sorted_global = get_sorted_global_cells(args.experiment, args.sequence)
    print(f"Total global cells in sequence: {len(sorted_global)}")

    # Slice the range of cell IDs
    target_global = sorted_global[args.range_start:args.range_end]
    print(f"Target global cells (indices {args.range_start} to {args.range_end}): {len(target_global)}")
    print(f"  First cell: {target_global[0] if target_global else None}")
    print(f"  Last cell: {target_global[-1] if target_global else None}")

    # Load linkage mapping
    exp_dir = BASE_MOVIE_ROOT / args.experiment
    with open(exp_dir / "sequence_linkage.json", "r") as f:
        linkage = json.load(f)
    films_list = linkage.get(args.sequence, {}).get("films", [])
    if args.film not in films_list:
        raise ValueError(f"Film {args.film} not found in sequence {args.sequence} films: {films_list}")
    film_idx = films_list.index(args.film)

    # Determine local cell IDs to process for this film
    all_local_cell_ids = []
    global_to_local = {}
    for g_id in target_global:
        local_ids = linkage[args.sequence]["global_cells"][g_id]
        local_id = local_ids[film_idx] if film_idx < len(local_ids) else -1
        if local_id != -1:
            all_local_cell_ids.append(local_id)
            global_to_local[g_id] = local_id

    # Filter out curated cells
    curated_ids = get_curated_cells(args.experiment, args.sequence, args.film, tracked_dir)
    curated_to_process = [cid for cid in all_local_cell_ids if cid in curated_ids]
    cells_to_process = [cid for cid in all_local_cell_ids if cid not in curated_ids]

    print(f"\nFor Film {args.film}:")
    print(f"  - Curated (skipped tracking, preserved in active): {len(curated_to_process)} cells")
    print(f"  - Pending (will run fine-tuned SAM2): {len(cells_to_process)} cells")

    # Copy curated cells to the SAM2 folder
    for cid in curated_to_process:
        src_csv = tracked_dir / f"cell_{cid}_masks.csv"
        dst_csv = tracked_out_dir / f"cell_{cid}_masks.csv"
        shutil.copy2(src_csv, dst_csv)

    if not cells_to_process:
        print("\nNo pending cells to track in this range for this film.")
        return

    # Discover frame files
    channel_idx = 0
    for f in sorted(masks_dir.iterdir()):
        if f.name.startswith("."):
            continue
        m = re.search(r"_t_\d+_c_(\d+)_seg\.(tif|npy)$", f.name)
        if m:
            channel_idx = int(m.group(1))
            break
            
    frame_files = sorted([
        f for f in frames_dir.iterdir()
        if f.name.lower().endswith(".tif")
        and not f.name.endswith("_seg.tif")
        and f"_c_{channel_idx}.tif" in f.name
        and not f.name.startswith(".")
    ])

    T = len(frame_files)
    if T == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # Copy and normalize frames to JPEGs once
    print(f"\nExporting {T} frames to local temporary JPEGs...")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = Path(temp_dir_obj.name)
    
    for t_idx, fpath in enumerate(tqdm(frame_files, desc="Exporting frames")):
        img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read frame: {fpath}")
        img_rgb = normalize_frame(img)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(temp_dir / f"{t_idx:05d}.jpg", quality=95)

    # Load model
    print(f"\nLoading Fine-Tuned SAM2 predictor from {SAM2_CKPT}...")
    if not SAM2_CKPT.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {SAM2_CKPT}")
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=args.device)

    # Initialize predictor state
    print("Initializing predictor state...")
    inference_state = predictor.init_state(video_path=str(temp_dir))

    # Track cells sequentially
    print(f"\nTracking {len(cells_to_process)} pending cells sequentially on {args.device}...")
    for idx, cid in enumerate(cells_to_process):
        out_csv = tracked_out_dir / f"cell_{cid}_masks.csv"
        if out_csv.exists():
            print(f"[{idx+1}/{len(cells_to_process)}] Skipping Pending Cell {cid} (already tracked)")
            continue

        t_start_cell = time.time()
        print(f"[{idx+1}/{len(cells_to_process)}] Tracking Pending Cell {cid}...")
        
        csv_path = tracked_dir / f"cell_{cid}_masks.csv"
        gt_df = pd.read_csv(csv_path).sort_values("time_point")
        H, W = int(gt_df.iloc[0]["height"]), int(gt_df.iloc[0]["width"])
        
        rle_col = "rle_bf"
        if "rle_gfp" in gt_df.columns and gt_df["rle_gfp"].dropna().any():
            rle_col = "rle_gfp"
            
        gt_masks = {}
        for _, row in gt_df.iterrows():
            t = int(row["time_point"])
            rle = row[rle_col]
            if isinstance(rle, str) and rle:
                gt_masks[t] = rle_decode(rle, (H, W)).astype(bool)
            else:
                gt_masks[t] = np.zeros((H, W), dtype=bool)
                
        valid_ts = [t for t, m in gt_masks.items() if m.any()]
        if not valid_ts:
            print(f"  [Skip] No valid non-empty masks found for cell {cid}.")
            continue
            
        t_start = min(valid_ts)
        initial_mask = gt_masks[t_start]

        # Reset state
        predictor.reset_state(inference_state)

        # Add initial mask prompt
        mask_tensor = torch.tensor(initial_mask, dtype=torch.bool, device=args.device)
        with torch.inference_mode():
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=t_start,
                obj_id=1,
                mask=mask_tensor
            )

        # Propagate
        cell_results = {}
        with torch.inference_mode():
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=t_start
            ):
                mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                cell_results[out_frame_idx] = mask

        # Save to CSV
        rows = []
        for t in range(T):
            row = {
                "time_point": t,
                "width": W,
                "height": H,
                "rle_bf": "",
                "touches_border_bf": False,
                "source_bf": "sam2_finetuned",
                "overlap_score_bf": 1.0,
                "smooth_score_bf": 1.0,
                "area_bf": 0,
                "area_penalty_bf": 0.0,
                "huge_jump_rejected_bf": False,
                "composition_bf": "",
                "pair_segA_rle_bf": "",
                "pair_segB_rle_bf": "",
                "selector_mode_bf": "sam2"
            }

            if t >= t_start:
                pred_mask = cell_results.get(t, np.zeros((H, W), dtype=bool))
                if pred_mask.any():
                    row["rle_bf"] = rle_encode(pred_mask)
                    row["area_bf"] = int(pred_mask.sum())
            rows.append(row)

        out_finetune = tracked_out_dir / f"cell_{cid}_masks.csv"
        pd.DataFrame(rows).to_csv(out_finetune, index=False)
        
        out_curated = tracked_dir / f"cell_{cid}_masks.csv"
        pd.DataFrame(rows).to_csv(out_curated, index=False)

        elapsed = time.time() - t_start_cell
        print(f"  Completed Cell {cid} in {elapsed:.2f}s  (Tracked frames: {T - t_start})")

    # Cleanup state
    predictor.reset_state(inference_state)
    del predictor
    temp_dir_obj.cleanup()
    print(f"\n🎉 Finished tracking target range for {args.film}!")


if __name__ == "__main__":
    main()
