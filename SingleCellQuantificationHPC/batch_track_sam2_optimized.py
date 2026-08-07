#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_track_sam2_optimized.py
-----------------------------
Tracks all curation-discovered cells in a target film using the Fine-Tuned SAM2
model, running sequentially and fully optimized (model loaded once in VRAM,
JPEG frames exported once to temp directory).

Writes output CSV masks to:
/Volumes/X10 Pro/Movies/{experiment}/{film}/TrackedCells_{film}_SAM2_Finetuned/

And compiles population movies at the end.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
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
    """Periodically writes to a file on the external drive to prevent it from sleeping."""
    keep_alive_file = drive_path / ".keep_alive"
    while True:
        try:
            with open(keep_alive_file, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        time.sleep(interval)


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


def make_side_by_side(v1_path, v2_path, out_path, label1, label2):
    print(f"Creating side-by-side: {v1_path} vs {v2_path} -> {out_path}")
    cap1 = cv2.VideoCapture(str(v1_path))
    cap2 = cv2.VideoCapture(str(v2_path))
    
    if not cap1.isOpened():
        raise RuntimeError(f"Could not open {v1_path}")
    if not cap2.isOpened():
        raise RuntimeError(f"Could not open {v2_path}")
        
    fps = cap1.get(cv2.CAP_PROP_FPS) or 10.0
    target_h, target_w = 800, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w * 2, target_h))
    
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        f1_resized = cv2.resize(frame1, (target_w, target_h))
        f2_resized = cv2.resize(frame2, (target_w, target_h))
        
        # Add labels
        cv2.rectangle(f1_resized, (15, 15), (780, 75), (0, 0, 0), -1)
        cv2.putText(f1_resized, label1, (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(f2_resized, (15, 15), (780, 75), (0, 0, 0), -1)
        cv2.putText(f2_resized, label2, (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
        
        combined = np.hstack((f1_resized, f2_resized))
        out.write(combined)
        frame_idx += 1
        
    cap1.release()
    cap2.release()
    out.release()
    print(f"Wrote {frame_idx} frames to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="2025_09_17", help="Experiment name")
    parser.add_argument("--film", type=str, default="A14_1TP2_BF_F1", help="Film name")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu, or cuda)")
    parser.add_argument("--limit-cells", type=int, default=0, help="Max cells to process (0 = all)")
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    film_dir = BASE_MOVIE_ROOT / args.experiment / args.film
    if not film_dir.exists():
        raise FileNotFoundError(f"Film directory not found: {film_dir}")

    frames_dir  = film_dir / f"Frames_{args.film}"
    masks_dir   = film_dir / f"Masks_{args.film}"
    tracked_dir = film_dir / f"TrackedCells_{args.film}"

    if not tracked_dir.exists():
        raise FileNotFoundError(f"Curated tracked cells directory not found: {tracked_dir}")

    # Create output directory for SAM2 masks
    tracked_out_dir = film_dir / f"TrackedCells_{args.film}_SAM2_Finetuned"
    tracked_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Start keep-alive daemon thread for the external SSD
    print("Starting SSD keep-alive thread...")
    t_alive = threading.Thread(target=keep_alive_drive, args=(BASE_MOVIE_ROOT,), daemon=True)
    t_alive.start()

    # 2. Discover cells to process
    all_cell_files = sorted([f for f in tracked_dir.iterdir() if f.name.startswith("cell_") and f.name.endswith("_masks.csv")])
    cell_ids = []
    for cf in all_cell_files:
        m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
        if m:
            cell_ids.append(int(m.group(1)))
    cell_ids = sorted(cell_ids)

    if args.limit_cells > 0:
        cell_ids = cell_ids[:args.limit_cells]
        print(f"Limiting execution to first {len(cell_ids)} cells.")

    print(f"Found {len(cell_ids)} cell tracks to process.")

    # Deduplicate / check which ones are already processed to support resume
    cells_to_process = []
    for cid in cell_ids:
        out_csv = tracked_out_dir / f"cell_{cid}_masks.csv"
        if out_csv.exists() and out_csv.stat().st_size > 100:
            # Check if valid
            try:
                test_df = pd.read_csv(out_csv)
                if not test_df.empty and "rle_bf" in test_df.columns:
                    continue
            except Exception:
                pass
        cells_to_process.append(cid)

    print(f"Already processed: {len(cell_ids) - len(cells_to_process)} cells. Remaining to process: {len(cells_to_process)} cells.")
    if not cells_to_process:
        print("All cells already processed. Proceeding directly to movie generation.")
    else:
        # 3. Discover frame files and channels
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

        # 4. Copy and normalize frames to JPEGs once in local temp directory
        print(f"Exporting {T} frames to local temporary JPEGs once...")
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
        
        for t_idx, fpath in enumerate(tqdm(frame_files, desc="Exporting frames")):
            img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Could not read frame: {fpath}")
            img_rgb = normalize_frame(img)
            img_pil = Image.fromarray(img_rgb)
            img_pil.save(temp_dir / f"{t_idx:05d}.jpg", quality=95)

        # 5. Load model checkpoint once
        print(f"Loading Fine-Tuned SAM2 predictor from {SAM2_CKPT}...")
        if not SAM2_CKPT.exists():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {SAM2_CKPT}")
        predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=args.device)

        # 6. Initialize tracking state once
        print("Initializing predictor state...")
        inference_state = predictor.init_state(video_path=str(temp_dir))

        # 7. Loop through cells and track sequentially
        print(f"Tracking {len(cells_to_process)} cells sequentially on {args.device}...")
        for idx, cid in enumerate(cells_to_process):
            t_start_cell = time.time()
            print(f"\n[{idx+1}/{len(cells_to_process)}] Tracking Cell {cid}...")
            
            csv_path = tracked_dir / f"cell_{cid}_masks.csv"
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
                    
            valid_ts = [t for t, m in gt_masks.items() if m.any()]
            if not valid_ts:
                print(f"  [Skip] No valid non-empty masks found in ground truth for cell {cid}.")
                continue
                
            t_start = min(valid_ts)
            initial_mask = gt_masks[t_start]

            # Clear any leftover prompt state
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

            # Propagate forward
            cell_results = {}
            with torch.inference_mode():
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=t_start
                ):
                    mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                    cell_results[out_frame_idx] = mask

            # Build output CSV rows for all frames 0..T-1
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

            # Save cell CSV
            out_csv = tracked_out_dir / f"cell_{cid}_masks.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            elapsed = time.time() - t_start_cell
            print(f"  Completed Cell {cid} in {elapsed:.2f}s  (Tracked frames: {T - t_start})")

        # Cleanup predictor state
        predictor.reset_state(inference_state)
        del predictor
        temp_dir_obj.cleanup()

    # 8. Compile population movie for fine-tuned SAM2
    py_exec = "/Users/user/miniforge3/envs/cellpose-sam/bin/python"
    sam2_out_local = HPC_DIR / f"{args.film}_SAM2_Finetuned_population.mp4"
    print(f"\nCompiling fine-tuned SAM2 population movie to: {sam2_out_local}...")
    
    cmd_sam2 = (
        f"KMP_DUPLICATE_LIB_OK=TRUE {py_exec} {ROOT_DIR}/make_population_movie.py "
        f"--base_dir \"{film_dir}\" --frames_dir \"Frames_{args.film}\" "
        f"--cells_dir \"TrackedCells_{args.film}_SAM2_Finetuned\" "
        f"--out \"{sam2_out_local}\" --fps 10 --alpha 0.4"
    )
    os.system(cmd_sam2)

    # 9. Compile side-by-side comparison with baseline curated GT
    baseline_out = film_dir / f"{args.film}_baseline_population.mp4"
    if not baseline_out.exists():
        print(f"Generating baseline population movie for comparison...")
        cmd_base = (
            f"KMP_DUPLICATE_LIB_OK=TRUE {py_exec} {ROOT_DIR}/make_population_movie.py "
            f"--base_dir \"{film_dir}\" --frames_dir \"Frames_{args.film}\" "
            f"--cells_dir \"TrackedCells_{args.film}\" "
            f"--out \"{baseline_out}\" --fps 10 --alpha 0.4"
        )
        os.system(cmd_base)

    sxs_out_local = HPC_DIR / f"{args.film}_SAM2_vs_Baseline_comparison.mp4"
    if baseline_out.exists() and sam2_out_local.exists():
        make_side_by_side(
            v1_path=baseline_out,
            v2_path=sam2_out_local,
            out_path=sxs_out_local,
            label1="Curated Ground Truth",
            label2="Fine-Tuned SAM2 Tracker (MPS)"
        )

    # 10. Copy final videos to SSD
    print("Copying final videos to SSD...")
    ssd_target_sam2 = film_dir / f"{args.film}_SAM2_Finetuned_population.mp4"
    ssd_target_sxs = film_dir / f"{args.film}_SAM2_vs_Baseline_comparison.mp4"

    if sam2_out_local.exists():
        shutil.copy2(sam2_out_local, ssd_target_sam2)
        print(f"Copied SAM2 movie to: {ssd_target_sam2}")
    if sxs_out_local.exists():
        shutil.copy2(sxs_out_local, ssd_target_sxs)
        print(f"Copied Comparison movie to: {ssd_target_sxs}")

    print("\n🎉 BATCH SAM2 TRACKING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
