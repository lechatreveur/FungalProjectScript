#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_pending_only.py
---------------------
Retracks ONLY the pending (uncurated) cells in a film using the fine-tuned SAM2 predictor,
leaving manual curations (good/corrected/bad) completely untouched.

Writes output CSV masks to both the active curated folder (TrackedCells_{film})
and the fine-tuned SAM2 tracker folder (TrackedCells_{film}_SAM2_Finetuned).
Also updates the population movies and comparisons at the end.
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


def get_curated_cells(experiment, sequence, film, tracked_dir):
    """Discovers which cell IDs in this film have been curated in the sequence QC json or film qc.csv."""
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
        
        if film in films:
            film_idx = films.index(film)
            
            # Scan global keys
            for g_id, status in qc.items():
                if status.lower() not in ["good", "corrected", "bad"]:
                    continue
                
                # Global cells
                if g_id in global_cells:
                    local_ids = global_cells[g_id]
                    if film_idx < len(local_ids):
                        local_id = local_ids[film_idx]
                        if local_id != -1:
                            curated_ids.add(local_id)
                # Local keys e.g. F1_A14_1TP2_BF_F1_cell_87
                else:
                    prefix = f"{sequence}_{film}_cell_"
                    if g_id.startswith(prefix):
                        try:
                            local_id = int(g_id[len(prefix):])
                            curated_ids.add(local_id)
                        except ValueError:
                            pass

    # 2. Check local qc.csv inside TrackedCells_{film}
    qc_csv = tracked_dir / "qc.csv"
    if qc_csv.exists():
        try:
            df = pd.read_csv(qc_csv)
            for _, row in df.iterrows():
                cid = int(row["cell_id"])
                status = str(row.get("status", "")).strip().lower()
                if status in ["good", "corrected", "bad"]:
                    curated_ids.add(cid)
        except Exception:
            pass
            
    return curated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="2025_09_17", help="Experiment name")
    parser.add_argument("--sequence", type=str, default="F1", help="Sequence name")
    parser.add_argument("--film", type=str, default="A14_1TP2_BF_F1", help="Film name")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu, or cuda)")
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if args.experiment == "2026_04_30_M135":
        import sys
        print(f"Skipping tracking for M135 experiment {args.film} as requested by user.")
        sys.exit(0)

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

    # Start keep-alive daemon thread for external SSD
    print("Starting SSD keep-alive thread...")
    t_alive = threading.Thread(target=keep_alive_drive, args=(BASE_MOVIE_ROOT,), daemon=True)
    t_alive.start()

    # Discover all local cells
    all_cell_files = sorted([f for f in tracked_dir.iterdir() if f.name.startswith("cell_") and f.name.endswith("_masks.csv")])
    all_cell_ids = []
    for cf in all_cell_files:
        m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
        if m:
            all_cell_ids.append(int(m.group(1)))
    all_cell_ids = sorted(all_cell_ids)

    # Get curated cell IDs
    curated_ids = get_curated_cells(args.experiment, args.sequence, args.film, tracked_dir)
    
    # Separate into curated and pending
    curated_to_process = [cid for cid in all_cell_ids if cid in curated_ids]
    cells_to_process = [cid for cid in all_cell_ids if cid not in curated_ids]

    print(f"\nDiscovered {len(all_cell_ids)} total cells:")
    print(f"  - Curated (skipped tracking, preserved in active): {len(curated_to_process)} cells")
    print(f"  - Pending (will run fine-tuned SAM2): {len(cells_to_process)} cells")

    # 1. Copy curated cells masks directly to the Finetuned SAM2 folder so they are represented in the final population videos
    for cid in curated_to_process:
        src_csv = tracked_dir / f"cell_{cid}_masks.csv"
        dst_csv = tracked_out_dir / f"cell_{cid}_masks.csv"
        shutil.copy2(src_csv, dst_csv)

    if not cells_to_process:
        print("\nAll cells are curated! No pending cells to track.")
    else:
        # 2. Discover frame files and channels
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

        # 3. Copy and normalize frames to JPEGs once in local temp directory
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

        # 4. Load model checkpoint once
        print(f"\nLoading Fine-Tuned SAM2 predictor from {SAM2_CKPT}...")
        if not SAM2_CKPT.exists():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {SAM2_CKPT}")
        predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=args.device)

        # 5. Initialize tracking state once
        print("Initializing predictor state...")
        inference_state = predictor.init_state(video_path=str(temp_dir))

        # 6. Run multi-object tracking in batches of 25 to optimize MPS performance
        batch_size = 25
        print(f"\nTracking {len(cells_to_process)} pending cells in batches of {batch_size} on {args.device}...")
        
        # Split cells_to_process into chunks
        cell_chunks = [cells_to_process[i:i + batch_size] for i in range(0, len(cells_to_process), batch_size)]
        
        all_results = {}
        cell_info = {}
        
        for chunk_idx, chunk in enumerate(cell_chunks):
            print(f"\n--- Batch {chunk_idx+1}/{len(cell_chunks)} ({len(chunk)} cells) ---")
            predictor.reset_state(inference_state)
            
            chunk_min_t = T
            for cid in chunk:
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
                chunk_min_t = min(chunk_min_t, t_start)
                
                cell_info[cid] = {
                    "t_start": t_start,
                    "H": H,
                    "W": W
                }
                all_results[cid] = {}
                
                # Add initial mask prompt with cell ID as obj_id
                mask_tensor = torch.tensor(initial_mask, dtype=torch.bool, device=args.device)
                with torch.inference_mode():
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=t_start,
                        obj_id=cid,
                        mask=mask_tensor
                    )
            
            # Propagate this batch
            print(f"Propagating batch {chunk_idx+1} starting from frame {chunk_min_t}...")
            with torch.inference_mode():
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=chunk_min_t
                ):
                    for i, obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i, 0] > 0.0).cpu().numpy().astype(bool)
                        if mask.any():
                            all_results[obj_id][out_frame_idx] = (rle_encode(mask), int(mask.sum()))
                            
            # Clear GPU cache to prevent memory build-up
            if "mps" in args.device:
                torch.mps.empty_cache()
            elif "cuda" in args.device:
                torch.cuda.empty_cache()
                        
        # 7. Write output CSV files
        print(f"\nSaving results for {len(cell_info)} cells...")
        for cid, info in cell_info.items():
            t_start = info["t_start"]
            H, W = info["H"], info["W"]
            
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
                    pred_data = all_results[cid].get(t, None)
                    if pred_data is not None:
                        row["rle_bf"] = pred_data[0]
                        row["area_bf"] = pred_data[1]
                rows.append(row)

            # Save cell CSV to BOTH:
            # 1) SAM2 Finetuned Output Folder
            out_finetune = tracked_out_dir / f"cell_{cid}_masks.csv"
            pd.DataFrame(rows).to_csv(out_finetune, index=False)
            
            # 2) Curated Active Folder (Overwriting only the pending track!)
            out_curated = tracked_dir / f"cell_{cid}_masks.csv"
            pd.DataFrame(rows).to_csv(out_curated, index=False)

        # Cleanup state
        predictor.reset_state(inference_state)
        del predictor
        temp_dir_obj.cleanup()

    # 3. Compile population movie for fine-tuned SAM2
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

    # 4. Compile side-by-side comparison
    baseline_out = film_dir / f"{args.film}_baseline_population.mp4"
    if not baseline_out.exists():
        print(f"Generating baseline population movie...")
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

    # 5. Copy final videos to SSD
    print("Copying final videos to SSD...")
    ssd_target_sam2 = film_dir / f"{args.film}_SAM2_Finetuned_population.mp4"
    ssd_target_sxs = film_dir / f"{args.film}_SAM2_vs_Baseline_comparison.mp4"

    if sam2_out_local.exists():
        shutil.copy2(sam2_out_local, ssd_target_sam2)
        print(f"Copied SAM2 movie to: {ssd_target_sam2}")
    if sxs_out_local.exists():
        shutil.copy2(sxs_out_local, ssd_target_sxs)
        print(f"Copied Comparison movie to: {ssd_target_sxs}")

    print("\n🎉 SELECTIVE BATCH SAM2 TRACKING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
