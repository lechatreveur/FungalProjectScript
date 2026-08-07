#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_trackers.py
------------------
Benchmark script to compare SAM2 cell tracking against the custom AI tracker
and the ground truth.
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Ensure local imports work
HPC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(HPC_DIR.parent))

from Cell_tracking_functions import rle_decode, iou, compute_overlap
from sam2_tracker import track_cell_with_sam2

# Discover GT logic (derived from discover_gt.py)
BASE_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")

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
        return []
    linkage_file = exp_dir / "sequence_linkage.json"
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
            if status.lower() not in ["good", "corrected"]:
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
                            "status": status.lower()
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
                            "status": status.lower()
                        })
    # Scan individual film qc.csv
    for film_dir in exp_dir.iterdir():
        if not film_dir.is_dir() or film_dir.name.startswith("."):
            continue
        tracked_dir = film_dir / f"TrackedCells_{film_dir.name}"
        if not tracked_dir.exists():
            continue
        qc_data = check_qc_csv(tracked_dir)
        for cell_id, status in qc_data.items():
            if status in ["good", "corrected"]:
                results.append({
                    "experiment": "2026_01_08_M93",
                    "film": film_dir.name,
                    "cell_id": cell_id,
                    "status": status
                })
    return results

def discover_gt_2025_09_17():
    exp_dir = BASE_MOVIE_ROOT / "2025_09_17"
    if not exp_dir.exists():
        return []
    linkage_file = exp_dir / "sequence_linkage.json"
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
            if status.lower() not in ["good", "corrected"]:
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
                            "status": status.lower()
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
                            "status": status.lower()
                        })
    # Scan individual film qc.csv
    for film_dir in exp_dir.iterdir():
        if not film_dir.is_dir() or film_dir.name.startswith("."):
            continue
        tracked_dir = film_dir / f"TrackedCells_{film_dir.name}"
        if not tracked_dir.exists():
            continue
        qc_data = check_qc_csv(tracked_dir)
        for cell_id, status in qc_data.items():
            if status in ["good", "corrected"]:
                results.append({
                    "experiment": "2025_09_17",
                    "film": film_dir.name,
                    "cell_id": cell_id,
                    "status": status
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

import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-cells", type=int, default=10, help="Limit cells evaluated per experiment for speed")
    parser.add_argument("--device", type=str, default="mps", help="Device for SAM2 (mps, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cell selection subset")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 1. Discover all GT cell tracks
    print("Discovering ground truth cell tracks...")
    m93_gt = discover_gt_m93()
    s0917_gt = discover_gt_2025_09_17()
    
    # Deduplicate
    m93_df = pd.DataFrame(m93_gt).drop_duplicates(subset=["experiment", "film", "cell_id"])
    s0917_df = pd.DataFrame(s0917_gt).drop_duplicates(subset=["experiment", "film", "cell_id"])
    
    print(f"Discovered {len(m93_df)} tracks in M93 and {len(s0917_df)} tracks in 2025_09_17.")
    
    # Select subset of cells for comparison
    if args.limit_cells > 0:
        if len(m93_df) > args.limit_cells:
            m93_subset = m93_df.sample(n=args.limit_cells, random_state=args.seed)
        else:
            m93_subset = m93_df
            
        if len(s0917_df) > args.limit_cells:
            s0917_subset = s0917_df.sample(n=args.limit_cells, random_state=args.seed)
        else:
            s0917_subset = s0917_df
            
        eval_df = pd.concat([m93_subset, s0917_subset]).reset_index(drop=True)
    else:
        eval_df = pd.concat([m93_df, s0917_df]).reset_index(drop=True)
        
    print(f"Selected {len(eval_df)} total cells for tracking comparison.")
    
    # SAM2 Setup
    # Use relative config name for Hydra's pkg://sam2 search path
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    # Resolve the checkpoint path for SAM2 (check SSD, NAS, then local repo fallback)
    sam2_ckpt_paths = [
        "/Volumes/X10 Pro/Movies/AI/sam2_checkpoints/sam2.1_hiera_tiny.pt",
        "/X10 Pro/Movies/AI/sam2_checkpoints/sam2.1_hiera_tiny.pt",
        "/Volumes/Movies/AI/sam2_checkpoints/sam2.1_hiera_tiny.pt",
        str(HPC_DIR / "../segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
    ]
    sam2_ckpt = None
    for path in sam2_ckpt_paths:
        if os.path.exists(path):
            sam2_ckpt = path
            break
    if sam2_ckpt is None:
        raise FileNotFoundError("Could not find sam2.1_hiera_tiny.pt checkpoint!")
    print(f"Using SAM2 checkpoint: {sam2_ckpt}")
    
    # Custom AI Tracker Setup
    custom_ckpt = "/Volumes/X10 Pro/Movies/AI/tracker_checkpoints/model_latest.pt"
    
    # Load custom tracker model
    from tracker_model import load_tracker
    from ai_tracking_inference import ai_track_one_direction
    
    print(f"Loading custom AI tracker model from {custom_ckpt}...")
    custom_model = load_tracker(custom_ckpt, device=args.device)
    custom_model.eval()
    
    comparison_results = []
    
    for idx, row in eval_df.iterrows():
        exp = row["experiment"]
        film = row["film"]
        cell_id = row["cell_id"]
        status = row["status"]
        
        print(f"\n[{idx+1}/{len(eval_df)}] Evaluating Cell {cell_id} in film {film} ({exp})...")
        
        film_dir = BASE_MOVIE_ROOT / exp / film
        frames_dir = film_dir / f"Frames_{film}"
        masks_dir = film_dir / f"Masks_{film}"
        tracked_dir = film_dir / f"TrackedCells_{film}"
        csv_path = tracked_dir / f"cell_{cell_id}_masks.csv"
        
        if not csv_path.exists():
            print(f"  [Skip] cell masks CSV does not exist: {csv_path}")
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
                
        # Find start time
        valid_ts = [t for t, m in gt_masks.items() if m.any()]
        if not valid_ts:
            print(f"  [Skip] No valid non-empty masks found in GT CSV.")
            continue
            
        t_start = min(valid_ts)
        t_end = max(valid_ts)
        initial_mask = gt_masks[t_start]
        
        # Discover frame paths
        channel_idx = detect_channel(masks_dir)
        frame_files = sorted([
            f for f in frames_dir.iterdir()
            if f.name.lower().endswith(".tif")
            and not f.name.endswith("_seg.tif")
            and f"_c_{channel_idx}.tif" in f.name
            and not f.name.startswith(".")
        ])
        
        T = len(frame_files)
        if T == 0:
            print(f"  [Skip] No frames found in {frames_dir}")
            continue
            
        print(f"  Tracking from t={t_start} to t={T-1} (Total frames: {T}, target track length: {T - t_start})")
        
        # 2a. Run SAM2 Tracker
        t0 = time.time()
        try:
            # We track forward starting from t_start
            sam2_res = track_cell_with_sam2(
                frame_paths=frame_files,
                initial_mask=initial_mask,
                checkpoint_path=sam2_ckpt,
                model_cfg=sam2_cfg,
                device=args.device,
                start_frame_idx=t_start
            )
            sam2_dur = time.time() - t0
            sam2_failed = False
        except Exception as e:
            print(f"  [SAM2 Error] {e}")
            sam2_res = {}
            sam2_dur = 0.0
            sam2_failed = True
            
        # 2b. Run Custom AI Tracker
        t0 = time.time()
        try:
            def get_seg_path_func(t):
                cand_tif = masks_dir / f"{film}_t_{t:03d}_c_{channel_idx}_seg.tif"
                if cand_tif.exists():
                    return str(cand_tif)
                return str(masks_dir / f"{film}_t_{t:03d}_c_{channel_idx}_seg.npy")
                
            def get_frame_path_func(t):
                return str(frames_dir / f"{film}_t_{t:03d}_c_{channel_idx}.tif")
                
            t_seq = list(range(t_start, T))
            custom_res_raw = ai_track_one_direction(
                t_seq=t_seq,
                ref_start_mask=initial_mask,
                bf_frame_path_func=get_frame_path_func,
                lab_seg_path_func=get_seg_path_func,
                model=custom_model,
                device=args.device,
                use_probabilistic=True,
                w_keep_prior=0.35
            )
            
            custom_res = {}
            for t, val in custom_res_raw.items():
                custom_res[t] = val["mask"].astype(bool)
                
            custom_dur = time.time() - t0
            custom_failed = False
        except Exception as e:
            print(f"  [Custom AI Error] {e}")
            custom_res = {}
            custom_dur = 0.0
            custom_failed = True
            
        # 3. Compute frame-by-frame IoUs
        sam2_ious = []
        custom_ious = []
        
        # We evaluate on all frames from t_start to the end of the GT track or video
        for t in range(t_start, T):
            gt_mask = gt_masks.get(t, np.zeros((H, W), dtype=bool))
            
            # If the cell touches the border or goes missing in the GT, the track ends
            # But let's evaluate on all frames where GT mask is non-empty
            if not gt_mask.any():
                continue
                
            # SAM2 IoU
            s_mask = sam2_res.get(t, np.zeros((H, W), dtype=bool))
            sam2_ious.append(iou(gt_mask, s_mask))
            
            # Custom AI IoU
            c_mask = custom_res.get(t, np.zeros((H, W), dtype=bool))
            custom_ious.append(iou(gt_mask, c_mask))
            
        # 4. Aggregate Metrics
        if sam2_ious:
            sam2_mean = np.mean(sam2_ious)
            sam2_survival = np.mean([1.0 if v >= 0.5 else 0.0 for v in sam2_ious])
            sam2_final = sam2_ious[-1]
        else:
            sam2_mean, sam2_survival, sam2_final = 0.0, 0.0, 0.0
            
        if custom_ious:
            custom_mean = np.mean(custom_ious)
            custom_survival = np.mean([1.0 if v >= 0.5 else 0.0 for v in custom_ious])
            custom_final = custom_ious[-1]
        else:
            custom_mean, custom_survival, custom_final = 0.0, 0.0, 0.0
            
        fps_sam2 = len(sam2_ious) / sam2_dur if sam2_dur > 0 else 0.0
        fps_custom = len(custom_ious) / custom_dur if custom_dur > 0 else 0.0
        
        print(f"  SAM2:      Mean IoU = {sam2_mean:.3f}, Survival = {sam2_survival*100:.1f}%, Final IoU = {sam2_final:.3f}, Duration = {sam2_dur:.2f}s ({fps_sam2:.1f} fps)")
        print(f"  Custom AI: Mean IoU = {custom_mean:.3f}, Survival = {custom_survival*100:.1f}%, Final IoU = {custom_final:.3f}, Duration = {custom_dur:.2f}s ({fps_custom:.1f} fps)")
        
        comparison_results.append({
            "experiment": exp,
            "film": film,
            "cell_id": cell_id,
            "qc_status": status,
            "track_length": len(sam2_ious),
            "sam2_mean_iou": sam2_mean,
            "sam2_survival": sam2_survival,
            "sam2_final_iou": sam2_final,
            "sam2_duration": sam2_dur,
            "sam2_failed": sam2_failed,
            "custom_mean_iou": custom_mean,
            "custom_survival": custom_survival,
            "custom_final_iou": custom_final,
            "custom_duration": custom_dur,
            "custom_failed": custom_failed
        })
        
    # Save results to CSV
    results_df = pd.DataFrame(comparison_results)
    results_csv = HPC_DIR / "tracker_comparison_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved detailed comparison metrics to {results_csv}")
    
    # 5. Output beautiful markdown summary
    summary_md_path = HPC_DIR / "tracker_comparison_summary.md"
    
    sam2_avg_iou = results_df["sam2_mean_iou"].mean()
    sam2_avg_survival = results_df["sam2_survival"].mean()
    sam2_avg_final = results_df["sam2_final_iou"].mean()
    sam2_avg_fps = (results_df["track_length"] / results_df["sam2_duration"].replace(0, np.nan)).mean()
    
    custom_avg_iou = results_df["custom_mean_iou"].mean()
    custom_avg_survival = results_df["custom_survival"].mean()
    custom_avg_final = results_df["custom_final_iou"].mean()
    custom_avg_fps = (results_df["track_length"] / results_df["custom_duration"].replace(0, np.nan)).mean()
    
    summary_md = f"""# Cell Tracker Benchmark Summary: SAM2 vs. Custom AI Tracker

Comparison evaluated on {len(results_df)} cell tracks from M93 and 2025_09_17 datasets.

## Aggregate Performance Metrics

| Metric | Meta SAM2 (Tiny) | Custom AI Tracker (ResNet/HMM) | Improvement / Difference |
| :--- | :--- | :--- | :--- |
| **Average Mean IoU** | {sam2_avg_iou:.3f} | {custom_avg_iou:.3f} | {sam2_avg_iou - custom_avg_iou:+.3f} |
| **Average Survival Rate** (IoU &ge; 0.5) | {sam2_avg_survival*100:.1f}% | {custom_avg_survival*100:.1f}% | {sam2_avg_survival - custom_avg_survival:+.1%} |
| **Average Final Frame IoU** | {sam2_avg_final:.3f} | {custom_avg_final:.3f} | {sam2_avg_final - custom_avg_final:+.3f} |
| **Propagation Speed** (FPS) | {sam2_avg_fps:.1f} | {custom_avg_fps:.1f} | {sam2_avg_fps - custom_avg_fps:+.1f} |

## Performance by Experiment Group

| Experiment | Tracker | Mean IoU | Survival Rate | Final IoU |
| :--- | :--- | :--- | :--- | :--- |
"""
    
    for exp_name, group in results_df.groupby("experiment"):
        s_iou = group["sam2_mean_iou"].mean()
        s_surv = group["sam2_survival"].mean()
        s_fin = group["sam2_final_iou"].mean()
        
        c_iou = group["custom_mean_iou"].mean()
        c_surv = group["custom_survival"].mean()
        c_fin = group["custom_final_iou"].mean()
        
        summary_md += f"| {exp_name} | Meta SAM2 | {s_iou:.3f} | {s_surv*100:.1f}% | {s_fin:.3f} |\n"
        summary_md += f"| {exp_name} | Custom AI | {c_iou:.3f} | {c_surv*100:.1f}% | {c_fin:.3f} |\n"
        
    summary_md += "\n## Individual Cell Tracking Detailed Breakdown\n\n"
    summary_md += "| Film | Cell ID | Status | Length (Frames) | SAM2 Mean IoU | Custom AI Mean IoU | SAM2 Final IoU | Custom AI Final IoU |\n"
    summary_md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, r in results_df.iterrows():
        summary_md += f"| {r['film']} | {r['cell_id']} | {r['qc_status']} | {r['track_length']} | {r['sam2_mean_iou']:.3f} | {r['custom_mean_iou']:.3f} | {r['sam2_final_iou']:.3f} | {r['custom_final_iou']:.3f} |\n"
        
    with open(summary_md_path, 'w') as f:
        f.write(summary_md)
        
    print(f"Saved summary report to {summary_md_path}")
    print("\nBenchmark Summary Statistics:")
    print(f"  SAM2 average mean IoU: {sam2_avg_iou:.3f}")
    print(f"  Custom AI average mean IoU: {custom_avg_iou:.3f}")
    
if __name__ == "__main__":
    main()
