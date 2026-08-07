#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_film.py
---------------
Benchmark script to compare SAM2 cell tracking against the custom AI tracker
on all or a subset of verified cells for a SPECIFIC film.
"""

import os
import sys
import time
import json
import random
import argparse
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Ensure local imports work
HPC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(HPC_DIR.parent))

from Cell_tracking_functions import rle_decode, iou
from sam2_tracker import track_cell_with_sam2

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

def discover_gt_for_film(exp_name, film_name):
    exp_dir = BASE_MOVIE_ROOT / exp_name
    if not exp_dir.exists():
        return []
        
    linkage_file = exp_dir / "sequence_linkage.json"
    linkage = {}
    if linkage_file.exists():
        with open(linkage_file, 'r') as f:
            linkage = json.load(f)
            
    results = []
    
    # 1. Scan the sequence_linkage JSON QCs
    seqs = ["F0", "F1"] if exp_name == "2025_09_17" else ["A14_F0", "A14_F1", "A14_F2"]
    for seq in seqs:
        qc_file = exp_dir / seq / f"qc_{seq}.json"
        if not qc_file.exists():
            continue
        with open(qc_file, 'r') as f:
            qc = json.load(f)
            
        films = linkage.get(seq, {}).get("films", [])
        global_cells = linkage.get(seq, {}).get("global_cells", {})
        
        # Check if the target film is in this sequence group
        if film_name not in films:
            continue
            
        film_idx = films.index(film_name)
        
        for key, status in qc.items():
            if status.lower() not in ["good", "corrected"]:
                continue
                
            prefix = f"{seq}_"
            suffix = key[len(prefix):]
            if suffix.startswith("cell_"):
                # Global cell
                local_ids = global_cells.get(key, [-1]*len(films))
                if local_ids[film_idx] != -1:
                    results.append({
                        "experiment": exp_name,
                        "film": film_name,
                        "cell_id": local_ids[film_idx],
                        "status": status.lower(),
                        "source": f"qc_{seq}.json (global)"
                    })
            else:
                # Local cell
                parts = suffix.split("_cell_")
                if len(parts) == 2:
                    film_part = parts[0]
                    local_id = int(parts[1])
                    if film_part == film_name:
                        results.append({
                            "experiment": exp_name,
                            "film": film_name,
                            "cell_id": local_id,
                            "status": status.lower(),
                            "source": f"qc_{seq}.json (local)"
                        })
                        
    # 2. Scan individual film qc.csv files
    film_dir = exp_dir / film_name
    tracked_dir = film_dir / f"TrackedCells_{film_name}"
    if tracked_dir.exists():
        qc_data = check_qc_csv(tracked_dir)
        for cell_id, status in qc_data.items():
            if status in ["good", "corrected"]:
                results.append({
                    "experiment": exp_name,
                    "film": film_name,
                    "cell_id": cell_id,
                    "status": status,
                    "source": "qc.csv"
                })
                
    # Deduplicate
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.drop_duplicates(subset=["cell_id"])
        return df.to_dict(orient="records")
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--film", type=str, default="A14_1TP2_BF_F1", help="Film name to benchmark")
    parser.add_argument("--limit-cells", type=int, default=10, help="Max cells to evaluate for time reasons (0 for all)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Auto-detect experiment folder
    exp_name = None
    for exp_dir in BASE_MOVIE_ROOT.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        if (exp_dir / args.film).exists():
            exp_name = exp_dir.name
            break
            
    if exp_name is None:
        raise FileNotFoundError(f"Could not find film {args.film} in {BASE_MOVIE_ROOT} subdirectories.")
        
    print(f"Detected experiment: {exp_name} for film {args.film}")
    
    # Discover verified cell tracks for this film
    verified_cells = discover_gt_for_film(exp_name, args.film)
    print(f"Discovered {len(verified_cells)} verified cell tracks in film {args.film}.")
    
    if not verified_cells:
        print("No verified cells to compare. Exiting.")
        return
        
    # Limit cells if requested
    if args.limit_cells > 0 and len(verified_cells) > args.limit_cells:
        verified_cells = random.sample(verified_cells, args.limit_cells)
        print(f"Limited evaluation to {len(verified_cells)} randomly sampled cells (seed={args.seed}).")
        
    # SAM2 Setup
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_ckpt = "/Volumes/X10 Pro/Movies/AI/sam2_checkpoints/sam2.1_hiera_tiny.pt"
    
    # Custom AI Tracker Setup
    custom_ckpt = "/Volumes/X10 Pro/Movies/AI/tracker_checkpoints/model_latest.pt"
    from tracker_model import load_tracker
    from ai_tracking_inference import ai_track_one_direction
    
    print(f"Loading custom AI tracker model from {custom_ckpt}...")
    custom_model = load_tracker(custom_ckpt, device=args.device)
    custom_model.eval()
    
    comparison_results = []
    
    for idx, cell_info in enumerate(verified_cells):
        cell_id = cell_info["cell_id"]
        status = cell_info["status"]
        source = cell_info["source"]
        
        print(f"\n[{idx+1}/{len(verified_cells)}] Evaluating Cell {cell_id} (status: {status}, source: {source})...")
        
        film_dir = BASE_MOVIE_ROOT / exp_name / args.film
        frames_dir = film_dir / f"Frames_{args.film}"
        masks_dir = film_dir / f"Masks_{args.film}"
        tracked_dir = film_dir / f"TrackedCells_{args.film}"
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
        # Detect channel index
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
            print(f"  [Skip] No frames found in {frames_dir}")
            continue
            
        print(f"  Tracking from t={t_start} to t={T-1} (Total frames: {T}, target track length: {T - t_start})")
        
        # 1. Run SAM2
        t0 = time.time()
        try:
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
            
        # 2. Run Custom AI
        t0 = time.time()
        try:
            def get_seg_path_func(t):
                cand_tif = masks_dir / f"{args.film}_t_{t:03d}_c_{channel_idx}_seg.tif"
                if cand_tif.exists():
                    return str(cand_tif)
                return str(masks_dir / f"{args.film}_t_{t:03d}_c_{channel_idx}_seg.npy")
                
            def get_frame_path_func(t):
                return str(frames_dir / f"{args.film}_t_{t:03d}_c_{channel_idx}.tif")
                
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
            
        # 3. Evaluate
        sam2_ious = []
        custom_ious = []
        
        for t in range(t_start, T):
            gt_mask = gt_masks.get(t, np.zeros((H, W), dtype=bool))
            if not gt_mask.any():
                continue
                
            s_mask = sam2_res.get(t, np.zeros((H, W), dtype=bool))
            sam2_ious.append(iou(gt_mask, s_mask))
            
            c_mask = custom_res.get(t, np.zeros((H, W), dtype=bool))
            custom_ious.append(iou(gt_mask, c_mask))
            
        # Stats
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
        
    results_df = pd.DataFrame(comparison_results)
    results_csv = HPC_DIR / f"{args.film}_comparison_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved detailed comparison metrics to {results_csv}")
    
    # Save a nice markdown report in the artifacts folder
    summary_md_path = Path("/Users/user/.gemini/antigravity-ide/brain/4dcb4bf4-544f-4647-96fd-e18a94325d83") / f"{args.film}_benchmark_report.md"
    
    sam2_avg_iou = results_df["sam2_mean_iou"].mean()
    sam2_avg_survival = results_df["sam2_survival"].mean()
    sam2_avg_final = results_df["sam2_final_iou"].mean()
    sam2_avg_fps = (results_df["track_length"] / results_df["sam2_duration"].replace(0, np.nan)).mean()
    
    custom_avg_iou = results_df["custom_mean_iou"].mean()
    custom_avg_survival = results_df["custom_survival"].mean()
    custom_avg_final = results_df["custom_final_iou"].mean()
    custom_avg_fps = (results_df["track_length"] / results_df["custom_duration"].replace(0, np.nan)).mean()
    
    summary_md = f"""# Tracking Performance Report: Film {args.film}

Benchmarked on {len(results_df)} verified cell tracks from {args.film} in experiment {exp_name}.

## Aggregate Metrics for Film {args.film}

| Metric | Meta SAM2 (Tiny) | Custom AI Tracker | Improvement |
| :--- | :---: | :---: | :---: |
| **Average Mean IoU** | {sam2_avg_iou:.3f} | {custom_avg_iou:.3f} | {sam2_avg_iou - custom_avg_iou:+.3f} |
| **Average Survival Rate** | {sam2_avg_survival*100:.1f}% | {custom_avg_survival*100:.1f}% | {sam2_avg_survival - custom_avg_survival:+.1%} |
| **Average Final Frame IoU** | {sam2_avg_final:.3f} | {custom_avg_final:.3f} | {sam2_avg_final - custom_avg_final:+.3f} |
| **Propagation Speed** (FPS) | {sam2_avg_fps:.1f} | {custom_avg_fps:.1f} | {sam2_avg_fps - custom_avg_fps:+.1f} |

## Cell-by-Cell Breakdown

| Cell ID | Status | Length (Frames) | SAM2 Mean IoU | Custom Mean IoU | SAM2 Final IoU | Custom Final IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in results_df.iterrows():
        summary_md += f"| {int(r['cell_id'])} | {r['qc_status']} | {int(r['track_length'])} | {r['sam2_mean_iou']:.3f} | {r['custom_mean_iou']:.3f} | {r['sam2_final_iou']:.3f} | {r['custom_final_iou']:.3f} |\n"
        
    with open(summary_md_path, 'w') as f:
        f.write(summary_md)
        
    print(f"Saved benchmark summary report to {summary_md_path}")

if __name__ == "__main__":
    main()
