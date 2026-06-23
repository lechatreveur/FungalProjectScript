#!/usr/bin/env python3
import os
import sys
import subprocess
import re
from pathlib import Path
import cv2
import numpy as np

os.environ["RUN_IN_PROCESS"] = "TRUE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_cmd(cmd, cwd=None):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(f"[Error] Command failed with return code {res.returncode}")
        print(res.stdout)
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout

def make_side_by_side(v1_path, v2_path, out_path, label1, label2):
    print(f"Creating side-by-side: {v1_path} vs {v2_path} -> {out_path}")
    cap1 = cv2.VideoCapture(v1_path)
    cap2 = cv2.VideoCapture(v2_path)
    
    if not cap1.isOpened():
        raise RuntimeError(f"Could not open {v1_path}")
    if not cap2.isOpened():
        raise RuntimeError(f"Could not open {v2_path}")
        
    fps = cap1.get(cv2.CAP_PROP_FPS) or 10.0
    target_h, target_w = 800, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (target_w * 2, target_h))
    
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

def track_cell_subprocess(args_tuple):
    cid, movie_root, exp, film = args_tuple
    hpc_dir = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC"
    py_exec = "python3"
    cmd = f"KMP_DUPLICATE_LIB_OK=TRUE {py_exec} one_cell_quantification_1CH.py --cell_id {cid} --experiment_path \"{os.path.join(movie_root, exp)}\" --file_name \"{film}\" --use_ai_tracker --no_plot"
    res = subprocess.run(cmd, shell=True, cwd=hpc_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return cid, False, res.stdout + "\n" + res.stderr
    return cid, True, ""

def process_movie(exp, film):
    print(f"\n==================================================")
    print(f"🎬 PROCESSING EXPERIMENT: {exp} | FILM: {film}")
    print(f"==================================================")
    
    movie_root = "/Volumes/X10 Pro/Movies"
    film_dir = os.path.join(movie_root, exp, film)
    tracked_dir = os.path.join(film_dir, f"TrackedCells_{film}")
    
    # 1. Discover all cell IDs from tracked_dir
    if not os.path.exists(tracked_dir):
        raise RuntimeError(f"Curation folder not found: {tracked_dir}")
        
    cell_ids = []
    for f in os.listdir(tracked_dir):
        m = re.match(r"^cell_(\d+)_masks\.csv$", f)
        if m:
            cell_ids.append(int(m.group(1)))
    cell_ids = sorted(cell_ids)
    print(f"Found {len(cell_ids)} cells: {cell_ids}")
    
    # Ensure AI tracked cells folder exists (do not delete to support resuming aborted tracking runs)
    ai_tracked_dir = os.path.join(film_dir, f"TrackedCells_{film}_AI")
    os.makedirs(ai_tracked_dir, exist_ok=True)

    # 2. Run AI tracking in parallel using multiprocessing
    import multiprocessing
    # Reduced num_workers to 4 to balance memory usage (safe on 16GB Mac) and tracking speed
    num_workers = 4
    print(f"Tracking {len(cell_ids)} cells in parallel using {num_workers} workers...")
    
    args_list = [(cid, movie_root, exp, film) for cid in cell_ids]
    hpc_dir = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC"
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = []
        for res in pool.imap_unordered(track_cell_subprocess, args_list):
            cid, success, logs = res
            results.append(res)
            if not success:
                print(f"[Error] Cell {cid} tracking failed:\n{logs}")
                raise RuntimeError(f"Cell {cid} failed")
            else:
                if len(results) % 20 == 0 or len(results) == len(cell_ids):
                    print(f"Progress: Completed {len(results)}/{len(cell_ids)} cells")
        
    # 3. Generate AI population movie
    py_exec = "/Users/user/miniforge3/envs/cellpose-sam/bin/python"
    hpc_dir = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC"
    ai_out_local = os.path.join(hpc_dir, f"{film}_AI_population.mp4")
    cmd_ai = f"KMP_DUPLICATE_LIB_OK=TRUE {py_exec} ../make_population_movie.py --base_dir \"{film_dir}\" --frames_dir \"Frames_{film}\" --cells_dir \"TrackedCells_{film}_AI\" --out \"{ai_out_local}\" --fps 10 --alpha 0.4"
    run_cmd(cmd_ai, cwd=hpc_dir)
    
    # 4. Generate Baseline population movie
    base_out_local = os.path.join(hpc_dir, f"{film}_baseline_population.mp4")
    cmd_base = f"KMP_DUPLICATE_LIB_OK=TRUE {py_exec} ../make_population_movie.py --base_dir \"{film_dir}\" --frames_dir \"Frames_{film}\" --cells_dir \"TrackedCells_{film}\" --out \"{base_out_local}\" --fps 10 --alpha 0.4"
    run_cmd(cmd_base, cwd=hpc_dir)
    
    # 5. Generate side-by-side movie
    sxs_out_local = os.path.join(hpc_dir, f"{film}_side_by_side_comparison.mp4")
    make_side_by_side(
        v1_path=base_out_local,
        v2_path=ai_out_local,
        out_path=sxs_out_local,
        label1=f"Old Baseline (Curated GT - {exp})",
        label2="New Tracker (Geometric AI Model)"
    )
    
    # 6. Copy final movies back to SSD
    ssd_target_ai = os.path.join(film_dir, f"{film}_AI_population.mp4")
    ssd_target_sxs = os.path.join(film_dir, f"{film}_side_by_side_comparison.mp4")
    print(f"Copying final videos back to SSD...")
    import shutil
    shutil.copy2(ai_out_local, ssd_target_ai)
    shutil.copy2(sxs_out_local, ssd_target_sxs)
    
    # Also copy to SingleCellQuantificationHPC for easy workspace access
    print(f"Saved on SSD: \n  - {ssd_target_ai}\n  - {ssd_target_sxs}")

def main():
    movies = [
        ("2026_04_30_M135", "A14_BF1_F0"),
        ("2026_04_30_M135", "A14_BF1_F1"),
        ("2026_04_30_M135", "A14_BF1_F2"),
        ("2026_04_30_M135", "A14_BF2_F0"),
        ("2026_04_30_M135", "A14_BF2_F1"),
        ("2026_04_30_M135", "A14_BF2_F2"),
        ("2026_04_30_M135", "A14_BF3_F0"),
        ("2026_04_30_M135", "A14_BF3_F1"),
        ("2026_04_30_M135", "A14_BF3_F2"),
    ]
    
    for exp, film in movies:
        # Check if side-by-side comparison video already exists on SSD
        movie_root = "/Volumes/X10 Pro/Movies"
        ssd_target_sxs = os.path.join(movie_root, exp, film, f"{film}_side_by_side_comparison.mp4")
        if os.path.exists(ssd_target_sxs):
            print(f"Skipping {film} because side-by-side comparison already exists on SSD.")
            continue
        process_movie(exp, film)
    print("\n🎉 ALL BATCH PROCESSING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
