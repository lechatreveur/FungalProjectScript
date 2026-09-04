import os
import sys
import random
import numpy as np
import pandas as pd
import tifffile
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

# Set project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from Cell_tracking_functions import rle_decode, touches_border

HPC_BASE = '/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156'
LOCAL_BASE = '/Volumes/X10 Pro/Movies/2026_07_16_M156'
MOVIE_ROOT = HPC_BASE if os.path.exists(HPC_BASE) else LOCAL_BASE

BASE_DIR = os.path.join(MOVIE_ROOT, '3_FL2_F0')
LEGACY_TRACKED_DIR = os.path.join(BASE_DIR, 'TrackedCells_3_FL2_F0')
NEW_TRACKED_DIR = os.path.join(BASE_DIR, 'TrackedCells_3_FL2_F0_cpsam_overexp')
FRAMES_DIR = os.path.join(BASE_DIR, 'Frames_3_FL2_F0')

TEST_OUTPUT_DIR = os.path.join(MOVIE_ROOT, 'cellpose_overexposure_test')
MOVIES_OUTPUT_DIR = os.path.join(TEST_OUTPUT_DIR, 'single_cell_movies')
os.makedirs(NEW_TRACKED_DIR, exist_ok=True)
os.makedirs(MOVIES_OUTPUT_DIR, exist_ok=True)

ORIGINAL_10_CELLS = [25, 116, 9, 18, 15, 11, 19, 140, 178, 34]

def calculate_jumps(areas, centroids):
    jumps = [False] * len(areas)
    for t in range(len(areas) - 1):
        a1, a2 = areas[t], areas[t+1]
        c1, c2 = centroids[t], centroids[t+1]
        if a1 == 0 or a2 == 0 or np.isnan(c1[0]) or np.isnan(c2[0]):
            jumps[t] = True
            continue
        area_ratio = max(a1, a2) / min(a1, a2)
        dist = np.hypot(c2[0] - c1[0], c2[1] - c1[1])
        if area_ratio > 1.5 or dist > 15.0:
            jumps[t] = True
    return jumps

def evaluate_cell_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
        
    rle_col = 'rle_gfp' if 'rle_gfp' in df.columns else 'rle_bf'
    areas = []
    centroids = []
    failures = []
    
    for idx, row in df.iterrows():
        rle = row.get(rle_col, '')
        mask = rle_decode(rle, (int(row['height']), int(row['width'])))
        area = int(mask.sum())
        areas.append(area)
        if area > 0:
            ys, xs = np.where(mask)
            centroids.append((np.mean(ys), np.mean(xs)))
            failures.append(False)
        else:
            centroids.append((np.nan, np.nan))
            failures.append(True)
            
    jumps = calculate_jumps(areas, centroids)
    return {
        'jumps': sum(jumps[:-1]),
        'failures': sum(failures),
        'areas': areas,
        'centroids': centroids,
        'df': df
    }

def generate_single_cell_movie(cell_id):
    """
    Renders a 2-panel side-by-side MP4 video for a cell:
    Left: Existing Pipeline mask contour on raw crop
    Right: CellposeSAM-OVEREXPOSED mask contour on raw crop
    """
    legacy_csv = os.path.join(LEGACY_TRACKED_DIR, f'cell_{cell_id}_masks.csv')
    new_csv = os.path.join(NEW_TRACKED_DIR, f'cell_{cell_id}_masks.csv')
    
    if not os.path.exists(legacy_csv) or not os.path.exists(new_csv):
        print(f"Skipping movie generation for cell {cell_id}: missing CSVs", flush=True)
        return
        
    leg_df = pd.read_csv(legacy_csv)
    new_df = pd.read_csv(new_csv)
    
    # Calculate crop box from cell centroids
    all_cy, all_cx = [], []
    for df in [leg_df, new_df]:
        rle_col = 'rle_gfp' if 'rle_gfp' in df.columns else 'rle_bf'
        for _, row in df.iterrows():
            m = rle_decode(row[rle_col], (int(row['height']), int(row['width'])))
            if m.sum() > 0:
                ys, xs = np.where(m)
                all_cy.append(np.mean(ys))
                all_cx.append(np.mean(xs))
                
    if len(all_cy) == 0:
        return
        
    med_y, med_x = np.median(all_cy), np.median(all_cx)
    cy = int(np.clip(round(med_y), 100, 1900))
    cx = int(np.clip(round(med_x), 100, 1900))
    r0, r1 = cy - 100, cy + 100
    c0, c1 = cx - 100, cx + 100
    
    out_mp4 = os.path.join(MOVIES_OUTPUT_DIR, f'cell_{cell_id}_comparison.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_mp4, fourcc, 7, (400, 240))
    
    for t in range(min(len(leg_df), len(new_df))):
        frame_p = os.path.join(FRAMES_DIR, f'3_FL2_F0_t_{t:03d}_c_0.tif')
        if not os.path.exists(frame_p):
            continue
            
        img_raw = tifffile.imread(frame_p)
        crop = img_raw[r0:r1, c0:c1]
        
        p1, p99 = np.percentile(crop, (0.5, 99.5))
        crop_norm = np.clip((crop - p1) / max(1e-5, p99 - p1), 0, 1)
        crop_u8 = (crop_norm * 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2BGR)
        
        # Panel 1: Existing Pipeline (Blue Contour)
        panel1 = crop_bgr.copy()
        leg_rle = leg_df.iloc[t]['rle_gfp'] if 'rle_gfp' in leg_df.columns else leg_df.iloc[t]['rle_bf']
        leg_m = rle_decode(leg_rle, (2000, 2000))[r0:r1, c0:c1]
        if leg_m.any():
            contours, _ = cv2.findContours(leg_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel1, contours, -1, (255, 130, 49), 2)  # Blue/Cyan
        cv2.putText(panel1, f"Existing (t={t:03d})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Panel 2: CellposeSAM-OVEREXPOSED (Green Contour)
        panel2 = crop_bgr.copy()
        new_rle = new_df.iloc[t]['rle_gfp'] if 'rle_gfp' in new_df.columns else new_df.iloc[t]['rle_bf']
        new_m = rle_decode(new_rle, (2000, 2000))[r0:r1, c0:c1]
        if new_m.any():
            contours, _ = cv2.findContours(new_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel2, contours, -1, (44, 160, 44), 2)  # Green
        cv2.putText(panel2, f"CPSAM-Overexp (t={t:03d})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Combine side-by-side
        combined_frame = np.zeros((240, 400, 3), dtype=np.uint8)
        combined_frame[20:220, 0:200] = cv2.resize(panel1, (200, 200))
        combined_frame[20:220, 200:400] = cv2.resize(panel2, (200, 200))
        
        # Add Header Title
        cv2.putText(combined_frame, f"Cell {cell_id} Tracking Comparison", (60, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(combined_frame)
        
    writer.release()
    print(f"Generated single-cell comparison movie for Cell {cell_id}: {out_mp4}", flush=True)

def main():
    print("================ PROMPT 1 HPC VALIDATION & MOVIE RUN ================", flush=True)
    
    # 1. Discover available cell IDs in film 3_FL2_F0
    legacy_csv_files = [f for f in os.listdir(LEGACY_TRACKED_DIR) if f.startswith('cell_') and f.endswith('_masks.csv')]
    all_available_cell_ids = []
    for f in legacy_csv_files:
        try:
            cid = int(f.split('_')[1])
            all_available_cell_ids.append(cid)
        except Exception:
            pass
            
    print(f"Total cells found in legacy tracked dir: {len(all_available_cell_ids)}", flush=True)
    
    # Select 20 random cell IDs excluding ORIGINAL_10_CELLS
    random.seed(42)
    candidate_random = [cid for cid in all_available_cell_ids if cid not in ORIGINAL_10_CELLS]
    random_20_cells = sorted(random.sample(candidate_random, min(20, len(candidate_random))))
    print(f"Selected 20 random non-worst-case cell IDs: {random_20_cells}", flush=True)
    
    all_test_cells = sorted(list(set(ORIGINAL_10_CELLS + random_20_cells)))
    
    # 2. Execute production script one_cell_quantification_1CH.py --seg-backend cellpose_overexposed for all 30 cells in parallel via SLURM
    cell_ids_file = os.path.join(TEST_OUTPUT_DIR, "prompt1_val_cell_ids.txt")
    with open(cell_ids_file, "w") as f:
        for cid in all_test_cells:
            f.write(f"{cid}\n")
            
    sb_work_dir = os.path.join(SCRIPT_DIR, "2026_07_16_M156", "3_FL2_F0", "sb_scripts_val")
    gen_script = os.path.join(SCRIPT_DIR, "generate_cell_jobs.py")
    one_cell_script = os.path.join(SCRIPT_DIR, "one_cell_quantification_1CH.py")
    job_id_file = os.path.join(TEST_OUTPUT_DIR, "slurm_val_job_ids.tsv")
    
    if os.path.exists(job_id_file):
        os.remove(job_id_file)
        
    cmd = [
        sys.executable, "-u", gen_script,
        "-w", sb_work_dir,
        "-s", one_cell_script,
        "-i", cell_ids_file,
        "-e", MOVIE_ROOT,
        "-f", "3_FL2_F0",
        "-c", "gfp",
        "-n", "15",
        "-d", "5",
        "-z", "0",
        "-a", "2000",
        "--direction", "forward",
        "--seg-backend", "cellpose_overexposed",
        "--job-name-prefix", "VAL_M156_",
        "--job-id-file", job_id_file,
        "--submit", "slurm"
    ]
    
    print(f"Submitting {len(all_test_cells)} cell quantification SLURM jobs...", flush=True)
    import subprocess, time
    subprocess.run(cmd, check=True)
    
    print("Waiting for all parallel SLURM jobs to complete...", flush=True)
    expected_csvs = [os.path.join(NEW_TRACKED_DIR, f"cell_{cid}_masks.csv") for cid in all_test_cells]
    
    start_time = time.time()
    while True:
        completed = sum(1 for p in expected_csvs if os.path.exists(p) and os.path.getsize(p) > 0)
        print(f"[{int(time.time() - start_time)}s] Progress: {completed}/{len(all_test_cells)} cells completed.", flush=True)
        if completed >= len(all_test_cells) or (time.time() - start_time) > 3600:
            break
        time.sleep(15)

    # 3. Evaluate Validation Metrics
    print("\n================ VALIDATION RESULTS ================", flush=True)
    stats_list = []
    
    for cid in all_test_cells:
        leg_res = evaluate_cell_csv(os.path.join(LEGACY_TRACKED_DIR, f'cell_{cid}_masks.csv'))
        new_res = evaluate_cell_csv(os.path.join(NEW_TRACKED_DIR, f'cell_{cid}_masks.csv'))
        
        if leg_res is not None and new_res is not None:
            group = "Original 10 Worst-Case" if cid in ORIGINAL_10_CELLS else "Random Sample (20 Typical)"
            stats_list.append({
                'cell_id': cid,
                'group': group,
                'existing_jumps': leg_res['jumps'],
                'existing_failures': leg_res['failures'],
                'new_cpsam_overexp_jumps': new_res['jumps'],
                'new_cpsam_overexp_failures': new_res['failures']
            })
            
    val_df = pd.DataFrame(stats_list)
    val_df_path = os.path.join(TEST_OUTPUT_DIR, 'prompt1_validation_summary.csv')
    val_df.to_csv(val_df_path, index=False)
    
    print("\n--- Summary by Group ---", flush=True)
    grp_df = val_df.groupby('group')[['existing_jumps', 'existing_failures', 'new_cpsam_overexp_jumps', 'new_cpsam_overexp_failures']].mean().round(2)
    print(grp_df.to_string(), flush=True)
    
    # 4. Generate Single-Cell Movies for Original 10 Cells
    print("\n--- Generating Single-Cell Comparison Movies ---", flush=True)
    for cid in ORIGINAL_10_CELLS:
        generate_single_cell_movie(cid)
        
    print("\nValidation run complete! All outputs saved to:", TEST_OUTPUT_DIR, flush=True)

def mask_to_rle_str(mask):
    if not mask.any():
        return ""
    flat = mask.flatten()
    diffs = np.diff(np.concatenate(([0], flat.astype(np.int8), [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths = ends - starts
    pairs = []
    for s, l in zip(starts, lengths):
        pairs.append(f"{s} {l}")
    return " ".join(pairs)

if __name__ == '__main__':
    main()
