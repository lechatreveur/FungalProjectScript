import os
import sys
import subprocess
import glob
import cv2
import numpy as np

PYTHON_ENV = "/home/hsushen/miniconda3/envs/cellpose_env/bin/python3"
SCRIPT_PATH = "/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/one_cell_quantification_1CH.py"
EXP_PATH = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156"
FILE_NAME = "3_FL2_F0"

LEGACY_PLOTS_DIR = os.path.join(EXP_PATH, FILE_NAME, "TrackedCells_3_FL2_F0", "cell_plots")
NEW_PLOTS_DIR = os.path.join(EXP_PATH, FILE_NAME, "TrackedCells_3_FL2_F0_cpsam_overexp", "cell_plots")
MOVIES_OUTPUT_DIR = os.path.join(EXP_PATH, "cellpose_overexposure_test", "single_cell_movies")

os.makedirs(MOVIES_OUTPUT_DIR, exist_ok=True)

def run_quantification_with_plot(cell_id, backend):
    print(f"[quant] Running cell {cell_id} with backend={backend} and --do_plot...", flush=True)
    cmd = [
        PYTHON_ENV, SCRIPT_PATH,
        "--cell_id", str(cell_id),
        "--experiment_path", EXP_PATH,
        "--file_name", FILE_NAME,
        "--seg-backend", backend,
        "--do_plot",
        "--update_existing"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[error] Quantification failed for cell {cell_id} ({backend}):\n{res.stderr}", flush=True)

def create_plot_comparison_movie(cell_id):
    leg_dir = os.path.join(LEGACY_PLOTS_DIR, f"cell_{cell_id}")
    new_dir = os.path.join(NEW_PLOTS_DIR, f"cell_{cell_id}")
    
    leg_imgs = sorted(glob.glob(os.path.join(leg_dir, "frame_t_*.png")))
    new_imgs = sorted(glob.glob(os.path.join(new_dir, "frame_t_*.png")))
    
    print(f"[movie] Found {len(leg_imgs)} legacy plot frames and {len(new_imgs)} new plot frames for cell {cell_id}", flush=True)
    
    if not leg_imgs and not new_imgs:
        print(f"[warn] No plot frames found for cell {cell_id}", flush=True)
        return
        
    out_mp4 = os.path.join(MOVIES_OUTPUT_DIR, f"cell_{cell_id}_plot_comparison.mp4")
    
    # Read sample image to get resolution
    sample_path = leg_imgs[0] if leg_imgs else new_imgs[0]
    sample_img = cv2.imread(sample_path)
    h, w, _ = sample_img.shape
    
    out_w = w * 2
    out_h = h + 40
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_mp4, fourcc, 5, (out_w, out_h))
    
    max_frames = max(101, len(leg_imgs), len(new_imgs))
    
    for t in range(max_frames):
        leg_p = os.path.join(leg_dir, f"frame_t_{t:03d}.png")
        new_p = os.path.join(new_dir, f"frame_t_{t:03d}.png")
        
        leg_frame = cv2.imread(leg_p) if os.path.exists(leg_p) else np.zeros((h, w, 3), dtype=np.uint8)
        new_frame = cv2.imread(new_p) if os.path.exists(new_p) else np.zeros((h, w, 3), dtype=np.uint8)
        
        if leg_frame.shape[:2] != (h, w):
            leg_frame = cv2.resize(leg_frame, (w, h))
        if new_frame.shape[:2] != (h, w):
            new_frame = cv2.resize(new_frame, (w, h))
            
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[40:40+h, 0:w] = leg_frame
        combined[40:40+h, w:2*w] = new_frame
        
        # Add Header text
        cv2.putText(combined, f"CELL {cell_id} PLOT COMPARISON (t={t:03d})", (out_w // 4, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "EXISTING PIPELINE (LEGACY)", (w // 4, out_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(combined, "CELLPOSE-SAM OVEREXPOSED", (w + w // 4, out_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
                    
        writer.write(combined)
        
    writer.release()
    print(f"[success] Created cell plot comparison movie: {out_mp4}", flush=True)

if __name__ == "__main__":
    cells_to_process = [25, 116, 9, 18, 15, 11, 19, 140, 178, 34]
    if len(sys.argv) > 1:
        cells_to_process = [int(x) for x in sys.argv[1:]]
        
    for cid in cells_to_process:
        print(f"\n=== Processing Cell {cid} ===")
        run_quantification_with_plot(cid, "legacy")
        run_quantification_with_plot(cid, "cellpose_overexposed")
        create_plot_comparison_movie(cid)
