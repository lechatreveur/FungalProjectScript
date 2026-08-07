#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_comparison_video.py
------------------------
Generates a 3-panel side-by-side comparison video showing:
[ Ground Truth ] [ Meta SAM2 ] [ Custom AI Tracker ]
For a specific cell track, with colored overlays and real-time IoU metrics.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from skimage.io import imread
from PIL import Image

# Ensure local imports work
HPC_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(HPC_DIR.parent))

from Cell_tracking_functions import rle_decode, iou
from sam2_tracker import track_cell_with_sam2, normalize_frame

def draw_mask_overlay(img_rgb, mask, color, opacity=0.3):
    """Overlay a semi-transparent colored mask and a solid contour onto an image."""
    overlay = img_rgb.copy()
    if not mask.any():
        return overlay
        
    # Apply color overlay
    overlay[mask] = (overlay[mask] * (1.0 - opacity) + np.array(color) * opacity).astype(np.uint8)
    
    # Draw contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="2026_01_08_M93")
    parser.add_argument("--film", type=str, default="A14_BF_1_F1")
    parser.add_argument("--cell-id", type=int, default=23)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--fps", type=int, default=7, help="Playback FPS of output video")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the video")
    args = parser.parse_args()
    
    if args.output_dir is None:
        # Default to brain conversation artifacts directory
        output_dir = Path("/Users/user/.gemini/antigravity-ide/brain/4dcb4bf4-544f-4647-96fd-e18a94325d83")
    else:
        output_dir = Path(args.output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating side-by-side comparison for Cell {args.cell_id} in film {args.film} ({args.experiment})...")
    
    # Paths setup
    BASE_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
    film_dir = BASE_MOVIE_ROOT / args.experiment / args.film
    frames_dir = film_dir / f"Frames_{args.film}"
    masks_dir = film_dir / f"Masks_{args.film}"
    tracked_dir = film_dir / f"TrackedCells_{args.film}"
    csv_path = tracked_dir / f"cell_{args.cell_id}_masks.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Cell mask CSV not found: {csv_path}")
        
    # 1. Load Ground Truth Masks
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
        raise ValueError("No non-empty ground truth masks found.")
        
    t_start = min(valid_ts)
    initial_mask = gt_masks[t_start]
    
    # 2. Discover frames
    # Detect channel
    channel_idx = 0
    for f in sorted(masks_dir.iterdir()):
        if f.name.startswith("."):
            continue
        import re
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
        raise FileNotFoundError(f"No frame files found in {frames_dir}")
        
    # 3. Run SAM2 Tracker
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_ckpt = "/Volumes/X10 Pro/Movies/AI/sam2_checkpoints/sam2.1_hiera_tiny.pt"
    if not os.path.exists(sam2_ckpt):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_ckpt}")
        
    print("Running Meta SAM2...")
    sam2_res = track_cell_with_sam2(
        frame_paths=frame_files,
        initial_mask=initial_mask,
        checkpoint_path=sam2_ckpt,
        model_cfg=sam2_cfg,
        device=args.device,
        start_frame_idx=t_start
    )
    
    # 4. Run Custom AI Tracker
    custom_ckpt = "/Volumes/X10 Pro/Movies/AI/tracker_checkpoints/model_latest.pt"
    print("Running Custom AI Tracker...")
    from tracker_model import load_tracker
    from ai_tracking_inference import ai_track_one_direction
    
    custom_model = load_tracker(custom_ckpt, device=args.device)
    custom_model.eval()
    
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
        
    # 5. Build side-by-side frames and write video
    temp_output_path = output_dir / f"comparison_raw_cell_{args.cell_id}.mp4"
    final_output_path = output_dir / f"comparison_cell_{args.cell_id}.mp4"
    
    # Frame shape is H, W. Three panels: H, W*3.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_output_path), fourcc, args.fps, (W * 3, H))
    
    print("Compositing video frames...")
    
    for t in range(t_start, T):
        # Read and normalize raw frame
        img = imread(str(frame_files[t]))
        img_rgb = normalize_frame(img)
        
        # Color space conversion: OpenCV expects BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Get masks
        gt_mask = gt_masks.get(t, np.zeros((H, W), dtype=bool))
        s_mask = sam2_res.get(t, np.zeros((H, W), dtype=bool))
        c_mask = custom_res.get(t, np.zeros((H, W), dtype=bool))
        
        # Compute IoUs
        s_iou = iou(gt_mask, s_mask) if gt_mask.any() else 0.0
        c_iou = iou(gt_mask, c_mask) if gt_mask.any() else 0.0
        
        # Ground Truth Panel (Green)
        panel_gt = draw_mask_overlay(img_bgr, gt_mask, color=(0, 255, 0), opacity=0.3)
        cv2.putText(panel_gt, "Ground Truth", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(panel_gt, f"Frame: {t}", (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # SAM2 Panel (Cyan overlay)
        panel_sam2 = draw_mask_overlay(img_bgr, s_mask, color=(255, 255, 0), opacity=0.3) # BGR Cyan
        cv2.putText(panel_sam2, "Meta SAM2 (Tiny)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(panel_sam2, f"IoU: {s_iou:.3f}", (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Custom AI Panel (Red overlay)
        panel_custom = draw_mask_overlay(img_bgr, c_mask, color=(0, 0, 255), opacity=0.3) # BGR Red
        cv2.putText(panel_custom, "Custom AI Tracker", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(panel_custom, f"IoU: {c_iou:.3f}", (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Combine
        combined_frame = np.hstack([panel_gt, panel_sam2, panel_custom])
        out.write(combined_frame)
        
    out.release()
    print(f"Raw video written to {temp_output_path}")
    
    # 6. Transcode with ffmpeg for HTML5 video player compatibility (H.264 / AAC)
    print("Transcoding video for browser compatibility...")
    cmd = f"ffmpeg -y -i '{temp_output_path}' -vcodec libx264 -pix_fmt yuv420p '{final_output_path}'"
    ret = os.system(cmd)
    if ret == 0:
        print(f"Transcoded video successfully saved to: {final_output_path}")
        try:
            os.remove(temp_output_path)
        except OSError:
            pass
    else:
        print(f"Warning: ffmpeg transcoding failed with exit code {ret}. Using raw video.")
        os.rename(temp_output_path, final_output_path)
        
    print("Done!")

if __name__ == "__main__":
    main()
