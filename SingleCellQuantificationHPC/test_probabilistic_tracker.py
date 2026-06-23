#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import cv2
from skimage.io import imread
from pathlib import Path
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker_model import load_tracker
from ai_tracking_inference import get_candidates
from tracker_dataset import _padded_bbox, _crop_and_resize, _norm_img
from Cell_tracking_functions import rle_decode, area_change_penalty, iou

def get_centroid(mask):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))

def track_cell_probabilistic(cell_id, model, exp_path, device, w_iou=1.0, w_dist=10.0, w_area=2.0, w_merge=0.02, w_split=0.5):
    movie_name = exp_path.name
    frames_dir = exp_path / f"Frames_{movie_name}"
    masks_dir = exp_path / f"Masks_{movie_name}"
    
    gt_csv = exp_path / f"TrackedCells_{movie_name}" / f"cell_{cell_id}_masks.csv"
    if not gt_csv.exists():
        print(f"GT CSV does not exist: {gt_csv}")
        return None
        
    df_gt = pd.read_csv(gt_csv)
    H, W = int(df_gt.iloc[0]['height']), int(df_gt.iloc[0]['width'])
    
    # Auto-detect rle column
    rle_col = 'rle_bf' if 'rle_bf' in df_gt.columns else 'rle_gfp'
    
    # Load frame-0 seed mask
    rle_0 = df_gt[df_gt['time_point'] == 0].iloc[0][rle_col]
    seed_mask = rle_decode(rle_0, (H, W)).astype(bool)
    
    prev_mask = seed_mask.copy()
    T = len(df_gt)
    ious = [1.0]
    
    # Track history
    centroid_history = [get_centroid(seed_mask)]
    area_history = [seed_mask.sum()]
    comp_history = ['single'] # 'single' or 'pair'
    
    divergence_found = False
    
    for t in range(1, T):
        lab_cur_path = masks_dir / f"{movie_name}_t_{t:03d}_c_0_seg.tif"
        if not lab_cur_path.exists():
            lab_cur_path = masks_dir / f"{movie_name}_t_{t:03d}_c_0_seg.npy"
            if not lab_cur_path.exists():
                ious.append(0.0)
                continue
                
        if lab_cur_path.suffix == '.tif':
            lab_cur = cv2.imread(str(lab_cur_path), cv2.IMREAD_UNCHANGED)
        else:
            lab_cur = np.load(str(lab_cur_path))
            
        cands, cand_ids = get_candidates(lab_cur, prev_mask)
        
        # Load GT for evaluation
        gt_row = df_gt[df_gt['time_point'] == t]
        if gt_row.empty:
            ious.append(0.0)
            continue
        rle_gt = gt_row.iloc[0][rle_col]
        if not isinstance(rle_gt, str) or not rle_gt or rle_gt == "nan":
            gt_mask = np.zeros((H, W), dtype=bool)
        else:
            gt_mask = rle_decode(rle_gt, (H, W)).astype(bool)
            
        gt_cand_idx = -1
        best_gt_cand_iou = 0.0
        for idx, cand_mask in enumerate(cands):
            inter = np.logical_and(gt_mask, cand_mask).sum()
            union = np.logical_or(gt_mask, cand_mask).sum()
            iou_val = inter / max(union, 1)
            if iou_val > 0.8:
                gt_cand_idx = idx
                best_gt_cand_iou = iou_val
                
        if not cands:
            best_mask = prev_mask
            best_comp = comp_history[-1]
        else:
            bf_t0_path = frames_dir / f"{movie_name}_t_{t-1:03d}_c_0.tif"
            bf_t1_path = frames_dir / f"{movie_name}_t_{t:03d}_c_0.tif"
            img_t0 = imread(str(bf_t0_path)) if bf_t0_path.exists() else imread(str(bf_t1_path))
            img_t1 = imread(str(bf_t1_path)) if bf_t1_path.exists() else img_t0
            
            r0, r1, c0, c1 = _padded_bbox(prev_mask, H, W, pad_frac=0.4)
            img_t0_norm = _norm_img(img_t0)
            img_t1_norm = _norm_img(img_t1)
            img_t0_crop = _crop_and_resize(img_t0_norm, r0, r1, c0, c1, 128)
            img_t1_crop = _crop_and_resize(img_t1_norm, r0, r1, c0, c1, 128)
            mask_t0_crop = _crop_and_resize(prev_mask.astype(np.float32), r0, r1, c0, c1, 128)
            
            img3ch = np.stack([img_t0_crop, img_t1_crop, mask_t0_crop], axis=0)
            img3ch_t = torch.from_numpy(img3ch).unsqueeze(0).to(device)
            
            # Predict expected centroid using velocity (motion vector)
            if len(centroid_history) >= 2:
                v = np.array(centroid_history[-1]) - np.array(centroid_history[-2])
                expected_centroid = np.array(centroid_history[-1]) + v
            else:
                expected_centroid = np.array(centroid_history[-1])
                
            # Running average area of last 5 frames
            ref_area = np.mean(area_history[-5:])
            
            prev_comp = comp_history[-1]
            
            cand_details = []
            for idx, (cand_mask, ids) in enumerate(zip(cands, cand_ids)):
                is_pair = (len(ids) == 2)
                
                # Overlap (IoU) with previous mask
                iou_prev = iou(prev_mask, cand_mask)
                
                # Distance to expected centroid
                cand_centroid = get_centroid(cand_mask)
                dist = np.sqrt((cand_centroid[0] - expected_centroid[0])**2 + (cand_centroid[1] - expected_centroid[1])**2)
                
                # Area change penalty compared to running average area
                cand_area = cand_mask.sum()
                pen_area = area_change_penalty(cand_area, ref_area, ratio_soft=1.3, ratio_hard=1.8)
                
                # AI Inference
                cand_crop = _crop_and_resize(cand_mask.astype(np.float32), r0, r1, c0, c1, 128)
                cand_t = torch.from_numpy(cand_crop).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model.score(img3ch_t, cand_t.unsqueeze(1))
                    prob = torch.sigmoid(out)[0, 0].item()
                    
                # HMM State transition (topology factor)
                if prev_comp == 'single':
                    P_topology = 1.0 if not is_pair else w_merge
                else: # prev was 'pair'
                    P_topology = w_split if not is_pair else 1.0
                    
                dist_factor = np.exp(-dist / w_dist)
                area_factor = np.exp(-pen_area * w_area)
                iou_factor = iou_prev ** w_iou
                
                score = prob * iou_factor * dist_factor * area_factor * P_topology
                
                cand_details.append({
                    'idx': idx,
                    'ids': ids,
                    'is_pair': is_pair,
                    'iou_prev': iou_prev,
                    'dist': dist,
                    'area': cand_area,
                    'prob': prob,
                    'score': score,
                    'mask': cand_mask
                })
                
            best = max(cand_details, key=lambda x: x['score'])
            best_mask = best['mask']
            best_comp = 'pair' if best['is_pair'] else 'single'
            
            # Print divergence info if it happens
            is_divergent = (best['idx'] != gt_cand_idx)
            if is_divergent and not divergence_found:
                divergence_found = True
                print(f"\nDIVERGENCE DETECTED at t={t}")
                print(f"GT Candidate Index: {gt_cand_idx} (IoU with GT: {best_gt_cand_iou:.4f})")
                print(f"Chosen Candidate Index: {best['idx']}")
                for c in cand_details:
                    mark = " [CHOSEN]" if c['idx'] == best['idx'] else ""
                    mark += " [GT]" if c['idx'] == gt_cand_idx else ""
                    print(f"  Cand {c['idx']} (ids={c['ids']}){mark}:")
                    print(f"    AI prob:    {c['prob']:.4f}")
                    print(f"    IoU prev:   {c['iou_prev']:.4f}")
                    print(f"    Dist expected: {c['dist']:.4f}")
                    print(f"    Area:       {c['area']} (ref_area={ref_area:.1f})")
                    print(f"    Final Score: {c['score']:.6f}")
                    
        # Calculate overlap with GT for stats
        gt_inter = np.logical_and(gt_mask, best_mask).sum()
        gt_union = np.logical_or(gt_mask, best_mask).sum()
        ious.append(gt_inter / max(gt_union, 1))
        
        # Update histories
        prev_mask = best_mask
        centroid_history.append(get_centroid(best_mask))
        area_history.append(best_mask.sum())
        comp_history.append(best_comp)
        
    mean_iou = np.mean(ious)
    print(f"\nFinal Tracking IoU with GT for Cell {cell_id}: {mean_iou:.4f}")
    if not divergence_found:
        print("No tracking divergence occurred! Success!")
    return mean_iou

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    ckpt_path = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/tracker_checkpoints/model_latest.pt"
    model = load_tracker(ckpt_path, device=device)
    model.eval()
    
    exp_path_m92 = Path("/Volumes/X10 Pro/Movies/2025_12_31_M92/A14-YES-1t-FBFBF-2_F2")
    track_cell_probabilistic(38, model, exp_path_m92, device)

if __name__ == "__main__":
    main()
