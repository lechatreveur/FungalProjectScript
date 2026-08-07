import os
import torch
import numpy as np
import cv2
from skimage.io import imread
from pathlib import Path

from tracker_dataset import _norm_img, _crop_and_resize, _padded_bbox, _segments_adjacent

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Cell_tracking_functions import mask_to_rle, rle_to_mask, area_change_penalty

if not hasattr(sys, '_ai_tracker_image_cache'):
    sys._ai_tracker_image_cache = {}
if not hasattr(sys, '_ai_tracker_seg_cache'):
    sys._ai_tracker_seg_cache = {}

def cached_imread(path):
    p_str = str(path)
    if len(sys._ai_tracker_image_cache) > 200:
        sys._ai_tracker_image_cache.clear()
    if p_str not in sys._ai_tracker_image_cache:
        sys._ai_tracker_image_cache[p_str] = imread(p_str)
    return sys._ai_tracker_image_cache[p_str]

def cached_cv2_imread(path):
    p_str = str(path)
    if len(sys._ai_tracker_seg_cache) > 200:
        sys._ai_tracker_seg_cache.clear()
    if p_str not in sys._ai_tracker_seg_cache:
        sys._ai_tracker_seg_cache[p_str] = cv2.imread(p_str, cv2.IMREAD_UNCHANGED)
    return sys._ai_tracker_seg_cache[p_str]

def get_centroid(mask):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))

def get_candidates(lab_cur, ref_mask, min_overlap=0.1):
    H, W = lab_cur.shape
    candidates = []
    cand_ids = []
    
    intersect = lab_cur[ref_mask]
    if len(intersect) == 0:
        return candidates, cand_ids
        
    uniq_ids, counts = np.unique(intersect, return_counts=True)
    ref_area = ref_mask.sum()
    
    # Valid singles
    valid_singles = []
    for uid, count in zip(uniq_ids, counts):
        if uid == 0:
            continue
        # Allow relatively low overlap so we catch moving cells
        valid_singles.append(uid)
        
    # Build single candidates
    for uid in valid_singles:
        cand_mask = (lab_cur == uid)
        candidates.append(cand_mask)
        cand_ids.append([uid])
        
    # Build pair candidates for any two valid singles that are adjacent
    for i in range(len(valid_singles)):
        for j in range(i+1, len(valid_singles)):
            id_a, id_b = valid_singles[i], valid_singles[j]
            mask_a = (lab_cur == id_a)
            mask_b = (lab_cur == id_b)
            if _segments_adjacent(mask_a, mask_b, max_dist_px=3):
                pair_mask = mask_a | mask_b
                candidates.append(pair_mask)
                cand_ids.append([id_a, id_b])
                
    return candidates, cand_ids

def ai_track_one_direction(t_seq, ref_start_mask, bf_frame_path_func, lab_seg_path_func, model, device="mps", target_size=128, use_probabilistic=True, w_keep_prior=0.35):
    results = {}
    prev_mask = ref_start_mask.copy()
    
    # Tracking hyperparameters for geometric regularization and state transitions
    w_iou = 1.0
    w_dist = 10.0
    w_area = 2.0
    w_merge = 0.02  # Transition prior: single -> pair (merge)
    w_split = 0.5   # Transition prior: pair -> single (split/division)

    # Track histories for temporal consistency
    centroid_history = [get_centroid(ref_start_mask)]
    area_history = [float(ref_start_mask.sum())]
    comp_history = ['single']

    for i, t in enumerate(t_seq):
        if i == 0:
            touch = False
            if ref_start_mask.any():
                touch = bool(ref_start_mask[0, :].any() or ref_start_mask[-1, :].any() or ref_start_mask[:, 0].any() or ref_start_mask[:, -1].any())
            results[t] = {
                'mask': ref_start_mask.copy(),
                'composition': 'single',
                'segments_rle': None,
                'ai_score': 1.0,
                'rejected': False,
                'touch': touch,
                'overlap': 1.0,
                'score': 1.0,
                'pen': 0.0,
                'sel_mode': 'seed_mask',
                'area': int(ref_start_mask.sum())
            }
            prev_mask = ref_start_mask.copy()
            continue

        t_prev = t_seq[i-1]
        
        lab_cur_path = lab_seg_path_func(t)
        if not os.path.exists(lab_cur_path):
            touch = False
            if prev_mask.any():
                touch = bool(prev_mask[0, :].any() or prev_mask[-1, :].any() or prev_mask[:, 0].any() or prev_mask[:, -1].any())
            results[t] = {
                'mask': prev_mask.copy(),
                'composition': comp_history[-1],
                'ai_score': 1.0,
                'rejected': True,
                'touch': touch,
                'overlap': 0.0,
                'score': -1e9,
                'pen': 0.0,
                'sel_mode': 'missing_file',
                'area': int(prev_mask.sum()) if prev_mask is not None else 0
            }
            # Maintain history consistency
            centroid_history.append(centroid_history[-1])
            area_history.append(area_history[-1])
            comp_history.append(comp_history[-1])
            continue
            
        lab_cur = cached_cv2_imread(lab_cur_path)
        
        # Load BF images
        bf_t0_path = bf_frame_path_func(t_prev)
        bf_t1_path = bf_frame_path_func(t)
        
        img_t0 = cached_imread(bf_t0_path) if os.path.exists(bf_t0_path) else cached_imread(bf_t1_path)
        img_t1 = cached_imread(bf_t1_path) if os.path.exists(bf_t1_path) else img_t0
        
        # Build candidates
        cands, cand_ids = get_candidates(lab_cur, prev_mask)
        if use_probabilistic:
            cands.append(prev_mask)
            cand_ids.append([-1])
            
        if not cands:
            touch = False
            if prev_mask.any():
                touch = bool(prev_mask[0, :].any() or prev_mask[-1, :].any() or prev_mask[:, 0].any() or prev_mask[:, -1].any())
            results[t] = {
                'mask': prev_mask.copy(),
                'composition': comp_history[-1],
                'ai_score': 0.0,
                'rejected': True,
                'touch': touch,
                'overlap': 0.0,
                'score': -1e9,
                'pen': 0.0,
                'sel_mode': 'no_candidates',
                'area': int(prev_mask.sum()) if prev_mask is not None else 0
            }
            # Maintain history consistency
            centroid_history.append(centroid_history[-1])
            area_history.append(area_history[-1])
            comp_history.append(comp_history[-1])
            continue
            
        # Prepare inputs for AI
        H, W = img_t0.shape
        r0, r1, c0, c1 = _padded_bbox(prev_mask, H, W, pad_frac=0.4)
        
        img_t0_norm = _norm_img(img_t0)
        img_t1_norm = _norm_img(img_t1)
        img_t0_crop = _crop_and_resize(img_t0_norm, r0, r1, c0, c1, target_size)
        img_t1_crop = _crop_and_resize(img_t1_norm, r0, r1, c0, c1, target_size)
        mask_t0_crop = _crop_and_resize(prev_mask.astype(np.float32), r0, r1, c0, c1, target_size)
        
        # Stack context: (bf_t0, bf_t1, mask_t0) -> 3 channels
        img3ch = np.stack([img_t0_crop, img_t1_crop, mask_t0_crop], axis=0) # [3, H, W]
        img3ch_t = torch.from_numpy(img3ch).unsqueeze(0).to(device) # [1, 3, H, W]
        
        # Predict expected centroid using velocity (motion vector)
        if len(centroid_history) >= 2:
            v = np.array(centroid_history[-1]) - np.array(centroid_history[-2])
            expected_centroid = np.array(centroid_history[-1]) + v
        else:
            expected_centroid = np.array(centroid_history[-1])
            
        # Running average area of last 5 frames
        ref_area = np.mean(area_history[-5:])
        
        prev_comp = comp_history[-1]
        
        # Pre-crop and stack all candidates to evaluate them in a single batch
        cand_crops = []
        for cand_mask in cands:
            cand_crop = _crop_and_resize(cand_mask.astype(np.float32), r0, r1, c0, c1, target_size)
            cand_crops.append(cand_crop)
            
        # Shape: (K, 1, H, W) -> add batch dim -> (1, K, 1, H, W)
        cands_array = np.stack(cand_crops, axis=0)[:, np.newaxis, :, :].astype(np.float32)
        cands_t = torch.from_numpy(cands_array).unsqueeze(0).to(device)
        
        # Batch inference
        with torch.no_grad():
            out = model.score(img3ch_t, cands_t) # returns (1, K) logits
            probs = torch.sigmoid(out)[0].tolist() # list of length K
            
        cand_details = []
        for idx, (cand_mask, ids) in enumerate(zip(cands, cand_ids)):
            prob = probs[idx]
            is_keep_prev = (ids == [-1])
            is_pair = (len(ids) == 2)
            
            # Compute geometric scores
            inter = np.logical_and(prev_mask, cand_mask).sum()
            union = np.logical_or(prev_mask, cand_mask).sum()
            iou_val = inter / float(union) if union > 0 else 0.0
            
            cand_centroid = get_centroid(cand_mask)
            dist = np.sqrt((cand_centroid[0] - expected_centroid[0])**2 + (cand_centroid[1] - expected_centroid[1])**2)
            
            cand_area = cand_mask.sum()
            pen_area = area_change_penalty(cand_area, ref_area, ratio_soft=1.3, ratio_hard=1.8)
            
            # HMM State transition (topology factor)
            if prev_comp == 'single':
                P_topology = 1.0 if not is_pair else w_merge
            else: # prev was 'pair'
                P_topology = w_split if not is_pair else 1.0
                
            dist_factor = np.exp(-dist / w_dist)
            area_factor = np.exp(-pen_area * w_area)
            iou_factor = (iou_val ** w_iou)
            
            prior = w_keep_prior if is_keep_prev else 1.0
            score = prob * iou_factor * dist_factor * area_factor * P_topology * prior
            
            cand_details.append({
                'ids': ids,
                'is_keep_prev': is_keep_prev,
                'is_pair': is_pair,
                'iou': iou_val,
                'score': score,
                'ai_prob': prob,
                'mask': cand_mask,
                'area': cand_area,
                'centroid': cand_centroid
            })
            
        # Select best candidate based on soft probabilistic joint score
        best = max(cand_details, key=lambda x: x['score'])
            
        best_mask = best['mask']
        best_ids = best['ids']
        best_score = best['ai_prob']
        
        composition = 'keep_prev' if best['is_keep_prev'] else ('pair' if best['is_pair'] else 'single')
        segments_rle = [mask_to_rle(lab_cur == uid) for uid in best_ids] if composition == 'pair' else None
        
        touch = False
        if best_mask.any():
            touch = bool(best_mask[0, :].any() or best_mask[-1, :].any() or best_mask[:, 0].any() or best_mask[:, -1].any())

        results[t] = {
            'mask': best_mask,
            'composition': composition,
            'segments_rle': segments_rle,
            'ai_score': best_score,
            'rejected': False,
            'touch': touch,
            'overlap': best['iou'], 
            'score': best['score'],
            'pen': 0.0,
            'sel_mode': 'keep_prev' if best['is_keep_prev'] else 'ai_tracker',
            'area': int(best_mask.sum())
        }
        
        prev_mask = best_mask
        centroid_history.append(best['centroid'])
        area_history.append(float(best['area']))
        comp_history.append(comp_history[-1] if best['is_keep_prev'] else composition)
        
    return results

