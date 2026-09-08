#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_missegmented_poles.py

Automated pole recovery helper module and CLI for single-cell missegmentations
in Schizosaccharomyces pombe time-lapse microscopy.

Because polarity site dynamics occur predominantly at cell poles, a shortened major
axis completely misses polar fluorescent signals. This tool:
1. Interpolates expected cell geometry, orientation, and physical endpoints from
   neighboring keyframes (±5 sliding window).
2. Identifies which pole(s) of the missegmented cell are cut off / truncated.
3. Automatically recovers the missing pole(s) by:
   - Fusing adjacent split segments in the Cellpose segmentation map (_seg.tif), or
   - Extruding a minor-axis width brush stroke along the major axis to the expected pole.
4. Updates both cell_<cid>_masks.csv and the Cellpose _seg.tif file (with automated backups).
5. Generates before-and-after diagnostic overlay images.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import disk, binary_closing

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SingleCellQuantificationHPC"))

from SingleCellQuantificationHPC.ground_truth_corrector.schemas import (
    validate_and_decode_rle,
    encode_mask_to_rle
)


def get_mask_geometry(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Computes precise morphological parameters and physical endpoint coordinates
    for a binary single-cell mask using principal eigenvector decomposition.
    """
    if mask is None or not mask.any():
        return None
        
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return None
        
    p = props[0]
    ys, xs = np.where(mask > 0)
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    
    if len(ys) > 3:
        cov = np.cov(ys, xs)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u_long = eigvecs[:, -1]  # (dy, dx) along major variance
    else:
        u_long = np.array([1.0, 0.0], dtype=np.float64)
        
    # Standardize orientation: ensure u_long has positive y-component (points downward)
    if u_long[0] < 0 or (u_long[0] == 0 and u_long[1] < 0):
        u_long = -u_long
    norm = np.linalg.norm(u_long)
    if norm > 0:
        u_long = u_long / norm
    else:
        u_long = np.array([1.0, 0.0], dtype=np.float64)
        
    u_short = np.array([u_long[1], -u_long[0]], dtype=np.float64)
    
    # Project all foreground pixels along the major axis relative to centroid
    proj = (ys - cy) * u_long[0] + (xs - cx) * u_long[1]
    p_max = float(np.max(proj))
    p_min = float(np.min(proj))
    
    # Physical tip coordinates
    tip1 = np.array([cy, cx]) + p_max * u_long  # +u_long pole
    tip2 = np.array([cy, cx]) + p_min * u_long  # -u_long pole
    
    return {
        "centroid": np.array([cy, cx], dtype=np.float64),
        "major_axis_length": float(p.major_axis_length),
        "minor_axis_length": float(p.minor_axis_length),
        "orientation": float(p.orientation),
        "u_long": u_long,
        "u_short": u_short,
        "p_max": p_max,
        "p_min": p_min,
        "physical_span": p_max - p_min,
        "tip1": tip1,
        "tip2": tip2,
        "area": int(p.area),
        "bbox": p.bbox
    }


def bisect_fused_mask(
    fused_mask: np.ndarray,
    target_centroid: np.ndarray,
    u_long: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Bisects an undersegmented mask covering two touching sister cells along the
    cleavage plane perpendicular to the major axis, returning the sister sub-mask
    closest to target_centroid.
    """
    if fused_mask is None or not fused_mask.any():
        return fused_mask

    ys, xs = np.where(fused_mask > 0)
    if len(ys) < 50:
        return fused_mask

    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    if u_long is None:
        coords = np.column_stack([ys - cy, xs - cx])
        cov = np.cov(coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        u_long = eigenvectors[:, np.argmax(eigenvalues)]
        u_long = u_long / np.linalg.norm(u_long)

    # Project pixels along major axis relative to centroid
    proj = (ys - cy) * u_long[0] + (xs - cx) * u_long[1]
    
    # Split into positive and negative projection halves
    half1_mask = np.zeros_like(fused_mask, dtype=bool)
    half2_mask = np.zeros_like(fused_mask, dtype=bool)
    
    pos_idx = np.where(proj >= 0)[0]
    neg_idx = np.where(proj < 0)[0]
    
    half1_mask[ys[pos_idx], xs[pos_idx]] = True
    half2_mask[ys[neg_idx], xs[neg_idx]] = True
    
    c1 = np.array([np.mean(ys[pos_idx]), np.mean(xs[pos_idx])]) if len(pos_idx) > 0 else np.array([cy, cx])
    c2 = np.array([np.mean(ys[neg_idx]), np.mean(xs[neg_idx])]) if len(neg_idx) > 0 else np.array([cy, cx])
    
    d1 = np.linalg.norm(c1 - target_centroid)
    d2 = np.linalg.norm(c2 - target_centroid)
    
    chosen_mask = half1_mask if d1 <= d2 else half2_mask
    return chosen_mask.astype(np.uint8)


def interpolate_expected_geometry(
    exp_dir: Path,
    sequence: str,
    cell_key: str,
    target_film: str,
    target_t: int,
    window_radius: int = 5,
    k_div: Optional[int] = None,
    cached_kf_entries: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Interpolates expected cell geometry, orientation, and endpoints at (target_film, target_t)
    using a phase-aware sliding window across the sequence trajectory.
    """
    if cached_kf_entries is not None:
        kf_entries = cached_kf_entries
        target_kf_idx = None
        for idx, kf in enumerate(kf_entries):
            if kf["film"] == target_film and kf["t"] == target_t:
                target_kf_idx = idx
                break
    else:
        link_path = exp_dir / "sequence_linkage.json"
        if not link_path.exists():
            raise FileNotFoundError(f"Sequence linkage not found at {link_path}")
            
        with open(link_path, "r") as f:
            link_data = json.load(f)
            
        seq_data = link_data.get("sequences", link_data).get(sequence)
        if not seq_data:
            raise ValueError(f"Sequence '{sequence}' not found in sequence_linkage.json")
            
        films = seq_data["films"]
        global_cells = seq_data["global_cells"]
        
        # Normalize cell key
        actual_cell_key = cell_key
        if cell_key not in global_cells:
            for k in global_cells:
                if k == cell_key or k.endswith(f"_{cell_key}") or cell_key.endswith(k):
                    actual_cell_key = k
                    break
                    
        if actual_cell_key not in global_cells:
            raise ValueError(f"Cell '{cell_key}' not found in global cells of sequence '{sequence}'")
            
        track = global_cells[actual_cell_key]
        
        # Build complete keyframe sequence (39 keyframes)
        kf_entries = []
        target_kf_idx = None
        
        for f_idx, film_name in enumerate(films):
            kts = [0, 50, 100] if "FL" in film_name else [0, 20, 40]
            local_cid = track[f_idx]
            
            masks_csv = exp_dir / film_name / f"TrackedCells_{film_name}" / f"cell_{local_cid}_masks.csv"
            df = None
            if local_cid > 0 and masks_csv.exists():
                try:
                    df = pd.read_csv(masks_csv)
                except Exception:
                    df = None
                    
            rle_col = "rle_gfp" if "FL" in film_name and df is not None and "rle_gfp" in df.columns and df["rle_gfp"].dropna().any() else "rle_bf"
            
            for kt in kts:
                curr_idx = len(kf_entries)
                if film_name == target_film and kt == target_t:
                    target_kf_idx = curr_idx
                    
                geom = None
                mask = None
                if df is not None and not df.empty:
                    rows = df[df["time_point"] == kt]
                    if not rows.empty:
                        rle_val = str(rows.iloc[0].get(rle_col, ""))
                        if rle_val and rle_val.strip() and rle_val.lower() != "nan":
                            try:
                                H = int(df.iloc[0]["height"]) if "height" in df.columns else 2000
                                W = int(df.iloc[0]["width"]) if "width" in df.columns else 2000
                                mask = validate_and_decode_rle(rle_val, H, W)
                                geom = get_mask_geometry(mask)
                            except Exception:
                                geom = None
                                
                kf_entries.append({
                    "kf_idx": curr_idx,
                    "film": film_name,
                    "t": kt,
                    "local_cid": local_cid,
                    "geom": geom,
                    "mask": mask
                })
            
    if target_kf_idx is None:
        raise ValueError(f"Target keyframe ({target_film}, t={target_t}) could not be indexed.")
        
    start_idx = max(0, target_kf_idx - window_radius)
    end_idx = min(len(kf_entries), target_kf_idx + window_radius + 1)
    
    # Phase-aware partitioning: never mix pre-division mother frames with post-division daughter frames
    if k_div is not None:
        if target_kf_idx >= k_div:
            start_idx = max(start_idx, k_div)  # Daughter phase only
        else:
            end_idx = min(end_idx, k_div)      # Mother phase only
            
    neighbor_geoms = []
    for i in range(start_idx, end_idx):
        if i == target_kf_idx:
            continue
        g = kf_entries[i]["geom"]
        if g is not None and g["area"] > 100:
            neighbor_geoms.append((i, g))
            
    if not neighbor_geoms:
        # Fallback to nearest valid neighbor in same phase
        if k_div is not None:
            phase_pool = [kf for kf in kf_entries if (kf["kf_idx"] >= k_div if target_kf_idx >= k_div else kf["kf_idx"] < k_div) and kf["kf_idx"] != target_kf_idx and kf["geom"] is not None]
        else:
            phase_pool = [kf for kf in kf_entries if kf["kf_idx"] != target_kf_idx and kf["geom"] is not None]
            
        if phase_pool:
            closest_kf = min(phase_pool, key=lambda x: abs(x["kf_idx"] - target_kf_idx))
            neighbor_geoms.append((closest_kf["kf_idx"], closest_kf["geom"]))
        else:
            raise RuntimeError(f"No valid neighbor keyframes found around keyframe {target_kf_idx}")
        
    all_maj = [g["major_axis_length"] for _, g in neighbor_geoms]
    med_maj = float(np.median(all_maj))
    valid_neighbors = [(i, g) for i, g in neighbor_geoms if g["major_axis_length"] >= 0.70 * med_maj]
    
    if not valid_neighbors:
        valid_neighbors = neighbor_geoms
        
    closest_idx, ref_g = min(valid_neighbors, key=lambda x: abs(x[0] - target_kf_idx))
    ref_u_long = ref_g["u_long"]
    
    aligned_geoms = []
    for i, g in valid_neighbors:
        u_l = g["u_long"].copy()
        p_mx = g["p_max"]
        p_mn = g["p_min"]
        t1 = g["tip1"].copy()
        t2 = g["tip2"].copy()
        
        if np.dot(u_l, ref_u_long) < 0:
            u_l = -u_l
            p_mx, p_mn = -p_mn, -p_mx
            t1, t2 = t2, t1
            
        aligned_geoms.append({
            "kf_idx": i,
            "major_axis": g["major_axis_length"],
            "minor_axis": g["minor_axis_length"],
            "orientation": g["orientation"],
            "centroid": g["centroid"],
            "u_long": u_l,
            "p_max": p_mx,
            "p_min": p_mn,
            "tip1": t1,
            "tip2": t2
        })
        
    exp_maj = float(np.median([g["major_axis"] for g in aligned_geoms]))
    exp_min = float(np.median([g["minor_axis"] for g in aligned_geoms]))
    
    # Average unit vector
    u_longs = np.array([g["u_long"] for g in aligned_geoms], dtype=np.float64)
    exp_u_long = np.mean(u_longs, axis=0)
    exp_u_long = exp_u_long / np.linalg.norm(exp_u_long)
    exp_u_short = np.array([exp_u_long[1], -exp_u_long[0]], dtype=np.float64)
    
    # Centroid interpolation (distance-weighted)
    indices = np.array([g["kf_idx"] for g in aligned_geoms], dtype=np.float64)
    centroids = np.array([g["centroid"] for g in aligned_geoms], dtype=np.float64)
    
    if len(indices) >= 2:
        weights = 1.0 / (np.abs(indices - target_kf_idx) + 0.1)
        weights /= weights.sum()
        exp_centroid = np.sum(centroids * weights[:, None], axis=0)
    else:
        exp_centroid = centroids[0]
        
    exp_p_max = float(np.median([g["p_max"] for g in aligned_geoms]))
    exp_p_min = float(np.median([g["p_min"] for g in aligned_geoms]))
    
    # Expected tips
    exp_tip1 = exp_centroid + exp_p_max * exp_u_long
    exp_tip2 = exp_centroid + exp_p_min * exp_u_long
    
    return {
        "target_kf_idx": target_kf_idx,
        "film": target_film,
        "t": target_t,
        "local_cid": kf_entries[target_kf_idx]["local_cid"],
        "curr_geom": kf_entries[target_kf_idx]["geom"],
        "curr_mask": kf_entries[target_kf_idx]["mask"],
        "expected_major_axis": exp_maj,
        "expected_minor_axis": exp_min,
        "expected_centroid": exp_centroid,
        "expected_u_long": exp_u_long,
        "expected_u_short": exp_u_short,
        "expected_p_max": exp_p_max,
        "expected_p_min": exp_p_min,
        "expected_tip1": exp_tip1,
        "expected_tip2": exp_tip2,
        "num_valid_neighbors": len(aligned_geoms)
    }


def detect_truncated_poles(
    curr_mask: np.ndarray,
    exp_geom: Dict[str, Any],
    truncation_threshold_px: float = 10.0
) -> Dict[str, Any]:
    """
    Compares the current mask against interpolated expected geometry to identify
    which pole(s) are cut off in the missegmentation.
    """
    curr_geom = get_mask_geometry(curr_mask)
    if curr_geom is None:
        raise ValueError("Current mask is empty or invalid.")
        
    u_long_exp = exp_geom["expected_u_long"]
    u_short_exp = exp_geom["expected_u_short"]
    c_exp = exp_geom["expected_centroid"]
    
    cy, cx = curr_geom["centroid"]
    u_curr = curr_geom["u_long"].copy()
    if np.dot(u_curr, u_long_exp) < 0:
        u_curr = -u_curr
        
    ys, xs = np.where(curr_mask > 0)
    proj = (ys - cy) * u_curr[0] + (xs - cx) * u_curr[1]
    tip1_curr = np.array([cy, cx]) + np.max(proj) * u_curr
    tip2_curr = np.array([cy, cx]) + np.min(proj) * u_curr
    
    exp_tip1 = exp_geom["expected_tip1"]
    exp_tip2 = exp_geom["expected_tip2"]
    
    delta_tip1 = float(np.dot(exp_tip1 - tip1_curr, u_long_exp))
    delta_tip2 = float(np.dot(tip2_curr - exp_tip2, u_long_exp))
    
    tip1_truncated = delta_tip1 > truncation_threshold_px
    tip2_truncated = delta_tip2 > truncation_threshold_px
    
    return {
        "curr_tip1": tip1_curr,
        "curr_tip2": tip2_curr,
        "expected_tip1": exp_tip1,
        "expected_tip2": exp_tip2,
        "delta_tip1": delta_tip1,
        "delta_tip2": delta_tip2,
        "tip1_truncated": tip1_truncated,
        "tip2_truncated": tip2_truncated,
        "is_missegmented": tip1_truncated or tip2_truncated
    }


def recover_missegmented_cell(
    exp_dir: Path,
    film: str,
    t: int,
    local_cid: int,
    exp_geom: Dict[str, Any],
    truncation_threshold_px: float = 10.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Recovers the missing pole(s) of a missegmented cell by:
    1. Matching full overlapping segments in _seg.tif, or
    2. Fusing adjacent split label pieces in _seg.tif along the major axis, or
    3. Extruding a minor-axis width brush stroke along the major axis to the expected pole.
    """
    csv_path = exp_dir / film / f"TrackedCells_{film}" / f"cell_{local_cid}_masks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cell mask CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    rle_col = "rle_gfp" if "FL" in film and "rle_gfp" in df.columns and df["rle_gfp"].dropna().any() else "rle_bf"
    rows = df[df["time_point"] == t]
    if rows.empty:
        raise ValueError(f"No mask entry for timepoint t={t} in {csv_path}")
        
    rle_val = str(rows.iloc[0].get(rle_col, ""))
    H = int(df.iloc[0]["height"]) if "height" in df.columns else 2000
    W = int(df.iloc[0]["width"]) if "width" in df.columns else 2000
    curr_mask = validate_and_decode_rle(rle_val, H, W)
    
    trunc_info = detect_truncated_poles(curr_mask, exp_geom, truncation_threshold_px)
    
    seg_file = exp_dir / film / f"Masks_{film}" / f"{film}_t_{t:03d}_c_0_seg.tif"
    if not seg_file.exists():
        candidates = list((exp_dir / film / f"Masks_{film}").glob(f"*t*{t:03d}*_seg.tif"))
        seg_file = candidates[0] if candidates else None
        
    seg_img = tifffile.imread(str(seg_file)) if (seg_file and seg_file.exists()) else None
    
    recovered_mask = curr_mask.copy().astype(np.uint8)
    recovery_methods = []
    
    u_long = exp_geom["expected_u_long"]
    u_short = exp_geom["expected_u_short"]
    L_exp = exp_geom["expected_major_axis"]
    W_exp = exp_geom["expected_minor_axis"]
    radius = int(round(W_exp / 2.0))
    
    # Strategy 1: Check if _seg.tif has a full segment
    full_seg_found = False
    if seg_img is not None:
        cy, cx = exp_geom["expected_centroid"]
        r0 = max(0, int(cy - 120))
        c0 = max(0, int(cx - 120))
        r1 = min(H, int(cy + 120))
        c1 = min(W, int(cx + 120))
        crop_seg = seg_img[r0:r1, c0:c1]
        crop_curr = curr_mask[r0:r1, c0:c1]
        labels_near = np.unique(crop_seg)
        
        for l in labels_near:
            if l == 0:
                continue
            l_mask_crop = (crop_seg == l).astype(np.uint8)
            inter = int(np.sum(crop_curr & l_mask_crop))
            curr_area = int(np.sum(crop_curr))
            if curr_area > 0 and inter >= 0.70 * curr_area:
                l_props = regionprops(l_mask_crop)
                if l_props:
                    p_l = l_props[0]
                    if p_l.major_axis_length >= 0.85 * L_exp and abs(p_l.major_axis_length - L_exp) < 18:
                        recovered_mask = (seg_img == l).astype(np.uint8)
                        full_seg_found = True
                        recovery_methods.append(f"Full segment match in _seg.tif (Label {l})")
                        break
                        
    if not full_seg_found:
        # Tip 1 Recovery (+u_long pole)
        if trunc_info["tip1_truncated"]:
            delta1 = trunc_info["delta_tip1"]
            tip1_curr = trunc_info["curr_tip1"]
            tip1_exp = trunc_info["expected_tip1"]
            
            piece_found = False
            if seg_img is not None:
                for l in labels_near:
                    if l == 0: continue
                    l_mask_crop = (crop_seg == l).astype(np.uint8)
                    if np.sum(crop_curr & l_mask_crop) > 0: continue
                    l_props = regionprops(l_mask_crop)
                    if not l_props: continue
                    p_l = l_props[0]
                    c_global = np.array([p_l.centroid[0] + r0, p_l.centroid[1] + c0])
                    vec = c_global - tip1_curr
                    proj_long = float(np.dot(vec, u_long))
                    proj_short = float(abs(np.dot(vec, u_short)))
                    if 0 < proj_long < delta1 + 25.0 and proj_short < 15.0:
                        recovered_mask[r0:r1, c0:c1] |= l_mask_crop
                        piece_found = True
                        recovery_methods.append(f"Fused split segment (Label {l}) at Tip 1 (+{delta1:.1f} px)")
                        break
                        
            if not piece_found:
                num_steps = max(2, int(np.ceil(delta1)))
                step_points = np.linspace(0, delta1, num_steps)
                stroke_mask = np.zeros((H, W), dtype=np.uint8)
                for s in step_points:
                    pt = tip1_curr + s * u_long
                    py, px = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= py < H and 0 <= px < W:
                        cv2.circle(stroke_mask, (px, py), radius, 1, -1)
                recovered_mask = (recovered_mask | stroke_mask).astype(np.uint8)
                recovery_methods.append(f"Extruded brush stroke at Tip 1 (+{delta1:.1f} px, radius={radius} px)")
                
        # Tip 2 Recovery (-u_long pole)
        if trunc_info["tip2_truncated"]:
            delta2 = trunc_info["delta_tip2"]
            tip2_curr = trunc_info["curr_tip2"]
            tip2_exp = trunc_info["expected_tip2"]
            
            piece_found = False
            if seg_img is not None:
                for l in labels_near:
                    if l == 0: continue
                    l_mask_crop = (crop_seg == l).astype(np.uint8)
                    if np.sum(crop_curr & l_mask_crop) > 0: continue
                    l_props = regionprops(l_mask_crop)
                    if not l_props: continue
                    p_l = l_props[0]
                    c_global = np.array([p_l.centroid[0] + r0, p_l.centroid[1] + c0])
                    vec = tip2_curr - c_global
                    proj_long = float(np.dot(vec, u_long))
                    proj_short = float(abs(np.dot(vec, u_short)))
                    if 0 < proj_long < delta2 + 25.0 and proj_short < 15.0:
                        recovered_mask[r0:r1, c0:c1] |= l_mask_crop
                        piece_found = True
                        recovery_methods.append(f"Fused split segment (Label {l}) at Tip 2 (+{delta2:.1f} px)")
                        break
                        
            if not piece_found:
                num_steps = max(2, int(np.ceil(delta2)))
                step_points = np.linspace(0, delta2, num_steps)
                stroke_mask = np.zeros((H, W), dtype=np.uint8)
                for s in step_points:
                    pt = tip2_curr - s * u_long
                    py, px = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= py < H and 0 <= px < W:
                        cv2.circle(stroke_mask, (px, py), radius, 1, -1)
                recovered_mask = (recovered_mask | stroke_mask).astype(np.uint8)
                recovery_methods.append(f"Extruded brush stroke at Tip 2 (+{delta2:.1f} px, radius={radius} px)")
                
        recovered_mask = binary_closing(recovered_mask, disk(2)).astype(np.uint8)
        
    rec_geom = get_mask_geometry(recovered_mask)
    
    result_info = {
        "film": film,
        "t": t,
        "local_cid": local_cid,
        "truncation_info": trunc_info,
        "recovery_methods": recovery_methods,
        "before_geometry": get_mask_geometry(curr_mask),
        "after_geometry": rec_geom,
        "expected_geometry": exp_geom
    }
    
    return recovered_mask, result_info


def save_recovered_mask(
    exp_dir: Path,
    film: str,
    t: int,
    local_cid: int,
    recovered_mask: np.ndarray,
    backup: bool = True
) -> Dict[str, Any]:
    """
    Saves the recovered binary mask back to cell_<cid>_masks.csv and updates the
    Cellpose segmentation map (_seg.tif) for future training.
    """
    csv_path = exp_dir / film / f"TrackedCells_{film}" / f"cell_{local_cid}_masks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Masks CSV not found at {csv_path}")
        
    if backup:
        bak_path = csv_path.with_suffix(".csv.bak")
        if not bak_path.exists():
            shutil.copy2(csv_path, bak_path)
            
    df = pd.read_csv(csv_path)
    rle_col = "rle_gfp" if "FL" in film and "rle_gfp" in df.columns and df["rle_gfp"].dropna().any() else "rle_bf"
    if rle_col not in df.columns:
        df[rle_col] = ""
    df[rle_col] = df[rle_col].astype(object)
    
    new_rle = encode_mask_to_rle(recovered_mask)
    rows = df[df["time_point"] == t]
    if not rows.empty:
        df.at[rows.index[0], rle_col] = new_rle
    else:
        new_row = {
            "time_point": t,
            "width": recovered_mask.shape[1],
            "height": recovered_mask.shape[0],
            rle_col: new_rle
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df.to_csv(csv_path, index=False)
    
    seg_file = exp_dir / film / f"Masks_{film}" / f"{film}_t_{t:03d}_c_0_seg.tif"
    seg_updated = False
    if seg_file.exists():
        if backup:
            seg_bak = seg_file.with_suffix(".tif.bak")
            if not seg_bak.exists():
                shutil.copy2(seg_file, seg_bak)
                
        seg_img = tifffile.imread(str(seg_file))
        ys, xs = np.where(recovered_mask > 0)
        labels_present = seg_img[ys, xs]
        non_zero = labels_present[labels_present > 0]
        
        target_label = int(np.bincount(non_zero).argmax()) if len(non_zero) > 0 else int(seg_img.max() + 1)
        if target_label == 0:
            target_label = int(seg_img.max() + 1)
            
        seg_img[recovered_mask > 0] = target_label
        tifffile.imwrite(str(seg_file), seg_img, compression="zlib")
        seg_updated = True
        
    return {
        "status": "success",
        "csv_path": str(csv_path),
        "seg_file": str(seg_file) if seg_updated else None,
        "new_rle_length": len(new_rle)
    }


def generate_diagnostic_figure(
    exp_dir: Path,
    film: str,
    t: int,
    local_cid: int,
    cell_key: str,
    curr_mask: np.ndarray,
    recovered_mask: np.ndarray,
    exp_geom: Dict[str, Any],
    out_path: Path
) -> Path:
    """
    Generates a 3-panel before-and-after diagnostic overlay figure:
    1. Raw FL Microscopy Image
    2. Before (Truncated Mask outline in Red with missing pole indicator)
    3. After (Recovered Mask outline in Green, Major Axis, and Pole markers)
    """
    raw_path = exp_dir / film / f"Frames_{film}" / f"{film}_t_{t:03d}_c_0.tif"
    if not raw_path.exists():
        candidates = list((exp_dir / film / f"Frames_{film}").glob(f"*t*{t:03d}*.tif"))
        raw_path = candidates[0] if candidates else None
        
    if raw_path and raw_path.exists():
        raw_img = tifffile.imread(str(raw_path))
    else:
        raw_img = np.zeros(curr_mask.shape, dtype=np.uint16)
        
    cy_exp, cx_exp = exp_geom["expected_centroid"]
    pad = 75
    H, W = raw_img.shape[:2]
    ymin, ymax = max(0, int(cy_exp - pad)), min(H, int(cy_exp + pad))
    xmin, xmax = max(0, int(cx_exp - pad)), min(W, int(cx_exp + pad))
    
    crop_raw = raw_img[ymin:ymax, xmin:xmax]
    crop_curr = curr_mask[ymin:ymax, xmin:xmax]
    crop_rec = recovered_mask[ymin:ymax, xmin:xmax]
    
    p_low, p_high = np.percentile(raw_img, (1.0, 99.8))
    disp_raw = np.clip((crop_raw - p_low) / max(1.0, p_high - p_low), 0.0, 1.0)
    
    p_before = regionprops(curr_mask)[0]
    p_after = regionprops(recovered_mask)[0]
    
    tip1_exp = exp_geom["expected_tip1"]
    tip2_exp = exp_geom["expected_tip2"]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#181824")
    
    for ax in axes:
        ax.set_facecolor("#12121c")
        ax.tick_params(colors="#8888aa")
        for spine in ax.spines.values():
            spine.set_color("#33334d")
            
    # Panel 1: Raw FL
    axes[0].imshow(disp_raw, cmap="magma")
    axes[0].set_title(f"Raw FL Channel ({film} t={t})\nGlobal Cell: {cell_key}", color="#e6edf3", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 2: Before
    axes[1].imshow(disp_raw, cmap="gray")
    contours_before = find_contours(crop_curr, 0.5)
    for c in contours_before:
        axes[1].plot(c[:, 1], c[:, 0], color="#ff4d4d", linewidth=2.2, label="Missegmented Mask")
    axes[1].set_title(
        f"Before Recovery\nLength: {p_before.major_axis_length:.1f} px | Area: {int(p_before.area)} px",
        color="#ff7b72", fontsize=11, fontweight="bold"
    )
    axes[1].axis("off")
    
    # Panel 3: After
    axes[2].imshow(disp_raw, cmap="gray")
    contours_after = find_contours(crop_rec, 0.5)
    for c in contours_after:
        axes[2].plot(c[:, 1], c[:, 0], color="#3fb950", linewidth=2.2, label="Recovered Mask")
        
    p1_y, p1_x = tip1_exp[0] - ymin, tip1_exp[1] - xmin
    p2_y, p2_x = tip2_exp[0] - ymin, tip2_exp[1] - xmin
    axes[2].plot([p1_x, p2_x], [p1_y, p2_y], color="#e3b341", linestyle="--", linewidth=1.5, alpha=0.8, label="Expected Long Axis")
    axes[2].plot([p1_x], [p1_y], marker="o", color="#58a6ff", markersize=6.5, label="Pole 1")
    axes[2].plot([p2_x], [p2_y], marker="o", color="#d2a8ff", markersize=6.5, label="Pole 2")
    
    axes[2].set_title(
        f"After Pole Recovery\nLength: {p_after.major_axis_length:.1f} px (Expected: {exp_geom['expected_major_axis']:.1f}) | Area: {int(p_after.area)} px",
        color="#7ee787", fontsize=11, fontweight="bold"
    )
    axes[2].legend(loc="lower left", fontsize=8, facecolor="#1e1e2f", edgecolor="#444466", labelcolor="#c9d1d9")
    axes[2].axis("off")
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Automated single-cell pole recovery for missegmentations.")
    parser.add_argument("--exp_dir", type=str, default="/Volumes/X10 Pro/Movies/2026_08_28_M160", help="Path to experiment root.")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence name (e.g. 5_1_N1_F1).")
    parser.add_argument("--cell_key", type=str, required=True, help="Global cell ID or track key.")
    parser.add_argument("--film", type=str, required=True, help="Film name containing the missegmentation.")
    parser.add_argument("--t", type=int, required=True, help="Keyframe timepoint.")
    parser.add_argument("--window", type=int, default=5, help="Sliding window radius (±keyframes).")
    parser.add_argument("--save", action="store_true", help="Save the recovered mask to CSV and _seg.tif.")
    parser.add_argument("--output_fig", type=str, default=None, help="Output image path for diagnostic figure.")
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    print(f"\n[1/4] Interpolating geometry from ±{args.window} keyframe window for {args.cell_key} at {args.film} t={args.t}...")
    exp_geom = interpolate_expected_geometry(exp_dir, args.sequence, args.cell_key, args.film, args.t, args.window)
    print(f"      Expected Major Axis: {exp_geom['expected_major_axis']:.1f} px, Minor Axis: {exp_geom['expected_minor_axis']:.1f} px, Orientation Vector: ({exp_geom['expected_u_long'][0]:.3f}, {exp_geom['expected_u_long'][1]:.3f})")
    
    local_cid = exp_geom["local_cid"]
    print(f"\n[2/4] Detecting truncated poles for local cell #{local_cid}...")
    rec_mask, res_info = recover_missegmented_cell(exp_dir, args.film, args.t, local_cid, exp_geom)
    
    trunc = res_info["truncation_info"]
    print(f"      Tip 1 Truncation: {trunc['delta_tip1']:.1f} px ({'TRUNCATED' if trunc['tip1_truncated'] else 'OK'})")
    print(f"      Tip 2 Truncation: {trunc['delta_tip2']:.1f} px ({'TRUNCATED' if trunc['tip2_truncated'] else 'OK'})")
    print(f"      Recovery Action : {'; '.join(res_info['recovery_methods']) if res_info['recovery_methods'] else 'No change needed'}")
    
    b_geom = res_info["before_geometry"]
    a_geom = res_info["after_geometry"]
    print(f"\n[3/4] Geometry Comparison:")
    print(f"      Before -> Major Axis: {b_geom['major_axis_length']:.1f} px, Minor Axis: {b_geom['minor_axis_length']:.1f} px, Area: {b_geom['area']} px")
    print(f"      After  -> Major Axis: {a_geom['major_axis_length']:.1f} px, Minor Axis: {a_geom['minor_axis_length']:.1f} px, Area: {a_geom['area']} px")
    
    if args.output_fig:
        out_path = Path(args.output_fig)
        fig_saved = generate_diagnostic_figure(exp_dir, args.film, args.t, local_cid, args.cell_key, exp_geom["curr_mask"], rec_mask, exp_geom, out_path)
        print(f"\n[4/4] Saved diagnostic figure to {fig_saved}")
        
    if args.save:
        save_res = save_recovered_mask(exp_dir, args.film, args.t, local_cid, rec_mask, backup=True)
        print(f"      Updated mask saved to {save_res['csv_path']}")
        if save_res["seg_file"]:
            print(f"      Updated Cellpose training mask in {save_res['seg_file']}")


if __name__ == "__main__":
    main()
