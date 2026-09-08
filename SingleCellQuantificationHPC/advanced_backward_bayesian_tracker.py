#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_backward_bayesian_tracker.py

Refined Hard-EM Backward Bayesian Cell-Tracking System for Schizosaccharomyces pombe.

Key Principles:
1. Backward Tracking (t_end -> t_0): Eliminates forward daughter bifurcation ambiguity
   and leverages monotonic backward length decrease.
2. Single Division Constraint: Every global cell divides at most once (max_divisions = 1).
3. Hypothesis-Conditioned Backward Bisection & Hard-EM / ICM:
   - Evaluates competing division hypotheses H(k*) over candidate drop events.
   - For candidate hypothesis H(k_div), applies cleavage-plane bisection to post-division frames (k >= k_div)
     if fused/undersegmented, restoring clean daughter geometry.
   - Computes exact joint posterior P(H(k*) | O) using 4D Feature HMM + Multivariate LLR + Geometric Prior.
4. Border-Touching Exclusion: Automatically identifies and flags cells touching image borders (x, y within 5 px).
5. Comprehensive Validation Benchmark across Curated Tracks in 2026_08_28_M160.
"""

import os
import sys
import json
import pickle
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_closing
from scipy.stats import norm, multivariate_normal

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SingleCellQuantificationHPC"))

from SingleCellQuantificationHPC.ground_truth_corrector.schemas import validate_and_decode_rle
from SingleCellQuantificationHPC.classify_division_events import (
    TemporalDivisionHMM,
    MultivariateLLRClassifier,
    build_gold_standard_training_set,
    detect_candidate_drop_events,
    extract_11frame_feature_window
)
from SingleCellQuantificationHPC.recover_missegmented_poles import (
    get_mask_geometry,
    bisect_fused_mask,
    interpolate_expected_geometry,
    detect_truncated_poles,
    recover_missegmented_cell
)
from quant_helpers import (
    pattern_score_touching_circles,
    pattern_score_split_rectangles,
    transform_to_mn_space
)


def fast_extract_39_keyframes(
    exp_dir: Path,
    sequence: str,
    cell_key: str,
    track: List[int],
    films: List[str]
) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """
    Fast extraction of 39 keyframe geometries and border-touching status.
    Directly extracts morphological parameters from CSV masks without loading raw TIFFs.
    """
    if len(track) != len(films) or not all(cid > 0 for cid in track):
        return None, False

    keyframe_list = []
    global_kf_idx = 0
    touches_border_any = False

    for f_idx, film_name in enumerate(films):
        local_cid = track[f_idx]
        film_dir = exp_dir / film_name
        masks_csv = film_dir / f"TrackedCells_{film_name}" / f"cell_{local_cid}_masks.csv"
        if not masks_csv.exists():
            return None, False

        df = pd.read_csv(masks_csv)
        rle_col = "rle_gfp" if "FL" in film_name and "rle_gfp" in df.columns else "rle_bf"
        channel_type = "FL" if "FL" in film_name else "BF"
        kpts = [0, 50, 100] if channel_type == "FL" else [0, 20, 40]

        H = int(df.iloc[0]["height"]) if "height" in df.columns else 2000
        W = int(df.iloc[0]["width"]) if "width" in df.columns else 2000

        for t in kpts:
            row = df[df["time_point"] == t]
            mask = None
            geom = None
            if not row.empty and isinstance(row[rle_col].values[0], str) and len(str(row[rle_col].values[0]).strip()) > 0 and str(row[rle_col].values[0]).lower() != "nan":
                mask = validate_and_decode_rle(str(row[rle_col].values[0]), H, W)
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    if ys.min() <= 5 or ys.max() >= H - 5 or xs.min() <= 5 or xs.max() >= W - 5:
                        touches_border_any = True
                    geom = get_mask_geometry(mask)

            area = int(geom["area"]) if geom else 0
            maj = float(geom["major_axis_length"]) if geom else 0.0
            min_len = float(geom["minor_axis_length"]) if geom else 0.0

            keyframe_list.append({
                "kf_idx": global_kf_idx,
                "f_idx": f_idx,
                "film_name": film_name,
                "channel_type": channel_type,
                "t": t,
                "local_cid": local_cid,
                "mask": mask,
                "geom": geom,
                "area": area,
                "maj_axis": maj,
                "min_axis": min_len,
                "hourglass_score": 0.0,
                "strip_score": 0.0,
                "septum_contrast": 0.0,
            })
            global_kf_idx += 1

    return keyframe_list, touches_border_any


_TIFF_CACHE = {}

def get_cached_tiff_frame(frame_file: Path) -> Optional[np.ndarray]:
    path_str = str(frame_file)
    if path_str in _TIFF_CACHE:
        return _TIFF_CACHE[path_str]
    try:
        import tifffile
        img = tifffile.imread(path_str)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., 0]
        _TIFF_CACHE[path_str] = img
        return img
    except Exception:
        return None


def compute_keyframe_shape_and_contrast(
    exp_dir: Path,
    kf: Dict[str, Any]
) -> Tuple[float, float, float]:
    """Computes hourglass score, strip score, and septum contrast on-demand for a single keyframe."""
    mask = kf.get("mask")
    if mask is None or not mask.any():
        return 0.0, 0.0, 0.0

    geom = kf.get("geom")
    if not geom:
        return 0.0, 0.0, 0.0

    cy, cx = geom["centroid"]
    min_len = geom["minor_axis_length"]
    orientation = geom["orientation"]

    H, W = mask.shape
    minr = max(0, int(cy - 60))
    maxr = min(H, int(cy + 60))
    minc = max(0, int(cx - 60))
    maxc = min(W, int(cx + 60))

    crop_mask = mask[minr:maxr, minc:maxc]
    cy_crop, cx_crop = cy - minr, cx - minc

    theta = orientation
    vy, vx = np.cos(theta), -np.sin(theta)
    a_minor_half = min_len / 2.0
    mid1_rc = (cy_crop - a_minor_half * vy, cx_crop - a_minor_half * vx)
    mid2_rc = (cy_crop + a_minor_half * vy, cx_crop + a_minor_half * vx)

    # Shape scores
    try:
        pat_hg = pattern_score_touching_circles(crop_mask, crop_mask, mid1_rc, mid2_rc, side_px=50, stride=1)
        hg = float(pat_hg["best_score_norm"])
    except Exception:
        hg = 0.0

    try:
        pat_strip = pattern_score_split_rectangles(crop_mask, crop_mask, mid1_rc, mid2_rc, side_px=50, stride=1)
        strip = float(pat_strip["best_score_norm"])
    except Exception:
        strip = 0.0

    # Contrast
    sept_contrast = 0.0
    film_dir = exp_dir / kf["film_name"]
    t = kf["t"]
    frame_candidates = [
        film_dir / f"Frames_{kf['film_name']}" / f"{kf['film_name']}_t_{t:03d}_c_0.tif",
        film_dir / f"Frames_{kf['film_name']}" / f"{kf['film_name']}_t_{t:03d}_z_1_c_1.tif",
        film_dir / f"Frames_{kf['film_name']}" / f"{kf['film_name']}_t_{t:03d}_z_2_c_1.tif",
        film_dir / f"Frames_{kf['film_name']}" / f"{kf['film_name']}_t_{t:03d}_c_1.tif",
    ]
    frame_file = next((cand for cand in frame_candidates if cand.exists()), None)
    if frame_file:
        img = get_cached_tiff_frame(frame_file)
        if img is not None:
            try:
                crop_img = img[minr:maxr, minc:maxc].astype(np.float32)
                m_map, n_map = transform_to_mn_space(mid1_rc, mid2_rc, crop_mask, reflect=False)
                center_band = (np.abs(n_map) <= 4.0) & crop_mask
                cyt_band = (np.abs(n_map) > 8.0) & (np.abs(n_map) <= 22.0) & crop_mask
                if center_band.any() and cyt_band.any():
                    if kf["channel_type"] == "FL":
                        sept_contrast = float(np.mean(crop_img[center_band]) - np.mean(crop_img[cyt_band]))
                    else:
                        sept_contrast = float(np.abs(np.mean(crop_img[center_band]) - np.mean(crop_img[cyt_band])))
            except Exception:
                sept_contrast = 0.0

    return hg, strip, sept_contrast


class HardEMBackwardBayesianTracker:
    """
    Hard-EM / Hypothesis-Conditioned Backward Bayesian Cell Tracker.
    Evaluates global division hypotheses with phase-conditioned cleavage bisection,
    sister swap detection, and phase-aware pole recovery.
    """
    def __init__(
        self,
        exp_dir: Path,
        hmm_model: TemporalDivisionHMM,
        llr_model: MultivariateLLRClassifier,
        sigma_length: float = 12.0,
        sigma_drift: float = 15.0,
        p_misseg_prior: float = 0.05,
        p_swap_prior: float = 0.10,
        max_divisions: int = 1
    ):
        self.exp_dir = exp_dir
        self.hmm = hmm_model
        self.llr = llr_model
        self.sigma_length = sigma_length
        self.sigma_drift = sigma_drift
        self.p_misseg_prior = p_misseg_prior
        self.p_swap_prior = p_swap_prior
        self.max_divisions = max_divisions

    def track_cell(
        self,
        sequence: str,
        cell_key: str,
        track: List[int],
        films: List[str]
    ) -> Dict[str, Any]:
        """
        Executes backward Bayesian tracking on a single global cell track.
        Returns the optimal MAP hypothesis, division keyframe, action series, and diagnostics.
        """
        keyframes, touches_border = fast_extract_39_keyframes(self.exp_dir, sequence, cell_key, track, films)
        if not keyframes or len(keyframes) < 10:
            return {"status": "error", "message": "Failed to extract keyframes", "touches_border": touches_border}

        if touches_border:
            return {
                "sequence": sequence,
                "cell_key": cell_key,
                "is_border_touch": True,
                "status_qc": "bad",
                "is_dividing": False,
                "division_keyframe": None,
                "division_info": None,
                "map_posterior_prob": 0.0,
                "actions": ["BorderTouch"] * len(keyframes),
                "num_missegmentations": 0,
                "missegmentations": [],
                "num_sister_swaps": 0,
                "sister_swaps": [],
                "touches_border": True
            }

        K = len(keyframes)
        cand_drops = detect_candidate_drop_events(keyframes)

        # Compute on-demand shape & contrast features for candidate drops
        for k in cand_drops:
            for offset in range(-5, 6):
                idx = np.clip(k + offset, 0, K - 1)
                if keyframes[idx]["mask"] is not None and keyframes[idx]["hourglass_score"] == 0.0:
                    hg, strip, sc = compute_keyframe_shape_and_contrast(self.exp_dir, keyframes[idx])
                    keyframes[idx]["hourglass_score"] = hg
                    keyframes[idx]["strip_score"] = strip
                    keyframes[idx]["septum_contrast"] = sc

        div_probs = {}
        for k in range(1, K - 1):
            if k in cand_drops:
                win = extract_11frame_feature_window(keyframes, k)
                p_hmm = float(self.hmm.score_division_posterior(win))
                p_llr = float(np.atleast_1d(self.llr.predict_proba(win.flatten()))[0])
                div_probs[k] = {"p_comb": 0.5 * (p_hmm + p_llr), "p_hmm": p_hmm, "p_llr": p_llr}
            else:
                div_probs[k] = {"p_comb": 1e-6, "p_hmm": 1e-6, "p_llr": 1e-6}

        hypotheses = []

        # -----------------------------------------------------------------
        # Hypothesis 0: No division (k* = None)
        # -----------------------------------------------------------------
        log_p_null = 0.0
        for k in cand_drops:
            log_p_null += np.log(max(1.0 - div_probs[k]["p_comb"], 1e-6))

        maj_lengths = [kf["maj_axis"] for kf in keyframes if kf["maj_axis"] > 0]
        med_mother_len = float(np.median(maj_lengths)) if maj_lengths else 150.0

        null_actions = []
        for k in range(K):
            kf = keyframes[k]
            maj = kf["maj_axis"]
            if maj <= 0:
                action = "Empty"
                ll_geom = -10.0
            elif maj < 0.75 * med_mother_len:
                action = "Missegmentation"
                ll_geom = np.log(self.p_misseg_prior) - 0.5
            else:
                action = "Good"
                ll_geom = -((maj - med_mother_len)**2) / (2 * self.sigma_length**2)
            log_p_null += ll_geom
            null_actions.append(action)

        hypotheses.append({
            "k_div": None,
            "log_posterior": log_p_null,
            "p_division": 0.0,
            "actions": null_actions,
            "division_info": None,
            "refined_keyframes": keyframes
        })

        # -----------------------------------------------------------------
        # Hypotheses H_k: Division at candidate drop keyframe k*
        # -----------------------------------------------------------------
        for k_div in cand_drops:
            h_kfs = [dict(kf) for kf in keyframes]

            # Expected daughter length from post-division frames (excluding obvious mother segments)
            post_lens = [h_kfs[i]["maj_axis"] for i in range(k_div, K) if h_kfs[i]["maj_axis"] > 0 and h_kfs[i]["maj_axis"] < 130]
            exp_daughter_len = float(np.median(post_lens)) if post_lens else 85.0

            clean_post = [h_kfs[i]["geom"] for i in range(k_div, K) if h_kfs[i]["geom"] is not None and h_kfs[i]["maj_axis"] < 130]
            ref_c = clean_post[-1]["centroid"].copy() if clean_post else np.array([1000.0, 1000.0])
            ref_u = clean_post[-1]["u_long"].copy() if clean_post else None

            # Backward bisection of fused post-division masks (k >= k_div)
            num_bisected = 0
            for k in reversed(range(k_div, K)):
                kf = h_kfs[k]
                if kf["maj_axis"] > 1.35 * exp_daughter_len and kf["mask"] is not None:
                    bis = bisect_fused_mask(kf["mask"], target_centroid=ref_c, u_long=ref_u)
                    bg = get_mask_geometry(bis)
                    if bg:
                        kf["mask"] = bis
                        kf["geom"] = bg
                        kf["area"] = bg["area"]
                        kf["maj_axis"] = bg["major_axis_length"]
                        kf["min_axis"] = bg["minor_axis_length"]
                        ref_c = bg["centroid"].copy()
                        ref_u = bg["u_long"].copy()
                        num_bisected += 1
                elif kf["geom"] is not None:
                    ref_c = kf["geom"]["centroid"].copy()
                    ref_u = kf["geom"]["u_long"].copy()

            # Window scoring at k_div with bisected features
            win = extract_11frame_feature_window(h_kfs, k_div)
            p_hmm = float(self.hmm.score_division_posterior(win))
            p_llr = float(np.atleast_1d(self.llr.predict_proba(win.flatten()))[0])
            p_div_k = 0.5 * (p_hmm + p_llr)

            log_p_k = np.log(max(p_div_k, 1e-6))
            for other_k in cand_drops:
                if other_k != k_div:
                    log_p_k += np.log(max(1.0 - div_probs[other_k]["p_comb"], 1e-6))

            pre_lengths = [h_kfs[i]["maj_axis"] for i in range(max(0, k_div - 6), k_div) if h_kfs[i]["maj_axis"] > 0]
            exp_mother_len = float(np.median(pre_lengths)) if pre_lengths else 150.0

            k_actions = []
            for k in range(K):
                kf = h_kfs[k]
                maj = kf["maj_axis"]
                is_mother = (k < k_div)
                exp_len = exp_mother_len if is_mother else exp_daughter_len

                if maj <= 0:
                    action = "Empty"
                    ll_geom = -10.0
                elif maj < 0.75 * exp_len:
                    action = "Missegmentation"
                    ll_geom = np.log(self.p_misseg_prior) - 0.5
                else:
                    action = "Good"
                    ll_geom = -((maj - exp_len)**2) / (2 * self.sigma_length**2)

                log_p_k += ll_geom
                k_actions.append(action)

            hypotheses.append({
                "k_div": k_div,
                "log_posterior": log_p_k,
                "p_division": p_div_k,
                "actions": k_actions,
                "division_info": {
                    "film_name": keyframes[k_div]["film_name"],
                    "t": keyframes[k_div]["t"],
                    "kf_idx": k_div,
                    "p_hmm": p_hmm,
                    "p_llr": p_llr,
                    "area_drop": keyframes[k_div - 1]["area"] - keyframes[k_div]["area"],
                    "maj_drop": keyframes[k_div - 1]["maj_axis"] - keyframes[k_div]["maj_axis"]
                },
                "refined_keyframes": h_kfs
            })

        # -----------------------------------------------------------------
        # Select MAP Hypothesis
        # -----------------------------------------------------------------
        best_hyp = max(hypotheses, key=lambda h: h["log_posterior"])
        final_keyframes = best_hyp["refined_keyframes"]

        # Compute normalized posterior probabilities across all hypotheses
        all_logs = np.array([h["log_posterior"] for h in hypotheses])
        max_log = np.max(all_logs)
        probs = np.exp(all_logs - max_log)
        probs /= np.sum(probs)
        for i, h in enumerate(hypotheses):
            h["posterior_prob"] = float(probs[i])

        # -----------------------------------------------------------------
        # Detect Sister Swaps across Post-Division Film Boundaries
        # -----------------------------------------------------------------
        sister_swaps = []
        if best_hyp["k_div"] is not None:
            k_div = best_hyp["k_div"]
            f_div = final_keyframes[k_div]["f_idx"]

            for f in range(f_div, len(films) - 1):
                f1_name = films[f]
                f2_name = films[f + 1]

                kf1 = final_keyframes[(f + 1) * 3 - 1]
                kf2 = final_keyframes[(f + 1) * 3]

                g1 = kf1.get("geom")
                g2 = kf2.get("geom")
                if g1 and g2:
                    delta_c = g2["centroid"] - g1["centroid"]
                    dist = float(np.linalg.norm(delta_c))
                    u_long = g1["u_long"]
                    proj_long = float(abs(np.dot(delta_c, u_long)))
                    proj_short = float(abs(np.dot(delta_c, g1["u_short"])))
                    collinearity = proj_long / max(dist, 1e-6)
                    if dist >= 35.0 and collinearity >= 0.70 and proj_short <= 25.0:
                        sister_swaps.append({
                            "transition": f"{f1_name} -> {f2_name}",
                            "f_idx": f,
                            "displacement": round(dist, 1),
                            "collinearity": round(collinearity, 3),
                            "proj_short": round(proj_short, 1)
                        })

        misseg_frames = [
            {
                "kf_idx": k,
                "film_name": final_keyframes[k]["film_name"],
                "t": final_keyframes[k]["t"],
                "maj_axis": final_keyframes[k]["maj_axis"],
                "exp_axis": (exp_mother_len if (best_hyp["k_div"] is not None and k < best_hyp["k_div"]) else exp_daughter_len) if 'exp_mother_len' in locals() else med_mother_len
            }
            for k, act in enumerate(best_hyp["actions"]) if act == "Missegmentation"
        ]

        return {
            "sequence": sequence,
            "cell_key": cell_key,
            "is_border_touch": False,
            "is_dividing": best_hyp["k_div"] is not None,
            "division_keyframe": best_hyp["k_div"],
            "division_info": best_hyp["division_info"],
            "map_posterior_prob": best_hyp["posterior_prob"],
            "actions": best_hyp["actions"],
            "num_missegmentations": len(misseg_frames),
            "missegmentations": misseg_frames,
            "num_sister_swaps": len(sister_swaps),
            "sister_swaps": sister_swaps,
            "all_hypotheses": hypotheses,
            "touches_border": False
        }


# Alias for backward compatibility
BackwardBayesianTracker = HardEMBackwardBayesianTracker


def run_benchmark_on_curated_cells(
    exp_dir: Path,
    linkage_data: Dict[str, Any],
    cache_file: Path
) -> pd.DataFrame:
    """
    Runs the Hard-EM Backward Bayesian Tracker across all curated complete tracks
    and benchmarks against gold-standard division, sister swap, and missegmentation ground truth.
    """
    print("\n" + "=" * 75)
    print("RUNNING REFINED HARD-EM BACKWARD BAYESIAN TRACKER BENCHMARK")
    print("=" * 75)

    # 1. Load models
    g1_windows, g2_windows, all_records = build_gold_standard_training_set(exp_dir, linkage_data, cache_file)
    hmm = TemporalDivisionHMM()
    hmm.fit_emissions(g1_windows, g2_windows)
    llr = MultivariateLLRClassifier(reg_cov=1e-2)
    llr.fit(g1_windows.reshape(len(g1_windows), -1), g2_windows.reshape(len(g2_windows), -1))

    tracker = HardEMBackwardBayesianTracker(exp_dir, hmm, llr)

    # 2. Build ground truth map from curated records
    gt_division_map = {}
    gt_misseg_map = {}
    for r in all_records:
        key = (r["seq"], r["cell_key"])
        if r["group"] == "Group 1 (Division)":
            gt_division_map[key] = {
                "kf_idx": r["kf_idx"],
                "film_name": r["film_name"],
                "t": r["t"]
            }
        else:
            if key not in gt_misseg_map:
                gt_misseg_map[key] = []
            gt_misseg_map[key].append({
                "kf_idx": r["kf_idx"],
                "film_name": r["film_name"],
                "t": r["t"]
            })

    # 3. Evaluate across all curated cells
    seqs = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]
    benchmark_results = []

    total_cells = 0
    border_touch_cells = 0
    clean_cells = 0
    correct_div_class = 0
    exact_div_timing = 0
    within1_div_timing = 0

    t_start = time.time()

    for seq in seqs:
        with open(exp_dir / f"qc_{seq}.json") as f:
            qc = json.load(f)
        gcells = linkage_data[seq]["global_cells"]
        films = linkage_data[seq]["films"]

        for cell_key, track in gcells.items():
            if len(track) != len(films) or not all(cid > 0 for cid in track):
                continue

            st = "unreviewed"
            if cell_key in qc:
                st = qc[cell_key].get("status", "unreviewed") if isinstance(qc[cell_key], dict) else str(qc[cell_key])
            elif cell_key.split(f"{seq}_")[-1] in qc:
                subk = cell_key.split(f"{seq}_")[-1]
                st = qc[subk].get("status", "unreviewed") if isinstance(qc[subk], dict) else str(qc[subk])

            if st not in ("good", "corrected"):
                continue

            total_cells += 1
            if total_cells % 10 == 0 or total_cells == 1:
                print(f"[{seq}] Processing curated cell {total_cells} ({cell_key})...", flush=True)
            res = tracker.track_cell(seq, cell_key, track, films)

            if res.get("is_border_touch"):
                border_touch_cells += 1
                benchmark_results.append({
                    "sequence": seq,
                    "cell_key": cell_key,
                    "status_qc": st,
                    "is_border_touch": True,
                    "gt_is_dividing": (seq, cell_key) in gt_division_map,
                    "pred_is_dividing": False,
                    "div_classification_correct": None,
                    "gt_division_kf": gt_division_map.get((seq, cell_key), {}).get("kf_idx"),
                    "pred_division_kf": None,
                    "timing_error_kfs": None,
                    "num_missegmentations_found": 0,
                    "num_sister_swaps_found": 0,
                    "map_posterior_prob": 0.0
                })
                continue

            clean_cells += 1

            # Compare with ground truth
            gt_div = gt_division_map.get((seq, cell_key))
            is_div_gt = (gt_div is not None)
            is_div_pred = res["is_dividing"]

            div_match = (is_div_gt == is_div_pred)
            if div_match:
                correct_div_class += 1

            timing_err = None
            if is_div_gt and is_div_pred:
                timing_err = abs(res["division_keyframe"] - gt_div["kf_idx"])
                if timing_err == 0:
                    exact_div_timing += 1
                if timing_err <= 1:
                    within1_div_timing += 1

            benchmark_results.append({
                "sequence": seq,
                "cell_key": cell_key,
                "status_qc": st,
                "is_border_touch": False,
                "gt_is_dividing": is_div_gt,
                "pred_is_dividing": is_div_pred,
                "div_classification_correct": div_match,
                "gt_division_kf": gt_div["kf_idx"] if is_div_gt else None,
                "pred_division_kf": res["division_keyframe"],
                "timing_error_kfs": timing_err,
                "num_missegmentations_found": res["num_missegmentations"],
                "num_sister_swaps_found": res["num_sister_swaps"],
                "map_posterior_prob": round(res["map_posterior_prob"], 4)
            })

    t_elapsed = time.time() - t_start
    df = pd.DataFrame(benchmark_results)

    # 4. Print Comprehensive Benchmark Report
    print(f"\n--- HARD-EM BENCHMARK RESULTS SUMMARY ({t_elapsed:.2f} s) ---")
    print(f"Total Curated Cells Evaluated : {total_cells}")
    print(f"Border-Touching Cells Flagged : {border_touch_cells}")
    print(f"Clean Interior Cells Tested   : {clean_cells}")
    if clean_cells > 0:
        print(f"Division Classification Acc   : {correct_div_class}/{clean_cells} ({correct_div_class / clean_cells * 100:.2f}%)")

    clean_df = df[~df["is_border_touch"]]
    clean_gt_divs = clean_df[clean_df["gt_is_dividing"] == True]
    total_clean_div_gt = len(clean_gt_divs)

    print(f"Total True Divisions in Clean : {total_clean_div_gt}")
    if total_clean_div_gt > 0:
        print(f"Exact Division Timing Acc     : {exact_div_timing}/{total_clean_div_gt} ({exact_div_timing / total_clean_div_gt * 100:.2f}%)")
        print(f"Timing within ±1 Keyframe     : {within1_div_timing}/{total_clean_div_gt} ({within1_div_timing / total_clean_div_gt * 100:.2f}%)")

    total_misseg_found = clean_df["num_missegmentations_found"].sum()
    total_swaps_found = clean_df["num_sister_swaps_found"].sum()
    print(f"Total Missegmentations Detected: {total_misseg_found}")
    print(f"Total Sister Swaps Detected    : {total_swaps_found}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Refined Hard-EM Backward Bayesian Cell Tracker")
    parser.add_argument("--exp_dir", type=str, default="/Volumes/X10 Pro/Movies/2026_08_28_M160")
    parser.add_argument("--cache_path", type=str, default="SingleCellQuantificationHPC/scratch/gold_standard_11frame_features.pkl")
    parser.add_argument("--out_csv", type=str, default="SingleCellQuantificationHPC/scratch/bayesian_tracker_benchmark_307.csv")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    linkage_file = exp_dir / "sequence_linkage.json"
    with open(linkage_file) as f:
        linkage = json.load(f)

    cache_file = REPO_ROOT / args.cache_path
    df_results = run_benchmark_on_curated_cells(exp_dir, linkage, cache_file)

    out_csv = REPO_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n✓ Saved complete benchmark results to {out_csv}")


if __name__ == "__main__":
    main()
