#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_division_events.py

Probabilistic sliding-window classification of single-cell area drop events
as True Cell Division (Group 1) vs. Missegmentation (Group 2) using 4D feature
dynamic patterns over an 11-timeframe window ([-5, +5] keyframes).

Global Constraint: Every global cell can have at most ONE true cell division.
Training Data: 307 Curated Cells (Good + Corrected) in 2026_08_28_M160.
Testing: Unreviewed complete tracks for verification in GTC.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label, regionprops
from scipy.stats import norm, multivariate_normal

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SingleCellQuantificationHPC"))

from SingleCellQuantificationHPC.ground_truth_corrector.schemas import validate_and_decode_rle
from quant_helpers import (
    pattern_score_touching_circles,
    pattern_score_split_rectangles,
    transform_to_mn_space
)


def compute_mask_midpoints(mask_bool: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Any]:
    """Computes endpoints along the minor axis passing through centroid."""
    lbl = label(mask_bool.astype(np.uint8))
    props = regionprops(lbl)
    if not props:
        return (0.0, 0.0), (0.0, 0.0), None
    r = props[0]
    cy, cx = r.centroid
    theta = getattr(r, "orientation", 0.0) or 0.0
    vy, vx = np.cos(theta), -np.sin(theta)
    a_minor = getattr(r, "minor_axis_length", 0.0) / 2.0
    mid1_rc = (cy - a_minor * vy, cx - a_minor * vx)
    mid2_rc = (cy + a_minor * vy, cx + a_minor * vx)
    return mid1_rc, mid2_rc, r


def extract_39_keyframe_features(
    exp_dir: Path,
    sequence: str,
    global_cell_id: str,
    track: List[int],
    films: List[str],
    side_px: int = 50
) -> Optional[List[Dict[str, Any]]]:
    """
    Extracts the 39 keyframe 4D features for a single global cell track across 13 films.
    Keyframes: 3 per film (t=0, 50, 100 for FL; t=0, 20, 40 for BF).
    """
    if len(track) != len(films) or not all(cid > 0 for cid in track):
        return None

    keyframe_list = []
    global_kf_idx = 0

    for f_idx, film_name in enumerate(films):
        local_cid = track[f_idx]
        film_dir = exp_dir / film_name
        masks_csv = film_dir / f"TrackedCells_{film_name}" / f"cell_{local_cid}_masks.csv"
        if not masks_csv.exists():
            return None

        df = pd.read_csv(masks_csv)
        rle_col = "rle_gfp" if "FL" in film_name and "rle_gfp" in df.columns else "rle_bf"
        channel_type = "FL" if "FL" in film_name else "BF"
        kpts = [0, 50, 100] if channel_type == "FL" else [0, 20, 40]

        H = int(df.iloc[0]["height"])
        W = int(df.iloc[0]["width"])

        for t in kpts:
            row = df[df["time_point"] == t]
            if row.empty or not isinstance(row[rle_col].values[0], str) or len(str(row[rle_col].values[0]).strip()) == 0:
                # Missing or empty mask
                area, maj, min_len, hg, strip, sept_contrast = 0, 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                mask = validate_and_decode_rle(str(row[rle_col].values[0]), H, W)
                area = int(mask.sum())
                if area < 50:
                    area, maj, min_len, hg, strip, sept_contrast = 0, 0.0, 0.0, 0.0, 0.0, 0.0
                else:
                    mid1, mid2, rp = compute_mask_midpoints(mask)
                    if rp is None:
                        maj, min_len, hg, strip, sept_contrast = 0.0, 0.0, 0.0, 0.0, 0.0
                    else:
                        maj = float(rp.major_axis_length)
                        min_len = float(rp.minor_axis_length)

                        minr, minc_box, maxr, maxc_box = rp.bbox
                        pad = 12
                        minr = max(0, minr - pad)
                        minc_box = max(0, minc_box - pad)
                        maxr = min(H, maxr + pad)
                        maxc_box = min(W, maxc_box + pad)

                        crop_mask = mask[minr:maxr, minc_box:maxc_box]
                        cy, cx = rp.centroid
                        cy_crop, cx_crop = cy - minr, cx - minc_box
                        theta = getattr(rp, "orientation", 0.0) or 0.0
                        vy, vx = np.cos(theta), -np.sin(theta)
                        a_minor_half = min_len / 2.0
                        mid1_rc = (cy_crop - a_minor_half * vy, cx_crop - a_minor_half * vx)
                        mid2_rc = (cy_crop + a_minor_half * vy, cx_crop + a_minor_half * vx)

                        # Shape pattern scores
                        try:
                            pat_hg = pattern_score_touching_circles(
                                crop_mask, crop_mask, mid1_rc, mid2_rc, side_px=side_px, stride=1
                            )
                            hg = float(pat_hg["best_score_norm"])
                        except Exception:
                            hg = 0.0

                        try:
                            pat_strip = pattern_score_split_rectangles(
                                crop_mask, crop_mask, mid1_rc, mid2_rc, side_px=side_px, stride=1
                            )
                            strip = float(pat_strip["best_score_norm"])
                        except Exception:
                            strip = 0.0

                        # Septum Intensity Contrast
                        sept_contrast = 0.0
                        frame_candidates = [
                            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{t:03d}_c_0.tif",
                            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{t:03d}_z_1_c_1.tif",
                            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{t:03d}_z_2_c_1.tif",
                            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{t:03d}_c_1.tif",
                        ]
                        frame_file = next((cand for cand in frame_candidates if cand.exists()), None)
                        if frame_file:
                            try:
                                img = tifffile.imread(str(frame_file))
                                if img.ndim == 3 and img.shape[-1] in (3, 4):
                                    img = img[..., 0]
                                crop_img = img[minr:maxr, minc_box:maxc_box].astype(np.float32)
                                m_map, n_map = transform_to_mn_space(mid1_rc, mid2_rc, crop_mask, reflect=False)
                                center_band = (np.abs(n_map) <= 4.0) & crop_mask
                                cyt_band = (np.abs(n_map) > 8.0) & (np.abs(n_map) <= 22.0) & crop_mask
                                if center_band.any() and cyt_band.any():
                                    if channel_type == "FL":
                                        sept_contrast = float(np.mean(crop_img[center_band]) - np.mean(crop_img[cyt_band]))
                                    else:
                                        sept_contrast = float(np.abs(np.mean(crop_img[center_band]) - np.mean(crop_img[cyt_band])))
                            except Exception:
                                sept_contrast = 0.0

            keyframe_list.append({
                "kf_idx": global_kf_idx,
                "f_idx": f_idx,
                "film_name": film_name,
                "channel_type": channel_type,
                "t": t,
                "area": area,
                "maj_axis": maj,
                "min_axis": min_len,
                "hourglass_score": hg,
                "strip_score": strip,
                "septum_contrast": sept_contrast,
            })
            global_kf_idx += 1

    return keyframe_list


def detect_candidate_drop_events(
    keyframes: List[Dict[str, Any]],
    area_drop_thresh: float = 0.25
) -> List[int]:
    """
    Detects candidate division/missegmentation keyframe indices k where normalized area
    drops by >= area_drop_thresh relative to previous keyframe.
    """
    candidates = []
    for k in range(1, len(keyframes) - 1):
        curr_area = keyframes[k]["area"]
        prev_area = keyframes[k - 1]["area"]
        if prev_area < 500:
            continue
        
        # Channel-aware area scaling (BF segmentations are ~1.25x FL segmentations)
        prev_ch = keyframes[k - 1]["channel_type"]
        curr_ch = keyframes[k]["channel_type"]
        scale = 1.0
        if prev_ch == "FL" and curr_ch == "BF":
            scale = 1.25
        elif prev_ch == "BF" and curr_ch == "FL":
            scale = 0.80
        
        expected_prev_area = prev_area * scale
        drop_ratio = (expected_prev_area - curr_area) / max(expected_prev_area, 1)
        if drop_ratio >= area_drop_thresh:
            candidates.append(k)

    return candidates


def extract_11frame_feature_window(
    keyframes: List[Dict[str, Any]],
    k_center: int
) -> np.ndarray:
    """
    Extracts an 11-timeframe window [-5, +5] centered at k_center.
    Output: 11 x 4 array of normalized features:
      col 0: Normalized Area (relative to pre-drop baseline k-1)
      col 1: Normalized Major Axis Length (relative to pre-drop baseline k-1)
      col 2: Shape Score max(hourglass, strip)
      col 3: Septum Intensity Contrast (normalized)
    """
    window = np.zeros((11, 4), dtype=np.float32)
    N = len(keyframes)

    baseline_idx = max(0, k_center - 1)
    base_area = max(keyframes[baseline_idx]["area"], 1)
    base_maj = max(keyframes[baseline_idx]["maj_axis"], 1.0)
    base_ch = keyframes[baseline_idx]["channel_type"]

    for i, offset in enumerate(range(-5, 6)):
        idx = np.clip(k_center + offset, 0, N - 1)
        kf = keyframes[idx]

        # Channel normalization for area
        curr_ch = kf["channel_type"]
        ch_scale = 1.0
        if base_ch == "FL" and curr_ch == "BF":
            ch_scale = 1.25
        elif base_ch == "BF" and curr_ch == "FL":
            ch_scale = 0.80

        norm_area = (kf["area"] / ch_scale) / base_area
        norm_maj = kf["maj_axis"] / base_maj
        shape_score = max(kf["hourglass_score"], kf["strip_score"])
        sept_contrast = max(0.0, kf["septum_contrast"])

        window[i, 0] = norm_area
        window[i, 1] = norm_maj
        window[i, 2] = shape_score
        window[i, 3] = sept_contrast / 100.0  # scale contrast into [0, 1] range

    return window


# -------------------------------------------------------------------------
# Probabilistic Classifiers
# -------------------------------------------------------------------------

class MultivariateLLRClassifier:
    """
    Multivariate Gaussian Log-Likelihood Ratio classifier over the 11x4 (44D) window.
    Group 1: True Sustained Division
    Group 2: Transient Missegmentation / Glitch
    """
    def __init__(self, reg_cov: float = 1e-3):
        self.reg_cov = reg_cov
        self.mu1 = None
        self.cov1 = None
        self.mu2 = None
        self.cov2 = None
        self.prior1 = 0.5

    def fit(self, X1: np.ndarray, X2: np.ndarray):
        """Fits Gaussian distributions on Group 1 (N1 x 44) and Group 2 (N2 x 44)."""
        N1, D = X1.shape
        N2, _ = X2.shape
        self.prior1 = N1 / max(N1 + N2, 1)

        self.mu1 = np.mean(X1, axis=0)
        self.cov1 = np.cov(X1, rowvar=False) + np.eye(D) * self.reg_cov

        self.mu2 = np.mean(X2, axis=0)
        self.cov2 = np.cov(X2, rowvar=False) + np.eye(D) * self.reg_cov

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes P(Division | X) via Bayes rule with log-likelihoods."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        ll1 = multivariate_normal.logpdf(X, mean=self.mu1, cov=self.cov1, allow_singular=True)
        ll2 = multivariate_normal.logpdf(X, mean=self.mu2, cov=self.cov2, allow_singular=True)
        
        llr = (ll1 - ll2) + np.log(self.prior1 / max(1.0 - self.prior1, 1e-6))
        llr_clipped = np.clip(llr, -50.0, 50.0)
        p1 = 1.0 / (1.0 + np.exp(-llr_clipped))
        return p1


class TemporalDivisionHMM:
    """
    Hidden Markov Model with irreversible biological division state transitions:
    S0: Pre-division Mother (high area, high length)
    S1: Active Cytokinesis (division furrow, peak septum contrast, peak shape score)
    S2: Post-division Daughter (sustained half area, sustained half length)
    Smis: Missegmentation (transient half area/length that returns to S0)
    """
    def __init__(self):
        # 4 states: S0, S1, S2, Smis
        self.states = ["S0", "S1", "S2", "Smis"]
        self.n_states = 4
        self.means = np.zeros((4, 4))
        self.vars = np.ones((4, 4))

        # Transition matrix A: P(S_j | S_i)
        # S0 -> S0, S1, Smis
        # S1 -> S2 (irreversible)
        # S2 -> S2 (sustained daughter)
        # Smis -> S0 (transient recovery)
        self.A = np.array([
            [0.80, 0.10, 0.00, 0.10], # S0
            [0.00, 0.05, 0.95, 0.00], # S1 -> S2
            [0.00, 0.00, 1.00, 0.00], # S2 -> S2
            [0.85, 0.00, 0.00, 0.15], # Smis -> S0
        ], dtype=np.float64)

        self.pi = np.array([0.95, 0.01, 0.01, 0.03], dtype=np.float64)

    def fit_emissions(self, group1_windows: np.ndarray, group2_windows: np.ndarray):
        """
        Calibrates state emission Gaussian parameters (mean, var) from curated trajectories.
        """
        # S0: Pre-division (offsets -5 to -1) in Group 1
        s0_obs = group1_windows[:, :5, :].reshape(-1, 4)
        self.means[0] = np.mean(s0_obs, axis=0)
        self.vars[0] = np.var(s0_obs, axis=0) + 1e-4

        # S1: Active Division (offset 5 = center t=0) in Group 1
        s1_obs = group1_windows[:, 5, :]
        self.means[1] = np.mean(s1_obs, axis=0)
        self.vars[1] = np.var(s1_obs, axis=0) + 1e-4

        # S2: Post-division Daughter (offsets 6 to 10) in Group 1
        s2_obs = group1_windows[:, 6:, :].reshape(-1, 4)
        self.means[2] = np.mean(s2_obs, axis=0)
        self.vars[2] = np.var(s2_obs, axis=0) + 1e-4

        # Smis: Missegmentation glitch at center t=0 in Group 2
        smis_obs = group2_windows[:, 5, :]
        self.means[3] = np.mean(smis_obs, axis=0)
        self.vars[3] = np.var(smis_obs, axis=0) + 1e-4

    def log_emission(self, obs: np.ndarray) -> np.ndarray:
        """Computes log P(obs_t | S_j) for 4 states."""
        log_e = np.zeros((len(obs), self.n_states))
        for j in range(self.n_states):
            log_e[:, j] = np.sum(norm.logpdf(obs, loc=self.means[j], scale=np.sqrt(self.vars[j])), axis=1)
        return log_e

    def score_division_posterior(self, window_11x4: np.ndarray) -> float:
        """
        Runs forward-backward over the 11-frame window to compute the posterior
        probability P(State = S1 at t=5 | sequence).
        """
        T = len(window_11x4)
        log_e = self.log_emission(window_11x4)

        # Forward pass (in log-domain with logsumexp)
        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = np.log(self.pi + 1e-12) + log_e[0]
        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = np.logaddexp.reduce(log_alpha[t - 1] + np.log(self.A[:, j] + 1e-12)) + log_e[t, j]

        # Backward pass
        log_beta = np.zeros((T, self.n_states))
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = np.logaddexp.reduce(np.log(self.A[i, :] + 1e-12) + log_e[t + 1] + log_beta[t + 1])

        # Posterior gamma at center t=5 (active division site)
        log_gamma_5 = log_alpha[5] + log_beta[5]
        log_norm = np.logaddexp.reduce(log_gamma_5)
        gamma_5 = np.exp(log_gamma_5 - log_norm)

        # Posterior for True Division state S1
        p_div = float(gamma_5[1])
        return p_div


# -------------------------------------------------------------------------
# Track-Level Global Constraint Solver
# -------------------------------------------------------------------------

def classify_cell_candidate_events(
    keyframes: List[Dict[str, Any]],
    hmm_model: TemporalDivisionHMM,
    llr_model: MultivariateLLRClassifier,
    div_threshold: float = 0.50
) -> List[Dict[str, Any]]:
    """
    Evaluates all candidate drop events in a cell track, enforcing the
    at-most-one-division global track constraint.
    """
    candidates = detect_candidate_drop_events(keyframes)
    if not candidates:
        return []

    event_evals = []
    for k in candidates:
        win = extract_11frame_feature_window(keyframes, k)
        p_hmm = hmm_model.score_division_posterior(win)
        p_llr = float(np.atleast_1d(llr_model.predict_proba(win.flatten()))[0])

        # Ensemble probability
        p_comb = 0.5 * (p_hmm + p_llr)

        kf = keyframes[k]
        event_evals.append({
            "kf_idx": k,
            "film_name": kf["film_name"],
            "t": kf["t"],
            "p_hmm": p_hmm,
            "p_llr": p_llr,
            "p_comb": p_comb,
            "area_drop": (keyframes[k-1]["area"] - kf["area"]) / max(keyframes[k-1]["area"], 1),
            "maj_drop": (keyframes[k-1]["maj_axis"] - kf["maj_axis"]) / max(keyframes[k-1]["maj_axis"], 1.0),
            "septum_contrast": kf["septum_contrast"],
            "shape_score": max(kf["hourglass_score"], kf["strip_score"]),
        })

    # Apply Global Constraint: At most 1 True Division per global cell
    best_cand = max(event_evals, key=lambda x: x["p_comb"])

    for ev in event_evals:
        if ev["kf_idx"] == best_cand["kf_idx"] and ev["p_comb"] >= div_threshold:
            ev["classification"] = "True Division (Group 1)"
            ev["status"] = "DIVISION"
        else:
            ev["classification"] = "Missegmentation (Group 2)"
            ev["status"] = "MISSEGMENTATION"

    return event_evals


# -------------------------------------------------------------------------
# Dataset Training & Benchmarking Pipeline
# -------------------------------------------------------------------------

def build_gold_standard_training_set(
    exp_dir: Path,
    linkage: Dict[str, Any],
    cache_path: Path
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts all candidate events from the 307 curated cells and partitions
    into Group 1 (True Division) and Group 2 (Missegmentation).
    """
    if cache_path.exists():
        print(f"Loading cached gold standard features from {cache_path}...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["group1_windows"], data["group2_windows"], data["all_records"]

    print("Extracting 39-keyframe features across all 307 curated cells...")
    seqs = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]
    
    curated_tracks = []
    for seq in seqs:
        with open(exp_dir / f"qc_{seq}.json") as f:
            qc = json.load(f)
        gcells = linkage[seq]["global_cells"]
        films = linkage[seq]["films"]
        for k, v in qc.items():
            if isinstance(v, dict) and v.get("status") in ["good", "corrected"]:
                full_k = f"{seq}_{k}" if not k.startswith(f"{seq}_") else k
                if full_k in gcells:
                    track = gcells[full_k]
                    if all(cid > 0 for cid in track) and len(track) == len(films):
                        curated_tracks.append((seq, full_k, track, films))

    print(f"Total curated tracks to process: {len(curated_tracks)}")

    group1_windows = []
    group2_windows = []
    all_records = []

    for i, (seq, cell_key, track, films) in enumerate(curated_tracks):
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(curated_tracks)} tracks...")

        keyframes = extract_39_keyframe_features(exp_dir, seq, cell_key, track, films)
        if not keyframes:
            continue

        cands = detect_candidate_drop_events(keyframes)
        for k in cands:
            win = extract_11frame_feature_window(keyframes, k)
            
            # Ground truth criteria for Group 1 vs Group 2 in curated cells:
            post_area_ratio = float(np.mean(win[6:, 0]))
            post_maj_ratio = float(np.mean(win[6:, 1]))
            immediate_bounce = bool((win[6, 0] > 0.85) or (win[6, 1] > 0.85))

            is_group1 = (post_area_ratio <= 0.70) and (post_maj_ratio <= 0.80) and (not immediate_bounce)

            record = {
                "seq": seq,
                "cell_key": cell_key,
                "kf_idx": k,
                "film_name": keyframes[k]["film_name"],
                "t": keyframes[k]["t"],
                "group": "Group 1 (Division)" if is_group1 else "Group 2 (Missegmentation)",
                "post_area_ratio": post_area_ratio,
                "post_maj_ratio": post_maj_ratio,
            }
            all_records.append(record)

            if is_group1:
                group1_windows.append(win)
            else:
                group2_windows.append(win)

    g1_arr = np.array(group1_windows, dtype=np.float32)
    g2_arr = np.array(group2_windows, dtype=np.float32)

    print(f"\nGold Standard Extraction Complete:")
    print(f"  Group 1 (True Divisions): {len(g1_arr)} events")
    print(f"  Group 2 (Missegmentations): {len(g2_arr)} events")

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "group1_windows": g1_arr,
            "group2_windows": g2_arr,
            "all_records": all_records
        }, f)

    return g1_arr, g2_arr, all_records


def main():
    parser = argparse.ArgumentParser(description="Probabilistic Division vs Missegmentation Classifier")
    parser.add_argument("--movie_root", type=str, default="/Volumes/X10 Pro/Movies/2026_08_28_M160")
    parser.add_argument("--cache_path", type=str, default="SingleCellQuantificationHPC/scratch/gold_standard_11frame_features.pkl")
    parser.add_argument("--test_unreviewed_count", type=int, default=50, help="Number of unreviewed cells to classify")
    parser.add_argument("--out_csv", type=str, default="SingleCellQuantificationHPC/scratch/unreviewed_division_classification.csv")
    args = parser.parse_args()

    exp_dir = Path(args.movie_root)
    linkage_file = exp_dir / "sequence_linkage.json"
    with open(linkage_file) as f:
        linkage = json.load(f)

    cache_file = REPO_ROOT / args.cache_path
    g1_windows, g2_windows, all_records = build_gold_standard_training_set(exp_dir, linkage, cache_file)

    # 1. Train Temporal HMM
    print("\nTraining Temporal Division HMM...")
    hmm = TemporalDivisionHMM()
    hmm.fit_emissions(g1_windows, g2_windows)
    print("✓ HMM trained successfully.")

    # 2. Train Multivariate LLR Classifier
    print("Training Multivariate Dynamic LLR Classifier...")
    llr = MultivariateLLRClassifier(reg_cov=1e-2)
    llr.fit(g1_windows.reshape(len(g1_windows), -1), g2_windows.reshape(len(g2_windows), -1))
    print("✓ Multivariate LLR trained successfully.")

    # 3. Test on Unreviewed Complete Tracks
    print(f"\nSelecting {args.test_unreviewed_count} unreviewed complete tracks for testing...")
    seqs = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]
    unreviewed_tracks = []

    for seq in seqs:
        with open(exp_dir / f"qc_{seq}.json") as f:
            qc = json.load(f)
        gcells = linkage[seq]["global_cells"]
        films = linkage[seq]["films"]
        for k, track in gcells.items():
            if all(cid > 0 for cid in track) and len(track) == len(films):
                st = "unreviewed"
                if k in qc:
                    st = qc[k].get("status", "unreviewed") if isinstance(qc[k], dict) else str(qc[k])
                elif k.split(f"{seq}_")[-1] in qc:
                    subk = k.split(f"{seq}_")[-1]
                    st = qc[subk].get("status", "unreviewed") if isinstance(qc[subk], dict) else str(qc[subk])
                if st == "unreviewed":
                    unreviewed_tracks.append((seq, k, track, films))

    print(f"Total unreviewed complete tracks pool: {len(unreviewed_tracks)}")
    np.random.seed(42)
    selected_indices = np.random.choice(len(unreviewed_tracks), min(args.test_unreviewed_count, len(unreviewed_tracks)), replace=False)
    test_batch = [unreviewed_tracks[i] for i in selected_indices]

    results = []
    print(f"Running sliding-window probabilistic classification on {len(test_batch)} test cells...")
    for idx, (seq, cell_key, track, films) in enumerate(test_batch):
        keyframes = extract_39_keyframe_features(exp_dir, seq, cell_key, track, films)
        if not keyframes:
            continue
        
        evals = classify_cell_candidate_events(keyframes, hmm, llr, div_threshold=0.50)
        for ev in evals:
            results.append({
                "sequence": seq,
                "cell_key": cell_key,
                "film_name": ev["film_name"],
                "keyframe_t": ev["t"],
                "kf_idx": ev["kf_idx"],
                "classification": ev["classification"],
                "p_division_combined": round(ev["p_comb"], 4),
                "p_division_hmm": round(ev["p_hmm"], 4),
                "p_division_llr": round(ev["p_llr"], 4),
                "area_drop_pct": round(ev["area_drop"] * 100, 1),
                "maj_drop_pct": round(ev["maj_drop"] * 100, 1),
                "septum_contrast": round(ev["septum_contrast"], 2),
                "shape_score": round(ev["shape_score"], 3),
            })

    df_out = pd.DataFrame(results)
    out_csv_path = REPO_ROOT / args.out_csv
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv_path, index=False)
    print(f"\n✓ Saved unreviewed classification results to {out_csv_path}")

    # Summary
    if not df_out.empty:
        print("\n--- CLASSIFICATION SUMMARY ON TEST UNREVIEWED CELLS ---")
        print(df_out["classification"].value_counts())
        print("\nSample Classified Events:")
        print(df_out[["cell_key", "film_name", "keyframe_t", "classification", "p_division_combined", "area_drop_pct", "septum_contrast"]].head(15))


if __name__ == "__main__":
    main()
