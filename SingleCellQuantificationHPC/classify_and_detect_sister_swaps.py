#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_and_detect_sister_swaps.py

Two-step single-cell division & tracking QC pipeline:
1. Probabilistic Classification: Classifies area drop events as True Division (Group 1) vs. Missegmentation (Group 2)
   using 4D feature dynamic patterns (Area, Major Axis, Shape Templates, Septum Contrast) over an 11-timeframe window.
2. Sister Swap Identification: For Group 1 (True Division) tracks, detects if the tracker swapped between the two
   collinear sister cells across post-division film transitions (located along the mother's long axis, not the short axis).

Testing: Runs on a fresh batch of unreviewed complete tracks for verification in GTC.
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
from skimage.measure import label, regionprops

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SingleCellQuantificationHPC"))

from SingleCellQuantificationHPC.ground_truth_corrector.schemas import validate_and_decode_rle
from SingleCellQuantificationHPC.classify_division_events import (
    extract_39_keyframe_features,
    detect_candidate_drop_events,
    extract_11frame_feature_window,
    TemporalDivisionHMM,
    MultivariateLLRClassifier,
    build_gold_standard_training_set,
    classify_cell_candidate_events
)


def detect_sister_swaps(
    exp_dir: Path,
    sequence: str,
    cell_key: str,
    track: List[int],
    films: List[str],
    div_kf_idx: int
) -> List[Dict[str, Any]]:
    """
    Identifies sister swaps across post-division film transitions.
    Geometric criteria:
    - Pre-division mother geometry defines the intrinsic major axis (u_long) and minor axis (u_short).
    - True sisters are located side-by-side along the mother cell's long axis (collinear with u_long).
    - A sister swap occurs when the tracked daughter jumps between opposite poles (+u_long <-> -u_long)
      across a film boundary with high long-axis collinearity (|delta_long| / dist >= 0.70)
      and minimal short-axis deviation (|delta_short| <= 25 px).
    """
    f_pre = max(0, (div_kf_idx - 1) // 3)
    t_idx_pre = (div_kf_idx - 1) % 3
    film_pre = films[f_pre]
    kpts_pre = [0, 50, 100] if "FL" in film_pre else [0, 20, 40]
    t_pre = kpts_pre[t_idx_pre]

    cid_pre = track[f_pre]
    df_pre = pd.read_csv(exp_dir / film_pre / f"TrackedCells_{film_pre}" / f"cell_{cid_pre}_masks.csv")
    rle_col_pre = "rle_gfp" if "FL" in film_pre and "rle_gfp" in df_pre.columns else "rle_bf"
    H, W = int(df_pre.iloc[0]["height"]), int(df_pre.iloc[0]["width"])

    row_pre = df_pre[df_pre["time_point"] == t_pre]
    if row_pre.empty or not isinstance(row_pre[rle_col_pre].values[0], str):
        return []

    mask_mother = validate_and_decode_rle(str(row_pre[rle_col_pre].values[0]), H, W)
    rp_m = regionprops(label(mask_mother.astype(np.uint8)))
    if not rp_m:
        return []

    cy_m, cx_m = rp_m[0].centroid
    theta_m = rp_m[0].orientation
    maj_m = float(rp_m[0].major_axis_length)

    # Unit vectors for mother orientation (row, col)
    u_long = np.array([np.cos(theta_m), -np.sin(theta_m)])
    u_short = np.array([np.sin(theta_m), np.cos(theta_m)])

    # Collect centroids across all keyframes
    film_projections: Dict[int, List[Tuple[int, float, float]]] = {}

    for f_idx in range(len(films)):
        film_name = films[f_idx]
        local_cid = track[f_idx]
        if local_cid <= 0:
            continue

        csv_p = exp_dir / film_name / f"TrackedCells_{film_name}" / f"cell_{local_cid}_masks.csv"
        if not csv_p.exists():
            continue

        df = pd.read_csv(csv_p)
        rle_col = "rle_gfp" if "FL" in film_name and "rle_gfp" in df.columns else "rle_bf"
        kpts = [0, 50, 100] if "FL" in film_name else [0, 20, 40]

        projs = []
        for t in kpts:
            row = df[df["time_point"] == t]
            if not row.empty and isinstance(row[rle_col].values[0], str) and len(str(row[rle_col].values[0]).strip()) > 0:
                m = validate_and_decode_rle(str(row[rle_col].values[0]), H, W)
                rp = regionprops(label(m.astype(np.uint8)))
                if rp:
                    cy, cx = rp[0].centroid
                    d_vec = np.array([cy - cy_m, cx - cx_m])
                    p_long = float(np.dot(d_vec, u_long))
                    p_short = float(np.dot(d_vec, u_short))
                    projs.append((t, p_long, p_short))
        film_projections[f_idx] = projs

    swaps = []
    f_div = div_kf_idx // 3

    # Check consecutive film transitions from division film onwards
    for f_i in range(f_div, len(films) - 1):
        projs_A = film_projections.get(f_i, [])
        projs_B = film_projections.get(f_i + 1, [])
        if not projs_A or not projs_B:
            continue

        last_t_A, p_long_A, p_short_A = projs_A[-1]
        first_t_B, p_long_B, p_short_B = projs_B[0]

        delta_long = p_long_B - p_long_A
        delta_short = p_short_B - p_short_A
        dist = np.sqrt(delta_long**2 + delta_short**2)
        if dist < 12.0:
            continue

        long_collinearity = abs(delta_long) / max(dist, 1e-6)

        # 1. Sign flip across mother center (pole A to pole B) OR large shift along long axis
        sign_flip = bool(p_long_A * p_long_B < -15.0)
        large_long_shift = bool(abs(delta_long) >= 0.35 * maj_m)

        # 2. Long-axis collinearity vs lateral neighbor check
        is_collinear_long = bool(long_collinearity >= 0.70)
        is_minimal_lateral = bool(abs(delta_short) <= 25.0)

        if (sign_flip or large_long_shift) and is_collinear_long and is_minimal_lateral:
            swaps.append({
                "from_film_idx": f_i,
                "to_film_idx": f_i + 1,
                "from_film": films[f_i],
                "to_film": films[f_i + 1],
                "p_long_before": round(p_long_A, 1),
                "p_long_after": round(p_long_B, 1),
                "delta_long": round(delta_long, 1),
                "delta_short": round(delta_short, 1),
                "collinearity_ratio": round(long_collinearity, 3),
                "swap_type": "Sister Swap (Bipolar Long-Axis Jump)" if sign_flip else "Sister Shift (Long-Axis Displacement)"
            })

    return swaps


def main():
    parser = argparse.ArgumentParser(description="Classify division events and identify sister swaps across film transitions.")
    parser.add_argument("--movie_root", type=str, default="/Volumes/X10 Pro/Movies/2026_08_28_M160")
    parser.add_argument("--cache_path", type=str, default="SingleCellQuantificationHPC/scratch/gold_standard_11frame_features.pkl")
    parser.add_argument("--test_count", type=int, default=50, help="Number of unreviewed cells to test")
    parser.add_argument("--random_seed", type=int, default=101, help="Random seed for test cohort selection")
    parser.add_argument("--out_csv", type=str, default="SingleCellQuantificationHPC/scratch/unreviewed_division_and_sister_swaps.csv")
    args = parser.parse_args()

    exp_dir = Path(args.movie_root)
    linkage_file = exp_dir / "sequence_linkage.json"
    with open(linkage_file) as f:
        linkage = json.load(f)

    cache_file = REPO_ROOT / args.cache_path
    g1_windows, g2_windows, all_records = build_gold_standard_training_set(exp_dir, linkage, cache_file)

    # Train Probabilistic Models
    print("\nTraining Temporal Division HMM & Multivariate LLR...")
    hmm = TemporalDivisionHMM()
    hmm.fit_emissions(g1_windows, g2_windows)
    llr = MultivariateLLRClassifier(reg_cov=1e-2)
    llr.fit(g1_windows.reshape(len(g1_windows), -1), g2_windows.reshape(len(g2_windows), -1))
    print("✓ Models calibrated successfully.")

    # Select fresh test cohort of unreviewed complete tracks
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

    print(f"\nTotal unreviewed complete tracks pool: {len(unreviewed_tracks)}")
    np.random.seed(args.random_seed)
    selected_indices = np.random.choice(len(unreviewed_tracks), min(args.test_count, len(unreviewed_tracks)), replace=False)
    test_batch = [unreviewed_tracks[i] for i in selected_indices]

    results = []
    swap_results = []

    print(f"Running two-step evaluation on {len(test_batch)} test cells...")
    for idx, (seq, cell_key, track, films) in enumerate(test_batch):
        keyframes = extract_39_keyframe_features(exp_dir, seq, cell_key, track, films)
        if not keyframes:
            continue

        # Step 1: Classify division candidate events
        evals = classify_cell_candidate_events(keyframes, hmm, llr, div_threshold=0.50)

        div_event = next((ev for ev in evals if ev["status"] == "DIVISION"), None)

        for ev in evals:
            results.append({
                "sequence": seq,
                "cell_key": cell_key,
                "film_name": ev["film_name"],
                "keyframe_t": ev["t"],
                "kf_idx": ev["kf_idx"],
                "classification": ev["classification"],
                "p_division": round(ev["p_comb"], 4),
                "area_drop_pct": round(ev["area_drop"] * 100, 1),
                "maj_drop_pct": round(ev["maj_drop"] * 100, 1),
                "septum_contrast": round(ev["septum_contrast"], 2),
                "shape_score": round(ev["shape_score"], 3),
            })

        # Step 2: Sister Swap detection for Group 1 (True Division) cells
        if div_event:
            swaps = detect_sister_swaps(exp_dir, seq, cell_key, track, films, div_event["kf_idx"])
            for sw in swaps:
                swap_results.append({
                    "sequence": seq,
                    "cell_key": cell_key,
                    "division_film": div_event["film_name"],
                    "division_t": div_event["t"],
                    **sw
                })

    df_events = pd.DataFrame(results)
    df_swaps = pd.DataFrame(swap_results)

    out_csv_path = REPO_ROOT / args.out_csv
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_events.to_csv(out_csv_path, index=False)

    out_swaps_csv = out_csv_path.parent / "unreviewed_sister_swaps_identified.csv"
    df_swaps.to_csv(out_swaps_csv, index=False)

    print("\n=======================================================")
    print("STEP 1: DIVISION VS MISSEGMENTATION SUMMARY")
    print("=======================================================")
    if not df_events.empty:
        print(df_events["classification"].value_counts())
        print("\nCandidate Division Events:")
        print(df_events[df_events["classification"].str.contains("True Division")][["cell_key", "film_name", "keyframe_t", "p_division", "area_drop_pct", "septum_contrast"]])

    print("\n=======================================================")
    print("STEP 2: SISTER SWAP IDENTIFICATION SUMMARY")
    print("=======================================================")
    print(f"Total Group 1 True Division cells with Sister Swaps: {len(df_swaps['cell_key'].unique()) if not df_swaps.empty else 0}")
    print(f"Total Sister Swap transitions detected: {len(df_swaps)}")
    if not df_swaps.empty:
        print("\nIdentified Sister Swaps across Film Transitions:")
        print(df_swaps[["cell_key", "from_film", "to_film", "p_long_before", "p_long_after", "delta_long", "collinearity_ratio", "swap_type"]])


if __name__ == "__main__":
    main()
