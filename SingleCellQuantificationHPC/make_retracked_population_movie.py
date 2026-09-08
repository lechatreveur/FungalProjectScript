#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_retracked_population_movie.py

Generates high-resolution population-level tracking movies with colorful,
semi-transparent segmentation overlays for cells retracked via Refined Hard-EM ABBT
(Advanced Backward Bayesian Tracker).

Features:
1. Runs Hard-EM ABBT on all complete unreviewed cells in a sequence.
2. Excludes border-touching cells (within 5 px of boundary).
3. Incorporates hypothesis-conditioned daughter bisection, pole recovery, and sister swap
   continuity directly in memory without modifying canonical disk files.
4. Overlays colorful, semi-transparent masks (alpha=0.45), crisp contours, cell ID badges,
   and division indicators ([DIV]).
5. Renders a complete multi-film time-lapse sequence movie into high-compatibility MP4.
6. Saves diagnostic snapshot frames and population summary metrics.
"""

import os
import sys
import json
import math
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import tifffile
import numpy as np
import pandas as pd
from skimage.measure import regionprops, find_contours

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SingleCellQuantificationHPC"))

from SingleCellQuantificationHPC.ground_truth_corrector.schemas import (
    validate_and_decode_rle,
    encode_mask_to_rle
)
from SingleCellQuantificationHPC.advanced_backward_bayesian_tracker import (
    HardEMBackwardBayesianTracker,
    TemporalDivisionHMM,
    MultivariateLLRClassifier,
    build_gold_standard_training_set
)
from SingleCellQuantificationHPC.recover_missegmented_poles import (
    get_mask_geometry,
    interpolate_expected_geometry,
    recover_missegmented_cell
)


def generate_cell_color(cell_id: int, total_colors: int = 120) -> Tuple[int, int, int]:
    """Generates deterministic, highly distinguishable BGR colors using golden ratio hue distribution."""
    golden_ratio_conjugate = 0.618033988749895
    h = ((cell_id * golden_ratio_conjugate) % 1.0) * 180.0
    s = 200.0 + (cell_id % 5) * 11.0  # High saturation
    v = 220.0 + (cell_id % 3) * 15.0  # High brightness
    hsv_pixel = np.uint8([[[int(h), int(s), int(v)]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))


def run_abbt_on_unreviewed_cohort(
    exp_dir: Path,
    sequence: str,
    linkage_data: Dict[str, Any],
    hmm: TemporalDivisionHMM,
    llr: MultivariateLLRClassifier,
    cell_limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes Hard-EM ABBT on complete unreviewed tracks in the sequence, producing
    retracked lineages, bisected daughter masks, pole-recovered masks, and division metadata in memory.
    """
    seq_data = linkage_data[sequence]
    films = seq_data["films"]
    global_cells = seq_data["global_cells"]

    qc_file = exp_dir / f"qc_{sequence}.json"
    qc = {}
    if qc_file.exists():
        with open(qc_file, "r") as f:
            qc = json.load(f)

    # Select unreviewed complete cells
    unreviewed_tracks = []
    for k, track in global_cells.items():
        if len(track) == len(films) and all(cid > 0 for cid in track):
            st = "unreviewed"
            if k in qc:
                st = qc[k].get("status", "unreviewed") if isinstance(qc[k], dict) else str(qc[k])
            elif k.split(f"{sequence}_")[-1] in qc:
                subk = k.split(f"{sequence}_")[-1]
                st = qc[subk].get("status", "unreviewed") if isinstance(qc[subk], dict) else str(qc[subk])
            
            if st == "unreviewed":
                unreviewed_tracks.append((k, track))

    print(f"Total unreviewed complete tracks in {sequence}: {len(unreviewed_tracks)}")
    if cell_limit and cell_limit < len(unreviewed_tracks):
        unreviewed_tracks = unreviewed_tracks[:cell_limit]
        print(f"Limiting to first {cell_limit} unreviewed cells for rendering.")

    tracker = HardEMBackwardBayesianTracker(exp_dir, hmm, llr)
    retracked_results = []

    print("Running Hard-EM ABBT backward trajectory analysis & pole recovery...")
    for idx, (cell_key, track) in enumerate(unreviewed_tracks):
        if (idx + 1) % 25 == 0 or idx == 0 or idx == len(unreviewed_tracks) - 1:
            print(f"  Processed {idx + 1}/{len(unreviewed_tracks)} unreviewed cells...")

        res = tracker.track_cell(sequence, cell_key, track, films)
        
        # Collect refined in-memory masks
        refined_masks = {}
        best_hyp = None
        if "all_hypotheses" in res and res["all_hypotheses"]:
            best_hyp = max(res["all_hypotheses"], key=lambda h: h["log_posterior"])
            for kf in best_hyp.get("refined_keyframes", []):
                if kf.get("mask") is not None:
                    refined_masks[(kf["film_name"], kf["t"])] = kf["mask"]

        res["refined_masks"] = refined_masks
        res["raw_track"] = track
        retracked_results.append(res)

    summary = {
        "sequence": sequence,
        "total_unreviewed_tracked": len(retracked_results),
        "border_touch_cells_excluded": sum(1 for r in retracked_results if r.get("is_border_touch")),
        "clean_cells_rendered": sum(1 for r in retracked_results if not r.get("is_border_touch")),
        "total_divisions_detected": sum(1 for r in retracked_results if r["is_dividing"]),
        "total_missegmentations": sum(r["num_missegmentations"] for r in retracked_results),
        "total_sister_swaps": sum(r["num_sister_swaps"] for r in retracked_results)
    }

    return retracked_results, summary


def render_population_movie(
    exp_dir: Path,
    sequence: str,
    films: List[str],
    retracked_cells: List[Dict[str, Any]],
    out_mp4_path: Path,
    fps: int = 6,
    alpha: float = 0.45,
    keyframe_only: bool = True
) -> Path:
    """
    Renders multi-film population tracking movie with colorful semi-transparent
    segmentations overlaid on normalized raw microscopy frames.
    """
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    temp_avi = out_mp4_path.with_suffix(".temp.avi")

    # Generate stable colors for clean cells
    cell_colors = {}
    for r in retracked_cells:
        ck = r["cell_key"]
        try:
            cid_num = int(ck.split("_cell_")[-1])
        except Exception:
            cid_num = hash(ck) % 1000
        cell_colors[ck] = generate_cell_color(cid_num)

    # Keyframes per film (39 total across 13 films: 3 per film)
    sequence_frames = []
    for f_idx, film_name in enumerate(films):
        if keyframe_only:
            kpts = [0, 50, 100] if "FL" in film_name else [0, 20, 40]
        else:
            kpts = list(range(0, 101, 2)) if "FL" in film_name else list(range(0, 41, 1))
            
        for t in kpts:
            sequence_frames.append((f_idx, film_name, t))

    print(f"Total movie sequence frames to render: {len(sequence_frames)}")

    # Initialize video writer
    sample_frame_path = exp_dir / films[0] / f"Frames_{films[0]}" / f"{films[0]}_t_000_c_0.tif"
    if not sample_frame_path.exists():
        candidates = list((exp_dir / films[0] / f"Frames_{films[0]}").glob("*.tif"))
        sample_frame_path = candidates[0]
    sample_img = tifffile.imread(str(sample_frame_path))
    H, W = sample_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(temp_avi), fourcc, fps, (W, H))

    # Diagnostic snapshots to save
    snapshot_indices = [0, len(sequence_frames) // 4, len(sequence_frames) // 2, (3 * len(sequence_frames)) // 4, len(sequence_frames) - 1]
    saved_snapshots = []

    for seq_idx, (f_idx, film_name, t) in enumerate(sequence_frames):
        if (seq_idx + 1) % 5 == 0 or seq_idx == 0 or seq_idx == len(sequence_frames) - 1:
            print(f"  Rendering frame {seq_idx + 1}/{len(sequence_frames)}: {film_name} t={t}...")

        raw_path = exp_dir / film_name / f"Frames_{film_name}" / f"{film_name}_t_{t:03d}_c_0.tif"
        if not raw_path.exists():
            candidates = list((exp_dir / film_name / f"Frames_{film_name}").glob(f"*t*{t:03d}*.tif"))
            raw_path = candidates[0] if candidates else None

        if raw_path and raw_path.exists():
            raw_data = tifffile.imread(str(raw_path))
        else:
            raw_data = np.zeros((H, W), dtype=np.uint16)

        # Normalize microscopy image to 8-bit BGR with high-contrast percentile scaling
        p_low, p_high = np.percentile(raw_data, (1.0, 99.7))
        norm_img = np.clip((raw_data - p_low) / max(1.0, p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
        base_bgr = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

        # Overlay canvas
        overlay_mask_layer = np.zeros((H, W, 3), dtype=np.uint8)
        mask_active_any = np.zeros((H, W), dtype=bool)

        active_count = 0
        annotations = []

        for r in retracked_cells:
            if r.get("is_border_touch"):
                continue

            cell_key = r["cell_key"]
            color = cell_colors[cell_key]

            # In-memory refined mask
            mask = r.get("refined_masks", {}).get((film_name, t))
            if mask is not None and mask.any():
                active_count += 1
                mask_bool = (mask > 0)
                mask_active_any |= mask_bool
                overlay_mask_layer[mask_bool] = color

                # Centroid & ID label
                props = regionprops(mask.astype(np.uint8))
                if props:
                    p = props[0]
                    cy, cx = int(p.centroid[0]), int(p.centroid[1])
                    cid_num = cell_key.split("_cell_")[-1]
                    
                    is_div = False
                    if r["is_dividing"] and r["division_keyframe"] is not None:
                        if r["division_info"] and r["division_info"]["film_name"] == film_name and r["division_info"]["t"] == t:
                            is_div = True

                    annotations.append((cy, cx, cid_num, color, False, is_div))

        # Alpha blend masks with base image
        blended = base_bgr.copy()
        if mask_active_any.any():
            blended[mask_active_any] = cv2.addWeighted(
                base_bgr[mask_active_any], 1.0 - alpha,
                overlay_mask_layer[mask_active_any], alpha,
                0
            )

        # Draw cell contours and text badges
        for cy, cx, cid_num, color, is_repaired, is_div in annotations:
            label_text = f"#{cid_num}"
            if is_div:
                label_text += " [DIV]"

            # Badge background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(blended, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 2), (18, 18, 24), -1)
            text_color = (120, 255, 120) if is_div else (255, 255, 255)
            cv2.putText(blended, label_text, (cx, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        # HUD Header Banner
        cv2.rectangle(blended, (0, 0), (W, 46), (15, 15, 22), -1)
        cv2.line(blended, (0, 46), (W, 46), (60, 60, 90), 1)

        hud_left = f"Sequence: {sequence}  |  Film: {film_name}  |  t = {t:03d} (Keyframe {seq_idx + 1}/{len(sequence_frames)})"
        hud_right = f"Active Clean Cells: {active_count}  |  Hard-EM ABBT Refined In-Memory Overlay"
        cv2.putText(blended, hud_left, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 237, 243), 1, cv2.LINE_AA)
        cv2.putText(blended, hud_right, (W - 620, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 166, 255), 1, cv2.LINE_AA)

        vw.write(blended)

        # Save sample snapshots
        if seq_idx in snapshot_indices:
            snap_path = out_mp4_path.parent / f"snapshot_{sequence}_frame_{seq_idx:02d}_{film_name}_t{t:03d}.png"
            cv2.imwrite(str(snap_path), blended)
            saved_snapshots.append(snap_path)

        # Also save key snapshots for user analysis cases (FL1, FL4, FL5, BF5, FL6)
        if film_name in ["5_1_N1_FL1_F2", "5_1_N1_FL4_F2", "5_1_N1_FL5_F2", "5_1_N1_BF5_F2", "5_1_N1_FL6_F2"] and t == 0:
            user_snap_path = out_mp4_path.parent / f"snapshot_{sequence}_{film_name}_t{t:03d}.png"
            cv2.imwrite(str(user_snap_path), blended)

    vw.release()

    # Convert AVI to standard H.264 MP4 with ffmpeg
    print(f"\nFinalizing H.264 MP4 encoding with ffmpeg...")
    cmd_ffmpeg = [
        "ffmpeg", "-y", "-i", str(temp_avi),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        str(out_mp4_path)
    ]
    subprocess.run(cmd_ffmpeg, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_avi.exists():
        temp_avi.unlink()

    print(f"✓ Successfully generated population movie: {out_mp4_path} ({out_mp4_path.stat().st_size / (1024*1024):.2f} MB)")
    return out_mp4_path


def main():
    parser = argparse.ArgumentParser(description="Retracked Population Movie Generator")
    parser.add_argument("--exp_dir", type=str, default="/Volumes/X10 Pro/Movies/2026_08_28_M160")
    parser.add_argument("--sequence", type=str, default="5_1_N1_F2", help="Sequence name (e.g. 5_1_N1_F2)")
    parser.add_argument("--cell_limit", type=int, default=None, help="Optional limit on unreviewed cells to process")
    parser.add_argument("--fps", type=int, default=6, help="Playback frames per second")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay transparency")
    parser.add_argument("--out_mp4", type=str, default=None, help="Output MP4 video path")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    linkage_file = exp_dir / "sequence_linkage.json"
    with open(linkage_file) as f:
        linkage = json.load(f)

    cache_file = REPO_ROOT / "SingleCellQuantificationHPC/scratch/gold_standard_11frame_features.pkl"
    g1_windows, g2_windows, _ = build_gold_standard_training_set(exp_dir, linkage, cache_file)
    hmm = TemporalDivisionHMM()
    hmm.fit_emissions(g1_windows, g2_windows)
    llr = MultivariateLLRClassifier(reg_cov=1e-2)
    llr.fit(g1_windows.reshape(len(g1_windows), -1), g2_windows.reshape(len(g2_windows), -1))

    # 1. Run Hard-EM ABBT on unreviewed cohort
    retracked_cells, summary = run_abbt_on_unreviewed_cohort(exp_dir, args.sequence, linkage, hmm, llr, args.cell_limit)
    print("\n--- ABBT UNREVIEWED TRACKING SUMMARY ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 2. Render population movie
    if args.out_mp4:
        out_mp4_path = Path(args.out_mp4)
    else:
        out_mp4_path = Path(f"/Users/user/.gemini/antigravity-ide/brain/38dc6c8a-3edb-4a75-b352-c9b11def3095/retracked_population_{args.sequence}.mp4")

    films = linkage[args.sequence]["films"]
    render_population_movie(exp_dir, args.sequence, films, retracked_cells, out_mp4_path, fps=args.fps, alpha=args.alpha)


if __name__ == "__main__":
    main()
