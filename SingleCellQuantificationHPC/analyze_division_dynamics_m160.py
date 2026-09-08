#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_division_dynamics_m160.py

Quantifies single-cell biophysical dynamics around the end of cell division
across the 303 validated tracks (134 Good + 169 Corrected) in experiment 2026_08_28_M160.

Features Quantified:
1. Major Axis Length: Single daughter length, 2x daughter length, and sister-merged combined length.
2. Hourglass Shape Score: Normalized touching-circles pattern score.
3. Septum / Strip Score: Split-rectangles pattern score and septum intensity contrast.

Window: [-10, +5] frames relative to division time (t_div = 0).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    # minor-axis direction = rotate major axis by 90°
    vy, vx = np.cos(theta), -np.sin(theta)
    a_minor = getattr(r, "minor_axis_length", 0.0) / 2.0
    mid1_rc = (cy - a_minor * vy, cx - a_minor * vx)
    mid2_rc = (cy + a_minor * vy, cx + a_minor * vx)
    return mid1_rc, mid2_rc, r


def find_sister_cell_mask(
    film_dir: Path,
    film_name: str,
    t: int,
    mother_mask: np.ndarray,
    curr_daughter_mask: np.ndarray,
    min_overlap_ratio: float = 0.15
) -> Optional[np.ndarray]:
    """
    Finds the sister cell in the full segmentation _seg.tif at frame t that
    overlaps the pre-division mother mask but is distinct from curr_daughter_mask.
    """
    # Try c_0_seg.tif (GFP) or c_1_seg.tif (BF)
    seg_candidates = [
        film_dir / f"Masks_{film_name}" / f"{film_name}_t_{t:03d}_c_0_seg.tif",
        film_dir / f"Masks_{film_name}" / f"{film_name}_t_{t:03d}_z_1_c_1_seg.tif",
        film_dir / f"Masks_{film_name}" / f"{film_name}_t_{t:03d}_z_2_c_1_seg.tif",
        film_dir / f"Masks_{film_name}" / f"{film_name}_t_{t:03d}_c_1_seg.tif",
        film_dir / f"Masks_{film_name}" / f"{film_name}_t_{t:03d}_seg.tif",
    ]
    seg_path = None
    for cand in seg_candidates:
        if cand.exists():
            seg_path = cand
            break
            
    if not seg_path:
        return None

    try:
        seg = tifffile.imread(str(seg_path))
        if seg.ndim == 3 and seg.shape[-1] in (3, 4):
            seg = seg[..., 0]
        
        # Overlapping labels in mother footprint
        overlap_labels = np.unique(seg[mother_mask > 0])
        best_sister = None
        max_sister_area = 0

        for lbl in overlap_labels:
            if lbl <= 0:
                continue
            lbl_mask = (seg == lbl)
            lbl_area = lbl_mask.sum()
            if lbl_area < 300: # discard noise/debris
                continue
                
            # Intersect with current daughter
            inter_daughter = np.logical_and(curr_daughter_mask, lbl_mask).sum()
            # If it doesn't overlap heavily with current daughter (< 30% of its size)
            if inter_daughter / max(lbl_area, 1) < 0.3:
                # Check overlap with mother
                inter_mother = np.logical_and(mother_mask, lbl_mask).sum()
                if inter_mother / max(lbl_area, 1) >= min_overlap_ratio:
                    if lbl_area > max_sister_area:
                        max_sister_area = lbl_area
                        best_sister = lbl_mask

        return best_sister
    except Exception:
        return None


def extract_division_event_metrics(
    exp_dir: Path,
    film_name: str,
    cid: int,
    t_div: int,
    window_pre: int = 10,
    window_post: int = 5,
    side_px: int = 50
) -> Optional[List[Dict[str, Any]]]:
    """
    Quantifies the sequence of frames around t_div for a single cell.
    """
    film_dir = exp_dir / film_name
    masks_csv = film_dir / f"TrackedCells_{film_name}" / f"cell_{cid}_masks.csv"
    if not masks_csv.exists():
        return None

    df_masks = pd.read_csv(masks_csv)
    H = int(df_masks.iloc[0]["height"])
    W = int(df_masks.iloc[0]["width"])

    rle_col = "rle_gfp" if "rle_gfp" in df_masks.columns else "rle_bf"
    channel_type = "FL" if "FL" in film_name else "BF"

    # Pre-division mother mask at t_div - 1
    rows_pre = df_masks[df_masks["time_point"] == t_div - 1]
    if rows_pre.empty:
        return None
    rle_mother = rows_pre[rle_col].values[0]
    if not isinstance(rle_mother, str) or len(rle_mother) == 0:
        return None
    mother_mask = validate_and_decode_rle(rle_mother, H, W)
    if mother_mask.sum() < 1000:
        return None

    event_rows = []

    for rel_t in range(-window_pre, window_post + 1):
        target_t = t_div + rel_t
        row_t = df_masks[df_masks["time_point"] == target_t]
        if row_t.empty:
            continue

        rle_t = row_t[rle_col].values[0]
        if not isinstance(rle_t, str) or len(rle_t) == 0:
            continue
        mask_curr = validate_and_decode_rle(rle_t, H, W)
        if mask_curr.sum() < 200:
            continue

        # Sister mask search if post-division
        combined_mask = mask_curr.copy()
        sister_found = False
        sister_area = 0

        if rel_t >= 0:
            sister_mask = find_sister_cell_mask(film_dir, film_name, target_t, mother_mask, mask_curr)
            if sister_mask is not None:
                combined_mask = np.logical_or(combined_mask, sister_mask)
                sister_found = True
                sister_area = int(sister_mask.sum())

        # Measure Single Daughter
        mid1_s, mid2_s, rp_s = compute_mask_midpoints(mask_curr)
        if rp_s is None:
            continue
        maj_s = float(rp_s.major_axis_length)
        min_s = float(rp_s.minor_axis_length)
        area_s = int(mask_curr.sum())

        # Measure Combined / Sister-Merged
        mid1_c, mid2_c, rp_c = compute_mask_midpoints(combined_mask)
        if rp_c is None:
            continue
        maj_c = float(rp_c.major_axis_length)
        min_c = float(rp_c.minor_axis_length)
        area_c = int(combined_mask.sum())

        # Crop bounding box for pattern scoring
        minr, minc_box, maxr, maxc_box = rp_c.bbox
        pad = 12
        minr = max(0, minr - pad)
        minc_box = max(0, minc_box - pad)
        maxr = min(H, maxr + pad)
        maxc_box = min(W, maxc_box + pad)

        crop_mask = combined_mask[minr:maxr, minc_box:maxc_box]
        crop_sup = crop_mask.copy()

        # Crop midpoints
        cy, cx = rp_c.centroid
        cy_crop, cx_crop = cy - minr, cx - minc_box
        theta = getattr(rp_c, "orientation", 0.0) or 0.0
        vy, vx = np.cos(theta), -np.sin(theta)
        a_minor_half = min_c / 2.0
        mid1_rc = (cy_crop - a_minor_half * vy, cx_crop - a_minor_half * vx)
        mid2_rc = (cy_crop + a_minor_half * vy, cx_crop + a_minor_half * vx)

        # 1. Hourglass Score (touching semicircles template)
        try:
            pat_hg = pattern_score_touching_circles(
                crop_sup, crop_mask, mid1_rc, mid2_rc, side_px=side_px, stride=1
            )
            hg_score = float(pat_hg["best_score_norm"])
        except Exception:
            hg_score = 0.0

        # 2. Strip Score (split rectangles template)
        try:
            pat_strip = pattern_score_split_rectangles(
                crop_sup, crop_mask, mid1_rc, mid2_rc, side_px=side_px, stride=1
            )
            strip_score = float(pat_strip["best_score_norm"])
        except Exception:
            strip_score = 0.0

        # 3. Septum Intensity Contrast
        septum_int_contrast = 0.0
        frame_candidates = [
            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{target_t:03d}_c_0.tif",
            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{target_t:03d}_z_1_c_1.tif",
            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{target_t:03d}_z_2_c_1.tif",
            film_dir / f"Frames_{film_name}" / f"{film_name}_t_{target_t:03d}_c_1.tif",
        ]
        frame_file = None
        for cand in frame_candidates:
            if cand.exists():
                frame_file = cand
                break

        if frame_file:
            try:
                img = tifffile.imread(str(frame_file))
                if img.ndim == 3 and img.shape[-1] in (3, 4):
                    img = img[..., 0]
                crop_img = img[minr:maxr, minc_box:maxc_box].astype(np.float32)

                m_map, n_map = transform_to_mn_space(mid1_rc, mid2_rc, crop_mask, reflect=False)
                # Septum band along minor axis (center |n| <= 4 px)
                center_band = (np.abs(n_map) <= 4.0) & crop_mask
                cyt_band = (np.abs(n_map) > 8.0) & (np.abs(n_map) <= 22.0) & crop_mask
                if center_band.any() and cyt_band.any():
                    if channel_type == "FL":
                        # Bright cytokinesis ring (FL > cytosol)
                        septum_int_contrast = float(np.mean(crop_img[center_band]) - np.mean(crop_img[cyt_band]))
                    else:
                        # BF ridge contrast: absolute difference or dark valley contrast
                        septum_int_contrast = float(np.abs(np.mean(crop_img[center_band]) - np.mean(crop_img[cyt_band])))
            except Exception:
                septum_int_contrast = 0.0

        # 2x major axis length
        maj_2x = (2.0 * maj_s) if rel_t >= 0 else maj_s

        event_rows.append({
            "rel_t": rel_t,
            "target_t": target_t,
            "channel_type": channel_type,
            "area_single": area_s,
            "area_sister": sister_area,
            "area_combined": area_c,
            "sister_found": sister_found,
            "maj_single": maj_s,
            "maj_2x": maj_2x,
            "maj_combined": maj_c,
            "min_combined": min_c,
            "hourglass_score": hg_score,
            "strip_score": strip_score,
            "septum_contrast": septum_int_contrast,
        })

    return event_rows if len(event_rows) >= (window_pre + 1) else None


def main():
    parser = argparse.ArgumentParser(description="Analyze cell division dynamics across validated tracks.")
    parser.add_argument("--movie_root", type=str, default="/Volumes/X10 Pro/Movies", help="Movie root directory")
    parser.add_argument("--exp", type=str, default="2026_08_28_M160", help="Experiment name")
    parser.add_argument("--window_pre", type=int, default=10, help="Frames before division")
    parser.add_argument("--window_post", type=int, default=5, help="Frames after division")
    parser.add_argument("--output_dir", type=str, default="SingleCellQuantificationHPC/scratch/division_dynamics_output", help="Output directory")
    args = parser.parse_args()

    exp_dir = Path(args.movie_root) / args.exp
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_file = exp_dir / "sequence_linkage.json"
    with open(seq_file) as f:
        linkage = json.load(f)

    sequences = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]
    all_events_data = []
    event_meta = []
    event_id = 0

    print(f"🚀 Starting Division Dynamics Analysis for {args.exp} across {sequences}...")

    for seq_name in sequences:
        qc_file = exp_dir / f"qc_{seq_name}.json"
        if not qc_file.exists():
            continue
        with open(qc_file) as f:
            qc = json.load(f)

        films = linkage[seq_name]["films"]
        global_cells = linkage[seq_name]["global_cells"]

        # Validated tracks (good + corrected)
        curated_gids = [
            gid for gid, val in qc.items()
            if (val.get("status") if isinstance(val, dict) else val) in ["good", "corrected"]
        ]
        print(f"\n📂 Processing {seq_name}: {len(curated_gids)} curated tracks")

        for gid in curated_gids:
            tr = global_cells.get(gid, [])
            for f_idx, film in enumerate(films):
                cid = tr[f_idx]
                if cid <= 0:
                    continue
                masks_csv = exp_dir / film / f"TrackedCells_{film}" / f"cell_{cid}_masks.csv"
                if not masks_csv.exists():
                    continue

                try:
                    df_m = pd.read_csv(masks_csv)
                    col = "area_gfp" if "area_gfp" in df_m.columns else "area_bf"
                    if col not in df_m.columns:
                        continue

                    areas = df_m[col].values
                    last_div_idx = -999

                    for i in range(args.window_pre, len(areas) - args.window_post):
                        if i - last_div_idx < 20:
                            continue  # refractory period between divisions
                        a_prev = areas[i - 1]
                        a_curr = areas[i]
                        if a_prev >= 2500 and a_curr >= 800:
                            ratio = a_curr / a_prev
                            if 0.35 <= ratio <= 0.65:
                                # Division event detected
                                t_div = int(df_m.iloc[i]["time_point"])
                                event_rows = extract_division_event_metrics(
                                    exp_dir=exp_dir,
                                    film_name=film,
                                    cid=cid,
                                    t_div=t_div,
                                    window_pre=args.window_pre,
                                    window_post=args.window_post
                                )
                                if event_rows:
                                    last_div_idx = i
                                    event_id += 1
                                    event_meta.append({
                                        "event_id": event_id,
                                        "seq": seq_name,
                                        "gid": gid,
                                        "film": film,
                                        "local_cid": cid,
                                        "t_div": t_div,
                                        "a_prev": a_prev,
                                        "a_curr": a_curr,
                                        "ratio": ratio
                                    })
                                    for r in event_rows:
                                        r["event_id"] = event_id
                                        r["seq"] = seq_name
                                        r["gid"] = gid
                                        r["film"] = film
                                        all_events_data.append(r)
                except Exception as ex:
                    print(f"  [warn] Error processing {gid} in {film}: {ex}")

    print(f"\n✅ Total quantified division events: {event_id}")
    if not all_events_data:
        print("❌ No division events could be fully quantified.")
        return

    df_all = pd.DataFrame(all_events_data)
    df_meta = pd.DataFrame(event_meta)

    df_all.to_csv(out_dir / "division_dynamics_all_frames.csv", index=False)
    df_meta.to_csv(out_dir / "division_events_metadata.csv", index=False)
    print(f"💾 Saved full tabular dataset to {out_dir / 'division_dynamics_all_frames.csv'}")

    # ==========================================
    # Statistical Aggregation & Plotting
    # ==========================================
    print("\n📊 Computing statistical summaries...")
    rel_times = np.arange(-args.window_pre, args.window_post + 1)
    
    # Group by rel_t
    grp = df_all.groupby("rel_t")
    
    summary = pd.DataFrame({
        "rel_t": rel_times,
        "area_single_mean": grp["area_single"].mean(),
        "area_single_std": grp["area_single"].std(),
        "area_combined_mean": grp["area_combined"].mean(),
        "area_combined_std": grp["area_combined"].std(),
        "maj_single_mean": grp["maj_single"].mean(),
        "maj_single_std": grp["maj_single"].std(),
        "maj_2x_mean": grp["maj_2x"].mean(),
        "maj_2x_std": grp["maj_2x"].std(),
        "maj_combined_mean": grp["maj_combined"].mean(),
        "maj_combined_std": grp["maj_combined"].std(),
        "hourglass_mean": grp["hourglass_score"].mean(),
        "hourglass_std": grp["hourglass_score"].std(),
        "hourglass_median": grp["hourglass_score"].median(),
        "strip_score_mean": grp["strip_score"].mean(),
        "strip_score_std": grp["strip_score"].std(),
        "strip_score_median": grp["strip_score"].median(),
        "septum_contrast_mean": grp["septum_contrast"].mean(),
        "septum_contrast_std": grp["septum_contrast"].std(),
    })
    summary.to_csv(out_dir / "division_dynamics_summary.csv", index=False)

    # ------------------------------------------
    # Master Figure: 4-Panel Biophysical Dynamics
    # ------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    plt.subplots_adjust(hspace=0.28, wspace=0.25)

    # Panel 1: Cell Area Dynamics
    ax = axes[0, 0]
    ax.plot(summary["rel_t"], summary["area_single_mean"], "o-", color="#ef4444", lw=2.5, label="Single Daughter Area")
    ax.plot(summary["rel_t"], summary["area_combined_mean"], "s--", color="#3b82f6", lw=2, label="Sister-Merged Area (Combined)")
    ax.axvline(0, color="gray", ls=":", lw=1.5, label="Division Moment ($t_0$)")
    ax.set_title("1. Cell Area Dynamics around Division", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Frame ($t - t_{\\mathrm{div}}$)", fontsize=10)
    ax.set_ylabel("Area (pixels)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Panel 2: Major Axis Length Dynamics
    ax = axes[0, 1]
    ax.plot(summary["rel_t"], summary["maj_single_mean"], "o-", color="#ef4444", lw=2, label="Single Cell Length")
    ax.plot(summary["rel_t"], summary["maj_2x_mean"], "^-", color="#8b5cf6", lw=2.5, label="$2\\times$ Daughter Length (post-div)")
    ax.plot(summary["rel_t"], summary["maj_combined_mean"], "s--", color="#10b981", lw=2, label="Sister-Merged Length")
    ax.axvline(0, color="gray", ls=":", lw=1.5)
    ax.set_title("2. Major Axis Length Dynamics (Sudden Elongation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Frame ($t - t_{\\mathrm{div}}$)", fontsize=10)
    ax.set_ylabel("Major Axis Length (pixels)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Panel 3: Hourglass Shape Score
    ax = axes[1, 0]
    ax.plot(summary["rel_t"], summary["hourglass_mean"], "D-", color="#f59e0b", lw=2.5, label="Hourglass Score (Mean)")
    ax.fill_between(
        summary["rel_t"],
        summary["hourglass_mean"] - summary["hourglass_std"] / np.sqrt(event_id),
        summary["hourglass_mean"] + summary["hourglass_std"] / np.sqrt(event_id),
        color="#f59e0b", alpha=0.2
    )
    ax.axvline(0, color="gray", ls=":", lw=1.5)
    ax.set_title("3. Hourglass Shape Score (Touching Semicircles)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Frame ($t - t_{\\mathrm{div}}$)", fontsize=10)
    ax.set_ylabel("Normalized Score", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Panel 4: Strip & Septum Contrast
    ax = axes[1, 1]
    ax.plot(summary["rel_t"], summary["strip_score_mean"], "p-", color="#06b6d4", lw=2.5, label="Strip Score (Split Rectangles)")
    ax_twin = ax.twinx()
    ax_twin.plot(summary["rel_t"], summary["septum_contrast_mean"], "v--", color="#ec4899", lw=2, label="Septum Intensity Contrast")
    ax.axvline(0, color="gray", ls=":", lw=1.5)
    ax.set_title("4. Septum Strip Score & Intensity Contrast", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Frame ($t - t_{\\mathrm{div}}$)", fontsize=10)
    ax.set_ylabel("Strip Score (Normalized)", fontsize=10, color="#06b6d4")
    ax_twin.set_ylabel("Intensity Contrast ($\Delta I$)", fontsize=10, color="#ec4899")
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    fig_path = out_dir / "division_dynamics_curves_m160.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Saved master dynamics figure to {fig_path}")

    # Print summary key metrics
    print("\n" + "="*60)
    print("🎯 KEY BIOPHYSICAL FINDINGS:")
    print("="*60)
    
    # Feature 1 Check: Sudden length increase
    len_pre = float(summary[summary["rel_t"] == -1]["maj_single_mean"].values[0])
    len_2x = float(summary[summary["rel_t"] == 0]["maj_2x_mean"].values[0])
    len_comb = float(summary[summary["rel_t"] == 0]["maj_combined_mean"].values[0])
    print(f"1. Major Axis Length:")
    print(f"   - Pre-division mother length (t=-1):  {len_pre:.2f} px")
    print(f"   - 2x Daughter length at division (t=0): {len_2x:.2f} px (+{((len_2x/len_pre)-1)*100:.1f}%)")
    print(f"   - Sister-merged length at division:    {len_comb:.2f} px (+{((len_comb/len_pre)-1)*100:.1f}%)")

    # Feature 2 Check: Hourglass peak
    hg_pre = float(summary[summary["rel_t"] == -1]["hourglass_mean"].values[0])
    hg_div = float(summary[summary["rel_t"] == 0]["hourglass_mean"].values[0])
    hg_post = float(summary[summary["rel_t"] == 5]["hourglass_mean"].values[0])
    print(f"\n2. Hourglass Shape Score:")
    print(f"   - Pre-division (t=-1): {hg_pre:+.4f}")
    print(f"   - Division peak (t=0): {hg_div:+.4f}")
    print(f"   - Post-division (t=+5): {hg_post:+.4f}")

    # Feature 3 Check: Strip score
    strip_pre = float(summary[summary["rel_t"] == -1]["strip_score_mean"].values[0])
    strip_div = float(summary[summary["rel_t"] == 0]["strip_score_mean"].values[0])
    strip_post = float(summary[summary["rel_t"] == 5]["strip_score_mean"].values[0])
    sept_div = float(summary[summary["rel_t"] == 0]["septum_contrast_mean"].values[0])
    print(f"\n3. Septum Strip Score & Intensity:")
    print(f"   - Pre-division strip score (t=-1): {strip_pre:+.4f}")
    print(f"   - Division strip score (t=0):     {strip_div:+.4f}")
    print(f"   - Post-division strip score (t=+5): {strip_post:+.4f}")
    print(f"   - Peak Septum Intensity Contrast (t=0): {sept_div:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
