#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 20:46:50 2025

@author: user
"""

# ==============================
# File: xcorr_utils.py
# ==============================
"""Rotation-aware cross-correlation helpers and debug plotting.

Public API:
- xcorr_best_of_six(...): pick best candidate mask by xcorr over angle sweep
- save_xcorr_debug_figure(...): optional PNG visualization of selection

All other functions are internal.
"""

from typing import Tuple, List, Dict, Any

import numpy as np
from numpy.fft import rfft2, irfft2
from itertools import combinations

from skimage.measure import regionprops, label as _label
from skimage.transform import rotate as _rotate

# Matplotlib is imported lazily inside save_xcorr_debug_figure to avoid
# pulling it in unless needed.

# ------------------------------
# Small utilities
# ------------------------------

def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    r0, r1 = int(np.clip(ys.min() - pad, 0, mask.shape[0])), int(np.clip(ys.max() + 1 + pad, 0, mask.shape[0]))
    c0, c1 = int(np.clip(xs.min() - pad, 0, mask.shape[1])), int(np.clip(xs.max() + 1 + pad, 0, mask.shape[1]))
    return r0, r1, c0, c1


def _union_bbox(b0, b1, H, W, extra_pad=0):
    r0 = max(0, min(b0[0], b1[0]) - extra_pad)
    r1 = min(H, max(b0[1], b1[1]) + extra_pad)
    c0 = max(0, min(b0[2], b1[2]) - extra_pad)
    c1 = min(W, max(b0[3], b1[3]) + extra_pad)
    return r0, r1, c0, c1


def _masked_norm(img: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(img, dtype=np.float32)
    vals = img[m].astype(np.float32)
    mu, sd = float(vals.mean()), float(vals.std() + eps)
    out = (img.astype(np.float32) - mu) * m
    out /= sd
    return out


def _phase_corr_peak(a: np.ndarray, b: np.ndarray, eps: float = 1e-12):
    Fa, Fb = rfft2(a), rfft2(b)
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), eps)
    c = irfft2(R, s=a.shape)
    peak_idx = np.unravel_index(np.argmax(c), c.shape)
    peak_val = float(c[peak_idx])
    sh = np.array(peak_idx, dtype=float)
    for d, n in enumerate(a.shape):
        if sh[d] > n // 2:
            sh[d] -= n
    return peak_val, (float(sh[0]), float(sh[1]))


def _mask_union_from_labels(labeled: np.ndarray, labels: List[int]) -> np.ndarray:
    m = np.zeros_like(labeled, dtype=bool)
    for lb in labels:
        if lb:
            m |= (labeled == lb)
    return m


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    lab = _label(mask.astype(np.uint8), connectivity=2)
    regs = regionprops(lab)
    if not regs:
        ys, xs = np.nonzero(mask)
        return (float(ys.mean()) if ys.size else 0.0, float(xs.mean()) if xs.size else 0.0)
    cy, cx = regs[0].centroid
    return float(cy), float(cx)


def _orientation_deg(mask: np.ndarray) -> float:
    lab = _label(mask.astype(np.uint8), connectivity=2)
    regs = regionprops(lab)
    if not regs or getattr(regs[0], "orientation", None) is None:
        return 0.0
    return float(np.degrees(regs[0].orientation))


# ------------------------------
# Core xcorr routine
# ------------------------------

def _max_xcorr_over_angles(ref_img, ref_mask, cand_img, cand_mask, angles_deg, pad_px=20):
    H, W = ref_img.shape
    b_ref = _bbox_from_mask(ref_mask, pad_px)
    b_cnd = _bbox_from_mask(cand_mask, pad_px)
    r0, r1, c0, c1 = _union_bbox(b_ref, b_cnd, H, W, extra_pad=pad_px)

    R = ref_img[r0:r1, c0:c1]
    Mref = ref_mask[r0:r1, c0:c1].astype(bool)
    Rn = _masked_norm(R, Mref)

    best = {"xcorr": -1e9, "angle_deg": 0.0, "shift_rc": (0.0, 0.0), "bbox": (r0, r1, c0, c1)}

    C_full = cand_img[r0:r1, c0:c1]
    Mc_full = cand_mask[r0:r1, c0:c1].astype(bool)

    for ang in angles_deg:
        C_rot = _rotate(C_full, angle=ang, resize=False, preserve_range=True, order=1)
        Mc_rot = _rotate(Mc_full.astype(float), angle=ang, resize=False, preserve_range=True, order=0) > 0.5
        if Mc_rot.sum() == 0 or Mref.sum() == 0:
            continue
        Cn = _masked_norm(C_rot, Mc_rot)
        val, sh = _phase_corr_peak(Rn, Cn)
        if val > best["xcorr"]:
            best.update({"xcorr": float(val), "angle_deg": float(ang), "shift_rc": (float(sh[0]), float(sh[1]))})
    return best


def xcorr_best_of_six(
    t0_mask: np.ndarray, t0_bf: np.ndarray,
    t1_labeled: np.ndarray, t1_bf: np.ndarray,
    *, num_singles: int = 3, num_pairs: int = 3, pair_pool_k: int = 6,
    min_area: int = 50, angle_pad_deg: float = 15.0, angle_step_deg: float = 3.0, pad_px: int = 20
) -> Dict[str, Any]:
    """Evaluate 3 single segments + 3 unions at t1 and choose by max xcorr.
    Returns dict with keys: best_mask, best, candidates (list with metadata per cand)
    """
    H, W = t0_mask.shape
    ref_cy, ref_cx = _centroid(t0_mask)
    ref_orient = _orientation_deg(t0_mask)

    regs = regionprops(t1_labeled)
    items = []
    for r in regs:
        if r.label == 0 or r.area < min_area:
            continue
        cy, cx = r.centroid
        dist = float(np.hypot(cy - ref_cy, cx - ref_cx))
        items.append({"label": int(r.label), "centroid": (float(cy), float(cx)), "area": int(r.area), "dist": dist})
    if not items:
        return {"best_mask": np.zeros((H, W), bool), "best": None, "candidates": []}

    items.sort(key=lambda d: d["dist"])
    singles = items[:max(1, num_singles)]

    pool = items[:max(pair_pool_k, num_pairs + 1)]
    pair_meta = []
    for (a, b) in combinations(pool, 2):
        mask_union = _mask_union_from_labels(t1_labeled, [a["label"], b["label"]])
        cy_u, cx_u = _centroid(mask_union)
        dist_u = float(np.hypot(cy_u - ref_cy, cx_u - ref_cx))
        pair_meta.append({"labels": [a["label"], b["label"]], "dist": dist_u, "area": int(mask_union.sum())})
    pair_meta.sort(key=lambda d: d["dist"])
    pairs = pair_meta[:max(0, num_pairs)]

    candidate_specs = (
        [{"type": "single", "labels": [s["label"]], "dist": s["dist"]} for s in singles] +
        [{"type": "pair", "labels": p["labels"], "dist": p["dist"]} for p in pairs]
    )

    results = []
    for spec in candidate_specs:
        cand_mask = _mask_union_from_labels(t1_labeled, spec["labels"])
        cand_orient = _orientation_deg(cand_mask)
        base = float(ref_orient - cand_orient)
        n_steps = int(np.ceil(angle_pad_deg / max(1e-6, angle_step_deg)))
        angs = [base + k * angle_step_deg for k in range(-n_steps, n_steps + 1)]
        if 0.0 not in angs:
            angs.append(0.0)

        xres = _max_xcorr_over_angles(t0_bf, t0_mask, t1_bf, cand_mask, angs, pad_px=pad_px)
        results.append({
            "type": spec["type"],
            "labels": spec["labels"],
            "dist": spec["dist"],
            "xcorr": float(xres["xcorr"]),
            "angle_deg": float(xres["angle_deg"]),
            "shift_rc": tuple(xres["shift_rc"]),
            "mask": cand_mask,
            "bbox": xres["bbox"],
        })

    results.sort(key=lambda d: d["xcorr"], reverse=True)
    best = results[0]
    return {"best_mask": best["mask"].astype(bool), "best": best, "candidates": results}


# ------------------------------
# Debug plotting
# ------------------------------

def _union_many_bboxes(bboxes, H, W, pad=0):
    if not bboxes:
        return 0, H, 0, W
    r0 = max(0, min(b[0] for b in bboxes) - pad)
    r1 = min(H, max(b[1] for b in bboxes) + pad)
    c0 = max(0, min(b[2] for b in bboxes) - pad)
    c1 = min(W, max(b[3] for b in bboxes) + pad)
    return int(r0), int(r1), int(c0), int(c1)


def save_xcorr_debug_figure(
    t: int,
    t0_img: np.ndarray, t1_img: np.ndarray,
    ref_mask: np.ndarray, best_mask: np.ndarray,
    candidates: List[Dict[str, Any]],
    out_png_path: str
) -> None:
    """Write a side-by-side PNG visualizing the chosen candidate and scores."""
    import matplotlib.pyplot as plt

    H, W = t0_img.shape
    ref_bb = _bbox_from_mask(ref_mask, pad=20)
    c_bbs = [c.get("bbox", ref_bb) for c in candidates if c is not None]
    r0, r1, c0, c1 = _union_many_bboxes([ref_bb] + c_bbs, H, W, pad=10)

    R = t0_img[r0:r1, c0:c1]
    I = t1_img[r0:r1, c0:c1]
    Mref = ref_mask[r0:r1, c0:c1].astype(bool)
    Mbest = best_mask[r0:r1, c0:c1].astype(bool)

    lab = []
    scores = []
    for c in candidates:
        lbl = "+".join([f"L{l}" for l in c["labels"]])
        lab.append(f'{c["type"][0]}:{lbl}')
        scores.append(c["xcorr"])
    order = np.argsort(scores)[::-1]
    lab = [lab[i] for i in order]
    scores = [scores[i] for i in order]

    best = candidates[order[0]] if candidates else None
    title_best = "n/a"
    if best:
        title_best = f'{best["type"]}: ' + "+".join([f"L{l}" for l in best["labels"]])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(R, cmap='gray')
    ax0.contour(Mref, levels=[0.5], linewidths=1.5)
    ax0.set_title(f"t0 (ref)  crop  [t={t - 1 if t > 0 else t}]")
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(I, cmap='gray')
    ax1.contour(Mbest, levels=[0.5], linewidths=1.5)
    if best:
        ang = best["angle_deg"]; dy, dx = best["shift_rc"]
        ax1.set_title(f"t1 crop — chosen: {title_best}\nangle={ang:.2f}°, shift=(dy={dy:.2f}, dx={dx:.2f})")
    else:
        ax1.set_title("t1 crop — chosen: n/a")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, :])
    ax2.bar(range(len(scores)), scores)
    ax2.set_xticks(range(len(scores)))
    ax2.set_xticklabels(lab, rotation=40, ha='right')
    ax2.set_ylabel("max XCorr (phase corr)")
    ax2.set_title("Candidate scores (higher is better)")

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=140)
    plt.close(fig)


# ==============================
# Patch notes for your main script
# ==============================
# 1) Add at top of your script, near other imports:
# from xcorr_utils import xcorr_best_of_six, save_xcorr_debug_figure
#
# 2) Delete the following definitions from your main file (now provided by module):
#    - _bbox_from_mask
#    - _union_bbox
#    - _masked_norm
#    - _phase_corr_peak
#    - _mask_union_from_labels
#    - _centroid
#    - _orientation_deg
#    - _max_xcorr_over_angles
#    - xcorr_best_of_six
#    - _union_many_bboxes
#    - _save_xcorr_debug_figure
#
# 3) Replace calls to _save_xcorr_debug_figure(...) with save_xcorr_debug_figure(...)
#
# 4) The rest of your tracking code remains unchanged. Ensure sys.path includes
#    the directory containing xcorr_utils.py (you already append your project path).
