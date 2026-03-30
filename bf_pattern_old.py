#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:32:00 2025

@author: user
"""

# bf_pattern.py
# -*- coding: utf-8 -*-
"""
Brightfield (BF) pattern-only scoring as a reusable function.

Extracted from the tracking/quantification script so it can be imported elsewhere.
Keeps the same behavior and outputs, with one small API improvement:
- Accept a `cell_id` argument instead of relying on a free variable.
"""

#from __future__ import annotations

import os
import math
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imsave

# Project-local helpers (paths must already be on sys.path in the caller)
from Image_quantification_functions import find_midpoints_on_minor_axis
from quant_helpers import (
    transform_to_mn_space,
    pattern_score_split_rectangles,
    save_split_rectangles_pattern_overlay,
)
from Cell_tracking_functions import touches_border


__all__ = ["bf_pattern_only"]


# -------------------------
# Module-level imports
# -------------------------
import os
import math
import numpy as np
from skimage.measure import label, regionprops
from imageio.v2 import imsave

# If these come from your codebase, keep their imports where you have them:
# from your_pkg.geometry import find_midpoints_on_minor_axis, transform_to_mn_space
# from your_pkg.patterns import pattern_score_split_rectangles, save_split_rectangles_pattern_overlay
# from your_pkg.masks import touches_border


# -------------------------
# Helpers (UNNESTED)
# -------------------------
def _score_split_rectangles_intensity(values_img: np.ndarray,
                                      m_map: np.ndarray,
                                      n_map: np.ndarray,
                                      center_mn: tuple[float, float] | None,
                                      side_px: int,
                                      *,
                                      eps: float = 1e-9) -> dict | None:
    """
    Intensity-weighted split-rectangles score at a given (m0, n0) center.
    We compute two adjacent rectangles along the minor-axis (n), each of height=side_px,
    and width=side_px centered on m0. Score is the intensity difference between them.

    Returns dict with raw and normalized scores (or None if center missing).
    """
    if center_mn is None:
        return None

    m0, n0 = float(center_mn[0]), float(center_mn[1])
    half = float(side_px) / 2.0

    # Two stacked rectangles along n: [n0, n0+side_px) and (n0-side_px, n0]
    rect_top = (np.abs(m_map - m0) <= half) & (n_map >= n0) & (n_map < n0 + side_px)
    rect_bot = (np.abs(m_map - m0) <= half) & (n_map >= n0 - side_px) & (n_map < n0)

    # Sum intensities (ignore NaNs if you place them outside support)
    s_top = float(np.nansum(values_img[rect_top]))
    s_bot = float(np.nansum(values_img[rect_bot]))

    raw = s_top - s_bot
    norm = raw / (s_top + s_bot + eps)

    return {
        "best_score_raw_intensity": raw,
        "best_score_norm_intensity": norm,
        "rect_top_sum": s_top,
        "rect_bot_sum": s_bot,
        "rect_top_N": int(np.count_nonzero(np.isfinite(values_img[rect_top]))),
        "rect_bot_N": int(np.count_nonzero(np.isfinite(values_img[rect_bot]))),
    }

def _project_min_separation_sorted(means_sorted, sigmas_sorted, weights, eps=1e-12):
    """
    Enforce mu_{i+1} - mu_i >= sigma_{i+1} + sigma_i for sorted components by
    projecting means with weighted isotonic regression (PAV) after an offset trick.

    Minimizes sum_i w_i * (mu_i - m_i)^2 subject to separation constraints.
    """
    m = np.asarray(means_sorted, np.float64)
    s = np.maximum(np.asarray(sigmas_sorted, np.float64), 1e-12)
    w = np.maximum(np.asarray(weights, np.float64), eps)

    K = m.size
    if K <= 1:
        return m.copy()

    # Convert pairwise "min gap" constraints into a simple monotone constraint
    gaps = s[:-1] + s[1:]
    c = np.zeros(K, dtype=np.float64)
    c[1:] = np.cumsum(gaps)

    y = m - c  # target sequence for isotonic regression

    # Weighted Pool-Adjacent Violators (nondecreasing)
    block_mean, block_weight, block_range = [], [], []
    for i in range(K):
        block_mean.append(y[i])
        block_weight.append(w[i])
        block_range.append([i, i + 1])
        while len(block_mean) >= 2 and block_mean[-2] > block_mean[-1]:
            m2, w2 = block_mean.pop(), block_weight.pop()
            s2, e2 = block_range.pop()
            m1, w1 = block_mean.pop(), block_weight.pop()
            s1, e1 = block_range.pop()
            new_w = w1 + w2
            new_m = (m1 * w1 + m2 * w2) / max(new_w, eps)
            block_mean.append(new_m)
            block_weight.append(new_w)
            block_range.append([s1, e2])

    v_proj = np.empty(K, dtype=np.float64)
    for mean, (s_idx, e_idx) in zip(block_mean, block_range):
        v_proj[s_idx:e_idx] = mean

    mu_proj = v_proj + c
    return mu_proj


def _em_3gauss_uniform(x, mus, sigmas, pis, a_, b_, max_iter=200, tol=1e-6):
    """
    EM for a 1D mixture: 3 Gaussians + 1 Uniform([a_, b_]).
    Enforces |mu_i - mu_j| >= sigma_i + sigma_j via projection after each M-step.
    Returns (mus, sigmas, pis, gamma_matrix, loglik, iters).
    """
    x = np.asarray(x, np.float64)
    eps = 1e-12

    width = max(b_ - a_, 1e-6)
    cU = 1.0 / width
    sigmas = np.maximum(np.asarray(sigmas, np.float64), 1e-9)
    mus = np.asarray(mus, np.float64)
    pis = np.asarray(pis, np.float64)
    pis = pis / max(pis.sum(), eps)

    prev_ll = -np.inf
    for it in range(max_iter):
        # --- E-step ---
        z = (x[None, :] - mus[:, None]) / sigmas[:, None]
        Ng = (1.0 / (np.sqrt(2.0 * np.pi) * sigmas[:, None])) * np.exp(-0.5 * (z ** 2))  # (3, N)
        mix_g = pis[:3, None] * Ng

        Nu = np.full_like(x, cU)
        mix_u = pis[3] * Nu

        denom = mix_g.sum(axis=0) + mix_u + eps
        gamma_g = mix_g / denom
        gamma_u = (mix_u / denom)[None, :]

        # --- M-step (unconstrained) ---
        sum_g = gamma_g.sum(axis=1) + eps  # effective counts per Gaussian
        new_mus = (gamma_g @ x) / sum_g
        diffs = x[None, :] - new_mus[:, None]
        new_vars = (gamma_g * (diffs ** 2)).sum(axis=1) / sum_g
        new_sigmas = np.sqrt(np.maximum(new_vars, 1e-12))

        new_pis = np.empty_like(pis)
        new_pis[:3] = (gamma_g.sum(axis=1) / x.size)
        new_pis[3] = (gamma_u.sum() / x.size)
        new_pis = new_pis / max(new_pis.sum(), eps)

        # --- Project means to enforce separation; keep stable ordering ---
        order = np.argsort(new_mus)
        new_mus_sorted = new_mus[order]
        new_sigmas_sorted = new_sigmas[order]
        new_pis_sorted = new_pis.copy()
        new_pis_sorted[:3] = new_pis_sorted[:3][order]
        weights_sorted = sum_g[order]

        new_mus_sorted = _project_min_separation_sorted(
            new_mus_sorted, new_sigmas_sorted, weights_sorted, eps=eps
        )

        # Keep sorted labeling
        mus, sigmas, pis = new_mus_sorted, new_sigmas_sorted, new_pis_sorted

        # --- Evaluate projected params for convergence ---
        z2 = (x[None, :] - mus[:, None]) / sigmas[:, None]
        Ng2 = (1.0 / (np.sqrt(2.0 * np.pi) * sigmas[:, None])) * np.exp(-0.5 * (z2 ** 2))
        mix_g2 = pis[:3, None] * Ng2
        Nu2 = Nu  # same uniform pdf vector
        mix_u2 = pis[3] * Nu2
        denom2 = mix_g2.sum(axis=0) + mix_u2 + eps
        ll = float(np.sum(np.log(denom2)))

        if abs(ll - prev_ll) < tol * (abs(ll) + eps):
            gamma_g2 = mix_g2 / denom2
            gamma_u2 = (mix_u2 / denom2)[None, :]
            return mus, sigmas, pis, np.concatenate([gamma_g2, gamma_u2], axis=0), ll, (it + 1)

        prev_ll = ll

    # Max-iter fallback: return responsibilities for final params
    z = (x[None, :] - mus[:, None]) / sigmas[:, None]
    Ng = (1.0 / (np.sqrt(2.0 * np.pi) * sigmas[:, None])) * np.exp(-0.5 * (z ** 2))
    mix_g = pis[:3, None] * Ng
    Nu = np.full_like(x, cU)
    mix_u = pis[3] * Nu
    denom = mix_g.sum(axis=0) + mix_u + eps
    gamma_g = mix_g / denom
    gamma_u = (mix_u / denom)[None, :]
    ll = float(np.sum(np.log(denom + eps)))
    return mus, sigmas, pis, np.concatenate([gamma_g, gamma_u], axis=0), ll, max_iter
def pattern_score_split_rectangles_intensity_at_center(
    *,
    support_crop: np.ndarray,
    raw_crop: np.ndarray,
    m_map: np.ndarray,
    n_map: np.ndarray,
    center_mn: tuple[float, float],
    side_px: int,
    rect_n_frac: float = 0.5,
    rect_m_frac: float = 0.35,
    gap_frac: float = 0.10,
    swap_axes: bool = True,
    eps: float = 1e-9,
    min_count_each_rect: int = 3,
) -> dict:
    """
    Intensity-based split-rectangles score at a single, fixed center.
    INVERTED POLARITY:
      +1 for central split (expects low intensity / dark septum)
      -1 for the two bars (expects higher intensity / empty cytoplasm)
    """
    sup = np.asarray(support_crop, bool)
    if sup.sum() == 0:
        return {
            "best_score_raw": 0.0, "best_score_norm": 0.0,
            "best_center_rc": None, "best_center_mn": None,
            "side_px": float(side_px), "evaluated_centers": 0,
            "n_length_px": 0.0, "band_half_px": 0.0, "filtered_by_center_n": False
        }

    vals = raw_crop.astype(np.float64, copy=False)
    vals = np.where(sup, vals, np.nan)  # only supported pixels contribute

    m0, n0 = float(center_mn[0]), float(center_mn[1])
    r = float(side_px) * 0.5

    # Offsets in (m,n)
    m_off = m_map - m0
    n_off = n_map - n0

    # Square window inside the segment
    square = (np.abs(m_off) <= r) & (np.abs(n_off) <= r) & sup

    # Template axes
    if swap_axes:
        u, v = m_off, n_off
    else:
        u, v = n_off, m_off

    u_half   = float(rect_n_frac) * r
    gap_half = float(gap_frac) * r
    bar_tk   = float(rect_m_frac) * r
    bar_tk   = float(np.clip(bar_tk, 0.0, max(0.0, r - gap_half)))

    in_u_span  = (np.abs(u) <= u_half)
    in_v_top   = (v >= +gap_half) & (v <= +gap_half + bar_tk)
    in_v_bot   = (v >= -gap_half - bar_tk) & (v <= -gap_half)
    in_v_split = (np.abs(v) < gap_half)

    rect_top_mask  = square & in_u_span & in_v_top
    rect_bot_mask  = square & in_u_span & in_v_bot
    split_mask     = square & in_u_span & in_v_split

    # Require some pixels in each region (esp. the split)
    c_top  = int(np.count_nonzero(rect_top_mask))
    c_bot  = int(np.count_nonzero(rect_bot_mask))
    c_split= int(np.count_nonzero(split_mask))
    if (c_top < min_count_each_rect) or (c_bot < min_count_each_rect) or (c_split < min_count_each_rect):
        return {
            "best_score_raw": 0.0, "best_score_norm": 0.0,
            "best_center_rc": None, "best_center_mn": (m0, n0),
            "side_px": float(side_px), "evaluated_centers": 1,
            "n_length_px": float(np.nanmax(n_map[sup]) - np.nanmin(n_map[sup])) if sup.any() else 0.0,
            "band_half_px": r, "filtered_by_center_n": False
        }

    s_top   = float(np.nansum(vals[rect_top_mask]))
    s_bot   = float(np.nansum(vals[rect_bot_mask]))
    s_split = float(np.nansum(vals[split_mask]))

    # INVERTED POLARITY:
    #   raw = (+1)*split  +  (-1)*bars
    # Use the average of the two bars so bar thickness asymmetry doesn't bias much
    s_bars = 0.5 * (s_top + s_bot)
    raw    = s_split - s_bars

    denom  = (s_split + s_top + s_bot)
    norm   = (raw / (denom + eps)) if np.isfinite(denom) and denom > eps else 0.0

    # nearest pixel for marking
    d2 = (m_map - m0)**2 + (n_map - n0)**2
    rr, cc = np.unravel_index(np.nanargmin(d2), d2.shape)
    center_rc = (int(rr), int(cc))

    return {
        "best_score_raw":   raw,
        "best_score_norm":  norm,
        "best_center_rc":   center_rc,
        "best_center_mn":   (m0, n0),
        "side_px":          float(side_px),
        "evaluated_centers": 1,
        "n_length_px": float(np.nanmax(n_map[sup]) - np.nanmin(n_map[sup])) if sup.any() else 0.0,
        "band_half_px": r,
        "filtered_by_center_n": False,
    }



# -------------------------
# Main function (now clean, calls helpers)
# -------------------------
def bf_pattern_only(
    img_bf: np.ndarray,
    mask_bf_full: np.ndarray,
    t: int,
    out_dir: str,
    *,
    cell_id: str | int | None = None,
    posterior_cutoff: float = 0,
    side_px: int = 80,
    sigma_n_px: float = 1.0,   # positional prior σ along n (pixels)
    sigma_m_px: float = 6.0,   # <-- NEW: positional prior σ along m (pixels)
) -> dict | None:


    """
    Build a boolean support in BF using a 1-D mixture (3 Gaussians + 1 Uniform)
    inside the cell mask, then score the support with the split-rectangles pattern scorer.
    """
    # Basic guard
    seg_bool = np.asarray(mask_bf_full, bool)
    if seg_bool.sum() == 0:
        return None

    # -------------------------
    # Stable crop around the cell
    # -------------------------
    labeled_mask = label(seg_bool.astype(np.uint8), connectivity=2)
    props_list = regionprops(labeled_mask)
    if not props_list:
        return None
    region = props_list[0]
    min_row, min_col, max_row, max_col = region.bbox

    raw_crop = img_bf[min_row:max_row, min_col:max_col]
    cropped_cell_mask = seg_bool[min_row:max_row, min_col:max_col]
    if cropped_cell_mask.size == 0:
        return None

    # Ensure borders are False to avoid boundary artifacts
    cropped_cell_mask = cropped_cell_mask.copy()
    cropped_cell_mask[0, :] = False
    cropped_cell_mask[-1, :] = False
    cropped_cell_mask[:, 0] = False
    cropped_cell_mask[:, -1] = False

    labeled_cell = label(cropped_cell_mask.astype(np.uint8))
    props_list2 = regionprops(labeled_cell)
    if not props_list2:
        return None
    props = props_list2[0]

    # Centroid/orientation in cropped coords
    P0 = np.array(props.centroid)  # (row, col)
    orientation = float(getattr(props, "orientation", 0.0) or 0.0)
    d = np.array([np.cos(orientation), np.sin(orientation)], dtype=float)  # major-axis dir (row,col)

    # Boundary pixels in cropped coords
    try:
        from skimage.segmentation import find_boundaries  # local import to minimize global deps
        boundary_mask = find_boundaries(cropped_cell_mask, mode="inner")
        boundary_coords = np.column_stack(np.nonzero(boundary_mask))
    except Exception:
        boundary_coords = None

    # Intensity values inside the cropped mask
    vals = raw_crop[cropped_cell_mask].astype(np.float64, copy=False)
    if vals.size == 0:
        return None

    # Quantize floats for a stable mode
    if np.issubdtype(raw_crop.dtype, np.floating):
        vals_q = np.rint(vals).astype(np.int64)
        raw_q = np.rint(raw_crop).astype(np.int64)
    else:
        vals_q = vals.astype(np.int64)
        raw_q = raw_crop.astype(np.int64)

    binc = np.bincount(vals_q.ravel())
    mode_val = int(np.argmax(binc))
    mu0 = float(mode_val)

    p1, p99 = np.percentile(vals, [1, 99])
    sigma0 = max((p99 - p1) / 4.0, 1e-3)
    a, b = float(vals.min()), float(vals.max())

    # -------------------------
    # EM: 3 Gaussians + Uniform (call UNNESTED helper)
    # -------------------------
    delta = max(1.0, 0.5 * sigma0)
    mu_inits = np.array([max(a, mu0 - 2.0 * delta), mu0, min(b, mu0 + 2.0 * delta)], dtype=np.float64)
    sigma_inits = np.array([sigma0, sigma0, sigma0], dtype=np.float64)
    pi_inits = np.array([0.35, 0.40, 0.15, 0.10], dtype=np.float64)
    pi_inits = pi_inits / pi_inits.sum()

    mus, sigmas, pis, gamma_mat, ll, n_iter = _em_3gauss_uniform(
        vals, mu_inits, sigma_inits, pi_inits, a, b, max_iter=200, tol=1e-6
    )

    # -------------------------
    # Debug histogram + fit overlay
    # -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals_plot = vals.astype(np.float64, copy=False)
        if np.issubdtype(raw_crop.dtype, np.integer):
            vmin, vmax = int(vals_plot.min()), int(vals_plot.max())
            if vmax - vmin <= 4096:
                edges = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
                centers = np.arange(vmin, vmax + 1, 1.0)
                binw = 1.0
            else:
                edges = np.linspace(vals_plot.min(), vals_plot.max(), 513)
                centers = 0.5 * (edges[:-1] + edges[1:])
                binw = float(edges[1] - edges[0])
        else:
            edges = np.linspace(vals_plot.min(), vals_plot.max(), 257)
            centers = 0.5 * (edges[:-1] + edges[1:])
            binw = float(edges[1] - edges[0])

        counts, _ = np.histogram(vals_plot, bins=edges)
        scale = vals_plot.size * binw

        gauss_counts_list = []
        for k in range(3):
            sigma_eff = max(float(sigmas[k]), 1e-9)
            gpdf = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_eff)) * np.exp(
                -0.5 * ((centers - mus[k]) / sigma_eff) ** 2
            )
            gauss_counts_list.append(pis[k] * gpdf * scale)

        widthU = max(b - a, 1e-9)
        updf = np.zeros_like(centers, dtype=np.float64)
        inU = (centers >= a) & (centers <= b)
        updf[inU] = 1.0 / widthU
        uni_counts = pis[3] * updf * scale

        mix_counts = uni_counts.copy()
        for gc in gauss_counts_list:
            mix_counts += gc

        order = np.argsort(mus)
        gauss_counts_s = [gauss_counts_list[i] for i in order]

        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
        ax.bar(centers, counts, width=binw, align="center", alpha=0.35, label="Masked pixels")
        ax.plot(centers, mix_counts, linewidth=2.4, label="Mixture fit (3G+U)")
        styles = ["--", "-.", ":"]
        for idx, gc in enumerate(gauss_counts_s):
            ax.plot(centers, gc, linewidth=1.6, linestyle=styles[idx % len(styles)],
                    label=f"G{idx}")
        ax.plot(centers, uni_counts, linewidth=1.6, linestyle="-", alpha=0.6, label="Uniform")
        ax.set_xlabel("BF intensity")
        ax.set_ylabel("Count")
        ax.set_title(f"EM fit 3G+U (t={t:03d})  iters={n_iter}")
        if counts.max() > 5000:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bf_hist_emfit_t_{t:03d}.png"))
        plt.close(fig)

        try:
            arr = [centers, counts, mix_counts, uni_counts] + gauss_counts_s
            np.savetxt(
                os.path.join(out_dir, f"bf_hist_emfit_t_{t:03d}.csv"),
                np.column_stack(arr),
                fmt="%.6g,%d," + ",".join(["%.6g"] * (len(arr) - 2)),
                header="center,count,fit_mix,fit_uniform," + ",".join([f"fit_gauss{k}" for k in range(3)]),
                comments="",
            )
        except Exception:
            pass
    except Exception as e:
        print(f"[warn] EM-fit histogram plotting failed at t={t}: {e}")

    # # -------------------------
    # # Support from mode-centered Gaussian
    # # -------------------------
    # k_mode = int(np.argmin(np.abs(mus - mode_val)))
    # posterior_mode = gamma_mat[k_mode, :]
    # gamma_map = np.zeros_like(raw_crop, dtype=np.float32)
    # gamma_map[cropped_cell_mask] = posterior_mode.astype(np.float32)
    # sup_gauss = (gamma_map >= posterior_cutoff) & cropped_cell_mask
    # --- Support from ALL three Gaussians (i.e., the full segment) ---
    # Sum of Gaussian responsibilities (exclude uniform); typically ~1 inside the cell.
    gamma_sum_vec = gamma_mat[:3, :].sum(axis=0)  # shape (N_in_mask,)
    gamma_sum_map = np.zeros_like(raw_crop, dtype=np.float32)
    gamma_sum_map[cropped_cell_mask] = gamma_sum_vec.astype(np.float32)
    
    # If you truly want the *entire* segment, ignore posterior and use the mask directly:
    sup_gauss = cropped_cell_mask.copy()
    
    # (Optional: if you prefer a soft cutoff, uncomment next line)
    # sup_gauss = (gamma_sum_map >= 0.3) & cropped_cell_mask  # 0.3 is generous


   


    # -------------------------
    # Transform to (m,n) and score
    # -------------------------
    TOL_AXIS = 2.0
    if boundary_coords is None:
        ys, xs = np.nonzero(cropped_cell_mask)
        if ys.size == 0:
            return None
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        mid1_rc = (cy - 5.0, cx)
        mid2_rc = (cy + 5.0, cx)
    else:
        try:
            mid1_rc, mid2_rc = find_midpoints_on_minor_axis(
                boundary_coords=boundary_coords, P0=P0, major_direction=d, tol=TOL_AXIS
            )
        except Exception:
            got = False
            for tol_try in (TOL_AXIS * 1.5, TOL_AXIS * 2.5, TOL_AXIS * 4.0):
                try:
                    mid1_rc, mid2_rc = find_midpoints_on_minor_axis(
                        boundary_coords=boundary_coords, P0=P0, major_direction=d, tol=tol_try
                    )
                    got = True
                    break
                except Exception:
                    continue
            if not got:
                raise

    m_map, n_map = transform_to_mn_space(mid1_rc, mid2_rc, cropped_cell_mask, reflect=False)
    # -------------------------
    # Septum posterior from intensity x position
    # -------------------------
    # (A) INTENSITY term: darkest Gaussian's responsibility
    k_dark = int(np.argmin(mus))
    k_mode = int(np.argmin(np.abs(mus - mu0)))
    gamma_gauss = gamma_mat[:3, :]
    gamma_dark_vec = gamma_gauss[k_dark, :]
    
    gamma_dark_map = np.zeros_like(raw_crop, dtype=np.float32)
    gamma_dark_map[cropped_cell_mask] = gamma_dark_vec.astype(np.float32)
    
    # Support = pixels likely to be septum (darkest Gaussian) AND inside the segment
    #sup_dark = (gamma_dark_map >= posterior_cutoff) & cropped_cell_mask
    sup_dark = (gamma_dark_map >= 0.01) & cropped_cell_mask
    # Build an intensity map gated by the *dark* binary support used by pattern scoring
    support_values = raw_crop.astype(np.float32)
    support_values[~sup_dark] = np.nan  # NaN outside the dark support (ignored in sums)
    
    # (B) POSITION prior: Gaussian along n AND m
    # n is already centered around 0 by transform_to_mn_space (midline).
    sigma_n = float(max(sigma_n_px, 1e-3))
    
    # center m at the middle of the m-extent (same as your fixed template center)
    m_min, m_max = float(np.nanmin(m_map[cropped_cell_mask])), float(np.nanmax(m_map[cropped_cell_mask]))
    m_center = 0.5 * (m_min + m_max)
    sigma_m = float(max(sigma_m_px, 1e-3))
    
    pos_prior_n = np.exp(-0.5 * (n_map / sigma_n) ** 2).astype(np.float32)
    pos_prior_m = np.exp(-0.5 * ((m_map - m_center) / sigma_m) ** 2).astype(np.float32)
    pos_prior   = (pos_prior_n * pos_prior_m).astype(np.float32)
    
    # (C) Combined unnormalized posterior weights on the crop
    w_combined = gamma_dark_map * pos_prior
    w_combined *= cropped_cell_mask.astype(np.float32)
    
    # Guard against degenerate all-zero
    w_sum = float(np.nansum(w_combined))
    if w_sum <= 0.0:
        septum_mean_intensity = float("nan")
        septum_weight_sum = 0.0
    else:
        septum_mean_intensity = float(np.nansum(w_combined * raw_crop) / w_sum)
        septum_weight_sum = w_sum
    
    # (D) Delta metric: mode-Gaussian mean minus septum mean intensity
    mu_mode_gauss = float(mus[k_mode])
    septum_delta_from_mode = float(mu_mode_gauss - septum_mean_intensity) if np.isfinite(septum_mean_intensity) else float("nan")
    
    # Optional debug saves
    try:
        os.makedirs(out_dir, exist_ok=True)
        gdm = gamma_dark_map.copy()
        gdm = (255 * (gdm - np.nanmin(gdm)) / max(np.nanmax(gdm) - np.nanmin(gdm), 1e-6)).astype(np.uint8)
        imsave(os.path.join(out_dir, f"bf_gamma_dark_t_{t:03d}.png"), gdm)
    
        ppm_n = (255 * (pos_prior_n - np.nanmin(pos_prior_n)) / max(np.nanmax(pos_prior_n) - np.nanmin(pos_prior_n), 1e-6)).astype(np.uint8)
        imsave(os.path.join(out_dir, f"bf_pos_prior_n_t_{t:03d}.png"), ppm_n)
    
        ppm_m = (255 * (pos_prior_m - np.nanmin(pos_prior_m)) / max(np.nanmax(pos_prior_m) - np.nanmin(pos_prior_m), 1e-6)).astype(np.uint8)
        imsave(os.path.join(out_dir, f"bf_pos_prior_m_t_{t:03d}.png"), ppm_m)
    
        wv = (255 * (w_combined - np.nanmin(w_combined)) / max(np.nanmax(w_combined) - np.nanmin(w_combined), 1e-6)).astype(np.uint8)
        imsave(os.path.join(out_dir, f"bf_septum_weights_t_{t:03d}.png"), wv)
    except Exception as _e:
        print(f"[warn] septum posterior debug save failed at t={t:03d}: {_e}")


   # Fix m-center to the middle of the segment in m
    m_min = float(np.nanmin(m_map[cropped_cell_mask]))
    m_max = float(np.nanmax(m_map[cropped_cell_mask]))
    m_center = 0.5 * (m_min + m_max)
    
    # Binary support = darkest Gaussian posterior above cutoff (within the cell)
    sup_dark = (gamma_dark_map >= posterior_cutoff) & cropped_cell_mask
    #sup_dark = (~(gamma_dark_map >= posterior_cutoff)) & cropped_cell_mask
    
    # Scan with n-only movement: pass m_center_fixed
    pat = pattern_score_split_rectangles(
        support_crop=sup_dark,                 # full dark support, no m strip
        cropped_cell_mask=cropped_cell_mask,
        midpoint1_rc=mid1_rc,
        midpoint2_rc=mid2_rc,
        side_px=side_px,
        stride=1,
        limit_n_half=None,                     # or set a pixel cap if you want tighter n-band
        rect_n_frac=0.5,
        rect_m_frac=0.35,
        gap_frac=0.30,
        swap_axes=True,
        m_center_fixed=m_center               # <-- constrain center to vary on n only
    )



    pat_int = None  # we’re already using intensity; keep field for compatibility


    
    if pat is None:
        # Fallback struct (overlay will show "No valid center")
        pat = {
            "best_score_raw":   0.0,
            "best_score_norm":  0.0,
            "best_center_rc":   None,
            "best_center_mn":   None,
            "side_px":          side_px,
            "evaluated_centers": 0,
            "filtered_by_center_n": False,
            "n_length_px": None,
            "band_half_px": side_px / 2.0,
        }
    else:
        # Optional: compute single-center intensity score at the chosen center
        pat_int = _score_split_rectangles_intensity(
            values_img=support_values,
            m_map=m_map,
            n_map=n_map,
            center_mn=pat.get("best_center_mn", None),
            side_px=side_px,
        )
    
    # Optional: save a quick debug view of the weighted support
    try:
        dbg_w = np.nan_to_num(support_values, nan=0.0)
        dbg_w = dbg_w - np.nanmin(dbg_w)
        rng = float(np.nanmax(dbg_w))
        if rng > 0:
            dbg_w = np.clip((dbg_w / rng) * 255.0, 0, 255).astype(np.uint8)
        else:
            dbg_w = np.zeros_like(dbg_w, dtype=np.uint8)
        imsave(os.path.join(out_dir, f"bf_support_intensity_t_{t:03d}.png"), dbg_w)
    except Exception as _e:
        print(f"[warn] intensity support debug save failed at t={t}: {_e}")


    # Optional overlay
    try:
        dbg_png = os.path.join(out_dir, f"pattern_overlay_BF_t_{t:03d}.png")
        save_split_rectangles_pattern_overlay(
            support_crop=sup_dark,                  # show full dark support in overlay
            cropped_cell_mask=cropped_cell_mask,
            midpoint1_rc=mid1_rc,
            midpoint2_rc=mid2_rc,
            best_result=pat,
            out_path=dbg_png,
            side_px=side_px,
            bg_image=raw_crop.astype(np.float32),
            gfp_min=None,
            gfp_max=None,
        )

    except Exception as _e:
        print(f"[warn] BF pattern overlay failed at t={t:03d}: {_e}")

    cid = None if cell_id is None else str(cell_id)

    # Save quick debug images
    def _save_u8(arr, path):
        a = np.asarray(arr, np.float32)
        a = a - np.nanmin(a)
        rng = float(np.nanmax(a))
        im = np.zeros_like(a, dtype=np.uint8) if (not np.isfinite(rng) or rng <= 0) \
             else np.clip((a / rng) * 255.0, 0, 255).astype(np.uint8)
        imsave(path, im)
        
    def _save_bin(mask_bool, path):
        imsave(path, (np.asarray(mask_bool, bool).astype(np.uint8) * 255))


    try:
        os.makedirs(out_dir, exist_ok=True)
    
        # 1) Raw cropped BF for visual reference
        _save_u8(raw_crop, os.path.join(out_dir, f"bf_cropped_img_RAW_t_{t:03d}.png"))
    
        # 2) Responsibilities (dark Gaussian only) – this is what was thresholded
        _save_u8(gamma_dark_map, os.path.join(out_dir, f"bf_gamma_dark_t_{t:03d}.png"))
    
        # 3) Positional priors that were multiplied with gamma_dark for the septum metric
        #_save_u8(pos_prior_n, os.path.join(out_dir, f"bf_pos_prior_n_t_{t:03d}.png"))
        #_save_u8(pos_prior_m, os.path.join(out_dir, f"bf_pos_prior_m_t_{t:03d}.png"))
    
        # 4) Combined weights actually used to compute ⟨I⟩_septum
        _save_u8(w_combined, os.path.join(out_dir, f"bf_septum_weights_t_{t:03d}.png"))
    
        # 5) Binary support used by pattern scoring (thresholded dark gamma)
        _save_bin(sup_dark, os.path.join(out_dir, f"bf_support_dark_binary_t_{t:03d}.png"))
    
        # 6) n-only search band (the constrained candidate region actually scanned)
        #_save_bin(sup_dark, os.path.join(out_dir, f"bf_support_nscan_band_t_{t:03d}.png"))
    
        # 7) Intensity gated by the dark support (used for any optional intensity-at-center calc)
        _save_u8(np.nan_to_num(support_values, nan=0.0),
                 os.path.join(out_dir, f"bf_support_intensity_dark_t_{t:03d}.png"))
    
    except Exception as _e:
        print(f"[warn] debug saves failed at t={t:03d}: {_e}")


    return {
        "cell_id": cid,
        "time_point": t,
        "channel": "bf",
        "touches_border": bool(touches_border(seg_bool)),
        "pattern_score_raw":   pat["best_score_raw"],
        "pattern_score_norm":  pat["best_score_norm"],
        "pattern_center_row":  None if pat["best_center_rc"] is None else pat["best_center_rc"][0],
        "pattern_center_col":  None if pat["best_center_rc"] is None else pat["best_center_rc"][1],
        "pattern_center_m":    None if pat["best_center_mn"] is None else pat["best_center_mn"][0],
        "pattern_center_n":    None if pat["best_center_mn"] is None else pat["best_center_mn"][1],
        "pattern_side_px":     pat["side_px"],
        "pattern_evalN":       pat["evaluated_centers"],
        "pattern_filtered":    pat.get("filtered_by_center_n", False),
        "pattern_n_length_px": pat.get("n_length_px", None),
        "pattern_band_half_px":pat.get("band_half_px", None),
        "pattern_score_raw_intensity":  None if pat_int is None else pat_int["best_score_raw_intensity"],
        "pattern_score_norm_intensity": None if pat_int is None else pat_int["best_score_norm_intensity"],
        "septum_mean_intensity":    septum_mean_intensity,
        "septum_weight_sum":        septum_weight_sum,           # total mass of the combined posterior
        "septum_delta_from_mode":   septum_delta_from_mode,      # μ_mode - ⟨I⟩_septum  (bigger ⇒ darker septum)
        "gaussian_mode_mean":       mu_mode_gauss,               # for convenience / logging
        "gaussian_dark_mean":       float(mus[k_dark]),


    }
