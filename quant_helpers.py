#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:52:43 2025

@author: user
"""

# quant_helpers.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.io import imsave
from skimage.segmentation import find_boundaries
from skimage.draw import line
from skimage.morphology import dilation, disk
#sys.path.append('/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC')
# --- required functions from your quantification module
from Image_quantification_functions import (
    ImageQuantification,
    plot_cell_and_gamma_overlay,
    transform_to_mn_space,
)

# Optional (guarded) debug util; ok if not present
try:
    from Image_quantification_functions import save_axis_distance_scatter_from_axis
except Exception:
    def save_axis_distance_scatter_from_axis(*args, **kwargs):
        return None


# -------------------------------------------------------------------------
# Public helpers
# -------------------------------------------------------------------------

def extract_midpoints_rc_from_plot_data(plot_data):
    """
    Return (mid1_rc, mid2_rc) from plot_data in (row, col).
    Falls back to regionprops on the cropped mask if missing.
    """
    try:
        mid1_xy = plot_data[10]  # (x, y)
        mid2_xy = plot_data[11]  # (x, y)
        return (float(mid1_xy[1]), float(mid1_xy[0])), (float(mid2_xy[1]), float(mid2_xy[0]))
    except Exception:
        cropped_cell_mask = np.asarray(plot_data[3], bool)
        r = regionprops(label(cropped_cell_mask.astype(np.uint8)))[0]
        cy, cx = r.centroid
        theta = getattr(r, "orientation", 0.0) or 0.0
        # minor-axis direction (rows, cols) = rotate major axis by 90°
        vy, vx = np.cos(theta), -np.sin(theta)
        a_minor = getattr(r, "minor_axis_length", 0.0) / 2.0
        mid1_rc = (cy - a_minor * vy, cx - a_minor * vx)
        mid2_rc = (cy + a_minor * vy, cx + a_minor * vx)
        return mid1_rc, mid2_rc


def combine_gammas_prob(plot_data, include_keys=('Y2Z0','Y2Z1','Y2Z2','Y2Z3','Y2Z4','Y2Z5')):
    """
    From ImageQuantification plot_data, sum per-pixel responsibilities (gammas)
    for the specified include_keys. Returns (prob_crop, kept_keys).
    """
    gammas_unlinked   = plot_data[2]
    cropped_cell_mask = np.asarray(plot_data[3], bool)
    y_idx             = np.asarray(plot_data[4], int)
    x_idx             = np.asarray(plot_data[5], int)

    kept = [k for k in include_keys if k in gammas_unlinked]
    if not kept:
        return np.zeros_like(cropped_cell_mask, dtype=np.float32), kept

    p_vec = None
    for k in kept:
        gk = np.asarray(gammas_unlinked[k], dtype=float)
        p_vec = gk if p_vec is None else (p_vec + gk)

    prob_crop = np.zeros_like(cropped_cell_mask, dtype=np.float32)
    if p_vec is not None and p_vec.size == y_idx.size:
        prob_crop[y_idx, x_idx] = p_vec.astype(np.float32)
    return prob_crop, kept


def prob_to_support_mask_crop(prob_crop, mode='adaptive', min_frac_active=0.002):
    """
    Probability (crop) -> boolean support (crop).
    - mode='adaptive' (default): percentile on nonzero prob
    - mode='nonzero' : include any pixel with prob > 0.0

    Fills small holes before returning.
    """
    p = np.asarray(prob_crop, float)
    min_area = max(8, int(min_frac_active * p.size))

    if mode == 'nonzero':
        m = (p > 0.0)
        if m.any():
            m = remove_small_holes(m.astype(bool), area_threshold=min_area)
        return m.astype(bool)

    nz = p[p > 0]
    if nz.size == 0:
        return np.zeros_like(p, dtype=bool)

    for q in (60.0, 55.0, 50.0, 45.0, 40.0):
        thr = np.percentile(nz, q)
        m   = (p >= thr)
        if m.sum() >= max(8, int(min_frac_active * p.size)):
            m = remove_small_holes(m.astype(bool), area_threshold=min_area)
            return m.astype(bool)

    m = (p >= np.percentile(nz, 35.0))
    if m.any():
        m = remove_small_holes(m.astype(bool), area_threshold=min_area)
    return m.astype(bool)


def save_prob_and_support_debug(prob_crop, support_crop, out_dir, t, prefix="crop"):
    """
    Save prob_crop and support_crop as images + a side-by-side debug figure.
    """
    os.makedirs(out_dir, exist_ok=True)

    p = np.asarray(prob_crop, dtype=float)
    s = np.asarray(support_crop, dtype=bool)

    # file paths
    path_prob_raw   = os.path.join(out_dir, f"{prefix}_prob_t_{t:03d}.png")
    path_support_bw = os.path.join(out_dir, f"{prefix}_support_t_{t:03d}.png")
    path_panel      = os.path.join(out_dir, f"{prefix}_prob_support_panel_t_{t:03d}.png")

    # grayscale prob
    p_min, p_max = np.nanmin(p), np.nanmax(p)
    if not np.isfinite(p_min) or not np.isfinite(p_max) or (p_max <= p_min):
        p_norm = np.zeros_like(p, dtype=np.uint8)
    else:
        p01 = (p - p_min) / (p_max - p_min)
        p_norm = np.clip(p01 * 255.0, 0, 255).astype(np.uint8)
    imsave(path_prob_raw, p_norm)

    # binary support
    imsave(path_support_bw, (s.astype(np.uint8) * 255))

    # side-by-side panel
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=150)
    ax0, ax1 = axes
    im0 = ax0.imshow(p, cmap="magma", origin="upper", interpolation="nearest")
    ax0.set_title("prob_crop (sum gammas)", fontsize=10)
    ax0.set_axis_off()
    cb = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

    ax1.imshow(s, cmap="gray", origin="upper", interpolation="nearest")
    ax1.set_title("support_crop (binary)", fontsize=10)
    ax1.set_axis_off()

    fig.tight_layout()
    fig.savefig(path_panel, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def touching_circles_score_at_center(m_vals, n_vals, side):
    """
    Per-center template score on support points in (m,n) shifted to center (0,0).
    Two facing semicircles of radius side/2 along the n-axis inside a square of side.
    +1 inside semicircles, -1 inside square but outside semicircles, weighted by
    linear radial falloff to 0 at square edge.
    """
    S = float(side)
    h = S * 0.5

    in_sq = (np.abs(m_vals) <= h) & (np.abs(n_vals) <= h)
    in_top    = (m_vals**2 + (n_vals - h)**2) <= (h*h)
    in_bottom = (m_vals**2 + (n_vals + h)**2) <= (h*h)
    in_semis  = (in_top | in_bottom) & in_sq

    base = np.where(in_semis, 1.0, np.where(in_sq, -1.0, 0.0))

    r = np.sqrt(m_vals**2 + n_vals**2)
    w = np.clip(1.0 - (r / h), 0.0, 1.0)

    val = base * w
    raw = float(val.sum())
    den = float(np.maximum(w[in_sq].sum(), 1.0))
    norm = raw / den
    return raw, norm, int(in_sq.sum())


def pattern_score_touching_circles(
    support_crop,            # bool mask (crop coords)
    cropped_cell_mask,       # bool mask (crop coords)
    midpoint1_rc, midpoint2_rc,
    *,
    side_px=50,
    stride=1,                # scan every `stride`-th support pixel
    limit_n_half=None        # optional tighter band around n=0 (px)
):
    """
    Scan the support in (m,n) space with the 'touching semicircles' template.
    Hard-prefilter candidate centers to the central ±5% of the segment's n-extent,
    then choose the best by normalized score.
    """
    sup = np.asarray(support_crop, bool)
    seg = np.asarray(cropped_cell_mask, bool)

    if sup.sum() == 0 or seg.sum() == 0:
        return {
            'best_score_raw': 0.0, 'best_score_norm': 0.0,
            'best_center_rc': None, 'best_center_mn': None,
            'side_px': float(side_px), 'evaluated_centers': 0,
            'n_length_px': 0.0, 'band_half_px': 0.0, 'filtered_by_center_n': False
        }

    m_map, n_map = transform_to_mn_space(midpoint1_rc, midpoint2_rc, seg, reflect=False)

    # n-extent → ±5% band
    n_vals_seg = n_map[seg]
    n_len = float(np.max(n_vals_seg) - np.min(n_vals_seg)) if n_vals_seg.size else 0.0
    band_half = max(0.05 * n_len, 1.0)
    if limit_n_half is not None:
        band_half = min(band_half, float(limit_n_half))

    rr, cc = np.nonzero(sup)
    if stride > 1 and rr.size > 0:
        rr = rr[::stride]; cc = cc[::stride]
    if rr.size:
        keep = np.abs(n_map[rr, cc]) <= band_half
        rr = rr[keep]; cc = cc[keep]
    if rr.size == 0:
        return {
            'best_score_raw': 0.0, 'best_score_norm': 0.0,
            'best_center_rc': None, 'best_center_mn': None,
            'side_px': float(side_px), 'evaluated_centers': 0,
            'n_length_px': n_len, 'band_half_px': band_half, 'filtered_by_center_n': False
        }

    ms = m_map[sup].astype(np.float64)
    ns = n_map[sup].astype(np.float64)

    best = (-np.inf, -np.inf, None, None)  # (raw, norm, (r,c), (m0,n0))
    for r0, c0 in zip(rr, cc):
        m0 = float(m_map[r0, c0])
        n0 = float(n_map[r0, c0])
        raw, norm, _ = touching_circles_score_at_center(ms - m0, ns - n0, side_px)
        if norm > best[1]:
            best = (raw, norm, (int(r0), int(c0)), (m0, n0))

    return {
        'best_score_raw': best[0],
        'best_score_norm': best[1],
        'best_center_rc': best[2],
        'best_center_mn': best[3],
        'side_px': float(side_px),
        'evaluated_centers': int(rr.size),
        'n_length_px': n_len,
        'band_half_px': band_half,
        'filtered_by_center_n': False
    }


def save_touching_circles_pattern_overlay(
    support_crop,                   # bool mask (crop coords)
    cropped_cell_mask,              # bool mask (crop coords)
    midpoint1_rc, midpoint2_rc,     # (row, col) midpoints (crop coords)
    best_result,                    # dict from pattern_score_touching_circles
    out_path,
    *,
    side_px=30,
    bg_image=None,
    gfp_min=None, gfp_max=None,
    alpha_scale=0.45
):
    """
    Visualize the winning touching-circles template overlapped with the support.
    Green: +1 inside two semicircles; Red: −1 inside square but outside semicircles.
    Opacity ∝ central weighting.
    """
    sup = np.asarray(support_crop, bool)
    seg = np.asarray(cropped_cell_mask, bool)
    Hc, Wc = seg.shape
    r = float(side_px) / 2.0

    # Empty / no center → simple view
    if sup.sum() == 0 or seg.sum() == 0 or best_result.get('best_center_rc') is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        if bg_image is not None:
            if gfp_min is not None and gfp_max is not None and gfp_max > gfp_min:
                disp = np.clip((bg_image.astype(np.float32) - gfp_min) / (gfp_max - gfp_min), 0, 1)
            else:
                disp = bg_image
            ax.imshow(disp, cmap='gray', origin='upper', interpolation='nearest')
        else:
            ax.imshow(sup.astype(float), cmap='gray', origin='upper', interpolation='nearest')
        ax.set_axis_off()
        ax.set_title("No valid pattern center", fontsize=10)
        fig.tight_layout(pad=0)
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return

    m_map, n_map = transform_to_mn_space(midpoint1_rc, midpoint2_rc, seg, reflect=False)
    r0, c0 = best_result['best_center_rc']
    m0, n0 = best_result['best_center_mn'] if best_result.get('best_center_mn') is not None else (m_map[r0, c0], n_map[r0, c0])

    m_off = m_map - float(m0)
    n_off = n_map - float(n0)

    square = (np.abs(m_off) <= r) & (np.abs(n_off) <= r) & seg
    circ_top    = (m_off**2 + (n_off - r)**2) <= (r*r)
    circ_bottom = (m_off**2 + (n_off + r)**2) <= (r*r)
    # inward-facing halves
    semi_top    = square & circ_top    & (n_off <= 0)
    semi_bottom = square & circ_bottom & (n_off >= 0)

    pos_mask = sup & (semi_top | semi_bottom)
    neg_mask = sup & (square & ~(semi_top | semi_bottom))

    rad = np.sqrt(m_off**2 + n_off**2)
    weight = np.zeros_like(m_off, dtype=np.float32)
    inside = (rad <= r) & square
    weight[inside] = 1.0 - (rad[inside] / r)

    overlay = np.zeros((Hc, Wc, 4), dtype=np.float32)
    overlay[pos_mask, 1] = 1.0
    overlay[pos_mask, 3] = np.clip(alpha_scale * weight[pos_mask], 0.0, 1.0)
    overlay[neg_mask, 0] = 1.0
    overlay[neg_mask, 3] = np.clip(alpha_scale * weight[neg_mask], 0.0, 1.0)

    # Display
    if bg_image is not None:
        if gfp_min is not None and gfp_max is not None and gfp_max > gfp_min:
            disp = np.clip((bg_image.astype(np.float32) - gfp_min) / (gfp_max - gfp_min), 0, 1)
        else:
            disp = bg_image
        bg_to_show = disp; cmap_bg = 'gray'
    else:
        bg_to_show = sup.astype(float); cmap_bg = 'gray'

    fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=160)
    ax.imshow(bg_to_show, cmap=cmap_bg, origin='upper', interpolation='nearest')
    ax.imshow(overlay, origin='upper', interpolation='nearest')

    # Outline support & template
    def _outline(mask, color, lw=1.0, ls='-'):
        try:
            for c in find_contours(mask.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw, linestyle=ls, alpha=0.9)
        except Exception:
            pass

    try:
        for c in find_contours(sup.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], linewidth=1.0, color='yellow', alpha=0.6)
    except Exception:
        pass

    _outline(square, color='cyan', lw=1.0, ls='--')
    _outline(semi_top, color='lime', lw=1.2, ls='-')
    _outline(semi_bottom, color='lime', lw=1.2, ls='-')

    # center marker
    ax.plot([best_result['best_center_rc'][1]], [best_result['best_center_rc'][0]],
            marker='x', color='white', ms=6, mew=1.2)

    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def axis_from_plot_data(plot_data, fallback_mask=None):
    """
    Return (center_rc, axis_unit_rc, half_len_a) from endpoints in plot_data.
    Fallback to region moments on fallback_mask if needed.
    """
    center_rc = axis_unit_rc = half_len_a = None
    try:
        ep1_xy = np.asarray(plot_data[8], dtype=float)  # (x,y)
        ep2_xy = np.asarray(plot_data[9], dtype=float)
        ep1_rc = np.array([ep1_xy[1], ep1_xy[0]], dtype=float)
        ep2_rc = np.array([ep2_xy[1], ep2_xy[0]], dtype=float)

        v = ep2_rc - ep1_rc
        L = float(np.linalg.norm(v))
        if L > 1e-6:
            axis_unit_rc = (v / L)
            half_len_a = 0.5 * L
            center_rc = 0.5 * (ep1_rc + ep2_rc)
    except Exception:
        pass

    if (center_rc is None or axis_unit_rc is None or half_len_a is None) and fallback_mask is not None:
        lab = label(np.asarray(fallback_mask, bool).astype(np.uint8))
        regs = regionprops(lab)
        if regs:
            r = regs[0]
            cy, cx = r.centroid
            theta = getattr(r, "orientation", 0.0) or 0.0
            theta_x = theta + np.pi / 2.0
            vy, vx = np.sin(theta_x), np.cos(theta_x)
            center_rc = np.array([cy, cx], float)
            axis_unit_rc = np.array([vy, vx], float)
            half_len_a = 0.5 * float(getattr(r, "major_axis_length", 0.0) or 0.0)

    return center_rc, axis_unit_rc, half_len_a



def _nearest_boundary_bridge(mask_a: np.ndarray, mask_b: np.ndarray, bridge_width: int = 1):
    """Return a boolean bridge connecting mask_a and mask_b via the
    shortest line between their boundary pixels."""
    H, W = mask_a.shape
    ba = np.column_stack(np.nonzero(find_boundaries(mask_a, mode='inner')))
    bb = np.column_stack(np.nonzero(find_boundaries(mask_b, mode='inner')))
    if ba.size == 0 or bb.size == 0:
        return np.zeros((H, W), bool)

    # Try fast KD-tree; fall back to brute force if SciPy isn't installed
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(bb)
        dists, idx = tree.query(ba)
        i = int(np.argmin(dists))
        p0 = ba[i]
        p1 = bb[int(idx[i])]
    except Exception:
        # Brute force (fine for typical cell masks)
        diffs = ba[:, None, :] - bb[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        p0 = ba[int(i)]
        p1 = bb[int(j)]

    rr, cc = line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
    bridge = np.zeros((H, W), bool)
    bridge[rr, cc] = True
    if bridge_width > 1:
        bridge = dilation(bridge, disk(max(1, bridge_width // 2)))
    return bridge

def merge_disconnected_components(seg: np.ndarray, keep: int = 2, bridge_width: int = 1) -> np.ndarray:
    """
    If seg has multiple components, connect the largest two with a minimal bridge.
    Returns a boolean mask with the bridge added. Shape is unchanged.
    """
    seg = np.asarray(seg, bool)
    lab = label(seg, connectivity=2)
    props = regionprops(lab)
    if len(props) <= 1:
        return seg

    # Take the largest 'keep' components
    props = sorted(props, key=lambda r: r.area, reverse=True)[:keep]
    comps = [(lab == p.label) for p in props]
    merged = np.logical_or.reduce(comps)

    if len(comps) >= 2:
        bridge = _nearest_boundary_bridge(comps[0], comps[1], bridge_width=bridge_width)
        merged = merged | bridge

    # Keep any remaining smaller components as-is if you want:
    # merged = merged | (seg & ~np.logical_or.reduce([(lab == p.label) for p in props]))
    # (Usually not needed for a single-cell track.)

    return merged

def smooth_edges(mask: np.ndarray, radius: int = 1, passes: int = 1) -> np.ndarray:
    fp = disk(radius)  # radius=1 or 2 is usually enough
    out = mask.astype(bool)
    for _ in range(passes):
        out = binary_closing(out, footprint=fp)
        out = binary_opening(out, footprint=fp)
    return out

def quantify_one_object(
    img, mask_bool, id_suffix, t,
    plot_dir, ep_refs,
    gfp_min, gfp_max, cell_id,
    *,
    do_plot=True,
    touches_border_flag=False,
    allow_split=True,    # When False (children), only single-object quant
    extra_cols=None
):
    """
    Top-level quantification:
      1) Quantify the segment as ONE object (emit 1 row).
      2) Run touching-circles pattern recognition (save overlay + scores in row).
      3) Split by the MINOR-axis line through the pattern center (n=n0),
         paste split masks back to full frame, and quantify TWO halves (emit 2 rows).
    For child calls (allow_split=False): only step (1). Returns: list of row dicts.
    """
    rows_out = []
    os.makedirs(plot_dir, exist_ok=True)

    key = 'single' if id_suffix == '' else id_suffix
    out_cell_id = f"{cell_id}" if id_suffix == '' else f"{cell_id}_{id_suffix}"

    seg_bool = np.asarray(mask_bool, bool)
    
    # Get region properties from the cropped mask.

    # make sure it's boolean
    
    
    # plt.figure()
    # plt.imshow(seg_bool, cmap='gray', interpolation='none')
    # plt.title('cropped_cell_mask')
    # plt.axis('off')
    # plt.show()
    

    if seg_bool.sum() == 0:
        print(f"[warn] {out_cell_id} empty mask at t={t}; skipping")
        return rows_out
    # NEW: merge the two largest components if they’re disconnected
    seg_bool_merged = merge_disconnected_components(seg_bool, keep=2, bridge_width=1)

    # Use the merged mask for bbox and downstream quantification
    _lab_full = label(seg_bool_merged.astype(np.uint8), connectivity=2)
    _props_full = regionprops(_lab_full)
    
    #print(len(_props_full))
    if not _props_full:
        print(f"[warn] {out_cell_id} no region props at t={t}; skipping")
        return rows_out
    # Union bbox (robust even if something else slips in)
    min_row = min(p.bbox[0] for p in _props_full)
    min_col = min(p.bbox[1] for p in _props_full)
    max_row = max(p.bbox[2] for p in _props_full)
    max_col = max(p.bbox[3] for p in _props_full)

    # EM (always run for gammas)
    ref_ep1 = ep_refs.get(key, {}).get('ep1', None)
    ref_ep2 = ep_refs.get(key, {}).get('ep2', None)
    is_new_track = (ref_ep1 is None) or (ref_ep2 is None)
    t_em = 0 if is_new_track else t

    # Grab previous parameters
    prev_params = ep_refs.get(key, {}).get('prev_params', None)


    # And pass the merged mask into ImageQuantification
    par, par_fixed, plot_data, ep1_new, ep2_new = ImageQuantification(
        fluorescent_img=img,
        cell_mask=seg_bool_merged,  # <— merged
        selected_label=out_cell_id,
        C1max=gfp_max,
        C1min=gfp_min,
        tp=t_em,
        ref_ep1=ref_ep1,
        ref_ep2=ref_ep2,
        skip_em=False,
        init_params_unlinked=prev_params,   # NEW
        init_blend=0.7,                     # NEW (0..1): how much to trust t-1 init
    )
    ep_refs.setdefault(key, {})['ep1'] = ep1_new
    ep_refs.setdefault(key, {})['ep2'] = ep2_new
    ep_refs.setdefault(key, {})['prev_params'] = par  # NEW: carry to t+1

    if is_new_track and t != 0:
        print(f"[info] Initialized EM refs for track {out_cell_id} at global t={t} (t_em=0)")

    # Crop-world data from ImageQuantification
    cropped_img        = plot_data[0]
    cropped_cell_mask  = np.asarray(plot_data[3], bool)
    prob_crop, kept_keys = combine_gammas_prob(plot_data)
    support_crop = prob_to_support_mask_crop(prob_crop, mode='adaptive')
    save_prob_and_support_debug(prob_crop, support_crop, plot_dir, t, prefix=f"{out_cell_id}")

    # (1) Single-object row
    parent_row = {
        'cell_id': out_cell_id,
        'time_point': t,
        'cell_length': par_fixed.get('major_axis_length', None),
        'cell_area': par_fixed.get('area', None),
        'nu_dis': par.get('mu_mn_Y2', [None, None])[1],
        'nu_int': par.get('mu_I_Y2', None),
        'cyt_int': par.get('mu_bg_Y2', None),
        'septum_int': None if not par.get('mu_S1_Y2') or not par.get('mu_S2_Y2')
                        else (par['mu_S1_Y2'] + par['mu_S2_Y2']) / 2,
        'pol1_int': par.get('mu_P1_Y2', None),
        'pol2_int': par.get('mu_P2_Y2', None),
        'touches_border': bool(touches_border_flag),
        'ellipse_model': 1, 'ellipse_angle_deg': None, 'iou_single': None, 'iou_double': None,
    }
    if isinstance(extra_cols, dict):
        parent_row.update(extra_cols)
    rows_out.append(parent_row)

    # Optional EM overlay
    if do_plot:
        try:
            plot_file = os.path.join(plot_dir, f"frame_t_{t:03d}.png")
            plot_cell_and_gamma_overlay(plot_data, plot_filename=plot_file)
        except Exception as _e:
            print(f"[warn] overlay failed at t={t}: {_e}")

    # (2) Pattern recognition (store on parent row)
    mid1_rc, mid2_rc = extract_midpoints_rc_from_plot_data(plot_data)
    side_px = 60
    pat = pattern_score_touching_circles(
        support_crop, cropped_cell_mask, mid1_rc, mid2_rc,
        side_px=side_px, stride=1
    )
    parent_row.update({
        'pattern_score_raw':   pat['best_score_raw'],
        'pattern_score_norm':  pat['best_score_norm'],
        'pattern_center_row':  None if pat['best_center_rc']  is None else pat['best_center_rc'][0],
        'pattern_center_col':  None if pat['best_center_rc']  is None else pat['best_center_rc'][1],
        'pattern_center_m':    None if pat['best_center_mn']  is None else pat['best_center_mn'][0],
        'pattern_center_n':    None if pat['best_center_mn']  is None else pat['best_center_mn'][1],
        'pattern_side_px':     pat['side_px'],
        'pattern_evalN':       pat['evaluated_centers'],
        'pattern_filtered':    pat.get('filtered_by_center_n', False),
        'pattern_n_length_px': pat.get('n_length_px', None),
        'pattern_band_half_px':pat.get('band_half_px', None),
    })
    #print(pat['best_center_mn'][1])
    # Save pattern overlay
    try:
        dbg_png = os.path.join(plot_dir, f"pattern_overlay_t_{t:03d}.png")
        save_touching_circles_pattern_overlay(
            support_crop, cropped_cell_mask, mid1_rc, mid2_rc,
            pat, dbg_png, side_px=side_px,
            bg_image=cropped_img, gfp_min=gfp_min, gfp_max=gfp_max
        )
    except Exception as _e:
        print(f"[warn] pattern overlay failed at t={t}: {_e}")

    # Children do not split again
    if not allow_split:
        return rows_out

    # (3) Split by minor-axis line through the pattern center (n = n0)
    m_map, n_map = transform_to_mn_space(mid1_rc, mid2_rc, cropped_cell_mask, reflect=False)
    n0 = float(pat['best_center_mn'][1]) if pat.get('best_center_mn') is not None else 0.0

    mask1_crop = (n_map >= n0) & cropped_cell_mask
    mask2_crop = (n_map <  n0) & cropped_cell_mask
    # Remove tiny islands (<10 px) from each split
    MIN_AREA = 10
    mask1_crop = remove_small_objects(mask1_crop.astype(bool), min_size=MIN_AREA, connectivity=2)
    mask2_crop = remove_small_objects(mask2_crop.astype(bool), min_size=MIN_AREA, connectivity=2)
    # Smooth edges a little, keep binary
    mask1_crop = smooth_edges(mask1_crop, radius=2, passes=1)
    mask2_crop = smooth_edges(mask2_crop, radius=2, passes=1)

    # plt.figure()
    # plt.imshow(mask1_crop, cmap='gray', interpolation='none')
    # plt.title('cropped_cell_mask')
    # plt.axis('off')
    # plt.show()

    # Paste crop masks back into full-frame masks
    mask1_full = np.zeros_like(seg_bool := np.asarray(mask_bool, bool), dtype=bool)
    mask2_full = np.zeros_like(seg_bool, dtype=bool)
    mask1_full[min_row:max_row, min_col:max_col] = mask1_crop
    mask2_full[min_row:max_row, min_col:max_col] = mask2_crop

    # Child meta
    extra_child = {
        'ellipse_model': 2, 'ellipse_angle_deg': None, 'iou_single': None, 'iou_double': None,
        'pattern_parent_norm': pat['best_score_norm'],
        'split_n0': float(n0),
    }

    # subfolders per child
    child1_dir = os.path.join(plot_dir, f"{out_cell_id}_1"); os.makedirs(child1_dir, exist_ok=True)
    child2_dir = os.path.join(plot_dir, f"{out_cell_id}_2"); os.makedirs(child2_dir, exist_ok=True)

    # quantify child 1
    if mask1_full.sum() == 0:
        print(f"[warn] {out_cell_id}_1 empty after split at t={t}")
    else:
        rows_out += quantify_one_object(
            img, mask1_full, '1' if id_suffix == '' else f'{id_suffix}1', t,
            child1_dir, ep_refs, gfp_min, gfp_max, cell_id,
            do_plot=do_plot, touches_border_flag=touches_border_flag,
            allow_split=False, extra_cols=extra_child
        )

    # quantify child 2
    if mask2_full.sum() == 0:
        print(f"[warn] {out_cell_id}_2 empty after split at t={t}")
    else:
        rows_out += quantify_one_object(
            img, mask2_full, '2' if id_suffix == '' else f'{id_suffix}2', t,
            child2_dir, ep_refs, gfp_min, gfp_max, cell_id,
            do_plot=do_plot, touches_border_flag=touches_border_flag,
            allow_split=False, extra_cols=extra_child
        )

    return rows_out


__all__ = [
    "extract_midpoints_rc_from_plot_data",
    "combine_gammas_prob",
    "prob_to_support_mask_crop",
    "save_prob_and_support_debug",
    "touching_circles_score_at_center",
    "pattern_score_touching_circles",
    "save_touching_circles_pattern_overlay",
    "axis_from_plot_data",
    "quantify_one_object",
]





# ---------- core scorer (rotated + corrected signs) ----------

def split_rectangles_score_at_center(
    m_vals, n_vals, side,
    *,
    rect_n_frac=0.5,   # AFTER rotation: half-length of bars along m as a fraction of h (= side/2)
    rect_m_frac=0.35,  # AFTER rotation: bar thickness along n as a fraction of h
    gap_frac=0.10,     # AFTER rotation: half-gap between bars along n as a fraction of h
    swap_axes=True     # rotate 90° by swapping (m,n) for the template geometry
):
    """
    Per-center template score on (m,n) shifted to (0,0), with a square of side `side`.
    Rotated 90°: two horizontal rectangles (bars) along the m-axis, separated by a central split along n.

      • +1 inside the two rectangles (bars)
      • -1 ONLY in the central split between the rectangles
      •  0 elsewhere inside the square; 0 outside the square

    Weighted by linear radial falloff to 0 at the square edge (radius = h = side/2).

    Parameters (with swap_axes=True):
      - rect_n_frac -> half-length of bars along m  (|m| <= rect_n_frac*h)
      - rect_m_frac -> bar thickness along n       (gap_half <= |n| <= gap_half + rect_m_frac*h)
      - gap_frac    -> half-gap between bars along n (|n| < gap_half is the split)
    """
    S = float(side)
    h = S * 0.5

    # Square (always in original coords)
    in_sq = (np.abs(m_vals) <= h) & (np.abs(n_vals) <= h)

    # Use (u,v) for template geometry; with rotation we swap axes.
    if swap_axes:
        u, v = m_vals, n_vals  # bars run along u (= m), gap/thickness along v (= n)
    else:
        u, v = n_vals, m_vals  # no rotation (kept for flexibility)

    # Geometry parameters
    gap_half   = float(gap_frac) * h
    bar_thick  = float(rect_m_frac) * h
    bar_thick  = float(np.clip(bar_thick, 0.0, max(0.0, h - gap_half)))
    u_half     = float(np.clip(float(rect_n_frac) * h, 0.0, h))

    in_u_span      = (np.abs(u) <= u_half)                           # along-bar length
    in_v_bars      = (np.abs(v) >= gap_half) & (np.abs(v) <= gap_half + bar_thick)
    in_v_split     = (np.abs(v) <  gap_half)                         # central split

    bars_mask      = in_sq & in_u_span & in_v_bars
    split_mask     = in_sq & in_u_span & in_v_split

    # Base values: +1 in bars, -1 in split, 0 elsewhere in the square
    base = np.zeros_like(u, dtype=float)
    base[bars_mask]  =  -1 #1.0
    base[split_mask] = 1 #-1.0

    # Radial weight in original (m,n)
    r = np.sqrt(m_vals**2 + n_vals**2)
    w = np.clip(1.0 - (r / h), 0.0, 1.0)

    val = base * w
    raw = float(val.sum())
    den = float(np.maximum(w[in_sq].sum(), 1.0))
    norm = raw / den
    return raw, norm, int(in_sq.sum())


# ---------- scanner (uses rotated scorer) ----------

def pattern_score_split_rectangles(
    support_crop,            # bool mask (crop coords)
    cropped_cell_mask,       # bool mask (crop coords)
    midpoint1_rc, midpoint2_rc,
    *,
    side_px=50,
    stride=1,
    limit_n_half=None,
    rect_n_frac=0.5,
    rect_m_frac=0.35,
    gap_frac=0.10,
    swap_axes=True,
    m_center_fixed: float | None = None,   # <-- NEW
):
    """
    Scan the support in (m,n) with the rotated split-rectangles template.
    Candidate centers are prefiltered to a central band along n.
    If m_center_fixed is given, the template center's m is fixed there and
    only n varies across candidates.
    """
    sup = np.asarray(support_crop, bool)
    seg = np.asarray(cropped_cell_mask, bool)

    if sup.sum() == 0 or seg.sum() == 0:
        return {
            'best_score_raw': 0.0, 'best_score_norm': 0.0,
            'best_center_rc': None, 'best_center_mn': None,
            'side_px': float(side_px), 'evaluated_centers': 0,
            'n_length_px': 0.0, 'band_half_px': 0.0, 'filtered_by_center_n': False
        }

    m_map, n_map = transform_to_mn_space(midpoint1_rc, midpoint2_rc, seg, reflect=False)

    # n-extent → ±5% band (optionally tighter)
    n_vals_seg = n_map[seg]
    n_len = float(np.max(n_vals_seg) - np.min(n_vals_seg)) if n_vals_seg.size else 0.0
    band_half = max(0.05 * n_len, 1.0)
    if limit_n_half is not None:
        band_half = min(band_half, float(limit_n_half))

    rr, cc = np.nonzero(sup)
    if stride > 1 and rr.size > 0:
        rr = rr[::stride]; cc = cc[::stride]
    if rr.size:
        keep = np.abs(n_map[rr, cc]) <= band_half
        rr = rr[keep]; cc = cc[keep]
    if rr.size == 0:
        return {
            'best_score_raw': 0.0, 'best_score_norm': 0.0,
            'best_center_rc': None, 'best_center_mn': None,
            'side_px': float(side_px), 'evaluated_centers': 0,
            'n_length_px': n_len, 'band_half_px': band_half, 'filtered_by_center_n': False
        }

    ms = m_map[sup].astype(np.float64)
    ns = n_map[sup].astype(np.float64)

    best = (-np.inf, -np.inf, None, None)  # (raw, norm, (r,c), (m0,n0))
    for r0, c0 in zip(rr, cc):
        m0 = float(m_center_fixed) if (m_center_fixed is not None) else float(m_map[r0, c0])  # <-- only n varies
        n0 = float(n_map[r0, c0])
        raw, norm, _ = split_rectangles_score_at_center(
            ms - m0, ns - n0, side_px,
            rect_n_frac=rect_n_frac, rect_m_frac=rect_m_frac, gap_frac=gap_frac,
            swap_axes=swap_axes
        )
        if norm > best[1]:
            best = (raw, norm, (int(r0), int(c0)), (m0, n0))

    return {
        'best_score_raw': best[0],
        'best_score_norm': best[1],
        'best_center_rc': best[2],
        'best_center_mn': best[3],
        'side_px': float(side_px),
        'evaluated_centers': int(rr.size),
        'n_length_px': n_len,
        'band_half_px': band_half,
        'filtered_by_center_n': False
    }



# ---------- overlay (rotated + corrected signs) ----------

def save_split_rectangles_pattern_overlay(
    support_crop,                   # bool mask (crop coords)
    cropped_cell_mask,              # bool mask (crop coords)
    midpoint1_rc, midpoint2_rc,     # (row, col) midpoints (crop coords)
    best_result,                    # dict from pattern_score_split_rectangles
    out_path,
    *,
    side_px=30,
    rect_n_frac=0.5,
    rect_m_frac=0.35,
    gap_frac=0.10,
    swap_axes=True,
    bg_image=None,
    gfp_min=None, gfp_max=None,
    alpha_scale=0.45
):
    """
    Visualize the rotated split-rectangles template.
    Green: +1 inside bars; Red: −1 in the central split; 0 elsewhere in the square.
    Opacity ∝ radial falloff.
    """
    sup = np.asarray(support_crop, bool)
    seg = np.asarray(cropped_cell_mask, bool)
    Hc, Wc = seg.shape
    r = float(side_px) * 0.5  # h

    # Empty / no center → simple view
    if sup.sum() == 0 or seg.sum() == 0 or best_result.get('best_center_rc') is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        if bg_image is not None:
            if gfp_min is not None and gfp_max is not None and gfp_max > gfp_min:
                disp = np.clip((bg_image.astype(np.float32) - gfp_min) / (gfp_max - gfp_min), 0, 1)
            else:
                disp = bg_image
            ax.imshow(disp, cmap='gray', origin='upper', interpolation='nearest')
        else:
            ax.imshow(sup.astype(float), cmap='gray', origin='upper', interpolation='nearest')
        ax.set_axis_off()
        ax.set_title("No valid pattern center", fontsize=10)
        fig.tight_layout(pad=0)
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return

    m_map, n_map = transform_to_mn_space(midpoint1_rc, midpoint2_rc, seg, reflect=False)
    r0, c0 = best_result['best_center_rc']
    m0, n0 = best_result.get('best_center_mn', (m_map[r0, c0], n_map[r0, c0]))

    m_off = m_map - float(m0)
    n_off = n_map - float(n0)

    # Square region (also restrict by seg)
    square = (np.abs(m_off) <= r) & (np.abs(n_off) <= r) & seg

    # Use (u,v) for template geometry with rotation
    if swap_axes:
        u, v = m_off, n_off   # bars along u=m, split along v=n
    else:
        u, v = n_off, m_off   # no rotation (kept for flexibility)

    gap_half   = float(gap_frac) * r
    bar_thick  = float(rect_m_frac) * r
    bar_thick  = float(np.clip(bar_thick, 0.0, max(0.0, r - gap_half)))
    u_half     = float(np.clip(float(rect_n_frac) * r, 0.0, r))

    in_u_span  = (np.abs(u) <= u_half)
    in_v_bars  = (np.abs(v) >= gap_half) & (np.abs(v) <= gap_half + bar_thick)
    in_v_split = (np.abs(v) <  gap_half)

    rect_mask  = square & in_u_span & in_v_bars
    split_mask = square & in_u_span & in_v_split

    # positives / negatives on the support
    pos_mask = sup & split_mask #rect_mask
    neg_mask = sup & rect_mask #split_mask

    # radial weight for alpha (in original coords)
    rad = np.sqrt(m_off**2 + n_off**2)
    weight = np.zeros_like(m_off, dtype=np.float32)
    inside = (rad <= r) & square
    weight[inside] = 1.0 - (rad[inside] / r)

    overlay = np.zeros((Hc, Wc, 4), dtype=np.float32)
    overlay[pos_mask, 1] = 1.0
    overlay[pos_mask, 3] = np.clip(alpha_scale * weight[pos_mask], 0.0, 1.0)
    overlay[neg_mask, 0] = 1.0
    overlay[neg_mask, 3] = np.clip(alpha_scale * weight[neg_mask], 0.0, 1.0)

    # Display background
    if bg_image is not None:
        if gfp_min is not None and gfp_max is not None and gfp_max > gfp_min:
            disp = np.clip((bg_image.astype(np.float32) - gfp_min) / (gfp_max - gfp_min), 0, 1)
        else:
            disp = bg_image
        bg_to_show = disp; cmap_bg = 'gray'
    else:
        bg_to_show = sup.astype(float); cmap_bg = 'gray'

    fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=160)
    ax.imshow(bg_to_show, cmap=cmap_bg, origin='upper', interpolation='nearest')
    ax.imshow(overlay, origin='upper', interpolation='nearest')

    # helpful outlines
    def _outline(mask, color, lw=1.0, ls='-'):
        try:
            for c in find_contours(mask.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw, linestyle=ls, alpha=0.9)
        except Exception:
            pass

    try:
        for c in find_contours(sup.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], linewidth=1.0, color='yellow', alpha=0.6)
    except Exception:
        pass

    _outline(square, color='cyan', lw=1.0, ls='--')
    rect_mask_vis  = rect_mask  & sup
    split_mask_vis = split_mask & sup
    _outline(rect_mask_vis,  color='lime', lw=1.2, ls='-')
    _outline(split_mask_vis, color='red',  lw=1.0, ls=':')

    # center marker
    ax.plot([c0], [r0], marker='x', color='white', ms=6, mew=1.2)

    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
