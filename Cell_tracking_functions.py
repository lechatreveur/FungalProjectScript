#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:55:29 2025

@author: user
"""

# Cell_tracking_functions.py
import numpy as np
from itertools import combinations
from skimage.io import imread
from skimage.measure import label

# ------------------------------
# IO
# ------------------------------
def load_segmentation(path):
    """Load .npy or image mask (binary or labeled)."""
    return np.load(path) if path.endswith('.npy') else imread(path)

def to_labeled_current(seg):
    """
    Convert a raw segmentation to an instance-labeled map without collapsing cells.
    - If seg is bool or binary (0/1 or 0/255) => label(seg>0)
    - Else assume it's already instance-labeled (0..N)
    """
    u = np.unique(seg)
    if seg.dtype == bool:
        return label(seg)
    if u.size <= 3 and set(u).issubset({0, 1, 255}):
        return label(seg > 0)
    return seg.astype(np.int32)

def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Simple row-major RLE for boolean masks.
    Returns {'shape': (H, W), 'counts': [run_lengths...]}.
    """
    m = np.ascontiguousarray(mask.astype(np.uint8).ravel())
    counts = []
    run_val = m[0] if m.size else 0
    run_len = 0
    for v in m:
        if v == run_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_val = v
            run_len = 1
    if m.size:
        counts.append(run_len)
    return {'shape': mask.shape, 'counts': counts}

def rle_to_mask(rle: dict) -> np.ndarray:
    """
    Inverse of mask_to_rle. Returns a boolean mask of rle['shape'].
    """
    H, W = rle['shape']
    counts = rle['counts']
    # runs alternate [0s,1s,0s,1s,...], starting with zeros
    vals = []
    cur = 0
    for run in counts:
        vals.extend([cur] * run)
        cur = 1 - cur
    arr = np.array(vals, dtype=np.uint8)
    if arr.size < H * W:  # pad if needed
        arr = np.pad(arr, (0, H * W - arr.size), constant_values=0)
    return arr.reshape((H, W)).astype(bool)


# ------------------------------
# Geometry / scoring
# ------------------------------
def compute_overlap(mask1, mask2):
    """Intersection over reference (mask1) area."""
    s1 = mask1.sum()
    if s1 == 0:
        return 0.0
    return np.logical_and(mask1, mask2).sum() / float(s1)

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / float(union) if union > 0 else 0.0

def area_change_penalty(area, prev_area, ratio_soft=1.1, ratio_hard=1.3):
    """
    Penalize abrupt multiplicative area changes using |log(A_t/A_{t-1})|.
    Stronger penalty beyond ratio_hard, milder beyond ratio_soft.
    """
    if prev_area <= 0 or area <= 0:
        return 0.0
    r = area / float(prev_area)
    pen = abs(np.log(r))
    if r > ratio_hard or r < 1.0 / ratio_hard:
        pen *= 2.5
    elif r > ratio_soft or r < 1.0 / ratio_soft:
        pen *= 1.5
    return pen

def get_cell_mask_area_aware(
    labeled_current, ref_mask, prev_area,
    threshold=0.7,
    max_segments=2,
    topk=5,
    area_lambda=0.35,
    ratio_soft=1.3, ratio_hard=1.8,
    ratio_huge=2.0
):
    """
    Singles must pass overlap >= threshold AND area ratio within [1/ratio_huge, ratio_huge].
    If no acceptable single, consider pairs with same constraints.
    If a candidate is rejected for huge jump, reuse previous mask (penalty=0).

    Returns:
      mask, overlap, score(=overlap - area_lambda*penalty), penalty, huge_jump_rejected, meta

    meta schema:
      {
        'composition': 'single' | 'pair' | 'keep_prev',
        'segments_rle': [rle_a, rle_b]  # only when composition == 'pair'
        'segments_labels': (lbl_a, lbl_b)  # only when composition == 'pair'
      }
    """
    labels = [l for l in np.unique(labeled_current) if l != 0] if labeled_current is not None else []
    if not labels:
        meta = {'composition': 'keep_prev'}
        return ref_mask.copy(), 1.0, 1.0, 0.0, False, meta

    def _pen(area):
        return area_change_penalty(area, prev_area, ratio_soft, ratio_hard)

    def _huge(area):
        if prev_area <= 0:
            return False
        r = float(area) / float(prev_area)
        return (r > ratio_huge) or (r < 1.0 / ratio_huge)

    # Rank by overlap and keep top-K
    ranked = []
    for lbl in labels:
        cm = (labeled_current == lbl)
        ranked.append((lbl, compute_overlap(ref_mask, cm), int(cm.sum())))
    ranked.sort(key=lambda x: x[1], reverse=True)
    cand_labels = [lbl for (lbl, _, _) in ranked[:topk]]

    huge_reject_flag = False

    # 1) Singles
    singles = []
    for lbl in cand_labels:
        cm = (labeled_current == lbl)
        ov = compute_overlap(ref_mask, cm)
        a  = int(cm.sum())
        pen = _pen(a)
        sc  = ov - area_lambda * pen
        singles.append((cm, ov, sc, pen, (lbl,), _huge(a)))

    passing_singles = [c for c in singles if c[1] >= threshold]
    ok_singles      = [c for c in passing_singles if not c[5]]
    if ok_singles:
        cm, ov, sc, pen, (lbl,), _ = max(ok_singles, key=lambda x: x[2])
        meta = {'composition': 'single'}
        return cm.astype(bool), float(ov), float(sc), float(pen), False, meta
    if passing_singles and all(c[5] for c in passing_singles):
        huge_reject_flag = True

    # 2) Pairs (only if no acceptable single)
    pairs = []
    if max_segments >= 2:
        for a_lbl, b_lbl in combinations(cand_labels, 2):
            cm_a = (labeled_current == a_lbl)
            cm_b = (labeled_current == b_lbl)
            cm   = np.logical_or(cm_a, cm_b)
            ov   = compute_overlap(ref_mask, cm)
            if ov < threshold:
                continue
            a    = int(cm.sum())
            huge = _huge(a)
            pen  = _pen(a)
            sc   = ov - area_lambda * pen
            pairs.append((cm, ov, sc, pen, (a_lbl, b_lbl), huge, cm_a, cm_b))

    ok_pairs = [c for c in pairs if not c[5]]
    if ok_pairs:
        cm, ov, sc, pen, (lbl_a, lbl_b), _, cm_a, cm_b = max(ok_pairs, key=lambda x: x[2])
        meta = {
            'composition': 'pair',
            'segments_labels': (int(lbl_a), int(lbl_b)),
            'segments_rle': [mask_to_rle(cm_a.astype(bool)), mask_to_rle(cm_b.astype(bool))],
        }
        return cm.astype(bool), float(ov), float(sc), float(pen), False, meta
    if pairs and all(c[5] for c in pairs):
        huge_reject_flag = True

    # 3) Fallback: reuse previous (penalty=0)
    keep_cm = ref_mask.copy()
    keep_ov = 1.0
    keep_sc = keep_ov
    meta = {'composition': 'keep_prev'}
    return keep_cm, keep_ov, keep_sc, 0.0, huge_reject_flag, meta


# ------------------------------
# Misc
# ------------------------------
def touches_border(mask):
    return (np.any(mask[0, :]) or np.any(mask[-1, :]) or
            np.any(mask[:, 0]) or np.any(mask[:, -1]))

def rle_encode(mask):
    """Run-length encode a 2D boolean mask as 'start length start length ...' (1-based, Fortran order)."""
    flat = mask.ravel(order='F').astype(np.uint8)
    diffs = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.where(diffs == 1)[0] + 1
    ends   = np.where(diffs == -1)[0] + 1
    lengths = ends - starts
    out = []
    for s, l in zip(starts, lengths):
        out.append(str(s)); out.append(str(l))
    return " ".join(out)

def rle_decode(rle, shape):
    """Decode RLE string back to boolean mask with given shape (H, W)."""
    if rle is None or rle == "" or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros(shape, dtype=bool)
    nums = list(map(int, rle.strip().split()))
    starts = np.array(nums[0::2]) - 1
    lengths = np.array(nums[1::2])
    ends = starts + lengths
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    for s, e in zip(starts, ends):
        flat[s:e] = True
    return flat.reshape(shape, order='F')
