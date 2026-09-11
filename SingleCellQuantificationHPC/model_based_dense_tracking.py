#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-based dense tracking — pipeline stage 3 (P14).

Every interior frame between two curated keyframes is decided against an
explicit SHAPE MODEL built from those keyframes, instead of being repaired by
stacked heuristics.

The model
---------
A cell is three landmarks on its major axis — the two pole tips E1 and E2 and
the centre C — plus a stroke radius r (half the cell width).  The mask is the
union of two round-capped strokes, E1->C and E2->C, each started one radius in
from its tip so the cap lands ON the tip rather than a radius past it.  The two
strokes hinge at C by an angle theta, so a bent cell is represented properly;
theta = 180 deg is a straight capsule.  Call this shape E1CE2.

The decision, per frame
-----------------------
    no segment                      -> E1CE2
    segment holds BOTH tips
        and is not over-long        -> the segment, unchanged        (GOOD)
        and is over-long (fused)    -> segment AND E1CE2             (cut)
    segment holds ONE tip
        and is too short            -> segment OR neighbour OR stroke to the
                                       missing tip, bridged to the segment
                                       centroid if that leaves fragments
        and is too long             -> segment AND E1CE2             (cut)
    segment holds NEITHER tip       -> E1CE2

Every branch output is length-checked against the expected span, because the
two failure modes of this family are an unbounded graft and an over-eager cut.

Anchoring (P14)
---------------
Arm LENGTHS and the radius come only from the curated keyframes and from frames
that were accepted as GOOD.  A repaired frame never feeds its length back into
the reference, which is the ratchet that inflated expected geometry in the
previous stage-3 implementation.  Only the ANGLE is fitted per frame.

Iteration
---------
No division in the interval: bisect.  Solve the middle frame; promote it to an
anchor only if it came out GOOD; recurse on both halves.
Division in the interval: run the interval twice, once with the pre-division
(mother) parameters and once with the post-division (daughter) parameters, and
take the division frame that maximises the number of GOOD frames.  A flat scan
is reported as low confidence and defers to the stage-1 keyframe call (P14).

Read-only with respect to canonical masks and keyframes (P5, P14).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import label as cclabel

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ground_truth_corrector.schemas import validate_and_decode_rle, encode_mask_to_rle

# ---------------------------------------------------------------- parameters
FUSE_K = 1.25          # segment longer than this x expected => fused, not good
MISS_K = 0.75          # result shorter than this x expected => reject the cut
TOL_K = 0.35           # "tip is inside the segment" tolerance, as a fraction of r
THETA_WIN = 25.0       # angle search half-window, degrees
THETA_STEP = 5.0
SCAN_ANGLE = False     # angle search during the division scan (see module docs)
PEAK_FRAC_MIN = 0.40   # division scan: min share of interior frames explained
PLATEAU_MAX = 10       # division scan: max width of the near-optimal plateau
BOX_FACTOR = 2.0       # crop half-size, as a multiple of the longer keyframe span
WIDTH_PCTL = 90        # cell width taken as this percentile of perpendicular extent
MIN_AREA = 30          # ignore labels smaller than this


# ---------------------------------------------------------------- geometry
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _seg_dist(pts, A, B):
    AB = B - A
    L2 = float(AB @ AB)
    if L2 < 1e-9:
        return np.linalg.norm(pts - A, axis=1)
    t = np.clip((pts - A) @ AB / L2, 0.0, 1.0)
    return np.linalg.norm(pts - (A + t[:, None] * AB), axis=1)


def capsule(A, B, r, shp):
    """Round-capped stroke of radius r with its axis running A -> B."""
    H, W = shp
    out = np.zeros((H, W), bool)
    lo = np.floor(np.minimum(A, B) - r - 2).astype(int)
    hi = np.ceil(np.maximum(A, B) + r + 2).astype(int)
    x0, y0 = max(lo[0], 0), max(lo[1], 0)
    x1, y1 = min(hi[0], W - 1), min(hi[1], H - 1)
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    pts = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    out[y0:y1 + 1, x0:x1 + 1] = (_seg_dist(pts, A, B) <= r).reshape(yy.shape)
    return out


def stroke_to_tip(tip, C, r, shp):
    """Stroke from `tip` to C, started one radius in so the cap lands on the tip."""
    return capsule(tip + r * _unit(C - tip), C, r, shp)


def bent_capsule(E1, E2, C, r, shp):
    return stroke_to_tip(E1, C, r, shp) | stroke_to_tip(E2, C, r, shp)


def iou(a, b):
    inter = int(np.count_nonzero(a & b))
    un = int(np.count_nonzero(a | b))
    return inter / un if un else 0.0


def _bbox(m, pad, shp):
    ys, xs = np.nonzero(m)
    if not ys.size:
        return None
    return (max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, shp[0]),
            max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, shp[1]))


def iou_win(a, b, win):
    """IoU restricted to a window.  The angle search runs thousands of these."""
    if win is None:
        return iou(a, b)
    y0, y1, x0, x1 = win
    return iou(a[y0:y1, x0:x1], b[y0:y1, x0:x1])


def includes(mask, P, tol):
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return False, float("inf")
    d = float(np.hypot(xs - P[0], ys - P[1]).min())
    return bool(d <= tol), d


def span_along(mask, u):
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return 0.0
    s = np.column_stack([xs, ys]).astype(float) @ u
    return float(s.max() - s.min())


def mask_span(mask):
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return 0.0
    P = np.column_stack([xs, ys]).astype(float)
    X = P - P.mean(0)
    u = np.linalg.svd(X, full_matrices=False)[2][0]
    s = X @ u
    return float(s.max() - s.min())


# ---------------------------------------------------------------- shape
def shape_tips(S):
    C = S["C"]
    return (C + S["L1"] * np.array([np.cos(S["a1"]), np.sin(S["a1"])]),
            C + S["L2"] * np.array([np.cos(S["a2"]), np.sin(S["a2"])]))


def shape_mask(S, shp):
    E1, E2 = shape_tips(S)
    return bent_capsule(E1, E2, S["C"], S["r"], shp)


def shape_theta(S):
    d = np.degrees(S["a1"] - S["a2"]) % 360.0
    return 360.0 - d if d > 180.0 else d


def rotate_theta(S, delta_deg):
    """Open or close the hinge by delta degrees, bisector fixed."""
    T = dict(S)
    h = np.radians(delta_deg) / 2.0
    sgn = 1.0 if ((S["a1"] - S["a2"]) % (2 * np.pi)) < np.pi else -1.0
    T["a1"] = S["a1"] + sgn * h
    T["a2"] = S["a2"] - sgn * h
    return T


def fit_shape(mask, seed=None, step=6.0):
    """Fit (C, both arm angles, both arm lengths, r) to a mask.
    Scored inside a tight window; seeded from the previous frame when given."""
    ys, xs = np.nonzero(mask)
    P0 = np.column_stack([xs, ys]).astype(float)
    C0f = P0.mean(0)
    X = P0 - C0f
    u = np.linalg.svd(X, full_matrices=False)[2][0]
    if u[0] < 0:
        u = -u
    alpha = float(np.arctan2(u[1], u[0]))
    r = float(np.percentile(np.abs(X @ np.array([-u[1], u[0]])), WIDTH_PCTL))
    win = _bbox(mask, int(r) + 8, mask.shape)
    y0, y1, x0, x1 = win
    sub = mask[y0:y1, x0:x1]
    off = np.array([x0, y0], float)
    P = P0 - off
    C0 = C0f - off
    shp = sub.shape

    if seed is not None:
        phi0 = (seed["a1"] + seed["a2"]) / 2.0
        th0 = shape_theta(seed)
        dphis = np.arange(-8, 8.1, step)
        thetas = np.arange(max(100.0, th0 - 24), min(180.0, th0 + 24) + .1, step)
    else:
        phi0 = alpha - np.pi / 2.0
        dphis = np.arange(-30, 30.1, step)
        thetas = np.arange(110.0, 180.1, step)
    offs = np.arange(-0.6 * r, 0.6 * r + .1, max(r / 3.0, 1.0))

    best_sc, best = -1.0, None
    for dphi in dphis:
        phi = phi0 + np.radians(dphi)
        bis = np.array([np.cos(phi), np.sin(phi)])
        for th in thetas:
            a1 = phi + np.radians(th) / 2.0
            a2 = phi - np.radians(th) / 2.0
            d1 = np.array([np.cos(a1), np.sin(a1)])
            d2 = np.array([np.cos(a2), np.sin(a2)])
            for o in offs:
                C = C0 + o * bis
                L1 = max(float(((P - C) @ d1).max()), 1.0)
                L2 = max(float(((P - C) @ d2).max()), 1.0)
                S = dict(C=C, a1=a1, a2=a2, L1=L1, L2=L2, r=r)
                sc = iou(sub, shape_mask(S, shp))
                if sc > best_sc:
                    best_sc, best = sc, dict(C=C + off, a1=a1, a2=a2, L1=L1, L2=L2, r=r)
    best["fit_iou"] = best_sc
    return best


def interp_shape(Sa, Sb, w):
    """Pair the arms by tip proximity so a cell cannot flip end for end, then
    interpolate centre, both arm angles, both arm lengths and the radius."""
    a1t, a2t = shape_tips(Sa)
    b1t, b2t = shape_tips(Sb)
    swap = (np.linalg.norm(a1t - b2t) + np.linalg.norm(a2t - b1t)) < \
           (np.linalg.norm(a1t - b1t) + np.linalg.norm(a2t - b2t))
    b1, b2 = (Sb["a2"], Sb["a1"]) if swap else (Sb["a1"], Sb["a2"])
    M1, M2 = (Sb["L2"], Sb["L1"]) if swap else (Sb["L1"], Sb["L2"])

    def lerp_ang(p, q):
        return p + w * ((q - p + np.pi) % (2 * np.pi) - np.pi)

    return dict(C=(1 - w) * Sa["C"] + w * Sb["C"],
                a1=lerp_ang(Sa["a1"], b1), a2=lerp_ang(Sa["a2"], b2),
                L1=(1 - w) * Sa["L1"] + w * M1,
                L2=(1 - w) * Sa["L2"] + w * M2,
                r=(1 - w) * Sa["r"] + w * Sb["r"])


# ---------------------------------------------------------------- data access
class Cell:
    """One (film, local cell id) over one keyframe interval [K_a, K_b].

    Everything is held in a crop around the interval rather than at full frame
    size, which is what makes a whole-experiment run tractable.
    """

    def __init__(self, exp_dir, film, lc, Ka, Kb, channel="FL"):
        self.exp = Path(exp_dir)
        csv = self.exp / film / f"TrackedCells_{film}" / f"cell_{lc}_masks.csv"
        if not csv.exists():
            raise FileNotFoundError(f"no mask csv: {csv}")
        df = pd.read_csv(csv)
        self.H0 = int(df.iloc[0]["height"])
        self.W0 = int(df.iloc[0]["width"])
        col = ("rle_gfp" if channel == "FL" and "rle_gfp" in df.columns
               and df["rle_gfp"].dropna().any() else "rle_bf")
        self.rle_col = col
        self.rle = {}
        for _, row in df.iterrows():
            v = str(row.get(col, ""))
            self.rle[int(row["time_point"])] = (None if (not v.strip() or v.lower() == "nan")
                                                else v)
        self.film, self.lc, self.Ka, self.Kb, self.channel = film, lc, Ka, Kb, channel

        ma, mb = self._full(Ka), self._full(Kb)
        if ma is None or mb is None:
            raise ValueError("KF_NO_MASK")
        ca = np.column_stack(np.nonzero(ma)[::-1]).mean(0)
        cb = np.column_stack(np.nonzero(mb)[::-1]).mean(0)
        c = (ca + cb) / 2.0
        half = int(BOX_FACTOR * max(mask_span(ma), mask_span(mb))) + 20
        self.x0 = max(0, int(c[0] - half))
        self.y0 = max(0, int(c[1] - half))
        self.x1 = min(self.W0, int(c[0] + half))
        self.y1 = min(self.H0, int(c[1] + half))
        self.shape = (self.y1 - self.y0, self.x1 - self.x0)
        self._lab = {}

    def _full(self, t):
        v = self.rle.get(t)
        if not v:
            return None
        try:
            m = validate_and_decode_rle(v, self.H0, self.W0).astype(bool)
        except Exception:
            return None
        return m if m.sum() >= MIN_AREA else None

    def crop(self, m):
        return None if m is None else m[self.y0:self.y1, self.x0:self.x1]

    def mask(self, t):
        return self.crop(self._full(t))

    def labels(self, t):
        """Every _seg label touching the crop, as crop-sized boolean masks."""
        if t not in self._lab:
            p = (self.exp / self.film / f"Masks_{self.film}" /
                 f"{self.film}_t_{t:03d}_c_0_seg.tif")
            if not p.exists():
                self._lab[t] = []
            else:
                cr = tifffile.imread(str(p))[self.y0:self.y1, self.x0:self.x1]
                out = []
                for L in np.unique(cr):
                    if L == 0:
                        continue
                    cm = (cr == L)
                    if cm.sum() >= MIN_AREA:
                        out.append(cm)
                self._lab[t] = out
        return self._lab[t]

    def release(self):
        self._lab.clear()


# ---------------------------------------------------------------- the decision
def decide(cell, t, S_exp, search_angle=True):
    shp = cell.shape
    exp = shape_mask(S_exp, shp)
    labs = cell.labels(t)

    seg, bov = None, 0
    for full in labs:
        ov = int(np.count_nonzero(full & exp))
        if ov > bov:
            seg, bov = full, ov

    res = dict(t=t, seg=seg, theta=shape_theta(S_exp), shape=dict(S_exp))

    if search_angle and seg is not None:
        win = _bbox(seg | exp, int(S_exp["r"]) + 6, shp)
        best_sc, best_d = iou_win(exp, seg, win), 0.0
        for d in np.arange(-THETA_WIN, THETA_WIN + .1, THETA_STEP):
            if d == 0.0:
                continue
            cand = rotate_theta(S_exp, d)
            th = shape_theta(cand)
            if th < 100.0 or th > 180.0:
                continue
            sc = iou_win(shape_mask(cand, shp), seg, win)
            if sc > best_sc:
                best_sc, best_d = sc, d
        if best_d != 0.0:
            S_exp = rotate_theta(S_exp, best_d)
            exp = shape_mask(S_exp, shp)
            res.update(theta_shift=float(best_d), theta=shape_theta(S_exp),
                       shape=dict(S_exp))

    E1, E2 = shape_tips(S_exp)
    C, r = S_exp["C"], S_exp["r"]
    u = _unit(E2 - E1)
    exp_span = S_exp["L1"] + S_exp["L2"]
    tol = max(3.0, TOL_K * r)
    res.update(exp=exp, exp_span=exp_span, tol=tol, E1=E1, E2=E2, C=C, r=r)

    if seg is None:
        res.update(branch="NO_SEG", out=exp, good=False)
        return res

    in1, d1 = includes(seg, E1, tol)
    in2, d2 = includes(seg, E2, tol)
    s_span = span_along(seg, u)
    res.update(seg_span=s_span, in1=in1, in2=in2, d1=d1, d2=d2)

    def cut():
        out = seg & exp
        if span_along(out, u) < MISS_K * exp_span:
            res["intersect_rejected"] = True
            return exp
        return out

    if in1 and in2:
        # A fused blob contains both tips AND a whole neighbour, so "both tips
        # present" is not on its own evidence that the segment is the cell.
        if s_span > FUSE_K * exp_span:
            res.update(branch="LONG_INTERSECT(fused)", out=cut(), good=False)
        else:
            res.update(branch="GOOD", out=seg, good=True)
        return res

    if not in1 and not in2:
        res.update(branch="BOTH_MISSED", out=exp, good=False)
        return res

    missed = E2 if in1 else E1
    tag = "E2" if in1 else "E1"
    if s_span < exp_span:
        strk = stroke_to_tip(missed, C, r, shp)
        extra = np.zeros(shp, bool)
        for full in labs:
            if np.any(full & seg):
                continue
            hit, _ = includes(full, missed, tol)
            if hit:
                extra |= full
        out = seg | extra | strk
        relinked = bool(extra.any())
        if relinked and span_along(out, u) > FUSE_K * exp_span:
            out = seg | strk
            relinked = False
            res["relink_rejected"] = True
        n = cclabel(out)[1]
        if n > 1:                      # bridge: extrude C -> segment centroid
            ys, xs = np.nonzero(seg)
            out = out | capsule(C, np.array([xs.mean(), ys.mean()]), r, shp)
            res.update(bridged=True, bridge_from=n, bridge_to=cclabel(out)[1])
        res.update(branch=f"SHORT_UNION({tag})", out=out, good=False, relinked=relinked)
    else:
        res.update(branch=f"LONG_INTERSECT({tag})", out=cut(), good=False)
    return res


# ---------------------------------------------------------------- iteration
def bisect_range(cell, lo, hi, S_lo, S_hi):
    anchors = {lo: S_lo, hi: S_hi}
    results, promoted, order = {}, set(), []
    queue = [(lo, hi)]
    while queue:
        a, b = queue.pop(0)
        if b - a < 2:
            continue
        mid = (a + b) // 2
        if mid not in results:
            a_lo = max([t for t in anchors if t < mid] or [lo])
            a_hi = min([t for t in anchors if t > mid] or [hi])
            w = 0.0 if a_hi == a_lo else (mid - a_lo) / float(a_hi - a_lo)
            res = decide(cell, mid, interp_shape(anchors[a_lo], anchors[a_hi], w))
            res["from"] = (a_lo, a_hi)
            results[mid] = res
            order.append(mid)
            if res["good"]:
                anchors[mid] = fit_shape(res["out"], seed=res["shape"])
                promoted.add(mid)
        queue.append((a, mid))
        queue.append((mid, b))
    return results, promoted, order


def division_scan(cell, Sa, Sb):
    Ka, Kb = cell.Ka, cell.Kb
    ts = list(range(Ka + 1, Kb))
    gA = {t: decide(cell, t, dict(Sa), search_angle=SCAN_ANGLE)["good"] for t in ts}
    gB = {t: decide(cell, t, dict(Sb), search_angle=SCAN_ANGLE)["good"] for t in ts}
    score = {d: sum(gA[t] for t in ts if t < d) + sum(gB[t] for t in ts if t >= d)
             for d in range(Ka + 1, Kb + 1)}
    best = max(score, key=lambda d: (score[d], -abs(d - (Ka + Kb) // 2)))
    plateau = [d for d in score if score[d] >= score[best] - 1]
    frac = score[best] / float(max(len(ts), 1))
    width = max(plateau) - min(plateau) + 1
    return dict(ts=ts, goodA=gA, goodB=gB, score=score, d=best, n_good=score[best],
                peak_frac=frac, plateau=width,
                confident=bool(frac >= PEAK_FRAC_MIN and width <= PLATEAU_MAX))


def solve(cell, dividing=False):
    """-> (results by frame, promoted anchor frames, division frame or None, scan)"""
    Ka, Kb = cell.Ka, cell.Kb
    Sa = fit_shape(cell.mask(Ka))
    Sb = fit_shape(cell.mask(Kb))
    if not dividing:
        res, prom, _ = bisect_range(cell, Ka, Kb, Sa, Sb)
        return res, prom, None, None
    scan = division_scan(cell, Sa, Sb)
    d = scan["d"]
    res, prom = {}, set()
    if d - 1 > Ka:
        r1, p1, _ = bisect_range(cell, Ka, d, Sa, Sa)
        res.update(r1)
        prom |= p1
    if Kb > d:
        r2, p2, _ = bisect_range(cell, d - 1, Kb, Sb, Sb)
        for t, v in r2.items():
            if t >= d or t not in res:
                res[t] = v
        prom |= p2
    for t in range(Ka + 1, Kb):
        if t not in res:
            res[t] = decide(cell, t, dict(Sa if t < d else Sb))
    return res, prom, d, scan


# ---------------------------------------------------------------- emit
def to_full(cell, crop_mask):
    if crop_mask is None or not crop_mask.any():
        return None
    m = np.zeros((cell.H0, cell.W0), bool)
    m[cell.y0:cell.y1, cell.x0:cell.x1] = crop_mask
    return m


def interval_rows(cell, res, d):
    """One row per interior frame, with the mask re-encoded at full frame size."""
    rows = []
    for t in sorted(res):
        r = res[t]
        full = to_full(cell, r["out"])
        rows.append(dict(
            film=cell.film, local_cid=cell.lc, channel=cell.channel,
            K_a=cell.Ka, K_b=cell.Kb, time_point=t,
            branch=r["branch"], good=bool(r["good"]),
            theta=round(float(r["theta"]), 1),
            theta_shift=round(float(r.get("theta_shift", 0.0)), 1),
            exp_span=round(float(r["exp_span"]), 2),
            out_span=round(mask_span(r["out"]), 2) if r["out"].any() else 0.0,
            seg_span=round(float(r.get("seg_span", 0.0)), 2),
            relinked=bool(r.get("relinked", False)),
            relink_rejected=bool(r.get("relink_rejected", False)),
            intersect_rejected=bool(r.get("intersect_rejected", False)),
            bridged=bool(r.get("bridged", False)),
            t_div=d, height=cell.H0, width=cell.W0,
            rle=encode_mask_to_rle(full.astype(np.uint8)) if full is not None else "",
        ))
    return rows
