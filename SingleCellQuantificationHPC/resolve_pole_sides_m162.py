#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make pole identity mean the same physical end across a global cell's films.

The problem
-----------
`PCA_utils.load_experiment_features` defines `pol1` as the *dominant* pole, so
a cell whose brighter end moves to the other pole still plots as a high `pol1`:
the label follows the brightness and the switch is invisible. The explorer must
therefore plot the GEOMETRIC assignment — and that assignment has to mean the
same physical end in every film of a cell.

Within a film, quantification already maintains it (`ep_refs` carries ep1/ep2
frame to frame). Across films it is arbitrary: each film's track is seeded
independently, so film N's `pol1` may be film N-1's `pol2`.

Why this is a reconstruction, not a lookup
------------------------------------------
ep1/ep2 are computed at stage 4 and then **discarded** — they never reach the
quant table. Persisting them per frame is the root fix (see P15 stage 6 rule 4).
Until that happens the correspondence has to be re-derived from the masks, which
is what this does, and the output is labelled `reconstructed` accordingly.

Method
------
1. Per frame, take the mask's major-axis endpoints from `regionprops`
   (centroid ± ½·major_axis_length along the orientation).
2. Measure the image inside a small disc at each endpoint, and ask which
   endpoint is brighter. Compare that against whether `pol1_int > pol2_int`.
   That gives, for this frame, whether quant's `pol1` sits at endpoint A or B.
   Frames vote with weight |pol1_int − pol2_int|, so ambiguous frames (the two
   poles nearly equal) count for little and clear ones decide. Within a film the
   true answer is constant, so ~101 weighted votes settle it comfortably.
3. Chain films in acquisition order: match film N's first-frame endpoints to
   film N−1's last-frame endpoints by position. Cells do not teleport across a
   ~20 min brightfield block, so proximity is a safe matcher — and the residual
   distance is reported so a bad match is visible rather than silent.
4. Emit, per datapoint, `swap_for_display`: whether the explorer should swap
   p1/p2 so that "pole 1" is the same physical end throughout the cell.

Output: <outputs>/2026_09_09_M162/pole_sides.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from tifffile import imread

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Cell_tracking_functions import rle_decode

EXP_NAME = "2026_09_09_M162"
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
_SSD = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_DENSE = _SSD / "dense_masks"
DEFAULT_QUANT = _SSD / "quant"
DEFAULT_OUT = _SSD / "pole_sides.csv"
SEQS = [f"NeonG_YES_F{i}" for i in range(4)]

DISC_R = 4          # radius of the intensity probe at each endpoint, px
MIN_VOTE = 1e-9
VOTE_CONFIDENT = 50.0   # weighted-vote margin below this is a near tie
RESID_SUSPECT = 30.0    # cross-film endpoint match worse than this is suspect


def build_id_map(exp_dir: Path):
    """(film, local_cell_id) -> global_cell_id, plus film -> FL ordinal (P12)."""
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    gid_of, order = {}, {}
    for seq in SEQS:
        if seq not in linkage:
            continue
        films = linkage[seq]["films"]
        for k, film in enumerate([f for f in films if "FL" in f]):
            order[film] = k
        fl_films = [f for f in films if "FL" in f]
        idx = [i for i, f in enumerate(films) if "FL" in f]
        for gid, locals_ in linkage[seq]["global_cells"].items():
            for i, film in zip(idx, fl_films):
                lc = locals_[i]
                if lc and lc > 0:
                    gid_of[(film, int(lc))] = gid
    return gid_of, order


def endpoints(mask: np.ndarray):
    """The two major-axis endpoints of the largest component, (row, col) each."""
    props = regionprops(label(mask.astype(np.uint8)))
    if not props:
        return None
    p = max(props, key=lambda x: x.area)
    r0, c0 = p.centroid
    half = 0.5 * float(p.major_axis_length)
    # skimage orientation is measured from the ROW axis (see build_strip_tile)
    th = p.orientation
    dr, dc = -half * np.cos(th), half * np.sin(th)
    return np.array([[r0 - dr, c0 - dc], [r0 + dr, c0 + dc]], float)


def disc_mean(img: np.ndarray, mask: np.ndarray, rc, radius=DISC_R) -> float:
    r, c = int(round(rc[0])), int(round(rc[1]))
    H, W = img.shape
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    if r1 <= r0 or c1 <= c0:
        return float("nan")
    sub_i = img[r0:r1, c0:c1].astype(np.float32)
    sub_m = mask[r0:r1, c0:c1]
    return float(sub_i[sub_m].mean()) if sub_m.any() else float(sub_i.mean())


def resolve_film(dense_csv: Path, quant_csv: Path, frames_dir: Path, film: str):
    """Which endpoint is quant's pol1, and where the endpoints sit at each end.

    Returns (pol1_at_index, first_eps, last_eps, margin) or None.
    `pol1_at_index` is 0 or 1 into the endpoint pair.
    """
    dm = pd.read_csv(dense_csv)
    if dm.empty or "rle" not in dm.columns:
        return None
    q = pd.read_csv(quant_csv, usecols=lambda c: c in
                    ("time_point", "cell_area", "pol1_int", "pol2_int"))
    if q.empty:
        return None
    q = q.loc[q.groupby("time_point")["cell_area"].idxmax()].set_index("time_point")

    H, W = int(dm.iloc[0]["height"]), int(dm.iloc[0]["width"])
    vote = 0.0
    first_eps = last_eps = None
    for _, r in dm.sort_values("time_point").iterrows():
        t = int(r["time_point"])
        rle = str(r.get("rle", ""))
        if not rle.strip() or rle.lower() == "nan" or t not in q.index:
            continue
        try:
            mask = np.asarray(rle_decode(rle, (H, W)), bool)
        except Exception:
            continue
        if mask.sum() < 30:
            continue
        eps = endpoints(mask)
        if eps is None:
            continue
        fp = frames_dir / f"{film}_t_{t:03d}_c_0.tif"
        if not fp.exists():
            continue
        img = imread(str(fp))
        iA = disc_mean(img, mask, eps[0])
        iB = disc_mean(img, mask, eps[1])
        p1, p2 = float(q.at[t, "pol1_int"]), float(q.at[t, "pol2_int"])
        if not all(np.isfinite([iA, iB, p1, p2])):
            continue
        # Weight by how clearly the two poles differ: an ambiguous frame should
        # not get the same say as an unambiguous one.
        w = abs(p1 - p2)
        if w > MIN_VOTE:
            # quant says pol1 is the brighter pole iff p1 > p2;
            # the image says endpoint A is brighter iff iA > iB.
            agree = ((p1 > p2) == (iA > iB))
            vote += w if agree else -w
        if first_eps is None:
            first_eps = eps
        last_eps = eps

    if first_eps is None:
        return None
    pol1_at = 0 if vote >= 0 else 1
    return pol1_at, first_eps, last_eps, abs(vote)


def match_endpoints(prev_eps, cur_eps):
    """Is cur endpoint 0 the same physical end as prev endpoint 0?

    Returns (same_order, residual_px). Endpoints do not teleport across a
    brightfield block, so nearest-neighbour is safe; the residual is returned
    so a bad match can be seen rather than assumed away.
    """
    d_same = (np.linalg.norm(cur_eps[0] - prev_eps[0])
              + np.linalg.norm(cur_eps[1] - prev_eps[1]))
    d_swap = (np.linalg.norm(cur_eps[0] - prev_eps[1])
              + np.linalg.norm(cur_eps[1] - prev_eps[0]))
    return (d_same <= d_swap), float(min(d_same, d_swap) / 2.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    ap.add_argument("--quant", type=Path, default=DEFAULT_QUANT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    gid_of, order = build_id_map(a.exp)
    by_gid = {}
    for d in sorted(a.dense.glob("*/cell_*.csv")):
        film = d.parent.name
        if "FL" not in film or film not in order:
            continue
        try:
            lc = int(d.stem.split("_")[1])
        except Exception:
            continue
        gid = gid_of.get((film, lc))
        if gid is None:
            continue
        by_gid.setdefault(gid, []).append((order[film], film, lc, d))
    for g in by_gid:
        by_gid[g].sort()

    gids = sorted(by_gid)
    if a.limit:
        gids = gids[:a.limit]
    print(f"global cells: {len(gids)}", flush=True)

    rows = []
    for n, gid in enumerate(gids, 1):
        prev_eps = None
        prev_canon_at = None      # which endpoint index is canonical pole 1
        for _, film, lc, dcsv in by_gid[gid]:
            qcsv = a.quant / film / f"cell_{lc}.csv"
            if not qcsv.exists():
                continue
            res = resolve_film(dcsv, qcsv, a.exp / film / f"Frames_{film}", film)
            if res is None:
                continue
            pol1_at, first_eps, last_eps, margin = res

            if prev_eps is None:
                canon_at = pol1_at          # first film defines the convention
                resid = 0.0
            else:
                same, resid = match_endpoints(prev_eps, first_eps)
                canon_at = prev_canon_at if same else 1 - prev_canon_at

            rows.append(dict(
                cell_id=f"{gid}__{film}", global_cell_id=gid, film=film,
                local_cid=lc,
                quant_pol1_at=pol1_at, canonical_pol1_at=canon_at,
                swap_for_display=bool(pol1_at != canon_at),
                vote_margin=round(margin, 2),
                match_residual_px=round(resid, 2),
                # A near-tied vote means the two poles were of similar
                # brightness throughout, so which endpoint quant called pol1
                # is not reliably recoverable for this datapoint. Flagged
                # rather than silently trusted.
                confident=bool(margin >= VOTE_CONFIDENT
                               and resid <= RESID_SUSPECT),
                source="reconstructed"))
            prev_eps, prev_canon_at = last_eps, canon_at
        if n % 100 == 0:
            print(f"  {n}/{len(gids)} cells", flush=True)

    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\ndatapoints resolved: {len(df)}")
    if len(df):
        print(f"  needing a display swap: {int(df.swap_for_display.sum())} "
              f"({100*df.swap_for_display.mean():.1f}%)")
        print(f"  median vote margin   : {df.vote_margin.median():.1f}")
        big = df[df.match_residual_px > 30]
        print(f"  cross-film matches over 30 px (suspect): {len(big)}")
        print(f"  low-confidence datapoints: {int((~df.confident).sum())} "
              f"({100*(~df.confident).mean():.1f}%) — near-tied vote or a "
              f"suspect cross-film match")
    print(f"table -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
