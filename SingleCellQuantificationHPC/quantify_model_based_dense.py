#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 — polarity-site quantification on model-based dense masks (P14).

Consumes the stage-3 output of `run_model_based_dense_tracking.py` instead of
re-deriving masks by overlap tracking, so the quantified series is exactly the
mask series stage 3 emitted and every row carries its stage-3 provenance.

Per frame it runs the EM mixture fit in `Image_quantification_functions.
ImageQuantification` and records the polarity-site dynamics: intensity at each
pole, septum intensity, nucleus position and intensity, cytoplasmic background,
and cell length and area.

The endpoint reference (`ep1`, `ep2`) is established on a cell's first emitted
frame and carried forward, which is what keeps pole 1 and pole 2 from swapping
identity down the series.

Frames whose stage-3 branch was `NO_SEG` or `BOTH_MISSED` carry no image
evidence — they are the shape model alone — and are marked `model_only` so they
can be excluded downstream (P13).

Output is scratch-only; canonical `cell_*_data.csv` files are never written
(P5).

    python SingleCellQuantificationHPC/quantify_model_based_dense.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from skimage.io import imread

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Image_quantification_functions import ImageQuantification
from ground_truth_corrector.schemas import validate_and_decode_rle

_SCRATCH = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/"
                "SingleCellQuantificationHPC/scratch")
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies/2026_08_28_M160")
DEFAULT_IN = _SCRATCH / "model_based_dense_out"
DEFAULT_OUT = _SCRATCH / "model_based_quant_out"
MODEL_ONLY = {"NO_SEG", "BOTH_MISSED"}


class FrameCache:
    """Frames are shared by every cell in a film, so cache them and drop the
    cache when the film changes."""

    def __init__(self, exp, cap=140):
        self.exp, self.cap, self.film, self.d = Path(exp), cap, None, {}

    def get(self, film, t):
        if film != self.film:
            self.film, self.d = film, {}
        if t not in self.d:
            if len(self.d) >= self.cap:
                self.d.pop(next(iter(self.d)))
            p = self.exp / film / f"Frames_{film}" / f"{film}_t_{t:03d}_c_0.tif"
            self.d[t] = imread(str(p)) if p.exists() else None
        return self.d[t]

    def scale(self, film):
        """Film intensity scale, computed exactly as `FindMovieMaxMin` in
        `quantify_cell.py` does it: pool every 10th pixel of EVERY frame in the
        film, then take the 99.5th and 1st percentiles.  Pooling matters — using
        the first frame alone ignores photobleaching and shifts the scale."""
        frames = sorted((self.exp / film / f"Frames_{film}").glob(f"{film}_t_*_c_0.tif"))
        if not frames:
            return None, None
        px = []
        for p in frames:
            img = imread(str(p))
            px.append(img.ravel()[::10])
        px = np.concatenate(px)
        return float(np.percentile(px, 99.5)), float(np.percentile(px, 1))


def touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any())


def quantify_cell(df, cache, gmax, gmin):
    df = df.sort_values("time_point")
    H, W = int(df.iloc[0]["height"]), int(df.iloc[0]["width"])
    lc = int(df.iloc[0]["local_cid"])
    film = df.iloc[0]["film"]
    ep1 = ep2 = None
    first = True
    rows = []
    for _, r in df.iterrows():
        t = int(r["time_point"])
        rle = str(r.get("rle", ""))
        if not rle.strip() or rle.lower() == "nan":
            continue
        img = cache.get(film, t)
        if img is None:
            continue
        try:
            mask = validate_and_decode_rle(rle, H, W).astype(bool)
        except Exception:
            continue
        if mask.sum() < 30:
            continue
        base = dict(
            film=film, local_cid=lc, time_point=t,
            branch=r["branch"], good=bool(r["good"]),
            model_only=bool(str(r["branch"]).split("(")[0] in MODEL_ONLY),
            theta=r.get("theta"), t_div=r.get("t_div"),
            stage3_span=r.get("out_span"), expected_span=r.get("exp_span"),
            touches_border=touches_border(mask),
        )
        try:
            if first:
                par, pf, _, ep1, ep2 = ImageQuantification(
                    img, mask, lc, gmax, gmin, 0, skip_em=False)
                first = False
            else:
                par, pf, _, _, _ = ImageQuantification(
                    img, mask, lc, gmax, gmin, t,
                    ref_ep1=ep1, ref_ep2=ep2, skip_em=False)
        except Exception as exc:
            base["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(base)
            continue
        s1, s2 = par.get("mu_S1_Y2"), par.get("mu_S2_Y2")

        def _nu_dis():
            """mu_mn_Y2 is the nucleus mean position; element 1 is the distance
            along the cell axis.  It comes back as a numpy array, so it cannot be
            truth-tested."""
            v = par.get("mu_mn_Y2")
            if v is None:
                return None
            v = np.asarray(v).ravel()
            return float(v[1]) if v.size > 1 else None

        base.update(
            error="",
            cell_length=pf.get("major_axis_length"),
            cell_area=pf.get("area"),
            nu_dis=_nu_dis(),
            nu_int=par.get("mu_I_Y2"),
            cyt_int=par.get("mu_bg_Y2"),
            septum_int=None if (not s1 or not s2) else (s1 + s2) / 2.0,
            pol1_int=par.get("mu_P1_Y2"),
            pol2_int=par.get("mu_P2_Y2"),
        )
        rows.append(base)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--dense", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--channel", default="FL", choices=["FL", "BF", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.dense.glob("*/cell_*.csv"))
    if a.channel != "both":
        files = [f for f in files if (a.channel in f.parent.name)]
    if a.limit:
        files = files[:a.limit]
    # group by film so the frame cache is useful
    files.sort(key=lambda f: (f.parent.name, f.name))
    print(f"cells: {len(files)}  channel {a.channel}", flush=True)
    print(f"out:   {a.out}", flush=True)

    cache = FrameCache(a.exp)
    scales = {}
    done = skipped = failed = 0
    t_start = time.time()
    for i, f in enumerate(files, 1):
        film = f.parent.name
        dst = a.out / film / f.name
        if dst.exists() and not a.force:
            skipped += 1
            continue
        try:
            df = pd.read_csv(f)
            if film not in scales:
                scales[film] = cache.scale(film)
            gmax, gmin = scales[film]
            if gmax is None:
                failed += 1
                continue
            rows = quantify_cell(df, cache, gmax, gmin)
            if rows:
                dst.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(dst, index=False)
                done += 1
            else:
                failed += 1
        except Exception:
            traceback.print_exc()
            failed += 1
        if i % 25 == 0 or i == len(files):
            el = time.time() - t_start
            rate = el / max(done, 1)
            print(f"[{i}/{len(files)}] done={done} skip={skipped} err={failed} "
                  f"{el/60:.1f}min, {rate:.1f}s/cell, "
                  f"eta {rate*(len(files)-i)/60:.0f}min", flush=True)

    print(f"FINISHED done={done} skipped={skipped} errors={failed} "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
