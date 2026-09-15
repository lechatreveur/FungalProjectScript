#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vertical cell strips for M160 model-based dense masks.

Strips are normally a by-product of stage 4: `one_cell_quantification_1CH.py`
passes a `strip_tiles` list into `quantify_one_object`, which appends one
`build_strip_tile()` output per frame, and `--make_strips` writes the PNG.
`quantify_model_based_dense.py` did not pass that list, so M160 has none.

This rebuilds them without requantifying, exactly as `build_strips_only.py`
does for tracked cells: `build_strip_tile` derives its own rotation from the
mask via `regionprops.orientation` and never touches the EM or pole fitting, so
a strip is fully determined by (frame image, mask). The only difference here is
the mask source — the stage-3 dense CSVs rather than `cell_<id>_masks.csv`.

One strip per **datapoint**, which is one cell in one film over its 101 frames,
matching `build_features_m160.datapoint_id`. Not one per global cell: a global
cell spans several consecutive films and is several datapoints, so a
concatenated strip would not line up with the trajectory shown beside it.

Contrast follows the production path: the crop is rescaled to the FILM's
intensity range (1st to 99.5th percentile, as `FindMovieMaxMin` computes it)
before tiling, exactly as `ImageQuantification` does before handing the crop to
`build_strip_tile`. Passing raw counts leaves the background offset in and
flattens the signal.

Tile size defaults to 16x56, half the house 32x112 in each dimension and so a
quarter of the pixels. At full size a strip is ~270 KB and the cohort is 6,246
of them: 1.7 GB on disk and well over 2 GB once base64-embedded in one page,
which neither this workstation nor a browser will hold. Halving keeps the cell
legible along its length while making the set tractable. Pass --tile-h 32
--tile-w 112 to restore the house size for a small cohort.

Output: <outputs>/2026_08_28_M160/strips/<global_cell_id>__<film>.png  (P4)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import label, regionprops

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Cell_tracking_functions import rle_decode
from quant_helpers import merge_disconnected_components, build_strip_tile, save_strip_from_tiles

EXP_NAME = "2026_08_28_M160"
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_DENSE = _SSD_OUT / "dense_masks"
DEFAULT_OUT = _SSD_OUT / "strips"
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]


def build_id_map(exp_dir):
    """(film, local_cell_id) -> global_cell_id, and film -> FL ordinal (P12)."""
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    gid_of, order = {}, {}
    for seq in SEQS:
        if seq not in linkage:
            continue
        films = linkage[seq]["films"]
        for k, film in enumerate([f for f in films if "FL" in f]):
            order[film] = k
        for gid, locals_ in linkage[seq]["global_cells"].items():
            for film, lc in zip(films, locals_):
                if lc and lc > 0:
                    gid_of[(film, int(lc))] = gid
    return gid_of, order


class Frames:
    """Frame cache plus the film intensity scale, dropped when the film changes."""

    def __init__(self, exp, cap=140):
        self.exp, self.cap, self.film, self.d = Path(exp), cap, None, {}
        self.scale = {}

    def get(self, film, t):
        if film != self.film:
            self.film, self.d = film, {}
        if t not in self.d:
            if len(self.d) >= self.cap:
                self.d.pop(next(iter(self.d)))
            p = self.exp / film / f"Frames_{film}" / f"{film}_t_{t:03d}_c_0.tif"
            self.d[t] = imread(str(p)) if p.exists() else None
        return self.d[t]

    def limits(self, film):
        """The film's (max, min) as `FindMovieMaxMin` computes them: pool every
        10th pixel of every frame, then the 99.5th and 1st percentiles."""
        if film not in self.scale:
            fs = sorted((self.exp / film / f"Frames_{film}").glob(f"{film}_t_*_c_0.tif"))
            fs = [p for p in fs if not p.name.endswith("_seg.tif")]
            if not fs:
                self.scale[film] = (None, None)
            else:
                px = np.concatenate([imread(str(p)).ravel()[::10] for p in fs])
                self.scale[film] = (float(np.percentile(px, 99.5)),
                                    float(np.percentile(px, 1)))
        return self.scale[film]


def tiles_for_film(df, frames, film, tile_h=16, tile_w=56):
    """One tile per frame, reproducing what --make_strips feeds build_strip_tile.

    The contrast step matters and is easy to miss: `ImageQuantification` rescales
    the crop to the FILM's intensity range before handing it over —

        cropped_img = clip((cropped_img - C1min) / (C1max - C1min) * 255, 0, 255)

    where C1min/C1max are the film's 1st and 99.5th percentiles. Passing raw
    camera counts instead leaves the background offset in, and the per-strip
    99th-percentile mapping in `save_strip_from_tiles` then has to absorb it,
    compressing the real signal into a narrow band."""
    out = []
    H, W = int(df.iloc[0]["height"]), int(df.iloc[0]["width"])
    c1max, c1min = frames.limits(film)
    if c1max is None or c1max <= c1min:
        return out
    span = float(c1max - c1min)
    for _, r in df.sort_values("time_point").iterrows():
        rle = str(r.get("rle", ""))
        if not rle.strip() or rle.lower() == "nan":
            continue
        img = frames.get(film, int(r["time_point"]))
        if img is None:
            continue
        try:
            mask = np.asarray(rle_decode(rle, (H, W)), bool)
        except Exception:
            continue
        if mask.sum() < 30:
            continue
        mask = merge_disconnected_components(mask, keep=2, bridge_width=1)
        props = regionprops(label(mask))
        if not props:
            continue
        r0, c0, r1, c1 = props[0].bbox
        crop = np.clip((img[r0:r1, c0:c1].astype(np.float32) - c1min) / span * 255.0,
                       0, 255).astype(np.uint8)
        out.append(build_strip_tile(crop, mask[r0:r1, c0:c1],
                                    frame_h=tile_h, frame_w=tile_w))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--films", nargs="+", default=None,
                    help="restrict to these films (one HPC array task per film)")
    ap.add_argument("--tile-h", type=int, default=16,
                    help="tile height in px; the house default is 32")
    ap.add_argument("--tile-w", type=int, default=56,
                    help="tile width in px; the house default is 112")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    gid_of, order = build_id_map(a.exp)

    # group the dense CSVs by global cell, films in sequence order
    by_gid = {}
    for f in sorted(a.dense.glob("*/cell_*.csv")):
        film = f.parent.name
        if "FL" not in film or film not in order:
            continue
        if a.films and film not in set(a.films):
            continue
        try:
            lc = int(f.stem.split("_")[1])
        except Exception:
            continue
        gid = gid_of.get((film, lc))
        if gid is None:
            continue
        by_gid.setdefault(gid, []).append((order[film], film, f))
    for gid in by_gid:
        by_gid[gid].sort()

    # One strip per DATAPOINT, which is one cell in one film — not one per
    # global cell. A datapoint is a 101-frame single-film trace (see
    # build_features_m160.datapoint_id), so its strip must cover the same
    # frames, or the strip and the trajectory beside it describe different
    # things.
    items = []
    for gid, entries in by_gid.items():
        for _, film, f in entries:
            items.append((f"{gid}__{film}", film, f))
    items.sort(key=lambda it: (it[1], it[0]))   # film-major, so the cache pays off
    if a.limit:
        items = items[:a.limit]
    print(f"datapoints (cell x film): {len(items)}", flush=True)

    frames = Frames(a.exp)
    done = skipped = failed = 0
    t0 = time.time()
    for i, (dp, film, f) in enumerate(items, 1):
        dst = a.out / f"{dp}.png"
        if dst.exists() and not a.force:
            skipped += 1
            continue
        try:
            tiles = tiles_for_film(pd.read_csv(f), frames, film,
                                   tile_h=a.tile_h, tile_w=a.tile_w)
        except Exception as exc:
            print(f"  {dp}: {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            continue
        if not tiles or not save_strip_from_tiles(tiles, str(dst)):
            failed += 1
            continue
        done += 1
        if i % 25 == 0 or i == len(items):
            el = time.time() - t0
            rate = el / max(done, 1)
            print(f"[{i}/{len(items)}] done={done} skip={skipped} err={failed} "
                  f"{el/60:.1f}min, {rate:.1f}s/strip, "
                  f"eta {rate*(len(items)-i)/60:.0f}min", flush=True)

    print(f"FINISHED done={done} skipped={skipped} errors={failed} "
          f"{(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
