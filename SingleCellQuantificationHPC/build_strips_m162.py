#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vertical cell strips for M162 model-based dense masks.

Copy-to-modify variant of build_strips_m160.py (P15).

**Contrast mode is the important difference.** The M160 script rescales each
crop to its OWN FILM's intensity range, which is what the production path does
and is correct when strips are only compared within a film. M162 spans FL1-FL4,
across which the polarity signal collapses — pol1_mid 20.9 -> 3.7, pole/cyt
excess 11.8% -> 1.7% — so per-film rescaling would renormalise that away and
make an FL4 strip look as bright as an FL1 strip. Two datapoints of the SAME
cell would then be incomparable, which defeats the point of linking strips into
a multi-film explorer.

`--scale global` (the default here) therefore pools every listed film once and
uses a single (C1max, C1min) for every strip. FL4 strips come out visibly
dimmer, which is the truth. `--scale film` restores the M160 behaviour for
within-film work.


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

Output: <outputs>/2026_09_09_M162/strips/<global_cell_id>__<film>.png  (P4)
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

EXP_NAME = "2026_09_09_M162"
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_DENSE = _SSD_OUT / "dense_masks"
DEFAULT_OUT = _SSD_OUT / "strips"
# Listed explicitly, never matched by substring: M161's NeonG_YES_1_ prefix
# is one underscore-delimited token longer than M162's NeonG_YES_, so a
# substring test silently cross-matches the two experiments.
SEQS = [f"NeonG_YES_F{i}" for i in range(4)]


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
        self._global = None

    def get(self, film, t):
        if film != self.film:
            self.film, self.d = film, {}
        if t not in self.d:
            if len(self.d) >= self.cap:
                self.d.pop(next(iter(self.d)))
            p = self.exp / film / f"Frames_{film}" / f"{film}_t_{t:03d}_c_0.tif"
            self.d[t] = imread(str(p)) if p.exists() else None
        return self.d[t]

    def set_global_limits(self, films):
        """One (max, min) pooled across every listed film.

        Needed because M162's signal collapses across FL1-FL4; a per-film scale
        would renormalise that away and make strips from different films look
        equally bright. Pools every 10th pixel of every frame of every film,
        then takes the same 99.5th/1st percentiles FindMovieMaxMin uses.
        """
        # Memory-bounded on purpose. Pooling every 10th pixel of every frame
        # across 16 films is ~650M values (~5 GB) and gets the process killed.
        # A percentile needs a representative sample, not every pixel: every
        # 5th frame at stride 97 gives ~7M values, which pins the 1st and
        # 99.5th percentiles far tighter than the contrast step can resolve.
        FRAME_STEP, PIX_STRIDE = 5, 97
        px = []
        for film in films:
            fs = sorted((self.exp / film / f"Frames_{film}").glob(f"{film}_t_*_c_0.tif"))
            fs = [q for q in fs if not q.name.endswith("_seg.tif")]
            for q in fs[::FRAME_STEP]:
                px.append(imread(str(q)).ravel()[::PIX_STRIDE].astype(np.float32))
        if not px:
            raise SystemExit("no frames found for the global contrast scale")
        px = np.concatenate(px)
        print(f"  pooled {px.size:,} sampled pixels from {len(films)} film(s)",
              flush=True)
        self._global = (float(np.percentile(px, 99.5)), float(np.percentile(px, 1)))
        print(f"global contrast scale over {len(films)} film(s): "
              f"C1max={self._global[0]:.1f} C1min={self._global[1]:.1f}", flush=True)
        return self._global

    def limits(self, film):
        """The film's (max, min) as `FindMovieMaxMin` computes them: pool every
        10th pixel of every frame, then the 99.5th and 1st percentiles.

        Returns the pooled cross-film scale instead when one has been set, so
        every strip shares one mapping and datapoints stay comparable."""
        if getattr(self, "_global", None) is not None:
            return self._global
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


def _orient_consistently(tiles):
    """Stop the cell flipping end-for-end partway down a strip.

    `regionprops.orientation` is defined modulo 180 degrees, so the angle that
    straightens a cell to horizontal is ambiguous by a half turn. A cell whose
    measured angle wanders across that boundary renders reversed from one frame
    to the next, and the strip appears to swap its endpoints.

    Fixed by continuity: each tile is compared against a running reference both
    as-is and mirrored, and the better match is kept. The comparison uses the
    MASK's width profile along the long axis — where the cell has body, not how
    bright it is.

    Using intensity here would be wrong, and the distinction matters. An
    oscillating bipolar cell legitimately changes which end is brighter; an
    intensity-anchored rule would mirror the image to chase that, and the
    oscillation — the very thing the strip is meant to show — would vanish.
    Shape does not oscillate, so it can anchor the frame while the signal moves
    within it.

    The reference is a running blend rather than frame 0 alone, so a single
    poorly segmented frame cannot flip the remainder of the strip.
    """
    if len(tiles) < 2:
        return tiles, 0

    def profile(t):
        # column sums of the cell's body; tiles are already rotated horizontal
        pr = (np.asarray(t, np.float32) > 0).sum(axis=0).astype(np.float32)
        if pr.sum() <= 0:
            return None
        pr = pr - pr.mean()
        n = float(np.linalg.norm(pr))
        return pr / n if n > 0 else None

    out = [tiles[0]]
    ref = profile(tiles[0])
    n_flipped = 0
    for t in tiles[1:]:
        pr = profile(t)
        if ref is None or pr is None:
            out.append(t)
            if pr is not None:
                ref = pr
            continue
        same = float(np.dot(ref, pr))
        flipped = float(np.dot(ref, pr[::-1]))
        if flipped > same:
            t = np.fliplr(np.asarray(t))
            pr = pr[::-1]
            n_flipped += 1
        out.append(t)
        # running blend: one bad frame cannot redefine the orientation
        ref = 0.7 * ref + 0.3 * pr
        nrm = float(np.linalg.norm(ref))
        if nrm > 0:
            ref = ref / nrm
    return out, n_flipped


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
    out, n_flip = _orient_consistently(out)
    if n_flip:
        print(f"    {film}: re-oriented {n_flip} of {len(out)} tiles", flush=True)
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
    ap.add_argument("--scale", choices=("global", "film"), default="global",
                    help="'global' (default) pools every film into one "
                         "(C1max, C1min) so strips are comparable BETWEEN "
                         "datapoints — necessary because M162's signal "
                         "collapses across FL1-FL4 and a per-film scale would "
                         "renormalise that away. 'film' restores the M160 "
                         "per-film behaviour, correct only within one film.")
    ap.add_argument("--c1max", type=float, default=None,
                    help="explicit contrast maximum, overriding --scale. REQUIRED "
                         "when strips are built film-by-film (e.g. one SLURM array "
                         "task per film): each task would otherwise pool only its "
                         "own film and silently produce per-film scaling, which is "
                         "exactly what --scale global exists to avoid. Compute the "
                         "pooled values once with --print-scale, then pass them to "
                         "every task.")
    ap.add_argument("--c1min", type=float, default=None,
                    help="explicit contrast minimum; see --c1max")
    ap.add_argument("--pole-sides", type=Path, default=None,
                    help="pole_sides.csv from resolve_pole_sides_m162.py. Where "
                         "a datapoint is marked swap_for_display, its whole "
                         "strip is mirrored so that 'pole 1' is the same "
                         "physical end in every film of that cell. The strip "
                         "and the trajectory MUST be flipped by the same "
                         "decision (P15 stage 6 rule 4): a strip showing the "
                         "bright end at the top while the red trace is pol2 is "
                         "worse than either error alone.")
    ap.add_argument("--print-scale", action="store_true",
                    help="pool the listed films, print (C1max, C1min), and exit")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    gid_of, order = build_id_map(a.exp)

    # Decided before any tile is rendered, because every strip must share one
    # mapping for them to be comparable at all.
    _scale_films = ([f for f in order if "FL" in f]
                    if not a.films else [f for f in a.films if "FL" in f])

    if a.print_scale:
        fr = Frames(a.exp)
        c1max, c1min = fr.set_global_limits([f for f in order if "FL" in f])
        print(f"POOLED_SCALE c1max={c1max:.4f} c1min={c1min:.4f}")
        return 0

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

    pole_swap = {}
    if a.pole_sides and Path(a.pole_sides).exists():
        _ps = pd.read_csv(a.pole_sides)
        pole_swap = dict(zip(_ps.cell_id.astype(str),
                             _ps.swap_for_display.astype(bool)))
        print(f"pole sides: {len(pole_swap)} datapoints, "
              f"{int(sum(pole_swap.values()))} to mirror", flush=True)

    frames = Frames(a.exp)
    if a.c1max is not None and a.c1min is not None:
        frames._global = (float(a.c1max), float(a.c1min))
        print(f"explicit contrast scale: C1max={a.c1max:.1f} C1min={a.c1min:.1f}",
              flush=True)
    elif a.scale == "global":
        frames.set_global_limits(_scale_films)
    else:
        print("per-film contrast: strips are NOT comparable between films",
              flush=True)
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
        # Mirror the whole datapoint when its poles are reversed relative to
        # the cell's convention, so "pole 1" is the same physical end in every
        # film. Driven by the SAME flag the trajectory colours use, so the two
        # panels can never disagree (P15 stage 6 rule 4).
        if pole_swap.get(dp, False):
            tiles = [np.fliplr(np.asarray(t)) for t in tiles]
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
