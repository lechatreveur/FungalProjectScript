#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone M162 UMAP explorer (stage 6, P14).

Copy-to-modify variant of the M162 original (P15): experiment-dated scripts
are frozen records, so the M162 file is untouched and only the experiment
constants change here. M162 is the healthy, well-segmented arm — YES medium,
0.21% model-only frames, BF doubling 3.93 h — and is therefore the natural
reference for a standalone manifold.


UMAP is fit on the **autoencoder latents**, as the Sept17 reference does, not on
the engineered features: the autoencoder folds the 101-frame Pol1/Pol2
trajectory together with the eleven features, and fitting on the features alone
throws the trajectory shape away.

A datapoint is one cell in ONE film over its 101 frames. A global cell followed
across consecutive films is several datapoints, and the **link lines** join them
in film order so you can see how a cell moves across the manifold as it lives.
The link styling follows the M156 explorer: each link is split into segments
with a graded opacity ramp so direction of travel is readable, drawn dull grey
for every cell and bold sky blue for the selected one, behind the markers and
out of the legend. Datapoints flagged mistracked or bad are excluded from the
grouping so a bad segment cannot draw a spurious jump.

Format follows `SingleCellDataAnalysis/FC_AE_3d_umap.py`: light theme, 3D/2D
toggle, "Color by" dropdown, Viridis, card sidebar. Colour limits default to the
2nd–98th percentile, with manual min/max inputs as the M156 page has.

**Standalone**: the autoencoder and the UMAP are both fit on M162 alone, so
these coordinates are M162's own and are not comparable with the Sept17
manifold. P1 requires the reference fit plus `.transform()` for cross-experiment
work.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import umap

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

EXP_NAME = "2026_09_09_M162"
# Derived, never hard-coded: a copied script that keeps the donor's
# literal filenames writes mislabelled artifacts into this experiment's
# folder. Has already happened three times in this pipeline (P3).
EXP_TAG = EXP_NAME.rsplit("_", 1)[-1].lower()   # "m162"
EXP_KEY = EXP_NAME.rsplit("_", 1)[-1]           # "M162", the loader key
_SSD = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_FEATURES_DIR = _SSD / "features"
DEFAULT_MODEL = _SSD / f"fc_ae_3d_{EXP_TAG}.pth"
DEFAULT_STRIPS = _SSD / "strips"
DEFAULT_OUT = _SSD / f"umap_{EXP_TAG}_standalone.html"
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]

# UMAP's n_neighbors is an ABSOLUTE count, but what governs the local-versus-
# global balance is the fraction of the population it covers. The Sept17
# reference used the library default of 15 on 378 cells — 3.97% — so matching
# that fraction, rather than the number, is what makes two maps comparable.
# On 861 FL1 datapoints it gives 34; on the full 6,243 it gives 248.
REF_NEIGHBORS, REF_N = 15, 378
NEIGHBORS_FRAC = REF_NEIGHBORS / REF_N

COLOR_AXES = [
    ("Pol1 Mid Intensity", "pol1_mid"), ("Pol2 Mid Intensity", "pol2_mid"),
    ("Pol1 Variability", "pol1_v"), ("Pol2 Variability", "pol2_v"),
    ("Pole Asymmetry (dd)", "dd"), ("Pole Distance (d)", "d"),
    ("Periodicity", "Periodicity"), ("NC Score", "NC_score"),
    ("Model-only %", "model_only_pct"), ("Stage-3 GOOD %", "stage3_good_pct"),
    ("Division film", "is_division_film"), ("Lineage depth", "segment_depth"),
    ("Cell cycle stage", "stage_min"),
]
# Computed in the page from pol1_mid / pol2_mid / Periodicity / NC_score rather
# than read from a column, because its thresholds are adjustable live.
MODE_AXIS = "Dynamic mode"

CSS = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width:100vw; height:100vh; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; display:flex; flex-direction:row; overflow:hidden; background:#f4f6f8; }
    #main { flex:2 2 0%; display:flex; flex-direction:column; border-right:2px solid #d1d5db; background:#fff; min-width:0; }
    #toolbar { padding:8px 12px; background:#1e293b; display:flex; align-items:center; gap:10px; flex-shrink:0; flex-wrap:wrap; }
    #toolbar label { color:#94a3b8; font-size:0.8rem; white-space:nowrap; }
    #toolbar .note { color:#fbbf24; font-size:0.72rem; margin-left:auto; text-align:right; line-height:1.25; }
    select, input[type=number] { background:#334155; color:#e2e8f0; border:1px solid #475569; border-radius:6px; padding:4px 8px; font-size:0.85rem; }
    input[type=number] { width:86px; }
    #toolbar .chk { color:#cbd5e1; font-size:0.8rem; display:flex; align-items:center; gap:5px; }
    #plot-div { flex:1 1 0%; width:100%; height:100%; min-height:0; }
    #sidebar { flex:1 1 0%; min-width:320px; max-width:460px; padding:20px; overflow-y:auto; background:#fff; box-shadow:-2px 0 12px rgba(0,0,0,0.06); display:flex; flex-direction:column; gap:14px; }
    .placeholder { color:#94a3b8; text-align:center; margin-top:80px; font-style:italic; font-size:0.95rem; }
    .card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px; }
    .card h2 { font-size:0.9rem; color:#1e293b; border-bottom:1px solid #e2e8f0; padding-bottom:6px; margin-bottom:10px; }
    .stat { display:flex; justify-content:space-between; font-size:0.82rem; color:#475569; padding:3px 0; border-bottom:1px dashed #e2e8f0; }
    .stat:last-child { border-bottom:none; }
    .val { font-weight:700; color:#0284c7; font-family:monospace; }
    .qual-badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; color:#fff; margin-left:6px; }
    #traj-div, #acor-div { width:100%; height:260px; margin-top:6px; }
    /* The trajectories stay visible while the rest of the sidebar scrolls;
       the negative top offsets the sidebar's own padding so it sits flush. */
    .sticky-card { position:sticky; top:-20px; z-index:10; background:#fff !important;
                   border:1px solid #cbd5e1; box-shadow:0 10px 15px -3px rgba(0,0,0,0.08);
                   margin-left:-4px; margin-right:-4px; padding-left:16px; padding-right:16px; }
    .collapse-head { display:flex; justify-content:space-between; align-items:center; cursor:pointer; }
    .collapse-head h2 { margin:0; border:none; padding:0; }
    .formula { font-size:0.72rem; color:#334155; background:#f1f5f9; padding:8px;
               border-radius:4px; border:1px solid #e2e8f0; line-height:1.6; margin-top:8px; }
    #thr-row { display:none; gap:6px; align-items:center; flex-wrap:wrap; }
    #thr-row input { width:64px; }
    #thr-row span { color:#94a3b8; font-size:0.72rem; }
    .legend { font-size:0.7rem; color:#64748b; text-align:center; margin-top:3px; }
"""


def plotly_js():
    import plotly
    return (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")


def robust_limits(vals, lo_pct=2.0, hi_pct=98.0):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


N_LAGS = 40      # positive lags shown in the ACF card


def detrended_acf(g):
    """Normalised autocorrelation of each detrended pole trace, positive lags.

    Matches `signal_cor`: detrend, full autocorrelation, divide by the zero-lag
    value, then keep lags 1 upward — which is why the fitted model's `acf0` is
    the value at lag 1, not lag 0.

    The canonical detrend is `model_selector_with_threshold`, an AIC model
    selection that costs as much as the whole feature build. Here the trend is
    removed linearly instead, which is what that selector reduces to for the
    common case and is close for the rest. The FITTED curve drawn over it comes
    from the stored canonical parameters, so a visible gap between raw and fit
    means the two detrends disagreed for that cell.
    """
    out = {}
    t = g["time_point"].values.astype(float)
    for key, col in (("a1", "pol1_int_corr"), ("a2", "pol2_int_corr")):
        y = g[col].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(t)
        if ok.sum() < 8:
            out[key] = []
            continue
        yy, tt = y[ok], t[ok]
        try:
            m, c = np.polyfit(tt, yy, 1)
            d = yy - (m * tt + c)
        except Exception:
            d = yy - np.nanmean(yy)
        ac = np.correlate(d, d, mode="full")
        mid = len(ac) // 2
        z = ac[mid]
        ac = ac / z if z > 1e-8 else np.zeros_like(ac)
        out[key] = [round(float(v), 3) for v in ac[mid + 1: mid + 1 + N_LAGS]]
    return out


def film_order(exp_dir):
    """film -> ordinal among the fluorescence films of its sequence, so a global
    cell's datapoints are linked in the order the cell was actually imaged."""
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    order = {}
    for seq in SEQS:
        if seq not in linkage:
            continue
        for k, film in enumerate([f for f in linkage[seq]["films"] if "FL" in f]):
            order[film] = k
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--exp", type=Path,
                    default=Path("/Volumes/X10 Pro/Movies") / EXP_NAME)
    ap.add_argument("--strips", type=Path, default=DEFAULT_STRIPS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-neighbors", type=int, default=None,
                    help="UMAP n_neighbors; default scales with the dataset to "
                         "hold the Sept17 reference fraction of 15/378")
    ap.add_argument("--fit-film-contains", nargs="+", default=None,
                    help="fit the UMAP on datapoints whose film matches ANY of "
                         "these, and PROJECT the rest with .transform() (P1's "
                         "reference-manifold pattern). M162's polarity signal "
                         "collapses across the series — pol1_mid 20.9 -> 3.7 "
                         "from FL1 to FL4, 52.8%% of FL4 below the polarity "
                         "threshold — so fitting on the late films would make "
                         "the dominant axis photobleaching. Datapoints outside "
                         "the fit set are marked `projected` in the explorer.")
    ap.add_argument("--film-contains", default=None,
                    help="restrict the map to films matching this string")
    ap.add_argument("--no-strips", action="store_true")
    ap.add_argument("--strips-mode", choices=("auto", "embed", "link"), default="auto",
                    help="embed inlines each PNG (self-contained but large); "
                         "link references them relatively (small page, must stay "
                         "beside the strips folder); auto embeds up to 1200 "
                         "datapoints and links above that")
    a = ap.parse_args()

    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(
        {EXP_KEY: str(a.features_dir)})
    if a.film_contains:
        keep = np.array([a.film_contains in g for g in gids], bool)
        X_traj, X_feat = X_traj[keep], X_feat[keep]
        gids = [g for g, k in zip(gids, keep) if k]
        print(f"film filter {a.film_contains!r}: {len(gids)} datapoints", flush=True)
    print(f"datapoints: {len(gids)}", flush=True)

    # Read the latent dimension out of the checkpoint itself rather than
    # hardcoding it, so the explorer can never drift from whatever the trainer
    # last produced. encoder_fc's final Linear maps 128 -> latent_dim.
    sd = torch.load(a.model, map_location="cpu")
    lat_dim = next((v.shape[0] for k, v in sd.items()
                    if k.startswith("encoder_fc") and k.endswith("weight")
                    and v.ndim == 2 and v.shape[1] == 128), 3)
    print(f"checkpoint latent_dim: {lat_dim}", flush=True)
    model = MultimodalAutoencoder3D(latent_dim=int(lat_dim))
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        _, _, lat = model(torch.from_numpy(X_traj).float(),
                          torch.from_numpy(X_feat).float())
    lat = lat.numpy()
    n_nb = a.n_neighbors or max(2, min(len(lat) - 1, round(NEIGHBORS_FRAC * len(lat))))
    print(f"latents: {lat.shape}; n_neighbors {n_nb} "
          f"({100*n_nb/len(lat):.2f}% of {len(lat)}, matching Sept17's "
          f"{100*NEIGHBORS_FRAC:.2f}%); fitting UMAP ...", flush=True)
    if a.fit_film_contains:
        pats = list(a.fit_film_contains)
        fit_mask = np.array([any(pat in g for pat in pats) for g in gids], bool)
        if fit_mask.sum() < 10:
            raise SystemExit(f"--fit-film-contains {pats} matched only "
                             f"{int(fit_mask.sum())} datapoints")
        # n_neighbors is taken on the FIT set, since that is the manifold being
        # learned; projecting more points later does not change its density.
        n_fit = int(fit_mask.sum())
        n_nb = a.n_neighbors or max(2, min(n_fit - 1,
                                           round(NEIGHBORS_FRAC * n_fit)))
        print(f"fitting on {n_fit} datapoints matching {pats}, projecting the "
              f"remaining {len(lat) - n_fit}; n_neighbors {n_nb} "
              f"({100*n_nb/n_fit:.2f}% of the fit set)", flush=True)
        r3 = umap.UMAP(n_components=3, n_neighbors=n_nb,
                       random_state=42, n_jobs=1).fit(lat[fit_mask])
        r2 = umap.UMAP(n_components=2, n_neighbors=n_nb,
                       random_state=42, n_jobs=1).fit(lat[fit_mask])
        e3 = r3.transform(lat)
        e2 = r2.transform(lat)
    else:
        fit_mask = np.ones(len(lat), bool)
        e3 = umap.UMAP(n_components=3, n_neighbors=n_nb,
                       random_state=42, n_jobs=1).fit_transform(lat)
        e2 = umap.UMAP(n_components=2, n_neighbors=n_nb,
                       random_state=42, n_jobs=1).fit_transform(lat)

    feats = pd.read_csv(a.features_dir / f"umap_features_{EXP_TAG}.csv")
    # Cell-cycle stage, from cell_cycle_regress_m160.py. `stage_source` marks
    # whether the cell's division was actually detected (curated) or the stage
    # comes from the regression (estimated, typical error ~49 min).
    cyc = a.features_dir.parent / "cell_cycle" / "cell_cycle_stage_by_datapoint.csv"
    if cyc.exists():
        cc = pd.read_csv(cyc)[["cell_id", "stage_min", "stage_source"]]
        feats = feats.merge(cc, on="cell_id", how="left")
        n_cur = int((feats.stage_source == "curated").sum())
        print(f"cell-cycle stage merged: {int(feats.stage_min.notna().sum())} "
              f"datapoints ({n_cur} curated, "
              f"{int(feats.stage_min.notna().sum()) - n_cur} estimated)", flush=True)
    else:
        feats["stage_min"] = np.nan
        feats["stage_source"] = None
        print("(no cell-cycle stage table; that axis will be empty)", flush=True)
    # the loader keys as M160_<global_cell_id>_<film>; rebuild that to join
    feats["loader_gid"] = (EXP_KEY + "_" + feats.global_cell_id.astype(str)
                           + "_" + feats.film.astype(str))
    feats = feats.set_index("loader_gid")
    order = film_order(a.exp)

    stacked = pd.read_csv(a.features_dir / "unaligned_pairs_quant"
                          / "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
    traj, acf = {}, {}
    for cid, g in stacked.groupby("cell_id"):
        g = g.sort_values("time_point")
        acf[str(cid)] = detrended_acf(g)
        traj[str(cid)] = dict(t=[int(v) for v in g.time_point],
                              p1=[round(float(v), 3) for v in g.pol1_int_corr],
                              p2=[round(float(v), 3) for v in g.pol2_int_corr])

    # fit parameters for the ACF card, straight from the canonical acor table
    acor = pd.read_csv(a.features_dir / "unaligned_pairs_quant"
                       / "acor_detrended_results.csv").set_index("cell_id")
    FITC = ["pol1_A1", "pol1_tau1", "pol1_tau2", "pol1_f", "pol1_phi", "pol1_C",
            "pol1_acf0", "pol2_A1", "pol2_tau1", "pol2_tau2", "pol2_f", "pol2_phi",
            "pol2_C", "pol2_acf0", "precision_sum", "freq_distance_sum", "NC_score"]
    FITC = [c for c in FITC if c in acor.columns]
    fitp = {str(i): {c: (None if pd.isna(r[c]) else round(float(r[c]), 5)) for c in FITC}
            for i, r in acor.iterrows()}

    meta_cols = ["film", "local_cid", "n_frames", "model_only_pct", "stage3_good_pct",
                 "segment_id", "segment_depth", "is_division_film", "div_frame",
                 "div_bounce", "stage_min", "stage_source",
                 "pol1_mid", "pol2_mid", "d", "dd", "Periodicity", "NC_score"]
    cells, color_arrays = [], {lab: [] for lab, _ in COLOR_AXES}
    for i, gid in enumerate(gids):
        if gid not in feats.index:
            continue
        r = feats.loc[gid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        cells.append(dict(
            gid=str(r["cell_id"]), gcid=str(r["global_cell_id"]),
            # lineage: a datapoint belongs to one SEGMENT of the mother/daughter
            # tree. Links run along a segment and then fork to its daughters.
            seg=(None if pd.isna(r.get("segment_id")) else str(r["segment_id"])),
            par=(None if pd.isna(r.get("segment_parent")) else str(r["segment_parent"])),
            divf=bool(r.get("is_division_film", False)),
            cur=bool(str(r.get("stage_source", "")) == "curated"),
            ord=int(order.get(str(r["film"]), 0)), i=len(cells),
            x3=float(e3[i, 0]), y3=float(e3[i, 1]), z3=float(e3[i, 2]),
            x2=float(e2[i, 0]), y2=float(e2[i, 1]),
            meta={c: (None if pd.isna(r.get(c)) else
                      (round(float(r[c]), 4) if isinstance(r[c], (int, float, np.floating))
                       else str(r[c]))) for c in meta_cols if c in feats.columns}))
        for lab, col in COLOR_AXES:
            v = r.get(col, np.nan)
            color_arrays[lab].append(None if pd.isna(v) else round(float(v), 4))

    limits = {lab: list(map(lambda z: round(z, 4), robust_limits(color_arrays[lab])))
              for lab, _ in COLOR_AXES}
    for lab, _ in COLOR_AXES:
        print(f"  {lab:<22} robust {limits[lab]}", flush=True)

    n_multi = sum(1 for _, n in pd.Series([c["gcid"] for c in cells]).value_counts().items() if n >= 2)
    print(f"cells on the map: {len(cells)}; global cells with >=2 datapoints: {n_multi}",
          flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    opts = ('<option value="%s">%s</option>' % (MODE_AXIS, MODE_AXIS)
            + "".join(f'<option value="{lab}">{lab}</option>' for lab, _ in COLOR_AXES))
    a.out.parent.mkdir(parents=True, exist_ok=True)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8"/>\n')
        f.write("<title>M162 Manifold Explorer 2D/3D (standalone)</title>\n")
        f.write(f"<style>{CSS}</style>\n</head>\n<body>\n<div id='main'>\n<div id='toolbar'>\n")
        f.write('<label>Dimension:</label><select id="dim-select">'
                '<option value="3D">3D UMAP</option><option value="2D">2D UMAP</option></select>\n')
        f.write(f'<label>Color by:</label><select id="color-select">{opts}</select>\n')
        f.write('<label>Range:</label><input type="number" id="cmin" step="any" placeholder="min">'
                '<input type="number" id="cmax" step="any" placeholder="max">\n')
        f.write('<span class="chk"><input type="checkbox" id="link-chk" checked>'
                '<label for="link-chk" style="color:#cbd5e1">Link cell across films</label></span>\n')
        f.write('<span class="chk"><input type="checkbox" id="cur-chk">'
                '<label for="cur-chk" style="color:#cbd5e1">Curated stage only</label></span>\n')
        f.write('<div id="thr-row">'
                '<span>pol1&ge;</span><input type="number" id="thr-pol1" step="any" value="4.04">'
                '<span>pol2&ge;</span><input type="number" id="thr-pol2" step="any" value="2.0">'
                '<span>mono osc&gt;</span><input type="number" id="thr-mono" step="any" value="1.14">'
                '<span>bi osc&gt;</span><input type="number" id="thr-bi" step="any" value="1.60">'
                '<span>NC&ge;</span><input type="number" id="thr-nc" step="any" value="0">'
                '</div>\n')
        f.write(f'<div class="note"><b>Standalone</b> &mdash; autoencoder and UMAP both fit on '
                f'M162 alone ({len(cells)} datapoints, {n_multi} cells spanning &ge;2 films).<br>'
                f'UMAP on AE latents. Colour limits 2nd&ndash;98th pct unless overridden. {stamp}.</div>\n')
        f.write('</div>\n<div id="plot-div"></div>\n</div>\n')
        f.write('<div id="sidebar"><div id="content"><div class="placeholder">'
                'Click any point to view its polarity dynamics, or follow a link line to see '
                'where that cell goes next.</div></div></div>\n')
        f.write("<script>")
        f.write(plotly_js())
        f.write("</script>\n<script>\nvar CELLS=")
        f.write(json.dumps(cells, separators=(",", ":")))
        f.write(";\nvar COLORS=")
        f.write(json.dumps(color_arrays, separators=(",", ":")))
        f.write(";\nvar LIMITS=")
        f.write(json.dumps(limits, separators=(",", ":")))
        # A COMMON y-range for every trajectory panel. Auto-scaling per cell
        # makes a flat cell look as dynamic as an oscillating one, and hides
        # the FL1->FL4 amplitude collapse entirely, so the panels have to share
        # an axis to be comparable at all. Robust percentiles rather than
        # min/max, so a single spike does not flatten every other cell.
        _all = []
        for _v in traj.values():
            _all.extend([x for x in _v.get("p1", []) if x is not None])
            _all.extend([x for x in _v.get("p2", []) if x is not None])
        if _all:
            _arr = np.asarray(_all, float)
            _arr = _arr[np.isfinite(_arr)]
            _lo, _hi = np.percentile(_arr, [0.5, 99.5])
            _pad = 0.05 * (_hi - _lo)
            traj_range = [float(_lo - _pad), float(_hi + _pad)]
        else:
            traj_range = None
        print(f"trajectory y-range (shared, 0.5-99.5 pct): {traj_range}", flush=True)
        f.write(";\nvar TRAJ_YRANGE=")
        f.write(json.dumps(traj_range))
        f.write(";\nvar TRAJ=")
        f.write(json.dumps(traj, separators=(",", ":")))
        f.write(";\nvar ACF=")
        f.write(json.dumps(acf, separators=(",", ":")))
        f.write(";\nvar FITP=")
        f.write(json.dumps(fitp, separators=(",", ":")))
        f.write(";\nvar STRIPS={};\n</script>\n")

        # Two ways to carry the strips.
        #
        # "embed" inlines each PNG as base64, which keeps the page a single
        # self-contained file. That is fine for a few hundred datapoints and
        # impossible for a full cohort: 6,244 strips at ~85 KB are 0.53 GB on
        # disk and about 0.7 GB once base64-expanded into one HTML file, which
        # no browser will open comfortably.
        #
        # "link" writes a relative path instead and lets the browser fetch each
        # strip on demand. The page is then ~10 MB and must sit next to the
        # strips directory, which it does by default.
        mode = a.strips_mode
        if mode == "auto":
            mode = "embed" if len(cells) <= 1200 else "link"
        if not a.no_strips:
            n = 0
            rel = os.path.relpath(a.strips, a.out.parent)
            for c in cells:
                p = a.strips / f"{c['gid']}.png"
                if not p.exists():
                    continue
                f.write('<script>STRIPS[')
                f.write(json.dumps(c["gid"]))
                if mode == "embed":
                    f.write(']="data:image/png;base64,')
                    f.write(base64.b64encode(p.read_bytes()).decode())
                    f.write('";</script>\n')
                else:
                    f.write("]=")
                    f.write(json.dumps(f"{rel}/{c['gid']}.png"))
                    f.write(";</script>\n")
                n += 1
            print(f"strips ({mode}): {n}", flush=True)
            if mode == "link":
                print(f"  page references {rel}/ relatively; keep them together",
                      flush=True)

        f.write(r"""<script>
var is3D = true, selected = null;
var MODE_AXIS = "Dynamic mode";

// Dynamic-mode thresholds.
//
// pol1 and pol2 keep the M156 values, which sit sensibly in M162's range: 16%
// of datapoints clear pol1 >= 4.04 and 34% clear pol2 >= 2.0.
//
// The OSCILLATION thresholds do not transfer. M156 used Periodicity > 5.0 and
// > 6.5, but M162's Periodicity runs -0.35 to 4.24 with a median of 0.46, so
// those cut off literally nothing — 0.00% of datapoints clear either. They are
// recalibrated here to this dataset's own distribution: 1.14 is the 90th
// percentile and 1.60 roughly the 96th, preserving the M156 intent that
// bipolar oscillation is the stricter call.
//
// NC score is POSITIVE when the two poles are negatively correlated, because
// signal_cor computes it as -(A + C) over the cross-correlation fit. So an
// oscillatory cell needs NC >= thr, not <=. 30% of datapoints are at or above
// zero, which is the default.
// Trajectory panels share one y-axis by default so cells are comparable.
// Auto-scaling per cell makes a flat cell look as dynamic as an
// oscillating one, and hides the FL1->FL4 amplitude collapse entirely.
// Toggle off to inspect one flat cell's own dynamics.
var TRAJ_YSHARED = true;
function setTrajShared(v){ TRAJ_YSHARED = !!v; if (selected) showCell(selected); }
var THR = { pol1: 4.04, pol2: 2.0, mono: 1.14, bi: 1.60, nc: 0.0 };
var MODE_LABELS = ["Non-polarized", "Monopolar", "Monopolar Osc", "Bipolar", "Bipolar Osc"];
var MODE_COLORS = ["#94a3b8", "#f59e0b", "#ef4444", "#10b981", "#3b82f6"];
var MODE_SCALE = [[0.0,'#94a3b8'],[0.2,'#94a3b8'],[0.2,'#f59e0b'],[0.4,'#f59e0b'],
                  [0.4,'#ef4444'],[0.6,'#ef4444'],[0.6,'#10b981'],[0.8,'#10b981'],
                  [0.8,'#3b82f6'],[1.0,'#3b82f6']];

function getCategory(p1, p2, per, nc){
  if (p1 === null || p1 === undefined || p1 < THR.pol1) return 0;   // Non-polarized
  // NC is positive for anti-correlated poles, so oscillation needs NC >= thr
  var osc = (nc !== null && nc !== undefined) ? (nc >= THR.nc) : true;
  if (p2 === null || p2 === undefined || p2 < THR.pol2)
    return (per > THR.mono && osc) ? 2 : 1;                          // Monopolar (Osc)
  return (per > THR.bi && osc) ? 4 : 3;                              // Bipolar (Osc)
}
function modeOf(c){
  return getCategory(c.meta.pol1_mid, c.meta.pol2_mid,
                     c.meta.Periodicity, c.meta.NC_score);
}
function readThresholds(){
  var g = function(id, d){ var v = parseFloat(document.getElementById(id).value);
                           return isNaN(v) ? d : v; };
  THR.pol1 = g('thr-pol1', 4.04); THR.pol2 = g('thr-pol2', 2.0);
  THR.mono = g('thr-mono', 1.14); THR.bi   = g('thr-bi', 1.60);
  THR.nc   = g('thr-nc', 0.0);
}
function debounce(fn, ms){ var t; return function(){ clearTimeout(t); t = setTimeout(fn, ms); }; }
var plotDiv = document.getElementById('plot-div');
var NSEG = 6;   // segments per link, for the opacity ramp

// Lineage tree: datapoints grouped by SEGMENT, in the order they were imaged.
// A segment is a stretch of one cell's life between divisions; its daughters
// hang off its end. Linking by segment rather than by global_cell_id is what
// makes a division draw as a fork instead of two overlapping paths.
var GROUPS = {}, KIDS = {};
CELLS.forEach(function(c){
  var k = c.seg || ("gc:" + c.gcid);
  (GROUPS[k] = GROUPS[k] || []).push(c);
});
Object.keys(GROUPS).forEach(function(k){
  GROUPS[k].sort(function(a,b){ return a.ord - b.ord; });
  var p = GROUPS[k][0].par;
  if (p) (KIDS[p] = KIDS[p] || []).push(k);
});

// every segment in the same lineage as this one: ancestors and descendants
function lineageOf(seg){
  if (!seg) return {};
  var out = {}, stack = [seg];
  while (stack.length){
    var s = stack.pop();
    if (out[s]) continue;
    out[s] = 1;
    (KIDS[s] || []).forEach(function(k){ stack.push(k); });
    var g = GROUPS[s];
    if (g && g[0].par) stack.push(g[0].par);
  }
  return out;
}

function xy(c){ return is3D ? [c.x3, c.y3, c.z3] : [c.x2, c.y2, 0]; }

// "Curated stage only" hides datapoints whose cell-cycle stage was estimated by
// the regression rather than anchored on a detected division. It filters the
// markers AND the link lines, so a lineage is not drawn through points that are
// no longer shown.
function curatedOnly(){ var e = document.getElementById('cur-chk'); return e && e.checked; }
function shown(c){ return !curatedOnly() || c.cur; }

function linkTraces(){
  if (!document.getElementById('link-chk').checked) return [];
  var dull = [], act = [];
  for (var s = 0; s < NSEG; s++){ dull.push({x:[],y:[],z:[]}); act.push({x:[],y:[],z:[]}); }
  var lin = selected ? lineageOf(selected.seg) : {};
  function span(A, B, buf){
    for (var s = 0; s < NSEG; s++){
      var f0 = s/NSEG, f1 = (s+1)/NSEG;
      buf[s].x.push(A[0]+f0*(B[0]-A[0]), A[0]+f1*(B[0]-A[0]), null);
      buf[s].y.push(A[1]+f0*(B[1]-A[1]), A[1]+f1*(B[1]-A[1]), null);
      buf[s].z.push(A[2]+f0*(B[2]-A[2]), A[2]+f1*(B[2]-A[2]), null);
    }
  }
  Object.keys(GROUPS).forEach(function(k){
    var g = GROUPS[k].filter(shown);
    if (!g.length) return;
    var isSel = selected && (lin[k] || (!selected.seg && selected.gcid === g[0].gcid));
    var buf = isSel ? act : dull;
    // along the segment
    for (var i = 0; i < g.length - 1; i++) span(xy(g[i]), xy(g[i+1]), buf);
    // and out to each daughter's first point: this is the fork
    (KIDS[k] || []).forEach(function(kid){
      var d = (GROUPS[kid] || []).filter(shown);
      if (d.length) span(xy(g[g.length-1]), xy(d[0]), buf);
    });
  });
  var out = [];
  function emit(buf, rgb, w0, lo, hi){
    for (var s = 0; s < NSEG; s++){
      if (!buf[s].x.length) continue;
      var op = lo + (hi - lo) * (s / (NSEG - 1));
      var t = { type: is3D ? 'scatter3d' : 'scatter', mode: 'lines',
                x: buf[s].x, y: buf[s].y,
                line: { color: 'rgba(' + rgb + ',' + op.toFixed(3) + ')', width: w0 },
                hoverinfo: 'none', showlegend: false };
      if (is3D) t.z = buf[s].z;
      out.push(t);
    }
  }
  emit(dull, '160,174,192', 1.2, 0.10, 0.25);
  emit(act, '2,132,199', is3D ? 4.8 : 3.8, 0.50, 1.00);
  return out;
}

function renderPlot(){
  var key = document.getElementById('color-select').value;
  var isMode = (key === MODE_AXIS);
  document.getElementById('thr-row').style.display = isMode ? 'flex' : 'none';
  var vals, cmin, cmax, scale, cbar;
  if (isMode){
    readThresholds();
    vals = CELLS.map(modeOf);
    cmin = -0.5; cmax = 4.5; scale = MODE_SCALE;
    cbar = { title:{ text:'Mode', side:'right' }, tickmode:'array',
             tickvals:[0,1,2,3,4], ticktext:MODE_LABELS };
  } else {
    vals = COLORS[key];
    var lim = LIMITS[key];
    var mn = parseFloat(document.getElementById('cmin').value);
    var mx = parseFloat(document.getElementById('cmax').value);
    cmin = isNaN(mn) ? lim[0] : mn; cmax = isNaN(mx) ? lim[1] : mx;
    scale = 'Viridis';
    cbar = { title:{ text:key, side:'right' } };
  }
  var VIS = CELLS.filter(shown);
  var markers = {
    type: is3D ? 'scatter3d' : 'scattergl', mode: 'markers',
    x: VIS.map(function(c){ return is3D ? c.x3 : c.x2; }),
    y: VIS.map(function(c){ return is3D ? c.y3 : c.y2; }),
    text: VIS.map(function(c){ return c.gid; }),
    hovertemplate: '%{text}<br>' + key + ': ' +
                   (isMode ? '%{customdata}' : '%{marker.color:.4g}') + '<extra></extra>',
    customdata: isMode ? VIS.map(function(c){ return MODE_LABELS[modeOf(c)]; }) : undefined,
    marker: { size: is3D ? 4 : 6,
              color: VIS.map(function(c){ return vals[c.i]; }),
              colorscale: scale, cmin:cmin, cmax:cmax, showscale:true,
              colorbar: cbar,
              line:{ width:0.3, color:'#334155' } },
    showlegend: false
  };
  if (is3D) markers.z = VIS.map(function(c){ return c.z3; });
  var ax = { showgrid:true, gridcolor:'#e2e8f0', zeroline:false };
  var layout = { margin:{l:0,r:0,b:0,t:10}, autosize:true,
                 paper_bgcolor:'#fff', plot_bgcolor:'#fff', font:{color:'#475569',size:11},
                 xaxis:Object.assign({title:'UMAP 1'},ax), yaxis:Object.assign({title:'UMAP 2'},ax),
                 scene:{ xaxis:{title:'UMAP 1'}, yaxis:{title:'UMAP 2'}, zaxis:{title:'UMAP 3'} } };
  Plotly.react(plotDiv, linkTraces().concat([markers]), layout,
               {responsive:true, displayModeBar:true});
  bindClick();
}

function qualBadge(p){
  if (p === null || p === undefined) return {txt:'N/A', col:'#6b7280'};
  if (p < 1)  return {txt:'segmentation throughout', col:'#16a34a'};
  if (p < 5)  return {txt:'mostly segmented',        col:'#2563eb'};
  if (p < 15) return {txt:'partly inferred',         col:'#d97706'};
              return {txt:'largely inferred',        col:'#dc2626'};
}

var bound = false;
function showCell(c){
    selected = c;
    var segKey = c.seg || ("gc:" + c.gcid);
    var sibs = GROUPS[segKey] || [];
    var kids = KIDS[segKey] || [];
    var mo = c.meta.model_only_pct, b = qualBadge(mo);
    var h = '<div class="card"><h2>' + c.gid + '</h2>';
    h += '<div class="stat"><span>global cell</span><span class="val">' + c.gcid + '</span></div>';
    h += '<div class="stat"><span>segment spans</span><span class="val">' +
         sibs.length + ' film' + (sibs.length === 1 ? '' : 's') + '</span></div>';
    h += '<div class="stat"><span>divides into</span><span class="val">' +
         (kids.length ? kids.length + ' daughters' : 'no division seen') + '</span></div>';
    if (c.divf) h += '<div class="stat"><span>division film</span>' +
         '<span class="val" style="color:#dc2626">yes, frame ' +
         (c.meta.div_frame === null ? '?' : Math.round(c.meta.div_frame)) +
         ' (excluded from AE training)</span></div>';
    h += '<div class="stat"><span>mask provenance</span><span class="val">' +
         (mo === null ? 'N/A' : mo.toFixed(2) + '%') +
         '<span class="qual-badge" style="background:' + b.col + '">' + b.txt + '</span></span></div>';
    ['film','local_cid','stage3_good_pct','pol1_mid','pol2_mid','d','dd','Periodicity','NC_score']
      .forEach(function(k){
        if (c.meta[k] === undefined) return;
        var v = c.meta[k];
        h += '<div class="stat"><span>' + k + '</span><span class="val">' +
             (v === null ? 'N/A' : (typeof v === 'number' ? v.toFixed(4) : v)) + '</span></div>';
      });
    h += '</div>';
    if (sibs.length > 1){
      h += '<div class="card"><h2>Path across the manifold</h2>';
      sibs.forEach(function(s, i){
        h += '<div class="stat"><span>' + (i+1) + '. ' + (s.meta.film || s.gid) + '</span>' +
             '<span class="val">' + (s.gid === c.gid ? 'here' : '') + '</span></div>';
      });
      h += '<p class="legend">Blue line on the map, fading light to dark in film order</p></div>';
    }
    if (c.meta.stage_min !== undefined && c.meta.stage_min !== null){
      var isCur = (c.meta.stage_source === 'curated');
      h += '<div class="card"><h2>Cell cycle stage</h2>' +
           '<div class="stat"><span>stage</span><span class="val">' +
           c.meta.stage_min.toFixed(0) + ' min ' +
           (c.meta.stage_min <= 0 ? 'before' : 'after') + ' division' +
           '<span class="qual-badge" style="background:' +
           (isCur ? '#16a34a' : '#d97706') + '">' +
           (isCur ? 'curated' : 'estimated') + '</span></span></div>' +
           (isCur ? '' : '<p class="legend">regressed from area, length, septum '
                       + 'and nuclear signals; typical error ~49 min</p>') +
           '</div>';
    }
    var cat = modeOf(c);
    h += '<div class="card"><h2>Dynamic mode</h2><div class="stat"><span>classified as</span>' +
         '<span class="val"><span class="qual-badge" style="background:' + MODE_COLORS[cat] +
         '">' + MODE_LABELS[cat] + '</span></span></div>' +
         '<div class="stat"><span>Periodicity</span><span class="val">' +
         (c.meta.Periodicity === null ? 'N/A' : c.meta.Periodicity.toFixed(4)) + '</span></div>' +
         '<div class="stat"><span>NC score</span><span class="val">' +
         (c.meta.NC_score === null ? 'N/A' : c.meta.NC_score.toFixed(4)) + '</span></div></div>';
    h += '<div class="card sticky-card"><h2>Polarity Site Dynamics</h2><div id="traj-div"></div>' +
         '<p class="legend">Red: Pol1 &nbsp;|&nbsp; Blue: Pol2 &nbsp;|&nbsp; grey = cytoplasm level</p></div>';
    h += '<div class="card"><div class="collapse-head" id="acor-head">' +
         '<h2>Autocorrelation &amp; Fit Details</h2>' +
         '<span id="acor-ind" style="color:#64748b">&#9658;</span></div>' +
         '<div id="acor-body" style="display:none"><div id="acor-div"></div>' +
         '<div class="formula" id="acor-formula"></div>' +
         '<p class="legend">Faint: measured ACF &nbsp;|&nbsp; dashed: fitted model</p></div></div>';
    if (STRIPS[c.gid]) h += '<div class="card"><h2>Cell Timelapse Strip</h2>' +
      '<img src="' + STRIPS[c.gid] + '" style="width:100%;image-rendering:pixelated;border-radius:4px;"/>' +
      '<p class="legend">Frame 0 → 100 (top → bottom)</p></div>';
    document.getElementById('content').innerHTML = h;

    var tr = TRAJ[c.gid];
    if (tr){
      Plotly.newPlot('traj-div',
        [{x:tr.t, y:tr.p1, mode:'lines', name:'Pol1', line:{color:'#ef4444',width:2}},
         {x:tr.t, y:tr.p2, mode:'lines', name:'Pol2', line:{color:'#3b82f6',width:2}}],
        {margin:{l:46,r:10,b:38,t:10},
         xaxis:{title:'Frame', showgrid:false},
         yaxis:Object.assign({title:'Intensity − cytoplasm', gridcolor:'#e2e8f0',
                              zeroline:false},
                             (TRAJ_YSHARED && TRAJ_YRANGE)
                               ? {range:TRAJ_YRANGE.slice(), autorange:false}
                               : {autorange:true}),
         shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,
                  line:{color:'#94a3b8',width:1,dash:'dot'}}],
         showlegend:false, paper_bgcolor:'transparent', plot_bgcolor:'transparent'},
        {displayModeBar:false, responsive:true});
    }
    setupAcor(c);
}

function bindClick(){
  if (bound) return; bound = true;
  plotDiv.on('plotly_click', function(ev){
    var pt = ev.points[0];
    if (!pt || pt.data.mode === 'lines') return;
    var c = CELLS.find(function(q){ return q.gid === pt.text; });
    if (c){ showCell(c); renderPlot(); }
  });
}

// Autocorrelation card. Collapsed by default, and the plot is only drawn the
// first time it is opened, so thousands of hidden Plotly canvases are never
// created.
function fitModel(lag, A1, tau1, tau2, f, phi, C, acf0){
  var A2 = Math.min(Math.max((acf0 || 0) - (A1 || 0) - (C || 0), 0.0), 1.0);
  var env = (A1 || 0) * Math.exp(-lag / (tau1 || 1)) +
            A2 * Math.exp(-lag / (tau2 || 1)) + (C || 0);
  return env * Math.cos(2 * Math.PI * (f || 0) * lag + (phi || 0));
}
function envText(fp, P){
  var A1 = fp[P+'_A1'] || 0, C = fp[P+'_C'] || 0, acf0 = fp[P+'_acf0'] || 0;
  var A2 = Math.min(Math.max(acf0 - A1 - C, 0), 1);
  return A1.toFixed(3) + '&middot;e<sup>&minus;t/' + (fp[P+'_tau1']||1).toFixed(1) +
         '</sup> + ' + A2.toFixed(3) + '&middot;e<sup>&minus;t/' +
         (fp[P+'_tau2']||1).toFixed(1) + '</sup> + ' + C.toFixed(3);
}
function drawAcor(c){
  var A = ACF[c.gid] || {}, fp = FITP[c.gid] || {};
  var a1 = A.a1 || [], a2 = A.a2 || [];
  var n = Math.max(a1.length, a2.length);
  if (!n){
    document.getElementById('acor-div').innerHTML =
      '<p class="legend">no autocorrelation for this cell</p>';
    return;
  }
  var lags = [], f1 = [], f2 = [];
  for (var i = 0; i < n; i++){
    var lag = i + 1;                       // positive lags start at 1
    lags.push(lag);
    f1.push(fitModel(lag, fp.pol1_A1, fp.pol1_tau1, fp.pol1_tau2, fp.pol1_f,
                     fp.pol1_phi, fp.pol1_C, fp.pol1_acf0));
    f2.push(fitModel(lag, fp.pol2_A1, fp.pol2_tau1, fp.pol2_tau2, fp.pol2_f,
                     fp.pol2_phi, fp.pol2_C, fp.pol2_acf0));
  }
  Plotly.newPlot('acor-div', [
    {x:lags, y:a1, name:'Pol1', mode:'lines', line:{color:'rgba(239,68,68,0.35)', width:2}},
    {x:lags, y:f1, name:'Pol1 fit', mode:'lines', line:{color:'#ef4444', width:2, dash:'dash'}},
    {x:lags, y:a2, name:'Pol2', mode:'lines', line:{color:'rgba(59,130,246,0.35)', width:2}},
    {x:lags, y:f2, name:'Pol2 fit', mode:'lines', line:{color:'#3b82f6', width:2, dash:'dash'}}],
   {margin:{l:44,r:10,b:34,t:8}, showlegend:false,
    xaxis:{title:'lag (frames)', gridcolor:'#e2e8f0'},
    yaxis:{title:'ACF', gridcolor:'#e2e8f0', zeroline:true, zerolinecolor:'#cbd5e1'},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent'},
   {displayModeBar:false, responsive:true});

  var ps = fp.precision_sum, fd = fp.freq_distance_sum;
  var num = function(v){ return (v === null || v === undefined) ? '?' : v.toFixed(4); };
  document.getElementById('acor-formula').innerHTML =
    '<b>ACF(t) = envelope &middot; cos(2&pi;&middot;f&middot;t + &phi;)</b><br>' +
    'Pol1 env = ' + envText(fp, 'pol1') + '<br>' +
    'Pol1 f = ' + num(fp.pol1_f) + ',&nbsp; &phi; = ' + num(fp.pol1_phi) + '<br>' +
    'Pol2 env = ' + envText(fp, 'pol2') + '<br>' +
    'Pol2 f = ' + num(fp.pol2_f) + ',&nbsp; &phi; = ' + num(fp.pol2_phi) + '<br><br>' +
    '<b>Periodicity = precision_sum &minus; freq_distance_sum</b><br>' +
    '&nbsp;&nbsp;= ' + num(ps) + ' &minus; ' + num(fd) + ' = ' + num(c.meta.Periodicity);
}
function setupAcor(c){
  var head = document.getElementById('acor-head');
  if (!head) return;
  var drawn = false;
  head.addEventListener('click', function(){
    var body = document.getElementById('acor-body');
    var ind = document.getElementById('acor-ind');
    if (body.style.display === 'none'){
      body.style.display = 'block';
      ind.innerHTML = '&#9660;';
      if (!drawn){ drawAcor(c); drawn = true; }
      else { Plotly.Plots.resize('acor-div'); }
    } else {
      body.style.display = 'none';
      ind.innerHTML = '&#9658;';
    }
  });
}

// Left/Right step through the datapoints of the same GLOBAL cell, in film
// order — the same navigation the M156 explorer has. Ignored while a form
// control has focus, so typing a threshold does not jump the selection.
function selectCell(c){ showCell(c); renderPlot(); }
document.addEventListener('keydown', function(e){
  var tag = document.activeElement ? document.activeElement.tagName : '';
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  if (!selected) return;
  var group = CELLS.filter(function(q){ return q.gcid === selected.gcid; });
  if (group.length <= 1) return;
  group.sort(function(a, b){ return a.ord - b.ord; });
  var i = group.findIndex(function(q){ return q.gid === selected.gid; });
  if (i === -1) return;
  e.preventDefault();
  var j = (e.key === 'ArrowRight') ? (i + 1) % group.length
                                   : (i - 1 + group.length) % group.length;
  selectCell(group[j]);
});

document.getElementById('dim-select').addEventListener('change', function(){
  is3D = this.value === '3D'; renderPlot(); });
document.getElementById('color-select').addEventListener('change', function(){
  document.getElementById('cmin').value = ''; document.getElementById('cmax').value = '';
  renderPlot(); });
document.getElementById('cmin').addEventListener('change', renderPlot);
document.getElementById('cmax').addEventListener('change', renderPlot);
document.getElementById('link-chk').addEventListener('change', renderPlot);
document.getElementById('cur-chk').addEventListener('change', renderPlot);
['thr-pol1','thr-pol2','thr-mono','thr-bi','thr-nc'].forEach(function(id){
  document.getElementById(id).addEventListener('input', debounce(renderPlot, 250));
});
renderPlot();
</script>
</body>
</html>
""")

    prov = dict(artifact=a.out.name, created=stamp,
                created_by="SingleCellQuantificationHPC/build_umap_html_m162.py",
                experiment=EXP_NAME, standalone=True,
                umap_fit_on=f"autoencoder latents (fc_ae_3d_{EXP_TAG}.pth)",
                n_neighbors=int(n_nb),
                n_neighbors_fraction=round(n_nb / len(lat), 5),
                latent_dim=int(lat_dim),
                model=str(a.model), n_datapoints=len(cells),
                n_global_cells_multi_film=int(n_multi),
                colour_limits="2nd-98th percentile per axis, manually overridable",
                links="global cell, film order, graded opacity")
    (a.out.parent / f"_provenance_umap_{EXP_TAG}.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {a.out}  ({a.out.stat().st_size/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
