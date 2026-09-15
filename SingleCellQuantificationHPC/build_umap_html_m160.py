#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone M160 UMAP explorer (stage 6, P14).

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

**Standalone**: the autoencoder and the UMAP are both fit on M160 alone, so
these coordinates are M160's own and are not comparable with the Sept17
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

EXP_NAME = "2026_08_28_M160"
_SSD = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_FEATURES_DIR = _SSD / "features"
DEFAULT_MODEL = _SSD / "fc_ae_3d_m160.pth"
DEFAULT_STRIPS = _SSD / "strips"
DEFAULT_OUT = _SSD / "umap_m160_standalone.html"
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]

COLOR_AXES = [
    ("Pol1 Mid Intensity", "pol1_mid"), ("Pol2 Mid Intensity", "pol2_mid"),
    ("Pol1 Variability", "pol1_v"), ("Pol2 Variability", "pol2_v"),
    ("Pole Asymmetry (dd)", "dd"), ("Pole Distance (d)", "d"),
    ("Periodicity", "Periodicity"), ("NC Score", "NC_score"),
    ("Model-only %", "model_only_pct"), ("Stage-3 GOOD %", "stage3_good_pct"),
]

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
    #traj-div { width:100%; height:300px; }
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
    ap.add_argument("--no-strips", action="store_true")
    a = ap.parse_args()

    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(
        {"M160": str(a.features_dir)})
    print(f"datapoints: {len(gids)}", flush=True)

    model = MultimodalAutoencoder3D()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        _, _, lat = model(torch.from_numpy(X_traj).float(),
                          torch.from_numpy(X_feat).float())
    lat = lat.numpy()
    print(f"latents: {lat.shape}; fitting UMAP on them (standalone) ...", flush=True)
    e3 = umap.UMAP(n_components=3, random_state=42, n_jobs=1).fit_transform(lat)
    e2 = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(lat)

    feats = pd.read_csv(a.features_dir / "umap_features_m160.csv")
    # the loader keys as M160_<global_cell_id>_<film>; rebuild that to join
    feats["loader_gid"] = ("M160_" + feats.global_cell_id.astype(str)
                           + "_" + feats.film.astype(str))
    feats = feats.set_index("loader_gid")
    order = film_order(a.exp)

    stacked = pd.read_csv(a.features_dir / "unaligned_pairs_quant"
                          / "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
    traj = {}
    for cid, g in stacked.groupby("cell_id"):
        g = g.sort_values("time_point")
        traj[str(cid)] = dict(t=[int(v) for v in g.time_point],
                              p1=[round(float(v), 3) for v in g.pol1_int_corr],
                              p2=[round(float(v), 3) for v in g.pol2_int_corr])

    meta_cols = ["film", "local_cid", "n_frames", "model_only_pct", "stage3_good_pct",
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
    opts = "".join(f'<option value="{lab}">{lab}</option>' for lab, _ in COLOR_AXES)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8"/>\n')
        f.write("<title>M160 Manifold Explorer 2D/3D (standalone)</title>\n")
        f.write(f"<style>{CSS}</style>\n</head>\n<body>\n<div id='main'>\n<div id='toolbar'>\n")
        f.write('<label>Dimension:</label><select id="dim-select">'
                '<option value="3D">3D UMAP</option><option value="2D">2D UMAP</option></select>\n')
        f.write(f'<label>Color by:</label><select id="color-select">{opts}</select>\n')
        f.write('<label>Range:</label><input type="number" id="cmin" step="any" placeholder="min">'
                '<input type="number" id="cmax" step="any" placeholder="max">\n')
        f.write('<span class="chk"><input type="checkbox" id="link-chk" checked>'
                '<label for="link-chk" style="color:#cbd5e1">Link cell across films</label></span>\n')
        f.write(f'<div class="note"><b>Standalone</b> &mdash; autoencoder and UMAP both fit on '
                f'M160 alone ({len(cells)} datapoints, {n_multi} cells spanning &ge;2 films).<br>'
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
        f.write(";\nvar TRAJ=")
        f.write(json.dumps(traj, separators=(",", ":")))
        f.write(";\nvar STRIPS={};\n</script>\n")

        if not a.no_strips:
            n = 0
            for c in cells:
                p = a.strips / f"{c['gid']}.png"
                if not p.exists():
                    continue
                f.write('<script>STRIPS[')
                f.write(json.dumps(c["gid"]))
                f.write(']="data:image/png;base64,')
                f.write(base64.b64encode(p.read_bytes()).decode())
                f.write('";</script>\n')
                n += 1
            print(f"embedded strips: {n}", flush=True)

        f.write(r"""<script>
var is3D = true, selected = null;
var plotDiv = document.getElementById('plot-div');
var NSEG = 6;   // segments per link, for the opacity ramp

// global cell -> its datapoints, in the order the cell was imaged
var GROUPS = {};
CELLS.forEach(function(c){ (GROUPS[c.gcid] = GROUPS[c.gcid] || []).push(c); });
Object.keys(GROUPS).forEach(function(k){ GROUPS[k].sort(function(a,b){ return a.ord - b.ord; }); });

function xy(c){ return is3D ? [c.x3, c.y3, c.z3] : [c.x2, c.y2, 0]; }

function linkTraces(){
  if (!document.getElementById('link-chk').checked) return [];
  var dull = [], act = [];
  for (var s = 0; s < NSEG; s++){ dull.push({x:[],y:[],z:[]}); act.push({x:[],y:[],z:[]}); }
  Object.keys(GROUPS).forEach(function(gc){
    var g = GROUPS[gc];
    if (g.length < 2) return;
    var isSel = selected && selected.gcid === gc;
    var buf = isSel ? act : dull;
    for (var i = 0; i < g.length - 1; i++){
      var A = xy(g[i]), B = xy(g[i+1]);
      for (var s = 0; s < NSEG; s++){
        var f0 = s/NSEG, f1 = (s+1)/NSEG;
        buf[s].x.push(A[0]+f0*(B[0]-A[0]), A[0]+f1*(B[0]-A[0]), null);
        buf[s].y.push(A[1]+f0*(B[1]-A[1]), A[1]+f1*(B[1]-A[1]), null);
        buf[s].z.push(A[2]+f0*(B[2]-A[2]), A[2]+f1*(B[2]-A[2]), null);
      }
    }
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
  var vals = COLORS[key], lim = LIMITS[key];
  var mn = parseFloat(document.getElementById('cmin').value);
  var mx = parseFloat(document.getElementById('cmax').value);
  var cmin = isNaN(mn) ? lim[0] : mn, cmax = isNaN(mx) ? lim[1] : mx;
  var markers = {
    type: is3D ? 'scatter3d' : 'scattergl', mode: 'markers',
    x: CELLS.map(function(c){ return is3D ? c.x3 : c.x2; }),
    y: CELLS.map(function(c){ return is3D ? c.y3 : c.y2; }),
    text: CELLS.map(function(c){ return c.gid; }),
    hovertemplate: '%{text}<br>' + key + ': %{marker.color:.4g}<extra></extra>',
    marker: { size: is3D ? 4 : 6, color: CELLS.map(function(c){ return vals[c.i]; }),
              colorscale:'Viridis', cmin:cmin, cmax:cmax, showscale:true,
              colorbar:{ title:{ text:key, side:'right' } },
              line:{ width:0.3, color:'#334155' } },
    showlegend: false
  };
  if (is3D) markers.z = CELLS.map(function(c){ return c.z3; });
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
function bindClick(){
  if (bound) return; bound = true;
  plotDiv.on('plotly_click', function(ev){
    var pt = ev.points[0];
    if (!pt || pt.data.mode === 'lines') return;
    var c = CELLS.find(function(q){ return q.gid === pt.text; });
    if (!c) return;
    selected = c;
    var sibs = GROUPS[c.gcid] || [];
    var mo = c.meta.model_only_pct, b = qualBadge(mo);
    var h = '<div class="card"><h2>' + c.gid + '</h2>';
    h += '<div class="stat"><span>global cell</span><span class="val">' + c.gcid + '</span></div>';
    h += '<div class="stat"><span>this cell appears in</span><span class="val">' +
         sibs.length + ' film' + (sibs.length === 1 ? '' : 's') + '</span></div>';
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
    h += '<div class="card"><h2>Polarity Site Dynamics</h2><div id="traj-div"></div>' +
         '<p class="legend">Red: Pol1 &nbsp;|&nbsp; Blue: Pol2 &nbsp;|&nbsp; grey = cytoplasm level</p></div>';
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
         yaxis:{title:'Intensity − cytoplasm', gridcolor:'#e2e8f0', zeroline:false},
         shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,
                  line:{color:'#94a3b8',width:1,dash:'dot'}}],
         showlegend:false, paper_bgcolor:'transparent', plot_bgcolor:'transparent'},
        {displayModeBar:false, responsive:true});
    }
    renderPlot();
  });
}

document.getElementById('dim-select').addEventListener('change', function(){
  is3D = this.value === '3D'; renderPlot(); });
document.getElementById('color-select').addEventListener('change', function(){
  document.getElementById('cmin').value = ''; document.getElementById('cmax').value = '';
  renderPlot(); });
document.getElementById('cmin').addEventListener('change', renderPlot);
document.getElementById('cmax').addEventListener('change', renderPlot);
document.getElementById('link-chk').addEventListener('change', renderPlot);
renderPlot();
</script>
</body>
</html>
""")

    prov = dict(artifact=a.out.name, created=stamp,
                created_by="SingleCellQuantificationHPC/build_umap_html_m160.py",
                experiment=EXP_NAME, standalone=True,
                umap_fit_on="autoencoder latents (fc_ae_3d_m160.pth)",
                model=str(a.model), n_datapoints=len(cells),
                n_global_cells_multi_film=int(n_multi),
                colour_limits="2nd-98th percentile per axis, manually overridable",
                links="global cell, film order, graded opacity")
    (a.out.parent / "_provenance_umap_m160.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {a.out}  ({a.out.stat().st_size/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
