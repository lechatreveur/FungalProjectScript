#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone M160 UMAP explorer (stage 6, P14).

Follows the house format set by `SingleCellDataAnalysis/FC_AE_3d_umap.py`, the
Sept17 reference explorer: light theme (`#f4f6f8` page, white panels, slate
toolbar), 3D/2D dimension toggle, "Color by" dropdown, Viridis, and a sidebar of
cards carrying the cell statistics, the Pol1/Pol2 intensity profile in red and
blue, and the vertical strip.

Two deliberate departures from that reference, both about readability:

1. **Colour range.** The reference sets `cmin`/`cmax` from the raw min and max of
   the colour array. On a skewed axis one outlier flattens everything else into
   the bottom of the scale. Here each axis gets robust limits from its 2nd and
   98th percentiles, so the bulk of the cells use the full colour range; values
   outside simply clamp. The reference already acknowledges the problem for one
   axis with its hand-tuned `remap_for_display`; this is the general form.
2. **Polarity dynamics.** The intensity profile is taller, carries a zero line,
   and marks the division frame, so the Pol1/Pol2 relationship is legible rather
   than a pair of thin traces in a 240 px box.

**Standalone, not the reference manifold.** The scaler and UMAP are fit on
M160's own cells, so the coordinates are M160's own and are not comparable with
the Sept17 manifold or the M156 maps. P1 requires the reference fit plus
`.transform()` for cross-experiment work.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

EXP_NAME = "2026_08_28_M160"
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_FEATURES = _SSD_OUT / "features" / "umap_features_m160.csv"
DEFAULT_STACKED = (_SSD_OUT / "features" / "unaligned_pairs_quant"
                   / "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
DEFAULT_STRIPS = _SSD_OUT / "strips"
DEFAULT_OUT = _SSD_OUT / "umap_m160_standalone.html"

FEATURES = ["pol1_a", "pol1_mid", "pol1_v", "pol2_a", "pol2_mid", "pol2_v",
            "NC_score", "Periodicity", "a1a2", "d", "dd"]

# label -> column. Polarity-site axes first, then tracking-quality axes.
COLOR_AXES = [
    ("Pol1 Mid Intensity", "pol1_mid"),
    ("Pol2 Mid Intensity", "pol2_mid"),
    ("Pol1 Variability", "pol1_v"),
    ("Pol2 Variability", "pol2_v"),
    ("Pole Asymmetry (dd)", "dd"),
    ("Pole Distance (d)", "d"),
    ("Periodicity", "Periodicity"),
    ("NC Score", "NC_score"),
    ("Model-only %", "model_only_pct"),
    ("Stage-3 GOOD %", "stage3_good_pct"),
    ("Frames", "n_frames"),
]

CSS = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100vw; height: 100vh; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: row; overflow: hidden; background: #f4f6f8; }
    #main { flex: 2 2 0%; display: flex; flex-direction: column; border-right: 2px solid #d1d5db; background: #fff; min-width: 0; }
    #toolbar { padding: 8px 12px; background: #1e293b; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
    #toolbar label { color: #94a3b8; font-size: 0.8rem; white-space: nowrap; }
    #toolbar .note { color: #fbbf24; font-size: 0.72rem; margin-left: auto; text-align: right; line-height: 1.25; }
    select { background: #334155; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: 4px 10px; font-size: 0.85rem; cursor: pointer; }
    #plot-div { flex: 1 1 0%; width: 100%; height: 100%; min-height: 0; }
    #sidebar { flex: 1 1 0%; min-width: 320px; max-width: 460px; padding: 20px; overflow-y: auto; background: #fff; box-shadow: -2px 0 12px rgba(0,0,0,0.06); display: flex; flex-direction: column; gap: 14px; }
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
    """Colour limits that the bulk of the data actually spans.

    Raw min/max lets a single outlier compress every other cell into one end of
    the scale, which is what makes a skewed axis unreadable. Percentile limits
    keep the gradient where the cells are; anything beyond clamps."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--stacked", type=Path, default=DEFAULT_STACKED)
    ap.add_argument("--strips", type=Path, default=DEFAULT_STRIPS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-strips", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    miss = [c for c in FEATURES if c not in df.columns]
    if miss:
        raise SystemExit(f"feature table is missing {miss}")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"cells: {len(df)}", flush=True)

    X = StandardScaler().fit_transform(df[FEATURES].values.astype(float))
    print("fitting 3D and 2D UMAP on the standard-scaled eleven features "
          "(standalone: fit on M160's own cells) ...", flush=True)
    e3 = umap.UMAP(n_components=3, random_state=42, n_jobs=1).fit_transform(X)
    e2 = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(X)

    axes = [(lab, col) for lab, col in COLOR_AXES if col in df.columns]
    color_arrays, color_limits = {}, {}
    for lab, col in axes:
        vals = [None if pd.isna(v) else round(float(v), 4) for v in df[col]]
        color_arrays[lab] = vals
        lo, hi = robust_limits(vals)
        color_limits[lab] = [round(lo, 4), round(hi, 4)]
        print(f"  {lab:<22} robust range [{lo:.3g}, {hi:.3g}]  "
              f"raw [{np.nanmin(df[col]):.3g}, {np.nanmax(df[col]):.3g}]", flush=True)

    traj = {}
    st = pd.read_csv(a.stacked)
    for cid, g in st.groupby("cell_id"):
        g = g.sort_values("time_point")
        traj[str(cid)] = dict(
            t=[int(v) for v in g["time_point"].values],
            p1=[round(float(v), 3) for v in g["pol1_int_corr"].values],
            p2=[round(float(v), 3) for v in g["pol2_int_corr"].values])

    meta_cols = [c for c in ("films", "local_cids", "n_films", "n_frames",
                             "model_only_pct", "stage3_good_pct",
                             "pol1_mid", "pol2_mid", "d", "dd",
                             "Periodicity", "NC_score") if c in df.columns]
    cells = []
    for i, r in df.iterrows():
        cells.append(dict(
            gid=str(r["cell_id"]), i=int(i),
            x3=float(e3[i, 0]), y3=float(e3[i, 1]), z3=float(e3[i, 2]),
            x2=float(e2[i, 0]), y2=float(e2[i, 1]),
            meta={c: (None if pd.isna(r[c]) else
                      (round(float(r[c]), 4) if isinstance(r[c], (int, float, np.floating))
                       else str(r[c]))) for c in meta_cols},
            tdiv=(None if ("t_div" not in df.columns or pd.isna(r.get("t_div")))
                  else float(r["t_div"]))))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    opts = "".join(f'<option value="{lab}">{lab}</option>' for lab, _ in axes)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8"/>\n')
        f.write("<title>M160 Manifold Explorer 2D/3D (standalone)</title>\n")
        f.write(f"<style>{CSS}</style>\n</head>\n<body>\n")
        f.write('<div id="main">\n  <div id="toolbar">\n')
        f.write('    <label>Dimension:</label><select id="dim-select">'
                '<option value="3D">3D UMAP</option><option value="2D">2D UMAP</option></select>\n')
        f.write(f'    <label>Color by:</label><select id="color-select">{opts}</select>\n')
        f.write(f'    <div class="note"><b>Standalone map</b> &mdash; fit on M160\'s own '
                f'{len(df)} cells; not comparable with the Sept17 manifold.<br>'
                f'Colour limits are 2nd&ndash;98th percentile. Built {stamp}.</div>\n')
        f.write('  </div>\n  <div id="plot-div"></div>\n</div>\n')
        f.write('<div id="sidebar"><div id="content"><div class="placeholder">'
                'Click any point in the manifold to view its polarity dynamics and features.'
                '</div></div></div>\n')
        f.write("<script>")
        f.write(plotly_js())
        f.write("</script>\n<script>\nvar CELLS=")
        f.write(json.dumps(cells, separators=(",", ":")))
        f.write(";\nvar COLORS=")
        f.write(json.dumps(color_arrays, separators=(",", ":")))
        f.write(";\nvar LIMITS=")
        f.write(json.dumps(color_limits, separators=(",", ":")))
        f.write(";\nvar TRAJ=")
        f.write(json.dumps(traj, separators=(",", ":")))
        f.write(";\nvar STRIPS={};\n</script>\n")

        if not a.no_strips:
            n = 0
            for _, r in df.iterrows():
                gid = str(r["cell_id"])
                p = a.strips / f"{gid}.png"
                if not p.exists():
                    continue
                f.write('<script>STRIPS[')
                f.write(json.dumps(gid))
                f.write(']="data:image/png;base64,')
                f.write(base64.b64encode(p.read_bytes()).decode())
                f.write('";</script>\n')
                n += 1
            print(f"embedded strips: {n}", flush=True)

        f.write(r"""<script>
var is3D = true;
var plotDiv = document.getElementById('plot-div');

function renderPlot(){
  var key = document.getElementById('color-select').value;
  var vals = COLORS[key], lim = LIMITS[key];
  var tr = {
    type: is3D ? 'scatter3d' : 'scattergl',
    mode: 'markers',
    x: CELLS.map(c => is3D ? c.x3 : c.x2),
    y: CELLS.map(c => is3D ? c.y3 : c.y2),
    text: CELLS.map(c => c.gid),
    hovertemplate: '%{text}<br>' + key + ': %{marker.color:.4g}<extra></extra>',
    marker: { size: is3D ? 4 : 6, color: CELLS.map(c => vals[c.i]),
              colorscale: 'Viridis', cmin: lim[0], cmax: lim[1], showscale: true,
              colorbar: { title: { text: key, side: 'right' } },
              line: { width: 0.3, color: '#334155' } }
  };
  if (is3D) tr.z = CELLS.map(c => c.z3);
  var ax = { showgrid:true, gridcolor:'#e2e8f0', zeroline:false };
  var layout = { margin:{l:0,r:0,b:0,t:10}, autosize:true,
                 paper_bgcolor:'#fff', plot_bgcolor:'#fff',
                 font:{color:'#475569',size:11},
                 xaxis: Object.assign({title:'UMAP 1'}, ax),
                 yaxis: Object.assign({title:'UMAP 2'}, ax),
                 scene: { xaxis:{title:'UMAP 1'}, yaxis:{title:'UMAP 2'}, zaxis:{title:'UMAP 3'} } };
  Plotly.newPlot(plotDiv, [tr], layout, {responsive:true, displayModeBar:true});
  bindClick();
}

function qualBadge(p){
  if (p === null || p === undefined) return {txt:'N/A', col:'#6b7280'};
  if (p < 1)  return {txt:'segmentation throughout', col:'#16a34a'};
  if (p < 5)  return {txt:'mostly segmented',        col:'#2563eb'};
  if (p < 15) return {txt:'partly inferred',         col:'#d97706'};
              return {txt:'largely inferred',        col:'#dc2626'};
}

function bindClick(){
  plotDiv.on('plotly_click', function(ev){
    var gid = ev.points[0].text;
    var c = CELLS.find(function(q){ return q.gid === gid; });
    if (!c) return;
    var mo = c.meta.model_only_pct, b = qualBadge(mo);
    var h = '<div class="card"><h2>Cell: ' + gid + '</h2>';
    h += '<div class="stat"><span>Mask provenance</span><span class="val">' +
         (mo === null ? 'N/A' : mo.toFixed(2) + '%') +
         '<span class="qual-badge" style="background:' + b.col + '">' + b.txt + '</span></span></div>';
    ['stage3_good_pct','n_films','n_frames','pol1_mid','pol2_mid','d','dd','Periodicity','NC_score']
      .forEach(function(k){
        if (c.meta[k] === undefined) return;
        var v = c.meta[k];
        h += '<div class="stat"><span>' + k + '</span><span class="val">' +
             (v === null ? 'N/A' : (typeof v === 'number' ? v.toFixed(4) : v)) + '</span></div>';
      });
    if (c.meta.films) h += '<div class="stat"><span>films</span><span class="val">' +
                            String(c.meta.films).replace(/\|/g, '<br>') + '</span></div>';
    h += '</div>';
    h += '<div class="card"><h2>Polarity Site Dynamics</h2><div id="traj-div"></div>' +
         '<p class="legend">Red: Pol1 &nbsp;|&nbsp; Blue: Pol2 &nbsp;|&nbsp; ' +
         'grey line = zero (cytoplasm level)</p></div>';
    if (STRIPS[gid]) h += '<div class="card"><h2>Cell Timelapse Strip</h2>' +
      '<img src="' + STRIPS[gid] + '" style="width:100%;image-rendering:pixelated;border-radius:4px;"/>' +
      '<p class="legend">First frame → last (top → bottom)</p></div>';
    document.getElementById('content').innerHTML = h;

    var tr = TRAJ[gid];
    if (!tr) return;
    var shapes = [{type:'line', xref:'paper', x0:0, x1:1, y0:0, y1:0,
                   line:{color:'#94a3b8', width:1, dash:'dot'}}];
    if (c.tdiv !== null && c.tdiv !== undefined)
      shapes.push({type:'line', x0:c.tdiv, x1:c.tdiv, yref:'paper', y0:0, y1:1,
                   line:{color:'#a855f7', width:1.5, dash:'dash'}});
    Plotly.newPlot('traj-div',
      [{x:tr.t, y:tr.p1, mode:'lines', name:'Pol1', line:{color:'#ef4444', width:2}},
       {x:tr.t, y:tr.p2, mode:'lines', name:'Pol2', line:{color:'#3b82f6', width:2}}],
      {margin:{l:46,r:10,b:38,t:10},
       xaxis:{title:'Frame (sequence-continuous)', showgrid:false},
       yaxis:{title:'Intensity − cytoplasm', gridcolor:'#e2e8f0', zeroline:false},
       shapes:shapes, showlegend:false,
       paper_bgcolor:'transparent', plot_bgcolor:'transparent'},
      {displayModeBar:false, responsive:true});
  });
}

document.getElementById('dim-select').addEventListener('change', function(){
  is3D = this.value === '3D'; renderPlot();
});
document.getElementById('color-select').addEventListener('change', renderPlot);
renderPlot();
</script>
</body>
</html>
""")

    mb = a.out.stat().st_size / 1e6
    prov = dict(artifact=a.out.name, created=stamp,
                created_by="SingleCellQuantificationHPC/build_umap_html_m160.py",
                experiment=EXP_NAME, standalone=True,
                format_follows="SingleCellDataAnalysis/FC_AE_3d_umap.py (Sept17 reference)",
                umap="n_components=3 and 2, random_state=42, n_jobs=1, StandardScaler",
                colour_limits="2nd-98th percentile per axis",
                n_cells=int(len(df)), features=FEATURES,
                colour_axes={lab: color_limits[lab] for lab, _ in axes})
    (a.out.parent / "_provenance_umap_m160.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {a.out}  ({mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
