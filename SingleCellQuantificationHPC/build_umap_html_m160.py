#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone M160 UMAP explorer (stage 6, P14).

**Standalone, not the reference manifold.** The scaler and the UMAP are fit on
M160's own cells, so the coordinates are M160's own. This map is NOT comparable
with the reference manifold, nor with the M156 maps, which are each standalone
in the same way. P1 and P15 cover the distinction: a cross-experiment comparison
requires fitting on the reference and projecting others with `.transform()`.

Why this is a new module rather than a copy of `build_umap_html_m156_*.py`:
those scripts do not build an explorer. They fit a UMAP and then re-embed into
an existing explorer HTML, lifting each cell's trajectories, autocorrelation
arrays, fit parameters and strips out of it. M160 has no such template, so the
cell objects are constructed here from the stage-4/5 artifacts directly.

Inputs (all from the M160 outputs folder, P4):
    features/umap_features_m160.csv                    eleven features + provenance
    features/unaligned_pairs_quant/stacked_...csv      cytoplasm-corrected traces
    strips/<global_cell_id>.png                        vertical strips

UMAP settings match the previous builds: standard-scaled features, then
`umap.UMAP(n_components=2, random_state=42, n_jobs=1)`.

The HTML is written incrementally, because 328 base64 strips do not want to be
held in memory at once on this machine.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
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
COLOR_AXES = [("model_only_pct", "Model-only %"), ("stage3_good_pct", "Stage-3 GOOD %"),
              ("pol1_mid", "Pol1 mid"), ("Periodicity", "Periodicity"),
              ("NC_score", "NC score"), ("d", "Pole distance"),
              ("n_frames", "Frames"), ("n_films", "Films")]


def plotly_js():
    import plotly
    p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return p.read_text(encoding="utf-8")


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
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"feature table is missing {missing}")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"cells: {len(df)}", flush=True)

    X = StandardScaler().fit_transform(df[FEATURES].values.astype(float))
    print("fitting 2D UMAP on the standard-scaled eleven features "
          "(standalone: fit on M160's own cells) ...", flush=True)
    emb = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(X)
    df["umap_x"], df["umap_y"] = emb[:, 0], emb[:, 1]

    # trajectories, kept as short lists so the page stays manageable
    traj = {}
    st = pd.read_csv(a.stacked)
    for cid, g in st.groupby("cell_id"):
        g = g.sort_values("time_point")
        traj[str(cid)] = dict(
            t=[int(v) for v in g["time_point"].values],
            p1=[round(float(v), 3) for v in g["pol1_int_corr"].values],
            p2=[round(float(v), 3) for v in g["pol2_int_corr"].values],
        )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = [c for c in ("films", "local_cids", "n_films", "n_frames",
                             "model_only_pct", "stage3_good_pct") if c in df.columns]

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write(f"<title>M160 UMAP (standalone)</title>\n")
        f.write("<style>body{margin:0;font:13px system-ui,sans-serif;background:#111;color:#eee;}"
                "#wrap{display:flex;height:100vh}#left{flex:1 1 60%;min-width:0}"
                "#right{flex:0 0 40%;max-width:560px;overflow:auto;border-left:1px solid #333;padding:10px}"
                "#strip{max-width:100%;image-rendering:pixelated;border:1px solid #333}"
                "h2{margin:4px 0;font-size:15px}table{font-size:12px;border-collapse:collapse}"
                "td{padding:1px 6px;border-bottom:1px solid #222}"
                "#banner{background:#3a2d00;color:#ffd97a;padding:6px 10px;font-size:12px}"
                "select{background:#222;color:#eee;border:1px solid #444;padding:3px}</style>\n")
        f.write("<script>")
        f.write(plotly_js())
        f.write("</script>\n")
        f.write(f"<div id='banner'><b>Standalone map.</b> Scaler and UMAP fit on M160's "
                f"own {len(df)} cells, so these coordinates are M160's own and are "
                f"<b>not</b> comparable with the reference manifold or the M156 maps. "
                f"Built {stamp}.</div>\n")
        f.write("<div id='wrap'><div id='left'><div id='plot' style='height:100%'></div></div>"
                "<div id='right'><div>Colour: <select id='caxis'></select></div>"
                "<h2 id='cid'>click a point</h2><div id='meta'></div>"
                "<div id='traj' style='height:260px'></div>"
                "<div><img id='strip'></div></div></div>\n")

        f.write("<script>\nconst CELLS=")
        f.write(json.dumps([
            dict(gid=str(r["cell_id"]), x=float(r["umap_x"]), y=float(r["umap_y"]),
                 meta={c: (None if pd.isna(r[c]) else
                           (float(r[c]) if isinstance(r[c], (int, float, np.floating)) else str(r[c])))
                       for c in meta_cols},
                 col={k: (None if pd.isna(r.get(k, np.nan)) else float(r[k]))
                      for k, _ in COLOR_AXES if k in df.columns})
            for _, r in df.iterrows()], separators=(",", ":")))
        f.write(";\n")
        f.write("const AXES=")
        f.write(json.dumps([[k, lab] for k, lab in COLOR_AXES if k in df.columns]))
        f.write(";\nconst TRAJ=")
        f.write(json.dumps(traj, separators=(",", ":")))
        f.write(";\nconst STRIPS={};\n")
        f.write("</script>\n")

        if not a.no_strips:
            n = 0
            for _, r in df.iterrows():
                gid = str(r["cell_id"])
                p = a.strips / f"{gid}.png"
                if not p.exists():
                    continue
                b64 = base64.b64encode(p.read_bytes()).decode()
                f.write("<script>STRIPS[")
                f.write(json.dumps(gid))
                f.write("]=\"data:image/png;base64,")
                f.write(b64)
                f.write("\";</script>\n")
                n += 1
                del b64
            print(f"embedded strips: {n}", flush=True)

        f.write("""<script>
const sel=document.getElementById('caxis');
AXES.forEach(([k,lab])=>{const o=document.createElement('option');o.value=k;o.textContent=lab;sel.appendChild(o);});
function draw(axis){
  const c=CELLS.map(d=>d.col[axis]);
  Plotly.react('plot',[{x:CELLS.map(d=>d.x),y:CELLS.map(d=>d.y),mode:'markers',type:'scattergl',
    text:CELLS.map(d=>d.gid),hovertemplate:'%{text}<extra></extra>',
    marker:{size:7,color:c,colorscale:'Viridis',showscale:true,
            colorbar:{title:{text:axis,side:'right'}},line:{width:0.4,color:'#000'}}}],
   {paper_bgcolor:'#111',plot_bgcolor:'#111',font:{color:'#ccc'},margin:{l:40,r:10,t:10,b:40},
    xaxis:{title:'UMAP 1',zeroline:false,gridcolor:'#222'},
    yaxis:{title:'UMAP 2',zeroline:false,gridcolor:'#222'}},{responsive:true});
}
sel.onchange=()=>draw(sel.value);
draw(AXES[0][0]);
document.getElementById('plot').on('plotly_click',ev=>{
  const gid=ev.points[0].text, d=CELLS.find(c=>c.gid===gid);
  document.getElementById('cid').textContent=gid;
  let h='<table>';for(const k in d.meta){h+='<tr><td>'+k+'</td><td>'+d.meta[k]+'</td></tr>';}
  document.getElementById('meta').innerHTML=h+'</table>';
  const tr=TRAJ[gid];
  if(tr){Plotly.react('traj',[
    {x:tr.t,y:tr.p1,name:'Pol1',mode:'lines',line:{color:'#ff5555',width:1.3}},
    {x:tr.t,y:tr.p2,name:'Pol2',mode:'lines',line:{color:'#5599ff',width:1.3}}],
   {paper_bgcolor:'#111',plot_bgcolor:'#111',font:{color:'#ccc',size:10},
    margin:{l:40,r:10,t:6,b:30},legend:{orientation:'h'},
    xaxis:{title:'frame (sequence-continuous)',gridcolor:'#222'},
    yaxis:{title:'intensity − cytoplasm',gridcolor:'#222'}},{responsive:true});}
  document.getElementById('strip').src=STRIPS[gid]||'';
});
</script>
""")

    size_mb = a.out.stat().st_size / 1e6
    prov = dict(artifact=a.out.name, created=stamp,
                created_by="SingleCellQuantificationHPC/build_umap_html_m160.py",
                experiment=EXP_NAME, standalone=True,
                note="scaler and UMAP fit on M160's own cells; not the reference manifold",
                umap="n_components=2, random_state=42, n_jobs=1, StandardScaler",
                n_cells=int(len(df)), features=FEATURES)
    (a.out.parent / "_provenance_umap_m160.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {a.out}  ({size_mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
