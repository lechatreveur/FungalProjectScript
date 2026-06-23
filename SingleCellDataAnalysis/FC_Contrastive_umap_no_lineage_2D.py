#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FC_Contrastive_umap.py
Self-contained 3D UMAP explorer with:
  - Raw (cytoplasm-corrected) trajectory display
  - Pre-built vertical cell strip images
  - Color-axis dropdown (Cycle Score, Pol1 Mid, Periodicity, NC Score, Pole Distance)
"""
import os, sys, json, base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch, umap, numpy as np, pandas as pd
import plotly.express as px, plotly.utils

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_Contrastive_train import FCTrajectoryEncoder, EXPERIMENTS, LATENT_DIM

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_contrastive_no_lineage_final.pth"
OUTPUT_HTML = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_manifold_explorer_no_lineage.html"
STRIPS_DIR  = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/vertical_strips/"
VIDEO_GIDS  = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/video_gids.txt"
CYCLE_SCORES= "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/cycle_stage_scores.npy"
DEVICE      = torch.device("cpu")

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_plotly_js() -> str:
    import plotly
    p = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    return open(p, encoding="utf-8").read()

def load_raw_trajectories(experiments_dict):
    raw = {}
    for exp_name, exp_dir in experiments_dict.items():
        csv_path = os.path.join(exp_dir, "unaligned_pairs_quant",
                                "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(exp_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df['global_cell_id'] = exp_name + "_" + df['cell_id'].astype(str)
        df.sort_values(['global_cell_id', 'time_point'], inplace=True)
        for gid, grp in df.groupby('global_cell_id'):
            if len(grp) != 101: continue
            p1 = grp['pol1_int_corr'].values
            p2 = grp['pol2_int_corr'].values
            if p2.sum() > p1.sum(): p1, p2 = p2, p1
            raw[gid] = {'p1': p1.tolist(), 'p2': p2.tolist()}
    return raw

def get_strip_b64(gid: str) -> str:
    path = os.path.join(STRIPS_DIR, f"{gid}.png")
    if not os.path.exists(path): return ""
    return "data:image/png;base64," + base64.b64encode(open(path,"rb").read()).decode()

def load_cycle_scores(gids):
    """Align video-AE cycle scores to the FC GID list. NaN for unmatched cells."""
    scores = np.full(len(gids), np.nan)
    if not (os.path.exists(VIDEO_GIDS) and os.path.exists(CYCLE_SCORES)):
        print("⚠️  Cycle score files not found — skipping.")
        return scores
    with open(VIDEO_GIDS) as f:
        v_gids = [l.strip() for l in f if l.strip()]
    v_scores = np.load(CYCLE_SCORES)
    score_map = dict(zip(v_gids, v_scores))
    for i, gid in enumerate(gids):
        if gid in score_map:
            scores[i] = float(score_map[gid])
    matched = np.sum(~np.isnan(scores))
    print(f"🔗 Cycle scores matched: {matched}/{len(gids)} cells")
    return scores


# Film folder mapping: (exp_label, source, field) -> film_subfolder_name
_FILM_FOLDER_MAP = {
    ('Sept17', 'GFP1', 1): 'A14_1TP1_F1',
    ('Sept17', 'GFP2', 2): 'A14_1TP2_F1',
    ('M92', 'GFP1', 'F0'): 'A14-YES-1t-FBFBF_F0',
    ('M92', 'GFP1', 'F1'): 'A14-YES-1t-FBFBF_F1',
    ('M92', 'GFP1', 'F2'): 'A14-YES-1t-FBFBF_F2',
    ('M92', 'GFP2', 'F0'): 'A14-YES-1t-FBFBF-5_F0',
    ('M92', 'GFP2', 'F1'): 'A14-YES-1t-FBFBF-5_F1',
    ('M92', 'GFP2', 'F2'): 'A14-YES-1t-FBFBF-5_F2',
    ('M93', 'GFP1', 'F0'): 'A14_FL_1_F0',
    ('M93', 'GFP1', 'F1'): 'A14_FL_1_F1',
    ('M93', 'GFP1', 'F2'): 'A14_FL_1_F2',
    ('M93', 'GFP2', 'F0'): 'A14_FL_3_F0',
    ('M93', 'GFP2', 'F1'): 'A14_FL_3_F1',
    ('M93', 'GFP2', 'F2'): 'A14_FL_3_F2',
    ('June25_20m', 'GFP1', 'F0'): 'A14_10_20min',
}

_ID_MAP_PATHS = {
    'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/unaligned_pairs_quant/id_map_unaligned.csv',
    'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/unaligned_pairs_quant/id_map_unaligned.csv',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/unaligned_pairs_quant/id_map_unaligned.csv',
}

_SEPT17_STACKED = '/Volumes/X10 Pro/Movies/2025_09_17/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv'


def _resolve_orig_cell(exp_label: str, local_id: int,
                       id_maps: dict, sept17_stacked) -> tuple:
    """Return (base_dir, film_name, orig_cell_id).
    film_name='' means TrackedCells is directly inside base (June25_20m case)."""
    base = EXPERIMENTS[exp_label]
    if exp_label == 'Sept17':
        if sept17_stacked is None: return None, None, None
        row = sept17_stacked[sept17_stacked.cell_id == local_id]
        if row.empty: return None, None, None
        row = row.iloc[0]
        film = _FILM_FOLDER_MAP.get((exp_label, row.source, int(row.tp)))
        return base, film, int(row.orig_gfp_id)
    elif exp_label == 'June25_20m':
        # EXPERIMENTS['June25_20m'] already ends inside the film folder;
        # TrackedCells_A14_10_20min sits directly under base.
        return base, '', local_id  # '' = no film subfolder
    else:
        df = id_maps.get(exp_label)
        if df is None: return None, None, None
        row = df[df.new_cell_id == local_id]
        if row.empty: return None, None, None
        row = row.iloc[0]
        orig_id = int(row.orig_str_id.split(':')[1])
        film = _FILM_FOLDER_MAP.get((exp_label, row.source, row.field))
        return base, film, orig_id


def load_cell_areas(gids):
    """
    For each FC GID, resolve the correct cell_data.csv via the id_map and
    return max(cell_area) over the 101 frames as a proxy for cell size.
    NaN is returned for cells whose data file cannot be found.
    """
    print("📐 Loading cell areas from cell_data.csv files...")
    id_maps = {}
    for exp, p in _ID_MAP_PATHS.items():
        if os.path.exists(p):
            id_maps[exp] = pd.read_csv(p)
    sept17_stacked = None
    if os.path.exists(_SEPT17_STACKED):
        sept17_stacked = pd.read_csv(_SEPT17_STACKED)

    areas = np.full(len(gids), np.nan)
    known_labels = sorted(EXPERIMENTS.keys(), key=lambda x: -len(x))
    n_found = 0

    # Special-case: June25_20m film name for TrackedCells directory
    JUNE25_FILM = 'A14_10_20min'

    for i, gid in enumerate(gids):
        exp_label = next((lbl for lbl in known_labels if gid.startswith(lbl + '_')), None)
        if exp_label is None: continue
        local_id = int(gid[len(exp_label) + 1:])
        base, film, orig_id = _resolve_orig_cell(exp_label, local_id, id_maps, sept17_stacked)
        if orig_id is None: continue

        if film == '':  # June25_20m: TrackedCells directly under base
            csv_p = os.path.join(base, f"TrackedCells_{JUNE25_FILM}",
                                 f"cell_{orig_id}_data.csv")
        elif film is None:
            continue
        else:
            csv_p = os.path.join(base, film, f"TrackedCells_{film}",
                                 f"cell_{orig_id}_data.csv")

        if not os.path.exists(csv_p): continue
        df = pd.read_csv(csv_p)
        if 'cell_area' not in df.columns: continue
        vals = pd.to_numeric(df['cell_area'], errors='coerce').dropna().values
        if len(vals) == 0: continue
        areas[i] = float(vals.max())
        n_found += 1

    print(f"📐 Cell areas loaded: {n_found}/{len(gids)} cells")
    return areas


def remap_for_display(score: float) -> float:
    """
    Piecewise-linear remap so Stage 1 (raw 0–0.4) fills 80% of the colorscale.
    This makes color variation in the growing stage clearly visible.

      Raw score   Stage          Display value
      0.0 – 0.4  Stage 1 (80%)  0.00 – 0.80
      0.4 – 0.7  Stage 2        0.80 – 0.90
      0.7 – 0.9  Stage 3        0.90 – 0.95
      0.9 – 1.0  Stage 4        0.95 – 1.00
    """
    if np.isnan(score): return np.nan
    if score < 0.4:  return score / 0.4 * 0.80
    if score < 0.7:  return 0.80 + (score - 0.4) / 0.3 * 0.10
    if score < 0.9:  return 0.90 + (score - 0.7) / 0.2 * 0.05
    return           0.95 + (score - 0.9) / 0.1 * 0.05

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("📥 Loading data...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)

    print("🧠 Loading model & extracting latents...")
    model = FCTrajectoryEncoder(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        latents = model(torch.from_numpy(X_traj).float(),
                        torch.from_numpy(X_feat).float()).numpy()

    print("🗺️ Running 3D UMAP...")
    embedding = umap.UMAP(n_components=3, random_state=42, n_jobs=1).fit_transform(latents)

    # ── Color arrays ────────────────────────────────────────────────────────────
    cycle_scores = load_cycle_scores(gids)
    cycle_display = [None if np.isnan(v) else round(remap_for_display(v), 4)
                     for v in cycle_scores]
    cell_areas = load_cell_areas(gids)
    area_display = [None if np.isnan(v) else round(float(v), 2) for v in cell_areas]
    color_arrays = {
        "Cell Area (max, px²)":  area_display,
        "Cycle Stage Score":      cycle_display,
        "Pol1 Mid Intensity":    [round(float(X_feat[i,1]),4) for i in range(len(gids))],
        "Pol2 Mid Intensity":    [round(float(X_feat[i,4]),4) for i in range(len(gids))],
        "Periodicity":           [round(float(X_feat[i,7]),4) for i in range(len(gids))],
        "NC Score":              [round(float(X_feat[i,6]),4) for i in range(len(gids))],
    }
    color_arrays_json = json.dumps(color_arrays)

    # ── Initial figure (coloured by Cell Area) ───────────────────────────────
    df = pd.DataFrame(embedding, columns=["x","y","z"])
    df["gid"] = gids
    df["exp"] = labels
    df["color"] = cell_areas

    fig = px.scatter_3d(df, x="x", y="y", z="z",
        color="color", symbol="exp", hover_data=["gid"],
        title="Fungal Multimodal Manifold Explorer (3D) - No Lineage Stitching",
        color_continuous_scale="Viridis")
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=40), autosize=True,
                      coloraxis_colorbar=dict(title="Cell Area"))
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ── Trajectory & strip data ─────────────────────────────────────────────
    print("📂 Loading raw trajectories...")
    raw_traj = load_raw_trajectories(EXPERIMENTS)
    print("🎥 Loading cell strips...")
    traj_dict = {}
    for i, gid in enumerate(gids):
        raw = raw_traj.get(gid, {})
        traj_dict[gid] = {
            "p1":    raw.get('p1', X_traj[i][:,0].tolist()),
            "p2":    raw.get('p2', X_traj[i][:,1].tolist()),
            "f":     [round(float(X_feat[i,1]),4), round(float(X_feat[i,7]),4),
                      round(float(X_feat[i,6]),4), round(float(cycle_scores[i]),4)
                      if not np.isnan(cycle_scores[i]) else None],
            "strip": get_strip_b64(gid),
        }
    traj_json = json.dumps(traj_dict)

    # ── Embed JS ────────────────────────────────────────────────────────────
    print("📦 Embedding Plotly JS...")
    plotly_js = get_plotly_js()

    # ── Build HTML ──────────────────────────────────────────────────────────
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Fungal Manifold Explorer 3D (No Lineage)</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100vw; height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex; flex-direction: row; overflow: hidden; background: #f4f6f8;
    }
    #main {
      flex: 2 2 0%; display: flex; flex-direction: column;
      border-right: 2px solid #d1d5db; background: #fff; min-width: 0;
    }
    #toolbar {
      padding: 8px 12px; background: #1e293b; display: flex;
      align-items: center; gap: 12px; flex-shrink: 0;
    }
    #toolbar label { color: #94a3b8; font-size: 0.8rem; white-space: nowrap; }
    #color-select {
      background: #334155; color: #e2e8f0; border: 1px solid #475569;
      border-radius: 6px; padding: 4px 10px; font-size: 0.85rem; cursor: pointer;
    }
    #plot-div { flex: 1 1 0%; width: 100%; height: 100%; min-height: 0; }
    #sidebar {
      flex: 1 1 0%; min-width: 320px; max-width: 460px;
      padding: 20px; overflow-y: auto; background: #fff;
      box-shadow: -2px 0 12px rgba(0,0,0,0.06);
      display: flex; flex-direction: column; gap: 14px;
    }
    .placeholder { color:#94a3b8; text-align:center; margin-top:80px; font-style:italic; font-size:0.95rem; }
    .card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px; }
    .card h2 { font-size:0.9rem; color:#1e293b; border-bottom:1px solid #e2e8f0; padding-bottom:6px; margin-bottom:10px; }
    .stat { display:flex; justify-content:space-between; font-size:0.82rem; color:#475569; padding:3px 0; border-bottom:1px dashed #e2e8f0; }
    .stat:last-child { border-bottom:none; }
    .val { font-weight:700; color:#0284c7; font-family:monospace; }
    .cycle-badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; color:#fff; margin-left:6px; }
    #traj-div { width:100%; height:240px; }
    .legend { font-size:0.7rem; color:#64748b; text-align:center; margin-top:3px; }
  </style>
</head>
<body>
  <div id="main">
    <div id="toolbar">
      <label>Color by:</label>
      <select id="color-select">
        <option value="Cell Area (max, px&#178;)">Cell Area (max, px²)</option>
        <option value="Cycle Stage Score">Cycle Stage Score</option>
        <option value="Pol1 Mid Intensity">Pol1 Mid Intensity</option>
        <option value="Pol2 Mid Intensity">Pol2 Mid Intensity</option>
        <option value="Periodicity">Periodicity</option>
        <option value="NC Score">NC Score</option>
      </select>
    </div>
    <div id="plot-div"></div>
  </div>
  <div id="sidebar">
    <div id="content">
      <div class="placeholder">Click any point in the 3D manifold to view its trajectory and features.</div>
    </div>
  </div>

  <script>PLOTLY_JS_PLACEHOLDER</script>
  <script>
    var plotData      = PLOT_JSON_PLACEHOLDER;
    var trajData      = TRAJ_JSON_PLACEHOLDER;
    var colorArrays   = COLOR_ARRAYS_PLACEHOLDER;

    var plotDiv = document.getElementById('plot-div');
    Plotly.newPlot(plotDiv, plotData.data, plotData.layout, {responsive:true, displayModeBar:true});

    // ── Color dropdown ────────────────────────────────────────────────────
    document.getElementById('color-select').addEventListener('change', function() {
      var key = this.value;
      var vals = colorArrays[key];
      var valid = vals.filter(function(v){ return v !== null; });
      var lo = Math.min.apply(null, valid);
      var hi = Math.max.apply(null, valid);
      Plotly.restyle(plotDiv, {
        'marker.color': [vals],
        'marker.cmin':  [lo],
        'marker.cmax':  [hi]
      });
      Plotly.relayout(plotDiv, {'coloraxis.colorbar.title.text': key});
    });

    // ── Cycle stage label helper — uses RAW score (not remapped display value)
    function cycleLabel(score) {
      if (score === null) return {txt: 'N/A',                  col: '#6b7280'};
      if (score < 0.4)   return {txt: 'Stage 1: Growing',      col: '#16a34a'};
      if (score < 0.7)   return {txt: 'Stage 2: Nuclear Div',  col: '#2563eb'};
      if (score < 0.9)   return {txt: 'Stage 3: Septation',    col: '#d97706'};
                         return {txt: 'Stage 4: Hourglass',     col: '#dc2626'};
    }

    // ── Click handler ─────────────────────────────────────────────────────
    plotDiv.on('plotly_click', function(eventData) {
      var pt   = eventData.points[0];
      var gid  = pt.customdata[0];
      var exp  = pt.fullData.name;
      var cell = trajData[gid];
      if (!cell) return;

      var cyc   = cell.f[3];
      var clbl  = cycleLabel(cyc);
      var cycStr = cyc !== null ? cyc.toFixed(3) : 'N/A';

      document.getElementById('content').innerHTML =
        '<div class="card">' +
          '<h2>Cell: ' + gid + '</h2>' +
          '<div class="stat"><span>Experiment</span><span class="val">' + exp + '</span></div>' +
          '<div class="stat"><span>Cycle Stage</span><span class="val">' +
            cycStr + '<span class="cycle-badge" style="background:' + clbl.col + '">' + clbl.txt + '</span></span></div>' +
          '<div class="stat"><span>Pol1 Mid Intensity</span><span class="val">' + cell.f[0].toFixed(4) + '</span></div>' +
          '<div class="stat"><span>Periodicity Score</span><span class="val">' + cell.f[1].toFixed(4) + '</span></div>' +
          '<div class="stat"><span>NC Score</span><span class="val">' + cell.f[2].toFixed(4) + '</span></div>' +
        '</div>' +
        '<div class="card">' +
          '<h2>Intensity Profile</h2>' +
          '<div id="traj-div"></div>' +
          '<p class="legend">Red: Pol1 &nbsp;|&nbsp; Blue: Pol2</p>' +
        '</div>' +
        (cell.strip ?
          '<div class="card">' +
            '<h2>Cell Timelapse Strip</h2>' +
            '<img src="' + cell.strip + '" style="width:100%;image-rendering:pixelated;border-radius:4px;"/>' +
            '<p class="legend">Frames 0 \u2192 100 (top \u2192 bottom)</p>' +
          '</div>' : '');

      var frames = Array.from({length: cell.p1.length}, function(_,i){ return i; });
      Plotly.newPlot('traj-div',
        [{x:frames, y:cell.p1, mode:'lines', name:'Pol1', line:{color:'#ef4444',width:2}},
         {x:frames, y:cell.p2, mode:'lines', name:'Pol2', line:{color:'#3b82f6',width:2}}],
        {margin:{l:40,r:10,b:36,t:10},
         xaxis:{title:'Frame',showgrid:false},
         yaxis:{title:'Intensity (corr.)'},
         showlegend:false,
         paper_bgcolor:'transparent', plot_bgcolor:'transparent'},
        {displayModeBar:false, responsive:true});
    });
  </script>
</body>
</html>"""

    html = html.replace("PLOTLY_JS_PLACEHOLDER",   plotly_js)
    html = html.replace("PLOT_JSON_PLACEHOLDER",    plot_json)
    html = html.replace("TRAJ_JSON_PLACEHOLDER",    traj_json)
    html = html.replace("COLOR_ARRAYS_PLACEHOLDER", color_arrays_json)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard saved: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
