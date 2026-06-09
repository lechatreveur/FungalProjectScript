import os, sys, json, base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch, umap, numpy as np, pandas as pd
import plotly.express as px, plotly.utils

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_Contrastive_train import FCTrajectoryEncoder, LATENT_DIM
from SingleCellDataAnalysis.FC_Contrastive_umap import (
    STRIPS_DIR, DEVICE, get_plotly_js, load_raw_trajectories, 
    load_cycle_scores, load_cell_areas, remap_for_display, get_strip_b64
)

MODEL_PATH_MIX = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_contrastive_M133_mix_final.pth"
OUTPUT_HTML = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_manifold_explorer_Strategy1_Mix.html"

EXPERIMENTS_MIX = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/",
    "M133":   "/Volumes/X10 Pro/Movies/2026_04_29_M133/"
}

def main():
    print("📥 Loading mixed data...")
    X_traj, X_feat, gids, labels, _, _ = load_feature_constrained_data(EXPERIMENTS_MIX)
    
    # Fill any NaNs just in case
    X_traj = np.nan_to_num(X_traj, posinf=0.0, neginf=0.0)
    X_feat = np.nan_to_num(X_feat, posinf=0.0, neginf=0.0)

    print("🧠 Loading mixed model & extracting latents...")
    model = FCTrajectoryEncoder(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH_MIX, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad():
        latents = model(torch.from_numpy(X_traj).float(),
                        torch.from_numpy(X_feat).float()).numpy()

    print("🗺️ Fitting UMAP on all latents...")
    reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(latents)

    # Color arrays
    cycle_scores = load_cycle_scores(gids)
    cycle_display = [None if np.isnan(v) else round(remap_for_display(v), 4) for v in cycle_scores]
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

    df = pd.DataFrame(embedding, columns=["x","y","z"])
    df["gid"] = gids
    df["exp"] = labels
    df["color"] = area_display

    fig = px.scatter_3d(df, x="x", y="y", z="z",
        color="color", symbol="exp", hover_data=["gid"],
        title="Fungal Manifold Explorer - WT & M133 (Mixed Training)",
        color_continuous_scale="Viridis")
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=40), autosize=True,
                      coloraxis_colorbar=dict(title="Cell Area"))
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ── Trajectory & strip data ─────────────────────────────────────────────
    print("📂 Loading raw trajectories...")
    raw_traj = load_raw_trajectories(EXPERIMENTS_MIX)
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

    # Embed JS
    plotly_js = get_plotly_js()

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Fungal Manifold Explorer 3D - Mixed Strategy</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100vw; height: 100vh; font-family: sans-serif; display: flex; flex-direction: row; overflow: hidden; }
    #main { flex: 2 2 0%; display: flex; flex-direction: column; border-right: 2px solid #d1d5db; }
    #toolbar { padding: 8px 12px; background: #1e293b; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
    #toolbar label { color: #94a3b8; font-size: 0.8rem; }
    #color-select { background: #334155; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
    #plot-div { flex: 1 1 0%; }
    #sidebar { flex: 1 1 0%; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
    .card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px; }
    .stat { display:flex; justify-content:space-between; font-size:0.82rem; color:#475569; padding:3px 0; border-bottom:1px dashed #e2e8f0; }
    .val { font-weight:700; color:#0284c7; font-family:monospace; }
    #traj-div { width:100%; height:240px; }
  </style>
</head>
<body>
  <div id="main">
    <div id="toolbar">
      <label>Color by:</label>
      <select id="color-select">
        <option value="Cell Area (max, px²)">Cell Area (max, px²)</option>
        <option value="Cycle Stage Score">Cycle Stage Score</option>
        <option value="Pol1 Mid Intensity">Pol1 Mid Intensity</option>
        <option value="Pol2 Mid Intensity">Pol2 Mid Intensity</option>
        <option value="Periodicity">Periodicity</option>
        <option value="NC Score">NC Score</option>
      </select>
    </div>
    <div id="plot-div"></div>
  </div>
  <div id="sidebar"><div id="content">Click any point...</div></div>
  <script>PLOTLY_JS_PLACEHOLDER</script>
  <script>
    var plotData      = PLOT_JSON_PLACEHOLDER;
    var trajData      = TRAJ_JSON_PLACEHOLDER;
    var colorArrays   = COLOR_ARRAYS_PLACEHOLDER;
    var plotDiv = document.getElementById('plot-div');
    Plotly.newPlot(plotDiv, plotData.data, plotData.layout, {responsive:true});
    document.getElementById('color-select').addEventListener('change', function() {
      var key = this.value; var vals = colorArrays[key];
      var valid = vals.filter(function(v){ return v !== null; });
      var lo = Math.min.apply(null, valid), hi = Math.max.apply(null, valid);
      Plotly.restyle(plotDiv, {'marker.color': [vals], 'marker.cmin': [lo], 'marker.cmax': [hi]});
      Plotly.relayout(plotDiv, {'coloraxis.colorbar.title.text': key});
    });
    plotDiv.on('plotly_click', function(eventData) {
      var pt = eventData.points[0], gid = pt.customdata[0], exp = pt.fullData.name, cell = trajData[gid];
      if(!cell) return;
      document.getElementById('content').innerHTML =
        '<div class="card"><h2>Cell: ' + gid + '</h2><div class="stat"><span>Experiment</span><span class="val">' + exp + '</span></div></div>' +
        '<div class="card"><h2>Intensity Profile</h2><div id="traj-div"></div></div>' +
        (cell.strip ? '<div class="card"><h2>Strip</h2><img src="' + cell.strip + '" style="width:100%;image-rendering:pixelated;"/></div>' : '');
      Plotly.newPlot('traj-div', [{y:cell.p1, name:'Pol1', line:{color:'red'}}, {y:cell.p2, name:'Pol2', line:{color:'blue'}}], {margin:{l:40,r:10,b:36,t:10}});
    });
  </script>
</body>
</html>"""

    html = html.replace("PLOTLY_JS_PLACEHOLDER", plotly_js)
    html = html.replace("PLOT_JSON_PLACEHOLDER", plot_json)
    html = html.replace("TRAJ_JSON_PLACEHOLDER", traj_json)
    html = html.replace("COLOR_ARRAYS_PLACEHOLDER", color_arrays_json)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Strategy 1 Dashboard saved: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
