import os, sys, json, base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch, umap, numpy as np, pandas as pd
import plotly.express as px, plotly.utils

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.AE_data_loader import load_and_preprocess_trajectories
from SingleCellDataAnalysis.PCA_utils import load_experiment_features
from SingleCellDataAnalysis.FC_Contrastive_train import FCTrajectoryEncoder, EXPERIMENTS, LATENT_DIM
from SingleCellDataAnalysis.FC_Contrastive_umap import (
    MODEL_PATH, STRIPS_DIR, DEVICE, get_plotly_js, load_raw_trajectories, 
    load_cycle_scores, load_cell_areas, remap_for_display, get_strip_b64
)

OUTPUT_HTML = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_manifold_explorer_Strategy2.html"

M133_EXPERIMENT = {
    "M133": "/Volumes/X10 Pro/Movies/2026_04_29_M133/"
}

def load_m133_with_wt_scalers(m133_dict, scaler_traj, scaler_feat):
    # 1. Load Trajectories manually without scaling
    df_list = []
    for exp_name, exp_dir in m133_dict.items():
        csv_path = os.path.join(exp_dir, "unaligned_pairs_quant", "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['experiment'] = exp_name
            df['global_cell_id'] = exp_name + "_" + df['cell_id'].astype(str)
            df_list.append(df)
            
    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined.sort_values(by=['global_cell_id', 'time_point'], inplace=True)
    
    gids = []
    labels = []
    trajectories = []
    for gid, grp in df_combined.groupby('global_cell_id'):
        if len(grp) != 101: continue
        pol1 = grp['pol1_int_corr'].values
        pol2 = grp['pol2_int_corr'].values
        if np.sum(pol2) > np.sum(pol1):
            pol1, pol2 = pol2, pol1
        trajectories.append(np.column_stack((pol1, pol2)))
        gids.append(gid)
        labels.append(grp['experiment'].iloc[0])
        
    X_traj_raw = np.array(trajectories)
    
    # 2. Load Features
    df_list_feat = []
    for exp_name, exp_dir in m133_dict.items():
        df_exp = load_experiment_features(exp_dir)
        df_exp['global_cell_id'] = exp_name + "_" + df_exp.index.astype(str)
        df_exp.set_index('global_cell_id', inplace=True)
        df_list_feat.append(df_exp)
    df_features_all = pd.concat(df_list_feat)
    feature_cols = ['pol1_a', 'pol1_mid', 'pol1_v', 'pol2_a', 'pol2_mid', 'pol2_v', 'NC_score', 'Periodicity', 'a1a2', 'd', 'dd']
    
    aligned_traj = []
    aligned_feat = []
    aligned_gids = []
    aligned_labels = []
    
    for i, gid in enumerate(gids):
        if gid in df_features_all.index:
            feat_row = df_features_all.loc[gid, feature_cols].values
            if pd.isna(feat_row).any():
                continue
            aligned_traj.append(X_traj_raw[i])
            aligned_feat.append(feat_row)
            aligned_gids.append(gid)
            aligned_labels.append(labels[i])
            
    X_traj_arr = np.array(aligned_traj)
    X_feat_raw = np.array(aligned_feat)
    
    # 3. Apply WT feature and traj scaler
    N, T, C = X_traj_arr.shape
    X_traj_scaled = scaler_traj.transform(X_traj_arr.reshape(-1, C)).reshape(N, T, C)
    X_feat_scaled = scaler_feat.transform(X_feat_raw)
    
    X_feat_scaled = np.nan_to_num(X_feat_scaled, posinf=0.0, neginf=0.0)
    X_traj_scaled = np.nan_to_num(X_traj_scaled, posinf=0.0, neginf=0.0)
    
    return X_traj_scaled, X_feat_scaled, aligned_gids, aligned_labels

def main():
    print("📥 Loading WT data...")
    X_traj_WT, X_feat_WT, gids_WT, labels_WT, scaler_traj, scaler_feat = load_feature_constrained_data(EXPERIMENTS)
    
    # Fill any NaNs in WT just in case
    X_traj_WT = np.nan_to_num(X_traj_WT)
    X_feat_WT = np.nan_to_num(X_feat_WT)

    print("📥 Loading M133 data with WT scalers...")
    X_traj_M133, X_feat_M133, gids_M133, labels_M133 = load_m133_with_wt_scalers(M133_EXPERIMENT, scaler_traj, scaler_feat)

    print("🧠 Loading previous WT model & extracting latents...")
    model = FCTrajectoryEncoder(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad():
        latents_WT = model(torch.from_numpy(X_traj_WT).float(),
                           torch.from_numpy(X_feat_WT).float()).numpy()
        latents_M133 = model(torch.from_numpy(X_traj_M133).float(),
                             torch.from_numpy(X_feat_M133).float()).numpy()

    print("🗺️ Fitting UMAP on WT latents...")
    reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
    embedding_WT = reducer.fit_transform(latents_WT)

    print("🗺️ Projecting M133 latents onto WT UMAP...")
    embedding_M133 = reducer.transform(latents_M133)
    
    # Combine everything for plotting
    embedding_all = np.vstack([embedding_WT, embedding_M133])
    gids_all = gids_WT + gids_M133
    labels_all = labels_WT + labels_M133
    X_traj_all = np.vstack([X_traj_WT, X_traj_M133])
    X_feat_all = np.vstack([X_feat_WT, X_feat_M133])

    # Color arrays
    cycle_scores = load_cycle_scores(gids_all)
    cycle_display = [None if np.isnan(v) else round(remap_for_display(v), 4) for v in cycle_scores]
    cell_areas = load_cell_areas(gids_all)
    area_display = [None if np.isnan(v) else round(float(v), 2) for v in cell_areas]
    
    color_arrays = {
        "Cell Area (max, px²)":  area_display,
        "Cycle Stage Score":      cycle_display,
        "Pol1 Mid Intensity":    [round(float(X_feat_all[i,1]),4) for i in range(len(gids_all))],
        "Pol2 Mid Intensity":    [round(float(X_feat_all[i,4]),4) for i in range(len(gids_all))],
        "Periodicity":           [round(float(X_feat_all[i,7]),4) for i in range(len(gids_all))],
        "NC Score":              [round(float(X_feat_all[i,6]),4) for i in range(len(gids_all))],
    }
    color_arrays_json = json.dumps(color_arrays)

    df = pd.DataFrame(embedding_all, columns=["x","y","z"])
    df["gid"] = gids_all
    df["exp"] = labels_all
    df["color"] = area_display

    fig = px.scatter_3d(df, x="x", y="y", z="z",
        color="color", symbol="exp", hover_data=["gid"],
        title="Fungal Manifold Explorer - WT vs M133 (Projected)",
        color_continuous_scale="Viridis")
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=40), autosize=True,
                      coloraxis_colorbar=dict(title="Cell Area"))
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ── Trajectory & strip data ─────────────────────────────────────────────
    print("📂 Loading raw trajectories...")
    ALL_EXP = {**EXPERIMENTS, **M133_EXPERIMENT}
    raw_traj = load_raw_trajectories(ALL_EXP)
    print("🎥 Loading cell strips...")
    traj_dict = {}
    for i, gid in enumerate(gids_all):
        raw = raw_traj.get(gid, {})
        traj_dict[gid] = {
            "p1":    raw.get('p1', X_traj_all[i][:,0].tolist()),
            "p2":    raw.get('p2', X_traj_all[i][:,1].tolist()),
            "f":     [round(float(X_feat_all[i,1]),4), round(float(X_feat_all[i,7]),4),
                      round(float(X_feat_all[i,6]),4), round(float(cycle_scores[i]),4)
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
  <title>Fungal Manifold Explorer 3D - WT vs M133</title>
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
    print(f"✅ Strategy 2 Dashboard saved: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
