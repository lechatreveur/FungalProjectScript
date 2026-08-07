#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, base64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch, umap, numpy as np, pandas as pd
import plotly.express as px, plotly.utils

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D, EXPERIMENTS
from SingleCellDataAnalysis.FC_Contrastive_umap_Strategy2 import load_m133_with_wt_scalers

MODEL_PATH  = "/Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d/fc_ae_3d_final.pth"
OUTPUT_HTML = "/Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d/fc_ae_3d_manifold_explorer_M133.html"
STRIPS_DIR  = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/vertical_strips/"
VIDEO_GIDS  = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/video_gids.txt"
CYCLE_SCORES= "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/cycle_stage_scores.npy"
DEVICE      = torch.device("cpu")

M133_EXPERIMENT = {
    "M133": "/Volumes/X10 Pro/Movies/2026_04_29_M133/"
}

def get_plotly_js() -> str:
    import plotly
    p = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    return open(p, encoding="utf-8").read()

def load_raw_trajectories(experiments_dict):
    raw = {}
    for exp_name, exp_dir in experiments_dict.items():
        csv_path = os.path.join(exp_dir, "unaligned_pairs_quant", "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
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
    scores = np.full(len(gids), np.nan)
    if not (os.path.exists(VIDEO_GIDS) and os.path.exists(CYCLE_SCORES)):
        return scores
    with open(VIDEO_GIDS) as f:
        v_gids = [l.strip() for l in f if l.strip()]
    v_scores = np.load(CYCLE_SCORES)
    score_map = dict(zip(v_gids, v_scores))
    for i, gid in enumerate(gids):
        if gid in score_map:
            scores[i] = float(score_map[gid])
    return scores

_FILM_FOLDER_MAP = {
    ('Sept17', 'GFP1', 1): 'A14_1TP1_F1', ('Sept17', 'GFP2', 2): 'A14_1TP2_F1',
    ('M92', 'GFP1', 'F0'): 'A14-YES-1t-FBFBF_F0', ('M92', 'GFP1', 'F1'): 'A14-YES-1t-FBFBF_F1', ('M92', 'GFP1', 'F2'): 'A14-YES-1t-FBFBF_F2',
    ('M92', 'GFP2', 'F0'): 'A14-YES-1t-FBFBF-5_F0', ('M92', 'GFP2', 'F1'): 'A14-YES-1t-FBFBF-5_F1', ('M92', 'GFP2', 'F2'): 'A14-YES-1t-FBFBF-5_F2',
    ('M93', 'GFP1', 'F0'): 'A14_FL_1_F0', ('M93', 'GFP1', 'F1'): 'A14_FL_1_F1', ('M93', 'GFP1', 'F2'): 'A14_FL_1_F2',
    ('M93', 'GFP2', 'F0'): 'A14_FL_3_F0', ('M93', 'GFP2', 'F1'): 'A14_FL_3_F1', ('M93', 'GFP2', 'F2'): 'A14_FL_3_F2',
    ('June25_20m', 'GFP1', 'F0'): 'A14_10_20min',
}

_ID_MAP_PATHS = {
    'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/unaligned_pairs_quant/id_map_unaligned.csv',
    'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/unaligned_pairs_quant/id_map_unaligned.csv',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/unaligned_pairs_quant/id_map_unaligned.csv',
}
_SEPT17_STACKED = '/Volumes/X10 Pro/Movies/2025_09_17/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv'

def _resolve_orig_cell(exp_label: str, local_id: int, id_maps: dict, sept17_stacked) -> tuple:
    base = EXPERIMENTS.get(exp_label)
    if not base:
        if exp_label == 'M133': return M133_EXPERIMENT['M133'], None, None # skip area for M133 right now
        return None, None, None
    if exp_label == 'Sept17':
        if sept17_stacked is None: return None, None, None
        row = sept17_stacked[sept17_stacked.cell_id == local_id]
        if row.empty: return None, None, None
        row = row.iloc[0]
        film = _FILM_FOLDER_MAP.get((exp_label, row.source, int(row.tp)))
        return base, film, int(row.orig_gfp_id)
    elif exp_label == 'June25_20m':
        return base, '', local_id
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
    id_maps = {exp: pd.read_csv(p) for exp, p in _ID_MAP_PATHS.items() if os.path.exists(p)}
    sept17_stacked = pd.read_csv(_SEPT17_STACKED) if os.path.exists(_SEPT17_STACKED) else None
    areas = np.full(len(gids), np.nan)
    known_labels = sorted(list(EXPERIMENTS.keys()) + ['M133'], key=lambda x: -len(x))
    
    for i, gid in enumerate(gids):
        exp_label = next((lbl for lbl in known_labels if gid.startswith(lbl + '_')), None)
        if exp_label is None: continue
        local_id = int(gid[len(exp_label) + 1:])
        base, film, orig_id = _resolve_orig_cell(exp_label, local_id, id_maps, sept17_stacked)
        if orig_id is None: continue
        
        if film == '': csv_p = os.path.join(base, f"TrackedCells_A14_10_20min", f"cell_{orig_id}_data.csv")
        elif film is None: continue
        else: csv_p = os.path.join(base, film, f"TrackedCells_{film}", f"cell_{orig_id}_data.csv")
            
        if not os.path.exists(csv_p): continue
        df = pd.read_csv(csv_p)
        if 'cell_area' not in df.columns: continue
        vals = pd.to_numeric(df['cell_area'], errors='coerce').dropna().values
        if len(vals) == 0: continue
        areas[i] = float(vals.max())
    return areas

def remap_for_display(score: float) -> float:
    if np.isnan(score): return np.nan
    if score < 0.4:  return score / 0.4 * 0.80
    if score < 0.7:  return 0.80 + (score - 0.4) / 0.3 * 0.10
    if score < 0.9:  return 0.90 + (score - 0.7) / 0.2 * 0.05
    return           0.95 + (score - 0.9) / 0.1 * 0.05

def main():
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    X_traj_M133, X_feat_M133, gids_M133, labels_M133 = load_m133_with_wt_scalers(M133_EXPERIMENT, s_traj, s_feat)
    
    # Split M133 into D_2 and D_4 based on id_map_unaligned.csv
    m133_map_path = os.path.join(M133_EXPERIMENT["M133"], "unaligned_pairs_quant", "id_map_unaligned.csv")
    if os.path.exists(m133_map_path):
        df_m133_map = pd.read_csv(m133_map_path)
        id_source_map = dict(zip(df_m133_map['new_cell_id'], df_m133_map['source']))
        labels_M133_split = []
        for gid in gids_M133:
            local_id = int(gid.split('_')[1])
            src = id_source_map.get(local_id, 'GFP1')
            if src == 'GFP1':
                labels_M133_split.append('M133 (D_2)')
            else:
                labels_M133_split.append('M133 (D_4)')
        labels_M133 = labels_M133_split
        
    model = MultimodalAutoencoder3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        _, _, latents = model(torch.from_numpy(X_traj).float().to(DEVICE), torch.from_numpy(X_feat).float().to(DEVICE))
        latents = latents.cpu().numpy()
        _, _, latents_M133 = model(torch.from_numpy(X_traj_M133).float().to(DEVICE), torch.from_numpy(X_feat_M133).float().to(DEVICE))
        latents_M133 = latents_M133.cpu().numpy()
        
    r3d = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
    e3d = r3d.fit_transform(latents)
    e3d_M133 = r3d.transform(latents_M133)
    emb3d_all = np.vstack([e3d, e3d_M133])
    
    r2d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    e2d = r2d.fit_transform(latents)
    e2d_M133 = r2d.transform(latents_M133)
    emb2d_all = np.vstack([e2d, e2d_M133])
    
    gids_all = gids + gids_M133
    labels_all = labels + labels_M133
    X_feat_all = np.vstack([X_feat, X_feat_M133])
    X_traj_all = np.vstack([X_traj, X_traj_M133])
    X_feat_all_raw = X_feat_all * (s_feat.std + 1e-8) + s_feat.mean

    
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
    
    # 3D
    df_3d = pd.DataFrame(emb3d_all, columns=["x","y","z"])
    df_3d["gid"] = gids_all
    df_3d["exp"] = labels_all
    df_3d["color"] = area_display
    fig_3d = px.scatter_3d(df_3d, x="x", y="y", z="z", color="color", symbol="exp", hover_data=["gid"],
                           title="Fungal Manifold (3D AE) + M133", color_continuous_scale="Viridis")
    fig_3d.update_layout(margin=dict(l=0,r=0,b=0,t=40), autosize=True, coloraxis_colorbar=dict(title="Cell Area"))
    
    # 2D
    df_2d = pd.DataFrame(emb2d_all, columns=["x","y"])
    df_2d["gid"] = gids_all
    df_2d["exp"] = labels_all
    df_2d["color"] = area_display
    fig_2d = px.scatter(df_2d, x="x", y="y", color="color", symbol="exp", hover_data=["gid"],
                        title="Fungal Manifold (3D AE) + M133", color_continuous_scale="Viridis")
    fig_2d.update_traces(marker=dict(size=5))
    fig_2d.update_layout(margin=dict(l=0,r=0,b=0,t=40), autosize=True, coloraxis_colorbar=dict(title="Cell Area"))
    
    ALL_EXP = {**EXPERIMENTS, **M133_EXPERIMENT}
    raw_traj = load_raw_trajectories(ALL_EXP)
    traj_dict = {}
    for i, gid in enumerate(gids_all):
        raw = raw_traj.get(gid, {})
        traj_dict[gid] = {
            "p1":    raw.get('p1', X_traj_all[i][:,0].tolist()),
            "p2":    raw.get('p2', X_traj_all[i][:,1].tolist()),
            "f":     [round(float(X_feat_all[i,1]),4), round(float(X_feat_all[i,7]),4),
                      round(float(X_feat_all[i,6]),4), round(float(cycle_scores[i]),4) if not np.isnan(cycle_scores[i]) else None],
            "raw_feats": {
                "pol1_mid": round(float(X_feat_all_raw[i, 1]), 4),
                "pol2_mid": round(float(X_feat_all_raw[i, 4]), 4),
                "periodicity": round(float(X_feat_all_raw[i, 7]), 4)
            },
            "strip": get_strip_b64(gid),
            "idx":   i
        }
        
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>AE Explorer 2D/3D + M133</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100vw; height: 100vh; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: row; overflow: hidden; background: #f4f6f8; }
    #main { flex: 2 2 0%; display: flex; flex-direction: column; border-right: 2px solid #d1d5db; background: #fff; min-width: 0; }
    #toolbar { padding: 8px 12px; background: #1e293b; display: flex; align-items: center; gap: 12px; flex-shrink: 0; flex-wrap: wrap; }
    #toolbar label { color: #94a3b8; font-size: 0.8rem; white-space: nowrap; display:flex; align-items:center; gap:4px; }
    select { background: #334155; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: 4px 10px; font-size: 0.85rem; cursor: pointer; }
    input[type=checkbox] { accent-color: #3b82f6; width:16px; height:16px; cursor:pointer; }
    #plot-div { flex: 1 1 0%; width: 100%; height: 100%; min-height: 0; }
    #sidebar { flex: 1 1 0%; min-width: 320px; max-width: 460px; padding: 20px; overflow-y: auto; background: #fff; box-shadow: -2px 0 12px rgba(0,0,0,0.06); display: flex; flex-direction: column; gap: 14px; }
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
      <label>Dimension:
      <select id="dim-select">
        <option value="3D">3D UMAP</option>
        <option value="2D">2D UMAP</option>
      </select>
      </label>
      <label style="margin-left:8px; border-left:1px solid #475569; padding-left:12px;">
      <input type="checkbox" id="mutant-d2-toggle" checked>
      Show M133 (D_2)
      </label>
      <label style="margin-left:8px; border-left:1px solid #475569; padding-left:12px;">
      <input type="checkbox" id="mutant-d4-toggle" checked>
      Show M133 (D_4)
      </label>
      <label style="margin-left:8px; border-left:1px solid #475569; padding-left:12px;">Color by:
      <select id="color-select">
        <option value="Cell Area (max, px&#178;)">Cell Area (max, px²)</option>
        <option value="Cycle Stage Score">Cycle Stage Score</option>
        <option value="Pol1 Mid Intensity">Pol1 Mid Intensity</option>
        <option value="Pol2 Mid Intensity">Pol2 Mid Intensity</option>
        <option value="Periodicity">Periodicity</option>
        <option value="NC Score">NC Score</option>
        <option value="Dynamic modes">Dynamic modes</option>
      </select>
      </label>
      
      <div id="thresholds-container" style="display: none; align-items: center; gap: 8px; margin-left: 8px; border-left: 1px solid #475569; padding-left: 12px;">
        <label style="color: #94a3b8; font-size: 0.8rem;">pol1_mid:</label>
        <input type="number" id="thresh-pol1-mid" value="4.04" step="0.01" style="width: 55px; background: #334155; color: white; border: 1px solid #475569; border-radius: 4px; padding: 2px 4px; font-size: 0.8rem;">
        <label style="color: #94a3b8; font-size: 0.8rem;">pol2_mid:</label>
        <input type="number" id="thresh-pol2-mid" value="2.0" step="0.01" style="width: 55px; background: #334155; color: white; border: 1px solid #475569; border-radius: 4px; padding: 2px 4px; font-size: 0.8rem;">
        <label style="color: #94a3b8; font-size: 0.8rem;">mono_osc:</label>
        <input type="number" id="thresh-mono-osc" value="5.0" step="0.1" style="width: 45px; background: #334155; color: white; border: 1px solid #475569; border-radius: 4px; padding: 2px 4px; font-size: 0.8rem;">
        <label style="color: #94a3b8; font-size: 0.8rem;">bi_osc:</label>
        <input type="number" id="thresh-bi-osc" value="6.5" step="0.1" style="width: 45px; background: #334155; color: white; border: 1px solid #475569; border-radius: 4px; padding: 2px 4px; font-size: 0.8rem;">
      </div>
    </div>
    <div id="plot-div"></div>
  </div>
  <div id="sidebar">
    <div id="content"><div class="placeholder">Click any point in the manifold to view its trajectory and features.</div></div>
  </div>

  <script>PLOTLY_JS_PLACEHOLDER</script>
  <script>
    var plotData3D = PLOT_JSON_3D_PLACEHOLDER;
    var plotData2D = PLOT_JSON_2D_PLACEHOLDER;
    var trajData = TRAJ_JSON_PLACEHOLDER;
    var colorArrays = COLOR_ARRAYS_PLACEHOLDER;

    var is3D = true;
    var plotDiv = document.getElementById('plot-div');
    var dynamicColorscale = [
      [0.0, '#94a3b8'], [0.2, '#94a3b8'],
      [0.2, '#f59e0b'], [0.4, '#f59e0b'],
      [0.4, '#ef4444'], [0.6, '#ef4444'],
      [0.6, '#10b981'], [0.8, '#10b981'],
      [0.8, '#3b82f6'], [1.0, '#3b82f6']
    ];

    function getCategory(pol1_mid, pol2_mid, periodicity) {
        var p1_mid_thresh = parseFloat(document.getElementById('thresh-pol1-mid').value);
        var p2_mid_thresh = parseFloat(document.getElementById('thresh-pol2-mid').value);
        var mono_osc_thresh = parseFloat(document.getElementById('thresh-mono-osc').value);
        var bi_osc_thresh = parseFloat(document.getElementById('thresh-bi-osc').value);
        
        if (pol1_mid < p1_mid_thresh) {
            return 0; // Non-polarized
        } else {
            if (pol2_mid < p2_mid_thresh) {
                return (periodicity > mono_osc_thresh) ? 2 : 1; // Monopolar Osc vs Monopolar
            } else {
                return (periodicity > bi_osc_thresh) ? 4 : 3; // Bipolar Osc vs Bipolar
            }
        }
    }

    function categoryLabel(cat) {
        var labels = ["Non-polarized", "Monopolar", "Monopolar Osc", "Bipolar", "Bipolar Osc"];
        var badgeColors = ["#94a3b8", "#f59e0b", "#ef4444", "#10b981", "#3b82f6"];
        return { txt: labels[cat], col: badgeColors[cat] };
    }

    function renderPlot() {
        var key = document.getElementById('color-select').value;
        var showD2 = document.getElementById('mutant-d2-toggle').checked;
        var showD4 = document.getElementById('mutant-d4-toggle').checked;
        
        var basePlot = is3D ? plotData3D : plotData2D;
        var currentPlot = JSON.parse(JSON.stringify(basePlot));
        
        currentPlot.data = currentPlot.data.filter(function(t) {
            if (t.name === 'M133 (D_2)') return showD2;
            if (t.name === 'M133 (D_4)') return showD4;
            return true;
        });
        
        if (key === "Dynamic modes") {
            document.getElementById('thresholds-container').style.display = 'flex';
            
            for (var i=0; i<currentPlot.data.length; i++) {
                var gids = currentPlot.data[i].customdata.map(d => d[0]);
                var traceColors = gids.map(gid => {
                    var cell = trajData[gid];
                    return getCategory(cell.raw_feats.pol1_mid, cell.raw_feats.pol2_mid, cell.raw_feats.periodicity);
                });
                currentPlot.data[i].marker.color = traceColors;
                currentPlot.data[i].marker.cmin = -0.5;
                currentPlot.data[i].marker.cmax = 4.5;
                currentPlot.data[i].marker.colorscale = dynamicColorscale;
            }
            
            if (currentPlot.layout.coloraxis) {
                currentPlot.layout.coloraxis.colorscale = dynamicColorscale;
                currentPlot.layout.coloraxis.cmin = -0.5;
                currentPlot.layout.coloraxis.cmax = 4.5;
                currentPlot.layout.coloraxis.showscale = true;
                if (currentPlot.layout.coloraxis.colorbar) {
                    currentPlot.layout.coloraxis.colorbar.title = { text: "Dynamic Mode" };
                    currentPlot.layout.coloraxis.colorbar.tickvals = [0, 1, 2, 3, 4];
                    currentPlot.layout.coloraxis.colorbar.ticktext = [
                        "Non-polarized", 
                        "Monopolar", 
                        "Monopolar Osc", 
                        "Bipolar", 
                        "Bipolar Osc"
                    ];
                }
            }
        } else {
            document.getElementById('thresholds-container').style.display = 'none';
            var vals = colorArrays[key];
            var valid = vals.filter(function(v){ return v !== null; });
            var lo = Math.min.apply(null, valid);
            var hi = Math.max.apply(null, valid);
            
            for (var i=0; i<currentPlot.data.length; i++) {
                var gids = currentPlot.data[i].customdata.map(d => d[0]);
                var traceColors = gids.map(gid => vals[trajData[gid].idx]);
                currentPlot.data[i].marker.color = traceColors;
                currentPlot.data[i].marker.cmin = lo;
                currentPlot.data[i].marker.cmax = hi;
                delete currentPlot.data[i].marker.colorscale;
            }
            
            if (currentPlot.layout.coloraxis) {
                currentPlot.layout.coloraxis.colorscale = "Viridis";
                currentPlot.layout.coloraxis.cmin = lo;
                currentPlot.layout.coloraxis.cmax = hi;
                if (currentPlot.layout.coloraxis.colorbar) {
                    currentPlot.layout.coloraxis.colorbar.title = { text: key };
                    delete currentPlot.layout.coloraxis.colorbar.tickvals;
                    delete currentPlot.layout.coloraxis.colorbar.ticktext;
                }
            }
        }
        
        Plotly.newPlot(plotDiv, currentPlot.data, currentPlot.layout, {responsive:true, displayModeBar:true});
        bindClick();
    }
    
    document.getElementById('dim-select').addEventListener('change', function() { is3D = this.value === '3D'; renderPlot(); });
    document.getElementById('mutant-d2-toggle').addEventListener('change', renderPlot);
    document.getElementById('mutant-d4-toggle').addEventListener('change', renderPlot);
    document.getElementById('color-select').addEventListener('change', renderPlot);
    
    document.getElementById('thresh-pol1-mid').addEventListener('input', renderPlot);
    document.getElementById('thresh-pol2-mid').addEventListener('input', renderPlot);
    document.getElementById('thresh-mono-osc').addEventListener('input', renderPlot);
    document.getElementById('thresh-bi-osc').addEventListener('input', renderPlot);

    function cycleLabel(score) {
      if (score === null) return {txt: 'N/A',                  col: '#6b7280'};
      if (score < 0.4)   return {txt: 'Stage 1: Growing',      col: '#16a34a'};
      if (score < 0.7)   return {txt: 'Stage 2: Nuclear Div',  col: '#2563eb'};
      if (score < 0.9)   return {txt: 'Stage 3: Septation',    col: '#d97706'};
                         return {txt: 'Stage 4: Hourglass',    col: '#dc2626'};
    }

    function bindClick() {
        plotDiv.on('plotly_click', function(eventData) {
          var pt = eventData.points[0];
          var gid = pt.customdata[0];
          var exp = pt.fullData.name;
          var cell = trajData[gid];
          if (!cell) return;

          var cyc = cell.f[3];
          var clbl = cycleLabel(cyc);
          var cycStr = cyc !== null ? cyc.toFixed(3) : 'N/A';
          
          var cat = getCategory(cell.raw_feats.pol1_mid, cell.raw_feats.pol2_mid, cell.raw_feats.periodicity);
          var catlbl = categoryLabel(cat);

          document.getElementById('content').innerHTML =
            '<div class="card">' +
              '<h2>Cell: ' + gid + '</h2>' +
              '<div class="stat"><span>Experiment</span><span class="val">' + exp + '</span></div>' +
              '<div class="stat"><span>Dynamic Mode</span><span class="val">' +
                '<span class="cycle-badge" style="background:' + catlbl.col + '; margin-left:0;">' + catlbl.txt + '</span></span></div>' +
              '<div class="stat"><span>Cycle Stage</span><span class="val">' +
                cycStr + '<span class="cycle-badge" style="background:' + clbl.col + '">' + clbl.txt + '</span></span></div>' +
              '<div class="stat"><span>Pol1 Mid (raw)</span><span class="val">' + cell.raw_feats.pol1_mid.toFixed(4) + '</span></div>' +
              '<div class="stat"><span>Pol2 Mid (raw)</span><span class="val">' + cell.raw_feats.pol2_mid.toFixed(4) + '</span></div>' +
              '<div class="stat"><span>Periodicity (raw)</span><span class="val">' + cell.raw_feats.periodicity.toFixed(4) + '</span></div>' +
              '<div class="stat"><span>NC Score (scaled)</span><span class="val">' + cell.f[2].toFixed(4) + '</span></div>' +
            '</div>' +
            '<div class="card" style="position: sticky; top: 0; z-index: 10; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">' +
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
            {margin:{l:40,r:10,b:36,t:10}, xaxis:{title:'Frame',showgrid:false}, yaxis:{title:'Intensity (corr.)'}, showlegend:false, paper_bgcolor:'transparent', plot_bgcolor:'transparent'},
            {displayModeBar:false, responsive:true});
        });
    }

    renderPlot();
  </script>
</body>
</html>"""
    html = html.replace("PLOTLY_JS_PLACEHOLDER", get_plotly_js())
    html = html.replace("PLOT_JSON_3D_PLACEHOLDER", json.dumps(fig_3d, cls=plotly.utils.PlotlyJSONEncoder))
    html = html.replace("PLOT_JSON_2D_PLACEHOLDER", json.dumps(fig_2d, cls=plotly.utils.PlotlyJSONEncoder))
    html = html.replace("TRAJ_JSON_PLACEHOLDER", json.dumps(traj_dict))
    html = html.replace("COLOR_ARRAYS_PLACEHOLDER", json.dumps(color_arrays))

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard saved: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
