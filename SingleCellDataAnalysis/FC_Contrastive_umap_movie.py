#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FC_Contrastive_umap_movie.py
Renders a rotating MP4 movie of the 3D UMAP coloured by Cell Area.
Camera follows 2 full azimuth rotations with a sinusoidal elevation
oscillation (15°→55°→15°) to reveal all angles of the manifold.
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_Contrastive_train import FCTrajectoryEncoder, EXPERIMENTS, LATENT_DIM

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_contrastive_final.pth"
OUTPUT_MP4  = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_umap_rotation.mp4"
DEVICE      = torch.device("cpu")

# ── ID resolution (copied from FC_Contrastive_umap.py) ────────────────────────
_FILM_FOLDER_MAP = {
    ('Sept17', 'GFP1', 1): 'A14_1TP1_F1',
    ('Sept17', 'GFP2', 2): 'A14_1TP2_F1',
    ('M92', 'GFP1', 'F0'): 'A14-YES-1t-FBFBF_F0',
    ('M92', 'GFP1', 'F1'): 'A14-YES-1t-FBFBF_F1',
    ('M92', 'GFP1', 'F2'): 'A14-YES-1t-FBFBF_F2',
    ('M92', 'GFP2', 'F0'): 'A14-YES-1t-FBFBF-5_F0',
    ('M92', 'GFP2', 'F1'): 'A14-YES-1t-FBFBF-5_F1',
    ('M92', 'GFP2', 'F2'): 'A14-YES-1t-FBFBF-5_F2',
    ('M93', 'GFP1', 'F0'): 'A14_FL_1_F0',  ('M93', 'GFP1', 'F1'): 'A14_FL_1_F1',
    ('M93', 'GFP1', 'F2'): 'A14_FL_1_F2',  ('M93', 'GFP2', 'F0'): 'A14_FL_3_F0',
    ('M93', 'GFP2', 'F1'): 'A14_FL_3_F1',  ('M93', 'GFP2', 'F2'): 'A14_FL_3_F2',
    ('June25_20m', 'GFP1', 'F0'): 'A14_10_20min',
}
_ID_MAP_PATHS = {
    'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/unaligned_pairs_quant/id_map_unaligned.csv',
    'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/unaligned_pairs_quant/id_map_unaligned.csv',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/unaligned_pairs_quant/id_map_unaligned.csv',
}
_SEPT17_STACKED = '/Volumes/X10 Pro/Movies/2025_09_17/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv'
JUNE25_FILM = 'A14_10_20min'

def _resolve_orig_cell(exp_label, local_id, id_maps, sept17_stacked):
    base = EXPERIMENTS[exp_label]
    if exp_label == 'Sept17':
        if sept17_stacked is None: return None, None, None
        row = sept17_stacked[sept17_stacked.cell_id == local_id]
        if row.empty: return None, None, None
        r = row.iloc[0]
        film = _FILM_FOLDER_MAP.get((exp_label, r.source, int(r.tp)))
        return base, film, int(r.orig_gfp_id)
    elif exp_label == 'June25_20m':
        return base, '', local_id
    else:
        df = id_maps.get(exp_label)
        if df is None: return None, None, None
        row = df[df.new_cell_id == local_id]
        if row.empty: return None, None, None
        r = row.iloc[0]
        orig_id = int(r.orig_str_id.split(':')[1])
        film = _FILM_FOLDER_MAP.get((exp_label, r.source, r.field))
        return base, film, orig_id

def load_cell_areas(gids):
    id_maps = {exp: pd.read_csv(p) for exp, p in _ID_MAP_PATHS.items() if os.path.exists(p)}
    s17 = pd.read_csv(_SEPT17_STACKED) if os.path.exists(_SEPT17_STACKED) else None
    areas = np.full(len(gids), np.nan)
    known = sorted(EXPERIMENTS.keys(), key=lambda x: -len(x))
    for i, gid in enumerate(gids):
        exp = next((lbl for lbl in known if gid.startswith(lbl + '_')), None)
        if exp is None: continue
        lid = int(gid[len(exp) + 1:])
        base, film, orig_id = _resolve_orig_cell(exp, lid, id_maps, s17)
        if orig_id is None: continue
        csv_p = (os.path.join(base, f"TrackedCells_{JUNE25_FILM}", f"cell_{orig_id}_data.csv")
                 if film == '' else
                 (None if film is None else
                  os.path.join(base, film, f"TrackedCells_{film}", f"cell_{orig_id}_data.csv")))
        if csv_p is None or not os.path.exists(csv_p): continue
        df = pd.read_csv(csv_p)
        if 'cell_area' not in df.columns: continue
        vals = pd.to_numeric(df['cell_area'], errors='coerce').dropna().values
        if len(vals): areas[i] = float(vals.max())
    return areas

# ── Experiment colour map ──────────────────────────────────────────────────────
EXP_COLORS = {
    'M92': '#60a5fa', 'M93': '#34d399', 'Sept17': '#f472b6', 'June25_20m': '#fb923c'
}

# ── Animation settings ─────────────────────────────────────────────────────────
FPS        = 30
N_FRAMES   = 600   # 20 s @ 30fps — 2 full rotations + elevation oscillation
POINT_SIZE = 18

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("📥 Loading data...")
    X_traj, X_feat, gids, labels, _, _ = load_feature_constrained_data(EXPERIMENTS)

    print("🧠 Extracting latents...")
    model = FCTrajectoryEncoder(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        latents = model(torch.from_numpy(X_traj).float(),
                        torch.from_numpy(X_feat).float()).numpy()

    print("🗺️ Running 3D UMAP...")
    embedding = umap.UMAP(n_components=3, random_state=42, n_jobs=1).fit_transform(latents)
    x, y, z = embedding[:, 0], embedding[:, 1], embedding[:, 2]

    print("📐 Loading cell areas...")
    areas = load_cell_areas(gids)

    # ── Normalise areas for colour ─────────────────────────────────────────────
    valid = areas[~np.isnan(areas)]
    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    colors_area = [cmap(norm(a)) if not np.isnan(a) else (0.55, 0.55, 0.55, 0.4)
                   for a in areas]

    # Per-experiment marker shapes
    MARKERS = {'M92': 'o', 'M93': 's', 'Sept17': '^', 'June25_20m': 'D'}

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9), facecolor='#0d1117')
    ax  = fig.add_subplot(111, projection='3d', facecolor='#0d1117')

    ax.tick_params(colors='#8b949e', labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#21262d')
    ax.grid(True, color='#21262d', linewidth=0.4)
    ax.set_xlabel('UMAP 1', color='#8b949e', fontsize=9, labelpad=6)
    ax.set_ylabel('UMAP 2', color='#8b949e', fontsize=9, labelpad=6)
    ax.set_zlabel('UMAP 3', color='#8b949e', fontsize=9, labelpad=6)

    # Draw scatter per experiment (so legend works)
    scatter_handles = []
    exp_list = sorted(set(labels))
    for exp in exp_list:
        mask = np.array(labels) == exp
        idx  = np.where(mask)[0]
        ec   = [colors_area[i] for i in idx]
        sc   = ax.scatter(x[idx], y[idx], z[idx],
                          c=ec, s=POINT_SIZE,
                          marker=MARKERS.get(exp, 'o'),
                          edgecolors='none', alpha=0.85,
                          depthshade=True, label=exp)
        scatter_handles.append(sc)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08, aspect=20)
    cb.set_label('Max Cell Area (px²)', color='#c9d1d9', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='#8b949e', labelsize=7)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#c9d1d9')

    # Legend
    leg = ax.legend(handles=scatter_handles, loc='upper left',
                    fontsize=8, framealpha=0.3,
                    facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    title = ax.set_title('Fungal Multimodal Manifold — Rotating 3D UMAP\n'
                         'Color: max cell area  |  Marker: experiment',
                         color='#c9d1d9', fontsize=11, pad=14)

    # ── Camera path ───────────────────────────────────────────────────────────
    # t ∈ [0, 1] over all frames
    t = np.linspace(0, 1, N_FRAMES)

    # Azimuth: 0 → 720 (two full rotations)
    azim_path = 30 + 720 * t

    # Elevation: oscillates 20°→55°→20°→55°→20° (twice per movie)
    elev_path = 20 + 35 * (np.sin(2 * np.pi * 2 * t - np.pi / 2) + 1) / 2

    # ── Animation function ────────────────────────────────────────────────────
    def update(frame):
        ax.view_init(elev=elev_path[frame], azim=azim_path[frame])
        return []

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"🎬 Rendering {N_FRAMES} frames @ {FPS} fps → {OUTPUT_MP4}")
    writer = FFMpegWriter(fps=FPS, bitrate=4000,
                          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    ani.save(OUTPUT_MP4, writer=writer, dpi=150,
             savefig_kwargs={'facecolor': '#0d1117'})
    print(f"✅ Movie saved: {OUTPUT_MP4}")
    plt.close(fig)


if __name__ == "__main__":
    main()
