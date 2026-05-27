#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_contrastive_umap.py

1. Loads the trained contrastive encoder (Strategy C).
2. Generates 128D latent embeddings for all 419 cells.
3. Runs UMAP to reduce to 2D/3D.
4. Generates an interactive Plotly HTML colored by cycle stage.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import umap
import pandas as pd
import plotly.express as px

# --- Paths ---
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
DATA_PATH = os.path.join(BASE_DIR, "video_cache_32x112_padded.npy")
MODEL_PATH = os.path.join(BASE_DIR, "video_contrastive_cycle_final.pth")
SCORES_PATH = os.path.join(BASE_DIR, "cycle_stage_scores.npy")
GIDS_PATH = os.path.join(BASE_DIR, "video_gids.txt")
TRANSITION_PATH = os.path.join(BASE_DIR, "cycle_transition_pairs.npy")
FEATURES_PATH = os.path.join(BASE_DIR, "cycle_stage_features.npy")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Import Model Architecture (must match training script) ---
from Video_AE_train_contrastive import PolarityEncoder

def main():
    # 1. Load Data
    print("Loading data...")
    videos = np.load(DATA_PATH) # (N, 101, 1, 32, 112)
    scores = np.load(SCORES_PATH)
    features = np.load(FEATURES_PATH) # [length, area, nu_dis, septum, pattern]
    with open(GIDS_PATH) as f:
        gids = [l.strip() for l in f if l.strip()]
    transitions = np.load(TRANSITION_PATH)

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = PolarityEncoder(latent_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. Generate Latents
    print("Generating latents...")
    latents = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(videos), batch_size):
            batch = torch.from_numpy(videos[i:i+batch_size]).float().to(DEVICE)
            z = model(batch)
            latents.append(z.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    np.save(os.path.join(BASE_DIR, "contrastive_latents.npy"), latents)
    print(f"Latents saved: {latents.shape}")

    # 4. UMAP Reduction
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)

    # 5. Prepare DataFrame for Plotly
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'z': embedding[:, 2],
        'gid': gids,
        'cycle_score': scores,
        'cell_length': features[:, 0],
        'nu_dis': features[:, 2],
        'pattern_score': features[:, 4]
    })

    # 6. Create Interactive Plot
    print("Creating interactive plot...")
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='cycle_score',
        hover_data=['gid', 'cell_length', 'nu_dis', 'pattern_score'],
        title="Cycle-Aware Contrastive Manifold (Strategy C)",
        color_continuous_scale='Viridis',
        opacity=0.8
    )

    # Add lines for transition pairs (The "Stitch")
    # For clarity, we'll only draw some of them
    for i in range(0, len(transitions), 5): 
        idx1, idx2 = transitions[i]
        fig.add_trace(px.line_3d(
            df.iloc[[idx1, idx2]], x='x', y='y', z='z'
        ).data[0])
        # Make transition lines grey and subtle
        fig.data[-1].line.color = 'rgba(100,100,100,0.2)'
        fig.data[-1].line.width = 1

    out_html = os.path.join(BASE_DIR, "contrastive_cycle_umap.html")
    fig.write_html(out_html)
    print(f"✅ Interactive UMAP saved to {out_html}")

if __name__ == "__main__":
    main()
