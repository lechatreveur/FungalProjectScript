#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_contrastive_invariant_umap.py

Creates a 2D UMAP of the Size-Invariant Latent Space (Strategy D).
Uses the model trained with random scaling to ignore absolute cell length.
"""

import os
import json
import numpy as np
import pandas as pd
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache_32x112_padded.npy")
LATENTS_PATH = os.path.join(BASE_DIR, "contrastive_latents_invariant.npy")
SCORES_PATH = os.path.join(BASE_DIR, "cycle_stage_scores.npy")
FEATURES_PATH = os.path.join(BASE_DIR, "cycle_stage_features.npy")
MODEL_PATH = os.path.join(BASE_DIR, "video_contrastive_invariant_final.pth")

STRIPS_DIR = "vertical_strips" 
OUTPUT_HTML = os.path.join(BASE_DIR, "contrastive_invariant_interactive_umap.html")

# Define model architecture (must match training script)
class InvariantEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), stride=(2,1,2), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten()
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # (B, C, T, H, W)
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

def main():
    # 1. Load data
    print("Loading data...")
    videos = np.load(CACHE_VIDEOS, mmap_mode='r')
    scores = np.load(SCORES_PATH)
    features = np.load(FEATURES_PATH)
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f if l.strip()]
    
    # 2. Generate Latents
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    model = InvariantEncoder(latent_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Generating invariant latents...")
    latents = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(videos), batch_size):
            batch = torch.from_numpy(videos[i:i+batch_size]).float().to(DEVICE)
            z = model(batch)
            latents.append(z.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    np.save(LATENTS_PATH, latents)
    print(f"Latents saved: {latents.shape}")

    # 3. Compute 2D UMAP
    print("Computing 2D UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    # 4. Prepare JSON
    points_data = []
    for i, gid in enumerate(gids):
        points_data.append({
            'x': float(embedding[i, 0]),
            'y': float(embedding[i, 1]),
            'id': gid,
            'score': float(scores[i]),
            'len': float(features[i, 0])
        })
    points_json = json.dumps(points_data)
    
    # 5. Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Size-Invariant Polarity Manifold</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; display: flex; height: 100vh; background: #0f111a; color: #e0e0e0; }}
        #plot-container {{ flex: 1; display: flex; flex-direction: column; }}
        #plot {{ flex: 1; background: #1a1c25; border-radius: 12px; }}
        #comparison-panel {{ width: 600px; display: flex; gap: 10px; padding: 20px; background: #1a1c25; border-radius: 12px; margin-left: 20px; overflow-y: auto; }}
        .image-col {{ flex: 1; display: flex; flex-direction: column; align-items: center; background: #12141d; padding: 10px; border-radius: 8px; border: 1px solid #333; }}
        .strip-img {{ width: 100%; image-rendering: pixelated; border: 1px solid #444; }}
        .title {{ font-size: 12px; margin-bottom: 8px; color: #bb86fc; text-align: center; height: 40px; }}
        .info {{ background: #252833; padding: 10px; border-radius: 8px; margin-bottom: 10px; font-size: 14px; }}
    </style>
</head>
<body>
    <div id="plot-container">
        <h2>Size-Invariant Polarity Manifold (Strategy D)</h2>
        <div class="info">
            This manifold ignores absolute cell size. Points close together share <b>similar intensity patterns</b> regardless of length.<br>
            Color: Biological Cycle Score | Size: Normalized by model
        </div>
        <div id="plot"></div>
    </div>
    <div id="comparison-panel">
        <div class="image-col">
            <div id="title1" class="title">Slot A</div>
            <img id="img1" class="strip-img" style="display:none;" />
        </div>
        <div class="image-col">
            <div id="title2" class="title">Slot B</div>
            <img id="img2" class="strip-img" style="display:none;" />
        </div>
    </div>
    <script>
        const data = {points_json};
        const trace = {{
            x: data.map(d => d.x), y: data.map(d => d.y),
            text: data.map(d => `ID: ${{d.id}}<br>Len: ${{d.len.toFixed(1)}}`),
            mode: 'markers',
            marker: {{ size: 8, color: data.map(d => d.score), colorscale: 'Viridis', showscale: true }}
        }};
        Plotly.newPlot('plot', [trace], {{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {{color: '#fff'}}, margin: {{t:10, b:10, l:10, r:10}} }});
        
        let slot = 1;
        document.getElementById('plot').on('plotly_click', function(e) {{
            const d = data[e.points[0].pointIndex];
            const img = document.getElementById('img' + slot);
            const title = document.getElementById('title' + slot);
            img.src = 'vertical_strips/' + d.id + '.png';
            img.style.display = 'block';
            title.innerHTML = `<b>${{d.id}}</b><br>Len: ${{d.len.toFixed(1)}}`;
            slot = slot === 1 ? 2 : 1;
        }});
    </script>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)
    print(f"✅ Success! Invariant Tool saved to: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
