#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_contrastive_interactive_umap.py

Creates a 2D UMAP of the Contrastive Latent Space (Strategy C)
with a side-by-side comparison tool for vertical time-lapse strips.
Points are colored by biological cycle score.
"""

import os
import json
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")
LATENTS_PATH = os.path.join(BASE_DIR, "contrastive_latents_sequential.npy")
SCORES_PATH = os.path.join(BASE_DIR, "cycle_stage_scores.npy")
FEATURES_PATH = os.path.join(BASE_DIR, "cycle_stage_features.npy")

STRIPS_DIR = "vertical_strips" # Relative path for HTML
OUTPUT_HTML = os.path.join(BASE_DIR, "contrastive_sequential_interactive_umap.html")
MODEL_PATH = os.path.join(BASE_DIR, "video_contrastive_sequential_final.pth")

def main():
    # 1. Load data
    print("Loading data...")
    videos = np.load(os.path.join(BASE_DIR, "video_cache_32x112_padded.npy"), mmap_mode='r')
    scores = np.load(SCORES_PATH)
    features = np.load(FEATURES_PATH)
    
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f if l.strip()]
        
    print(f"Loaded {len(gids)} cells.")

    # 1.5 Generate Latents using the new model
    print(f"Loading model from {MODEL_PATH}...")
    from Video_AE_train_contrastive import PolarityEncoder
    import torch
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PolarityEncoder(latent_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Generating latents...")
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

    # 2. Compute 2D UMAP for easier navigation
    print("Computing 2D UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    # 3. Prepare data for JSON
    points_data = []
    for i, gid in enumerate(gids):
        points_data.append({
            'x': float(embedding[i, 0]),
            'y': float(embedding[i, 1]),
            'id': gid,
            'score': float(scores[i]),
            'len': float(features[i, 0]),
            'nu': float(features[i, 2]),
            'pat': float(features[i, 4])
        })
        
    points_json = json.dumps(points_data)
    
    # 4. Generate HTML with Side-by-Side Tool
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Contrastive Cycle Manifold - Comparison Tool</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            height: 100vh;
            box-sizing: border-box;
            background-color: #0f111a;
            color: #e0e0e0;
        }}
        #plot-container {{
            flex: 1;
            padding-right: 20px;
            display: flex;
            flex-direction: column;
        }}
        #plot {{
            flex: 1;
            background-color: #1a1c25;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        #comparison-panel {{
            width: 700px;
            display: flex;
            flex-direction: row;
            gap: 15px;
            overflow-y: auto;
            background-color: #1a1c25;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .image-col {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
            background: #12141d;
        }}
        .image-title {{
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 10px;
            word-break: break-all;
            text-align: center;
            height: 50px;
            color: #bb86fc;
        }}
        .strip-img {{
            width: 100%;
            image-rendering: pixelated;
            border-radius: 4px;
            border: 1px solid #444;
        }}
        .info-box {{
            margin-bottom: 15px;
            padding: 10px;
            background: #252833;
            border-radius: 8px;
            font-size: 14px;
        }}
        .highlight {{ color: #03dac6; font-weight: bold; }}
        h2 {{ margin-top: 0; color: #ffffff; }}
    </style>
</head>
<body>

    <div id="plot-container">
        <h2>Sequential Biological Cycle Manifold</h2>
        <div class="info-box">
            <b>Color</b>: Sequential Cycle Stage (Stage 1 to 4)<br>
            <b>Stage 1</b>: Growing | <b>Stage 2</b>: Nuclear Div | <b>Stage 3</b>: Septation | <b>Stage 4</b>: Division<br>
            <b>Action</b>: Click a point to view its timelapse. Click another to <span class="highlight">Compare Side-by-Side</span>.
        </div>
        <div id="plot"></div>
    </div>

    <div id="comparison-panel">
        <div class="image-col">
            <div id="title1" class="image-title">Slot A: Select a cell</div>
            <img id="img1" class="strip-img" style="display:none;" />
        </div>
        <div class="image-col">
            <div id="title2" class="image-title">Slot B: Select a cell</div>
            <img id="img2" class="strip-img" style="display:none;" />
        </div>
    </div>

    <script>
        const data = {points_json};
        
        const trace = {{
            x: data.map(d => d.x),
            y: data.map(d => d.y),
            text: data.map(d => `ID: ${{d.id}}<br>Score: ${{d.score.toFixed(3)}}<br>Len: ${{d.len.toFixed(1)}}<br>NuDis: ${{d.nu.toFixed(1)}}`),
            mode: 'markers',
            marker: {{
                size: 8,
                color: data.map(d => d.score),
                colorscale: 'Viridis',
                reversescale: true,
                showscale: true,
                colorbar: {{ title: 'Cycle Score', thickness: 15 }},
                opacity: 0.8,
                line: {{ color: '#000', width: 0.5 }}
            }},
            hoverinfo: 'text'
        }};
        
        const layout = {{
            hovermode: 'closest',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#e0e0e0' }},
            xaxis: {{ showgrid: false, zeroline: false, visible: false }},
            yaxis: {{ showgrid: false, zeroline: false, visible: false }},
            margin: {{ t: 10, b: 10, l: 10, r: 10 }}
        }};
        
        Plotly.newPlot('plot', [trace], layout);
        
        let currentSlot = 1;
        let selectedIndices = [null, null];

        document.getElementById('plot').on('plotly_click', function(eventData) {{
            const pointIndex = eventData.points[0].pointIndex;
            const clicked = data[pointIndex];
            const imgSrc = '{STRIPS_DIR}/' + clicked.id + '.png';
            
            // Update UI
            const imgEl = document.getElementById('img' + currentSlot);
            const titleEl = document.getElementById('title' + currentSlot);
            
            imgEl.src = imgSrc;
            imgEl.style.display = 'block';
            titleEl.innerHTML = `<b>${{clicked.id}}</b><br>Score: ${{clicked.score.toFixed(3)}}`;
            
            // Track selection
            selectedIndices[currentSlot - 1] = pointIndex;
            
            // Update highlights
            const colors = data.map(d => d.score);
            const sizes = new Array(data.length).fill(8);
            const symbols = new Array(data.length).fill('circle');
            
            selectedIndices.forEach((idx, i) => {{
                if (idx !== null) {{
                    sizes[idx] = 16;
                    symbols[idx] = i === 0 ? 'diamond' : 'square';
                }}
            }});
            
            Plotly.restyle('plot', {{
                'marker.size': [sizes],
                'marker.symbol': [symbols]
            }});
            
            // Toggle slot
            currentSlot = currentSlot === 1 ? 2 : 1;
        }});
    </script>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)
        
    print(f"✅ Success! Interactive Comparison Tool saved to:")
    print(f"   {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
