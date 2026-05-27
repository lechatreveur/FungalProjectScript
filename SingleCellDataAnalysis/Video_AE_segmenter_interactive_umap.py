import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from PIL import Image

# Paths
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS = os.path.join(BASE_DIR, "video_cache_32x112_padded.npy")
CACHE_GIDS = os.path.join(BASE_DIR, "video_gids.txt")
OUTPUT_LATENTS = os.path.join(BASE_DIR, "segmenter_latents.npy")

STRIPS_DIR = os.path.join(BASE_DIR, "vertical_strips")
OUTPUT_HTML = os.path.join(BASE_DIR, "interactive_segmenter_umap.html")

def generate_strips_and_html():
    os.makedirs(STRIPS_DIR, exist_ok=True)
    
    # 1. Load data
    videos = np.load(CACHE_VIDEOS, mmap_mode='r')
    latents = np.load(OUTPUT_LATENTS) # (419, 16)
    
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]
        
    print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    # 2. Generate vertical strips for all 419 cells
    print("Generating vertical strips...")
    for i, gid in enumerate(gids):
        # We will take every 2nd frame to keep the strip reasonable (51 frames)
        # or all 101 frames? "full strip" -> let's do all 101.
        # But wait, 101 frames * 48 = 4848 pixels height. Perfectly fine for scrolling.
        
        # Get video (101, 1, 48, 96)
        v = videos[i, :, 0, :, :] # (101, 48, 96)
        
        # Create a vertical stack
        strip = np.vstack(v) # (101*64, 224)
        
        # Normalize to 0-255
        # The video is normalized such that 99th percentile is 1.0, but let's just clip to 0-1.5 and scale
        strip_norm = np.clip(strip / 1.5, 0, 1) * 255
        strip_uint8 = strip_norm.astype(np.uint8)
        
        # Apply a colormap (viridis) to make it look nicer, or just grayscale?
        # Videos are usually grayscale, but applying a colormap can help visibility.
        # Let's use matplotlib's viridis to convert to RGB
        cm = plt.get_cmap('viridis')
        strip_rgb = (cm(strip_uint8 / 255.0)[:, :, :3] * 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(strip_rgb)
        img.save(os.path.join(STRIPS_DIR, f"{gid}.png"))
        
        if (i+1) % 50 == 0:
            print(f"  Generated {i+1}/{len(gids)} strips...")
            
    # 3. Create JSON data for Plotly
    points_data = []
    for i, gid in enumerate(gids):
        points_data.append({
            'x': float(embedding[i, 0]),
            'y': float(embedding[i, 1]),
            'id': gid
        })
        
    points_json = json.dumps(points_data)
    
    # 4. Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Segmenter UMAP</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            height: 100vh;
            box-sizing: border-box;
            background-color: #121212;
            color: #ffffff;
        }}
        #plot-container {{
            flex: 1;
            padding-right: 20px;
            display: flex;
            flex-direction: column;
        }}
        #plot {{
            flex: 1;
            background-color: #1e1e1e;
            border-radius: 8px;
        }}
        #images-container {{
            width: 800px;
            display: flex;
            flex-direction: row;
            gap: 20px;
            overflow-y: auto;
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 8px;
        }}
        .image-col {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .image-title {{
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
            word-break: break-all;
            text-align: center;
            height: 40px;
        }}
        .strip-img {{
            width: 100%;
            image-rendering: pixelated;
            border: 1px solid #444;
        }}
        .instructions {{
            margin-bottom: 15px;
            font-size: 16px;
            color: #aaa;
        }}
    </style>
</head>
<body>

    <div id="plot-container">
        <h2>Segmenter Latent Space UMAP</h2>
        <div class="instructions">
            Click on a point in the scatter plot to view its full vertical timelapse strip. <br>
            Click a second point to compare side-by-side. A third click will replace the first image.
        </div>
        <div id="plot"></div>
    </div>

    <div id="images-container">
        <div class="image-col">
            <div id="title1" class="image-title">Slot 1: Empty</div>
            <img id="img1" class="strip-img" style="display:none;" />
        </div>
        <div class="image-col">
            <div id="title2" class="image-title">Slot 2: Empty</div>
            <img id="img2" class="strip-img" style="display:none;" />
        </div>
    </div>

    <script>
        const data = {points_json};
        
        const x = data.map(d => d.x);
        const y = data.map(d => d.y);
        const text = data.map(d => d.id);
        
        const trace = {{
            x: x,
            y: y,
            text: text,
            mode: 'markers',
            type: 'scatter',
            marker: {{
                size: 8,
                color: '#bb86fc',
                opacity: 0.7,
                line: {{
                    color: '#000000',
                    width: 1
                }}
            }},
            hoverinfo: 'text'
        }};
        
        const layout = {{
            hovermode: 'closest',
            paper_bgcolor: '#1e1e1e',
            plot_bgcolor: '#1e1e1e',
            font: {{ color: '#ffffff' }},
            xaxis: {{ showgrid: false, zeroline: false, visible: false }},
            yaxis: {{ showgrid: false, zeroline: false, visible: false }},
            margin: {{ t: 20, b: 20, l: 20, r: 20 }}
        }};
        
        Plotly.newPlot('plot', [trace], layout);
        
        let currentSlot = 1;
        
        document.getElementById('plot').on('plotly_click', function(eventData) {{
            const pointIndex = eventData.points[0].pointIndex;
            const clickedId = data[pointIndex].id;
            const imgSrc = 'vertical_strips/' + clickedId + '.png';
            
            // Update the current slot
            const imgEl = document.getElementById('img' + currentSlot);
            const titleEl = document.getElementById('title' + currentSlot);
            
            imgEl.src = imgSrc;
            imgEl.style.display = 'block';
            titleEl.innerText = clickedId;
            
            // Highlight selected point
            const colors = new Array(data.length).fill('#bb86fc');
            const sizes = new Array(data.length).fill(8);
            
            // If we want to keep track of the two selected points to highlight both, we could.
            // But just highlighting the newly clicked one is easy:
            colors[pointIndex] = '#03dac6';
            sizes[pointIndex] = 12;
            
            Plotly.restyle('plot', {{'marker.color': [colors], 'marker.size': [sizes]}});
            
            // Toggle slot for next click
            currentSlot = currentSlot === 1 ? 2 : 1;
        }});
    </script>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)
        
    print(f"Done! HTML saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_strips_and_html()
