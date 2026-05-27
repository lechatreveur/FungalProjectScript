#!/usr/bin/env python3
"""
Strategy A: Direct UMAP on Pol1 + Pol2 temporal traces.

For each cell, we compute the spatially-averaged Pol1 (Ch2) and Pol2 (Ch3)
probability at each timepoint from the gamma cache. This gives a pure
temporal signal of polarity site dynamics, with no compression or loss.

We then run UMAP on this (419, 202) feature matrix and embed the result
in an interactive HTML dashboard identical to the existing one.
"""

import os
import sys
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64
from io import BytesIO

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

BASE_DIR     = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
GAMMA_CACHE  = os.path.join(BASE_DIR, "gamma_cache_32x112_padded.npy")
VIDEO_CACHE  = os.path.join(BASE_DIR, "video_cache_32x112_padded.npy")
CACHE_GIDS   = os.path.join(BASE_DIR, "video_gids.txt")
OUT_LATENTS  = os.path.join(BASE_DIR, "polarity_traces.npy")
OUT_UMAP_PNG = os.path.join(BASE_DIR, "polarity_umap.png")
OUT_HTML     = os.path.join(BASE_DIR, "polarity_umap_interactive.html")

# Channel indices in gamma cache
CH_POL1 = 2
CH_POL2 = 3
CH_CYT0 = 0

def main():
    print("📥 Loading gamma cache (memory-mapped)...")
    gammas = np.load(GAMMA_CACHE, mmap_mode='r')  # (N, 101, 7, 32, 112)
    videos = np.load(VIDEO_CACHE, mmap_mode='r')  # (N, 101, 1, 32, 112)
    with open(CACHE_GIDS) as f:
        gids = [l.strip() for l in f]
    n_cells = gammas.shape[0]
    print(f"   {n_cells} cells loaded.")

    # --- 1. Extract temporal traces ---
    print("📊 Computing polarity temporal traces...")
    pol1_traces = np.zeros((n_cells, 101), dtype=np.float32)  # mean Pol1 over space
    pol2_traces = np.zeros((n_cells, 101), dtype=np.float32)  # mean Pol2 over space
    cyto_traces = np.zeros((n_cells, 101), dtype=np.float32)  # mean Cyto (for normalization)

    for i in range(n_cells):
        pol1 = gammas[i, :, CH_POL1, :, :]  # (101, 32, 112)
        pol2 = gammas[i, :, CH_POL2, :, :]
        cyto = gammas[i, :, CH_CYT0, :, :]
        pol1_traces[i] = pol1.mean(axis=(1, 2))
        pol2_traces[i] = pol2.mean(axis=(1, 2))
        cyto_traces[i] = cyto.mean(axis=(1, 2))

    # Smooth with a light Gaussian (sigma=2 frames)
    for i in range(n_cells):
        pol1_traces[i] = gaussian_filter1d(pol1_traces[i], sigma=2)
        pol2_traces[i] = gaussian_filter1d(pol2_traces[i], sigma=2)

    # Concatenate Pol1 and Pol2 traces -> (N, 202) feature matrix
    features = np.hstack([pol1_traces, pol2_traces])
    np.save(OUT_LATENTS, features)
    print(f"   Feature matrix shape: {features.shape}")

    # --- 2. UMAP ---
    print("🔵 Computing UMAP...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features_scaled)
    print(f"   Embedding shape: {embedding.shape}")

    # --- 3. Static plot coloured by total polarity intensity ---
    total_pol = (pol1_traces + pol2_traces).mean(axis=1)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1],
                     c=total_pol, cmap='plasma', s=20, alpha=0.8)
    plt.colorbar(sc, label='Mean Polarity Intensity')
    plt.title('Polarity Dynamics UMAP (Strategy A)', fontsize=16)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(OUT_UMAP_PNG, dpi=150)
    print(f"   Saved UMAP to {OUT_UMAP_PNG}")
    plt.close()

    # --- 4. Generate vertical strips from video cache ---
    print("🎞️  Generating vertical strips...")
    strips_b64 = []
    for i in range(n_cells):
        vid = videos[i, :, 0, :, :]     # (101, 32, 112)
        pol1 = gammas[i, :, CH_POL1, :, :]  # (101, 32, 112)
        pol2 = gammas[i, :, CH_POL2, :, :]

        # Stack all 101 frames vertically; overlay polarity heatmap
        frames = []
        for t in range(101):
            frame_gray = vid[t]
            frame_norm = ((frame_gray - frame_gray.min()) /
                          (frame_gray.max() - frame_gray.min() + 1e-6) * 255).astype(np.uint8)
            rgb = np.stack([frame_norm, frame_norm, frame_norm], axis=-1)
            # Overlay Pol1 in green, Pol2 in magenta
            p1 = (pol1[t] * 255).clip(0, 255).astype(np.uint8)
            p2 = (pol2[t] * 255).clip(0, 255).astype(np.uint8)
            rgb[:, :, 1] = np.clip(rgb[:, :, 1].astype(int) + p1.astype(int), 0, 255).astype(np.uint8)
            rgb[:, :, 0] = np.clip(rgb[:, :, 0].astype(int) + p2.astype(int), 0, 255).astype(np.uint8)
            rgb[:, :, 2] = np.clip(rgb[:, :, 2].astype(int) - p2.astype(int) // 2, 0, 255).astype(np.uint8)
            frames.append(rgb)

        strip = np.vstack(frames)  # (101*32, 112, 3)
        img = Image.fromarray(strip)
        buf = BytesIO()
        img.save(buf, format='PNG', optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        strips_b64.append(b64)
        if (i + 1) % 50 == 0:
            print(f"   Generated {i+1}/{n_cells} strips...")

    # --- 5. Generate Interactive HTML ---
    print("🌐 Building interactive HTML...")
    points_json = []
    for i in range(n_cells):
        pol_intensity = float((pol1_traces[i] + pol2_traces[i]).mean())
        points_json.append({
            'x': float(embedding[i, 0]),
            'y': float(embedding[i, 1]),
            'gid': gids[i],
            'pol': pol_intensity,
            'img': strips_b64[i],
        })

    import json
    points_json_str = json.dumps(points_json)

    # Colour scale for intensity
    pols = np.array([p['pol'] for p in json.loads(points_json_str)])
    pol_min, pol_max = float(pols.min()), float(pols.max())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Polarity Dynamics UMAP</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d1a; color: #e0e0ff; font-family: 'Inter', sans-serif; display: flex; height: 100vh; overflow: hidden; }}
  #controls {{ position: absolute; top: 16px; left: 16px; z-index: 10; background: rgba(20,20,40,0.9); padding: 12px 16px; border-radius: 10px; border: 1px solid #333; font-size: 13px; max-width: 260px; }}
  #controls h2 {{ font-size: 15px; margin-bottom: 8px; color: #a78bfa; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  #canvas-wrapper {{ flex: 1; position: relative; }}
  canvas {{ cursor: crosshair; }}
  #panels {{ width: 480px; background: #0a0a18; border-left: 1px solid #1e1e3a; display: flex; gap: 0; overflow: hidden; }}
  .panel {{ flex: 1; display: flex; flex-direction: column; padding: 12px; border-right: 1px solid #1e1e3a; }}
  .panel:last-child {{ border-right: none; }}
  .panel-title {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; text-align: center; min-height: 32px; }}
  .panel-img {{ overflow-y: auto; flex: 1; text-align: center; }}
  .panel-img img {{ width: 112px; image-rendering: pixelated; }}
  #tooltip {{ position: absolute; background: rgba(20,20,40,0.95); border: 1px solid #4444aa; padding: 6px 10px; border-radius: 6px; font-size: 12px; pointer-events: none; display: none; z-index: 100; }}
</style>
</head>
<body>
<div id="controls">
  <h2>🔬 Polarity Dynamics UMAP</h2>
  <p style="color:#aaa;font-size:11px;margin-bottom:8px">Strategy A: Direct UMAP on Pol1+Pol2 traces<br>Click points to compare cells side by side.</p>
  <div style="margin-top:8px">
    <span class="legend-dot" style="background:#00ff88"></span>Green = Pol1 site<br>
    <span class="legend-dot" style="background:#ff44aa"></span>Magenta = Pol2 site
  </div>
  <div id="status" style="margin-top:10px;color:#aaa;font-size:11px">Click a point to load its strip.</div>
</div>
<div id="canvas-wrapper">
  <canvas id="umap"></canvas>
  <div id="tooltip"></div>
</div>
<div id="panels">
  <div class="panel" id="panelA">
    <div class="panel-title" id="titleA">— Click a point —</div>
    <div class="panel-img" id="imgA"></div>
  </div>
  <div class="panel" id="panelB">
    <div class="panel-title" id="titleB">— Click a second —</div>
    <div class="panel-img" id="imgB"></div>
  </div>
</div>
<script>
const DATA = {points_json_str};
const POL_MIN = {pol_min};
const POL_MAX = {pol_max};

const canvas = document.getElementById('umap');
const ctx = canvas.getContext('2d');
let W, H, transform, selected = [];

function resize() {{
  const wrapper = document.getElementById('canvas-wrapper');
  W = wrapper.clientWidth; H = wrapper.clientHeight;
  canvas.width = W; canvas.height = H;
  computeTransform(); draw();
}}

function computeTransform() {{
  const xs = DATA.map(d => d.x), ys = DATA.map(d => d.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = 60;
  const sx = (W - 2*pad) / (xMax - xMin);
  const sy = (H - 2*pad) / (yMax - yMin);
  const s = Math.min(sx, sy);
  transform = {{ s, ox: pad - xMin*s + (W - 2*pad - (xMax-xMin)*s)/2, oy: pad - yMin*s + (H - 2*pad - (yMax-yMin)*s)/2 }};
}}

function toCanvas(x, y) {{ return [x * transform.s + transform.ox, y * transform.s + transform.oy]; }}

function polarColor(pol) {{
  const t = (pol - POL_MIN) / (POL_MAX - POL_MIN + 1e-9);
  const r = Math.round(80 + t * 175);
  const g = Math.round(20 + t * 60);
  const b = Math.round(180 - t * 100);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function draw() {{
  ctx.clearRect(0, 0, W, H);
  DATA.forEach((d, i) => {{
    const [cx, cy] = toCanvas(d.x, d.y);
    const isA = selected[0] === i, isB = selected[1] === i;
    ctx.beginPath();
    ctx.arc(cx, cy, isA || isB ? 7 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = isA ? '#00ffaa' : isB ? '#ff6699' : polarColor(d.pol);
    ctx.fill();
    if (isA || isB) {{ ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke(); }}
  }});
}}

function nearestPoint(mx, my) {{
  let best = -1, bestD = Infinity;
  DATA.forEach((d, i) => {{
    const [cx, cy] = toCanvas(d.x, d.y);
    const dist = (mx-cx)**2 + (my-cy)**2;
    if (dist < bestD) {{ bestD = dist; best = i; }}
  }});
  return bestD < 400 ? best : -1;
}}

function showStrip(idx, slot) {{
  const d = DATA[idx];
  const titleEl = document.getElementById('title' + slot);
  const imgEl = document.getElementById('img' + slot);
  titleEl.textContent = d.gid + ' | pol=' + d.pol.toFixed(4);
  imgEl.innerHTML = `<img src="data:image/png;base64,${{d.img}}" alt="strip"/>`;
}}

canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const idx = nearestPoint(e.clientX - rect.left, e.clientY - rect.top);
  if (idx < 0) return;
  if (selected.length === 0 || selected.length === 2) {{
    selected = [idx]; showStrip(idx, 'A');
    document.getElementById('titleB').textContent = '— Click a second —';
    document.getElementById('imgB').innerHTML = '';
  }} else {{
    if (idx === selected[0]) return;
    selected.push(idx); showStrip(idx, 'B');
  }}
  document.getElementById('status').textContent = `Selected: ${{selected.length}} cell(s)`;
  draw();
}});

const tooltip = document.getElementById('tooltip');
canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const idx = nearestPoint(e.clientX - rect.left, e.clientY - rect.top);
  if (idx >= 0) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX - rect.left + 14) + 'px';
    tooltip.style.top  = (e.clientY - rect.top  - 10) + 'px';
    tooltip.textContent = DATA[idx].gid + ' | pol=' + DATA[idx].pol.toFixed(4);
  }} else {{ tooltip.style.display = 'none'; }}
}});

window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>"""

    with open(OUT_HTML, 'w') as f:
        f.write(html)
    print(f"✅ Interactive HTML saved to {OUT_HTML}")

if __name__ == "__main__":
    main()
