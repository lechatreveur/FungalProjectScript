#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import sys
import json

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# ==== 1. Configuration ====
FINAL_4D_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/fc_final_4d/"
CLUSTERED_CSV = os.path.join(FINAL_4D_DIR, "fc_ae_4d_clustered.csv")
OUTPUT_HTML = os.path.join(FINAL_4D_DIR, "fc_interactive_umap.html")

EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

def generate_interactive_dashboard():
    # 1. Load Data
    print("📥 Loading clustered data and trajectories...")
    df = pd.read_csv(CLUSTERED_CSV, index_col=0)
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    
    # 2. Prepare JSON for D3.js
    points = []
    for i, gid in enumerate(gids):
        # Inverse transform trajectory
        traj_raw = s_traj.inverse_transform(X_traj[i])
        
        points.append({
            "id": gid,
            "u1": float(df.loc[gid, 'UMAP1']),
            "u2": float(df.loc[gid, 'UMAP2']),
            "cluster": int(df.loc[gid, 'phenotype_cluster']),
            "exp": labels[i],
            "pol1": traj_raw[:, 0].tolist(),
            "pol2": traj_raw[:, 1].tolist()
        })
    
    # 3. HTML Template with Embedded Data and D3.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fungal Phenotype Explorer (4D Constrained)</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: sans-serif; display: flex; height: 100vh; margin: 0; background: #1a1a1a; color: white; }}
        #umap-container {{ flex: 2; border-right: 1px solid #444; position: relative; }}
        #side-panel {{ flex: 1; padding: 20px; display: flex; flex-direction: column; overflow-y: auto; background: #222; }}
        .dot {{ cursor: pointer; transition: r 0.2s; }}
        .dot:hover {{ r: 8; stroke: white; stroke-width: 2px; }}
        #tooltip {{ position: absolute; background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px; pointer-events: none; opacity: 0; }}
        h2 {{ margin-top: 0; color: #00ffcc; }}
        .trace-line {{ fill: none; stroke-width: 2; }}
        .pol1 {{ stroke: #00ffff; }}
        .pol2 {{ stroke: #ff3366; }}
    </style>
</head>
<body>
    <div id="umap-container">
        <div id="tooltip"></div>
    </div>
    <div id="side-panel">
        <h2>Cell Details</h2>
        <div id="cell-info">Hover over a point to see the trace.</div>
        <svg id="trace-plot" width="100%" height="300"></svg>
        <div id="feature-list" style="font-size: 0.9em; margin-top: 20px;"></div>
    </div>

    <script>
        const data = {json.dumps(points)};
        const colors = d3.schemeCategory10;

        const width = document.getElementById('umap-container').clientWidth;
        const height = document.getElementById('umap-container').clientHeight;

        const svg = d3.select("#umap-container").append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .call(d3.zoom().on("zoom", (e) => g.attr("transform", e.transform)));

        const g = svg.append("g");

        const x = d3.scaleLinear().domain(d3.extent(data, d => d.u1)).range([50, width - 50]);
        const y = d3.scaleLinear().domain(d3.extent(data, d => d.u2)).range([height - 50, 50]);

        const dots = g.selectAll(".dot")
            .data(data)
            .enter().append("circle")
            .attr("class", "dot")
            .attr("cx", d => x(d.u1))
            .attr("cy", d => y(d.u2))
            .attr("r", 4)
            .attr("fill", d => colors[d.cluster % 10])
            .on("mouseover", showDetail);

        function showDetail(event, d) {{
            d3.select("#cell-info").html(`<b>ID:</b> ${{d.id}}<br><b>Exp:</b> ${{d.exp}}<br><b>Cluster:</b> ${{d.cluster}}`);
            updateTrace(d);
        }}

        function updateTrace(d) {{
            const tsvg = d3.select("#trace-plot");
            tsvg.selectAll("*").remove();
            
            const tw = tsvg.node().clientWidth;
            const th = 300;
            const tx = d3.scaleLinear().domain([0, 100]).range([10, tw-10]);
            const ty = d3.scaleLinear().domain([0, d3.max([...d.pol1, ...d.pol2])]).range([th-20, 20]);

            const line = d3.line().x((v,i) => tx(i)).y(v => ty(v));

            tsvg.append("path").datum(d.pol1).attr("class", "trace-line pol1").attr("d", line);
            tsvg.append("path").datum(d.pol2).attr("class", "trace-line pol2").attr("d", line);
            
            tsvg.append("text").attr("x", 10).attr("y", 15).attr("fill", "#00ffff").text("Pol1");
            tsvg.append("text").attr("x", 60).attr("y", 15).attr("fill", "#ff3366").text("Pol2");
        }}

        // Add IDs as labels (toggleable or always on if zoomed?)
        g.selectAll(".label")
            .data(data)
            .enter().append("text")
            .attr("x", d => x(d.u1) + 5)
            .attr("y", d => y(d.u2))
            .text(d => d.id)
            .style("font-size", "8px")
            .style("fill", "#888")
            .style("pointer-events", "none");

    </script>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)
    print(f"💾 Interactive dashboard saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_interactive_dashboard()
