#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Web Tool for Manual Correction of Fungal Cell Tracking and Segmentation.
Provides a modern Flask-based web interface to review, edit, and re-quantify tracked cells,
as well as stitch sequences of films and correct cross-film cell linkages.
"""

import os
import sys
import re
import json
import subprocess
import webbrowser
import cv2
import threading
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template_string
from skimage.io import imread
from skimage.measure import label


# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Cell_tracking_functions import rle_decode, rle_encode, load_segmentation

app = Flask(__name__)

# Default movie root
BASE_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
NAS_MOVIE_ROOT = Path("/Volumes/Movies")

RELEVANT_EXPERIMENTS = [
    "2025_06_25",
    "2025_09_17",
    "2025_12_31_M92",
    "2026_01_08_M93",
    "2026_01_16_M96",
    "2026_01_18_M97",
    "2026_04_09_M125",
    "2026_04_23_M130",
    "2026_04_29_M133",
    "2026_04_30_M135",
    "2026_06_03_M143"
]

# In-memory cache for suspicious cell analysis results: key = "exp::film"
SUSPICIOUS_CACHE = {}

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fungal Cell Tracking Corrector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --border-color: #334155;
            --accent-primary: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        header {
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 10;
        }

        h1 {
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.025em;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        select {
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.875rem;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }

        main {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .left-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-color);
            position: relative;
            background-color: #0b0f19;
        }

        .right-panel {
            width: 420px;
            background-color: var(--bg-card);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            border-left: 1px solid var(--border-color);
        }

        .canvas-container {
            flex: 1;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: crosshair;
        }

        #imageCanvas {
            position: absolute;
            left: 0;
            top: 0;
            transform-origin: 0 0;
            image-rendering: pixelated;
        }

        .viewer-controls {
            background-color: var(--bg-card);
            border-top: 1px solid var(--border-color);
            padding: 16px 24px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .control-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }

        .btn-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        button {
            background-color: var(--border-color);
            color: var(--text-main);
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        button:hover {
            background-color: #475569;
        }

        button.active {
            background-color: var(--accent-primary);
        }

        button.btn-success { background-color: #064e3b; color: #d1fae5; }
        button.btn-success.active { background-color: var(--accent-green); color: var(--text-main); box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
        button.btn-danger { background-color: #7f1d1d; color: #fee2e2; }
        button.btn-danger.active { background-color: var(--accent-red); color: var(--text-main); box-shadow: 0 0 8px rgba(239, 68, 68, 0.5); }
        button.btn-corrected { background-color: #0c4a6e; color: #e0f2fe; }
        button.btn-corrected.active { background-color: var(--accent-primary); color: var(--text-main); box-shadow: 0 0 8px rgba(59, 130, 246, 0.5); }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 12px;
            flex: 1;
        }

        input[type="range"] {
            flex: 1;
            accent-color: var(--accent-primary);
            height: 6px;
            border-radius: 3px;
            background: var(--border-color);
            outline: none;
            cursor: pointer;
        }

        .panel-section {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .panel-section-title {
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }

        .cell-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }
        
        .global-cell-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }

        .cell-item {
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
            position: relative;
        }

        .cell-item:hover { border-color: #475569; }
        .cell-item.active { background-color: var(--accent-primary); border-color: var(--accent-primary); }
        .cell-item.qc-good { background-color: #064e3b; color: #d1fae5; border-color: #047857; }
        .cell-item.qc-bad { background-color: #7f1d1d; color: #fee2e2; border-color: #b91c1c; }
        .cell-item.qc-corrected, .cell-item.qc-review { background-color: #0c4a6e; color: #e0f2fe; border-color: #0369a1; }
        .cell-item.qc-good.active, .cell-item.qc-bad.active, .cell-item.qc-corrected.active, .cell-item.qc-review.active {
            border-color: var(--text-main);
            box-shadow: 0 0 5px var(--text-main);
        }
        .cell-item.suspicious::after {
            content: '⚠';
            font-size: 0.65rem;
            color: #fbbf24;
            position: absolute;
            top: 2px;
            right: 3px;
        }

        .info-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.9rem; }
        .info-label { color: var(--text-muted); }
        .info-value { font-weight: 500; }

        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-pending { background-color: #581c87; color: #f3e8ff; }
        .badge-good { background-color: #064e3b; color: #d1fae5; }
        .badge-bad { background-color: #7f1d1d; color: #fee2e2; }
        .badge-corrected { background-color: #0369a1; color: #e0f2fe; }

        .slider-wrapper { position: relative; flex: 1; display: flex; flex-direction: column; gap: 3px; }
        :root { --tick-inset: 8px; }
        .suspicious-ticks { position: absolute; top: -12px; left: var(--tick-inset); right: var(--tick-inset); height: 8px; pointer-events: none; }
        .suspicious-tick { position: absolute; width: 3px; height: 8px; background-color: #f59e0b; border-radius: 1px; transform: translateX(-50%); opacity: 0.85; }
        .film-boundary-tick { position: absolute; width: 2px; height: 16px; background-color: #38bdf8; top: 0px; transform: translateX(-50%); z-index: 5; }
        .film-boundary-label { position: absolute; top: 18px; font-size: 0.65rem; color: #38bdf8; transform: translateX(-50%); white-space: nowrap; }

        .suspicious-frame-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
        .suspicious-frame-btn { background-color: #451a03; border: 1px solid #92400e; color: #fde68a; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer; transition: all 0.15s; }
        .suspicious-frame-btn:hover { background-color: #78350f; color: #fef3c7; }

        .zoom-status {
            position: absolute;
            top: 16px;
            left: 16px;
            background-color: rgba(15, 23, 42, 0.85);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            pointer-events: none;
            backdrop-filter: blur(4px);
            z-index: 10;
        }

        .strip-container { display: flex; overflow-x: auto; gap: 8px; padding: 8px; background-color: #0b0f19; border-radius: 6px; border: 1px solid var(--border-color); min-height: 100px; }
        .strip-crop { flex-shrink: 0; border: 1px solid var(--border-color); border-radius: 4px; background-color: #000; box-sizing: border-box; }
        .strip-crop.active { border-color: var(--accent-primary); box-shadow: 0 0 8px rgba(59, 130, 246, 0.5); }
        .strip-crop.septum-start-frame { border: 3px solid var(--accent-green) !important; box-shadow: 0 0 10px rgba(16, 185, 129, 0.8) !important; }
        .strip-crop.septum-end-frame { border: 3px solid var(--accent-red) !important; box-shadow: 0 0 10px rgba(239, 68, 68, 0.8) !important; }
        .strip-crop.septum-during-frame { background-color: rgba(16, 185, 129, 0.15) !important; border-bottom: 3px solid var(--accent-green) !important; }
        
        .strip-crop.septum-start-frame-2 { border: 3px solid var(--accent-green) !important; box-shadow: 0 0 10px rgba(16, 185, 129, 0.8) !important; }
        .strip-crop.septum-end-frame-2 { border: 3px solid var(--accent-red) !important; box-shadow: 0 0 10px rgba(239, 68, 68, 0.8) !important; }
        .strip-crop.septum-during-frame-2 { background-color: rgba(16, 185, 129, 0.15) !important; border-bottom: 3px solid var(--accent-green) !important; }


        .linkage-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px;
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }

        /* Modal */
        #modalOverlay {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        #modalContent {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 24px;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
        }
        #modalGallery {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            overflow-y: auto;
            margin-top: 16px;
            padding: 8px;
            background: var(--bg-dark);
            border-radius: 6px;
            flex: 1;
        }
        .candidate-crop {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            padding: 8px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .candidate-crop:hover {
            border-color: var(--accent-primary);
            background: #1e3a8a;
        }
        .candidate-crop img {
            width: 80px; height: 80px; object-fit: contain; background: black;
        }
        .candidate-crop span {
            font-size: 0.8rem; font-weight: 600;
        }
    </style>
</head>
<body>

    <header>
        <h1>🦠 Fungal Cell Tracking Corrector</h1>
        <div class="header-controls">
            <div>
                <span style="color: var(--text-muted); font-size: 0.85rem; margin-right: 8px;">Experiment:</span>
                <select id="experimentSelect"></select>
            </div>
            <div id="sequenceContainer">
                <span style="color: var(--text-muted); font-size: 0.85rem; margin-right: 8px;">Sequence/Dataset:</span>
                <select id="sequenceSelect"></select>
            </div>
        </div>
    </header>

    <main>
        <div class="left-panel">
            <div class="zoom-status">
                <div>Scale: <span id="scaleLabel">1.0</span>x</div>
                <div>Pan: (<span id="panXLabel">0</span>, <span id="panYLabel">0</span>)</div>
                <div style="margin-top: 4px; color: var(--text-main);">Mode: <strong id="modeLabel" style="color: var(--accent-primary);">Click-Select</strong></div>
            </div>

            <div class="canvas-container" id="canvasContainer">
                <canvas id="imageCanvas"></canvas>
            </div>

            <div class="viewer-controls">
                <div class="control-row">
                    <div class="btn-group">
                        <button id="prevBtn" title="Previous Frame (Left Arrow)">◀</button>
                        <button id="playBtn" title="Play/Pause (Space)">Play</button>
                        <button id="nextBtn" title="Next Frame (Right Arrow)">▶</button>
                    </div>

                    <div class="slider-container">
                        <span id="currentTimeLabel">t=0</span>
                        <div class="slider-wrapper">
                            <div class="suspicious-ticks" id="suspiciousTicks"></div>
                            <div id="filmBoundaries"></div>
                            <input type="range" id="timeSlider" min="0" max="0" value="0">
                        </div>
                        <span id="maxTimeLabel">t=0</span>
                    </div>

                    <div class="btn-group">
                        <span style="font-size: 0.85rem; color: var(--text-muted);">Channel:</span>
                        <button id="chanBfBtn" class="active">BF</button>
                        <button id="chanGfpBtn">GFP</button>
                    </div>

                    <div class="btn-group">
                        <span style="font-size: 0.85rem; color: var(--text-muted);">View:</span>
                        <button id="viewModeBtn" style="background-color: #581c87; color: white; font-weight: 500;">Single Cell</button>
                    </div>
                </div>

                <div class="control-row" style="border-top: 1px solid var(--border-color); padding-top: 12px; margin-top: 4px;">
                    <div class="btn-group">
                        <button id="toolSelectBtn" class="active">Select Segment (S)</button>
                        <button id="toolBrushBtn">Brush Draw (B)</button>
                        <button id="toolEraserBtn">Eraser (E)</button>
                    </div>
                    <div class="btn-group" id="brushControls" style="display: none;">
                        <span style="font-size: 0.85rem; color: var(--text-muted);">Brush Size:</span>
                        <input type="range" id="brushSizeSlider" min="1" max="50" value="10" style="width: 100px;">
                        <span id="brushSizeLabel">10px</span>
                    </div>
                    <div class="btn-group">
                        <button id="undoBtn">Undo (Ctrl+Z)</button>
                        <button id="usePrevSegmentBtn" title="Use segment from previous frame (P)">Use Previous Segment (P)</button>
                        <button id="clearBtn">Clear Frame Mask</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="right-panel">
            <div class="panel-section" id="linkageSection">
                <div class="panel-section-title">
                    Tracking Linkages
                    <span id="currentGlobalCellLabel" style="color: var(--accent-primary);"></span>
                </div>
                <div id="linkageList"></div>
            </div>

            <div class="panel-section">
                <div class="panel-section-title" style="display: flex; justify-content: space-between; align-items: center;">
                    Cell Selection
                    <button id="trackNewCellBtn" style="padding: 4px 8px; font-size: 0.75rem; background-color: var(--accent-primary); color: white; display: none;">+ Track New Cell</button>
                </div>
                <div class="cell-grid" id="cellGrid"></div>
            </div>

            <div class="panel-section">
                <div class="panel-section-title">Cell Details & QC</div>
                <div class="info-row">
                    <span class="info-label">Active Cell:</span>
                    <span class="info-value" id="cellIdLabel">None</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Tracking Channel:</span>
                    <span class="info-value" id="cellChannelLabel">-</span>
                </div>
                <div class="info-row" id="localFilmRow" style="display:none;">
                    <span class="info-label">Current Film:</span>
                    <span class="info-value" id="localFilmLabel">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Mask Status:</span>
                    <span id="maskStatusText" class="status-badge badge-pending">Pending</span>
                </div>
                <div class="btn-group" style="margin-top: 16px; width: 100%; justify-content: space-between;">
                    <button id="btnMarkGood" class="btn-success">Good</button>
                    <button id="btnMarkBad" class="btn-danger">Bad</button>
                    <button id="btnMarkCorrected" class="btn-corrected">Corrected</button>
                </div>
                <div class="btn-group" id="quantifyHpcGroup" style="margin-top: 16px; width: 100%; display: none;">
                    <button id="btnQuantifyHpc" style="width: 100%; background-color: var(--accent-secondary); color: white;">💻 Quantify Locally (Seed from CSV)</button>
                </div>
                <button id="btnExitLocalEdit" style="margin-top: 12px; width: 100%; display: none; background-color: #4b5563; color: white;">🔙 Return to Linkage</button>
            </div>

            <div class="panel-section" id="suspiciousSection" style="display:none;">
                <div class="panel-section-title" style="color: #f59e0b;">⚠ Suspicious Frames</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px;">Frames where the segment's centroid jumps significantly.</div>
                <div class="suspicious-frame-list" id="suspiciousFrameList"></div>
            </div>

            <div class="panel-section" id="divisionSection" style="display:none;">
                <div class="panel-section-title" style="display: flex; justify-content: space-between; align-items: center;">
                    Division & Septum AI
                    <button id="predictSeptumBtn" style="padding: 4px 8px; font-size: 0.75rem; background-color: #581c87; color: white;">🤖 Run AI</button>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 8px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: bold;">Septum 1</span>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 0.85rem; color: var(--text-muted);">Has Septum:</span>
                        <input type="checkbox" id="hasSeptumCheckbox" style="width: 16px; height: 16px; accent-color: var(--accent-green);">
                    </div>
                    <div id="divisionIntervalContainer" style="display: none; flex-direction: column; gap: 8px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                            <span style="font-size: 0.85rem; color: var(--text-muted); width: 80px;">Start Frame:</span>
                            <input type="number" id="septumStartInput" style="width: 70px; background-color: var(--bg-dark); border: 1px solid var(--border-color); color: white; padding: 4px; border-radius: 4px;" min="0">
                            <button id="setSeptumStartBtn" style="padding: 4px 8px; font-size: 0.7rem;">Set Current</button>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                            <span style="font-size: 0.85rem; color: var(--text-muted); width: 80px;">End Frame:</span>
                            <input type="number" id="septumEndInput" style="width: 70px; background-color: var(--bg-dark); border: 1px solid var(--border-color); color: white; padding: 4px; border-radius: 4px;" min="0">
                            <button id="setSeptumEndBtn" style="padding: 4px 8px; font-size: 0.7rem;">Set Current</button>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; background: rgba(255,255,255,0.03); padding: 6px; border-radius: 4px; border: 1px solid var(--border-color); margin-top: 4px;">
                            <span style="font-size: 0.8rem; color: var(--text-muted);">Click Gallery to:</span>
                            <div style="display: flex; gap: 4px;">
                                <button id="galleryClickNavBtn" class="active" style="padding: 2px 6px; font-size: 0.7rem;">Nav</button>
                                <button id="galleryClickStartBtn" style="padding: 2px 6px; font-size: 0.7rem;">Start 1</button>
                                <button id="galleryClickEndBtn" style="padding: 2px 6px; font-size: 0.7rem;">End 1</button>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span style="font-size: 0.85rem; color: var(--text-muted);">White Septum:</span>
                            <input type="checkbox" id="whiteSeptumCheckbox" style="width: 16px; height: 16px; accent-color: #eab308;">
                        </div>
                    </div>

                    <hr style="border-color: var(--border-color); margin: 8px 0;">

                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: bold;">Septum 2</span>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 0.85rem; color: var(--text-muted);">Has Septum 2:</span>
                        <input type="checkbox" id="hasSeptumCheckbox2" style="width: 16px; height: 16px; accent-color: var(--accent-green);">
                    </div>
                    <div id="divisionIntervalContainer2" style="display: none; flex-direction: column; gap: 8px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                            <span style="font-size: 0.85rem; color: var(--text-muted); width: 80px;">Start Frame:</span>
                            <input type="number" id="septumStartInput2" style="width: 70px; background-color: var(--bg-dark); border: 1px solid var(--border-color); color: white; padding: 4px; border-radius: 4px;" min="0">
                            <button id="setSeptumStartBtn2" style="padding: 4px 8px; font-size: 0.7rem;">Set Current</button>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                            <span style="font-size: 0.85rem; color: var(--text-muted); width: 80px;">End Frame:</span>
                            <input type="number" id="septumEndInput2" style="width: 70px; background-color: var(--bg-dark); border: 1px solid var(--border-color); color: white; padding: 4px; border-radius: 4px;" min="0">
                            <button id="setSeptumEndBtn2" style="padding: 4px 8px; font-size: 0.7rem;">Set Current</button>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; background: rgba(255,255,255,0.03); padding: 6px; border-radius: 4px; border: 1px solid var(--border-color); margin-top: 4px;">
                            <span style="font-size: 0.8rem; color: var(--text-muted);">Click Gallery to:</span>
                            <div style="display: flex; gap: 4px;">
                                <button id="galleryClickStartBtn2" style="padding: 2px 6px; font-size: 0.7rem;">Start 2</button>
                                <button id="galleryClickEndBtn2" style="padding: 2px 6px; font-size: 0.7rem;">End 2</button>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span style="font-size: 0.85rem; color: var(--text-muted);">White Septum:</span>
                            <input type="checkbox" id="whiteSeptumCheckbox2" style="width: 16px; height: 16px; accent-color: #eab308;">
                        </div>
                    </div>
                    <!-- AI Probability Chart -->
                    <div id="septumAiChart" style="display: none; flex-direction: column; gap: 4px; margin-top: 8px;">
                        <div style="font-size: 0.75rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                            <span>AI Septum Probabilities</span>
                            <span id="septumAiPeakText" style="color: var(--accent-green); font-weight: bold;"></span>
                        </div>
                        <div id="septumAiSparkline" style="height: 32px; background: #0b0f19; border: 1px solid var(--border-color); border-radius: 4px; display: flex; align-items: flex-end; padding: 2px; overflow: hidden; position: relative;">
                        </div>
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <div class="panel-section-title">Dynamic Cell Gallery</div>
                <div class="strip-container" id="stripContainer">
                    <span style="color: var(--text-muted); font-size: 0.85rem;">Select a cell to view strip crops...</span>
                </div>
            </div>

            <div class="panel-section">
                <div class="panel-section-title">Bulk Auto-Fix from Raw Segments</div>
                <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
                    <input type="number" id="autofixStartInput" placeholder="Start T" style="width: 80px;" class="custom-input">
                    <span style="color: var(--text-muted);">to</span>
                    <input type="number" id="autofixEndInput" placeholder="End T" style="width: 80px;" class="custom-input">
                </div>
                <div class="btn-group" style="width: 100%;">
                    <button class="btn btn-secondary" id="setAutofixStartBtn" title="Set start to current frame" style="flex: 1; padding: 4px;">Set Start</button>
                    <button class="btn btn-secondary" id="setAutofixEndBtn" title="Set end to current frame" style="flex: 1; padding: 4px;">Set End</button>
                    <button class="btn btn-primary" id="runAutofixBtn" style="flex: 2; padding: 4px;" title="Auto-fix masks in range based on highest IoU with raw segments">Run Auto-Fix</button>
                </div>
            </div>

            <div class="panel-section" style="margin-top: auto;">
                <div class="btn-group" style="width: 100%; flex-direction: column; gap: 10px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
                        <span style="font-size: 0.75rem; color: var(--text-muted);">Autosaves on change</span>
                        <span id="autosaveStatus" style="font-size: 0.75rem; padding: 2px 8px; border-radius: 10px; background: #1e293b; color: var(--text-muted);">Idle</span>
                    </div>
                    <button id="btnSaveCell" class="btn-success" style="width: 100%; padding: 12px;">💾 Save now</button>
                    <button id="btnRequantify" style="width: 100%; padding: 12px; background-color: #581c87; display: none;">⚙️ Re-run quantification on cell</button>
                </div>
            </div>
        </div>
    </main>

    <!-- Modal for Linkage Editing -->
    <div id="modalOverlay">
        <div id="modalContent">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="font-size: 1.25rem;">Replace Linkage for <span id="modalFilmName" style="color: var(--accent-primary);"></span></h2>
                <button onclick="closeModal()" style="background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer;">×</button>
            </div>
            <p style="color: var(--text-muted); margin-top: 8px;">Select the correct cell from the gallery below (showing T=0 crops).</p>
            <div id="modalGallery"></div>
        </div>
    </div>

    <script>
        let state = {
            experiments: [],
            films: [],
            sequences: [],
            cells: [],
            selectedExp: '',
            selectedSequence: '',
            isLocalEdit: false,
            prevGlobalCell: null,
            prevLinkEditFilmIdx: null,
            prevLinkEditFilmName: null,
            isEditingLink: false,
            linkEditFilmIdx: -1,
            linkEditFilmName: '',
            selectedCell: null,
            cellMasks: [],
            numFrames: 0,
            currentFrame: 0,
            channel: 'bf',
            viewMode: 'single',
            tool: 'select',
            brushSize: 10,
            isPlaying: false,
            playInterval: null,
            imgWidth: 2000,
            imgHeight: 2000,
            scale: 1.0,
            panX: 0,
            panY: 0,
            isPanning: false,
            isDrawing: false,
            startX: 0,
            startY: 0,
            drawingHistory: [],
            qc: {},
            suspicious: {},
            autosaveTimer: null,
            
            // Sequence specific
            linkageDetails: {}, // global_cell -> { film: local_id }
            filmBoundaries: [], // list of start frames for each film
            // Septum specific
            galleryClickMode: 'nav',
            lastActiveFilm: null,
            lastActiveCellId: null
        };

        const canvas = document.getElementById('imageCanvas');
        const ctx = canvas.getContext('2d');
        const canvasContainer = document.getElementById('canvasContainer');
        let activeModalFilmIdx = -1;
        let activeModalFilmName = "";

        window.addEventListener('DOMContentLoaded', async () => {
            await loadExperiments();
            setupEventListeners();
        });

        async function fetchQC() {
            let url = `/api/get_qc?experiment=${state.selectedExp}&sequence=${state.selectedSequence}`;
            const res = await fetch(url);
            const data = await res.json();
            state.qc = data.qc || {};
            updateQCUI();
        }

        async function setQC(status) {
            if (state.selectedCell === null) return;
            
            const currentStatus = state.qc[state.selectedCell];
            if (currentStatus === status) {
                status = 'pending';
            }
            
            const body = { experiment: state.selectedExp, cell_id: state.selectedCell, status: status, sequence: state.selectedSequence };
            const res = await fetch('/api/save_qc', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const data = await res.json();
            if (data.status === 'success') {
                if (status === 'pending') {
                    delete state.qc[state.selectedCell];
                } else {
                    state.qc[state.selectedCell] = status;
                }
                updateQCUI();
            } else alert("Failed to save QC");
        }

        function updateQCUI() {
            // Update cell list button colors based on QC
            state.cells.forEach(c => {
                const btn = document.getElementById(`cell-item-${c.global_id}`);
                if (btn) {
                    const st = state.qc[c.global_id] || 'pending';
                    btn.classList.remove('qc-good', 'qc-bad', 'qc-corrected', 'qc-review');
                    if (st !== 'pending') {
                        btn.classList.add(`qc-${st}`);
                    }
                    
                    const isSuspicious = !!(state.suspicious[c.global_id] && state.suspicious[c.global_id].length > 0);
                    if (isSuspicious) {
                        btn.classList.add('suspicious');
                    } else {
                        btn.classList.remove('suspicious');
                    }
                }
            });

            if (state.selectedCell === null) {
                document.getElementById('maskStatusText').innerText = 'None';
                document.getElementById('maskStatusText').className = 'status-badge badge-pending';
                return;
            }
            const status = state.qc[state.selectedCell] || 'pending';
            document.getElementById('maskStatusText').innerText = status.charAt(0).toUpperCase() + status.slice(1);
            
            let color = 'var(--text-muted)';
            let badgeClass = 'badge-pending';
            if (status === 'good') { color = '#10b981'; badgeClass = 'badge-good'; }
            if (status === 'bad') { color = '#ef4444'; badgeClass = 'badge-bad'; }
            if (status === 'corrected' || status === 'review') { color = '#f59e0b'; badgeClass = 'badge-pending'; }
            
            document.getElementById('maskStatusText').style.color = color;
            document.getElementById('maskStatusText').className = `status-badge ${badgeClass}`;
        }

        async function loadExperiments() {
            const res = await fetch('/api/list_experiments');
            const data = await res.json();
            state.experiments = data.experiments;
            
            const expSelect = document.getElementById('experimentSelect');
            expSelect.innerHTML = state.experiments.map(e => `<option value="${e}">${e}</option>`).join('');
            
            if (state.experiments.length > 0) {
                state.selectedExp = state.experiments[0];
                await loadFilmsAndSequences(state.selectedExp);
            }
        }

        async function loadFilmsAndSequences(exp) {
            const res = await fetch(`/api/list_films_and_sequences?experiment=${exp}`);
            const data = await res.json();
            state.sequences = data.sequences;
            
            const seqSelect = document.getElementById('sequenceSelect');
            if (state.sequences.length > 0) {
                seqSelect.innerHTML = state.sequences.map(s => `<option value="${s}">${s}</option>`).join('');
            } else {
                seqSelect.innerHTML = '<option value="">No datasets available</option>';
            }
            
            if (state.sequences.length > 0) {
                state.selectedSequence = state.sequences[0];
                await loadCells(exp, state.selectedSequence);
            }
        }

        async function loadCells(exp, target) {
            const res = await fetch(`/api/list_cells?experiment=${exp}&sequence=${target}`);
            const data = await res.json();
            state.cells = data.cells;
            state.lineageTree = data.lineage || {};
            
            state.qc = {};
            state.suspicious = {};
            
            const cellGrid = document.getElementById('cellGrid');
            cellGrid.innerHTML = state.cells.map(c => `
                <div class="cell-item" id="cell-item-${c.global_id}" onclick="selectCell('${c.global_id}')">${c.display_name}</div>
            `).join('');
            

            if (state.cells.length > 0) {
                await fetchQC();
                const exists = state.cells.some(c => c.global_id === state.selectedCell);
                if (exists && state.selectedCell !== null) {
                    await selectCell(state.selectedCell);
                } else {
                    await selectCell(state.cells[0].global_id);
                }
            } else {
                state.selectedCell = null;
                document.getElementById('cellIdLabel').innerText = 'None';
                document.getElementById('stripContainer').innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No cells available to display.</span>';
                document.getElementById('linkageList').innerHTML = '';
                document.getElementById('currentGlobalCellLabel').innerText = '';
                const canvas = document.getElementById('imageCanvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            // Asynchronously fetch suspicious cells to prevent blocking the UI
            fetch(`/api/suspicious_cells?experiment=${exp}&sequence=${target}`)
                .then(r => r.json())
                .then(suspData => {
                    state.suspicious = suspData.suspicious || {};
                    updateQCUI();
                    renderSuspiciousTicks();
                })
                .catch(e => console.error("Error fetching suspicious cells:", e));
        }

        function getActiveFilmAndLocalCell() {
            if (state.isLocalEdit) {
                return { film: state.localFilmId, cellId: state.selectedCell, filmIdx: 0 };
            }
            let fIdx = 0;
            if (state.filmBoundaries && state.filmBoundaries.length > 0) {
                for (let i = 0; i < state.filmBoundaries.length; i++) {
                    if (state.currentFrame >= state.filmBoundaries[i]) {
                        fIdx = i;
                    }
                }
            }
            const film = state.linkageDetails && state.linkageDetails.films ? state.linkageDetails.films[fIdx] : null;
            const cellId = state.linkageDetails && state.linkageDetails.local_ids ? state.linkageDetails.local_ids[fIdx] : null;
            return { film, cellId, filmIdx: fIdx };
        }

        async function selectCell(cellId) {
            if (state.selectedCell !== null) {
                const prevActive = document.getElementById(`cell-item-${state.selectedCell}`);
                if (prevActive) prevActive.classList.remove('active');
            }
            cancelAutosave();
            state.selectedCell = cellId;
            state.isEditingLink = false;
            state.isLocalEdit = false;
            document.getElementById('trackNewCellBtn').style.display = 'none';
            const exitBtn = document.getElementById('btnExitLocalEdit');
            if (exitBtn) exitBtn.style.display = 'none';
            document.getElementById('modeLabel').style.color = 'var(--accent-primary)';
            if (state.tool === 'select') document.getElementById('modeLabel').innerText = 'Click-Select';
            
            const activeItem = document.getElementById(`cell-item-${cellId}`);
            if (activeItem) activeItem.classList.add('active');
            
            document.getElementById('cellIdLabel').innerText = cellId;
            
            const modeParam = `sequence=${state.selectedSequence}`;
            const res = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&${modeParam}&cell_id=${cellId}`);
            const data = await res.json();
            
            state.cellMasks = data.masks;
            state.numFrames = data.num_frames;
            state.imgWidth = data.width;
            state.imgHeight = data.height;
            state.channel = data.track_channel;
            state.filmBoundaries = data.film_boundaries || [];
            state.localFilmId = data.local_film;
            
            state.linkageDetails = data.linkage_details;
            renderLinkageBoard();
            renderFilmBoundaries();
            
            document.getElementById('cellChannelLabel').innerText = data.track_channel.toUpperCase();
            
            const slider = document.getElementById('timeSlider');
            slider.max = state.numFrames - 1;
            slider.value = 0;
            state.currentFrame = 0;
            document.getElementById('maxTimeLabel').innerText = `t=${state.numFrames - 1}`;
            
            // Removed quantifyHpcGroup visibility check for single film mode
            
            updateChannelButtons();
            resetView();
            await displayFrame();
            updateQCUI();
            renderGallery();
            await loadSeptumLabels(cellId);
            renderSuspiciousTicks();
        }

        function renderLinkageBoard() {
            const list = document.getElementById('linkageList');
            document.getElementById('currentGlobalCellLabel').innerText = `(${state.selectedCell})`;
            let html = '';
            const films = state.linkageDetails.films;
            const local_ids = state.linkageDetails.local_ids;
            
            for (let i = 0; i < films.length; i++) {
                html += `
                <div class="linkage-row">
                    <div>
                        <div style="color: var(--text-muted); font-size: 0.75rem;">${films[i]}</div>
                        <div style="font-weight: 600;">Local Cell: ${local_ids[i]}</div>
                    </div>
                    <button onclick="openLinkageModal(${i}, '${films[i]}')" style="padding: 4px 8px; font-size: 0.75rem;">✏ Edit Link</button>
                </div>
                `;
            }
            list.innerHTML = html;
        }

        function renderFilmBoundaries() {
            const container = document.getElementById('filmBoundaries');
            container.innerHTML = '';
            if (state.isLocalEdit || !state.filmBoundaries || state.filmBoundaries.length === 0 || !state.linkageDetails) {
                return;
            }
            if (state.numFrames <= 1) return;
            
            const films = state.linkageDetails.films;
            for (let i = 0; i < state.filmBoundaries.length; i++) {
                const startFrame = state.filmBoundaries[i];
                const pct = (startFrame / (state.numFrames - 1)) * 100;
                
                const tick = document.createElement('div');
                tick.className = 'film-boundary-tick';
                tick.style.left = `calc(var(--tick-inset) + (100% - 2 * var(--tick-inset)) * ${pct / 100})`;
                
                const lbl = document.createElement('div');
                lbl.className = 'film-boundary-label';
                lbl.style.left = `calc(var(--tick-inset) + (100% - 2 * var(--tick-inset)) * ${pct / 100})`;
                lbl.innerText = films[i];
                
                container.appendChild(tick);
                container.appendChild(lbl);
            }
        }

        function renderSuspiciousTicks() {
            const container = document.getElementById('suspiciousTicks');
            if (!container) return;
            container.innerHTML = '';
            
            const cellId = state.selectedCell;
            const list = document.getElementById('suspiciousFrameList');
            const section = document.getElementById('suspiciousSection');
            
            if (cellId === null) {
                if (section) section.style.display = 'none';
                if (list) list.innerHTML = '';
                return;
            }
            
            const frames = state.suspicious[cellId] || [];
            
            if (frames.length === 0) {
                if (section) section.style.display = 'none';
                if (list) list.innerHTML = '';
                return;
            }
            
            if (section) section.style.display = 'block';
            if (list) {
                list.innerHTML = frames.map(t => 
                    `<button class="suspicious-frame-btn" onclick="goToFrame(${t})">t=${t}</button>`
                ).join('');
            }
            
            if (state.numFrames <= 1) return;
            
            frames.forEach(t => {
                const pct = (t / (state.numFrames - 1)) * 100;
                const tick = document.createElement('div');
                tick.className = 'suspicious-tick';
                tick.style.left = `calc(var(--tick-inset) + (100% - 2 * var(--tick-inset)) * ${pct / 100})`;
                container.appendChild(tick);
            });
        }

        window.goToFrame = function(t) {
            state.currentFrame = t;
            document.getElementById('timeSlider').value = t;
            displayFrame();
        }

        
        function openLinkageModal(filmIdx, filmName) {
            state.isEditingLink = true;
            state.linkEditFilmIdx = filmIdx;
            state.linkEditFilmName = filmName;
            
            // Jump to the start frame of this film
            let targetFrame = 0;
            if (filmIdx > 0 && state.filmBoundaries.length > filmIdx) {
                targetFrame = state.filmBoundaries[filmIdx];
            }
            state.currentFrame = targetFrame;
            document.getElementById('timeSlider').value = targetFrame;
            
            document.getElementById('modeLabel').innerText = `Pick Link Cell for ${filmName}`;
            document.getElementById('modeLabel').style.color = '#f59e0b';
            document.getElementById('trackNewCellBtn').style.display = 'block';
            
            displayFrame();
        }

        async function updateLinkage(newLocalId) {
            const body = {
                experiment: state.selectedExp,
                sequence: state.selectedSequence,
                global_cell: state.selectedCell,
                film_idx: state.linkEditFilmIdx,
                new_local_cell: newLocalId
            };
            const res = await fetch('/api/update_linkage', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            if (data.status === 'success') {
                state.isEditingLink = false;
                document.getElementById('modeLabel').innerText = 'Click-Select';
                document.getElementById('modeLabel').style.color = 'var(--accent-primary)';
                selectCell(state.selectedCell);
            } else {
                alert("Failed to update linkage: " + data.message);
            }
        }

        // The rest of the JS functions are mostly identical, passing modeParam everywhere.
        function updateChannelButtons() {
            document.getElementById('chanBfBtn').classList.toggle('active', state.channel === 'bf');
            document.getElementById('chanGfpBtn').classList.toggle('active', state.channel === 'gfp');
        }

        function updateGalleryClickModeButtons() {
            document.getElementById('galleryClickNavBtn').classList.toggle('active', state.galleryClickMode === 'nav');
            document.getElementById('galleryClickStartBtn').classList.toggle('active', state.galleryClickMode === 'start1');
            document.getElementById('galleryClickEndBtn').classList.toggle('active', state.galleryClickMode === 'end1');
            document.getElementById('galleryClickStartBtn2').classList.toggle('active', state.galleryClickMode === 'start2');
            document.getElementById('galleryClickEndBtn2').classList.toggle('active', state.galleryClickMode === 'end2');
        }


        function resetView() {
            state.scale = canvasContainer.clientHeight / state.imgHeight * 0.95;
            state.panX = (canvasContainer.clientWidth - state.imgWidth * state.scale) / 2;
            state.panY = (canvasContainer.clientHeight - state.imgHeight * state.scale) / 2;
            updateTransformLabels();
        }

        function updateTransformLabels() {
            document.getElementById('scaleLabel').innerText = state.scale.toFixed(1);
            document.getElementById('panXLabel').innerText = Math.round(state.panX);
            document.getElementById('panYLabel').innerText = Math.round(state.panY);
        }

        async function displayFrame() {
            document.getElementById('currentTimeLabel').innerText = `t=${state.currentFrame}`;
            updateGalleryHighlight();
            
            // Reload septum labels if active film or local cell ID changes
            const current = getActiveFilmAndLocalCell();
            if (current.film !== state.lastActiveFilm || current.cellId !== state.lastActiveCellId) {
                state.lastActiveFilm = current.film;
                state.lastActiveCellId = current.cellId;
                await loadSeptumLabels(state.selectedCell);
            }

            
            // Update local film label
            if (state.isLocalEdit) {
                document.getElementById('localFilmLabel').innerText = `${state.localFilmId} (Local Edit)`;
            } else if (state.filmBoundaries.length > 0) {
                let fIdx = 0;
                for (let i = 0; i < state.filmBoundaries.length; i++) {
                    if (state.currentFrame >= state.filmBoundaries[i]) {
                        fIdx = i;
                    }
                }
                const films = state.linkageDetails.films;
                const local_ids = state.linkageDetails.local_ids;
                document.getElementById('localFilmLabel').innerText = `${films[fIdx]} (Cell ${local_ids[fIdx]})`;
            }
            
            const modeParam = state.isLocalEdit ? `film=${state.localFilmId}` : `sequence=${state.selectedSequence}`;
            
            const loadImage = (src) => new Promise((resolve) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => resolve(null);
                img.src = src;
            });
            
            const ts = Date.now();
            if (state.viewMode === 'population') {
                const popUrl = `/api/population_frame?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${state.currentFrame}&_ts=${ts}`;
                const img = await loadImage(popUrl);
                if (!img) return;
                
                canvas.width = state.imgWidth;
                canvas.height = state.imgHeight;
                canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                return;
            }
            
            const imgUrl = `/api/frame_image?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${state.currentFrame}&channel=${state.channel}&_ts=${ts}`;
            const promises = [loadImage(imgUrl)];
            
            if (state.tool === 'select') {
                const boundsUrl = `/api/frame_boundaries?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${state.currentFrame}&_ts=${ts}`;
                promises.push(loadImage(boundsUrl));
            } else {
                promises.push(Promise.resolve(null));
            }
            
            const [img, boundariesImg] = await Promise.all(promises);
            
            if (!img) return;
            
            canvas.width = state.imgWidth;
            canvas.height = state.imgHeight;
            canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            
            if (boundariesImg) {
                ctx.drawImage(boundariesImg, 0, 0);
            }
            
            drawMask();
        }

        function drawMask() {
            const currentRle = state.cellMasks[state.currentFrame];
            if (!currentRle) return;
            const maskArr = decodeRle(currentRle, state.imgWidth, state.imgHeight);
            const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imgData.data;
            for (let i = 0; i < maskArr.length; i++) {
                if (maskArr[i]) {
                    const idx = i * 4;
                    data[idx] = Math.min(255, data[idx] + 20);      // R
                    data[idx + 1] = Math.min(255, data[idx + 1] + 160); // G
                    data[idx + 2] = Math.min(255, data[idx + 2] + 40);  // B
                }
            }
            ctx.putImageData(imgData, 0, 0);
        }

        function decodeRle(rleStr, W, H) {
            const flat = new Uint8Array(W * H);
            if (!rleStr || rleStr.trim() === "") return flat;
            const nums = rleStr.trim().split(/\s+/).map(Number);
            for (let i = 0; i < nums.length; i += 2) {
                const start = nums[i] - 1;
                const length = nums[i + 1];
                for (let j = 0; j < length; j++) {
                    const idx = start + j;
                    if (idx < flat.length) flat[idx] = 1;
                }
            }
            const rowMajor = new Uint8Array(W * H);
            let k = 0;
            for (let x = 0; x < W; x++) {
                for (let y = 0; y < H; y++) {
                    rowMajor[y * W + x] = flat[k++];
                }
            }
            return rowMajor;
        }

        function rleEncode(rowMajorArr, W, H) {
            const flat = new Uint8Array(W * H);
            let k = 0;
            for (let x = 0; x < W; x++) {
                for (let y = 0; y < H; y++) {
                    flat[k++] = rowMajorArr[y * W + x];
                }
            }
            const starts = [];
            const lengths = [];
            let prev = 0;
            for (let i = 0; i <= flat.length; i++) {
                const v = (i < flat.length) ? flat[i] : 0;
                const diff = v - prev;
                if (diff === 1) starts.push(i + 1);
                else if (diff === -1) lengths.push(i + 1 - starts[starts.length - 1]);
                prev = v;
            }
            const out = [];
            for (let i = 0; i < starts.length; i++) {
                out.push(starts[i]);
                out.push(lengths[i]);
            }
            return out.join(" ");
        }

        function getCanvasMouseCoords(e) {
            const containerRect = canvasContainer.getBoundingClientRect();
            const mx = e.clientX - containerRect.left;
            const my = e.clientY - containerRect.top;
            const mouseX = Math.round((mx - state.panX) / state.scale);
            const mouseY = Math.round((my - state.panY) / state.scale);
            return { x: mouseX, y: mouseY };
        }

        function setupEventListeners() {
            const slider = document.getElementById('timeSlider');
            slider.addEventListener('input', (e) => {
                state.currentFrame = parseInt(e.target.value);
                displayFrame();
            });

            document.getElementById('prevBtn').onclick = () => {
                if (state.currentFrame > 0) {
                    state.currentFrame--;
                    document.getElementById('timeSlider').value = state.currentFrame;
                    displayFrame();
                }
            };
            document.getElementById('nextBtn').onclick = () => {
                if (state.currentFrame < state.numFrames - 1) {
                    state.currentFrame++;
                    document.getElementById('timeSlider').value = state.currentFrame;
                    displayFrame();
                }
            };

            document.getElementById('playBtn').onclick = () => {
                const btn = document.getElementById('playBtn');
                if (state.isPlaying) {
                    clearInterval(state.playInterval);
                    state.isPlaying = false;
                    btn.innerText = 'Play';
                } else {
                    state.isPlaying = true;
                    btn.innerText = 'Pause';
                    state.playInterval = setInterval(() => {
                        if (state.currentFrame < state.numFrames - 1) {
                            state.currentFrame++;
                            document.getElementById('timeSlider').value = state.currentFrame;
                            displayFrame();
                        } else {
                            state.currentFrame = 0;
                            document.getElementById('timeSlider').value = 0;
                            displayFrame();
                        }
                    }, 150);
                }
            };

            document.getElementById('experimentSelect').onchange = async (e) => {
                state.selectedExp = e.target.value;
                await loadFilmsAndSequences(state.selectedExp);
            };

            document.getElementById('sequenceSelect').onchange = (e) => {
                state.selectedSequence = e.target.value;
                loadCells(state.selectedExp, state.selectedSequence);
            };
            
            document.getElementById('trackNewCellBtn').onclick = async () => {
                const targetFilm = state.isEditingLink ? state.linkEditFilmName : state.localFilmId;
                if (!targetFilm) return alert("Select a film first.");
                
                try {
                    const res = await fetch('/api/create_new_cell', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            experiment: state.selectedExp,
                            film: targetFilm
                        })
                    });
                    const data = await res.json();
                    if (data.status === 'success') {
                        state.prevGlobalCell = state.selectedCell;
                        state.prevLinkEditFilmIdx = state.linkEditFilmIdx;
                        state.prevLinkEditFilmName = state.linkEditFilmName;
                        state.isEditingLink = false;
                        
                        await selectLocalCell(data.cell_id, targetFilm);
                        
                        alert(`Created custom local cell #${data.cell_id} in ${targetFilm}.\n\nInstructions:\n1. Paint the seed mask on Frame 0 using the Brush tool.\n2. Click "Save Changes" (or wait for autosave).\n3. Click "💻 Quantify Locally (Seed from CSV)" to track the cell.\n4. Once finished, click "🔙 Return to Linkage" to link it!`);
                    } else {
                        alert(data.message);
                    }
                } catch(e) {
                    alert(e);
                }
            };

            document.getElementById('btnQuantifyHpc').onclick = async () => {
                if (!state.selectedCell) return alert("Select a cell first.");
                
                let targetFilm, targetCellId;
                if (state.isLocalEdit) {
                    targetFilm = state.localFilmId;
                    targetCellId = state.selectedCell;
                } else {
                    const activeInfo = getActiveFilmAndLocalCell();
                    if (!activeInfo || !activeInfo.film || activeInfo.cellId === null || activeInfo.cellId === -1) {
                        return alert("The selected cell is not tracked or mapped in the currently viewed film.");
                    }
                    targetFilm = activeInfo.film;
                    targetCellId = activeInfo.cellId;
                }
                
                document.getElementById('btnQuantifyHpc').innerText = '⏳ Quantifying...';
                document.getElementById('btnQuantifyHpc').disabled = true;
                
                try {
                    const res = await fetch('/api/quantify_on_hpc', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            experiment: state.selectedExp,
                            film: targetFilm,
                            label_id: targetCellId,
                            seed_from_csv: true,
                            track_channel: state.channel
                        })
                    });
                    const data = await res.json();
                    alert(data.message);
                    if (data.status === 'success') {
                        if (state.isLocalEdit) {
                            const masksRes = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&film=${targetFilm}&cell_id=${targetCellId}`);
                            const masksData = await masksRes.json();
                            state.cellMasks = masksData.masks;
                        } else {
                            await loadCells(state.selectedExp, state.selectedSequence);
                            const curBtn = document.getElementById(`cell-item-${state.selectedCell}`);
                            if (curBtn) curBtn.click();
                        }
                        displayFrame();
                    }
                } catch (e) {
                    alert(e);
                } finally {
                    document.getElementById('btnQuantifyHpc').innerText = '💻 Quantify Locally (Seed from CSV)';
                    document.getElementById('btnQuantifyHpc').disabled = false;
                }
            };

            document.getElementById('btnExitLocalEdit').onclick = async () => {
                await exitLocalCellEdit();
            };

            document.getElementById('chanBfBtn').onclick = () => { state.channel = 'bf'; updateChannelButtons(); displayFrame(); renderGallery(); };
            document.getElementById('chanGfpBtn').onclick = () => { state.channel = 'gfp'; updateChannelButtons(); displayFrame(); renderGallery(); };

            document.getElementById('viewModeBtn').onclick = () => {
                state.viewMode = state.viewMode === 'single' ? 'population' : 'single';
                const btn = document.getElementById('viewModeBtn');
                if (state.viewMode === 'population') {
                    btn.innerText = 'Population';
                    btn.style.backgroundColor = 'var(--accent-primary)';
                    selectTool('select');
                    document.getElementById('toolBrushBtn').disabled = true;
                    document.getElementById('toolEraserBtn').disabled = true;
                    document.getElementById('toolBrushBtn').style.opacity = 0.5;
                    document.getElementById('toolEraserBtn').style.opacity = 0.5;
                } else {
                    btn.innerText = 'Single Cell';
                    btn.style.backgroundColor = '#581c87';
                    document.getElementById('toolBrushBtn').disabled = false;
                    document.getElementById('toolEraserBtn').disabled = false;
                    document.getElementById('toolBrushBtn').style.opacity = 1.0;
                    document.getElementById('toolEraserBtn').style.opacity = 1.0;
                }
                displayFrame();
            };

            const toolBtns = {
                select: document.getElementById('toolSelectBtn'),
                brush: document.getElementById('toolBrushBtn'),
                eraser: document.getElementById('toolEraserBtn')
            };

            const selectTool = (t) => {
                state.tool = t;
                // Cancel link edit mode if they switch tools to draw
                if (state.isEditingLink) {
                    state.isEditingLink = false;
                    document.getElementById('trackNewCellBtn').style.display = 'none';
                    document.getElementById('modeLabel').innerText = t === 'select' ? 'Click-Select' : (t === 'brush' ? 'Brush Draw' : 'Eraser');
                    document.getElementById('modeLabel').style.color = 'var(--accent-primary)';
                } else {
                    document.getElementById('modeLabel').innerText = t === 'select' ? 'Click-Select' : (t === 'brush' ? 'Brush Draw' : 'Eraser');
                }
                
                Object.keys(toolBtns).forEach(k => toolBtns[k].classList.toggle('active', k === t));
                document.getElementById('brushControls').style.display = (t === 'select' ? 'none' : 'flex');
                displayFrame();
            };

            toolBtns.select.onclick = () => selectTool('select');
            toolBtns.brush.onclick = () => selectTool('brush');
            toolBtns.eraser.onclick = () => selectTool('eraser');

            document.getElementById('brushSizeSlider').addEventListener('input', (e) => {
                state.brushSize = parseInt(e.target.value);
                document.getElementById('brushSizeLabel').innerText = `${state.brushSize}px`;
            });

            document.getElementById('undoBtn').onclick = undoStroke;
            document.getElementById('usePrevSegmentBtn').onclick = () => {
                if (state.currentFrame > 0 && state.cellMasks && state.cellMasks.length > 0) {
                    const prevRle = state.cellMasks[state.currentFrame - 1] || "";
                    state.drawingHistory.push(state.cellMasks[state.currentFrame] || "");
                    if (state.drawingHistory.length > 20) state.drawingHistory.shift();
                    
                    state.cellMasks[state.currentFrame] = prevRle;
                    displayFrame();
                    markDirty();
                } else {
                    alert("Cannot copy. No previous frame mask exists.");
                }
            };
            document.getElementById('clearBtn').onclick = () => {
                state.cellMasks[state.currentFrame] = "";
                displayFrame();
                markDirty();
            };

            document.getElementById('btnSaveCell').onclick = () => saveCorrectedMasks(false);

            document.getElementById('btnMarkGood').onclick = () => setQC('good');
            document.getElementById('btnMarkBad').onclick = () => setQC('bad');
            document.getElementById('btnMarkCorrected').onclick = () => setQC('corrected');

            document.getElementById('hasSeptumCheckbox').onchange = (e) => {
                document.getElementById('divisionIntervalContainer').style.display = e.target.checked ? 'flex' : 'none';
                saveSeptumLabels();
            };
            document.getElementById('septumStartInput').onchange = () => saveSeptumLabels();
            document.getElementById('septumEndInput').onchange = () => saveSeptumLabels();
            document.getElementById('whiteSeptumCheckbox').onchange = () => saveSeptumLabels();
            document.getElementById('predictSeptumBtn').onclick = () => runSeptumAi();
            
            document.getElementById('setSeptumStartBtn').onclick = () => {
                document.getElementById('septumStartInput').value = state.currentFrame;
                saveSeptumLabels();
            };
            
            document.getElementById('setSeptumEndBtn').onclick = () => {
                document.getElementById('septumEndInput').value = state.currentFrame;
                saveSeptumLabels();
            };
            
            document.getElementById('hasSeptumCheckbox2').onchange = (e) => {
                document.getElementById('divisionIntervalContainer2').style.display = e.target.checked ? 'flex' : 'none';
                saveSeptumLabels();
            };
            document.getElementById('septumStartInput2').onchange = () => saveSeptumLabels();
            document.getElementById('septumEndInput2').onchange = () => saveSeptumLabels();
            document.getElementById('whiteSeptumCheckbox2').onchange = () => saveSeptumLabels();
            
            document.getElementById('setSeptumStartBtn2').onclick = () => {
                document.getElementById('septumStartInput2').value = state.currentFrame;
                saveSeptumLabels();
            };
            
            document.getElementById('setSeptumEndBtn2').onclick = () => {
                document.getElementById('septumEndInput2').value = state.currentFrame;
                saveSeptumLabels();
            };

            document.getElementById('galleryClickNavBtn').onclick = () => {
                state.galleryClickMode = 'nav';
                document.getElementById('galleryClickNavBtn').classList.add('active');
                document.getElementById('galleryClickStartBtn').classList.remove('active');
                document.getElementById('galleryClickEndBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn2').classList.remove('active');
                document.getElementById('galleryClickEndBtn2').classList.remove('active');
            };
            document.getElementById('galleryClickStartBtn').onclick = () => {
                state.galleryClickMode = 'start1';
                document.getElementById('galleryClickNavBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn').classList.add('active');
                document.getElementById('galleryClickEndBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn2').classList.remove('active');
                document.getElementById('galleryClickEndBtn2').classList.remove('active');
            };
            document.getElementById('galleryClickEndBtn').onclick = () => {
                state.galleryClickMode = 'end1';
                document.getElementById('galleryClickNavBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn').classList.remove('active');
                document.getElementById('galleryClickEndBtn').classList.add('active');
                document.getElementById('galleryClickStartBtn2').classList.remove('active');
                document.getElementById('galleryClickEndBtn2').classList.remove('active');
            };
            document.getElementById('galleryClickStartBtn2').onclick = () => {
                state.galleryClickMode = 'start2';
                document.getElementById('galleryClickNavBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn').classList.remove('active');
                document.getElementById('galleryClickEndBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn2').classList.add('active');
                document.getElementById('galleryClickEndBtn2').classList.remove('active');
            };
            document.getElementById('galleryClickEndBtn2').onclick = () => {
                state.galleryClickMode = 'end2';
                document.getElementById('galleryClickNavBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn').classList.remove('active');
                document.getElementById('galleryClickEndBtn').classList.remove('active');
                document.getElementById('galleryClickStartBtn2').classList.remove('active');
                document.getElementById('galleryClickEndBtn2').classList.add('active');
            };

            document.getElementById('setAutofixStartBtn').onclick = () => {
                document.getElementById('autofixStartInput').value = state.currentFrame;
            };
            document.getElementById('setAutofixEndBtn').onclick = () => {
                document.getElementById('autofixEndInput').value = state.currentFrame;
            };
            document.getElementById('runAutofixBtn').onclick = () => runAutofix();

            window.addEventListener('keydown', (e) => {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
                if (e.key === ' ') { e.preventDefault(); document.getElementById('playBtn').click(); }
                else if (e.key === 'ArrowRight') document.getElementById('nextBtn').click();
                else if (e.key === 'ArrowLeft') document.getElementById('prevBtn').click();
                else if (e.key.toLowerCase() === 's') selectTool('select');
                else if (e.key.toLowerCase() === 'b') selectTool('brush');
                else if (e.key.toLowerCase() === 'e') selectTool('eraser');
                else if (e.key.toLowerCase() === 'p') document.getElementById('usePrevSegmentBtn').click();
                else if (e.key.toLowerCase() === 'z' && e.ctrlKey) undoStroke();
            });

            canvasContainer.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomIntensity = 0.1;
                const containerRect = canvasContainer.getBoundingClientRect();
                const mx = e.clientX - containerRect.left;
                const my = e.clientY - containerRect.top;
                const canvasMouseX = (mx - state.panX) / state.scale;
                const canvasMouseY = (my - state.panY) / state.scale;
                const zoomFactor = e.deltaY < 0 ? (1 + zoomIntensity) : (1 - zoomIntensity);
                state.scale = Math.min(20, Math.max(0.1, state.scale * zoomFactor));
                state.panX = mx - canvasMouseX * state.scale;
                state.panY = my - canvasMouseY * state.scale;
                canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
                updateTransformLabels();
            });

            
            canvasContainer.addEventListener('mousedown', (e) => {
                const coords = getCanvasMouseCoords(e);
                if (e.ctrlKey || e.button === 1 || state.tool === 'pan') {
                    state.isPanning = true;
                    state.startX = e.clientX - state.panX;
                    state.startY = e.clientY - state.panY;
                } else if (state.isEditingLink) {
                    // We are picking a cell to link!
                    identifyAndLinkCell(coords.x, coords.y);
                } else if (state.isAddingDaughter) {
                    identifyAndAddDaughter(coords.x, coords.y);
                } else if (state.tool === 'select') {
                    clickSelectSegment(coords.x, coords.y, e.shiftKey);
                } else if (state.tool === 'brush' || state.tool === 'eraser') {
                    state.isDrawing = true;
                    state.drawingHistory.push(state.cellMasks[state.currentFrame] || "");
                    if (state.drawingHistory.length > 20) state.drawingHistory.shift();
                    drawStroke(coords.x, coords.y, state.tool === 'brush');
                }
            });

            window.addEventListener('mousemove', (e) => {
                if (state.isPanning) {
                    state.panX = e.clientX - state.startX;
                    state.panY = e.clientY - state.startY;
                    canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
                    updateTransformLabels();
                } else if (state.isDrawing) {
                    const coords = getCanvasMouseCoords(e);
                    drawStroke(coords.x, coords.y, state.tool === 'brush');
                }
            });

            window.addEventListener('mouseup', () => {
                state.isPanning = false;
                state.isDrawing = false;
            });
        }

        
        async function identifyAndLinkCell(x, y) {
            document.getElementById('modeLabel').innerText = 'Searching...';
            let localT = 0;
            if (state.filmBoundaries && state.filmBoundaries.length > state.linkEditFilmIdx) {
                localT = state.currentFrame - state.filmBoundaries[state.linkEditFilmIdx];
                if (localT < 0) localT = 0;
            } else {
                localT = state.currentFrame;
            }
            const body = {
                experiment: state.selectedExp,
                film: state.linkEditFilmName,
                t: localT,
                x: x,
                y: y
            };
            const res = await fetch('/api/identify_cell', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            if (data.status === 'success') {
                updateLinkage(data.cell_id);
            } else if (data.status === 'untracked') {
                const conf = confirm(`This cell (Label #${data.label_id}) is not currently tracked. Do you want to quantify it locally on the laptop now? (This will take a moment)`);
                if (conf) {
                    await quantifyCellLocally(data.label_id);
                } else {
                    document.getElementById('modeLabel').innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
                }
            } else {
                alert(data.message);
                document.getElementById('modeLabel').innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
            }
        }
        
        async function quantifyCellLocally(labelId) {
            document.getElementById('modeLabel').innerText = `Quantifying Cell #${labelId} locally. Please wait...`;
            const body = {
                experiment: state.selectedExp,
                film: state.linkEditFilmName,
                label_id: labelId
            };
            try {
                const res = await fetch('/api/quantify_on_hpc', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    alert(data.message);
                    // Reload cells list to show the new cell
                    await loadCells(state.selectedExp, state.selectedSequence);
                    // Select it automatically in linkage map
                    updateLinkage(labelId);
                } else {
                    alert('Quantification Error: ' + data.message);
                    document.getElementById('modeLabel').innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
                }
            } catch (e) {
                alert('Error connecting to backend: ' + e);
                document.getElementById('modeLabel').innerText = `Pick Link Cell for ${state.linkEditFilmName}`;
            }
        }

        async function selectLocalCell(cellId, filmId) {
            cancelAutosave();
            state.selectedCell = cellId;
            state.localFilmId = filmId;
            state.isLocalEdit = true;
            state.isEditingLink = false; // Disable link editing mode temporarily to allow brush drawing
            
            // Set UI headers
            document.getElementById('modeLabel').innerText = `Editing Local Cell #${cellId} in ${filmId}`;
            document.getElementById('modeLabel').style.color = '#10b981';
            
            // Fetch local cell masks
            const res = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&film=${filmId}&cell_id=${cellId}`);
            const data = await res.json();
            
            state.cellMasks = data.masks;
            state.numFrames = data.num_frames;
            state.imgWidth = data.width;
            state.imgHeight = data.height;
            state.channel = data.track_channel;
            state.filmBoundaries = []; // No boundaries within a single film
            renderFilmBoundaries(); // Clear timeline boundaries
            
            document.getElementById('cellChannelLabel').innerText = data.track_channel.toUpperCase();
            document.getElementById('cellIdLabel').innerText = `Local #${cellId}`;
            
            const slider = document.getElementById('timeSlider');
            slider.max = state.numFrames - 1;
            slider.value = 0;
            state.currentFrame = 0;
            document.getElementById('maxTimeLabel').innerText = `t=${state.numFrames - 1}`;
            
            // Hide global-only sections and show local-only sections
            document.getElementById('linkageSection').style.display = 'none';
            document.getElementById('localFilmRow').style.display = 'flex';
            document.getElementById('localFilmLabel').innerText = `${filmId} (Local Edit)`;
            
            // Show quantify locally button
            document.getElementById('quantifyHpcGroup').style.display = 'flex';
            document.getElementById('trackNewCellBtn').style.display = 'none';
            
            const exitBtn = document.getElementById('btnExitLocalEdit');
            if (exitBtn) exitBtn.style.display = 'block';
            
            updateChannelButtons();
            resetView();
            await displayFrame();
            updateQCUI();
            renderGallery();
            await loadSeptumLabels(cellId);
        }

        async function exitLocalCellEdit() {
            cancelAutosave();
            state.isLocalEdit = false;
            
            // Restore UI sections
            document.getElementById('linkageSection').style.display = 'block';
            
            const exitBtn = document.getElementById('btnExitLocalEdit');
            if (exitBtn) exitBtn.style.display = 'none';
            
            // Hide quantify locally button
            document.getElementById('quantifyHpcGroup').style.display = 'none';
            
            // Re-select the global cell
            if (state.prevGlobalCell) {
                const globalCellId = state.prevGlobalCell;
                state.prevGlobalCell = null;
                await selectCell(globalCellId);
                // Also trigger edit linkage modal for the film again so they can link it
                if (state.prevLinkEditFilmIdx !== null && state.prevLinkEditFilmIdx !== undefined && state.prevLinkEditFilmName) {
                    openLinkageModal(state.prevLinkEditFilmIdx, state.prevLinkEditFilmName);
                }
            } else {
                // Fallback
                await loadCells(state.selectedExp, state.selectedSequence);
            }
        }

        function undoStroke() {
            if (state.drawingHistory.length > 0) {
                state.cellMasks[state.currentFrame] = state.drawingHistory.pop();
                displayFrame();
                markDirty();
            }
        }

        async function loadSeptumLabels(cellId) {
            if (!cellId) return;
            const current = getActiveFilmAndLocalCell();
            if (!current.film || current.cellId === -1 || current.cellId === undefined || current.cellId === null) {
                document.getElementById('divisionSection').style.display = 'none';
                return;
            }
            
            document.getElementById('divisionSection').style.display = 'block';
            document.getElementById('septumAiChart').style.display = 'none'; // reset prediction chart
            
            try {
                if (state.isLocalEdit) {
                    const res = await fetch(`/api/get_septum_label?experiment=${state.selectedExp}&film=${current.film}&cell_id=${current.cellId}`);
                    const d = await res.json();
                    if (d.status === 'success') {
                        document.getElementById('hasSeptumCheckbox').checked = d.has_septum;
                        document.getElementById('septumStartInput').value = (d.local_start !== null && d.local_start !== -1) ? d.local_start : '';
                        document.getElementById('septumEndInput').value   = (d.local_end   !== null && d.local_end   !== -1) ? d.local_end   : '';
                        document.getElementById('whiteSeptumCheckbox').checked = d.white_septum;
                        document.getElementById('divisionIntervalContainer').style.display = d.has_septum ? 'flex' : 'none';

                        document.getElementById('hasSeptumCheckbox2').checked = d.has_septum_2;
                        document.getElementById('septumStartInput2').value = (d.local_start_2 !== null && d.local_start_2 !== -1) ? d.local_start_2 : '';
                        document.getElementById('septumEndInput2').value   = (d.local_end_2   !== null && d.local_end_2   !== -1) ? d.local_end_2   : '';
                        document.getElementById('whiteSeptumCheckbox2').checked = d.white_septum_2;
                        document.getElementById('divisionIntervalContainer2').style.display = d.has_septum_2 ? 'flex' : 'none';
                    }
                } else if (state.filmBoundaries && state.filmBoundaries.length > 0 && state.linkageDetails) {
                    const boundaries = state.filmBoundaries;
                    const films      = state.linkageDetails.films;
                    const localIds   = state.linkageDetails.local_ids;
                    
                    let mergedHasSeptum  = false;
                    let mergedGlobalStart = null;
                    let mergedGlobalEnd   = null;
                    let mergedWhite       = false;
                    
                    let mergedHasSeptum2  = false;
                    let mergedGlobalStart2 = null;
                    let mergedGlobalEnd2   = null;
                    let mergedWhite2       = false;
                    
                    for (let fi = 0; fi < films.length; fi++) {
                        const film  = films[fi];
                        const lcid  = localIds[fi];
                        if (!film || lcid === -1 || lcid === undefined || lcid === null) continue;
                        
                        const fb = boundaries[fi] || 0;
                        const res = await fetch(`/api/get_septum_label?experiment=${state.selectedExp}&film=${film}&cell_id=${lcid}`);
                        const d = await res.json();
                        if (d.status !== 'success') continue;
                        
                        if (d.has_septum) {
                            mergedHasSeptum = true;
                            if (d.local_start !== null && d.local_start !== undefined && d.local_start !== -1) {
                                const gs = d.local_start + fb;
                                if (mergedGlobalStart === null || gs < mergedGlobalStart) mergedGlobalStart = gs;
                            }
                            if (d.local_end !== null && d.local_end !== undefined && d.local_end !== -1) {
                                const ge = d.local_end + fb;
                                if (mergedGlobalEnd === null || ge > mergedGlobalEnd) mergedGlobalEnd = ge;
                            }
                            if (d.white_septum) mergedWhite = true;
                        }
                        
                        if (d.has_septum_2) {
                            mergedHasSeptum2 = true;
                            if (d.local_start_2 !== null && d.local_start_2 !== undefined && d.local_start_2 !== -1) {
                                const gs = d.local_start_2 + fb;
                                if (mergedGlobalStart2 === null || gs < mergedGlobalStart2) mergedGlobalStart2 = gs;
                            }
                            if (d.local_end_2 !== null && d.local_end_2 !== undefined && d.local_end_2 !== -1) {
                                const ge = d.local_end_2 + fb;
                                if (mergedGlobalEnd2 === null || ge > mergedGlobalEnd2) mergedGlobalEnd2 = ge;
                            }
                            if (d.white_septum_2) mergedWhite2 = true;
                        }
                    }
                    
                    // Display merged global frames
                    document.getElementById('hasSeptumCheckbox').checked = mergedHasSeptum;
                    document.getElementById('septumStartInput').value = mergedGlobalStart !== null ? mergedGlobalStart : '';
                    document.getElementById('septumEndInput').value   = mergedGlobalEnd   !== null ? mergedGlobalEnd   : '';
                    document.getElementById('whiteSeptumCheckbox').checked = mergedWhite;
                    document.getElementById('divisionIntervalContainer').style.display = mergedHasSeptum ? 'flex' : 'none';

                    document.getElementById('hasSeptumCheckbox2').checked = mergedHasSeptum2;
                    document.getElementById('septumStartInput2').value = mergedGlobalStart2 !== null ? mergedGlobalStart2 : '';
                    document.getElementById('septumEndInput2').value   = mergedGlobalEnd2   !== null ? mergedGlobalEnd2   : '';
                    document.getElementById('whiteSeptumCheckbox2').checked = mergedWhite2;
                    document.getElementById('divisionIntervalContainer2').style.display = mergedHasSeptum2 ? 'flex' : 'none';
                }
            } catch (e) {
                console.error("Error loading septum labels: ", e);
            }
        }


        async function saveSeptumLabels() {
            const cellId = state.selectedCell;
            if (!cellId) return;
            const current = getActiveFilmAndLocalCell();
            if (!current.film || current.cellId === -1 || current.cellId === undefined || current.cellId === null) return;
            
            const has_septum = document.getElementById('hasSeptumCheckbox').checked;
            const startVal = document.getElementById('septumStartInput').value;
            const endVal = document.getElementById('septumEndInput').value;
            const white_septum = document.getElementById('whiteSeptumCheckbox').checked;
            
            const has_septum_2 = document.getElementById('hasSeptumCheckbox2').checked;
            const startVal2 = document.getElementById('septumStartInput2').value;
            const endVal2 = document.getElementById('septumEndInput2').value;
            const white_septum_2 = document.getElementById('whiteSeptumCheckbox2').checked;
            
            const globalStart = startVal !== '' ? parseInt(startVal) : null;
            const globalEnd   = endVal   !== '' ? parseInt(endVal)   : null;
            const globalStart2 = startVal2 !== '' ? parseInt(startVal2) : null;
            const globalEnd2   = endVal2   !== '' ? parseInt(endVal2)   : null;
            
            // We may have cross-film start/end.
            // Dispatch one save call per film, keeping only the point(s) that belong to it.
            if (state.isLocalEdit) {
                const body = {
                    experiment: state.selectedExp,
                    film: current.film,
                    cell_id: current.cellId,
                    has_septum: has_septum,
                    local_start: globalStart,
                    local_end:   globalEnd,
                    white_septum,
                    has_septum_2: has_septum_2,
                    local_start_2: globalStart2,
                    local_end_2:   globalEnd2,
                    white_septum_2
                };
                
                document.getElementById('autosaveStatus').innerText = 'Saving...';
                document.getElementById('autosaveStatus').style.background = '#0f5132';
                
                try {
                    const res = await fetch('/api/save_septum_label', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(body)
                    });
                    const d = await res.json();
                    if (d.status === 'success') {
                        document.getElementById('autosaveStatus').innerText = 'Saved';
                        document.getElementById('autosaveStatus').style.background = '#0f5132';
                        renderGallery();
                        setTimeout(() => {
                            if (document.getElementById('autosaveStatus').innerText === 'Saved') {
                                document.getElementById('autosaveStatus').innerText = 'Idle';
                                document.getElementById('autosaveStatus').style.background = '#1e293b';
                            }
                        }, 1000);
                    } else {
                        document.getElementById('autosaveStatus').innerText = 'Error';
                        document.getElementById('autosaveStatus').style.background = '#ef4444';
                    }
                } catch(e) {
                    document.getElementById('autosaveStatus').innerText = 'Error';
                    document.getElementById('autosaveStatus').style.background = '#ef4444';
                }
            } else if (state.filmBoundaries && state.filmBoundaries.length > 0 && state.linkageDetails) {
                
                const boundaries = state.filmBoundaries;  // global offset of each film
                const films      = state.linkageDetails.films;
                const localIds   = state.linkageDetails.local_ids;
                
                // Helper: which film index owns a given global frame?
                function filmIdxForGlobal(g) {
                    let idx = 0;
                    for (let i = 0; i < boundaries.length; i++) {
                        if (g >= boundaries[i]) idx = i;
                    }
                    return idx;
                }
                
                // Collect the set of film indices that are affected
                const affectedIdx = new Set();
                if (has_septum) {
                    if (globalStart !== null) affectedIdx.add(filmIdxForGlobal(globalStart));
                    if (globalEnd   !== null) affectedIdx.add(filmIdxForGlobal(globalEnd));
                    affectedIdx.add(current.filmIdx);
                } else {
                    affectedIdx.add(current.filmIdx);
                }
                if (has_septum_2) {
                    if (globalStart2 !== null) affectedIdx.add(filmIdxForGlobal(globalStart2));
                    if (globalEnd2   !== null) affectedIdx.add(filmIdxForGlobal(globalEnd2));
                    affectedIdx.add(current.filmIdx);
                } else {
                    affectedIdx.add(current.filmIdx);
                }
                
                document.getElementById('autosaveStatus').innerText = 'Saving...';
                document.getElementById('autosaveStatus').style.background = '#0f5132';
                
                let allOk = true;
                for (const fi of affectedIdx) {
                    const film   = films[fi];
                    const lcid   = localIds[fi];
                    if (!film || lcid === -1 || lcid === undefined || lcid === null) continue;
                    
                    const fb = boundaries[fi] || 0;
                    const nextFb = (fi + 1 < boundaries.length) ? boundaries[fi + 1] : state.numFrames;
                    
                    // Which points belong to this film?
                    let filmLocalStart = null;
                    let filmLocalEnd   = null;
                    let filmLocalStart2 = null;
                    let filmLocalEnd2   = null;
                    if (has_septum) {
                        if (globalStart !== null && globalStart >= fb && globalStart < nextFb) {
                            filmLocalStart = globalStart - fb;
                        }
                        if (globalEnd !== null && globalEnd >= fb && globalEnd < nextFb) {
                            filmLocalEnd = globalEnd - fb;
                        }
                    }
                    if (has_septum_2) {
                        if (globalStart2 !== null && globalStart2 >= fb && globalStart2 < nextFb) {
                            filmLocalStart2 = globalStart2 - fb;
                        }
                        if (globalEnd2 !== null && globalEnd2 >= fb && globalEnd2 < nextFb) {
                            filmLocalEnd2 = globalEnd2 - fb;
                        }
                    }
                    
                    const body = {
                        experiment: state.selectedExp,
                        film: film,
                        cell_id: lcid,
                        has_septum: has_septum && (filmLocalStart !== null || filmLocalEnd !== null),
                        local_start: filmLocalStart,
                        local_end:   filmLocalEnd,
                        white_septum,
                        has_septum_2: has_septum_2 && (filmLocalStart2 !== null || filmLocalEnd2 !== null),
                        local_start_2: filmLocalStart2,
                        local_end_2: filmLocalEnd2,
                        white_septum_2
                    };
                    
                    try {
                        const res = await fetch('/api/save_septum_label', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(body)
                        });
                        const d = await res.json();
                        if (d.status !== 'success') allOk = false;
                    } catch (e) {
                        console.error(`Error saving septum label for film ${film}:`, e);
                        allOk = false;
                    }
                }
                
                if (allOk) {
                    document.getElementById('autosaveStatus').innerText = 'Saved';
                    document.getElementById('autosaveStatus').style.background = '#0f5132';
                    renderGallery();
                    setTimeout(() => {
                        if (document.getElementById('autosaveStatus').innerText === 'Saved') {
                            document.getElementById('autosaveStatus').innerText = 'Idle';
                            document.getElementById('autosaveStatus').style.background = '#1e293b';
                        }
                    }, 1000);
                } else {
                    document.getElementById('autosaveStatus').innerText = 'Error';
                    document.getElementById('autosaveStatus').style.background = '#842029';
                }
                return;
            }
            
            // Non-sequence mode (or single-film sequence): simple single save
            const body = {
                experiment: state.selectedExp,
                film: current.film,
                cell_id: current.cellId,
                has_septum,
                local_start: globalStart,
                local_end: globalEnd,
                white_septum,
                has_septum_2,
                local_start_2: globalStart2,
                local_end_2: globalEnd2,
                white_septum_2
            };
            
            try {
                document.getElementById('autosaveStatus').innerText = 'Saving...';
                document.getElementById('autosaveStatus').style.background = '#0f5132';
                
                const res = await fetch('/api/save_septum_label', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    document.getElementById('autosaveStatus').innerText = 'Saved';
                    document.getElementById('autosaveStatus').style.background = '#0f5132';
                    renderGallery();
                    setTimeout(() => {
                        if (document.getElementById('autosaveStatus').innerText === 'Saved') {
                            document.getElementById('autosaveStatus').innerText = 'Idle';
                            document.getElementById('autosaveStatus').style.background = '#1e293b';
                        }
                    }, 1000);
                } else {
                    document.getElementById('autosaveStatus').innerText = 'Error';
                    document.getElementById('autosaveStatus').style.background = '#842029';
                }
            } catch (e) {
                console.error("Error saving septum label: ", e);
                document.getElementById('autosaveStatus').innerText = 'Error';
                document.getElementById('autosaveStatus').style.background = '#842029';
            }
        }


        async function runSeptumAi() {
            const cellId = state.selectedCell;
            if (!cellId) return;
            const current = getActiveFilmAndLocalCell();
            if (!current.film || current.cellId === -1 || current.cellId === undefined || current.cellId === null) return;
            
            const btn = document.getElementById('predictSeptumBtn');
            const originalText = btn.innerText;
            btn.innerText = '🤖 Predicting...';
            btn.disabled = true;
            
            const body = {
                experiment: state.selectedExp,
                film: current.film,
                cell_id: current.cellId
            };
            
            try {
                const res = await fetch('/api/predict_septum', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    renderSeptumAiChart(data.probs);
                    
                    const probs = data.probs;
                    const maxVal = Math.max(...probs);
                    if (maxVal > 0.3) {
                        const peakIdx = probs.indexOf(maxVal);
                        
                        let startSug = peakIdx;
                        while (startSug > 0 && probs[startSug - 1] > 0.2) {
                            startSug--;
                        }
                        let endSug = peakIdx;
                        while (endSug < probs.length - 1 && probs[endSug + 1] > 0.2) {
                            endSug++;
                        }
                        
                        document.getElementById('hasSeptumCheckbox').checked = true;
                        document.getElementById('divisionIntervalContainer').style.display = 'flex';
                        document.getElementById('septumStartInput').value = startSug;
                        document.getElementById('septumEndInput').value = endSug;
                        
                        await saveSeptumLabels();
                    }
                } else {
                    alert("AI prediction error: " + data.message);
                }
            } catch (e) {
                alert("Error calling AI: " + e);
            } finally {
                btn.innerText = originalText;
                btn.disabled = false;
            }
        }

        function renderSeptumAiChart(probs) {
            const container = document.getElementById('septumAiSparkline');
            container.innerHTML = '';
            if (!probs || probs.length === 0) {
                document.getElementById('septumAiChart').style.display = 'none';
                return;
            }
            document.getElementById('septumAiChart').style.display = 'flex';
            
            const maxVal = Math.max(...probs);
            const peakIdx = probs.indexOf(maxVal);
            document.getElementById('septumAiPeakText').innerText = `Peak: Frame ${peakIdx} (${(maxVal * 100).toFixed(0)}%)`;
            
            const barWidth = 100 / probs.length;
            for (let i = 0; i < probs.length; i++) {
                const bar = document.createElement('div');
                bar.style.width = `calc(${barWidth}% - 1px)`;
                bar.style.height = `${probs[i] * 100}%`;
                bar.style.backgroundColor = i === peakIdx ? 'var(--accent-green)' : (probs[i] > 0.5 ? '#10b981' : '#334155');
                bar.style.marginRight = '1px';
                bar.style.cursor = 'pointer';
                bar.title = `Frame ${i}: ${(probs[i] * 100).toFixed(1)}%`;
                
                bar.onclick = () => {
                    const current = getActiveFilmAndLocalCell();
                    let globalFrame = i;
                    if (current.filmIdx > 0 && state.filmBoundaries.length > current.filmIdx) {
                        globalFrame += state.filmBoundaries[current.filmIdx];
                    }
                    state.currentFrame = globalFrame;
                    document.getElementById('timeSlider').value = state.currentFrame;
                    displayFrame();
                };
                container.appendChild(bar);
            }
        }

        function drawStroke(x, y, isDraw) {
            const currentRle = state.cellMasks[state.currentFrame] || "";
            const W = state.imgWidth;
            const H = state.imgHeight;
            const maskArr = decodeRle(currentRle, W, H);
            const r = state.brushSize;
            for (let dy = -r; dy <= r; dy++) {
                for (let dx = -r; dx <= r; dx++) {
                    if (dx*dx + dy*dy <= r*r) {
                        const px = x + dx;
                        const py = y + dy;
                        if (px >= 0 && px < W && py >= 0 && py < H) {
                            maskArr[py * W + px] = isDraw ? 1 : 0;
                        }
                    }
                }
            }
            state.cellMasks[state.currentFrame] = rleEncode(maskArr, W, H);
            displayFrame();
            markDirty();
        }

        async function clickSelectSegment(x, y, unionExisting = false) {
            const modeParam = `sequence=${state.selectedSequence}`;
            const body = {
                experiment: state.selectedExp,
                t: state.currentFrame,
                channel: state.channel,
                cell_id: state.selectedCell,
                x: x,
                y: y,
                sequence: state.selectedSequence
            };
            
            const res = await fetch('/api/click_segment', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            const data = await res.json();
            
            if (data.status === 'success' && data.rle) {
                state.drawingHistory.push(state.cellMasks[state.currentFrame] || "");
                if (state.drawingHistory.length > 20) state.drawingHistory.shift();

                if (unionExisting) {
                    const W = state.imgWidth;
                    const H = state.imgHeight;
                    const maskArr = decodeRle(state.cellMasks[state.currentFrame] || "", W, H);
                    const newMaskArr = decodeRle(data.rle, W, H);
                    for (let i = 0; i < maskArr.length; i++) {
                        if (newMaskArr[i]) maskArr[i] = 1;
                    }
                    state.cellMasks[state.currentFrame] = rleEncode(maskArr, W, H);
                } else {
                    state.cellMasks[state.currentFrame] = data.rle;
                }
                displayFrame();
                markDirty();
            }
        }

        function setAutosaveStatus(text, color) {
            const el = document.getElementById('autosaveStatus');
            el.innerText = text;
            el.style.color = color;
        }

        function markDirty() {
            setAutosaveStatus('Unsaved…', '#f59e0b');
            cancelAutosave();
            state.autosaveTimer = setTimeout(() => {
                saveCorrectedMasks(true);
            }, 1500);
        }

        function cancelAutosave() {
            if (state.autosaveTimer) {
                clearTimeout(state.autosaveTimer);
                state.autosaveTimer = null;
            }
        }

        async function runAutofix() {
            const start = document.getElementById('autofixStartInput').value;
            const end = document.getElementById('autofixEndInput').value;
            if (start === '' || end === '') {
                alert("Please set both start and end frames.");
                return;
            }
            const tStart = parseInt(start);
            const tEnd = parseInt(end);
            if (tStart > tEnd) {
                alert("Start frame must be <= End frame.");
                return;
            }
            if (!state.selectedCell) return;
            
            const body = {
                experiment: state.selectedExp,
                cell_id: state.selectedCell,
                start_t: tStart,
                end_t: tEnd
            };
            if (state.isLocalEdit) {
                body.film = state.localFilmId;
            } else {
                body.sequence = state.selectedSequence;
            }
            
            setAutosaveStatus('Auto-fixing...', '#f59e0b');
            try {
                const res = await fetch('/api/auto_fix_segments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    setAutosaveStatus(`✓ Fixed ${data.fixed_count} frames`, '#10b981');
                    setTimeout(() => setAutosaveStatus('Idle', 'var(--text-muted)'), 3000);
                    const modeParamMasks = state.isLocalEdit ? `film=${state.localFilmId}` : `sequence=${state.selectedSequence}`;
                    const resMasks = await fetch(`/api/cell_masks?experiment=${state.selectedExp}&${modeParamMasks}&cell_id=${state.selectedCell}`);
                    const masksData = await resMasks.json();
                    state.cellMasks = masksData.masks;
                    displayFrame();
                    renderGallery();
                } else {
                    setAutosaveStatus('✗ Error', '#ef4444');
                    alert('Error during auto-fix: ' + data.message);
                }
            } catch (e) {
                setAutosaveStatus('✗ Error', '#ef4444');
                alert('Error: ' + e);
            }
        }

        async function saveCorrectedMasks(silent = false) {
            if (!state.selectedCell) return;
            setAutosaveStatus('Saving…', '#94a3b8');
            const body = {
                experiment: state.selectedExp,
                cell_id: state.selectedCell,
                channel: state.channel,
                masks: state.cellMasks
            };
            if (state.isLocalEdit) {
                body.film = state.localFilmId;
            } else {
                body.sequence = state.selectedSequence;
            }
            const res = await fetch('/api/save_masks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                setAutosaveStatus('✓ Saved', '#10b981');
                setTimeout(() => setAutosaveStatus('Idle', 'var(--text-muted)'), 3000);
                
                // Update the thumbnail for the current frame
                const thumb = document.getElementById('gallery-thumb-' + state.currentFrame);
                if (thumb) {
                    const ts = Date.now();
                    const modeParam = state.isLocalEdit ? `film=${state.localFilmId}` : `sequence=${state.selectedSequence}`;
                    thumb.src = `/api/frame_crop?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${state.currentFrame}&channel=${state.channel}&_ts=${ts}`;
                }
            } else {
                setAutosaveStatus('✗ Error', '#ef4444');
                if (!silent) alert('❌ Error saving masks: ' + data.message);
            }
        }

        function renderGallery() {
            const container = document.getElementById('stripContainer');
            container.innerHTML = '';
            const modeParam = state.isLocalEdit ? `film=${state.localFilmId}` : `sequence=${state.selectedSequence}`;
            
            const current = getActiveFilmAndLocalCell();
            const startVal = document.getElementById('septumStartInput').value;
            const endVal = document.getElementById('septumEndInput').value;
            const hasSeptum = document.getElementById('hasSeptumCheckbox').checked;
            
            const startVal2 = document.getElementById('septumStartInput2').value;
            const endVal2 = document.getElementById('septumEndInput2').value;
            const hasSeptum2 = document.getElementById('hasSeptumCheckbox2').checked;
            
            let globalStart = (hasSeptum && startVal !== '') ? parseInt(startVal) : null;
            let globalEnd = (hasSeptum && endVal !== '') ? parseInt(endVal) : null;
            let globalStart2 = (hasSeptum2 && startVal2 !== '') ? parseInt(startVal2) : null;
            let globalEnd2 = (hasSeptum2 && endVal2 !== '') ? parseInt(endVal2) : null;
            
            for (let t = 0; t < state.numFrames; t++) {
                const img = document.createElement('img');
                
                let classes = ['strip-crop'];
                if (t === state.currentFrame) {
                    classes.push('active');
                }
                if (hasSeptum) {
                    if (globalStart !== null && t === globalStart) classes.push('septum-start-frame');
                    if (globalEnd !== null && t === globalEnd) classes.push('septum-end-frame');
                    
                    let effectiveStart = globalStart !== null ? globalStart : 0;
                    let effectiveEnd = globalEnd !== null ? globalEnd : state.numFrames - 1;
                    if (t >= effectiveStart && t <= effectiveEnd) {
                        classes.push('septum-during-frame');
                    }
                }
                if (hasSeptum2) {
                    if (globalStart2 !== null && t === globalStart2) classes.push('septum-start-frame-2');
                    if (globalEnd2 !== null && t === globalEnd2) classes.push('septum-end-frame-2');
                    
                    let effectiveStart2 = globalStart2 !== null ? globalStart2 : 0;
                    let effectiveEnd2 = globalEnd2 !== null ? globalEnd2 : state.numFrames - 1;
                    if (t >= effectiveStart2 && t <= effectiveEnd2) {
                        classes.push('septum-during-frame-2');
                    }
                }
                
                img.className = classes.join(' ');
                img.id = 'gallery-thumb-' + t;
                img.style.height = '80px';
                img.style.width = '80px';
                img.style.objectFit = 'contain';
                const ts = Date.now();
                img.src = `/api/frame_crop?experiment=${state.selectedExp}&${modeParam}&cell_id=${state.selectedCell}&t=${t}&channel=${state.channel}&_ts=${ts}`;
                img.onclick = () => {
                    if (state.galleryClickMode === 'start1') {
                        state.currentFrame = t;
                        document.getElementById('timeSlider').value = t;
                        displayFrame();
                        
                        document.getElementById('hasSeptumCheckbox').checked = true;
                        document.getElementById('divisionIntervalContainer').style.display = 'flex';
                        document.getElementById('septumStartInput').value = t;
                        
                        saveSeptumLabels();
                        state.galleryClickMode = 'nav';
                        updateGalleryClickModeButtons();
                        renderGallery();
                    } else if (state.galleryClickMode === 'end1') {
                        state.currentFrame = t;
                        document.getElementById('timeSlider').value = t;
                        displayFrame();
                        
                        document.getElementById('hasSeptumCheckbox').checked = true;
                        document.getElementById('divisionIntervalContainer').style.display = 'flex';
                        document.getElementById('septumEndInput').value = t;
                        
                        saveSeptumLabels();
                        state.galleryClickMode = 'nav';
                        updateGalleryClickModeButtons();
                        renderGallery();
                    } else if (state.galleryClickMode === 'start2') {
                        state.currentFrame = t;
                        document.getElementById('timeSlider').value = t;
                        displayFrame();
                        
                        document.getElementById('hasSeptumCheckbox2').checked = true;
                        document.getElementById('divisionIntervalContainer2').style.display = 'flex';
                        document.getElementById('septumStartInput2').value = t;
                        
                        saveSeptumLabels();
                        state.galleryClickMode = 'nav';
                        updateGalleryClickModeButtons();
                        renderGallery();
                    } else if (state.galleryClickMode === 'end2') {
                        state.currentFrame = t;
                        document.getElementById('timeSlider').value = t;
                        displayFrame();
                        
                        document.getElementById('hasSeptumCheckbox2').checked = true;
                        document.getElementById('divisionIntervalContainer2').style.display = 'flex';
                        document.getElementById('septumEndInput2').value = t;
                        
                        saveSeptumLabels();
                        state.galleryClickMode = 'nav';
                        updateGalleryClickModeButtons();
                        renderGallery();
                    } else {
                        // nav mode
                        state.currentFrame = t;
                        document.getElementById('timeSlider').value = t;
                        displayFrame();
                    }
                };
                container.appendChild(img);
            }
        }

        function updateGalleryHighlight() {
            const crops = document.querySelectorAll('.strip-crop');
            crops.forEach((crop, idx) => {
                crop.classList.toggle('active', idx === state.currentFrame);
            });
        }
    </script>
</body>
</html>
"""

# ==============================================================================
# Backend Translation Layer
# ==============================================================================

def get_sequence_linkage_data(exp):
    seq_file = BASE_MOVIE_ROOT / exp / "sequence_linkage.json"
    data = {}
    if seq_file.exists():
        with open(seq_file, 'r') as f:
            data = json.load(f)
            
    # Also add isolated films as pseudo-sequences
    exp_dir = BASE_MOVIE_ROOT / exp
    if exp_dir.exists():
        films = sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        
        # Identify which films are already part of a true sequence
        filmed_in_seqs = set()
        for seq_data in data.values():
            filmed_in_seqs.update(seq_data.get("films", []))
            
        for film in films:
            if film not in filmed_in_seqs and film not in data:
                data[film] = {
                    "films": [film],
                    "global_cells": {}, # populated on demand
                    "lineage": {}
                }
    return data

def ensure_pseudo_sequence_cells(exp, sequence, data):
    if sequence not in data:
        return
    seq_data = data[sequence]
    if len(seq_data.get("films", [])) == 1 and not seq_data.get("global_cells"):
        film = seq_data["films"][0]
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        cells = []
        if tracked_dir.exists():
            for cf in tracked_dir.glob("cell_*_masks.csv"):
                if cf.name.startswith("."): continue
                m = re.search(r'cell_(\d+)_masks\.csv', cf.name)
                if m:
                    cells.append(int(m.group(1)))
        seq_data["global_cells"] = {str(c): [c] for c in sorted(list(set(cells)))}

def get_film_frame_count_and_size(exp, film):
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    for cf in tracked_dir.iterdir():
        if cf.name.startswith("."): continue
        if cf.name.endswith("_masks.csv"):
            try:
                df = pd.read_csv(cf)
                return len(df), int(df.iloc[0]['width']), int(df.iloc[0]['height'])
            except Exception:
                continue
    # Fallback to frames dir
    frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
    files = [f for f in frames_dir.glob("*.tif") if not f.name.startswith(".")]
    if not files:
        return 0, 0, 0
    img = imread(str(files[0]))
    # count unique frame numbers
    t_set = set()
    for f in files:
        m = re.search(r'_t_(\d+)_', f.name)
        if m: t_set.add(int(m.group(1)))
    return len(t_set), img.shape[1], img.shape[0]

def resolve_global_t(exp, sequence, global_cell_id, global_t):
    """
    Returns (local_film, local_cell_id, local_t)
    """
    data = get_sequence_linkage_data(exp)
    if sequence not in data:
        raise ValueError("Sequence not found")
        
    ensure_pseudo_sequence_cells(exp, sequence, data)
    films = data[sequence]["films"]
    local_ids = data[sequence]["global_cells"].get(global_cell_id, [-1]*len(films))
    
    current_t = 0
    for i, film in enumerate(films):
        local_id = local_ids[i]
        L, W, H = get_film_frame_count_and_size(exp, film)
        if global_t < current_t + L:
            return film, local_id, global_t - current_t
        current_t += L
        
    # If out of bounds
    return films[-1], local_ids[-1], L - 1

def find_time_from_name(name):
    m = re.search(r"_t_(\d+)_", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def id_to_color(cell_id: int):
    rng = abs(hash(int(cell_id)))
    h = rng % 180
    s = 200 + (rng // 180) % 56
    v = 220 + (rng // (180 * 56)) % 36
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def generate_population_frame_image(exp, film, t_val, cell_maps=None, files=None):
    if files is None:
        frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
        files = sorted([f for f in frames_dir.glob(f"{film}_t_*_c_*.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in frames_dir.glob(f"*_t_*_c_*.tif") if not f.name.startswith(".")])
            
    t_files = [f for f in files if find_time_from_name(f.name) == t_val]
    if not t_files:
        return None
        
    img = imread(str(t_files[0]))
    H, W = img.shape[:2]
    
    p_lo = np.percentile(img, 1.0)
    p_hi = np.percentile(img, 99.5)
    if p_hi > p_lo:
        img_scaled = np.clip((img - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_scaled = (img / img.max() * 255.0).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
        
    if len(img_scaled.shape) == 2:
        img_bgr = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_scaled.copy()
        
    if cell_maps is None:
        cell_maps = []
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        if tracked_dir.is_dir():
            for cf in tracked_dir.iterdir():
                if cf.name.startswith("."): continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
                if m:
                    cell_id = int(m.group(1))
                    try:
                        df = pd.read_csv(cf)
                        time_to_rle = {}
                        for idx, row in df.iterrows():
                            t_pt = int(row['time_point'])
                            rle_col = 'rle_bf'
                            if 'rle_gfp' in df.columns and pd.notna(row.get('rle_gfp')) and str(row.get('rle_gfp')).strip():
                                rle_col = 'rle_gfp'
                            if rle_col in df.columns:
                                rle = row[rle_col]
                                if isinstance(rle, str) and rle.strip():
                                    time_to_rle[t_pt] = rle
                        cell_maps.append((cell_id, time_to_rle))
                    except Exception:
                        pass
                        
    overlay = np.zeros_like(img_bgr, dtype=np.uint8)
    alpha = 0.4
    
    for cell_id, time_to_rle in cell_maps:
        rle = time_to_rle.get(t_val)
        if rle is None:
            continue
        try:
            mask = rle_decode(rle, (H, W))
            if not mask.any():
                continue
                
            color = id_to_color(cell_id)
            overlay[mask] = color
            
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                text = str(cell_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.8
                thickness = 2
                cv2.putText(img_bgr, text, (cx, cy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(img_bgr, text, (cx, cy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        except Exception:
            pass
            
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1.0, 0.0)
    
    max_dim = 1000
    if max(H, W) > max_dim:
        scale = max_dim / max(H, W)
        new_W = int(W * scale)
        new_H = int(H * scale)
        blended_resized = cv2.resize(blended, (new_W, new_H), interpolation=cv2.INTER_AREA)
    else:
        blended_resized = blended
        
    _, jpeg_encoded = cv2.imencode('.jpg', blended_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return jpeg_encoded.tobytes()

RUNNING_PREGENERATIONS = set()
PREGENERATION_LOCK = threading.Lock()

def background_generate_population_frames(exp, film):
    try:
        frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
        if not frames_dir.is_dir():
            return
        files = sorted([f for f in frames_dir.glob(f"{film}_t_*_c_*.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in frames_dir.glob(f"*_t_*_c_*.tif") if not f.name.startswith(".")])
        if not files:
            return
            
        t_points = sorted(list(set([find_time_from_name(f.name) for f in files])))
        t_points = [t for t in t_points if t is not None]
        
        cache_dir = BASE_MOVIE_ROOT / exp / film / f"PopulationFrames_{film}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cell_maps = []
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        if tracked_dir.is_dir():
            for cf in tracked_dir.iterdir():
                if cf.name.startswith("."): continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
                if m:
                    cell_id = int(m.group(1))
                    try:
                        df = pd.read_csv(cf)
                        time_to_rle = {}
                        for idx, row in df.iterrows():
                            t_val = int(row['time_point'])
                            rle_col = 'rle_bf'
                            if 'rle_gfp' in df.columns and pd.notna(row.get('rle_gfp')) and str(row.get('rle_gfp')).strip():
                                rle_col = 'rle_gfp'
                            if rle_col in df.columns:
                                rle = row[rle_col]
                                if isinstance(rle, str) and rle.strip():
                                    time_to_rle[t_val] = rle
                        cell_maps.append((cell_id, time_to_rle))
                    except Exception:
                        pass
                        
        if not cell_maps:
            return
            
        for t_val in t_points:
            cache_file = cache_dir / f"frame_{t_val:03d}.jpg"
            if not cache_file.exists():
                try:
                    img_data = generate_population_frame_image(exp, film, t_val, cell_maps, files)
                    if img_data is not None:
                        with open(cache_file, "wb") as f:
                            f.write(img_data)
                except Exception as e:
                    print(f"Error pre-generating frame {t_val} for {film}: {e}")
    except Exception as e:
        print(f"Error in background generation thread for {film}: {e}")
    finally:
        with PREGENERATION_LOCK:
            RUNNING_PREGENERATIONS.discard((exp, film))

def trigger_pregeneration_for_films(exp, films):
    with PREGENERATION_LOCK:
        for film in films:
            key = (exp, film)
            if key not in RUNNING_PREGENERATIONS:
                RUNNING_PREGENERATIONS.add(key)
                t = threading.Thread(target=background_generate_population_frames, args=(exp, film))
                t.daemon = True
                t.start()

# ==============================================================================
# API Endpoints
# ==============================================================================


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/list_experiments")
def list_experiments():
    experiments = sorted([d.name for d in BASE_MOVIE_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name in RELEVANT_EXPERIMENTS])
    return jsonify({"experiments": experiments})

@app.route("/api/list_films_and_sequences")
def list_films_and_sequences():
    exp = request.args.get("experiment")
    exp_dir = BASE_MOVIE_ROOT / exp
    films = sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    seq_data = get_sequence_linkage_data(exp)
    sequences = list(seq_data.keys())
    
    return jsonify({"films": films, "sequences": sequences})

@app.route("/api/list_cells")
def list_cells():
    exp = request.args.get("experiment")
    if "sequence" in request.args:
        sequence = request.args.get("sequence")
        data = get_sequence_linkage_data(exp)
        if sequence in data:
            ensure_pseudo_sequence_cells(exp, sequence, data)
            trigger_pregeneration_for_films(exp, data[sequence]["films"])
            def get_sort_key(k):
                s = str(k)
                m = re.search(r"(\d+)$", s)
                if m:
                    return (0, int(m.group(1)))
                return (1, s)
            cells = sorted(list(data[sequence]["global_cells"].keys()), key=get_sort_key)
            # Filter to keep only cells that have a valid tracking label in the last film
            cells = [c for c in cells if data[sequence]["global_cells"][c][-1] != -1]
            
            def display_name_for(global_id):
                # Extract trailing number from global ID (e.g. "A14_F1_cell_18" → "Cell 18")
                m = re.search(r'(\d+)$', str(global_id))
                return f"Cell {m.group(1)}" if m else str(global_id)
            
            # Build display names, appending film hint on collision
            base_names = {c: display_name_for(c) for c in cells}
            name_count = {}
            for n in base_names.values():
                name_count[n] = name_count.get(n, 0) + 1
            
            def origin_film_hint(global_id, local_ids):
                # Check if global_id contains an inner film name
                # e.g., "A14_F1_A14_BF_2_F1_cell_11" -> "A14_BF_2_F1"
                prefix = f"{sequence}_"
                gid_str = str(global_id)
                if gid_str.startswith(prefix) and "_cell_" in gid_str:
                    inner_part = gid_str[len(prefix):gid_str.rfind("_cell_")]
                    if inner_part:
                        parts = inner_part.split("_")
                        return "_".join(parts[-3:]) if len(parts) >= 3 else inner_part
                
                # Fallback to the first film where the cell is tracked
                films_local = data[sequence]["films"]
                for i, lid in enumerate(local_ids):
                    if lid != -1:
                        # Use a short film suffix, e.g. "FL_1_F1"
                        film = films_local[i] if i < len(films_local) else ""
                        parts = film.split("_")
                        hint = "_".join(parts[-3:]) if len(parts) >= 3 else film
                        return hint
                return "root"
            
            cells_data = []
            for c in cells:
                name = base_names[c]
                if name_count[name] > 1:
                    hint = origin_film_hint(c, data[sequence]["global_cells"][c])
                    name = f"{name} ({hint})"
                cells_data.append({"global_id": c, "display_name": name})
            lineage = data[sequence].get("lineage", {})
            return jsonify({"cells": cells_data, "lineage": lineage})
        return jsonify({"cells": [], "lineage": {}})
        
    film = request.args.get("film")
    trigger_pregeneration_for_films(exp, [film])
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    
    cells = []
    if tracked_dir.is_dir():
        for f in tracked_dir.iterdir():
            if f.name.startswith("."):
                continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                cells.append(int(m.group(1)))
                
    cells_data = [{"global_id": str(c), "display_name": f"Cell {c}"} for c in sorted(list(set(cells)))]
    return jsonify({"cells": cells_data})

@app.route("/api/cell_masks")
def cell_masks():
    exp = request.args.get("experiment")
    cell_id = request.args.get("cell_id")
    
    if "sequence" in request.args:
        sequence = request.args.get("sequence")
        data = get_sequence_linkage_data(exp)
        if sequence not in data:
            return jsonify({"error": "Sequence not found"}), 404
            
        ensure_pseudo_sequence_cells(exp, sequence, data)
        trigger_pregeneration_for_films(exp, data[sequence]["films"])
        films = data[sequence]["films"]
        local_ids = data[sequence]["global_cells"].get(cell_id, [-1]*len(films))
        
        all_masks = []
        boundaries = []
        current_len = 0
        w, h = 0, 0
        track_channel = 'bf'
        
        for i, film in enumerate(films):
            boundaries.append(current_len)
            
            L, fW, fH = get_film_frame_count_and_size(exp, film)
            if w == 0 and fW > 0:
                w, h = fW, fH
                
            local_id = local_ids[i]
            if local_id == -1:
                all_masks.extend([""] * L)
                current_len += L
                continue
                
            csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_id}_masks.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                    track_channel = 'gfp'
                    rle_col = 'rle_gfp'
                    
                masks = df[rle_col].fillna("").tolist()
                
                # pad or truncate to L
                if len(masks) < L:
                    masks.extend([""] * (L - len(masks)))
                elif len(masks) > L:
                    masks = masks[:L]
                    
                all_masks.extend(masks)
            else:
                all_masks.extend([""] * L)
                
            current_len += L
            
        return jsonify({
            "masks": all_masks,
            "num_frames": len(all_masks),
            "width": w,
            "height": h,
            "track_channel": track_channel,
            "film_boundaries": boundaries,
            "linkage_details": {"films": films, "local_ids": local_ids},
            "local_film": films[0] if films else None
        })
        
    else:
        film = request.args.get("film")
        trigger_pregeneration_for_films(exp, [film])
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
        df = pd.read_csv(csv_path)
        
        # Determine tracking channel: default to "gfp" if it's a GFP film, else "bf"
        track_channel = "gfp" if "FL" in film else "bf"
        rle_col = 'rle_gfp' if track_channel == 'gfp' else 'rle_bf'
        
        # Override default if either channel has non-empty masks in df
        if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
            track_channel = 'gfp'
            rle_col = 'rle_gfp'
        elif 'rle_bf' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_bf'].dropna()):
            track_channel = 'bf'
            rle_col = 'rle_bf'
                
        masks = df[rle_col].fillna("").tolist()
        return jsonify({
            "masks": masks,
            "num_frames": len(df),
            "width": int(df.iloc[0]['width']),
            "height": int(df.iloc[0]['height']),
            "track_channel": track_channel,
            "local_film": film
        })

@app.route("/api/get_candidates")
def get_candidates():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    
    cells = []
    if tracked_dir.is_dir():
        for f in tracked_dir.iterdir():
            if f.name.startswith("."): continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                cells.append(int(m.group(1)))
    return jsonify({"cells": sorted(list(set(cells)))})


@app.route("/api/identify_cell", methods=["POST"])
def identify_cell():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    t = int(data.get("t", 0))
    
    if not film:
        return jsonify({"status": "error", "message": "Film name is required but was empty."})
        
    try:
        x, y = int(data.get("x")), int(data.get("y"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Coordinates x and y must be integers."})
    
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    if not tracked_dir.is_dir():
        return jsonify({"status": "error", "message": f"TrackedCells directory not found: {tracked_dir}"})
    
    # Fast path: Use the segmentation mask directly to find the label, then see which cell has that label?
    # Wait, cells might not map 1:1 to labels. Just iterate csvs.
    for cf in tracked_dir.iterdir():
        if cf.name.startswith("."): continue
        m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
        if m:
            cid = int(m.group(1))
            try:
                df = pd.read_csv(cf)
                if t >= len(df): continue
                
                # Check both masks
                W = int(df.iloc[0]['width'])
                H = int(df.iloc[0]['height'])
                found = False
                for rle_col in ['rle_bf', 'rle_gfp']:
                    if rle_col in df.columns:
                        rle = df.iloc[t][rle_col]
                        if isinstance(rle, str) and rle.strip():
                            mask = rle_decode(rle, (H, W))
                            if y < H and x < W and mask[y, x]:
                                found = True
                                break
                if found:
                    return jsonify({"status": "success", "cell_id": cid})
            except Exception:
                pass
                
    # If no tracked cell is found, check the raw segmentation mask
    try:
        masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
        files = sorted([f for f in masks_dir.glob(f"{film}_t_{t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in masks_dir.glob(f"*_t_{t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if files:
            from Cell_tracking_functions import load_segmentation
            from skimage.measure import label
            seg = load_segmentation(str(files[0]))
            seg_lbl = label(seg) if seg.dtype == bool else seg
            if y < seg_lbl.shape[0] and x < seg_lbl.shape[1]:
                val = seg_lbl[y, x]
                if val > 0:
                    return jsonify({"status": "untracked", "label_id": int(val), "message": f"Found raw segment #{val}, but it is not tracked."})
    except Exception as e:
        print(f"Error checking seg mask: {e}")
        
    return jsonify({"status": "error", "message": "No tracked cell found at this location."})

@app.route('/api/get_qc', methods=['GET'])
def get_qc():
    exp = request.args.get('experiment')
    film = request.args.get('film')
    seq = request.args.get('sequence')
    
    target = seq if seq else film
    qc_file = BASE_MOVIE_ROOT / exp / target / f"qc_{target}.json"
    
    if qc_file.exists():
        import json
        with open(qc_file, 'r') as f:
            return jsonify({"status": "success", "qc": json.load(f)})
    return jsonify({"status": "success", "qc": {}})

@app.route("/api/suspicious_cells")
def suspicious_cells():
    exp = request.args.get("experiment")
    sequence = request.args.get("sequence")
    film = request.args.get("film")
    threshold = float(request.args.get("threshold", 15.0))
    
    target = sequence if sequence else film
    cache_key = f"{exp}::{target}::thresh_{threshold}"
    
    if cache_key in SUSPICIOUS_CACHE:
        return jsonify({"suspicious": SUSPICIOUS_CACHE[cache_key]})
        
    # Check disk cache
    target_dir = BASE_MOVIE_ROOT / exp / target
    cache_file = target_dir / f"suspicious_{target}.json"
    if cache_file.exists():
        try:
            import json
            with open(cache_file, "r") as f:
                disk_data = json.load(f)
                SUSPICIOUS_CACHE[cache_key] = disk_data
                return jsonify({"suspicious": disk_data})
        except Exception as e:
            print(f"Error reading disk cache: {e}")
            
    suspicious_data = {}
    
    if sequence:
        seq_data = get_sequence_linkage_data(exp)
        if sequence not in seq_data:
            return jsonify({"suspicious": {}})
        ensure_pseudo_sequence_cells(exp, sequence, seq_data)
        films = seq_data[sequence]["films"]
        cell_mappings = seq_data[sequence]["global_cells"]
    else:
        films = [film]
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        cell_mappings = {}
        if tracked_dir.is_dir():
            for f in tracked_dir.iterdir():
                if f.name.startswith("."):
                    continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
                if m:
                    cid = m.group(1)
                    cell_mappings[cid] = [int(cid)]
                    
    for cell_id, local_ids in cell_mappings.items():
        film_dfs = []
        
        for i, f_name in enumerate(films):
            local_id = local_ids[i] if i < len(local_ids) else -1
            if local_id == -1:
                film_dfs.append(None)
                continue
            csv_path = BASE_MOVIE_ROOT / exp / f_name / f"TrackedCells_{f_name}" / f"cell_{local_id}_masks.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    film_dfs.append(df)
                except Exception:
                    film_dfs.append(None)
            else:
                film_dfs.append(None)
                
        all_rles = []
        H, W = 0, 0
        for i, df in enumerate(film_dfs):
            f_name = films[i]
            L, fW, fH = get_film_frame_count_and_size(exp, f_name)
            if df is not None and len(df) > 0:
                if H == 0:
                    H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
                
                # Determine rle_col for this dataframe
                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                    rle_col = 'rle_gfp'
                elif 'rle_bf' not in df.columns and 'rle_gfp' in df.columns:
                    rle_col = 'rle_gfp'
                    
                masks = df[rle_col].fillna("").tolist()
                if len(masks) < L:
                    masks.extend([""] * (L - len(masks)))
                elif len(masks) > L:
                    masks = masks[:L]
                all_rles.extend(masks)
            else:
                all_rles.extend([""] * L)
                
        if H == 0 or W == 0:
            continue
            
        centroids = []
        for rle in all_rles:
            if not isinstance(rle, str) or not rle.strip() or rle == "nan":
                centroids.append(None)
                continue
            try:
                mask = rle_decode(rle, (H, W))
                if not mask.any():
                    centroids.append(None)
                else:
                    ys, xs = np.nonzero(mask)
                    centroids.append((float(np.mean(ys)), float(np.mean(xs))))
            except Exception:
                centroids.append(None)
                
        suspicious_frames = []
        for t in range(1, len(centroids)):
            c1 = centroids[t-1]
            c2 = centroids[t]
            if c1 is not None and c2 is not None:
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if dist > threshold:
                    suspicious_frames.append(t)
                    
        if suspicious_frames:
            suspicious_data[str(cell_id)] = suspicious_frames
            
    SUSPICIOUS_CACHE[cache_key] = suspicious_data
    
    # Save to disk cache
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(cache_file, "w") as f:
            json.dump(suspicious_data, f)
    except Exception as e:
        print(f"Error writing disk cache: {e}")
        
    return jsonify({"suspicious": suspicious_data})

@app.route('/api/save_qc', methods=['POST'])
def save_qc():
    data = request.json
    exp = data.get('experiment')
    film = data.get('film')
    seq = data.get('sequence')
    cell_id = str(data.get('cell_id'))
    status = data.get('status')
    
    target = seq if seq else film
    target_dir = BASE_MOVIE_ROOT / exp / target
    target_dir.mkdir(parents=True, exist_ok=True)
    qc_file = target_dir / f"qc_{target}.json"
    
    import json
    qc_data = {}
    if qc_file.exists():
        with open(qc_file, 'r') as f:
            qc_data = json.load(f)
            
    if status == "pending":
        if cell_id in qc_data:
            del qc_data[cell_id]
    else:
        qc_data[cell_id] = status
    
    with open(qc_file, 'w') as f:
        json.dump(qc_data, f)
        
    return jsonify({"status": "success"})

@app.route("/api/create_new_cell", methods=["POST"])
def create_new_cell():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    tracked_dir.mkdir(parents=True, exist_ok=True)
    
    max_id = 9999
    for f in tracked_dir.iterdir():
        if f.name.startswith("."): continue
        m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
        if m:
            cid = int(m.group(1))
            if cid > max_id:
                max_id = cid
                
    new_id = max_id + 1
    
    L, W, H = get_film_frame_count_and_size(exp, film)
    if L == 0:
        return jsonify({"status": "error", "message": "No frame images or existing cell files found to determine dimensions."})
        
    rows = []
    for t in range(L):
        rows.append({
            "time_point": t,
            "width": W, "height": H,
            "rle_bf": "",
            "touches_border_bf": False,
            "source_bf": "manual" if t == 0 else "",
            "overlap_score_bf": 1.0,
            "smooth_score_bf": 0.0,
            "area_bf": 0,
            "rle_gfp": "",
            "touches_border_gfp": False,
            "source_gfp": "manual" if t == 0 else "",
            "overlap_score_gfp": 1.0,
            "smooth_score_gfp": 0.0,
            "area_gfp": 0
        })
        
    df = pd.DataFrame(rows)
    out_csv = tracked_dir / f"cell_{new_id}_masks.csv"
    df.to_csv(out_csv, index=False)
    
    return jsonify({"status": "success", "cell_id": new_id})

@app.route("/api/quantify_on_hpc", methods=["POST"])
def quantify_on_hpc():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    label_id = data.get("label_id")
    seed_from_csv = data.get("seed_from_csv", False)
    
    track_channel = data.get("track_channel", "gfp" if "FL" in film else "bf")
    
    import subprocess
    
    try:
        exp_dir = BASE_MOVIE_ROOT / exp
        seed_flag = " --seed_from_csv" if seed_from_csv else ""
        
        script_path = Path(__file__).parent / "one_cell_quantification_1CH.py"
        
        import sys
        # Run quantification locally
        local_cmd = f""" "{sys.executable}" "{script_path}" --cell_id {label_id} --experiment_path "{exp_dir}" --file_name "{film}" --track_channel {track_channel} --update_existing{seed_flag} """
        
        result = subprocess.run(local_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            err_msg = (result.stderr or "").strip() or (result.stdout or "").strip() or f"Process exited with code {result.returncode}"
            print(f"Quantification Error: {err_msg}")
            return jsonify({"status": "error", "message": f"Quantification Error: {err_msg}"})
            
        return jsonify({"status": "success", "message": f"Successfully quantified cell #{label_id} locally!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/update_linkage", methods=["POST"])
def update_linkage():
    data = request.json
    exp = data.get("experiment")
    sequence = data.get("sequence")
    global_cell = data.get("global_cell")
    film_idx = int(data.get("film_idx"))
    new_local_cell = int(data.get("new_local_cell"))
    
    seq_file = BASE_MOVIE_ROOT / exp / "sequence_linkage.json"
    if not seq_file.exists():
        return jsonify({"status": "error", "message": f"Sequence linkage file not found: {seq_file}"})
        
    with open(seq_file, 'r') as f:
        linkage_data = json.load(f)
        
    if sequence not in linkage_data:
        return jsonify({"status": "error", "message": f"Sequence '{sequence}' not found in linkage data."})
        
    global_cell_str = str(global_cell)
    if global_cell_str not in linkage_data[sequence]["global_cells"]:
        return jsonify({"status": "error", "message": f"Global cell '{global_cell_str}' not found in sequence '{sequence}'."})
        
    linkage_data[sequence]["global_cells"][global_cell_str][film_idx] = new_local_cell
    
    with open(seq_file, 'w') as f:
        json.dump(linkage_data, f, indent=2)
        
    return jsonify({"status": "success"})

# ==============================================================================
# Septum Prediction AI & Labeling Endpoints
# ==============================================================================

def get_inference_runner(exp):
    chkpts = [
        BASE_MOVIE_ROOT / exp / "training_dataset" / "checkpoints_binary" / "model_latest.pt",
        BASE_MOVIE_ROOT / exp / "training_dataset" / "checkpoints" / "model_latest.pt",
        Path("/Volumes/X10 Pro/Movies/2025_12_31_M92/training_dataset/checkpoints_binary/model_latest.pt"),
        Path("/Volumes/X10 Pro/Movies/2025_12_31_M92/training_dataset/checkpoints/model_latest.pt")
    ]
    for cp in chkpts:
        if cp.exists():
            try:
                from SingleCellDataAnalysis.inference_core import FungalInferenceCore
                print(f"Loading FungalInferenceCore with checkpoint {cp}...")
                return FungalInferenceCore(str(cp), device="cpu")
            except Exception as e:
                print(f"Error loading checkpoint {cp}: {e}")
    return None

def get_cell_crop_tile(exp, film, t, rle, pad=10, tile_size=96):
    try:
        frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
        files = sorted([f for f in frames_dir.glob(f"{film}_t_{t:03d}_c_0.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in frames_dir.glob(f"*_t_{t:03d}_c_0.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in frames_dir.glob(f"{film}_t_{t:03d}_c_*.tif") if not f.name.startswith(".")])
            if not files:
                files = sorted([f for f in frames_dir.glob(f"*_t_{t:03d}_c_*.tif") if not f.name.startswith(".")])
                
        if not files:
            return None
            
        img = imread(str(files[0]))
        H, W = img.shape[:2]
        mask = rle_decode(rle, (H, W))
        
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
            
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        y0 = max(0, y0 - pad)
        y1 = min(H - 1, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(W - 1, x1 + pad)
        crop = img[y0:y1 + 1, x0:x1 + 1]
        
        Ht, Wt = tile_size, tile_size
        a = np.asarray(crop)
        
        if a.dtype != np.uint8:
            af = a.astype(np.float32)
            lo, hi = np.nanpercentile(af, [1, 99]) if np.isfinite(af).any() else (0.0, 1.0)
            if not np.isfinite(lo): lo = 0.0
            if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
            af = np.clip((af - lo) / (hi - lo), 0, 1)
            a = (255 * af).astype(np.uint8)
        else:
            a = a.copy()
            
        h, w = a.shape[:2]
        if h > Ht:
            y_start = (h - Ht) // 2
            a = a[y_start:y_start + Ht, :]
            h = Ht
        if w > Wt:
            x_start = (w - Wt) // 2
            a = a[:, x_start:x_start + Wt]
            w = Wt
            
        out = np.zeros((Ht, Wt), dtype=np.uint8)
        y_start = (Ht - h) // 2
        x_start = (Wt - w) // 2
        out[y_start:y_start + h, x_start:x_start + w] = a
        return out
    except Exception as e:
        print(f"Error cropping cell at t={t}: {e}")
        return None

@app.route("/api/get_septum_label", methods=["GET"])
def get_septum_label():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    cell_id = str(request.args.get("cell_id"))
    
    label_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels"
    json_path = label_dir / "global_septum_alignment.json"
    
    has_septum = False
    local_start = None
    local_end = None
    white_septum = False
    
    has_septum_2 = False
    local_start_2 = None
    local_end_2 = None
    white_septum_2 = False
    
    offset = 0
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                js = json.load(f)
            offsets = js.get("offsets", {})
            cell_intervals = js.get("cell_intervals", {})
            
            offset = int(offsets.get(cell_id, 0))
            ci = cell_intervals.get(cell_id, {})
            
            has_septum = bool(ci.get("has_septum", False))
            start_aligned = ci.get("start_aligned")
            end_aligned = ci.get("end_aligned")
            white_septum = bool(ci.get("white_septum", False))
            
            has_septum_2 = bool(ci.get("has_septum_2", False))
            start_aligned_2 = ci.get("start_aligned_2")
            end_aligned_2 = ci.get("end_aligned_2")
            white_septum_2 = bool(ci.get("white_septum_2", False))
            
            if start_aligned is not None:
                local_start = int(start_aligned - offset)
            if end_aligned is not None:
                local_end = int(end_aligned - offset)
                
            if start_aligned_2 is not None:
                local_start_2 = int(start_aligned_2 - offset)
            if end_aligned_2 is not None:
                local_end_2 = int(end_aligned_2 - offset)
        except Exception as e:
            print(f"Error loading global_septum_alignment.json: {e}")
            
    return jsonify({
        "status": "success",
        "has_septum": has_septum,
        "local_start": local_start,
        "local_end": local_end,
        "white_septum": white_septum,
        "has_septum_2": has_septum_2,
        "local_start_2": local_start_2,
        "local_end_2": local_end_2,
        "white_septum_2": white_septum_2,
        "offset": offset
    })

@app.route("/api/save_septum_label", methods=["POST"])
def save_septum_label():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    cell_id = str(data.get("cell_id"))
    
    has_septum = bool(data.get("has_septum", False))
    local_start = data.get("local_start")
    local_end = data.get("local_end")
    white_septum = bool(data.get("white_septum", False))
    
    has_septum_2 = bool(data.get("has_septum_2", False))
    local_start_2 = data.get("local_start_2")
    local_end_2 = data.get("local_end_2")
    white_septum_2 = bool(data.get("white_septum_2", False))
    
    label_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    json_path = label_dir / "global_septum_alignment.json"
    
    js = {
        "working_dir": str(BASE_MOVIE_ROOT / exp),
        "film_name": film,
        "cell_order": [],
        "offsets": {},
        "global_interval": {"G0": 0, "G1": 55},
        "cell_intervals": {}
    }
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                js = json.load(f)
        except Exception as e:
            print(f"Error loading global_septum_alignment.json: {e}")
            
    offsets = js.setdefault("offsets", {})
    cell_intervals = js.setdefault("cell_intervals", {})
    
    offset = int(offsets.setdefault(cell_id, 0))
    
    start_aligned = (local_start + offset) if (local_start is not None and has_septum) else None
    end_aligned = (local_end + offset) if (local_end is not None and has_septum) else None
    
    start_aligned_2 = (local_start_2 + offset) if (local_start_2 is not None and has_septum_2) else None
    end_aligned_2 = (local_end_2 + offset) if (local_end_2 is not None and has_septum_2) else None
    
    cell_intervals[cell_id] = {
        "has_septum": has_septum,
        "start_aligned": start_aligned,
        "end_aligned": end_aligned,
        "white_septum": white_septum,
        "has_septum_2": has_septum_2,
        "start_aligned_2": start_aligned_2,
        "end_aligned_2": end_aligned_2,
        "white_septum_2": white_septum_2
    }
    
    from datetime import datetime
    js["updated_at"] = datetime.now().isoformat()
    
    try:
        with open(json_path, 'w') as f:
            json.dump(js, f, indent=2)
            
        # Discover all cell IDs in the film to export the CSV
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        all_cids = []
        for f in tracked_dir.iterdir():
            if f.name.startswith("."): continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                all_cids.append(int(m.group(1)))
        all_cids.sort()
        
        gi = js.get("global_interval", {})
        a_left = int(gi.get("G0", 0))
        
        csv_path = label_dir / "septum_interval_per_cell.csv"
        rows = []
        for cid in all_cids:
            cid_str = str(cid)
            ci = cell_intervals.get(cid_str, {})
            rows.append({
                "cell_id": cid,
                "a_left": a_left,
                "start_aligned": ci.get("start_aligned") if ci.get("start_aligned") is not None else "",
                "end_aligned": ci.get("end_aligned") if ci.get("end_aligned") is not None else "",
                "has": 1 if ci.get("has_septum") else 0,
                "white_septum": 1 if ci.get("white_septum") else 0,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        # Build strip and export training sample in the exact training dataset format
        csv_path_cell = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
        if csv_path_cell.exists():
            try:
                df_cell = pd.read_csv(csv_path_cell)
                rle_col = 'rle_bf'
                if 'rle_gfp' in df_cell.columns and df_cell['rle_gfp'].dropna().any():
                    rle_col = 'rle_gfp'
                    
                tiles = []
                L = len(df_cell)
                for t in range(L):
                    rle = df_cell.iloc[t][rle_col]
                    tile = None
                    if isinstance(rle, str) and rle.strip():
                        tile = get_cell_crop_tile(exp, film, t, rle)
                    if tile is None:
                        tile = np.zeros((96, 96), dtype=np.uint8)
                    tiles.append(tile)
                    
                strip = np.hstack(tiles)
                
                from SingleCellDataAnalysis.septum_training_utils import export_cell_training_sample
                export_cell_training_sample(
                    working_dir=str(BASE_MOVIE_ROOT / exp),
                    film_name=film,
                    cell_id=int(cell_id),
                    strip=strip,
                    tp0=0,
                    offset=offset,
                    start_idx=local_start if (local_start is not None and has_septum) else -1,
                    end_idx=local_end if (local_end is not None and has_septum) else -1,
                    label_source="cell",
                    start_aligned=start_aligned,
                    end_aligned=end_aligned,
                    white_septum=white_septum,
                )
                print(f"Successfully exported training sample for cell {cell_id} in {film} to training_dataset")
            except Exception as e:
                print(f"Error calling export_cell_training_sample for cell {cell_id}: {e}")
        
        return jsonify({"status": "success", "message": "Septum labels saved and CSV/training sample exported successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/predict_septum", methods=["POST"])
def predict_septum():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    cell_id = str(data.get("cell_id"))
    
    inference_runner = get_inference_runner(exp)
    if inference_runner is None:
        return jsonify({"status": "error", "message": "Septum AI model checkpoint not found or could not be loaded."})
        
    csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
    if not csv_path.exists():
        return jsonify({"status": "error", "message": f"Cell masks CSV not found: {csv_path}"})
        
    df = pd.read_csv(csv_path)
    rle_col = 'rle_bf'
    if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
        rle_col = 'rle_gfp'
        
    tiles = []
    L = len(df)
    for t in range(L):
        rle = df.iloc[t][rle_col]
        tile = None
        if isinstance(rle, str) and rle.strip():
            tile = get_cell_crop_tile(exp, film, t, rle)
        if tile is None:
            tile = np.zeros((96, 96), dtype=np.uint8)
        tiles.append(tile)
        
    strip = np.hstack(tiles)
    
    try:
        probs = inference_runner.predict_strip(strip)
        if probs is not None:
            return jsonify({"status": "success", "probs": probs.tolist()})
        else:
            return jsonify({"status": "error", "message": "Model inference failed."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Inference exception: {str(e)}"})



def get_actual_film_and_t(args):
    exp = args.get("experiment")
    t = int(args.get("t"))
    if "sequence" in args:
        seq = args.get("sequence")
        gid = args.get("cell_id")
        return resolve_global_t(exp, seq, gid, t)
    return args.get("film"), int(args.get("cell_id")), t

@app.route("/api/frame_boundaries")
def frame_boundaries():
    film, _, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    
    masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
    files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files:
        files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        
    if not files:
        return jsonify({"error": "No segment file found"}), 404
        
    try:
        seg = load_segmentation(str(files[0]))
        from skimage.measure import label
        seg_lbl = (label(seg) if seg.dtype == bool else seg).copy()
        
        # Burn tracked local cells into seg_lbl so their outlines are visible for linking
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        if tracked_dir.is_dir():
            max_lbl = int(seg_lbl.max()) if seg_lbl.size > 0 else 0
            next_lbl = max_lbl + 100
            for cf in tracked_dir.iterdir():
                if cf.name.startswith("."): continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
                if m:
                    try:
                        df = pd.read_csv(cf)
                        if local_t < len(df):
                            H, W = seg_lbl.shape
                            for rle_col in ['rle_bf', 'rle_gfp']:
                                if rle_col in df.columns:
                                    rle = df.iloc[local_t][rle_col]
                                    if isinstance(rle, str) and rle.strip():
                                        source_col = 'source_bf' if rle_col == 'rle_bf' else 'source_gfp'
                                        is_manual = False
                                        if source_col in df.columns:
                                            is_manual = (df.iloc[local_t][source_col] == 'manual')
                                        if is_manual:
                                            mask = rle_decode(rle, (H, W))
                                            seg_lbl[mask] = next_lbl
                                            next_lbl += 1
                                            break
                    except Exception:
                        pass
        from skimage.segmentation import find_boundaries
        from scipy.ndimage import binary_dilation
        
        boundaries = find_boundaries(seg_lbl, mode='outer')
        thick_boundaries = binary_dilation(boundaries, structure=np.ones((3, 3)))
        H, W = seg_lbl.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[thick_boundaries] = [234, 179, 8, 140]
        
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgba, 'RGBA')
        img_io = BytesIO()
        pil_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/frame_image")
def frame_image():
    film, _, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    
    frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
    files = sorted([f for f in frames_dir.glob(f"{film}_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
    if not files:
        files = sorted([f for f in frames_dir.glob(f"*_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
        
    if not files:
        return jsonify({"error": f"Frame image not found"}), 404
        
    img = imread(str(files[0]))
    p_lo = np.percentile(img, 1.0)
    p_hi = np.percentile(img, 99.5)
    if p_hi > p_lo:
        img_scaled = np.clip((img - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_scaled = (img / img.max() * 255.0).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
        
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_scaled)
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route("/api/frame_crop")
def frame_crop():
    film, local_cid, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    channel = request.args.get("channel", "bf")
    
    if local_cid == -1:
        # Blank crop
        img_scaled = np.zeros((100, 100), dtype=np.uint8)
    else:
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_cid}_masks.csv"
        if not csv_path.exists():
            img_scaled = np.zeros((100, 100), dtype=np.uint8)
        else:
            df = pd.read_csv(csv_path)
            H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
            cy, cx = H // 2, W // 2
            
            if local_t < len(df):
                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
                    rle_col = 'rle_gfp'
                
                if rle_col in df.columns:
                    rle = df.iloc[local_t][rle_col]
                    if isinstance(rle, str) and rle.strip():
                        mask = rle_decode(rle, (H, W))
                        ys, xs = np.where(mask)
                        if len(ys) > 0:
                            cy, cx = int(np.mean(ys)), int(np.mean(xs))
                        
            frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
            files = sorted([f for f in frames_dir.glob(f"{film}_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
            if not files: files = sorted([f for f in frames_dir.glob(f"*_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
            if files:
                img = imread(str(files[0]))
                crop_size = 100
                y0 = max(0, cy - crop_size // 2); y1 = min(H, cy + crop_size // 2)
                x0 = max(0, cx - crop_size // 2); x1 = min(W, cx + crop_size // 2)
                crop = img[y0:y1, x0:x1]
                
                p_lo = np.percentile(crop, 1.0); p_hi = np.percentile(crop, 99.5)
                if p_hi > p_lo: img_scaled = np.clip((crop - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
                else: img_scaled = crop.astype(np.uint8)
            else:
                img_scaled = np.zeros((100, 100), dtype=np.uint8)
        
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_scaled)
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route("/api/population_frame")
def population_frame():
    film, _, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    
    cache_dir = BASE_MOVIE_ROOT / exp / film / f"PopulationFrames_{film}"
    cache_file = cache_dir / f"frame_{local_t:03d}.jpg"
    
    if cache_file.exists():
        return send_file(str(cache_file), mimetype='image/jpeg')
        
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        img_data = generate_population_frame_image(exp, film, local_t)
        if img_data is not None:
            with open(cache_file, "wb") as f:
                f.write(img_data)
            img_io = BytesIO(img_data)
            return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    return jsonify({"error": "Failed to generate population frame"}), 404

@app.route("/api/click_segment", methods=["POST"])
def click_segment():
    data = request.json
    exp = data.get("experiment")
    t = int(data.get("t"))
    x, y = int(data.get("x")), int(data.get("y"))
    
    if "sequence" in data:
        seq = data.get("sequence")
        gid = data.get("cell_id")
        film, local_cid, local_t = resolve_global_t(exp, seq, gid, t)
    else:
        film, local_cid, local_t = data.get("film"), int(data.get("cell_id")), t
        
    if local_cid == -1:
        return jsonify({"status": "error", "message": "Cannot select segment for an unassigned cell mapping."})
        
    masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
    files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files: files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files: return jsonify({"status": "error", "message": "Segmentation file not found"}), 404
        
    seg = load_segmentation(str(files[0]))
    seg_lbl = label(seg) if seg.dtype == bool else seg
    H, W = seg_lbl.shape
    if y >= H or x >= W: return jsonify({"status": "error", "message": "Click coordinates out of range"}), 400
        
    lbl = seg_lbl[y, x]
    if lbl == 0: return jsonify({"status": "success", "rle": ""})
        
    segment_mask = (seg_lbl == lbl)
    rle = rle_encode(segment_mask)
    return jsonify({"status": "success", "rle": rle})

@app.route("/api/save_masks", methods=["POST"])
def save_masks():
    data = request.json
    exp = data.get("experiment")
    cell_id = data.get("cell_id")
    channel = data.get("channel", "bf")
    new_masks = data.get("masks")
    
    # Partial update of suspicious cells cache
    seq = data.get("sequence")
    film_param = data.get("film")
    target = seq if seq else film_param
    if exp and target and cell_id:
        try:
            target_dir = BASE_MOVIE_ROOT / exp / target
            cache_file = target_dir / f"suspicious_{target}.json"
            
            disk_data = {}
            if cache_file.exists():
                import json
                with open(cache_file, "r") as f:
                    disk_data = json.load(f)
                    
            if seq:
                seq_data = get_sequence_linkage_data(exp)
                ensure_pseudo_sequence_cells(exp, seq, seq_data)
                films = seq_data[seq]["films"]
                local_ids = seq_data[seq]["global_cells"].get(str(cell_id), [-1]*len(films))
            else:
                films = [film_param]
                local_ids = [int(cell_id)]
                
            film_dfs = []
            for i, f_name in enumerate(films):
                local_id = local_ids[i] if i < len(local_ids) else -1
                if local_id == -1:
                    film_dfs.append(None)
                    continue
                csv_path = BASE_MOVIE_ROOT / exp / f_name / f"TrackedCells_{f_name}" / f"cell_{local_id}_masks.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        film_dfs.append(df)
                    except Exception:
                        film_dfs.append(None)
                else:
                    film_dfs.append(None)
                    
            all_rles = []
            H, W = 0, 0
            for i, df in enumerate(film_dfs):
                f_name = films[i]
                L, fW, fH = get_film_frame_count_and_size(exp, f_name)
                if df is not None and len(df) > 0:
                    if H == 0:
                        H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
                    
                    rle_col = 'rle_bf'
                    if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                        rle_col = 'rle_gfp'
                    elif 'rle_bf' not in df.columns and 'rle_gfp' in df.columns:
                        rle_col = 'rle_gfp'
                        
                    masks = df[rle_col].fillna("").tolist()
                    if len(masks) < L:
                        masks.extend([""] * (L - len(masks)))
                    elif len(masks) > L:
                        masks = masks[:L]
                    all_rles.extend(masks)
                else:
                    all_rles.extend([""] * L)
                    
            if H > 0 and W > 0:
                centroids = []
                for rle in all_rles:
                    if not isinstance(rle, str) or not rle.strip() or rle == "nan":
                        centroids.append(None)
                        continue
                    try:
                        mask = rle_decode(rle, (H, W))
                        if not mask.any():
                            centroids.append(None)
                        else:
                            ys, xs = np.nonzero(mask)
                            centroids.append((float(np.mean(ys)), float(np.mean(xs))))
                    except Exception:
                        centroids.append(None)
                        
                susp_frames = []
                threshold = 15.0
                for t in range(1, len(centroids)):
                    c1 = centroids[t-1]
                    c2 = centroids[t]
                    if c1 is not None and c2 is not None:
                        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        if dist > threshold:
                            susp_frames.append(t)
                            
                if susp_frames:
                    disk_data[str(cell_id)] = susp_frames
                else:
                    if str(cell_id) in disk_data:
                        del disk_data[str(cell_id)]
                        
            target_dir.mkdir(parents=True, exist_ok=True)
            import json
            with open(cache_file, "w") as f:
                json.dump(disk_data, f)
                
            for k in list(SUSPICIOUS_CACHE.keys()):
                if k.startswith(f"{exp}::{target}"):
                    SUSPICIOUS_CACHE[k] = disk_data
                    
        except Exception as e:
            print(f"Error updating suspicious cache: {e}")
    
    if "sequence" in data:
        seq = data.get("sequence")
        seq_data = get_sequence_linkage_data(exp)
        ensure_pseudo_sequence_cells(exp, seq, seq_data)
        films = seq_data[seq]["films"]
        local_ids = seq_data[seq]["global_cells"][cell_id]
        
        current_t = 0
        for i, film in enumerate(films):
            L, _, _ = get_film_frame_count_and_size(exp, film)
            film_masks = new_masks[current_t:current_t+L]
            local_id = local_ids[i]
            
            if local_id != -1:
                csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_id}_masks.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if 'rle_gfp' in df.columns and 'rle_bf' not in df.columns:
                        rle_col = 'rle_gfp'
                    elif 'rle_bf' in df.columns and 'rle_gfp' not in df.columns:
                        rle_col = 'rle_bf'
                    else:
                        rle_col = 'rle_bf' if channel == 'bf' else 'rle_gfp'
                        
                    source_col = 'source_bf' if rle_col == 'rle_bf' else 'source_gfp'
                    area_col = 'area_bf' if rle_col == 'rle_bf' else 'area_gfp'
                    
                    if rle_col not in df.columns:
                        df[rle_col] = ""
                    if source_col not in df.columns:
                        df[source_col] = ""
                    
                    if len(df) > len(film_masks):
                        film_masks.extend([""] * (len(df) - len(film_masks)))
                    elif len(df) < len(film_masks):
                        film_masks = film_masks[:len(df)]
                        
                    any_modified = False
                    modified_t_indices = []
                    for t in range(len(df)):
                        old_rle = df.loc[t, rle_col] if pd.notna(df.loc[t, rle_col]) else ""
                        new_rle = film_masks[t] if film_masks[t] is not None else ""
                        if old_rle != new_rle:
                            df.loc[t, rle_col] = new_rle
                            df.loc[t, source_col] = "manual"
                            any_modified = True
                            modified_t_indices.append(t)
                            
                        rle = film_masks[t]
                        H, W = int(df.iloc[t]['height']), int(df.iloc[t]['width'])
                        if isinstance(rle, str) and rle.strip():
                            mask = rle_decode(rle, (H, W))
                            area = int(mask.sum())
                        else:
                            area = 0
                        if area_col in df.columns: df.loc[t, area_col] = area
                    df.to_csv(csv_path, index=False)
                    
                    if any_modified:
                        for t_idx in modified_t_indices:
                            cache_dir = BASE_MOVIE_ROOT / exp / f_name / f"PopulationFrames_{f_name}"
                            cache_file = cache_dir / f"frame_{t_idx:03d}.jpg"
                            if cache_file.exists():
                                try:
                                    cache_file.unlink()
                                except Exception:
                                    pass
                            
            current_t += L
            
    else:
        film = data.get("film")
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
        df = pd.read_csv(csv_path)
        
        if 'rle_gfp' in df.columns and 'rle_bf' not in df.columns:
            rle_col = 'rle_gfp'
        elif 'rle_bf' in df.columns and 'rle_gfp' not in df.columns:
            rle_col = 'rle_bf'
        else:
            rle_col = 'rle_bf' if channel == 'bf' else 'rle_gfp'
            
        source_col = 'source_bf' if rle_col == 'rle_bf' else 'source_gfp'
        area_col = 'area_bf' if rle_col == 'rle_bf' else 'area_gfp'
        
        if rle_col not in df.columns:
            df[rle_col] = ""
        if source_col not in df.columns:
            df[source_col] = ""
        
        any_modified = False
        modified_t_indices = []
        for t in range(len(df)):
            if t < len(new_masks):
                old_rle = df.loc[t, rle_col] if pd.notna(df.loc[t, rle_col]) else ""
                new_rle = new_masks[t] if new_masks[t] is not None else ""
                if old_rle != new_rle:
                    df.loc[t, rle_col] = new_rle
                    df.loc[t, source_col] = "manual"
                    any_modified = True
                    modified_t_indices.append(t)
                
                rle = new_masks[t]
                H, W = int(df.iloc[t]['height']), int(df.iloc[t]['width'])
                if isinstance(rle, str) and rle.strip():
                    mask = rle_decode(rle, (H, W))
                    area = int(mask.sum())
                else:
                    area = 0
                if area_col in df.columns: df.loc[t, area_col] = area
        df.to_csv(csv_path, index=False)
        
        if any_modified:
            for t_idx in modified_t_indices:
                cache_dir = BASE_MOVIE_ROOT / exp / film / f"PopulationFrames_{film}"
                cache_file = cache_dir / f"frame_{t_idx:03d}.jpg"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
        
    return jsonify({"status": "success"})
@app.route("/api/auto_fix_segments", methods=["POST"])
def auto_fix_segments():
    data = request.json
    exp = data.get("experiment")
    start_t = int(data.get("start_t"))
    end_t = int(data.get("end_t"))
    
    fixed_count = 0
    modified_dfs = {}
    
    for t in range(start_t, end_t + 1):
        frame_data = dict(data)
        frame_data["t"] = t
        film, local_cid, local_t = get_actual_film_and_t(frame_data)
        
        if local_cid == -1:
            continue
            
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_cid}_masks.csv"
        
        if csv_path not in modified_dfs:
            if not csv_path.exists():
                continue
            modified_dfs[csv_path] = pd.read_csv(csv_path)
            
        df = modified_dfs[csv_path]
        
        if local_t >= len(df):
            continue
            
        H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
        
        if 'rle_gfp' in df.columns and 'rle_bf' not in df.columns:
            rle_col = 'rle_gfp'
            source_col = 'source_gfp'
        elif 'rle_bf' in df.columns and 'rle_gfp' not in df.columns:
            rle_col = 'rle_bf'
            source_col = 'source_bf'
        else:
            rle_col = 'rle_bf'
            source_col = 'source_bf'
            if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
                rle_col = 'rle_gfp'
                source_col = 'source_gfp'
            
        if source_col not in df.columns:
            df[source_col] = ""
            
        if rle_col not in df.columns:
            continue
            
        existing_rle = df.loc[local_t, rle_col]
        if not isinstance(existing_rle, str) or not str(existing_rle).strip() or str(existing_rle) == "nan":
            continue
            
        existing_mask = rle_decode(str(existing_rle), (H, W))
        if not existing_mask.any():
            continue
            
        masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
        files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files: files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files:
            continue
            
        raw_seg = imread(str(files[0]))
        
        overlapping_labels, counts = np.unique(raw_seg[existing_mask], return_counts=True)
        
        selected_labels = []
        best_label = 0
        max_iou = 0.0
        existing_area = existing_mask.sum()
        
        for label, count in zip(overlapping_labels, counts):
            if label == 0: continue
            raw_area = np.sum(raw_seg == label)
            coverage = count / raw_area
            iou = count / (existing_area + raw_area - count)
            
            if coverage >= 0.4:
                selected_labels.append(label)
                
            if iou > max_iou:
                max_iou = iou
                best_label = label
                
        if not selected_labels and best_label > 0:
            selected_labels.append(best_label)
            
        if selected_labels:
            new_mask = np.isin(raw_seg, selected_labels)
            new_rle = rle_encode(new_mask)
            df.loc[local_t, rle_col] = new_rle
            df.loc[local_t, source_col] = "manual"
            area_col = 'area_bf' if rle_col == 'rle_bf' else 'area_gfp'
            if area_col in df.columns: df.loc[local_t, area_col] = int(new_mask.sum())
            fixed_count += 1
            
    for csv_path, df in modified_dfs.items():
        df.to_csv(csv_path, index=False)
        
    return jsonify({"status": "success", "fixed_count": fixed_count})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fungal Cell Tracking Corrector")
    parser.add_argument("--sync-nas", type=str, nargs='?', const='all',
                        help="Sync data from NAS to local SSD. Can specify a specific experiment name (e.g. 2026_01_08_M93) or leave blank/'all' to sync everything.")
    parser.add_argument("--nas-root", type=str, default=str(NAS_MOVIE_ROOT),
                        help="Path to the NAS movie directory")
    parser.add_argument("--local-root", type=str, default=str(BASE_MOVIE_ROOT),
                        help="Path to the local SSD movie directory")
    args = parser.parse_args()
    
    BASE_MOVIE_ROOT = Path(args.local_root)
    NAS_MOVIE_ROOT = Path(args.nas_root)
    
    if args.sync_nas:
        print("🔄 Initiating NAS to Local SSD Sync (Pull)...")
        local_path = Path(args.local_root)
        nas_path = Path(args.nas_root)
        
        local_path.mkdir(parents=True, exist_ok=True)
        
        sync_list = RELEVANT_EXPERIMENTS if args.sync_nas == 'all' else [args.sync_nas]
        pull_errors = False
        
        for exp in sync_list:
            src = str(nas_path / exp) + "/"
            dst = str(local_path / exp) + "/"
            
            print(f"Pulling from NAS: {src} -> Local SSD: {dst}")
            
            if not os.path.exists(src.rstrip("/")):
                print(f"⚠️ Warning: NAS directory '{src}' does not exist. Skipping.")
                continue
                
            local_path_exp = local_path / exp
            local_path_exp.mkdir(parents=True, exist_ok=True)
            
            rsync_cmd = f"rsync -avz --update --exclude='__pycache__' --exclude='*.ims' '{src}' '{dst}'"
            try:
                subprocess.run(rsync_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Pull Sync Failed for {exp}: {e}")
                pull_errors = True
                
        if pull_errors:
            print("⚠️ Pull Sync finished with errors.")
            sys.exit(1)
        else:
            print("✅ Pull Sync Completed Successfully!")
            
    port = 5001
    print(f"🚀 Starting Corrector Tool at http://127.0.0.1:{port}")
    
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    finally:
        if args.sync_nas:
            print("\n🔄 Initiating Local SSD to NAS Sync (Push)...")
            local_path = Path(args.local_root)
            nas_path = Path(args.nas_root)
            
            sync_list = RELEVANT_EXPERIMENTS if args.sync_nas == 'all' else [args.sync_nas]
            
            for exp in sync_list:
                src = str(local_path / exp) + "/"
                dst = str(nas_path / exp) + "/"
                
                if not os.path.exists(src.rstrip("/")):
                    continue
                    
                print(f"Pushing from Local SSD: {src} -> NAS: {dst}")
                
                nas_path_exp = nas_path / exp
                nas_path_exp.mkdir(parents=True, exist_ok=True)
                
                rsync_cmd = f"rsync -avz --update --exclude='__pycache__' --exclude='*.ims' '{src}' '{dst}'"
                try:
                    subprocess.run(rsync_cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Push Sync Failed for {exp}: {e}")
            print("✅ Push Sync Completed Successfully!")

