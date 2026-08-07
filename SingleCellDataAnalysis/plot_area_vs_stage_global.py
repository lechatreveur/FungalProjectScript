import os
import json
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript')
from Cell_tracking_functions import rle_decode

MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies/2025_09_17")
ARTIFACTS_DIR = Path("/Users/user/.gemini/antigravity-ide/brain/32e12c76-45ee-4552-bcd9-4d4c3033438f")

# Define the film sequence and durations
FILM_SEQUENCE = [
    "A14_1TP1_F1",     # GFP1
    "A14_1TP1_BF_F1",  # BF1
    "A14_1TP2_F1",     # GFP2
    "A14_1TP2_BF_F1"   # BF2
]

# Film start times in minutes relative to the start of the first film (GFP1)
# GFP1 (20m) -> BF1 (120m) -> GFP2 (20m) -> BF2 (120m)
FILM_START_TIMES = [0.0, 20.0, 140.0, 160.0]
FILM_TIME_RES = [0.2, 1.0, 0.2, 1.0]

def main():
    # Load sequence linkages
    linkage_path = MOVIE_ROOT / "sequence_linkage.json"
    with open(linkage_path) as f:
        linkage = json.load(f)
    
    global_cells = linkage["F1"]["global_cells"]
    
    # Load curated status
    qc_path = MOVIE_ROOT / "F1" / "qc_F1.json"
    with open(qc_path) as f:
        qc_data = json.load(f)
        
    # Load septum alignments for each film
    alignments = {}
    for i, film in enumerate(FILM_SEQUENCE):
        json_path = MOVIE_ROOT / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels" / "global_septum_alignment.json"
        if json_path.exists():
            with open(json_path) as f:
                alignments[i] = json.load(f)
                
    data_points = []
    
    # First Pass: collect cells WITH a septum endpoint and determine their division time
    has_div = {}
    for gcid, local_ids in global_cells.items():
        global_div_time = None
        for k, lcid in enumerate(local_ids):
            if lcid == -1 or k not in alignments:
                continue
            offsets = alignments[k].get("offsets", {})
            intervals = alignments[k].get("cell_intervals", {})
            lcid_str = str(lcid)
            
            if lcid_str in intervals:
                interval = intervals[lcid_str]
                if interval.get("has_septum"):
                    end_val = interval.get("end_aligned")
                    if end_val is not None and end_val != -1 and str(end_val) != "-1":
                        offset = int(offsets.get(lcid_str, 0))
                        local_end = int(end_val) - offset
                        
                        # Calculate global time of division
                        div_time = FILM_START_TIMES[k] + local_end * FILM_TIME_RES[k]
                        if global_div_time is None:
                            global_div_time = div_time
                            
        if global_div_time is not None:
            has_div[gcid] = global_div_time
            
            # Extract area from all linked local films for this cell
            for k, lcid in enumerate(local_ids):
                if lcid == -1:
                    continue
                film = FILM_SEQUENCE[k]
                is_bf = "BF" in film
                masks_csv_path = MOVIE_ROOT / film / f"TrackedCells_{film}" / f"cell_{lcid}_masks.csv"
                
                if masks_csv_path.exists():
                    df_masks = pd.read_csv(masks_csv_path)
                    for _, row in df_masks.iterrows():
                        t = int(row['time_point'])
                        rle_col = None
                        candidates = ['rle_bf', 'rle_gfp'] if is_bf else ['rle_gfp', 'rle_bf']
                        for col in candidates:
                            if col in row and isinstance(row[col], str) and row[col].strip():
                                rle_col = col
                                break
                        if rle_col is not None:
                            rle_val = row[rle_col]
                            h, w = int(row['height']), int(row['width'])
                            mask = rle_decode(rle_val, (h, w))
                            area = int(mask.sum())
                            
                            if area > 0:
                                global_frame_time = FILM_START_TIMES[k] + t * FILM_TIME_RES[k]
                                data_points.append({
                                    'global_cell_id': gcid,
                                    'stage_min': global_frame_time - global_div_time,
                                    'area': area,
                                    'channel': 'BF' if is_bf else 'GFP',
                                    'group': 'with_septum'
                                })
                                
    # Calculate fit lines from the pre-division points of the 'with_septum' cells
    fits = {}
    for ch in ['BF', 'GFP']:
        sub = [p for p in data_points if p['channel'] == ch and p['group'] == 'with_septum' and p['stage_min'] <= 0]
        if len(sub) > 1:
            x_pre = np.array([p['stage_min'] for p in sub])
            y_pre = np.array([p['area'] for p in sub])
            slope, intercept = np.polyfit(x_pre, y_pre, 1)
            r = np.corrcoef(x_pre, y_pre)[0, 1]
            fits[ch] = {'slope': slope, 'intercept': intercept, 'r': r}
        else:
            fits[ch] = None
            
    print(f"Fits calculated: {fits}")
    
    # Second Pass: collect curated cells WITHOUT a septum endpoint and align them using regression fits
    no_septum_aligned_count = 0
    
    for gcid, local_ids in global_cells.items():
        if gcid in has_div:
            continue
            
        status = qc_data.get(gcid)
        if status not in ['good', 'corrected']:
            continue
            
        # For this cell, load all its frames for each channel separately
        no_septum_aligned_count += 1
        
        # We group frames by channel
        channel_frames = {'BF': [], 'GFP': []}
        
        for k, lcid in enumerate(local_ids):
            if lcid == -1:
                continue
            film = FILM_SEQUENCE[k]
            is_bf = "BF" in film
            masks_csv_path = MOVIE_ROOT / film / f"TrackedCells_{film}" / f"cell_{lcid}_masks.csv"
            
            if masks_csv_path.exists():
                df_masks = pd.read_csv(masks_csv_path)
                for _, row in df_masks.iterrows():
                    t = int(row['time_point'])
                    rle_col = None
                    candidates = ['rle_bf', 'rle_gfp'] if is_bf else ['rle_gfp', 'rle_bf']
                    for col in candidates:
                        if col in row and isinstance(row[col], str) and row[col].strip():
                            rle_col = col
                            break
                    if rle_col is not None:
                        rle_val = row[rle_col]
                        h, w = int(row['height']), int(row['width'])
                        mask = rle_decode(rle_val, (h, w))
                        area = int(mask.sum())
                        
                        if area > 0:
                            global_frame_time = FILM_START_TIMES[k] + t * FILM_TIME_RES[k]
                            ch = 'BF' if is_bf else 'GFP'
                            channel_frames[ch].append((global_frame_time, area))
                            
        # For each channel, find optimal shift tau and align
        for ch in ['BF', 'GFP']:
            frames = channel_frames[ch]
            if len(frames) > 0 and fits[ch] is not None:
                T = np.array([f[0] for f in frames])
                A = np.array([f[1] for f in frames])
                
                mean_T = np.mean(T)
                mean_A = np.mean(A)
                
                m = fits[ch]['slope']
                c = fits[ch]['intercept']
                
                # Calculate optimal shift tau
                tau = mean_T - (mean_A - c) / m
                
                # Shift all frames
                T_aligned = T - tau
                
                for aligned_t, area_val in zip(T_aligned, A):
                    data_points.append({
                        'global_cell_id': gcid,
                        'stage_min': aligned_t,
                        'area': area_val,
                        'channel': ch,
                        'group': 'no_septum'
                    })
                    
    print(f"Aligned {no_septum_aligned_count} curated cells without a division endpoint.")
    
    df_plot = pd.DataFrame(data_points)
    print(f"Total data points collected: {len(df_plot)}")
    print(df_plot['group'].value_counts())
    
    # Save CSV of the data points for verification/future use
    df_plot.to_csv(ARTIFACTS_DIR / "global_area_vs_stage_data.csv", index=False)
    
    # Create plot
    plt.figure(figsize=(11, 7))
    
    pre_colors = {'BF': 'darkred', 'GFP': 'darkgreen'}
    post_colors = {'BF': 'red', 'GFP': 'green'}
    no_septum_colors = {'BF': 'orange', 'GFP': 'limegreen'}
    
    for ch in ['BF', 'GFP']:
        sub = df_plot[df_plot['channel'] == ch]
        if sub.empty:
            continue
            
        # Plot pre-division (t <= 0)
        pre_sub = sub[(sub['stage_min'] <= 0) & (sub['group'] == 'with_septum')]
        if len(pre_sub) > 1:
            x_pre = pre_sub['stage_min'].values
            y_pre = pre_sub['area'].values
            plt.scatter(x_pre, y_pre, alpha=0.15, c=pre_colors[ch], edgecolors='none', s=8, label=f'{ch} Pre-div')
            
            # Plot the pre-division fit line
            fit_info = fits[ch]
            if fit_info is not None:
                x_fit = np.linspace(x_pre.min(), x_pre.max(), 100)
                plt.plot(x_fit, fit_info['slope']*x_fit + fit_info['intercept'], color=pre_colors[ch], linewidth=2.5, linestyle='-',
                         label=f"{ch} Pre Fit (r = {fit_info['r']:.3f}, slope = {fit_info['slope']:.1f})")
                         
        # Plot post-division (t > 0)
        post_sub = sub[(sub['stage_min'] > 0) & (sub['group'] == 'with_septum')]
        if not post_sub.empty:
            plt.scatter(post_sub['stage_min'].values, post_sub['area'].values, alpha=0.15, c=post_colors[ch], edgecolors='none', s=8,
                        label=f'{ch} Post-div')
                        
        # Plot curated cells without division (no_septum)
        no_div_sub = sub[sub['group'] == 'no_septum']
        if not no_div_sub.empty:
            plt.scatter(no_div_sub['stage_min'].values, no_div_sub['area'].values, alpha=0.08, c=no_septum_colors[ch], edgecolors='none', s=6,
                        label=f'{ch} Curated (no septum, aligned)')
                        
    plt.title("Continuous Fission Yeast Cell Area vs Time Relative to Division", fontsize=12, fontweight='bold')
    plt.xlabel("Time relative to division (minutes)")
    plt.ylabel("Cell Area (px²)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    out_img = ARTIFACTS_DIR / "cell_area_vs_stage_global.png"
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_img}")

if __name__ == '__main__':
    main()
