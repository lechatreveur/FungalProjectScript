import os
import json
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from concurrent.futures import ProcessPoolExecutor

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript')
from Cell_tracking_functions import rle_decode

PIXEL_SIZE = 6.5 / 63 # 0.1031746 um/pixel

def axis_extrema_points(boundary_coords, P0, direction, tol=2.0):
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("`direction` must be nonzero.")
    d = direction / norm  # ensure unit vector

    rel = boundary_coords - P0  # (M, 2)
    t = rel @ d                # (M,)
    q_line = P0 + np.outer(t, d)
    dist = np.linalg.norm(boundary_coords - q_line, axis=1)

    on_axis = dist < tol
    if not np.any(on_axis):
        on_axis = dist < (tol * 2.0)
    if not np.any(on_axis):
        on_axis = np.ones(len(dist), dtype=bool)

    t_sel = t[on_axis]
    pts_sel = boundary_coords[on_axis]

    i_min = np.argmin(t_sel)
    i_max = np.argmax(t_sel)
    return pts_sel[i_min], pts_sel[i_max], float(t_sel[i_min]), float(t_sel[i_max])

def get_endpoint_length_from_mask(mask_rle, h, w):
    try:
        mask = rle_decode(mask_rle, (h, w))
        labeled = label(mask)
        props = regionprops(labeled)
        if len(props) == 0:
            return None
        props = sorted(props, key=lambda x: x.area, reverse=True)
        r = props[0]
        
        P0 = np.array(r.centroid)
        orientation = r.orientation
        d = np.array([np.cos(orientation), np.sin(orientation)])
        boundary_mask = find_boundaries(mask, mode='inner')
        boundary_coords = np.column_stack(np.nonzero(boundary_mask))
        
        ep1, ep2, _, _ = axis_extrema_points(boundary_coords, P0, d, tol=2.0)
        tip_len_px = np.linalg.norm(np.array(ep1) - np.array(ep2))
        return tip_len_px
    except Exception:
        return None

def process_single_frame(task):
    mask_rle, h, w = task['mask_rle'], task['height'], task['width']
    px_len = get_endpoint_length_from_mask(mask_rle, h, w)
    if px_len is not None:
        return px_len * PIXEL_SIZE
    return None

def collect_tasks_for_exp(exp_root, is_m135=False):
    exp_root = Path(exp_root)
    linkage_path = exp_root / "sequence_linkage.json"
    if not linkage_path.exists():
        return []
        
    with open(linkage_path) as f:
        linkage = json.load(f)
        
    tasks = []
    
    # Define start times and resolutions
    if is_m135:
        fields = list(linkage.keys())
        FILM_START_TIMES = [0.0, 20.0, 140.0, 160.0, 280.0]
        FILM_TIME_RES = [0.2, 2.0, 0.2, 2.0, 0.2]
    else:
        fields = ["F0", "F1"]
        FILM_START_TIMES = [0.0, 20.0, 140.0, 160.0]
        FILM_TIME_RES = [0.2, 1.0, 0.2, 1.0]
        
    for seq in fields:
        seq_link = linkage.get(seq)
        if not seq_link: continue
        global_cells = seq_link['global_cells']
        films = seq_link['films']
        
        # Load QC file
        qc_path = exp_root / seq / f"qc_{seq}.json"
        if not qc_path.exists():
            continue
        with open(qc_path) as f:
            qc_data = json.load(f)
            
        # Load alignments
        alignments = {}
        for k, film in enumerate(films):
            json_path = exp_root / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels" / "global_septum_alignment.json"
            if json_path.exists():
                with open(json_path) as f:
                    alignments[film] = json.load(f)
                    
        for gcid, local_ids in global_cells.items():
            status = qc_data.get(gcid)
            if status not in ['good', 'corrected']:
                continue
                
            # Find the movie and frame of division
            global_div_time = None
            for k, lcid in enumerate(local_ids):
                if is_m135 and k == 4: # Skip GFP3
                    continue
                film = films[k]
                if lcid == -1 or film not in alignments:
                    continue
                align = alignments[film]
                offsets = align.get("offsets", {})
                intervals = align.get("cell_intervals", {})
                lcid_str = str(lcid)
                
                if lcid_str in intervals:
                    interval = intervals[lcid_str]
                    if interval.get("has_septum"):
                        end_val = interval.get("end_aligned")
                        if end_val is not None and end_val != -1 and str(end_val) != "-1":
                            offset = int(offsets.get(lcid_str, 0))
                            local_end = int(end_val) - offset
                            global_div_time = FILM_START_TIMES[k] + local_end * FILM_TIME_RES[k]
                            break
                            
            if global_div_time is not None:
                # Retrieve all masks for this cell and collect tasks
                for k, lcid in enumerate(local_ids):
                    if is_m135 and k == 4:
                        continue
                    if lcid == -1:
                        continue
                    film = films[k]
                    masks_csv = exp_root / film / f"TrackedCells_{film}" / f"cell_{lcid}_masks.csv"
                    if masks_csv.exists():
                        try:
                            df_m = pd.read_csv(masks_csv)
                            for _, row in df_m.iterrows():
                                t = int(row['time_point'])
                                is_bf = "BF" in film
                                rle_col = None
                                candidates = ['rle_bf', 'rle_gfp'] if is_bf else ['rle_gfp', 'rle_bf']
                                for col in candidates:
                                    if col in row and isinstance(row[col], str) and row[col].strip():
                                        rle_col = col
                                        break
                                if rle_col is not None:
                                    gtime = FILM_START_TIMES[k] + t * FILM_TIME_RES[k]
                                    rel_time = gtime - global_div_time
                                    
                                    # We only care about -30 to 0 minutes relative to division
                                    if -30.5 <= rel_time <= 0.5:
                                        tasks.append({
                                            'gcid': gcid,
                                            'rel_time': rel_time,
                                            'mask_rle': row[rle_col],
                                            'height': int(row['height']),
                                            'width': int(row['width']),
                                            'experiment': 'M135' if is_m135 else 'Sept17',
                                            'channel': 'BF' if is_bf else 'GFP'
                                        })
                        except Exception:
                            continue
    return tasks

def main():
    out_dir = Path("/Users/user/.gemini/antigravity-ide/brain/32e12c76-45ee-4552-bcd9-4d4c3033438f")
    csv_path = out_dir / "time_course_lengths_microns.csv"
    
    # Reload existing data only if the CSV contains the 'channel' column
    if csv_path.exists():
        df_all = pd.read_csv(csv_path)
        if 'channel' not in df_all.columns:
            print("Cached data lacks 'channel' column. Rerunning computation...")
            df_all = None
    else:
        df_all = None
        
    if df_all is None:
        print("Collecting candidate tasks for Sept17...")
        sept17_tasks = collect_tasks_for_exp("/Volumes/X10 Pro/Movies/2025_09_17", is_m135=False)
        
        print("Collecting candidate tasks for M135...")
        m135_tasks = collect_tasks_for_exp("/Volumes/X10 Pro/Movies/2026_04_30_M135", is_m135=True)
        
        all_tasks = sept17_tasks + m135_tasks
        total_frames = len(all_tasks)
        print(f"Total tasks collected: {total_frames}. Starting parallel execution...")
        
        # Process tasks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            for idx, length_um in enumerate(executor.map(process_single_frame, all_tasks)):
                if length_um is not None:
                    task = all_tasks[idx]
                    results.append({
                        'gcid': task['gcid'],
                        'rel_time': task['rel_time'],
                        'length_um': length_um,
                        'experiment': task['experiment'],
                        'channel': task['channel']
                    })
                if (idx + 1) % 1000 == 0 or idx + 1 == total_frames:
                    print(f"Processed {idx + 1}/{total_frames} frames...")
                    
        df_all = pd.DataFrame(results)
        df_all.to_csv(csv_path, index=False)
        print(f"Saved raw time course data to {csv_path}")
        
    # FILTER FOR GFP data only
    df_gfp = df_all[df_all['channel'] == 'GFP'].copy()
    print(f"Plotting GFP data only. Remaining data points: {len(df_gfp)}")
    
    plt.figure(figsize=(11, 6.5))
    colors = {'Sept17': 'royalblue', 'M135': 'darkorange'}
    
    for exp in ['Sept17', 'M135']:
        sub = df_gfp[df_gfp['experiment'] == exp].copy()
        if sub.empty:
            continue
            
        # For GFP:
        # Round relative times to the nearest 0.2 minutes (12s resolution)
        sub['time_round'] = np.round(sub['rel_time'] / 0.2) * 0.2
        
        # Calculate mean and standard deviation per unique 0.2-minute time point
        grouped = sub.groupby('time_round', observed=False)['length_um'].agg(['mean', 'std', 'count']).reset_index()
        grouped['time_round'] = grouped['time_round'].astype(float)
        
        # Filter points with at least 3 cells to avoid noise
        grouped = grouped[grouped['count'] >= 3]
        
        # Calculate standard error of the mean (sem)
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
        
        # Plot continuous line with markers showing GFP time resolution (12s / 0.2 min)
        plt.plot(grouped['time_round'], grouped['mean'], color=colors[exp], 
                 label=f"{exp} Mean (GFP)", linewidth=1.8, marker='o', markersize=3, alpha=0.9)
                 
        # Add shaded 1.96 SEM region
        plt.fill_between(
            grouped['time_round'], 
            grouped['mean'] - 1.96 * grouped['sem'], 
            grouped['mean'] + 1.96 * grouped['sem'], 
            color=colors[exp], 
            alpha=0.15,
            label=f"{exp} ± 1.96 SEM"
        )
        
    plt.title("GFP-Only Cell Length Elongation kinetics Leading to Division (t = 0)", fontsize=14, fontweight='bold')
    plt.xlabel("Time Relative to Division (minutes)", fontsize=12)
    plt.ylabel("Cell Length (µm)", fontsize=12)
    plt.xlim(-30, 0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    out_plot = out_dir / "length_time_course_with_markers.png"
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved GFP-only time-course plot to {out_plot}")

if __name__ == '__main__':
    main()
