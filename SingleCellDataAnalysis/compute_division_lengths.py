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

def get_division_and_pre10_lengths(exp_root, is_m135=False):
    exp_root = Path(exp_root)
    linkage_path = exp_root / "sequence_linkage.json"
    if not linkage_path.exists():
        return []
        
    with open(linkage_path) as f:
        linkage = json.load(f)
        
    lengths = []
    
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
            div_film_k = None
            div_local_frame = None
            
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
                            div_film_k = k
                            div_local_frame = local_end
                            break
                            
            if global_div_time is not None:
                # Target time is 10 minutes before division
                target_time = global_div_time - 10.0
                
                # Scan all available masks for this cell across all movies to find the closest frame to target_time
                best_dist = float('inf')
                best_mask_info = None
                best_channel = None
                
                # Also collect division frame info
                div_mask_info = None
                
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
                                    
                                    # Check division frame
                                    if k == div_film_k and t == div_local_frame:
                                        div_mask_info = (row[rle_col], int(row['height']), int(row['width']))
                                        
                                    # Check distance to target_time
                                    dist = abs(gtime - target_time)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_mask_info = (row[rle_col], int(row['height']), int(row['width']))
                                        best_channel = 'BF' if is_bf else 'GFP'
                        except Exception:
                            continue
                            
                # Compute lengths if both division and pre-10 minute masks were found
                if div_mask_info is not None and best_mask_info is not None and best_dist < 4.0:
                    try:
                        div_px = get_endpoint_length_from_mask(*div_mask_info)
                        pre10_px = get_endpoint_length_from_mask(*best_mask_info)
                        
                        if div_px is not None and pre10_px is not None:
                            lengths.append({
                                'gcid': gcid,
                                'channel': best_channel,
                                'div_um': div_px * PIXEL_SIZE,
                                'pre10_um': pre10_px * PIXEL_SIZE,
                                'diff_um': (div_px - pre10_px) * PIXEL_SIZE,
                                'time_diff_min': best_dist
                            })
                    except Exception as e:
                        print(f"Error computing length for {gcid}: {e}")
                        continue
    return lengths

def main():
    print("Computing Sept17 division and pre-10 lengths...")
    sept17_lengths = get_division_and_pre10_lengths("/Volumes/X10 Pro/Movies/2025_09_17", is_m135=False)
    
    print("Computing M135 division and pre-10 lengths...")
    m135_lengths = get_division_and_pre10_lengths("/Volumes/X10 Pro/Movies/2026_04_30_M135", is_m135=True)
    
    df_s = pd.DataFrame(sept17_lengths)
    df_s['experiment'] = 'Sept17'
    df_m = pd.DataFrame(m135_lengths)
    df_m['experiment'] = 'M135'
    
    df_all = pd.concat([df_s, df_m])
    print(f"Total division events analyzed: {len(df_all)}")
    
    # Print summary statistics
    print("\n--- Length at Division (t=0) vs 10-minutes before (t=-10) ---")
    summary = df_all.groupby(['experiment', 'channel'])[['pre10_um', 'div_um', 'diff_um']].mean()
    print(summary)
    
    # Save CSV
    out_dir = Path("/Users/user/.gemini/antigravity-ide/brain/32e12c76-45ee-4552-bcd9-4d4c3033438f")
    df_all.to_csv(out_dir / "division_and_pre10_lengths.csv", index=False)
    
    # Plot histograms of pre10_um vs div_um
    plt.figure(figsize=(12, 8))
    
    channels = ['GFP', 'BF']
    colors_div = {'Sept17': 'royalblue', 'M135': 'darkorange'}
    colors_pre10 = {'Sept17': 'lightblue', 'M135': 'wheat'}
    
    for idx, ch in enumerate(channels, 1):
        plt.subplot(1, 2, idx)
        for exp in ['Sept17', 'M135']:
            sub = df_all[(df_all['channel'] == ch) & (df_all['experiment'] == exp)]
            if not sub.empty:
                plt.hist(sub['div_um'], bins=12, alpha=0.7, edgecolor='black',
                         label=f"{exp} t=0 ({sub['div_um'].mean():.2f}±{sub['div_um'].std():.2f} µm)", 
                         color=colors_div[exp])
                plt.hist(sub['pre10_um'], bins=12, alpha=0.4, edgecolor='black', linestyle='--',
                         label=f"{exp} t=-10 ({sub['pre10_um'].mean():.2f}±{sub['pre10_um'].std():.2f} µm)", 
                         color=colors_pre10[exp])
        plt.title(f"{ch} Channel: Length t=0 vs t=-10 min")
        plt.xlabel("Cell Length (µm)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
    plt.suptitle("Fission Yeast Cell Length: Division (t=0) vs 10-Minutes Before (t=-10)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_plot = out_dir / "pre10_vs_div_lengths_histogram.png"
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {out_plot}")

if __name__ == '__main__':
    main()
