import os
import json
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript')
from Cell_tracking_functions import rle_decode

MOVIE_ROOT = "/Volumes/X10 Pro/Movies/2025_09_17"
ARTIFACTS_DIR = "/Users/user/.gemini/antigravity-ide/brain/32e12c76-45ee-4552-bcd9-4d4c3033438f"

def load_septum_data(film_name):
    json_path = os.path.join(MOVIE_ROOT, film_name, f"TrackedCells_{film_name}", "cell_plots", "gui_labels", "global_septum_alignment.json")
    if not os.path.exists(json_path):
        return {}, {}
    with open(json_path) as f:
        js = json.load(f)
    return js.get("offsets", {}), js.get("cell_intervals", {})

def main():
    films = ["A14_1TP1_F1", "A14_1TP1_BF_F1", "A14_1TP2_F1", "A14_1TP2_BF_F1"]
    
    data_points = []
    
    for film in films:
        offsets, intervals = load_septum_data(film)
        if not intervals:
            continue
            
        is_bf = "BF" in film
        time_res_min = 1.0 if is_bf else 0.2
        
        print(f"Processing film {film} (is_bf={is_bf}) with {len(intervals)} intervals...")
        
        for cell_id_str, interval in intervals.items():
            if not interval.get("has_septum") or interval.get("end_aligned") is None:
                continue
                
            offset = int(offsets.get(cell_id_str, 0))
            t_div = int(interval["end_aligned"]) - offset
            
            # Load masks CSV
            masks_csv_path = os.path.join(MOVIE_ROOT, film, f"TrackedCells_{film}", f"cell_{cell_id_str}_masks.csv")
            if os.path.exists(masks_csv_path):
                df_masks = pd.read_csv(masks_csv_path)
                
                # Filter to only the exact cell_id row (exclude suffixes if any)
                col0 = df_masks.columns[0]
                # In masks.csv, first column is time_point. We want to iterate over time points.
                df_filtered = df_masks[df_masks['time_point'] <= t_div]
                
                for _, row in df_filtered.iterrows():
                    t = int(row['time_point'])
                    rle_col = 'rle_bf' if is_bf else 'rle_gfp'
                    
                    if rle_col in row and isinstance(row[rle_col], str) and row[rle_col].strip():
                        rle_val = row[rle_col]
                        h, w = int(row['height']), int(row['width'])
                        mask = rle_decode(rle_val, (h, w))
                        area = int(mask.sum())
                        
                        if area > 0:
                            data_points.append({
                                'cell_id': f"{film}_{cell_id_str}",
                                'stage_min': (t - t_div) * time_res_min,
                                'area': area,
                                'channel': 'BF' if is_bf else 'GFP'
                            })
                            
    df_plot = pd.DataFrame(data_points)
    print(f"Total data points collected: {len(df_plot)}")
    
    if df_plot.empty:
        print("No data points found!")
        return
        
    # Print channel counts
    print(df_plot['channel'].value_counts())
    
    plt.figure(figsize=(10, 6))
    
    colors = {'BF': 'darkred', 'GFP': 'darkgreen'}
    
    for ch, sub in df_plot.groupby('channel'):
        x = sub['stage_min'].values
        y = sub['area'].values
        plt.scatter(x, y, alpha=0.3, c=colors[ch], edgecolors='none', s=10, label=f'{ch} points')
        
        # Calculate fit line
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        x_fit = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_fit, slope*x_fit + intercept, color=colors[ch], linewidth=2.5,
                 label=f'{ch} Fit: Area = {slope:.2f}*t + {intercept:.2f} (r = {r:.3f})')
        
    plt.title("Fission Yeast Cell Area vs Time Relative to Division", fontsize=12, fontweight='bold')
    plt.xlabel("Time relative to division (minutes)")
    plt.ylabel("Cell Area (px²)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    out_img = os.path.join(ARTIFACTS_DIR, "cell_area_vs_stage.png")
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_img}")

if __name__ == '__main__':
    main()
