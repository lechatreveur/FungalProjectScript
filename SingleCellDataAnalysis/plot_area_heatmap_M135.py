import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

csv_path = Path("/Users/user/.gemini/antigravity-ide/brain/32e12c76-45ee-4552-bcd9-4d4c3033438f/global_area_vs_stage_data_M135.csv")
out_img = Path("/Users/user/.gemini/antigravity-ide/brain/32e12c76-45ee-4552-bcd9-4d4c3033438f/m135_area_heatmap.png")

def main():
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    # Filter only cells with septum
    df_septum = df[df["group"] == "with_septum"].copy()
    
    if df_septum.empty:
        print("No cells with septum found in data.")
        return
        
    print(f"Loaded {df_septum['global_cell_id'].nunique()} unique cells with septum.")
    
    # Bin times relative to division (stage_min)
    # Bin size of 2 minutes
    bin_size = 2
    df_septum["time_bin"] = (df_septum["stage_min"] / bin_size).round() * bin_size
    
    # Pivot to create a grid of cells vs time bins
    grid = df_septum.groupby(["global_cell_id", "time_bin"])["area"].mean().unstack()
    
    # Sort cells by their start time (minimum time bin observed)
    start_times = grid.apply(lambda row: row.first_valid_index(), axis=1)
    grid = grid.loc[start_times.sort_values().index]
    
    # Include all valid time bins to avoid cutoffs
    valid_cols = sorted(grid.columns)
    grid = grid[valid_cols]
    
    # Set up plot style
    plt.figure(figsize=(14, 9))
    
    # Heatmap with dark background for NaNs
    ax = sns.heatmap(
        grid, 
        cmap="magma", 
        cbar_kws={"label": "Cell Area (px²)"},
        xticklabels=10, # Label every 10th bin (20 mins)
        yticklabels=True
    )
    
    # Set background color for NaNs to a very dark grey to match magma theme
    ax.set_facecolor("#111111")
    
    plt.title("M135 Cell Area Over Time Relative to Division (Cells with Septum)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time Relative to Division (minutes)", fontsize=12)
    plt.ylabel("Cell ID (Sorted by earliest observation time)", fontsize=12)
    
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_img}")

if __name__ == "__main__":
    main()
