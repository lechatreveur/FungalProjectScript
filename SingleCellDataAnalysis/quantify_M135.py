import os
import sys
import json
import re
import pandas as pd
import numpy as np

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.signal_cor import quantify_all_cells_acor
from SingleCellDataAnalysis.signal_analysis import quantify_all_cells

WORKING_DIR = "/Volumes/X10 Pro/Movies/2026_04_30_M135/"
FOVS = ["F0", "F1", "F2"]


def process_m135():
    print(f"\n{'='*60}")
    print(f"Processing M135 from {WORKING_DIR}")
    print(f"{'='*60}")
    
    linkage_path = os.path.join(WORKING_DIR, "sequence_linkage.json")
    if not os.path.exists(linkage_path):
        print(f"Error: {linkage_path} not found.")
        return
        
    with open(linkage_path, "r") as f:
        linkage = json.load(f)
        
    # Load QC data
    qc_data = {}
    qc_path = os.path.join(WORKING_DIR, "A14_F0", "qc_A14_F0.json")
    if os.path.exists(qc_path):
        with open(qc_path) as f:
            qc_data.update(json.load(f))
            
    stack_rows = []
    map_rows = []
    new_id = 1
    
    for fov in FOVS:
        fov_key = f"A14_{fov}"
        if fov_key not in linkage:
            continue
            
        film_names = linkage[fov_key]["films"]
        global_cells = linkage[fov_key]["global_cells"]
        
        for gcid, indices in global_cells.items():
            # Only keep main global cells (e.g. A14_F0_cell_12)
            if not re.match(r'^A14_F[0-2]_cell_\d+$', gcid):
                continue
            status = qc_data.get(gcid)
            if status not in ['good', 'corrected']:
                continue
                
            fl1_idx, bf1_idx, fl2_idx, bf2_idx, fl3_idx = indices
            
            # Enforce that all three GFP parts (FL1, FL2, FL3) are present
            if fl1_idx != -1 and fl2_idx != -1 and fl3_idx != -1:
                # Pre-verify and load all three files
                all_valid = True
                loaded_dfs = []
                for i, fl_idx in enumerate([fl1_idx, fl2_idx, fl3_idx]):
                    film_name = film_names[i * 2]
                    csv_path = os.path.join(WORKING_DIR, film_name, f"TrackedCells_{film_name}", f"cell_{fl_idx}_data.csv")
                    if not os.path.exists(csv_path):
                        all_valid = False
                        break
                    try:
                        df_tmp = pd.read_csv(csv_path)
                        if len(df_tmp) == 0:
                            all_valid = False
                            break
                        loaded_dfs.append(df_tmp)
                    except Exception:
                        all_valid = False
                        break
                if not all_valid:
                    continue
                
                # Now process the pre-loaded FL dataframes
                # We load FL1, FL2, and FL3 to represent GFP1, GFP2, GFP3
                for i, df in enumerate(loaded_dfs):
                    fl_idx = [fl1_idx, fl2_idx, fl3_idx][i]
                    film_name = film_names[i * 2]
                    source_name = f"GFP{i+1}"
                        
                    # Keep only main cell row per time_point (where split_n0 is NaN)
                    if 'split_n0' in df.columns:
                        df = df[df['split_n0'].isna()].copy()
                    
                    # Compute pol1_int_corr
                    df["pol1_int"] = pd.to_numeric(df["pol1_int"], errors="coerce").fillna(0)
                    df["pol2_int"] = pd.to_numeric(df["pol2_int"], errors="coerce").fillna(0)
                    df["cyt_int"] = pd.to_numeric(df["cyt_int"], errors="coerce").fillna(0)
                    df["pol1_int_corr"] = df["pol1_int"] - df["cyt_int"]
                    df["pol2_int_corr"] = df["pol2_int"] - df["cyt_int"]
                    
                    # Reindex to ensure exactly 101 frames (0 to 100)
                    df = df.set_index('time_point')
                    df = df.reindex(range(101))
                    df['pol1_int_corr'] = df['pol1_int_corr'].ffill().bfill().fillna(0.0)
                    df['pol2_int_corr'] = df['pol2_int_corr'].ffill().bfill().fillna(0.0)
                    df = df.reset_index()
                    
                    # Assign unique cell ID for each source
                    df["cell_id"] = new_id
                    df["source"] = source_name
                    df["field"] = fov
                    
                    stack_rows.append(df[["time_point", "cell_id", "pol1_int_corr", "pol2_int_corr", "source", "field"]])
                    
                    map_rows.append({
                        "new_cell_id": new_id,
                        "global_cell_id": gcid,
                        "local_fl_id": fl_idx,
                        "source": source_name,
                        "field": fov
                    })
                    
                    new_id += 1

    if not stack_rows:
        print("No valid traces found for M135!")
        return

    df_stacked = pd.concat(stack_rows, ignore_index=True)
    df_map = pd.DataFrame(map_rows)
    
    out_dir = os.path.join(WORKING_DIR, "unaligned_pairs_quant")
    os.makedirs(out_dir, exist_ok=True)
    
    data_csv = os.path.join(out_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
    map_csv = os.path.join(out_dir, "id_map_unaligned.csv")
    
    df_stacked.to_csv(data_csv, index=False)
    df_map.to_csv(map_csv, index=False)
    
    print(f"Saved stacked data to: {data_csv}")
    print(f"Total global traces extracted: {df_map['new_cell_id'].nunique()}")
    
    cell_ids = df_map["new_cell_id"].unique().tolist()
    
    print("\nRunning standard quantification (model fits)...")
    fits_csv = os.path.join(out_dir, "model_fits_by_cell.csv")
    _ = quantify_all_cells(
        df_stacked, cell_ids, 
        feature1='pol1_int_corr', feature2='pol2_int_corr', 
        delta_threshold=10, filename=fits_csv
    )
    
    print("\nRunning autocorrelation quantification...")
    acor_csv = os.path.join(out_dir, "acor_detrended_results.csv")
    _ = quantify_all_cells_acor(
        df_stacked, cell_ids, 
        delta_threshold=10, visualize=False, filename=acor_csv
    )

if __name__ == "__main__":
    process_m135()
