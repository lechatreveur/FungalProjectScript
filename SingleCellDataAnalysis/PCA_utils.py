import os
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_cell_data(csv_path, target_id=None):
    """
    Loads a single cell's data and computes corrected intensities.
    Filters by target_id to exclude daughter cell parts (_1, _2).
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 1. Filter by target_id if provided
        if target_id is not None:
            # Ensure we are comparing strings/numbers correctly
            # We want rows where the first column is exactly target_id (no suffix)
            col0 = df.columns[0]
            df[col0] = df[col0].astype(str)
            df = df[df[col0] == str(target_id)]
            
        # 2. Required columns: pol1_int, pol2_int, cyt_int
        if not {'pol1_int', 'pol2_int', 'cyt_int'}.issubset(df.columns):
            # Try to see if it's the BF version which has different columns
            # For 09_17, we usually want the GFP intensities
            return None
        
        # 3. Correct intensities
        # (Assuming the main part is already chronologically ordered in the filtered DF)
        df['pol1_int_corr'] = df['pol1_int'] - df['cyt_int']
        df['pol2_int_corr'] = df['pol2_int'] - df['cyt_int']
        
        # Re-index to ensure time goes from 0..N
        df = df.reset_index(drop=True)
        
        return df[['pol1_int_corr', 'pol2_int_corr']]
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def get_trajectories_from_dir(exp_dir, target_len=100):
    """
    Finds all cell_X_data.csv files in exp_dir and extracts 
    corrected trajectories of fixed length.
    """
    search_pattern = os.path.join(exp_dir, "**", "cell_*_data.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    all_trajectories = []
    cell_metadata = []
    
    for f in csv_files:
        df = load_cell_data(f)
        if df is not None and len(df) >= 10: # Minimum frames filter
            # Extract trajectories
            p1 = df['pol1_int_corr'].values
            p2 = df['pol2_int_corr'].values
            
            # Helper to pad or truncate to target_len
            def adjust_len(arr, length):
                if len(arr) >= length:
                    return arr[:length]
                else:
                    return np.pad(arr, (0, length - len(arr)), mode='constant', constant_values=np.nan)
            
            p1_fix = adjust_len(p1, target_len)
            p2_fix = adjust_len(p2, target_len)
            
            # Combine poles: [p1_0...p1_99, p2_0...p2_99]
            combined = np.concatenate([p1_fix, p2_fix])
            
            # Check for NaNs (if padding was used or original data had NaNs)
            # We will handle NaNs later in the matrix level if needed, 
            # but for PCA common practice is to drop rows or fill.
            
            all_trajectories.append(combined)
            cell_metadata.append({
                'cell_id': os.path.basename(f).split('_')[1],
                'source_file': f,
                'original_len': len(df)
            })
            
    return np.array(all_trajectories), pd.DataFrame(cell_metadata)

def run_pca_workflow(X, n_components=10):
    """Standardizes data, handles NaNs, and runs PCA."""
    # Handle NaNs: For now, fill with 0 or mean? 
    # Let's fill NaNs with the mean of the column (temporal mean across cells)
    X_filled = np.nan_to_num(X, nan=0.0) # Simple fallback
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca, scaler

def plot_top_cells_for_pc(metadata, X, pc_idx=1, n_cells=3, target_len=100):
    """
    Plots the trajectories of cells centered at the extremes of a PC axis.
    """
    import matplotlib.pyplot as plt
    
    pc_col = f'PC{pc_idx}'
    if pc_col not in metadata.columns:
        print(f"Error: {pc_col} not found in metadata.")
        return
        
    sorted_indices = metadata.sort_values(pc_col).index
    
    lowest_idx = sorted_indices[:n_cells]
    highest_idx = sorted_indices[-n_cells:]
    
    fig, axes = plt.subplots(2, n_cells, figsize=(15, 8), sharex=True, sharey=True)
    
    # Plot Lowest
    for i, idx in enumerate(lowest_idx):
        traj = X[idx]
        p1 = traj[:target_len]
        p2 = traj[target_len:]
        
        ax = axes[0, i]
        ax.plot(p1, color='red', label='Pole 1')
        ax.plot(p2, color='blue', label='Pole 2')
        ax.set_title(f"Low {pc_col}\nCell {metadata.iloc[idx]['cell_id']}")
        if i == 0: ax.set_ylabel("Intensity")
        
    # Plot Highest
    for i, idx in enumerate(highest_idx):
        traj = X[idx]
        p1 = traj[:target_len]
        p2 = traj[target_len:]
        
        ax = axes[1, i]
        ax.plot(p1, color='red')
        ax.plot(p2, color='blue')
        ax.set_title(f"High {pc_col}\nCell {metadata.iloc[idx]['cell_id']}")
        ax.set_xlabel("Time (frames)")
        if i == 0: ax.set_ylabel("Intensity")
        
    plt.suptitle(f"Trajectories at Extremes of {pc_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def load_experiment_features(exp_dir):
    """
    Loads and combines model fits and autocorrelation features for any given experiment.
    Implements polarity alignment (pol1 = stronger signal).

    Polarity alignment rule (from clustering.py):
        mid = a * 50 + b  (signal value at the midpoint frame)
        if pol2_mid > pol1_mid -> swap labels so pol1 is always the dominant pole.

    Caveat: for cells whose pol1 was best-fit as 'constant' or bare 'linear',
    trend_params.a and trend_params.b are NaN in the CSV (those params are only
    stored for linear+harmonic / linear+sine models). In those cases we fall back
    to the raw trajectory mean from the stacked GFP data as the mid proxy.
    """
    out_dir = os.path.join(exp_dir, "unaligned_pairs_quant")
    
    fits_path = os.path.join(out_dir, "model_fits_by_cell.csv")
    if not os.path.exists(fits_path):
        fits_path = os.path.join(exp_dir, "model_fits_by_cell.csv")
        
    acor_path = os.path.join(out_dir, "acor_detrended_results.csv")
    if not os.path.exists(acor_path):
        acor_path = os.path.join(exp_dir, "acor_detrended_results.csv")
        
    stacked_path = os.path.join(out_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")

    if not os.path.exists(fits_path) or not os.path.exists(acor_path):
        raise FileNotFoundError(f"Missing required feature files in {exp_dir} or {out_dir}")

    # 1. Load Model Fits
    df_fits = pd.read_csv(fits_path)

    # 2. Pre-compute raw trajectory means and linear variances
    if os.path.exists(stacked_path):
        df_stacked = pd.read_csv(stacked_path)
        
        raw_data = {}
        for c_id, grp in df_stacked.groupby("cell_id"):
            t = pd.to_numeric(grp['time_point'], errors='coerce').values
            y1 = pd.to_numeric(grp['pol1_int_corr'], errors='coerce').values
            y2 = pd.to_numeric(grp['pol2_int_corr'], errors='coerce').values
            
            # Helper to calculate residual variance of linear fit
            def lin_var(t_arr, y_arr):
                valid = ~np.isnan(t_arr) & ~np.isnan(y_arr)
                t_v, y_v = t_arr[valid], y_arr[valid]
                if len(t_v) < 2: return 0.0
                a, b = np.polyfit(t_v, y_v, 1)
                return np.var(y_v - (a * t_v + b))
            
            raw_data[c_id] = {
                'pol1_raw_mean': np.nanmean(y1),
                'pol2_raw_mean': np.nanmean(y2),
                'pol1_v': lin_var(t, y1),
                'pol2_v': lin_var(t, y2)
            }
        raw_stats = pd.DataFrame.from_dict(raw_data, orient='index')
    else:
        print(f"[warn] Stacked CSV not found at {stacked_path}; "
              "polarity alignment may be incorrect and variance will be 0.")
        raw_stats = pd.DataFrame()

    # 3. Extract trend parameters and pivot, with polarity alignment
    rows = []
    for cell_id in df_fits['cell_id'].unique():
        sub = df_fits[df_fits['cell_id'] == cell_id]

        feats = {}
        for pol in ['pol1', 'pol2']:
            p_sub = sub[sub['feature'] == pol]
            if p_sub.empty:
                continue

            a = p_sub['trend_params.a'].values[0] if 'trend_params.a' in p_sub.columns else np.nan
            b = p_sub['trend_params.b'].values[0] if 'trend_params.b' in p_sub.columns else np.nan

            if pd.notna(a) and pd.notna(b):
                mid = a * 50 + b
            elif cell_id in raw_stats.index:
                # Fallback: raw trajectory mean for cells with constant/linear model
                mid = raw_stats.loc[cell_id, f"{pol}_raw_mean"]
                # Reconstruct a plausible 'a' as 0 (flat) for the feature matrix
                a = 0.0
            else:
                mid = 0.0
                a   = 0.0
                
            # Extract the raw variance for this pole
            v = raw_stats.loc[cell_id, f"{pol}_v"] if cell_id in raw_stats.index else 0.0

            feats[pol] = {'a': a, 'mid': mid, 'v': v}

        if 'pol1' not in feats or 'pol2' not in feats:
            continue

        # Polarity Alignment: ensure pol1 is always the stronger (higher mid) pole
        if feats['pol2']['mid'] > feats['pol1']['mid']:
            primary, secondary = 'pol2', 'pol1'
        else:
            primary, secondary = 'pol1', 'pol2'

        row = {
            'cell_id':  cell_id,
            'pol1_a':   feats[primary]['a'],
            'pol1_mid': feats[primary]['mid'],
            'pol1_v':   feats[primary]['v'],
            'pol2_a':   feats[secondary]['a'],
            'pol2_mid': feats[secondary]['mid'],
            'pol2_v':   feats[secondary]['v']
        }
        rows.append(row)

    df_pivoted = pd.DataFrame(rows).set_index('cell_id')

    # 4. Load Acor Results
    df_acor = pd.read_csv(acor_path).set_index('cell_id')
    df_acor['Periodicity'] = df_acor['precision_sum'] - df_acor['freq_distance_sum']

    # 5. Merge
    df_combined = df_pivoted.join(df_acor[['NC_score', 'Periodicity']], how='inner')

    # 6. Add Interaction Features
    df_combined['a1a2'] = df_combined['pol1_a'] * df_combined['pol2_a']
    df_combined['d']    = (df_combined['pol1_mid'] - df_combined['pol2_mid']).abs()
    df_combined['dd']   = df_combined['pol1_a'] - df_combined['pol2_a']

    return df_combined
