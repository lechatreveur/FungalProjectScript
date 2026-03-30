#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:13:54 2025

@author: user
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import to_tree
import matplotlib.cm as cm
from matplotlib.colors import Normalize
def cluster_cells_from_model_params(df_result, n_harmonics=10, method='ward', figsize=(18, 10)):
    """
    Cluster single cells using model-derived features:
    - Assign pol1 to the higher-magnitude signal (based on average value)
    - Use slope + mid, amplitude, and delay from harmonics (up to n_harmonics)
    - Plot a heatmap clustered by cells only (features grouped by type)
    """
    rows = []

    for cell_id in df_result['cell_id'].unique():
        features = {}
        for feature in ['pol1', 'pol2']:
            sub = df_result[(df_result['cell_id'] == cell_id) & (df_result['feature'] == feature)]
            if sub.empty:
                continue

            a_val = sub.get('trend_params.a', pd.Series([0])).values[0]
            b_val = sub.get('trend_params.b', pd.Series([0])).values[0]
            mid_val = a_val * 50 + b_val

            features[feature] = {
                'a': a_val,
                'mid': mid_val,
                'A': {},
                'delay': {}
            }

            f = sub.get('osc_params.f', pd.Series([0])).values[0]
            for j in range(1, n_harmonics + 1):
                A = sub.get(f'osc_params.A{j}', pd.Series([0])).values[0]
                n = sub.get(f'osc_params.n{j}', pd.Series([j])).values[0]
                delay = sub.get(f'osc_params.phi{j}_offset', pd.Series([0])).values[0]

                features[feature]['A'][j] = A
                features[feature]['delay'][j] = delay

        # 🔁 Swap if pol2 has larger mid
        if features.get('pol2', {}).get('mid', 0) > features.get('pol1', {}).get('mid', 0):
            features['pol1'], features['pol2'] = features['pol2'], features['pol1']

        # ✅ Store values in row with consistent pol1/pol2 assignment
        row = {'cell_id': cell_id}
        for f in ['pol1', 'pol2']:
            row[f'{f}_a'] = features[f].get('a', 0)
            row[f'{f}_mid'] = features[f].get('mid', 0)
            for j in range(1, n_harmonics + 1):
                row[f'{f}_A{j}'] = features[f]['A'].get(j, 0)
            for j in range(1,3+1):    
                row[f'{f}_delay{j}'] = features[f]['delay'].get(j, 0)

        rows.append(row)

    # Create DataFrame
    df_features = pd.DataFrame(rows).set_index('cell_id').fillna(0)

    # Order columns for visual grouping
    col_order = []
    for prefix in ['a', 'mid']:
        col_order += [f'pol1_{prefix}', f'pol2_{prefix}']
    for metric in ['A', 'delay']:
        for j in range(1, n_harmonics + 1):
            col_order += [f'pol1_{metric}{j}', f'pol2_{metric}{j}']
    col_order = [c for c in col_order if c in df_features.columns]
    df_features = df_features[col_order]

    # Scale
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features),
                             index=df_features.index,
                             columns=df_features.columns)

    # Row linkage for cells
    row_linkage = linkage(df_scaled.values, method=method)

    # Plot clustermap (no column clustering)
    sns.set(style='white')
    sns.clustermap(
        df_scaled,
        row_linkage=row_linkage,
        col_cluster=False,
        cmap='vlag',
        figsize=figsize,
        xticklabels=True,
        yticklabels=True
    )
    plt.title("Cell Clustering with Polarity-Sorted Signal Assignment", pad=100)
    plt.show()

    return df_features

def plot_amplitude_distributions(clustered_df, n_harmonics=10, figsize=(16, 6)):
    """
    Plot violin plots of amplitude (A) parameters for pol1 and pol2 side-by-side.
    Only uses columns named like 'pol1_A1', 'pol2_A1', ..., 'pol1_A10', 'pol2_A10'.
    """
    # Step 1: Build long-format DataFrame
    records = []

    for j in range(1, n_harmonics + 1):
        for pol in ['pol1', 'pol2']:
            col_name = f'{pol}_A{j}'
            if col_name in clustered_df.columns:
                for cell_id, val in clustered_df[col_name].items():
                    records.append({
                        'cell_id': cell_id,
                        'harmonic': j,
                        'feature': pol,
                        'amplitude': val
                    })

    df_long = pd.DataFrame(records)

    # Step 2: Plot violin
    plt.figure(figsize=figsize)
    sns.violinplot(
        data=df_long,
        x='harmonic',
        y='amplitude',
        hue='feature',
        dodge=True,
        inner='quartile',
        cut=0,
        bw=0.8
    )

    plt.title("Distribution of Harmonic Amplitudes (pol1 vs pol2)")
    plt.xlabel("Harmonic Number")
    plt.ylabel("Amplitude")
    plt.xticks(ticks=np.arange(n_harmonics), labels=[str(j + 1) for j in range(n_harmonics)])
    plt.tight_layout()
    plt.show()

def cluster_cells_by_amplitude_and_delay(df_result, method='ward', figsize=(16, 30), verbose=False):
    """
    Cluster cells based on normalized pol1/pol2 feature pairs + xcor metrics:
    - trend slope (a) and midpoint (mid)
    - A1–A3 (long), A_mid (A4–A6), A_short (A7–A10)
    - delay1–3
    - xcor_max, xcor_lag (flipped if pols swapped), xcor_zero_lag
    """
   

    rows = []

    for cell_id in df_result['cell_id'].unique():
        sub_all = df_result[df_result['cell_id'] == cell_id]
        feats = {}

        for pol in ['pol1', 'pol2']:
            sub = sub_all[sub_all['feature'] == pol]
            if sub.empty:
                continue

            # Trend slope and midpoint
            a = sub['trend_params.a'].values[0] if 'trend_params.a' in sub.columns else 0
            b = sub['trend_params.b'].values[0] if 'trend_params.b' in sub.columns else 0
            a = 0 if pd.isna(a) else a
            b = 0 if pd.isna(b) else b
            mid = a * 50 + b

            # Amplitudes
            # A = {}
            # for j in range(1, 11):
            #     key = f'osc_params.A{j}'
            #     A[j] = sub[key].values[0] if key in sub.columns else 0
            #     A[j] = 0 if pd.isna(A[j]) else A[j]
            
            # # Frequency
            # f = sub['osc_params.f'].values[0] if 'osc_params.f' in sub.columns else 0
            
            # # Delays
            # delay = {}
            # for j in range(1, 4):
            #     key = f'osc_params.phi{j}_offset'
            #     delay[j] = sub[key].values[0] if key in sub.columns else 0
            #     delay[j] = 0 if pd.isna(delay[j]) else delay[j]
            
            # # Identify harmonics 4 to 10 with their amplitudes
            # harmonic_range = range(4, 11)
            # harmonic_amplitudes = {j: A[j] for j in harmonic_range}
            
            # # Find the harmonic number (n_max) with the largest amplitude
            # n_max = max(harmonic_amplitudes, key=harmonic_amplitudes.get)
            
            # # Compute the frequency of that dominant harmonic
            # freq_max_amp = n_max * f

            # feats[pol] = {
            #     'a': a,
            #     'mid': mid,
            #     'A_1': A[1],
            #     'A_2': A[2],
            #     'A_3': A[3],
            #     'A_mid': A[4] + A[5] + A[6],
            #     'A_short': A[7] + A[8] + A[9] + A[10],
            #     'delay1': delay[1],
            #     'delay2': delay[2],
            #     'delay3': delay[3]
            # }
            feats[pol] = {
                'a': a,
                'mid': mid,
        
  
            }

        # Skip incomplete data
        if 'pol1' not in feats or 'pol2' not in feats:
            continue

        mid1, mid2 = feats['pol1']['mid'], feats['pol2']['mid']
        
        if np.isnan(mid1) and np.isnan(mid2):
            continue
        elif np.isnan(mid1):
            primary, secondary = 'pol2', 'pol1'
        elif np.isnan(mid2):
            primary, secondary = 'pol1', 'pol2'
        else:
            primary, secondary = ('pol1', 'pol2') if mid1 >= mid2 else ('pol2', 'pol1')

        if verbose:
            print(f"{cell_id}: primary={primary} mid={feats[primary]['mid']:.2f}, "
                  f"secondary={secondary} mid={feats[secondary]['mid']:.2f}")

        # Extract acor and xcor values directly from df columns
        try:
            freq_distance_sum = sub_all['freq_distance_sum'].values[0]
            precision_sum = sub_all['precision_sum'].values[0]
            NC_score = sub_all['NC_score'].values[0]
        except (KeyError, IndexError):
            freq_distance_sum = 0
            precision_sum = 0
            NC_score = 0
            


        #if primary == 'pol2':  # flip if pol2 is reassigned as pol1
        #    xcor_lag *= -1
        
        # Extract acor values 
        # try:
        #     acor1_max = sub_all['acor1_max'].values[0]
        #     acor2_max = sub_all['acor2_max'].values[0]
        #     acor1_lag = sub_all['acor1_lag'].values[0]
        #     acor2_lag = sub_all['acor2_lag'].values[0]
        #     acor1_zero_lag = sub_all['acor1_zero_lag'].values[0]
        #     acor2_zero_lag = sub_all['acor2_zero_lag'].values[0]
        # except (KeyError, IndexError):
        #     acor1_max = 0 
        #     acor2_max = 0 
        #     acor1_lag = 0
        #     acor2_lag = 0
        #     acor1_zero_lag = 0
        #     acor2_zero_lag = 0
        
        # Compose final row
        row = {'cell_id': cell_id}
        for prefix, source in zip(['pol1', 'pol2'], [primary, secondary]):
            for k, v in feats[source].items():
                row[f'{prefix}_{k}'] = v

        # Add acor and xcor features
        #row['precision_sum'] = precision_sum 
        #row['freq_distance_sum'] = freq_distance_sum
        row['Periodicity'] = precision_sum - freq_distance_sum
        row['NC_score'] = NC_score
        
        # # Add acor features
        # row['acor1_max'] = acor1_max
        # row['acor2_max'] = acor2_max
        # row['acor1_lag'] = acor1_lag
        # row['acor2_lag'] = acor2_lag
        # row['acor1_zero_lag'] = acor1_zero_lag
        # row['acor2_zero_lag'] = acor2_zero_lag
        
        
        # Add designed features
        a1, a2 = row['pol1_a'], row['pol2_a']
        row['a1a2'] = a1 * a2
        mid1, mid2 = row['pol1_mid'], row['pol2_mid']
        row['d'] = np.abs(mid1-mid2)
        row['dd'] = a1 - a2

        rows.append(row)
        
    # Build DataFrame
    df_raw = pd.DataFrame(rows).set_index('cell_id').fillna(0)

    # Metrics to include in normalization
    # metrics = ['a', 'mid', 'A_1', 'A_2', 'A_3', 'A_mid', 'A_short',
    #            'delay1', 'delay2', 'delay3']
    metrics = ['a', 'mid', #'f', 'A'
               ]
    xcor_metrics = [#'precision_sum',
                    #'freq_distance_sum',
                    'Periodicity',
                    'NC_score'
                    ]
    # acor_metrics = [
    # 'acor1_max', 'acor2_max',
    # 'acor1_lag', 'acor2_lag',
    # 'acor1_zero_lag', 'acor2_zero_lag'
    # ]
    poles_metrics = ['a1a2','d','dd']
    weights = {
        'a': 3,
        'mid': 3,
        #'f': 1,
        #'A': 1,
        #'A_1': 3,
        #'A_2': 3,
        #'A_3': 3,
        #'A_mid': 3,
        #'A_short': 3,
        #'delay1': 2,
        #'delay2': 1,
        #'delay3': 1,
        #'xcor_max': 1,
        #'xcor_lag': 1,
        #'xcor_zero_lag': 1,
        #'acor1_max': 1,
        #'acor2_max': 1,
        #'acor1_lag': 1,
        #'acor2_lag': 1,
        #'acor1_zero_lag': 1,
        #'acor2_zero_lag': 1,
        'a1a2':3,
        'd':3,
        'dd':3,
        #'precision_sum':1,
        #'freq_distance_sum':1,
        'Periodicity':1,
        'NC_score':1
        
    }

    df_norm = df_raw.copy()

    # Normalize pol1/pol2 features
    # for m in metrics:
    #     pair_cols = [f'pol1_{m}', f'pol2_{m}']
    #     scaler = StandardScaler()
    #     df_norm[pair_cols] = scaler.fit_transform(df_raw[pair_cols])
    #     df_norm[pair_cols] *= weights[m]

    # # Normalize xcor metrics
    # for m in xcor_metrics:
    #     scaler = StandardScaler()
    #     df_norm[m] = scaler.fit_transform(df_raw[[m]])
    #     df_norm[m] *= weights[m]
    # # Normalize poles metrics
    # for m in poles_metrics:
    #     scaler = StandardScaler()
    #     df_norm[m] = scaler.fit_transform(df_raw[[m]])
    #     df_norm[m] *= weights[m]
        
        # Scale only (no centering): x / std(x)
   # Normalize features by std and weight
    for m in metrics:
        pair_cols = [f'pol1_{m}', f'pol2_{m}']
        for col in pair_cols:
            std = df_raw[col].std()
            df_norm[col] = df_raw[col] / std * weights[m]
    
    for m in xcor_metrics:
        std = df_raw[m].std()
        #ave = df_raw[m].mean()
        df_norm[m] = df_raw[m] / std * weights[m]
    
    for m in poles_metrics:
        std = df_raw[m].std()
        df_norm[m] = df_raw[m] / std * weights[m]
    
    # for m in acor_metrics:
    #     std = df_raw[m].std()
    #     df_norm[m] = df_raw[m] / std * weights[m]
    
    # Reorder columns
    col_order = []
    for m in metrics:
        col_order += [f'pol1_{m}', f'pol2_{m}']
    col_order += poles_metrics
    col_order += xcor_metrics
    #col_order += acor_metrics
    df_norm = df_norm[col_order]
    
    # Hierarchical clustering
    row_linkage = linkage(df_norm.values, method=method)
    
    # Heatmap
    sns.set_theme(font_scale=0.5)   # try 1.1–1.5
    clustergrid = sns.clustermap(
        df_norm,
        row_linkage=row_linkage,
        col_cluster=False,
        cmap='vlag',
        figsize=figsize,
        vmin=-10,
        vmax=10,
        xticklabels=True,
        yticklabels=True
    )
    plt.title("Cell Clustering by Amplitude, Delay, X-Correlation, and Autocorrelation", pad=100)
    plt.show()
    
    # Return ordered cell IDs
    ordered_cell_ids = df_norm.index[clustergrid.dendrogram_row.reordered_ind].tolist()

    return df_norm, ordered_cell_ids, row_linkage

def plot_aligned_time_by_dendrogram_order(row_linkage, df_norm, aligned_time_series):
    # --- Ensure consistent string index for all inputs ---
    df_norm = df_norm.copy()
    df_norm.index = df_norm.index.astype(str)

    # If aligned_time_series is a Series, convert its index to string
    if hasattr(aligned_time_series, 'index'):
        aligned_time_series = aligned_time_series.copy()
        aligned_time_series.index = aligned_time_series.index.astype(str)
    else:
        # If it's a dict, convert keys to string
        aligned_time_series = {str(k): v for k, v in aligned_time_series.items()}

    # Filter aligned_time_series to match df_norm
    aligned_times = pd.Series(aligned_time_series)
    aligned_times = aligned_times[df_norm.index]

    # Get dendrogram leaf order (no plot)
    dendro = dendrogram(row_linkage, no_plot=True, labels=df_norm.index.tolist())
    ordered_cell_ids = dendro['ivl']

    # Prepare x (aligned time) and y (row position)
    aligned_x = [aligned_times[cell_id] for cell_id in ordered_cell_ids]
    y_pos = np.arange(len(ordered_cell_ids))[::-1]  # reversed order

    # Plot
    plt.figure(figsize=(8, len(ordered_cell_ids) * 0.3))
    plt.scatter(aligned_x, y_pos, color='blue')
    plt.yticks(y_pos, ordered_cell_ids, fontsize=20)  # You can set 14, 16, etc.
    plt.ylim(-0.5, len(ordered_cell_ids) - 0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("Aligned Time")
    plt.title("Aligned Time per Cell (Ordered by Dendrogram)")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
def plot_tree_with_annotations(tree_root, y_offset=0, x_offset=0, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 60))

    # Flatten the tree to get all values for color normalization
    all_means = []

    def collect_means(node):
        if hasattr(node, 'aligned_times') and node.aligned_times:
            mu = np.mean(node.aligned_times)
            all_means.append(mu)
        if not node.is_leaf():
            collect_means(node.left)
            collect_means(node.right)

    collect_means(tree_root)

    # Normalize for colormap
    norm = Normalize(vmin=min(all_means), vmax=max(all_means))
    cmap = cm.get_cmap('viridis')  # You can change to 'plasma', 'coolwarm', etc.

    def draw_node(node, x):
        nonlocal y_offset

        if node.is_leaf():
            y = y_offset
            y_offset += 1
            value = np.mean(node.aligned_times) if hasattr(node, 'aligned_times') else 0
            color = cmap(norm(value))
            ax.text(x, y, node.label, va='center', ha='left', fontsize=20, color=color)
        else:
            y_left = draw_node(node.left, x + node.dist)
            y_right = draw_node(node.right, x + node.dist)
            y = (y_left + y_right) / 2
            ax.plot([x + node.dist, x + node.dist], [y_left, y_right], 'k-')
            ax.plot([x, x + node.dist], [y, y], 'k-')
            value = np.mean(node.aligned_times) if hasattr(node, 'aligned_times') else 0
            color = cmap(norm(value))
            ax.text(x, y, node.label, va='center', ha='right', fontsize=20, color=color)
        return y

    draw_node(tree_root, x_offset)
    ax.axis('off')
    plt.title("Clustering Tree Annotated with Aligned Time Stats (Colored by Mean)", fontsize=18)
    plt.tight_layout()
    plt.show()
    





def _annotate_node(node, df_norm, aligned_time_series):
    if node.is_leaf():
        cell_id = df_norm.index[node.id]
        value = aligned_time_series[cell_id]
        node.aligned_times = [value]
        node.label = f"{cell_id} ({value:.2f})"
    else:
        _annotate_node(node.left, df_norm, aligned_time_series)
        _annotate_node(node.right, df_norm, aligned_time_series)
        node.aligned_times = node.left.aligned_times + node.right.aligned_times
        mu = np.mean(node.aligned_times)
        sigma = np.std(node.aligned_times)
        node.label = f"[{node.id}]"#" {mu:.2f} ± {1.96 * sigma:.2f}"  # <-- show internal node ID

        
def annotate_tree_with_aligned_time(df_norm, row_linkage, aligned_time_series):
    tree_root, node_list = to_tree(row_linkage, rd=True)
    _annotate_node(tree_root, df_norm, aligned_time_series)
    return tree_root, node_list
