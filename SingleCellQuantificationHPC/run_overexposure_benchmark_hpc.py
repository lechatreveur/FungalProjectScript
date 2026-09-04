import os
import sys
import gc
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from cellpose import models

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
torch.set_num_threads(8)

# Determine path based on environment (HPC vs Local)
HPC_BASE = '/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156'
LOCAL_BASE = '/Volumes/X10 Pro/Movies/2026_07_16_M156'

if os.path.exists(HPC_BASE):
    MOVIE_ROOT = HPC_BASE
else:
    MOVIE_ROOT = LOCAL_BASE

BASE_DIR = os.path.join(MOVIE_ROOT, '3_FL2_F0')
TRACKED_DIR = os.path.join(BASE_DIR, 'TrackedCells_3_FL2_F0')
FRAMES_DIR = os.path.join(BASE_DIR, 'Frames_3_FL2_F0')
OUTPUT_DIR = os.path.join(MOVIE_ROOT, 'cellpose_overexposure_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CELLS = [25, 116, 9, 18, 15, 11, 19, 140, 178, 34]

def decode_rle(rle_str, width=2000, height=2000):
    if pd.isna(rle_str) or str(rle_str).strip() == '':
        return np.zeros((height, width), dtype=bool)
    nums = list(map(int, str(rle_str).split()))
    flat = np.zeros(width*height, dtype=bool)
    for i in range(0, len(nums), 2):
        start, length = nums[i], nums[i+1]
        flat[start:start+length] = True
    return flat.reshape(height, width)

def preprocess_raw_u8(img_raw):
    p1, p99 = np.percentile(img_raw, (0.1, 99.9))
    if p99 > p1:
        norm = np.clip((img_raw - p1) / (p99 - p1), 0, 1)
    else:
        norm = np.zeros_like(img_raw, dtype=float)
    return (norm * 255).astype(np.uint8)

def preprocess_overexposed_u8(img_raw):
    smooth = gaussian_filter(img_raw.astype(float), sigma=1.0)
    bg = gaussian_filter(img_raw.astype(float), sigma=20.0)
    dog = np.maximum(0, smooth - bg)
    p_low, p_high = np.percentile(dog, (5, 95))
    if p_high > p_low:
        norm_dog = np.clip((dog - p_low) / (p_high - p_low), 0, 1)
    else:
        norm_dog = np.zeros_like(dog)
    overexp = np.power(norm_dog, 0.4)
    return (overexp * 255).astype(np.uint8)

def calculate_jumps(areas, centroids):
    jumps = [False] * len(areas)
    for t in range(len(areas) - 1):
        a1, a2 = areas[t], areas[t+1]
        c1, c2 = centroids[t], centroids[t+1]
        if a1 == 0 or a2 == 0 or np.isnan(c1[0]) or np.isnan(c2[0]):
            jumps[t] = True
            continue
        area_ratio = max(a1, a2) / min(a1, a2)
        dist = np.hypot(c2[0] - c1[0], c2[1] - c1[1])
        if area_ratio > 1.5 or dist > 15.0:
            jumps[t] = True
    return jumps

def run_cell_benchmark(cell_id, model):
    print(f"\n================ Processing Cell {cell_id} ================", flush=True)
    csv_path = os.path.join(TRACKED_DIR, f'cell_{cell_id}_masks.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Mask file not found for cell {cell_id}", flush=True)
        return None

    masks_df = pd.read_csv(csv_path)
    
    existing_masks_crop = []
    existing_areas = []
    existing_centroids = []
    existing_mean_ints = []
    
    all_centroids = []
    for idx, row in masks_df.iterrows():
        mask = decode_rle(row['rle_gfp'])
        if mask.sum() > 0:
            ys, xs = np.where(mask)
            all_centroids.append((np.mean(ys), np.mean(xs)))
    
    if len(all_centroids) == 0:
        print(f"Error: No valid masks found for cell {cell_id}", flush=True)
        return None
        
    all_centroids = np.array(all_centroids)
    med_y, med_x = np.median(all_centroids[:, 0]), np.median(all_centroids[:, 1])
    
    cy = int(np.clip(round(med_y), 150, 2000 - 150))
    cx = int(np.clip(round(med_x), 150, 2000 - 150))
    print(f"Cell {cell_id} crop center: (y={cy}, x={cx}), bounds: [{cy-150}:{cy+150}, {cx-150}:{cx+150}]", flush=True)
    
    for t in range(101):
        row = masks_df.iloc[t]
        mask_full = decode_rle(row['rle_gfp'])
        mask_crop = mask_full[cy-150:cy+150, cx-150:cx+150]
        existing_masks_crop.append(mask_crop)
        area = mask_crop.sum()
        existing_areas.append(area)
        if area > 0:
            ys, xs = np.where(mask_crop)
            existing_centroids.append((np.mean(ys) + cy - 150, np.mean(xs) + cx - 150))
        else:
            existing_centroids.append((np.nan, np.nan))
            
    raw_crops = []
    for t in range(101):
        frame_path = os.path.join(FRAMES_DIR, f'3_FL2_F0_t_{t:03d}_c_0.tif')
        img = tifffile.imread(frame_path)
        crop = img[cy-150:cy+150, cx-150:cx+150]
        raw_crops.append(crop)
        
        m_crop = existing_masks_crop[t]
        if m_crop.sum() > 0:
            existing_mean_ints.append(float(np.mean(crop[m_crop])))
        else:
            existing_mean_ints.append(np.nan)
            
    m0 = existing_masks_crop[0]
    if m0.sum() > 0:
        ey, ex = np.where(m0)
        seed_y, seed_x = np.mean(ey), np.mean(ex)
    else:
        seed_y, seed_x = med_y - (cy - 150), med_x - (cx - 150)
        
    def track_stream(stream_type):
        masks_crop_list = []
        areas = []
        centroids = []
        mean_ints = []
        failures = [False] * 101
        
        curr_y, curr_x = seed_y, seed_x
        
        for t in range(101):
            img_raw = raw_crops[t]
            if stream_type == 'raw':
                inp_img = preprocess_raw_u8(img_raw)
            else:
                inp_img = preprocess_overexposed_u8(img_raw)
                
            m_instances, _, _ = model.eval(inp_img)
            n_instances = np.max(m_instances)
            
            best_id = 0
            min_dist = float('inf')
            for label in range(1, n_instances + 1):
                ys, xs = np.where(m_instances == label)
                if len(ys) == 0: continue
                my, mx = np.mean(ys), np.mean(xs)
                dist = np.hypot(my - curr_y, mx - curr_x)
                if dist < min_dist and dist <= 30.0:
                    min_dist = dist
                    best_id = label
                    
            if best_id > 0:
                mask_sel = (m_instances == best_id)
                ys, xs = np.where(mask_sel)
                my, mx = np.mean(ys), np.mean(xs)
                curr_y, curr_x = my, mx
                
                area = len(ys)
                mean_val = float(np.mean(img_raw[mask_sel]))
                
                masks_crop_list.append(mask_sel)
                areas.append(area)
                centroids.append((my + cy - 150, mx + cx - 150))
                mean_ints.append(mean_val)
            else:
                failures[t] = True
                masks_crop_list.append(np.zeros((300, 300), dtype=bool))
                areas.append(0)
                centroids.append((curr_y + cy - 150, curr_x + cx - 150))
                mean_ints.append(np.nan)
                
        return masks_crop_list, areas, centroids, mean_ints, failures

    print(f"Tracking CellposeSAM-RAW for cell {cell_id}...", flush=True)
    raw_masks, raw_areas, raw_cents, raw_means, raw_fails = track_stream('raw')
    
    print(f"Tracking CellposeSAM-OVEREXPOSED for cell {cell_id}...", flush=True)
    ov_masks, ov_areas, ov_cents, ov_means, ov_fails = track_stream('overexposed')
    
    ex_jumps = calculate_jumps(existing_areas, existing_centroids)
    raw_jumps = calculate_jumps(raw_areas, raw_cents)
    ov_jumps = calculate_jumps(ov_areas, ov_cents)
    
    df_out = pd.DataFrame({
        'frame': np.arange(101),
        'existing_area': existing_areas,
        'existing_mean_int': existing_mean_ints,
        'existing_cy': [c[0] for c in existing_centroids],
        'existing_cx': [c[1] for c in existing_centroids],
        'existing_jump': ex_jumps,
        
        'raw_area': raw_areas,
        'raw_mean_int': raw_means,
        'raw_cy': [c[0] for c in raw_cents],
        'raw_cx': [c[1] for c in raw_cents],
        'raw_fail': raw_fails,
        'raw_jump': raw_jumps,
        
        'overexp_area': ov_areas,
        'overexp_mean_int': ov_means,
        'overexp_cy': [c[0] for c in ov_cents],
        'overexp_cx': [c[1] for c in ov_cents],
        'overexp_fail': ov_fails,
        'overexp_jump': ov_jumps
    })
    
    csv_out_path = os.path.join(OUTPUT_DIR, f'cell_{cell_id}_track_comparison.csv')
    df_out.to_csv(csv_out_path, index=False)
    print(f"Saved track comparison CSV: {csv_out_path}", flush=True)
    
    cell_stats = {
        'cell_id': cell_id,
        'existing_jumps': sum(ex_jumps[:-1]),
        'cellpose_raw_jumps': sum(raw_jumps[:-1]),
        'cellpose_raw_failures': sum(raw_fails),
        'cellpose_overexp_jumps': sum(ov_jumps[:-1]),
        'cellpose_overexp_failures': sum(ov_fails)
    }
    
    return {
        'stats': cell_stats,
        'df': df_out,
        'raw_crops': raw_crops,
        'existing_masks': existing_masks_crop,
        'raw_masks': raw_masks,
        'ov_masks': ov_masks,
        'cy': cy,
        'cx': cx
    }

def generate_timeseries_plot(cell_id, df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    frames = df['frame']
    
    axes[0].plot(frames, df['existing_area'], label='Existing Pipeline', color='#3182bd', lw=2)
    axes[0].plot(frames, df['raw_area'], label='CellposeSAM-RAW', color='#e6550d', lw=2, ls='--')
    axes[0].plot(frames, df['overexp_area'], label='CellposeSAM-OVEREXPOSED', color='#2ca02c', lw=2)
    axes[0].set_ylabel('Area (pixels)', fontsize=12)
    axes[0].set_title(f'Cell {cell_id}: Track Comparison Time Series', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    
    axes[1].plot(frames, df['existing_mean_int'], label='Existing Pipeline', color='#3182bd', lw=2)
    axes[1].plot(frames, df['raw_mean_int'], label='CellposeSAM-RAW', color='#e6550d', lw=2, ls='--')
    axes[1].plot(frames, df['overexp_mean_int'], label='CellposeSAM-OVEREXPOSED', color='#2ca02c', lw=2)
    axes[1].set_ylabel('Mean Raw Intensity (a.u.)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')
    
    def calc_drift(cy, cx):
        if len(cy) == 0 or np.isnan(cy[0]):
            return np.zeros(len(cy))
        return np.sqrt((cy - cy[0])**2 + (cx - cx[0])**2)
        
    ex_drift = calc_drift(df['existing_cy'].values, df['existing_cx'].values)
    raw_drift = calc_drift(df['raw_cy'].values, df['raw_cx'].values)
    ov_drift = calc_drift(df['overexp_cy'].values, df['overexp_cx'].values)
    
    axes[2].plot(frames, ex_drift, label='Existing Pipeline', color='#3182bd', lw=2)
    axes[2].plot(frames, raw_drift, label='CellposeSAM-RAW', color='#e6550d', lw=2, ls='--')
    axes[2].plot(frames, ov_drift, label='CellposeSAM-OVEREXPOSED', color='#2ca02c', lw=2)
    axes[2].set_ylabel('Centroid Drift from t=0 (px)', fontsize=12)
    axes[2].set_xlabel('Frame t', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'cell_{cell_id}_timeseries_comparison.png')
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved timeseries plot: {plot_path}", flush=True)

def generate_overlay_figure(cell_id, raw_crops, ex_masks, raw_masks, ov_masks, timepoints=[0, 7, 50, 100]):
    fig, axes = plt.subplots(len(timepoints), 3, figsize=(12, 4 * len(timepoints)))
    if len(timepoints) == 1:
        axes = np.array([axes])
        
    for idx, t in enumerate(timepoints):
        crop = raw_crops[t]
        p1, p99 = np.percentile(crop, (0.5, 99.5))
        crop_disp = np.clip((crop - p1) / max(1e-5, p99 - p1), 0, 1)
        
        axes[idx, 0].imshow(crop_disp, cmap='gray')
        if np.any(ex_masks[t]):
            try:
                axes[idx, 0].contour(ex_masks[t], levels=[0.5], colors=['#3182bd'], linewidths=1.5)
            except Exception:
                pass
        axes[idx, 0].set_title(f'Existing Pipeline (t={t:03d})', fontsize=11)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(crop_disp, cmap='gray')
        if np.any(raw_masks[t]):
            try:
                axes[idx, 1].contour(raw_masks[t], levels=[0.5], colors=['#e6550d'], linewidths=1.5)
            except Exception:
                pass
        axes[idx, 1].set_title(f'CellposeSAM-RAW (t={t:03d})', fontsize=11)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(crop_disp, cmap='gray')
        if np.any(ov_masks[t]):
            try:
                axes[idx, 2].contour(ov_masks[t], levels=[0.5], colors=['#2ca02c'], linewidths=1.5)
            except Exception:
                pass
        axes[idx, 2].set_title(f'CellposeSAM-OVEREXPOSED (t={t:03d})', fontsize=11)
        axes[idx, 2].axis('off')
        
    plt.suptitle(f'Cell {cell_id}: Segmentation Mask Contours Overlaid on Raw Crops', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'cell_{cell_id}_overlay_multi_t.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved overlay figure: {fig_path}", flush=True)

def main():
    use_gpu = torch.cuda.is_available()
    print(f"Initializing CellposeModel (pretrained_model='cpsam', gpu={use_gpu})...", flush=True)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model='cpsam')
    
    summary_stats = []
    summary_csv_path = os.path.join(OUTPUT_DIR, 'summary_table.csv')
    plot_cells = [25, 116, 9, 178]
    
    for cid in TARGET_CELLS:
        res = run_cell_benchmark(cid, model)
        if res is not None:
            summary_stats.append(res['stats'])
            if cid in plot_cells:
                try:
                    generate_timeseries_plot(cid, res['df'])
                except Exception as e:
                    print(f"Timeseries plot error cell {cid}: {e}", flush=True)
                try:
                    generate_overlay_figure(cid, res['raw_crops'], res['existing_masks'], res['raw_masks'], res['ov_masks'])
                except Exception as e:
                    print(f"Overlay plot error cell {cid}: {e}", flush=True)
            
            # Save summary table incrementally
            pd.DataFrame(summary_stats).to_csv(summary_csv_path, index=False)
            print(f"Updated summary table incrementally ({len(summary_stats)} cells completed).", flush=True)
            
            del res
            gc.collect()
            if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
            
    summary_df = pd.DataFrame(summary_stats)
    print("\n================ FINAL SUMMARY TABLE ================", flush=True)
    print(summary_df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
