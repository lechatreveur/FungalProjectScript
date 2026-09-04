import os
import sys
import glob
import pandas as pd
import numpy as np
import cv2
import tifffile

def rle_decode(rle_str, shape):
    if not isinstance(rle_str, str) or not rle_str or rle_str == 'nan':
        return np.zeros(shape, dtype=bool)
    s = rle_str.split()
    if len(s) < 2:
        return np.zeros(shape, dtype=bool)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=bool)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = True
    return img.reshape(shape)


LEGACY_TRACKED_DIR = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/3_FL2_F0/TrackedCells_3_FL2_F0/"
NEW_TRACKED_DIR = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/3_FL2_F0/TrackedCells_3_FL2_F0_cpsam_overexp/"
FRAMES_DIR = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/3_FL2_F0/Frames_3_FL2_F0/"
MOVIES_OUTPUT_DIR = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/cellpose_overexposure_test/single_cell_movies/"

os.makedirs(MOVIES_OUTPUT_DIR, exist_ok=True)

def generate_single_cell_movie(cell_id):
    legacy_csv = os.path.join(LEGACY_TRACKED_DIR, f'cell_{cell_id}_masks.csv')
    new_csv = os.path.join(NEW_TRACKED_DIR, f'cell_{cell_id}_masks.csv')
    
    if not os.path.exists(legacy_csv) or not os.path.exists(new_csv):
        print(f"Skipping cell {cell_id}: missing CSV", flush=True)
        return
        
    leg_df = pd.read_csv(legacy_csv)
    new_df = pd.read_csv(new_csv)
    
    all_cy, all_cx = [], []
    for df in [leg_df, new_df]:
        rle_col = 'rle_gfp' if 'rle_gfp' in df.columns else 'rle_bf'
        for _, row in df.iterrows():
            m = rle_decode(row[rle_col], (int(row['height']), int(row['width'])))
            if m.sum() > 0:
                ys, xs = np.where(m)
                all_cy.append(np.mean(ys))
                all_cx.append(np.mean(xs))
                
    if len(all_cy) == 0:
        return
        
    med_y, med_x = np.median(all_cy), np.median(all_cx)
    cy = int(np.clip(round(med_y), 100, 1900))
    cx = int(np.clip(round(med_x), 100, 1900))
    r0, r1 = cy - 100, cy + 100
    c0, c1 = cx - 100, cx + 100
    
    out_mp4 = os.path.join(MOVIES_OUTPUT_DIR, f'cell_{cell_id}_comparison.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_mp4, fourcc, 7, (400, 240))
    
    for t in range(min(len(leg_df), len(new_df))):
        frame_p = os.path.join(FRAMES_DIR, f'3_FL2_F0_t_{t:03d}_c_0.tif')
        if not os.path.exists(frame_p):
            continue
            
        img_raw = tifffile.imread(frame_p)
        crop = img_raw[r0:r1, c0:c1]
        
        p1, p99 = np.percentile(crop, (0.5, 99.5))
        crop_norm = np.clip((crop - p1) / max(1e-5, p99 - p1), 0, 1)
        crop_u8 = (crop_norm * 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2BGR)
        
        # Panel 1: Existing Pipeline (Blue Contour)
        panel1 = crop_bgr.copy()
        leg_rle = leg_df.iloc[t]['rle_gfp'] if 'rle_gfp' in leg_df.columns else leg_df.iloc[t]['rle_bf']
        leg_m = rle_decode(leg_rle, (2000, 2000))[r0:r1, c0:c1]
        if leg_m.any():
            contours, _ = cv2.findContours(leg_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel1, contours, -1, (255, 130, 49), 2)
        cv2.putText(panel1, f'Existing (t={t:03d})', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Panel 2: CellposeSAM-OVEREXPOSED (Green Contour)
        panel2 = crop_bgr.copy()
        new_rle = new_df.iloc[t]['rle_gfp'] if 'rle_gfp' in new_df.columns else new_df.iloc[t]['rle_bf']
        new_m = rle_decode(new_rle, (2000, 2000))[r0:r1, c0:c1]
        if new_m.any():
            contours, _ = cv2.findContours(new_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel2, contours, -1, (44, 160, 44), 2)
        cv2.putText(panel2, f'CPSAM-Overexp (t={t:03d})', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        combined_frame = np.zeros((240, 400, 3), dtype=np.uint8)
        combined_frame[20:220, 0:200] = cv2.resize(panel1, (200, 200))
        combined_frame[20:220, 200:400] = cv2.resize(panel2, (200, 200))
        
        cv2.putText(combined_frame, f'Cell {cell_id} Tracking Comparison', (60, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(combined_frame)
        
    writer.release()
    print(f'Done rendering MP4 for cell {cell_id}', flush=True)

if __name__ == '__main__':
    finished_files = glob.glob(os.path.join(NEW_TRACKED_DIR, 'cell_*_masks.csv'))
    finished_ids = sorted([int(os.path.basename(f).split('_')[1]) for f in finished_files])
    print(f'Rendering MP4 movies for finished cells: {finished_ids}', flush=True)
    for cid in finished_ids:
        generate_single_cell_movie(cid)
