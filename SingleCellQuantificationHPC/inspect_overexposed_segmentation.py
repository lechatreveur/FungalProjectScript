import os
import sys
import numpy as np
import cv2
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cellpose import models

FRAMES_DIR = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/3_FL2_F0/Frames_3_FL2_F0/"
OUT_DIR = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/cellpose_overexposure_test/inspection/"

os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_overexposed(img_raw, smooth_sigma=1.0, bg_sigma=20.0, p_stretch=(5, 95), gamma=0.4):
    """Preprocess raw 16-bit image for overexposed-style segmentation.
    Returns uint16 (0-65535) — does NOT convert to 8-bit.
    Cellpose/CellposeSAM handles uint16 natively.
    """
    img_f = img_raw.astype(np.float32)
    if smooth_sigma > 0:
        img_f = cv2.GaussianBlur(img_f, (0, 0), smooth_sigma)
    if bg_sigma > 0:
        bg = cv2.GaussianBlur(img_f, (0, 0), bg_sigma)
        img_f = np.maximum(0.0, img_f - bg)
    p_low, p_high = np.percentile(img_f, p_stretch)
    if p_high > p_low:
        img_f = np.clip((img_f - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        img_f = np.zeros_like(img_f)
    if gamma != 1.0:
        img_f = np.power(img_f, gamma)
    return (img_f * 65535.0).astype(np.uint16)

def inspect_frame(t=0, smooth_sigma=1.0, bg_sigma=20.0, p_low=5, p_high=95, gamma=0.40, diameter=None, min_size=None):
    frame_path = os.path.join(FRAMES_DIR, f"3_FL2_F0_t_{t:03d}_c_0.tif")
    print(f"Reading raw 16-bit TIFF frame t={t}: {frame_path}...", flush=True)
    img_raw_full = tifffile.imread(frame_path)

    # 600x600 ROI crop
    r0, r1 = 800, 1400
    c0, c1 = 800, 1400
    img_raw = img_raw_full[r0:r1, c0:c1]

    print(f"Applying overexposure preprocessing (uint16 output): smooth={smooth_sigma}, bg={bg_sigma}, P=({p_low},{p_high}), gamma={gamma}...", flush=True)
    img_proc = preprocess_overexposed(img_raw, smooth_sigma=smooth_sigma, bg_sigma=bg_sigma,
                                      p_stretch=(p_low, p_high), gamma=gamma)

    print(f"Running CellposeModel ('cpsam') segmentation on preprocessed uint16 crop (diameter={diameter}, min_size={min_size})...", flush=True)
    model = models.CellposeModel(gpu=False, pretrained_model='cpsam')
    m_instances, flows, styles = model.eval(img_proc, diameter=diameter, min_size=min_size)
    num_cells = int(np.max(m_instances))
    print(f"CellposeSAM detected {num_cells} cells on the crop at t={t}.", flush=True)

    # --- Normalize both images for display ---
    # Raw: percentile stretch for display only
    raw_p1, raw_p99 = np.percentile(img_raw, (1, 99.5))
    raw_disp = np.clip((img_raw.astype(np.float32) - raw_p1) / max(1e-5, raw_p99 - raw_p1), 0, 1)

    # Preprocessed uint16: normalize to [0,1] for display
    proc_disp = img_proc.astype(np.float32) / 65535.0

    # Build uint8 BGR base for contour overlay (from proc_disp)
    proc_u8 = (proc_disp * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(proc_u8, cv2.COLOR_GRAY2BGR)
    colored_masks = np.zeros((*img_proc.shape, 3), dtype=np.uint8)
    cmap_mask = plt.colormaps.get_cmap('hsv').resampled(max(2, num_cells + 1))

    for cid in range(1, num_cells + 1):
        cell_m = (m_instances == cid)
        if not cell_m.any():
            continue
        color = (np.array(cmap_mask(cid)[:3]) * 255).astype(np.uint8)
        colored_masks[cell_m] = color
        contours, _ = cv2.findContours(cell_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(img_bgr, [cnt], -1, (0, 255, 0), 2)

    overlay = cv2.addWeighted(img_bgr, 0.7, colored_masks, 0.3, 0)

    # --- Main 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(raw_disp, cmap='gray')
    axes[0].set_title(f"Raw 16-Bit Crop (t={t:03d})\nPercentile Display (1%-99.5%)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(proc_disp, cmap='gray')
    axes[1].set_title(f"Preprocessed uint16\nSmooth={smooth_sigma}, DoG={bg_sigma}, P=({p_low},{p_high}), γ={gamma}", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"CellposeSAM ('cpsam') Segmentation\nDetected Cells: {num_cells}", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f"Overexposure Inspection — Frame t={t:03d} | smooth={smooth_sigma}, bg={bg_sigma}, P=({p_low},{p_high}), γ={gamma}", fontsize=12, color='gray')
    plt.tight_layout()
    diam_str = f"diam_{diameter}" if diameter is not None else "diam_None"
    min_str = f"min_{min_size}" if min_size is not None else "min_None"
    roi_fig_path = os.path.join(OUT_DIR, f"overexposed_inspection_roi_t_{t:03d}_{diam_str}_{min_str}.png")
    plt.savefig(roi_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved inspection plot to: {roi_fig_path}", flush=True)

if __name__ == "__main__":
    t_inspect     = int(sys.argv[1])   if len(sys.argv) > 1 else 0
    p_low_arg     = int(sys.argv[2])   if len(sys.argv) > 2 else 5
    p_high_arg    = int(sys.argv[3])   if len(sys.argv) > 3 else 95
    smooth_arg    = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    bg_arg        = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
    gamma_arg     = float(sys.argv[6]) if len(sys.argv) > 6 else 0.40
    
    # Allow optional diameter and min_size arguments
    diameter_arg  = float(sys.argv[7]) if (len(sys.argv) > 7 and sys.argv[7] != 'None') else None
    min_size_arg  = int(sys.argv[8])   if (len(sys.argv) > 8 and sys.argv[8] != 'None') else None
    
    inspect_frame(t_inspect, smooth_sigma=smooth_arg, bg_sigma=bg_arg,
                  p_low=p_low_arg, p_high=p_high_arg, gamma=gamma_arg,
                  diameter=diameter_arg, min_size=min_size_arg)
