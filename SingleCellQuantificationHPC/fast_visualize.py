import os
import sys
import torch
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from SingleCellQuantificationHPC.tracker_model import load_tracker
from SingleCellQuantificationHPC.tracker_dataset import _padded_bbox, _crop_and_resize, _norm_img
from Cell_tracking_functions import rle_decode, to_labeled_current, load_segmentation, compute_overlap

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_tracker("./tracker_checkpoints/model_best.pt", device=device)
    
    # Manually load 3 samples from A14-YES-1t-FBFBF-4_F2
    film_dir = Path("/Volumes/X10 Pro/Movies/2025_12_31_M92/A14-YES-1t-FBFBF-4_F2")
    if not film_dir.exists():
        print("Film dir not found")
        sys.exit(1)
        
    frames_dir = film_dir / "Frames_A14-YES-1t-FBFBF-4_F2"
    masks_dir = film_dir / "Masks_A14-YES-1t-FBFBF-4_F2"
    tracked_dir = film_dir / "TrackedCells_A14-YES-1t-FBFBF-4_F2"
    
    # Get a few cells
    qc = pd.read_csv(tracked_dir / "qc.csv")
    corrected = qc[qc["status"].str.lower() == "corrected"]["cell_id"].tolist()
    
    if not corrected:
        print("No corrected cells found!")
        return
    
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    row_idx = 0
    
    for cell_id in corrected[:10]:
        df = pd.read_csv(tracked_dir / f"cell_{cell_id}_masks.csv")
        H, W = int(df.iloc[0]["height"]), int(df.iloc[0]["width"])
        
        # just pick first t and t+1
        t = df["time_point"].min()
        row_t = df[df["time_point"] == t]
        row_t1 = df[df["time_point"] == t+1]
        
        if row_t.empty or row_t1.empty:
            continue
            
        rle_t = str(row_t.iloc[0]["rle_bf"])
        rle_t1 = str(row_t1.iloc[0]["rle_bf"])
        
        mask_t = rle_decode(rle_t, (H, W)).astype(bool)
        mask_t1 = rle_decode(rle_t1, (H, W)).astype(bool)
        
        img_t = imread(str(frames_dir / f"A14-YES-1t-FBFBF-4_F2_t_{t:03d}_c_0.tif"))
        img_t1 = imread(str(frames_dir / f"A14-YES-1t-FBFBF-4_F2_t_{t+1:03d}_c_0.tif"))
        
        r0, r1, c0, c1 = _padded_bbox(mask_t, H, W)
        bf_t = _crop_and_resize(_norm_img(img_t), r0, r1, c0, c1)
        bf_t1 = _crop_and_resize(_norm_img(img_t1), r0, r1, c0, c1)
        mk_t = _crop_and_resize(mask_t.astype(np.float32), r0, r1, c0, c1)
        
        print(f"Cell {cell_id} | img min/max: {img_t.min()}/{img_t.max()} | crop min/max: {bf_t.min()}/{bf_t.max()}")
        
        img3ch = np.stack([bf_t, bf_t1, mk_t], axis=0).astype(np.float32)
        img3ch_t = torch.from_numpy(img3ch).unsqueeze(0).to(device)
        
        # Load seg candidates at t=1
        seg_t1 = to_labeled_current(load_segmentation(str(masks_dir / "A14-YES-1t-FBFBF-4_F2_t_001_c_0_seg.tif")))
        labels = [l for l in np.unique(seg_t1) if l != 0]
        ranked = sorted([(compute_overlap(mask_t1, seg_t1 == l), l) for l in labels], reverse=True)
        
        candidates = np.zeros((3, 1, 128, 128), dtype=np.float32)
        candidates[0, 0] = _crop_and_resize(mask_t1.astype(np.float32), r0, r1, c0, c1)
        if len(ranked) > 0:
            candidates[1, 0] = _crop_and_resize((seg_t1 == ranked[0][1]).astype(np.float32), r0, r1, c0, c1)
        if len(ranked) > 1:
            candidates[2, 0] = _crop_and_resize((seg_t1 == ranked[1][1]).astype(np.float32), r0, r1, c0, c1)
            
        cands_t = torch.from_numpy(candidates).unsqueeze(0).to(device)
        
        sep_t = torch.tensor([[0.0]], dtype=torch.float32).to(device)
        sep_t1 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
        area_ratio = torch.tensor([[1.0]], dtype=torch.float32).to(device)
        adjacency = torch.tensor([[0.0]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            scores, _, _ = model(img3ch_t, cands_t, sep_t, area_ratio, sep_t1, adjacency)
        scores = scores.squeeze(0).cpu().numpy()
        
        axes[row_idx, 0].imshow(bf_t, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_title("BF at t")
        axes[row_idx, 0].axis("off")
        
        axes[row_idx, 1].imshow(bf_t1, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("BF at t+1")
        axes[row_idx, 1].axis("off")
        
        axes[row_idx, 2].imshow(mk_t, cmap="hot", vmin=0, vmax=1)
        axes[row_idx, 2].set_title("Mask at t")
        axes[row_idx, 2].axis("off")
        
        for k in range(3):
            axes[row_idx, 3+k].imshow(candidates[k, 0], cmap="Blues", vmin=0, vmax=1)
            is_gt = " (GT)" if k == 0 else ""
            axes[row_idx, 3+k].set_title(f"Cand {k}{is_gt}\nScore: {scores[k]:.2f}")
            axes[row_idx, 3+k].axis("off")
            
        row_idx += 1
        if row_idx >= 3:
            break
            
    plt.tight_layout()
    plt.savefig("tracking_viz.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
