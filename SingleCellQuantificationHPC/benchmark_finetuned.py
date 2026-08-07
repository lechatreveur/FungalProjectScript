#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_finetuned.py
----------------------
Compares base SAM2 Tiny vs fine-tuned SAM2 Tiny (fungal-domain adapted)
on the 37 held-out validation sequences from the fine-tuning dataset.

Sequences are sourced directly from the exported VOS dataset so the
benchmark is completely independent from the training set.
"""

import os
import sys
import time
import json
import tempfile
import shutil
import argparse
import threading
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Ensure local imports work
HPC_DIR = Path(__file__).parent.resolve()
SAM2_DIR = HPC_DIR.parent / "segment-anything-2"
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(HPC_DIR.parent))
sys.path.insert(0, str(SAM2_DIR))

from sam2.build_sam import build_sam2_video_predictor

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT   = Path("/Volumes/X10 Pro/Movies/AI/sam2_finetune_dataset")
VAL_FILELIST   = DATASET_ROOT / "val_filelist.txt"
JPEG_DIR       = DATASET_ROOT / "JPEGImages"
ANN_DIR        = DATASET_ROOT / "Annotations"

BASE_CKPT      = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
FINETUNED_CKPT = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/segment-anything-2/checkpoints/checkpoint.pt")
EXTRACTED_CKPT = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/segment-anything-2/checkpoints/sam2.1_hiera_tiny_fungal_finetuned.pt")

MODEL_CFG      = "configs/sam2.1/sam2.1_hiera_t.yaml"

ARTIFACT_DIR   = Path("/Users/user/.gemini/antigravity-ide/brain/4dcb4bf4-544f-4647-96fd-e18a94325d83")


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_model_weights(training_ckpt_path: Path, out_path: Path):
    """
    The SAM2 trainer saves a dict with keys:
      model, optimizer, epoch, loss, steps, time_elapsed, scaler
    build_sam2_video_predictor expects the same raw format as the
    official checkpoints: {'model': state_dict}.
    We extract just the 'model' key and save it in that format.
    """
    print(f"Extracting model weights from training checkpoint: {training_ckpt_path}")
    ckpt = torch.load(training_ckpt_path, map_location="cpu")
    model_state = ckpt["model"]
    torch.save({"model": model_state}, out_path)
    print(f"Saved inference-ready weights to: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def load_palette_mask(ann_path: Path, H: int, W: int) -> np.ndarray:
    """Load a palettised PNG annotation and return a binary bool mask for object 1."""
    img = Image.open(ann_path).convert("P")
    arr = np.array(img)
    return (arr == 1).astype(bool)


def track_sequence_with_sam2(
    seq_name: str,
    predictor,
    device: str,
) -> dict:
    """
    Run SAM2 video predictor on a VOS-format sequence.
    Returns dict {frame_idx: bool_mask}.
    """
    img_seq_dir  = JPEG_DIR / seq_name
    ann_seq_dir  = ANN_DIR  / seq_name

    frame_files = sorted([f for f in img_seq_dir.iterdir()
                          if f.suffix.lower() == ".jpg" and not f.name.startswith(".")])
    ann_files   = sorted([f for f in ann_seq_dir.iterdir()
                          if f.suffix.lower() == ".png" and not f.name.startswith(".")])

    if not frame_files or not ann_files:
        return {}

    # Use the first annotation frame as the initial mask (SAM2 prompt)
    first_ann = ann_files[0]
    first_idx_str = first_ann.stem  # e.g. "00000"
    first_idx = int(first_idx_str)

    ref_img = Image.open(frame_files[first_idx]).convert("RGB")
    H, W = ref_img.size[1], ref_img.size[0]
    initial_mask = load_palette_mask(first_ann, H, W)

    if not initial_mask.any():
        return {}

    # Build a temp directory of RGB JPEGs for the video predictor
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, fp in enumerate(frame_files):
            shutil.copy(fp, os.path.join(tmpdir, f"{i:05d}.jpg"))

        with torch.inference_mode():
            inference_state = predictor.init_state(video_path=tmpdir, async_loading_frames=False)

        # Prompt with the GT mask at the first annotated frame
        with torch.inference_mode():
            _, obj_ids, mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=first_idx,
                obj_id=1,
                mask=initial_mask,
            )

        # Propagate forward
        results = {}
        with torch.inference_mode():
            for out_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if 1 in out_obj_ids:
                    obj_pos = out_obj_ids.index(1)
                    pred_mask = (out_mask_logits[obj_pos] > 0.0).cpu().numpy().squeeze()
                    results[out_idx] = pred_mask.astype(bool)

        predictor.reset_state(inference_state)

    return results


def evaluate_sequence(seq_name: str, pred_masks: dict) -> dict:
    """Compute IoU metrics against the ground truth annotations."""
    ann_seq_dir = ANN_DIR / seq_name
    ann_files = sorted([f for f in ann_seq_dir.iterdir()
                        if f.suffix.lower() == ".png" and not f.name.startswith(".")])

    if not ann_files:
        return {}

    ref_img_path = JPEG_DIR / seq_name / ann_files[0].name.replace(".png", ".jpg")
    ref_img = Image.open(ref_img_path)
    H, W = ref_img.size[1], ref_img.size[0]

    ious_list = []
    for ann_f in ann_files:
        frame_idx = int(ann_f.stem)
        gt_mask = load_palette_mask(ann_f, H, W)
        if not gt_mask.any():
            continue
        pred = pred_masks.get(frame_idx, np.zeros((H, W), dtype=bool))
        ious_list.append(iou(gt_mask, pred))

    if not ious_list:
        return {}

    return {
        "mean_iou":    float(np.mean(ious_list)),
        "survival":    float(np.mean([1.0 if v >= 0.5 else 0.0 for v in ious_list])),
        "final_iou":   float(ious_list[-1]),
        "n_frames":    len(ious_list),
    }


def run_benchmark(sequences, predictor, label, device) -> list:
    rows = []
    for seq in tqdm(sequences, desc=f"  [{label}]"):
        t0 = time.time()
        try:
            preds = track_sequence_with_sam2(seq, predictor, device)
            elapsed = time.time() - t0
            metrics = evaluate_sequence(seq, preds)
            if metrics:
                metrics["seq"]      = seq
                metrics["model"]    = label
                metrics["duration"] = elapsed
                metrics["fps"]      = metrics["n_frames"] / elapsed if elapsed > 0 else 0.0
                rows.append(metrics)
                print(f"    {seq}: Mean IoU={metrics['mean_iou']:.3f}, "
                      f"Surv={metrics['survival']*100:.1f}%, "
                      f"Final={metrics['final_iou']:.3f}")
        except Exception as e:
            print(f"    [Error] {seq}: {e}")
    return rows


def print_summary(df: pd.DataFrame, label: str):
    sub = df[df["model"] == label]
    if sub.empty:
        return
    print(f"\n  {label}:")
    print(f"    Mean IoU  : {sub['mean_iou'].mean():.4f}")
    print(f"    Survival  : {sub['survival'].mean()*100:.1f}%")
    print(f"    Final IoU : {sub['final_iou'].mean():.4f}")
    print(f"    Avg Speed : {sub['fps'].mean():.2f} fps")


def keep_alive_drive(drive_path: Path, interval=10):
    """Periodically writes to a file on the external drive to prevent it from sleeping."""
    keep_alive_file = drive_path / ".keep_alive"
    while True:
        try:
            with open(keep_alive_file, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, mps, or cuda")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N sequences (0 = all 37)")
    args = parser.parse_args()

    # Start keep-alive daemon thread for the external drive
    t = threading.Thread(target=keep_alive_drive, args=(DATASET_ROOT,), daemon=True)
    t.start()

    # ── Load val sequences ────────────────────────────────────────────────────
    with open(VAL_FILELIST) as f:
        sequences = [line.strip() for line in f if line.strip()]

    if args.limit > 0:
        sequences = sequences[:args.limit]

    print(f"\n{'='*60}")
    print(f"Fine-Tuned SAM2 Benchmark — {len(sequences)} validation sequences")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # ── Extract fine-tuned weights ────────────────────────────────────────────
    if not EXTRACTED_CKPT.exists():
        extract_model_weights(FINETUNED_CKPT, EXTRACTED_CKPT)
    else:
        print(f"Using cached extracted weights: {EXTRACTED_CKPT}")

    all_rows = []

    # ── Run Base SAM2 ─────────────────────────────────────────────────────────
    print("\n[1/2] Benchmarking BASE SAM2 Tiny...")
    base_predictor = build_sam2_video_predictor(MODEL_CFG, str(BASE_CKPT), device=args.device)
    base_rows = run_benchmark(sequences, base_predictor, "Base SAM2 Tiny", args.device)
    all_rows.extend(base_rows)
    del base_predictor

    # ── Run Fine-Tuned SAM2 ───────────────────────────────────────────────────
    print("\n[2/2] Benchmarking FINE-TUNED SAM2 Tiny (fungal-adapted)...")
    ft_predictor = build_sam2_video_predictor(MODEL_CFG, str(EXTRACTED_CKPT), device=args.device)
    ft_rows = run_benchmark(sequences, ft_predictor, "Fine-Tuned SAM2", args.device)
    all_rows.extend(ft_rows)
    del ft_predictor

    # ── Aggregate ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print_summary(df, "Base SAM2 Tiny")
    print_summary(df, "Fine-Tuned SAM2")

    # Per-sequence comparison
    base_df = df[df["model"] == "Base SAM2 Tiny"].set_index("seq")
    ft_df   = df[df["model"] == "Fine-Tuned SAM2"].set_index("seq")
    common  = base_df.index.intersection(ft_df.index)

    if len(common) > 0:
        delta_iou  = (ft_df.loc[common, "mean_iou"]  - base_df.loc[common, "mean_iou"]).mean()
        delta_surv = (ft_df.loc[common, "survival"]  - base_df.loc[common, "survival"]).mean()
        delta_fin  = (ft_df.loc[common, "final_iou"] - base_df.loc[common, "final_iou"]).mean()
        improved   = (ft_df.loc[common, "mean_iou"]  > base_df.loc[common, "mean_iou"]).sum()
        print(f"\n  Δ Mean IoU (FT − Base)  : {delta_iou:+.4f}")
        print(f"  Δ Survival  (FT − Base) : {delta_surv*100:+.1f}%")
        print(f"  Δ Final IoU (FT − Base) : {delta_fin:+.4f}")
        print(f"  Sequences improved      : {improved}/{len(common)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = HPC_DIR / "benchmark_finetuned_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nDetailed results saved to: {out_csv}")

    # ── Save Markdown report ──────────────────────────────────────────────────
    base_sub = df[df["model"] == "Base SAM2 Tiny"]
    ft_sub   = df[df["model"] == "Fine-Tuned SAM2"]

    report_path = ARTIFACT_DIR / "finetuned_benchmark_report.md"
    md = f"""# SAM2 Fine-Tuned vs Base: Validation Benchmark

Comparison of **Base SAM2 Tiny** vs **Fine-Tuned SAM2 Tiny** (fungal-domain adapted)
evaluated on **{len(common)} held-out validation sequences** from the fine-tuning dataset.

> These sequences were **not seen during training** (20% held-out split).

## Aggregate Results

| Metric | Base SAM2 Tiny | Fine-Tuned SAM2 | Δ Change |
| :--- | :---: | :---: | :---: |
| **Mean IoU** | {base_sub['mean_iou'].mean():.4f} | {ft_sub['mean_iou'].mean():.4f} | **{ft_sub['mean_iou'].mean() - base_sub['mean_iou'].mean():+.4f}** |
| **Survival Rate** (IoU ≥ 0.5) | {base_sub['survival'].mean()*100:.1f}% | {ft_sub['survival'].mean()*100:.1f}% | **{(ft_sub['survival'].mean() - base_sub['survival'].mean())*100:+.1f}%** |
| **Final Frame IoU** | {base_sub['final_iou'].mean():.4f} | {ft_sub['final_iou'].mean():.4f} | **{ft_sub['final_iou'].mean() - base_sub['final_iou'].mean():+.4f}** |
| **Speed** (FPS) | {base_sub['fps'].mean():.2f} | {ft_sub['fps'].mean():.2f} | {ft_sub['fps'].mean() - base_sub['fps'].mean():+.2f} |

**Sequences where FT > Base:** {improved}/{len(common)}

## Per-Sequence Breakdown

| Sequence | Base IoU | FT IoU | Δ IoU | Base Surv | FT Surv |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for seq in sorted(common):
        b_iou = base_df.loc[seq, "mean_iou"]
        f_iou = ft_df.loc[seq, "mean_iou"]
        b_s   = base_df.loc[seq, "survival"]
        f_s   = ft_df.loc[seq, "survival"]
        delta = f_iou - b_iou
        arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "≈")
        md += f"| {seq} | {b_iou:.3f} | {f_iou:.3f} | {arrow} {delta:+.3f} | {b_s*100:.0f}% | {f_s*100:.0f}% |\n"

    with open(report_path, "w") as f:
        f.write(md)
    print(f"Markdown report saved to: {report_path}")


if __name__ == "__main__":
    main()
