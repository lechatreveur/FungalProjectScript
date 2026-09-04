#!/usr/bin/env python3
"""
train_cellposesam.py - Reproducible Cellpose-SAM Fine-Tuning Pipeline

Fine-tunes the Cellpose-SAM (`cpsam`) model on human-curated microscopy ground truth
datasets (e.g. NeonGreenGFP / M160) using Apple Silicon MPS GPU acceleration.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python SingleCellQuantificationHPC/train_cellposesam.py \
        --train_dir "/Volumes/X10 Pro/Movies/cellpose_training_data/NeonGreenGFP" \
        --n_epochs 50 \
        --learning_rate 1e-5 \
        --seed 0
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Ensure OpenMP compatibility on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from cellpose import io, models, train, core


def get_git_info() -> dict:
    """Extract git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "dirty": len(status) > 0}
    except Exception:
        return {"commit": "unknown", "dirty": False}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Cellpose-SAM on curated microscopy masks.")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/Volumes/X10 Pro/Movies/cellpose_training_data/NeonGreenGFP",
        help="Path to folder containing raw .tif and _masks.tif pairs.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Optional separate test dataset folder. If omitted, an 85/15 train/val split is generated.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="cpsam",
        help="Base Cellpose model to fine-tune from (default: cpsam).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Custom name for the output model. Default: cpsam_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for AdamW optimizer (default: 1e-5).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for AdamW optimizer (default: 0.1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    parser.add_argument(
        "--bsize",
        type=int,
        default=256,
        help="Block size for tiles (default: 256).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10).",
    )
    parser.add_argument(
        "--nimg_per_epoch",
        type=int,
        default=None,
        help="Number of images sampled per epoch (default: None, trains on all available images).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Fraction of dataset to reserve for validation if test_dir is not provided (default: 0.15).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic train/val splitting and training (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fix random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or f"cpsam_{timestamp_str}"

    train_path = Path(args.train_dir).resolve()
    if not train_path.exists():
        print(f"ERROR: train_dir does not exist: {train_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"Cellpose-SAM Fine-Tuning Pipeline: {model_name}")
    print("=" * 70)
    print(f"Training Directory: {train_path}")
    print(f"Base Model:         {args.base_model}")
    print(f"Epochs:             {args.n_epochs}")
    print(f"Learning Rate:      {args.learning_rate}")
    print(f"Weight Decay:       {args.weight_decay}")
    print(f"Batch Size / Tile:  {args.batch_size} (tile size: {args.bsize})")
    print(f"Random Seed:        {args.seed}")

    # Discover and match image and mask files
    img_files = sorted(io.get_image_files(str(train_path), mask_filter="_masks", look_one_level_down=False))
    lbl_files_res = io.get_label_files(img_files, mask_filter="_masks")
    lbl_files = lbl_files_res[0] if isinstance(lbl_files_res, tuple) else lbl_files_res

    if len(img_files) == 0:
        print(f"ERROR: No image files found in {train_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(img_files)} total matched image-mask pairs.")

    # Train / Val Split
    if args.test_dir:
        test_path = Path(args.test_dir).resolve()
        test_img_files = sorted(io.get_image_files(str(test_path), mask_filter="_masks", look_one_level_down=False))
        test_lbl_res = io.get_label_files(test_img_files, mask_filter="_masks")
        test_lbl_files = test_lbl_res[0] if isinstance(test_lbl_res, tuple) else test_lbl_res
        train_img_files = img_files
        train_lbl_files = lbl_files

    else:
        n_total = len(img_files)
        indices = np.arange(n_total)
        # Deterministic shuffle
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)

        n_val = max(1, int(n_total * args.val_fraction))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_img_files = [img_files[i] for i in train_idx]
        train_lbl_files = [lbl_files[i] for i in train_idx]
        test_img_files = [img_files[i] for i in val_idx]
        test_lbl_files = [lbl_files[i] for i in val_idx]

    print(f"Dataset Split: {len(train_img_files)} Training Images, {len(test_img_files)} Validation Images.")

    # Destination directories
    save_models_dir = train_path / "models"
    save_models_dir.mkdir(parents=True, exist_ok=True)
    user_cellpose_dir = Path.home() / ".cellpose" / "models"
    user_cellpose_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Base Model
    use_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    print(f"\nInitializing base CellposeModel '{args.base_model}' (use_gpu={use_gpu})...")
    cp_model = models.CellposeModel(gpu=use_gpu, pretrained_model=args.base_model)

    start_time = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    print(f"\n--- Starting Training ({args.n_epochs} epochs) ---")
    saved_model_path, train_losses, test_losses = train.train_seg(
        net=cp_model.net,
        train_files=train_img_files,
        train_labels_files=train_lbl_files,
        test_files=test_img_files,
        test_labels_files=test_lbl_files,
        load_files=True,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        nimg_per_epoch=args.nimg_per_epoch,
        save_path=str(save_models_dir),
        save_every=args.save_every,
        save_each=False,
        bsize=args.bsize,
        model_name=model_name,
    )


    elapsed_time = time.time() - start_time
    print(f"\n--- Training Complete in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min) ---")

    final_model_path = Path(saved_model_path) if saved_model_path else (save_models_dir / model_name)
    if not final_model_path.exists():
        # Look for model in save_models_dir
        candidates = list(save_models_dir.glob(f"{model_name}*"))
        if candidates:
            final_model_path = candidates[0]

    print(f"Saved Checkpoint: {final_model_path}")

    # Register in user ~/.cellpose/models/ for Cellpose GUI / scripts
    user_model_target = user_cellpose_dir / final_model_path.name
    if final_model_path.exists():
        shutil.copy2(str(final_model_path), str(user_model_target))
        print(f"Registered in ~/.cellpose/models/: {user_model_target}")

        # Also register in gui_models.txt so the Cellpose GUI shows it in the models list
        gui_txt = user_cellpose_dir / "gui_models.txt"
        existing_models = []
        if gui_txt.exists():
            existing_models = [l.strip() for l in gui_txt.read_text().splitlines() if l.strip()]
        if final_model_path.name not in existing_models:
            existing_models.append(final_model_path.name)
            gui_txt.write_text("\n".join(existing_models) + "\n")
            print(f"Registered in {gui_txt}")


    # Compute provenance record
    git_info = get_git_info()
    provenance = {
        "artifact": final_model_path.name,
        "created": start_utc,
        "created_by": "SingleCellQuantificationHPC/train_cellposesam.py",
        "command": f"python {' '.join(sys.argv)}",
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "host": os.uname().nodename,
        "pythonhashseed": str(args.seed),
        "training_data": {
            "train_dir": str(train_path),
            "total_pairs": len(img_files),
            "train_count": len(train_img_files),
            "val_count": len(test_img_files),
        },
        "hyperparams": {
            "base_model": args.base_model,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "bsize": args.bsize,
            "seed": args.seed,
        },
        "elapsed_seconds": round(elapsed_time, 2),
        "final_train_loss": float(train_losses[-1]) if len(train_losses) > 0 else None,
        "final_test_loss": float(test_losses[-1]) if len(test_losses) > 0 else None,
        "registered_paths": [
            str(final_model_path),
            str(user_model_target),
        ],
    }

    # Save provenance sidecars
    prov_file_1 = final_model_path.parent / f"{final_model_path.name}.provenance.json"
    prov_file_2 = user_cellpose_dir / f"{final_model_path.name}.provenance.json"

    with open(prov_file_1, "w") as f:
        json.dump(provenance, f, indent=2)
    with open(prov_file_2, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"Provenance sidecar written: {prov_file_1}")

    # Quick Verification Inference
    print("\n--- Running Verification Inference on Sample Test Frame ---")
    if test_img_files:
        val_sample = test_img_files[0]
        test_img = io.imread(val_sample)
        trained_model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(final_model_path))
        masks, flows, styles = trained_model.eval(test_img, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
        n_detected = len(np.unique(masks)) - (1 if 0 in masks else 0)
        print(f"Validation sample: {Path(val_sample).name}")
        print(f"Detected cell instances: {n_detected}")
        print("Verification Successful!")

    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
