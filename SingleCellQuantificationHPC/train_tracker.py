#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_tracker.py
----------------
Training script for CellTrackerModel.

Usage (from the SingleCellQuantificationHPC directory):
    python train_tracker.py \
        --movie_root "/Volumes/X10 Pro/Movies" \
        --hold_out_film "A14-YES-1t-FBFBF-4_F2" \
        --out_dir ./tracker_checkpoints \
        --epochs 60 \
        --batch_size 16 \
        --lr 3e-4

After training, run evaluation on the held-out film:
    python train_tracker.py --eval_only \
        --movie_root "/Volumes/X10 Pro/Movies" \
        --hold_out_film "A14-YES-1t-FBFBF-4_F2" \
        --ckpt ./tracker_checkpoints/model_best.pt
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracker_model   import CellTrackerModel, load_tracker
from tracker_dataset import TrackerDataset


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

def tracker_loss(
    scores:    torch.Tensor,   # (B, K) raw logits
    div_logit: torch.Tensor,   # (B, 1)
    mrg_logit: torch.Tensor,   # (B, 1)
    gt_idx:    torch.Tensor,   # (B,)  always 0
    div_label: torch.Tensor,   # (B, 1)
    mrg_label: torch.Tensor,   # (B, 1)
    w_div: float = 1.0,
    w_mrg: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Score head: cross-entropy over the K candidates (gt always at index 0).
    Division & merge heads: binary cross-entropy.
    """
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    L_score = ce(scores, gt_idx)
    L_div   = bce(div_logit, div_label) * w_div
    L_mrg   = bce(mrg_logit, mrg_label) * w_mrg
    total   = L_score + L_div + L_mrg

    return total, {
        "loss_score": L_score.item(),
        "loss_div":   L_div.item(),
        "loss_mrg":   L_mrg.item(),
        "loss_total": total.item(),
    }


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(scores, div_logit, mrg_logit,
                    gt_idx, div_label, mrg_label) -> dict:
    """Compute accuracy metrics (detached, on CPU)."""
    with torch.no_grad():
        # Score top-1 accuracy
        pred_idx    = scores.argmax(dim=1)
        score_acc   = (pred_idx == gt_idx).float().mean().item()

        # Division accuracy
        div_pred    = (torch.sigmoid(div_logit) > 0.5).float()
        div_acc     = (div_pred == div_label).float().mean().item()

        # Merge accuracy
        mrg_pred    = (torch.sigmoid(mrg_logit) > 0.5).float()
        mrg_acc     = (mrg_pred == mrg_label).float().mean().item()

    return {
        "score_acc": score_acc,
        "div_acc":   div_acc,
        "mrg_acc":   mrg_acc,
    }


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(args):
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"[train] device={device}")

    # Dataset
    print("[train] Building dataset …")
    ds = TrackerDataset(
        movie_root    = args.movie_root,
        hold_out_film = args.hold_out_film,
        augment       = True,
        topk_neg      = 4,
    )

    # Train / val split (90 / 10)
    n_val   = max(1, int(0.10 * len(ds)))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  train={n_train}  val={n_val}")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = CellTrackerModel().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history       = []
    start_epoch   = 1

    ckpt_path = out_dir / "model_latest.pt"
    if ckpt_path.exists():
        print(f"[train] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1

        hist_path = out_dir / "train_history.json"
        if hist_path.exists():
            with open(hist_path, "r") as f:
                history = json.load(f)
            for row in history:
                if row.get("val_loss_total", float("inf")) < best_val_loss:
                    best_val_loss = row["val_loss_total"]

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────
        model.train()
        train_metrics = {k: 0.0 for k in
                         ["loss_total", "loss_score", "loss_div", "loss_mrg",
                          "score_acc", "div_acc", "mrg_acc"]}
        n_batches = 0

        for batch in dl_train:
            img3ch      = batch["img3ch"].to(device)
            candidates  = batch["candidates"].to(device)
            sep_t       = batch["sep_t"].to(device)
            sep_t1      = batch["sep_t1"].to(device)
            area_ratio  = batch["area_ratio"].to(device)
            adjacency   = batch["adjacency"].to(device)
            div_label   = batch["div_label"].to(device)
            mrg_label   = batch["mrg_label"].to(device)
            gt_idx      = batch["gt_idx"].to(device)

            opt.zero_grad()
            scores, div_logit, mrg_logit = model(
                img3ch, candidates, sep_t, area_ratio, sep_t1, adjacency)

            loss, loss_d = tracker_loss(
                scores, div_logit, mrg_logit, gt_idx, div_label, mrg_label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            met = compute_metrics(scores, div_logit, mrg_logit,
                                   gt_idx, div_label, mrg_label)
            for k, v in {**loss_d, **met}.items():
                train_metrics[k] += v
            n_batches += 1

        for k in train_metrics:
            train_metrics[k] /= max(n_batches, 1)

        # ── validation ─────────────────────────────────────────────
        model.eval()
        val_metrics = {k: 0.0 for k in train_metrics}
        n_v = 0

        with torch.no_grad():
            for batch in dl_val:
                img3ch     = batch["img3ch"].to(device)
                candidates = batch["candidates"].to(device)
                sep_t      = batch["sep_t"].to(device)
                sep_t1     = batch["sep_t1"].to(device)
                area_ratio = batch["area_ratio"].to(device)
                adjacency  = batch["adjacency"].to(device)
                div_label  = batch["div_label"].to(device)
                mrg_label  = batch["mrg_label"].to(device)
                gt_idx     = batch["gt_idx"].to(device)

                scores, div_logit, mrg_logit = model(
                    img3ch, candidates, sep_t, area_ratio, sep_t1, adjacency)
                _, loss_d = tracker_loss(
                    scores, div_logit, mrg_logit, gt_idx, div_label, mrg_label)
                met = compute_metrics(scores, div_logit, mrg_logit,
                                       gt_idx, div_label, mrg_label)
                for k, v in {**loss_d, **met}.items():
                    val_metrics[k] += v
                n_v += 1

        for k in val_metrics:
            val_metrics[k] /= max(n_v, 1)

        sched.step()
        elapsed = time.time() - t0

        # ── log ───────────────────────────────────────────────────
        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}":   v for k, v in val_metrics.items()},
               "lr": sched.get_last_lr()[0],
               "time_s": elapsed}
        history.append(row)

        print(
            f"[ep {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"score_acc={train_metrics['score_acc']:.3f} | "
            f"val_loss={val_metrics['loss_total']:.4f} "
            f"score_acc={val_metrics['score_acc']:.3f} "
            f"div_acc={val_metrics['div_acc']:.3f} "
            f"mrg_acc={val_metrics['mrg_acc']:.3f} | "
            f"{elapsed:.0f}s"
        )

        # ── save checkpoints ──────────────────────────────────────
        state = {"state_dict": model.state_dict(), "epoch": epoch}
        torch.save(state, out_dir / "model_latest.pt")

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            torch.save(state, out_dir / "model_best.pt")
            print(f"  → New best val loss: {best_val_loss:.4f}")

    # Save history
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Done. Best val loss: {best_val_loss:.4f}")
    print(f"[train] Checkpoints saved to: {out_dir}")


# ─────────────────────────────────────────────
# Evaluation on held-out film
# ─────────────────────────────────────────────

def evaluate(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[eval] device={device}  ckpt={args.ckpt}")

    model = load_tracker(args.ckpt, device=device)
    model.eval()

    ds = TrackerDataset(
        movie_root    = args.movie_root,
        hold_out_film = None,          # load everything then filter
        augment       = False,
        topk_neg      = 4,
    )

    # Keep only samples from the held-out film
    held = [s for s in ds.samples if s["film"] == args.hold_out_film]
    if not held:
        print(f"[eval] No samples found for {args.hold_out_film}")
        return

    # Temporarily replace the dataset samples
    ds.samples = held
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    score_accs, div_accs, mrg_accs = [], [], []

    with torch.no_grad():
        for batch in dl:
            img3ch     = batch["img3ch"].to(device)
            candidates = batch["candidates"].to(device)
            sep_t      = batch["sep_t"].to(device)
            sep_t1     = batch["sep_t1"].to(device)
            area_ratio = batch["area_ratio"].to(device)
            adjacency  = batch["adjacency"].to(device)
            div_label  = batch["div_label"].to(device)
            mrg_label  = batch["mrg_label"].to(device)
            gt_idx     = batch["gt_idx"].to(device)

            scores, div_logit, mrg_logit = model(
                img3ch, candidates, sep_t, area_ratio, sep_t1, adjacency)
            met = compute_metrics(scores, div_logit, mrg_logit,
                                   gt_idx, div_label, mrg_label)
            score_accs.append(met["score_acc"])
            div_accs.append(met["div_acc"])
            mrg_accs.append(met["mrg_acc"])

    print(f"\n[eval] Held-out film: {args.hold_out_film}")
    print(f"  Score accuracy (top-1): {np.mean(score_accs):.3f}")
    print(f"  Division accuracy:      {np.mean(div_accs):.3f}")
    print(f"  Merge accuracy:         {np.mean(mrg_accs):.3f}")
    print(f"  N samples:              {len(held)}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train/evaluate CellTrackerModel")
    parser.add_argument("--movie_root",    default="/Volumes/X10 Pro/Movies")
    parser.add_argument("--hold_out_film", default="A14-YES-1t-FBFBF-4_F2",
                        help="Film to hold out for evaluation")
    parser.add_argument("--out_dir",       default="./tracker_checkpoints")
    parser.add_argument("--ckpt",          default=None,
                        help="Checkpoint path for --eval_only")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--eval_only",  action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        if args.ckpt is None:
            args.ckpt = str(Path(args.out_dir) / "model_best.pt")
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
