#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:34:50 2026

@author: user
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from data import WindowEndpointDataset
from model import EndpointMILModel

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total = 0
    loss_sum = 0.0
    acc_s = acc_e = 0
    pos_s = pos_e = 0

    for batch in dl:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_s = batch["y_start"].to(device)
        y_e = batch["y_end"].to(device)

        _, _, win_s, win_e = model(x, mask, pool="logsumexp", temperature=1.0)
        loss = bce(win_s, y_s) + bce(win_e, y_e)

        p_s = (torch.sigmoid(win_s) > 0.5).float()
        p_e = (torch.sigmoid(win_e) > 0.5).float()

        b = x.size(0)
        total += b
        loss_sum += float(loss.item())
        acc_s += int((p_s == y_s).sum().item())
        acc_e += int((p_e == y_e).sum().item())
        pos_s += int(y_s.sum().item())
        pos_e += int(y_e.sum().item())

    return {
        "loss": loss_sum / max(1, total),
        "acc_start": acc_s / max(1, total),
        "acc_end": acc_e / max(1, total),
        "pos_rate_start": pos_s / max(1, total),
        "pos_rate_end": pos_e / max(1, total),
    }

def train(
    working_dir: str,
    L_max: int = 81,
    min_len: int = 16,
    D: int = 64,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    num_workers: int = 2,
    device: str = None,
):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    ds_tr = WindowEndpointDataset(working_dir, "train", L_max=L_max, min_len=min_len, invert_p=0.5, rot90_p=0.0)
    ds_va = WindowEndpointDataset(working_dir, "val",   L_max=L_max, min_len=min_len, invert_p=0.0, rot90_p=0.0)

    # --- train sampler: oversample strips that have ANY endpoint (start or end) ---
    s = pd.to_numeric(ds_tr.df["start_idx"], errors="coerce").fillna(-1).astype(int).values
    e = pd.to_numeric(ds_tr.df["end_idx"], errors="coerce").fillna(-1).astype(int).values
    is_pos = (s >= 0) | (e >= 0)
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())

    if n_pos > 0 and n_neg > 0:
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        weights = np.where(is_pos, w_pos, w_neg).astype(np.float32)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(ds_tr), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        pos_weight = max(1.0, n_neg / max(1, n_pos))
    else:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        pos_weight = 1.0

    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = EndpointMILModel(D=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # BCE with pos_weight helps if windows are often negative
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    ckpt_dir = os.path.join(working_dir, "training_dataset", "checkpoints_binary")
    os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in dl_tr:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y_s = batch["y_start"].to(device)
            y_e = batch["y_end"].to(device)

            _, _, win_s, win_e = model(x, mask, pool="logsumexp", temperature=1.0)
            loss = bce(win_s, y_s) + bce(win_e, y_e)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        tr_loss = running / max(1, n)
        va = eval_epoch(model, dl_va, device)

        print(
            f"[ep {ep:03d}] train_loss={tr_loss:.4f} "
            f"val_loss={va['loss']:.4f} "
            f"acc_s={va['acc_start']:.3f} acc_e={va['acc_end']:.3f} "
            f"val_pos_rate_s={va['pos_rate_start']:.3f} val_pos_rate_e={va['pos_rate_end']:.3f}"
        )

        torch.save({"state_dict": model.state_dict(), "D": D, "L_max": L_max},
                   os.path.join(ckpt_dir, f"model_ep{ep:03d}.pt"))

    torch.save({"state_dict": model.state_dict(), "D": D, "L_max": L_max},
               os.path.join(ckpt_dir, "model_latest.pt"))
    return model