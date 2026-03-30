#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class SeptumWindowDataset(Dataset):
    def __init__(
        self,
        working_dir: str,
        split: str,
        L_min: int = 16,
        L_max: int = 81,
        include_pos_prob: float = 0.7,
        seed: int = 0,
    ):
        self.working_dir = working_dir
        self.root = os.path.join(working_dir, "training_dataset")
        self.df = pd.read_csv(os.path.join(self.root, "manifest.csv"))

        # your deterministic split
        film = self.df["film_name"].astype(str)
        cell = pd.to_numeric(self.df["cell_id"], errors="coerce").fillna(-1).astype(int)
        key = (film + "__" + cell.astype(str)).apply(lambda s: abs(hash(s)) % 100)

        if split == "train":
            self.df = self.df[key < 85].reset_index(drop=True)
        elif split == "val":
            self.df = self.df[key >= 85].reset_index(drop=True)
        else:
            raise ValueError("split must be train/val")

        self.L_min = int(L_min)
        self.L_max = int(L_max)
        self.include_pos_prob = float(include_pos_prob)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _load_tiles(self, npz_fp: str):
        with np.load(npz_fp, allow_pickle=True) as z:
            strip = np.asarray(z["strip"], dtype=np.uint8)  # (H, H*L)
            H = int(strip.shape[0])
            L = int(strip.shape[1] // H)
            start_idx = int(z["start_idx"][0])
            end_idx   = int(z["end_idx"][0])

        tiles = strip.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]  # (L,1,H,H)
        x_full = tiles.astype(np.float32) / 255.0
        return x_full, start_idx, end_idx  # x_full: (L,1,H,H)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        npz_fp = os.path.join(self.working_dir, str(row["npz_path"]))
        x_full, s_idx, e_idx = self._load_tiles(npz_fp)
        Lfull = x_full.shape[0]

        # choose window length
        Lw = int(self.rng.integers(self.L_min, min(self.L_max, Lfull) + 1))

        # pick a crop start j
        def pick_j(center_idx: int):
            # choose j so that center_idx is inside [j, j+Lw)
            lo = max(0, center_idx - (Lw - 1))
            hi = min(center_idx, Lfull - Lw)
            if hi < lo:
                return int(self.rng.integers(0, Lfull - Lw + 1))
            return int(self.rng.integers(lo, hi + 1))

        # decide crop location (biased to include positives sometimes)
        has_any = (s_idx >= 0) or (e_idx >= 0)
        if has_any and (self.rng.random() < self.include_pos_prob):
            # if both exist, randomly choose which to target
            candidates = [idx for idx in [s_idx, e_idx] if idx >= 0]
            target = int(self.rng.choice(candidates))
            j = pick_j(target)
        else:
            j = int(self.rng.integers(0, Lfull - Lw + 1))

        x = x_full[j:j+Lw]  # (Lw,1,H,H)

        # binary labels: endpoint is inside this window?
        start_yes = 1.0 if (s_idx >= 0 and (j <= s_idx < j + Lw)) else 0.0
        end_yes   = 1.0 if (e_idx >= 0 and (j <= e_idx < j + Lw)) else 0.0

        return {
            "x": torch.from_numpy(x),                 # (Lw,1,H,H)
            "start_yes": torch.tensor(start_yes),
            "end_yes": torch.tensor(end_yes),
        }
def collate_pad(batch):
    B = len(batch)
    Ls = [b["x"].shape[0] for b in batch]
    Lmax = max(Ls)
    _, _, H, W = batch[0]["x"].shape

    x = torch.zeros((B, Lmax, 1, H, W), dtype=torch.float32)
    mask = torch.zeros((B, Lmax), dtype=torch.float32)
    y_s = torch.stack([b["start_yes"] for b in batch]).float()
    y_e = torch.stack([b["end_yes"]   for b in batch]).float()

    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        x[i, :L] = b["x"]
        mask[i, :L] = 1.0

    return {"x": x, "mask": mask, "y_start": y_s, "y_end": y_e}

import torch
from torch import nn

class EndpointMIL(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.enc = TileEncoder(D=D)
        self.temporal = nn.Sequential(
            nn.Conv1d(D, D, 3, padding=1), nn.ReLU(),
            nn.Conv1d(D, D, 3, padding=1), nn.ReLU(),
        )
        self.head_start = nn.Conv1d(D, 1, 1)
        self.head_end   = nn.Conv1d(D, 1, 1)

    def forward(self, x, mask):
        # x: (B,L,1,H,W) mask:(B,L)
        B, L, _, H, W = x.shape
        emb = self.enc(x.reshape(B*L, 1, H, W)).reshape(B, L, -1)  # (B,L,D)
        feat = self.temporal(emb.transpose(1, 2))                  # (B,D,L)

        s_t = self.head_start(feat)[:, 0, :]  # (B,L)
        e_t = self.head_end(feat)[:, 0, :]    # (B,L)

        neg_inf = torch.finfo(s_t.dtype).min
        s_t = s_t.masked_fill(mask == 0, neg_inf)
        e_t = e_t.masked_fill(mask == 0, neg_inf)

        # MIL pooling (max). You can swap to logsumexp later.
        s_win = s_t.max(dim=1).values  # (B,)
        e_win = e_t.max(dim=1).values  # (B,)

        return s_t, e_t, s_win, e_win
    
def compute_pos_weight(working_dir: str):
    df = pd.read_csv(os.path.join(working_dir, "training_dataset", "manifest.csv"))
    film = df["film_name"].astype(str)
    cell = pd.to_numeric(df["cell_id"], errors="coerce").fillna(-1).astype(int)
    key = (film + "__" + cell.astype(str)).apply(lambda s: abs(hash(s)) % 100)
    tr = df[key < 85].copy()

    s_pos = (pd.to_numeric(tr["start_idx"], errors="coerce").fillna(-1).astype(int) >= 0).sum()
    s_neg = len(tr) - s_pos
    e_pos = (pd.to_numeric(tr["end_idx"], errors="coerce").fillna(-1).astype(int) >= 0).sum()
    e_neg = len(tr) - e_pos

    # pos_weight = neg/pos (PyTorch expects this)
    s_w = float(s_neg / max(1, s_pos))
    e_w = float(e_neg / max(1, e_pos))
    return s_w, e_w, (s_pos, s_neg, e_pos, e_neg)

from torch.utils.data import DataLoader
import torch

def train_binary_mil(
    working_dir: str,
    D: int = 64,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    num_workers: int = 2,
    L_min: int = 16,
    L_max: int = 81,
    include_pos_prob: float = 0.7,
    device: str = None,
):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # data
    ds_tr = SeptumWindowDataset(working_dir, "train", L_min=L_min, L_max=L_max,
                               include_pos_prob=include_pos_prob, seed=0)
    ds_va = SeptumWindowDataset(working_dir, "val",   L_min=L_min, L_max=L_max,
                               include_pos_prob=0.0, seed=1)  # val: unbiased crops

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, collate_fn=collate_pad)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, collate_fn=collate_pad)

    # pos weights
    s_w, e_w, counts = compute_pos_weight(working_dir)
    print("[pos_weight] start:", s_w, "end:", e_w, "counts:", counts)

    bce_s = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([s_w], device=device))
    bce_e = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([e_w], device=device))

    model = EndpointMIL(D=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    @torch.no_grad()
    def eval_epoch():
        model.eval()
        loss_sum = 0.0
        n = 0
        acc_s = 0
        acc_e = 0
        for batch in dl_va:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y_s = batch["y_start"].to(device)
            y_e = batch["y_end"].to(device)

            _, _, s_win, e_win = model(x, mask)
            loss = bce_s(s_win, y_s) + bce_e(e_win, y_e)

            ps = (torch.sigmoid(s_win) > 0.5).float()
            pe = (torch.sigmoid(e_win) > 0.5).float()

            b = x.size(0)
            loss_sum += float(loss.item()) * b
            n += b
            acc_s += int((ps == y_s).sum().item())
            acc_e += int((pe == y_e).sum().item())

        return loss_sum / max(1, n), acc_s / max(1, n), acc_e / max(1, n)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in dl_tr:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y_s = batch["y_start"].to(device)
            y_e = batch["y_end"].to(device)

            _, _, s_win, e_win = model(x, mask)
            loss = bce_s(s_win, y_s) + bce_e(e_win, y_e)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            b = x.size(0)
            running += float(loss.item()) * b
            n += b

        tr_loss = running / max(1, n)
        va_loss, va_acc_s, va_acc_e = eval_epoch()
        print(f"[ep {ep:03d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_acc_start={va_acc_s:.3f} val_acc_end={va_acc_e:.3f}")

    return model
@torch.no_grad()
def sliding_scores(model, x_full, window=81, stride=1, device="cpu"):
    """
    x_full: torch float32 (L,1,H,W) in [0,1]
    returns:
      s_win_probs: (num_windows,)
      e_win_probs: (num_windows,)
    """
    model.eval()
    L = x_full.shape[0]
    outs_s, outs_e = [], []

    for j in range(0, L - window + 1, stride):
        x = x_full[j:j+window][None, ...].to(device)          # (1,window,1,H,W)
        mask = torch.ones((1, window), device=device)
        _, _, s_win, e_win = model(x, mask)
        outs_s.append(torch.sigmoid(s_win).item())
        outs_e.append(torch.sigmoid(e_win).item())

    return np.array(outs_s), np.array(outs_e)



# -------------------------
# Soft target helpers
# -------------------------
def gaussian_target(L_max: int, keep: int, idx: int, sigma: float, NONE: int):
    """
    Returns a (L_max+1,) float32 target distribution summing to 1.
    - If idx==NONE: one-hot on NONE
    - Else: Gaussian over [0..keep-1], zeros on padded positions, and 0 on NONE.
    """
    t = np.zeros((L_max + 1,), dtype=np.float32)

    if keep <= 0:
        t[NONE] = 1.0
        return t

    if idx == NONE:
        t[NONE] = 1.0
        return t

    xs = np.arange(keep, dtype=np.float32)
    g = np.exp(-0.5 * ((xs - float(idx)) / float(sigma)) ** 2)
    g = g / (g.sum() + 1e-8)

    t[:keep] = g
    t[NONE] = 0.0
    return t


def masked_kld_loss(
    ls: torch.Tensor,
    le: torch.Tensor,
    keep_np: np.ndarray,
    y_s_np: np.ndarray,
    y_e_np: np.ndarray,
    *,
    L_max: int,
    sigma: float,
    NONE: int,
    device: str,
    w_none_sample: float = 0.0,   # set >0 only when you *really* trust NONE labels
) -> torch.Tensor:
    """
    KLDiv on Gaussian soft targets, but:
      - if y == NONE => either downweight or IGNORE entirely (recommended)
      - avoids training collapse due to abundant missing endpoints
    """
    B = len(keep_np)

    logps = torch.log_softmax(ls, dim=1)
    logpe = torch.log_softmax(le, dim=1)

    kld_none = nn.KLDivLoss(reduction="none")

    # --- START ---
    mask_s = (y_s_np != NONE)
    if mask_s.any():
        ts = np.stack(
            [gaussian_target(L_max, int(keep_np[i]), int(y_s_np[i]), sigma, NONE) for i in range(B)],
            axis=0
        )
        ts = torch.from_numpy(ts).to(device)
        loss_s = kld_none(logps, ts).sum(dim=1)  # (B,)
        loss_s = loss_s[torch.from_numpy(mask_s).to(device)].mean()
    else:
        loss_s = torch.zeros((), device=device)

    # --- END ---
    mask_e = (y_e_np != NONE)
    if mask_e.any():
        te = np.stack(
            [gaussian_target(L_max, int(keep_np[i]), int(y_e_np[i]), sigma, NONE) for i in range(B)],
            axis=0
        )
        te = torch.from_numpy(te).to(device)
        loss_e = kld_none(logpe, te).sum(dim=1)  # (B,)
        loss_e = loss_e[torch.from_numpy(mask_e).to(device)].mean()
    else:
        loss_e = torch.zeros((), device=device)

    # Optional: if you REALLY want to train NONE as a true class:
    # keep a *small* penalty for y==NONE samples
    if w_none_sample > 0:
        # encourage NONE only weakly
        y_s_t = torch.from_numpy(y_s_np).to(device)
        y_e_t = torch.from_numpy(y_e_np).to(device)

        ts_none = torch.zeros((B, L_max + 1), device=device)
        te_none = torch.zeros((B, L_max + 1), device=device)
        ts_none[:, NONE] = 1.0
        te_none[:, NONE] = 1.0

        loss_s_none = kld_none(logps, ts_none).sum(dim=1)
        loss_e_none = kld_none(logpe, te_none).sum(dim=1)

        m_s_none = (y_s_t == NONE)
        m_e_none = (y_e_t == NONE)

        if m_s_none.any():
            loss_s = loss_s + w_none_sample * loss_s_none[m_s_none].mean()
        if m_e_none.any():
            loss_e = loss_e + w_none_sample * loss_e_none[m_e_none].mean()

    return loss_s + loss_e

def masked_kld_loss_one(
    logits: torch.Tensor,
    keep_np: np.ndarray,
    y_np: np.ndarray,
    *,
    L_max: int,
    sigma: float,
    NONE: int,
    device: str,
) -> torch.Tensor:
    """
    KL loss for ONE endpoint stream (start or end), ignoring NONE labels.
    Assumes the dataset guarantees y != NONE (pos_start/pos_end).
    """
    B = len(keep_np)
    t = np.stack(
        [gaussian_target(L_max, int(keep_np[i]), int(y_np[i]), sigma, NONE) for i in range(B)],
        axis=0
    )
    t = torch.from_numpy(t).to(device)

    logp = torch.log_softmax(logits, dim=1)
    kld_none = nn.KLDivLoss(reduction="none")
    loss = kld_none(logp, t).sum(dim=1).mean()
    return loss
# -------------------------
# Dataset
# -------------------------
class SeptumDataset(Dataset):
    def __init__(self, working_dir: str, split: str, L_max: int = 81, mode: str = "all"):
        """
        mode:
          - "all": use all samples
          - "pos_any": only samples with at least one endpoint (start_idx!=-1 OR end_idx!=-1)
          - "pos_both": only samples with both endpoints
        """
        self.working_dir = working_dir
        self.root = os.path.join(working_dir, "training_dataset")
        self.manifest_fp = os.path.join(self.root, "manifest.csv")
        if not os.path.isfile(self.manifest_fp):
            raise FileNotFoundError(f"manifest.csv not found: {self.manifest_fp}")

        df = pd.read_csv(self.manifest_fp)
        if df.empty:
            raise ValueError(f"manifest.csv is empty: {self.manifest_fp}")

        film = df["film_name"].astype(str)
        cell = pd.to_numeric(df["cell_id"], errors="coerce").fillna(-1).astype(int)
        key = (film + "__" + cell.astype(str)).apply(lambda s: abs(hash(s)) % 100)

        if split == "train":
            self.df = df[key < 85].reset_index(drop=True)
        elif split in ("val", "valid", "validation"):
            self.df = df[key >= 85].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        # optional filter (only after split)
        if mode != "all":
            if "start_idx" not in self.df.columns or "end_idx" not in self.df.columns:
                raise ValueError("manifest.csv must contain start_idx and end_idx for mode filtering")

            s = pd.to_numeric(self.df["start_idx"], errors="coerce").fillna(-1).astype(int).values
            e = pd.to_numeric(self.df["end_idx"], errors="coerce").fillna(-1).astype(int).values

            if mode == "pos_any":
                self.df = self.df[(s >= 0) | (e >= 0)].reset_index(drop=True)
            elif mode == "pos_both":
                self.df = self.df[(s >= 0) & (e >= 0)].reset_index(drop=True)
            elif mode == "pos_start":
                self.df = self.df[(s >= 0)].reset_index(drop=True)
            elif mode == "pos_end":
                self.df = self.df[(e >= 0)].reset_index(drop=True)
            else:
                raise ValueError("mode must be 'all','pos_any','pos_both','pos_start','pos_end'")

        self.L_max = int(L_max)
        self.NONE = int(L_max)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        npz_rel = str(row["npz_path"])
        npz_fp = os.path.join(self.working_dir, npz_rel)
        if not os.path.isfile(npz_fp):
            raise FileNotFoundError(f"NPZ not found for row {i}: {npz_fp}")

        with np.load(npz_fp, allow_pickle=True) as z:
            strip = np.asarray(z["strip"], dtype=np.uint8)   # (H, H*L)
            H = int(strip.shape[0])
            if strip.ndim != 2 or H <= 0 or (strip.shape[1] % H) != 0:
                raise ValueError(f"Bad strip shape in {npz_fp}: {strip.shape}")
            L = int(strip.shape[1] // H)

            start_idx = int(z["start_idx"][0])
            end_idx   = int(z["end_idx"][0])

        tiles = strip.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]  # (L,1,H,H)

        Lm = self.L_max
        x = np.zeros((Lm, 1, H, H), dtype=np.float32)
        keep = min(L, Lm)
        x[:keep] = tiles[:keep].astype(np.float32) / 255.0

        mask = np.zeros((Lm,), dtype=np.float32)
        mask[:keep] = 1.0

        def map_label(idx: int) -> int:
            if idx < 0:
                return self.NONE
            return idx if idx < keep else self.NONE

        y_start = map_label(start_idx)
        y_end   = map_label(end_idx)

        return {
            "x": torch.from_numpy(x),                 # (Lm,1,H,H)
            "mask": torch.from_numpy(mask),           # (Lm,)
            "y_start": torch.tensor(y_start, dtype=torch.long),
            "y_end": torch.tensor(y_end, dtype=torch.long),
            "keep": torch.tensor(keep, dtype=torch.long),
        }


# -------------------------
# Model
# -------------------------
class TileEncoder(nn.Module):
    def __init__(self, D: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, D, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x)[:, :, 0, 0]  # (B*L,D)


class SeptumModel(nn.Module):
    """
    logits_start/end: (B, L_max+1), last index is NONE

    Key fix:
      - NONE logit comes from a *learned* "none head" over temporal features,
        not from copying max() and not from a free scalar bias.
    """
    def __init__(self, D: int = 64, L_max: int = 81):
        super().__init__()
        self.L_max = int(L_max)
        self.NONE = int(L_max)

        self.enc = TileEncoder(D=D)
        self.temporal = nn.Sequential(
            nn.Conv1d(D, D, 3, padding=1), nn.ReLU(),
            nn.Conv1d(D, D, 3, padding=1), nn.ReLU(),
        )
        self.head_start = nn.Conv1d(D, 1, 1)  # -> (B,1,L)
        self.head_end   = nn.Conv1d(D, 1, 1)  # -> (B,1,L)

        # NONE head: maps temporal features -> per-time none score, pooled to scalar
        self.none_start_head = nn.Conv1d(D, 1, 1)
        self.none_end_head   = nn.Conv1d(D, 1, 1)

    def forward(self, x, mask):
        B, L, _, H, W = x.shape
        x2 = x.reshape(B * L, 1, H, W)
        emb = self.enc(x2).reshape(B, L, -1)          # (B,L,D)
        feat = self.temporal(emb.transpose(1, 2))     # (B,D,L)

        ls = self.head_start(feat)[:, 0, :]           # (B,L)
        le = self.head_end(feat)[:, 0, :]             # (B,L)

        neg_inf = torch.finfo(ls.dtype).min
        ls = ls.masked_fill(mask == 0, neg_inf)
        le = le.masked_fill(mask == 0, neg_inf)

        # NONE logits from learned head + masked mean pooling
        ns_t = self.none_start_head(feat)[:, 0, :]    # (B,L)
        ne_t = self.none_end_head(feat)[:, 0, :]      # (B,L)
        ns_t = ns_t.masked_fill(mask == 0, 0.0)
        ne_t = ne_t.masked_fill(mask == 0, 0.0)
        denom = mask.sum(dim=1).clamp(min=1.0)
        none_s = (ns_t.sum(dim=1, keepdim=True) / denom[:, None])  # (B,1)
        none_e = (ne_t.sum(dim=1, keepdim=True) / denom[:, None])  # (B,1)

        logits_start = torch.cat([ls, none_s], dim=1)
        logits_end   = torch.cat([le, none_e], dim=1)
        return logits_start, logits_end


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_epoch(model, dl, device, NONE: int, tol: int = 10):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    loss_sum = 0.0

    acc_s = 0
    acc_e = 0
    has_match = 0

    n_s_pos = 0
    n_e_pos = 0
    acc_s_pos = 0
    acc_e_pos = 0
    acc_s_pos_tol = 0
    acc_e_pos_tol = 0

    pred_s_none = 0
    pred_e_none = 0
    true_s_none = 0
    true_e_none = 0

    for batch in dl:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_s = batch["y_start"].to(device)
        y_e = batch["y_end"].to(device)

        ls, le = model(x, mask)
        loss = ce(ls, y_s) + ce(le, y_e)

        ps = ls.argmax(dim=1)
        pe = le.argmax(dim=1)

        b = x.size(0)
        total += b
        loss_sum += float(loss.item()) * b

        acc_s += int((ps == y_s).sum().item())
        acc_e += int((pe == y_e).sum().item())

        has_true = ((y_s != NONE) | (y_e != NONE))
        has_pred = ((ps != NONE) | (pe != NONE))
        has_match += int((has_true == has_pred).sum().item())

        s_pos = (y_s != NONE)
        e_pos = (y_e != NONE)

        n_s_pos += int(s_pos.sum().item())
        n_e_pos += int(e_pos.sum().item())

        if s_pos.any():
            acc_s_pos += int((ps[s_pos] == y_s[s_pos]).sum().item())
            acc_s_pos_tol += int((torch.abs(ps[s_pos] - y_s[s_pos]) <= tol).sum().item())

        if e_pos.any():
            acc_e_pos += int((pe[e_pos] == y_e[e_pos]).sum().item())
            acc_e_pos_tol += int((torch.abs(pe[e_pos] - y_e[e_pos]) <= tol).sum().item())

        pred_s_none += int((ps == NONE).sum().item())
        pred_e_none += int((pe == NONE).sum().item())
        true_s_none += int((y_s == NONE).sum().item())
        true_e_none += int((y_e == NONE).sum().item())

    return {
        "loss": loss_sum / max(1, total),
        "acc_start": acc_s / max(1, total),
        "acc_end": acc_e / max(1, total),
        "acc_has_derived": has_match / max(1, total),

        "acc_start_non_none": (acc_s_pos / max(1, n_s_pos)),
        "acc_end_non_none": (acc_e_pos / max(1, n_e_pos)),
        "acc_start_non_none_tol": (acc_s_pos_tol / max(1, n_s_pos)),
        "acc_end_non_none_tol": (acc_e_pos_tol / max(1, n_e_pos)),
        "n_start_non_none": n_s_pos,
        "n_end_non_none": n_e_pos,

        "pred_start_none_rate": pred_s_none / max(1, total),
        "pred_end_none_rate": pred_e_none / max(1, total),
        "true_start_none_rate": true_s_none / max(1, total),
        "true_end_none_rate": true_e_none / max(1, total),
    }


# -------------------------
# Train
# -------------------------
def train(
    working_dir: str,
    L_max: int = 81,
    D: int = 64,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    num_workers: int = 2,
    device: str = None,
    warmup_epochs: int = 10,
    sigma: float = 6.0,
    tol: int = 10,
):
    # -------------------------
    # Device
    # -------------------------
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    NONE = int(L_max)

    # -------------------------
    # Datasets / Loaders
    # -------------------------
    # Warmup (guaranteed labels)
    ds_tr_start = SeptumDataset(working_dir, "train", L_max=L_max, mode="pos_start")
    ds_tr_end   = SeptumDataset(working_dir, "train", L_max=L_max, mode="pos_end")
    ds_va_start = SeptumDataset(working_dir, "val",   L_max=L_max, mode="pos_start")
    ds_va_end   = SeptumDataset(working_dir, "val",   L_max=L_max, mode="pos_end")

    dl_tr_start = DataLoader(ds_tr_start, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_tr_end   = DataLoader(ds_tr_end,   batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va_start = DataLoader(ds_va_start, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_va_end   = DataLoader(ds_va_end,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if len(ds_tr_start) == 0 and len(ds_tr_end) == 0:
        raise ValueError("No positive start/end samples found. Check manifest start_idx/end_idx export.")
    if len(ds_tr_start) == 0:
        print("[WARN] pos_start is empty; warmup will only train end head.")
    if len(ds_tr_end) == 0:
        print("[WARN] pos_end is empty; warmup will only train start head.")

    # Main phase (all + balanced sampler)
    ds_tr_all = SeptumDataset(working_dir, "train", L_max=L_max, mode="all")
    ds_va_all = SeptumDataset(working_dir, "val",   L_max=L_max, mode="all")

    dl_va_all = DataLoader(ds_va_all, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Balanced sampler on train/all by pos_any from manifest columns
    if "start_idx" in ds_tr_all.df.columns and "end_idx" in ds_tr_all.df.columns:
        s = pd.to_numeric(ds_tr_all.df["start_idx"], errors="coerce").fillna(-1).astype(int).values
        e = pd.to_numeric(ds_tr_all.df["end_idx"], errors="coerce").fillna(-1).astype(int).values
        is_pos_any = (s >= 0) | (e >= 0)
        n_pos = int(is_pos_any.sum())
        n_neg = int((~is_pos_any).sum())

        if n_pos > 0 and n_neg > 0:
            w_pos = 0.5 / n_pos
            w_neg = 0.5 / n_neg
            weights = np.where(is_pos_any, w_pos, w_neg).astype(np.float32)
        else:
            weights = np.ones(len(ds_tr_all), dtype=np.float32)
    else:
        print("[WARN] manifest missing start_idx/end_idx columns; using uniform sampling for train/all.")
        weights = np.ones(len(ds_tr_all), dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(ds_tr_all),
        replacement=True,
    )

    dl_tr_all_bal = DataLoader(
        ds_tr_all,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # -------------------------
    # Model / Optim
    # -------------------------
    model = SeptumModel(D=D, L_max=L_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ckpt_dir = os.path.join(working_dir, "training_dataset", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def _train_step_one_head(batch, which: str):
        """
        which: "start" or "end"
        """
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        ls, le = model(x, mask)

        keep = batch["keep"].cpu().numpy()
        if which == "start":
            y = batch["y_start"].cpu().numpy()
            loss = masked_kld_loss_one(ls, keep, y, L_max=L_max, sigma=sigma, NONE=NONE, device=device)
        else:
            y = batch["y_end"].cpu().numpy()
            loss = masked_kld_loss_one(le, keep, y, L_max=L_max, sigma=sigma, NONE=NONE, device=device)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.item()), x.size(0)

    def _train_step_both_heads(batch):
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        ls, le = model(x, mask)

        keep = batch["keep"].cpu().numpy()
        y_s = batch["y_start"].cpu().numpy()
        y_e = batch["y_end"].cpu().numpy()

        # ignore NONE via masked_kld_loss as you implemented
        loss = masked_kld_loss(
            ls, le, keep, y_s, y_e,
            L_max=L_max, sigma=sigma, NONE=NONE, device=device,
            w_none_sample=0.0,  # keep 0 until you trust NONE labels
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.item()), x.size(0)

    # -------------------------
    # Epoch loop
    # -------------------------
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        if ep <= warmup_epochs:
            phase = "WARM"

            it_s = iter(dl_tr_start) if len(dl_tr_start) else None
            it_e = iter(dl_tr_end)   if len(dl_tr_end) else None

            steps = 0
            if len(dl_tr_start):
                steps = max(steps, len(dl_tr_start))
            if len(dl_tr_end):
                steps = max(steps, len(dl_tr_end))
            steps = max(1, steps)

            for _ in range(steps):
                if it_s is not None:
                    try:
                        batch = next(it_s)
                    except StopIteration:
                        it_s = iter(dl_tr_start)
                        batch = next(it_s)
                    loss_val, bsz = _train_step_one_head(batch, "start")
                    running += loss_val * bsz
                    n += bsz

                if it_e is not None:
                    try:
                        batch = next(it_e)
                    except StopIteration:
                        it_e = iter(dl_tr_end)
                        batch = next(it_e)
                    loss_val, bsz = _train_step_one_head(batch, "end")
                    running += loss_val * bsz
                    n += bsz

            tr_loss = running / max(1, n)

            va_s = eval_epoch(model, dl_va_start, device, NONE, tol=tol) if len(ds_va_start) else None
            va_e = eval_epoch(model, dl_va_end,   device, NONE, tol=tol) if len(ds_va_end) else None

            print(
                f"[{phase} ep {ep:03d}] train_loss={tr_loss:.4f} "
                f"val_start acc*={(va_s['acc_start_non_none'] if va_s else float('nan')):.3f} "
                f"val_end   acc*={(va_e['acc_end_non_none']   if va_e else float('nan')):.3f}"
            )

        else:
            phase = "ALL"

            for batch in dl_tr_all_bal:
                loss_val, bsz = _train_step_both_heads(batch)
                running += loss_val * bsz
                n += bsz

            tr_loss = running / max(1, n)
            va = eval_epoch(model, dl_va_all, device, NONE, tol=tol)

            print(
                f"[{phase} ep {ep:03d}] train_loss={tr_loss:.4f} val_loss={va['loss']:.4f} "
                f"acc_s={va['acc_start']:.3f} acc_e={va['acc_end']:.3f} "
                f"acc_s*={va['acc_start_non_none']:.3f}({va['n_start_non_none']}) "
                f"acc_e*={va['acc_end_non_none']:.3f}({va['n_end_non_none']}) "
                f"acc_s±{tol}={va['acc_start_non_none_tol']:.3f} acc_e±{tol}={va['acc_end_non_none_tol']:.3f} "
                f"none_pred_s={va['pred_start_none_rate']:.3f} none_true_s={va['true_start_none_rate']:.3f} "
                f"none_pred_e={va['pred_end_none_rate']:.3f} none_true_e={va['true_end_none_rate']:.3f} "
                f"acc_has={va['acc_has_derived']:.3f}"
            )

        # checkpoint every epoch (simple + safe)
        ckpt_path = os.path.join(ckpt_dir, f"model_ep{ep:03d}.pt")
        torch.save({"state_dict": model.state_dict(), "L_max": L_max, "D": D}, ckpt_path)

    torch.save({"state_dict": model.state_dict(), "L_max": L_max, "D": D},
               os.path.join(ckpt_dir, "model_latest.pt"))

    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("working_dir")
    ap.add_argument("--L_max", type=int, default=81)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    train(
        working_dir=args.working_dir,
        L_max=args.L_max,
        D=args.D,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
    )