#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    film = df["film_name"].astype(str)
    cell = pd.to_numeric(df["cell_id"], errors="coerce").fillna(-1).astype(int)
    key = (film + "__" + cell.astype(str)).apply(lambda s: abs(hash(s)) % 100)

    if split == "train":
        return df[key < 85].reset_index(drop=True)
    if split in ("val", "valid", "validation"):
        return df[key >= 85].reset_index(drop=True)
    raise ValueError("split must be train/val")


class WindowEndpointDataset(Dataset):
    """
    Randomly samples a sub-window from a strip each time __getitem__ is called.

    Returns:
      x:    (L_max, 1, H, H) float32
      mask: (L_max,) float32
      y_start: float32 {0,1}
      y_end:   float32 {0,1}
      win_len: int64
      offset:  int64
    """

    def __init__(
        self,
        working_dir: str,
        split: str,
        L_max: int = 81,
        min_len: int = 16,
        endpoint: str = "both",     # "start" | "end" | "both"
        # --- sampling control ---
        focus_p: float = 0.75,      # train: chance to force window to include a labeled endpoint
        focus_jitter: float = 0.30, # fraction of win_len used as jitter (0..0.5 recommended)
        # --- augment ---
        augment: bool = True,
        invert_p: float = 0.5,
        rot90_p: float = 0.0,
    ):
        self.working_dir = working_dir
        self.root = os.path.join(working_dir, "training_dataset")
        self.manifest_fp = os.path.join(self.root, "manifest.csv")
        if not os.path.isfile(self.manifest_fp):
            raise FileNotFoundError(f"manifest.csv not found: {self.manifest_fp}")

        df = pd.read_csv(self.manifest_fp, sep=None, engine="python")
        self.df = _split_df(df, split)

        self.split = split
        self.L_max = int(L_max)
        self.min_len = int(min_len)

        endpoint = endpoint.lower()
        if endpoint not in ("start", "end", "both"):
            raise ValueError("endpoint must be 'start', 'end', or 'both'")
        self.endpoint = endpoint

        self.focus_p = float(focus_p)
        self.focus_jitter = float(focus_jitter)

        self.augment = bool(augment) and (split == "train")
        self.invert_p = float(invert_p)
        self.rot90_p = float(rot90_p)

    def __len__(self):
        return len(self.df)

    def _choose_win_len(self, L: int) -> int:
        max_len = min(L, self.L_max)
        if max_len <= 0:
            return 0
        if max_len < self.min_len:
            return max_len
        return int(np.random.randint(self.min_len, max_len + 1))

    def _choose_offset_uniform(self, L: int, win_len: int) -> int:
        if L <= win_len:
            return 0
        return int(np.random.randint(0, L - win_len + 1))

    def _choose_offset_focused(self, L: int, win_len: int, target_idx: int) -> int:
        """
        Choose an offset so that target_idx falls inside [offset, offset+win_len),
        but with jitter so target isn't always at the same position.
        """
        if target_idx < 0:
            return self._choose_offset_uniform(L, win_len)
        if L <= win_len:
            return 0

        # pick where inside the window the target should land
        # center-ish with jitter
        center = win_len // 2
        jitter = int(round(self.focus_jitter * win_len))
        jitter = max(0, min(jitter, win_len - 1))

        desired_pos = center
        if jitter > 0:
            desired_pos = center + int(np.random.randint(-jitter, jitter + 1))
        desired_pos = max(0, min(win_len - 1, desired_pos))

        offset = target_idx - desired_pos
        offset = max(0, min(offset, L - win_len))
        return int(offset)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        npz_fp = os.path.join(self.working_dir, str(row["npz_path"]))
        if not os.path.isfile(npz_fp):
            raise FileNotFoundError(f"NPZ not found: {npz_fp}")

        with np.load(npz_fp, allow_pickle=True) as z:
            strip = np.asarray(z["strip"], dtype=np.uint8)  # (H, H*L)
            H = int(strip.shape[0])
            if strip.ndim != 2 or H <= 0 or (strip.shape[1] % H) != 0:
                raise ValueError(f"Bad strip shape in {npz_fp}: {strip.shape}")
            L = int(strip.shape[1] // H)
            start_idx = int(z["start_idx"][0])
            end_idx   = int(z["end_idx"][0])

        tiles = strip.reshape(H, L, H).transpose(1, 0, 2)  # (L, H, H)

        win_len = self._choose_win_len(L)
        if win_len <= 0:
            # degenerate: return all zeros
            x = torch.zeros((self.L_max, 1, H, H), dtype=torch.float32)
            mask = torch.zeros((self.L_max,), dtype=torch.float32)
            return {
                "x": x, "mask": mask,
                "y_start": torch.tensor(0.0),
                "y_end": torch.tensor(0.0),
                "win_len": torch.tensor(0, dtype=torch.long),
                "offset": torch.tensor(0, dtype=torch.long),
            }

        # --- choose offset ---
        use_focus = (self.split == "train") and (np.random.rand() < self.focus_p)

        # pick which endpoint to focus on (depending on mode and availability)
        target_idx = -1
        if use_focus:
            candidates = []
            if self.endpoint in ("start", "both") and start_idx >= 0:
                candidates.append(("start", start_idx))
            if self.endpoint in ("end", "both") and end_idx >= 0:
                candidates.append(("end", end_idx))

            if candidates:
                _, target_idx = candidates[np.random.randint(0, len(candidates))]
                offset = self._choose_offset_focused(L, win_len, target_idx)
            else:
                offset = self._choose_offset_uniform(L, win_len)
        else:
            offset = self._choose_offset_uniform(L, win_len)

        win = tiles[offset:offset + win_len].astype(np.float32) / 255.0  # (win_len,H,H)

        # pack to fixed L_max
        x = np.zeros((self.L_max, 1, H, H), dtype=np.float32)
        mask = np.zeros((self.L_max,), dtype=np.float32)
        x[:win_len, 0] = win
        mask[:win_len] = 1.0

        x = torch.from_numpy(x)
        mask = torch.from_numpy(mask)

        # augment (train only)
        if self.augment:
            if torch.rand(()) < self.invert_p:
                x = 1.0 - x
            if self.rot90_p > 0 and torch.rand(()) < self.rot90_p:
                k = int(torch.randint(0, 4, (1,)))
                x = torch.rot90(x, k=k, dims=(-2, -1))

        def inside(idx: int) -> float:
            if idx < 0:
                return 0.0
            return 1.0 if (offset <= idx < offset + win_len) else 0.0

        # labels
        y_start = inside(start_idx) if self.endpoint in ("start", "both") else 0.0
        y_end   = inside(end_idx)   if self.endpoint in ("end", "both") else 0.0

        return {
            "x": x,
            "mask": mask,
            "y_start": torch.tensor(y_start, dtype=torch.float32),
            "y_end": torch.tensor(y_end, dtype=torch.float32),
            "win_len": torch.tensor(win_len, dtype=torch.long),
            "offset": torch.tensor(offset, dtype=torch.long),
        }