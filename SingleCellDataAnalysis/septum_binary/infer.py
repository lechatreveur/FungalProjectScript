#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:35:17 2026

@author: user
"""

import numpy as np
import torch

@torch.no_grad()
def score_strip_sliding(model, strip_uint8, device, L_max=81, win_len=81, step=1, invert=False):
    """
    strip_uint8: (H, H*L) uint8
    Returns:
      starts: (n_windows,) probabilities
      ends:   (n_windows,) probabilities
    """
    model.eval()
    H = strip_uint8.shape[0]
    L = strip_uint8.shape[1] // H
    tiles = strip_uint8.reshape(H, L, H).transpose(1,0,2)  # (L,H,H)

    win_len = min(win_len, L_max, L)
    nwin = max(1, (L - win_len) // step + 1)

    starts = np.zeros((nwin,), dtype=np.float32)
    ends   = np.zeros((nwin,), dtype=np.float32)

    for k in range(nwin):
        off = k * step
        win = tiles[off:off+win_len].astype(np.float32) / 255.0  # (win_len,H,H)
        if invert:
            win = 1.0 - win

        x = np.zeros((L_max,1,H,H), dtype=np.float32)
        mask = np.zeros((L_max,), dtype=np.float32)
        x[:win_len,0] = win
        mask[:win_len] = 1.0

        x = torch.from_numpy(x)[None].to(device)      # (1,L_max,1,H,H)
        mask = torch.from_numpy(mask)[None].to(device) # (1,L_max)

        _, _, win_s, win_e = model(x, mask, pool="max")  # max pooling is good for detection
        starts[k] = torch.sigmoid(win_s)[0].item()
        ends[k]   = torch.sigmoid(win_e)[0].item()

    return starts, ends