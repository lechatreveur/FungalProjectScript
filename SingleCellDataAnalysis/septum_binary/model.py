#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


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
        return self.net(x)[:, :, 0, 0]  # (B*L, D)


def masked_logsumexp(x, mask, dim=1, temperature=1.0):
    x = x / temperature
    neg_inf = torch.finfo(x.dtype).min
    x = x.masked_fill(mask == 0, neg_inf)
    return torch.logsumexp(x, dim=dim) * temperature


def masked_softmax(x, mask, dim=1):
    neg_inf = torch.finfo(x.dtype).min
    x = x.masked_fill(mask == 0, neg_inf)
    return torch.softmax(x, dim=dim)


class EndpointMILModel(nn.Module):
    """
    Produces:
      tile_s, tile_e: (B, L) tile logits
      win_s,  win_e:  (B,)   window logits via MIL pooling
    """

    def __init__(self, D: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = TileEncoder(D=D)
        self.temporal = nn.Sequential(
            nn.Conv1d(D, D, 3, padding=1), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(D, D, 3, padding=1), nn.ReLU(),
        )
        self.head_start = nn.Conv1d(D, 1, 1)
        self.head_end   = nn.Conv1d(D, 1, 1)

        # optional: attention pooling (separate for start/end)
        self.attn_start = nn.Conv1d(D, 1, 1)
        self.attn_end   = nn.Conv1d(D, 1, 1)

    def forward(self, x, mask, pool="logsumexp", temperature=1.0):
        # x: (B, L, 1, H, W) mask: (B, L)
        B, L, _, H, W = x.shape
        emb = self.enc(x.reshape(B * L, 1, H, W)).reshape(B, L, -1)  # (B, L, D)
        feat = self.temporal(emb.transpose(1, 2))                    # (B, D, L)

        tile_s = self.head_start(feat)[:, 0, :]  # (B, L)
        tile_e = self.head_end(feat)[:, 0, :]    # (B, L)

        if pool == "max":
            neg_inf = torch.finfo(tile_s.dtype).min
            win_s = tile_s.masked_fill(mask == 0, neg_inf).max(dim=1).values
            win_e = tile_e.masked_fill(mask == 0, neg_inf).max(dim=1).values

        elif pool == "attn":
            # attention weights computed from features, then weighted sum of tile logits
            a_s = self.attn_start(feat)[:, 0, :]  # (B, L)
            a_e = self.attn_end(feat)[:, 0, :]    # (B, L)
            w_s = masked_softmax(a_s, mask, dim=1)
            w_e = masked_softmax(a_e, mask, dim=1)
            win_s = (w_s * tile_s).sum(dim=1)
            win_e = (w_e * tile_e).sum(dim=1)

        else:  # default logsumexp
            win_s = masked_logsumexp(tile_s, mask, dim=1, temperature=temperature)
            win_e = masked_logsumexp(tile_e, mask, dim=1, temperature=temperature)

        return tile_s, tile_e, win_s, win_e