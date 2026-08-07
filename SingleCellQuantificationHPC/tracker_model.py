#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracker_model.py
----------------
Siamese ResNet-18 cell tracker with three output heads:

  - Score head  : scalar match score for a candidate segment at t+1
  - Division head: p(true division at t→t+1)
  - Merge head   : p(two adjacent Cellpose segments should be merged at t+1)

The score head tells the tracker which candidate to follow.
The division/merge heads resolve the septa-vs-division ambiguity using
the external septum AI probability as an additional scalar feature.

Input:  3 channels [bf_t, bf_t1, mask_t], crop 128×128 px
Output: three independent heads, all sigmoid-activated
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _resnet18_backbone(
    in_channels: int = 3,
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """Return a ResNet-18 stem+body with modified first conv for `in_channels`."""
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    net = tv_models.resnet18(weights=weights)
    # Replace first conv: keep everything except in_channels
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                          padding=3, bias=False)
    nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
    # Remove the final FC layer; output of avgpool = 512-d
    out_dim = net.fc.in_features          # 512 for ResNet-18
    net.fc = nn.Identity()
    return net, out_dim


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, out_dim),
    )


# ─────────────────────────────────────────────
# Candidate feature extractor (small 3-layer CNN)
# ─────────────────────────────────────────────

class CandidateMaskEncoder(nn.Module):
    """
    Encodes the candidate binary mask (single channel, same crop) to a
    fixed-dim vector so the score head can condition on shape.
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),  # 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32
            nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1), # 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) – binary mask of candidate segment
        return self.net(x).squeeze(-1).squeeze(-1)   # (B, out_dim)


# ─────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────

class CellTrackerModel(nn.Module):
    """
    Siamese tracker.

    Forward signature:

        scores   = model.score(img3ch, candidate_masks)
        div_prob = model.division(img3ch, septum_t, area_ratio)
        mrg_prob = model.merge(img3ch, septum_t1, adjacency)

    Or all at once (for training):

        score, div, mrg = model(img3ch, candidate_masks,
                                septum_t, area_ratio,
                                septum_t1, adjacency)

    Parameters
    ----------
    img3ch          : (B, 3, H, W) – [bf_t, bf_t1, mask_t] normalised to [0,1]
    candidate_masks : (B, K, 1, H, W) – K candidate binary masks from Cellpose
    septum_t        : (B, 1) – septum AI probability at frame t
    area_ratio      : (B, 1) – area_t1 / area_t  (log-transformed internally)
    septum_t1       : (B, 1) – septum AI probability at frame t+1
    adjacency       : (B, 1) – 1 if two candidate segments are spatially adjacent

    Returns
    -------
    scores  : (B, K) – match score for each candidate (before sigmoid)
    div_log : (B, 1) – logit for p(dividing)
    mrg_log : (B, 1) – logit for p(merge)
    """

    CANDIDATE_DIM = 128   # mask encoder output dim
    EMBED_DIM     = 512   # ResNet-18 output
    HIDDEN        = 256

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()

        # ── shared encoder ──────────────────────────────
        self.backbone, embed_dim = _resnet18_backbone(
            in_channels=3,
            pretrained=pretrained_backbone,
        )
        assert embed_dim == self.EMBED_DIM

        # ── candidate shape encoder ─────────────────────
        self.cand_enc = CandidateMaskEncoder(out_dim=self.CANDIDATE_DIM)

        # ── score head: embed ⊕ cand_feat → scalar ──────
        self.score_head = _mlp(self.EMBED_DIM + self.CANDIDATE_DIM,
                               self.HIDDEN, 1)

        # ── division head: embed ⊕ sep_t ⊕ log(area_ratio) → scalar ──
        self.div_head = _mlp(self.EMBED_DIM + 2, self.HIDDEN, 1)

        # ── merge head: embed ⊕ sep_t1 ⊕ adjacency → scalar ──────────
        self.mrg_head = _mlp(self.EMBED_DIM + 2, self.HIDDEN, 1)

    # ── sub-methods (callable independently at inference) ──

    def embed(self, img3ch: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B, EMBED_DIM)"""
        return self.backbone(img3ch)

    def score(self, img3ch: torch.Tensor,
              candidate_masks: torch.Tensor) -> torch.Tensor:
        """
        img3ch          : (B, 3, H, W)
        candidate_masks : (B, K, 1, H, W)
        returns scores  : (B, K)  — raw logits
        """
        B, K, _, H, W = candidate_masks.shape
        emb = self.embed(img3ch)                          # (B, E)
        emb_rep = emb.unsqueeze(1).expand(-1, K, -1)     # (B, K, E)
        cands_flat = candidate_masks.view(B * K, 1, H, W)
        cand_feat = self.cand_enc(cands_flat).view(B, K, self.CANDIDATE_DIM)
        feat = torch.cat([emb_rep, cand_feat], dim=-1)   # (B, K, E+C)
        scores = self.score_head(feat).squeeze(-1)        # (B, K)
        return scores

    def division(self, img3ch: torch.Tensor,
                 septum_t: torch.Tensor,
                 area_ratio: torch.Tensor) -> torch.Tensor:
        """Returns (B,1) logit for p(dividing)."""
        emb = self.embed(img3ch)
        log_ratio = torch.log(area_ratio.clamp(min=1e-3))
        aux = torch.cat([septum_t, log_ratio], dim=-1)   # (B, 2)
        return self.div_head(torch.cat([emb, aux], dim=-1))

    def merge(self, img3ch: torch.Tensor,
              septum_t1: torch.Tensor,
              adjacency: torch.Tensor) -> torch.Tensor:
        """Returns (B,1) logit for p(should merge two segs)."""
        emb = self.embed(img3ch)
        aux = torch.cat([septum_t1, adjacency], dim=-1)  # (B, 2)
        return self.mrg_head(torch.cat([emb, aux], dim=-1))

    def forward(self,
                img3ch: torch.Tensor,
                candidate_masks: torch.Tensor,
                septum_t: torch.Tensor,
                area_ratio: torch.Tensor,
                septum_t1: torch.Tensor,
                adjacency: torch.Tensor):
        emb = self.embed(img3ch)

        # Score head
        B, K, _, H, W = candidate_masks.shape
        emb_rep = emb.unsqueeze(1).expand(-1, K, -1)
        cands_flat = candidate_masks.view(B * K, 1, H, W)
        cand_feat = self.cand_enc(cands_flat).view(B, K, self.CANDIDATE_DIM)
        feat = torch.cat([emb_rep, cand_feat], dim=-1)
        scores = self.score_head(feat).squeeze(-1)        # (B, K)

        # Division head
        log_ratio = torch.log(area_ratio.clamp(min=1e-3))
        div_logit = self.div_head(torch.cat(
            [emb, septum_t, log_ratio], dim=-1))          # (B, 1)

        # Merge head
        mrg_logit = self.mrg_head(torch.cat(
            [emb, septum_t1, adjacency], dim=-1))         # (B, 1)

        return scores, div_logit, mrg_logit


# ─────────────────────────────────────────────
# Convenience: load from checkpoint
# ─────────────────────────────────────────────

def load_tracker(ckpt_path: str, device: str = "cpu") -> CellTrackerModel:
    # The checkpoint contains the complete model state, so inference does not
    # need torchvision's pretrained weights (or network/cache access).
    model = CellTrackerModel(pretrained_backbone=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model
