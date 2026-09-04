#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracker_dataset.py
------------------
PyTorch Dataset for training the CellTrackerModel.

For each corrected cell track it extracts consecutive frame pairs (t, t+1)
and assembles:

  img3ch          : (3, 128, 128) – [bf_t, bf_t1, mask_t] in [0,1]
  candidate_masks : (K, 1, 128, 128) – K candidates from Cellpose seg at t+1
                    index 0 = ground-truth positive, rest = negatives
  gt_score_idx    : int – index of the positive candidate (always 0 after shuffle)
  septum_t        : float – septum AI probability at frame t
  septum_t1       : float – septum AI probability at frame t+1
  area_ratio      : float – area_t1 / area_t
  div_label       : int – 1 if true division at t→t+1
  mrg_label       : int – 1 if Cellpose wrongly split the cell at t+1
  adjacency       : float – 1.0 if two top-2 candidates are spatially adjacent
"""

from __future__ import annotations

import os
import sys
import re
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.measure import regionprops
from skimage.transform import resize

# Project-level imports (caller must set sys.path)
sys.path.insert(0, str(Path(__file__).parent.parent))
from Cell_tracking_functions import (
    load_segmentation, to_labeled_current,
    rle_decode, compute_overlap,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CROP_SIZE  = 128   # pixels – all crops resized to this
TOPK_NEG   = 4     # how many negative candidates to include per sample
TILE_SIZE  = 96    # for FungalInferenceCore strip (must match training)
SEPTUM_MODEL_PATH = (
    "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
    "training_dataset/checkpoints_binary/model_best.pt"
)


# ─────────────────────────────────────────────
# Film discovery
# ─────────────────────────────────────────────

def discover_corrected_films(movie_root: str) -> List[Dict]:
    """
    Walk MOVIE_ROOT and find all (experiment, film) pairs that have corrected
    cells in qc.csv.  Returns a list of dicts with film metadata.
    """
    records = []
    movie_root = Path(movie_root)
    for qc_path in sorted(movie_root.rglob("qc.csv")):
        try:
            qc = pd.read_csv(qc_path)
        except Exception:
            continue
        from tracking_corrector.qc_schema import GlobalCellQC
        corrected = set(
            int(r["cell_id"])
            for _, r in qc.iterrows()
            if str(r.get("status", "")).lower() == GlobalCellQC.CORRECTED.value
        )

        if not corrected:
            continue

        tracked_dir = qc_path.parent
        film_match  = re.match(r"TrackedCells_(.+)$", tracked_dir.name)
        if not film_match:
            continue
        film = film_match.group(1)
        film_dir  = tracked_dir.parent
        exp_dir   = film_dir.parent
        frames_dir = film_dir / f"Frames_{film}"
        masks_dir  = film_dir / f"Masks_{film}"

        if not frames_dir.is_dir() or not masks_dir.is_dir():
            continue

        records.append({
            "experiment": exp_dir.name,
            "film":        film,
            "film_dir":    film_dir,
            "frames_dir":  frames_dir,
            "masks_dir":   masks_dir,
            "tracked_dir": tracked_dir,
            "corrected_cells": corrected,
        })
    return records


def _detect_channel(masks_dir: Path) -> int:
    for f in sorted(masks_dir.iterdir()):
        m = re.search(r"_t_\d+_c_(\d+)_seg\.(tif|npy)$", f.name)
        if m:
            return int(m.group(1))
    return 0


# ─────────────────────────────────────────────
# Septum score cache (per film)
# ─────────────────────────────────────────────

def _build_septum_scores(
    film_meta: Dict,
    cell_id: int,
    masks_df: pd.DataFrame,
    rle_col: str,
    num_frames: int,
    channel_idx: int,
) -> np.ndarray:
    """
    Returns septum_probs: (T,) float32 using FungalInferenceCore.
    Falls back to zeros if the model or frames are unavailable.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "SingleCellDataAnalysis"))
        from inference_core import FungalInferenceCore
    except ImportError:
        return np.zeros(num_frames, dtype=np.float32)

    ckpt = SEPTUM_MODEL_PATH
    if not os.path.exists(ckpt):
        return np.zeros(num_frames, dtype=np.float32)

    try:
        runner = FungalInferenceCore(ckpt, device="cpu")
    except Exception:
        return np.zeros(num_frames, dtype=np.float32)

    frames_dir = film_meta["frames_dir"]
    film       = film_meta["film"]

    # Build strip (TILE_SIZE × TILE_SIZE*T)
    strip = np.zeros((TILE_SIZE, TILE_SIZE * num_frames), dtype=np.uint8)
    H_img, W_img = None, None

    for t in range(num_frames):
        row = masks_df[masks_df["time_point"] == t]
        if row.empty:
            continue
        rle_str = str(row.iloc[0][rle_col]).strip()
        if not rle_str or rle_str == "nan":
            continue

        frame_p = frames_dir / f"{film}_t_{t:03d}_c_{channel_idx}.tif"
        if not frame_p.exists():
            continue
        try:
            img = imread(str(frame_p))
        except Exception:
            continue
        if H_img is None:
            H_img, W_img = img.shape[:2]
            mask_shape = (H_img, W_img)

        mask = rle_decode(rle_str, (H_img, W_img)).astype(bool)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        r0, r1 = ys.min(), ys.max() + 1
        c0, c1 = xs.min(), xs.max() + 1
        pad = max(10, (max(r1-r0, c1-c0)) // 4)
        r0, r1 = max(0, r0-pad), min(H_img, r1+pad)
        c0, c1 = max(0, c0-pad), min(W_img, c1+pad)

        crop = img[r0:r1, c0:c1].astype(np.float32)
        crop_min, crop_max = crop.min(), crop.max()
        if crop_max > crop_min:
            crop = (crop - crop_min) / (crop_max - crop_min) * 255
        tile_resized = resize(crop, (TILE_SIZE, TILE_SIZE),
                              order=1, anti_aliasing=True,
                              preserve_range=True).astype(np.uint8)
        strip[:, t * TILE_SIZE:(t + 1) * TILE_SIZE] = tile_resized

    try:
        probs = runner.predict_strip(strip)
        if probs is None:
            return np.zeros(num_frames, dtype=np.float32)
        # probs may be shorter than num_frames (if strip has padding)
        out = np.zeros(num_frames, dtype=np.float32)
        out[:len(probs)] = probs[:num_frames]
        return out
    except Exception:
        return np.zeros(num_frames, dtype=np.float32)


# ─────────────────────────────────────────────
# Crop helpers
# ─────────────────────────────────────────────

def _padded_bbox(mask: np.ndarray, H: int, W: int,
                 pad_frac: float = 0.4) -> Tuple[int, int, int, int]:
    """Return (r0,r1,c0,c1) with padding around the mask bounding box."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, H, 0, W
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    pad = max(16, int(pad_frac * max(h, w)))
    r0 = max(0, ys.min() - pad)
    r1 = min(H, ys.max() + pad + 1)
    c0 = max(0, xs.min() - pad)
    c1 = min(W, xs.max() + pad + 1)
    return int(r0), int(r1), int(c0), int(c1)


def _crop_and_resize(img: np.ndarray, r0: int, r1: int, c0: int, c1: int,
                     size: int = CROP_SIZE) -> np.ndarray:
    crop = img[r0:r1, c0:c1]
    out  = resize(crop.astype(np.float32), (size, size),
                  order=1, anti_aliasing=True, preserve_range=True)
    return out


def _norm_img(arr: np.ndarray) -> np.ndarray:
    """Normalise a single-channel image to [0,1] float32."""
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    return a


def _augment(img3ch: np.ndarray, cand_masks: np.ndarray,
             p_flip: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Random left-right flip (shared across all channels)."""
    if random.random() < p_flip:
        img3ch    = img3ch[:, :, ::-1].copy()
        cand_masks = cand_masks[:, :, :, ::-1].copy()
    return img3ch, cand_masks


def _segments_adjacent(mask_a: np.ndarray, mask_b: np.ndarray,
                        max_dist_px: int = 10) -> bool:
    """True if masks share a border pixel within max_dist_px."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((max_dist_px * 2 + 1,) * 2, bool)
    return bool((binary_dilation(mask_a, structure=struct) & mask_b).any())


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TrackerDataset(Dataset):
    """
    Each item is one (t → t+1) frame-pair for one corrected cell.

    Only non-trivial frames are included:
      - mask at t must be non-empty
      - at least one Cellpose segment exists at t+1

    Parameters
    ----------
    movie_root     : path to the Movies NAS directory
    hold_out_film  : if set, skip all cells from this film (for train/val split)
    augment        : apply random flips at train time
    topk_neg       : how many negative candidates to sample
    """

    def __init__(
        self,
        movie_root: str,
        hold_out_film: Optional[str] = None,
        curated_csv: Optional[str] = None,
        augment: bool = True,
        topk_neg: int = TOPK_NEG,
    ):
        self.movie_root = Path(movie_root)
        self.hold_out_film = hold_out_film
        self.augment  = augment
        self.topk_neg = topk_neg
        self.samples: List[Dict] = []

        # Determine pickle cache path
        cache_name = f"samples_cache_{hold_out_film or 'none'}.pkl"
        if curated_csv:
            cache_name = f"samples_cache_curated_{hold_out_film or 'none'}.pkl"
        samples_cache_path = Path(__file__).parent / cache_name

        # Check cache validity
        cache_valid = False
        if samples_cache_path.exists():
            cache_valid = True
            if curated_csv:
                csv_path = Path(curated_csv)
                if csv_path.exists() and samples_cache_path.stat().st_mtime < csv_path.stat().st_mtime:
                    cache_valid = False

        if cache_valid:
            print(f"[TrackerDataset] Loading indexed samples from cache: {samples_cache_path}")
            try:
                import pickle
                with open(samples_cache_path, "rb") as f:
                    self.samples = pickle.load(f)
                print(f"[TrackerDataset] {len(self.samples)} samples loaded from pickle cache.")
                return
            except Exception as e:
                print(f"[TrackerDataset] Failed to load pickle cache: {e}. Rebuilding...")
                self.samples = []

        if curated_csv and Path(curated_csv).exists():
            print(f"[TrackerDataset] Loading directly from CSV: {curated_csv}")
            self._load_from_csv(curated_csv)
            print(f"[TrackerDataset] {len(self.samples)} frame-pair samples loaded.")
        else:
            films = discover_corrected_films(movie_root)
            if not films:
                raise RuntimeError(f"No corrected films found under {movie_root}")

            for meta in films:
                if hold_out_film and meta["film"] == hold_out_film:
                    print(f"  [dataset] Hold-out: skipping {meta['film']}")
                    continue
                self._index_film(meta)

            print(f"[TrackerDataset] {len(self.samples)} frame-pair samples "
                  f"from {len(films)} films")

        # Save to pickle cache
        try:
            import pickle
            with open(samples_cache_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"[TrackerDataset] Saved samples cache to {samples_cache_path}")
        except Exception as e:
            print(f"[TrackerDataset] Failed to save samples cache: {e}")
              
    def _load_from_csv(self, csv_path: str):
        df_csv = pd.read_csv(csv_path)
        grouped = df_csv.groupby(["film", "cell_id"])
        
        for (film_name, cell_id), group in grouped:
            if self.hold_out_film and film_name == self.hold_out_film:
                print(f"  [dataset] Hold-out: skipping {film_name}")
                continue
            film_dir = self.movie_root / "2026_01_08_M93" / film_name
            if not film_dir.exists():
                film_dir = self.movie_root / film_name
            if not film_dir.exists():
                continue
                
            frames_dir = film_dir / f"Frames_{film_name}"
            masks_dir = film_dir / f"Masks_{film_name}"
            tracked_dir = film_dir / f"TrackedCells_{film_name}"
            
            csv_file = tracked_dir / f"cell_{cell_id}_masks.csv"
            if not csv_file.exists():
                continue
                
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
                
            rle_col = "rle_bf"
            if "rle_gfp" in df.columns and df["rle_gfp"].dropna().any():
                rle_col = "rle_gfp"
                
            H = int(df.iloc[0]["height"])
            W = int(df.iloc[0]["width"])
            
            channel_idx = 0
            
            # Discover frame count
            frame_files = sorted([
                f for f in frames_dir.iterdir()
                if f.name.endswith(".tif")
                and not f.name.endswith("_seg.tif")
                and f"_c_{channel_idx}.tif" in f.name
            ])
            T = len(frame_files)
            if T < 2:
                continue
                
            film_meta = {
                "film": film_name,
                "film_dir": film_dir,
                "frames_dir": frames_dir,
                "masks_dir": masks_dir,
                "tracked_dir": tracked_dir,
            }
            sep_probs = _build_septum_scores(
                film_meta, cell_id, df, rle_col, T, channel_idx)
                
            for _, row_item in group.iterrows():
                t = int(row_item["t"])
                if t >= T - 1:
                    continue
                row_t = df[df["time_point"] == t]
                row_t1 = df[df["time_point"] == t + 1]
                if row_t.empty or row_t1.empty:
                    continue
                    
                rle_t = str(row_t.iloc[0][rle_col]).strip()
                rle_t1 = str(row_t1.iloc[0][rle_col]).strip()
                if not rle_t or rle_t == "nan":
                    continue
                if not rle_t1 or rle_t1 == "nan":
                    continue
                    
                mask_t = rle_decode(rle_t, (H, W)).astype(bool)
                mask_t1 = rle_decode(rle_t1, (H, W)).astype(bool)
                if not mask_t.any() or not mask_t1.any():
                    continue
                    
                area_t = float(mask_t.sum())
                area_t1 = float(mask_t1.sum())
                
                comp_col = "composition_bf" if "composition_bf" in df.columns else None
                comp_t1 = ""
                if comp_col:
                    comp_t1 = str(row_t1.iloc[0].get(comp_col, ""))
                div_label = int(
                    comp_t1 == "pair" or
                    (area_t > 0 and area_t1 / area_t < 0.60)
                )
                
                mrg_label = 0
                seg_t1_path = masks_dir / f"{film_name}_t_{t+1:03d}_c_{channel_idx}_seg.tif"
                if seg_t1_path.exists():
                    try:
                        seg_t1 = to_labeled_current(load_segmentation(str(seg_t1_path)))
                        mrg_label = int(self._is_merge_frame(
                            mask_t1, seg_t1, sep_probs[t + 1]))
                    except Exception:
                        pass
                        
                self.samples.append({
                    "film":        film_name,
                    "film_dir":    film_dir,
                    "frames_dir":  frames_dir,
                    "masks_dir":   masks_dir,
                    "channel_idx": channel_idx,
                    "H": H, "W": W,
                    "cell_id":     cell_id,
                    "t":           t,
                    "rle_t":       rle_t,
                    "rle_t1":      rle_t1,
                    "area_t":      area_t,
                    "area_t1":     area_t1,
                    "sep_t":       float(sep_probs[t]),
                    "sep_t1":      float(sep_probs[t + 1]),
                    "div_label":   div_label,
                    "mrg_label":   mrg_label,
                })

    # ── film indexing ───────────────────────────────────────────────

    def _index_film(self, meta: Dict) -> None:
        channel_idx  = _detect_channel(meta["masks_dir"])
        tracked_dir  = meta["tracked_dir"]
        frames_dir   = meta["frames_dir"]
        masks_dir    = meta["masks_dir"]
        film         = meta["film"]
        corrected    = meta["corrected_cells"]

        # Discover frame count
        frame_files = sorted([
            f for f in frames_dir.iterdir()
            if f.name.endswith(".tif")
            and not f.name.endswith("_seg.tif")
            and f"_c_{channel_idx}.tif" in f.name
        ])
        T = len(frame_files)
        if T < 2:
            return

        for cell_id in corrected:
            csv_path = tracked_dir / f"cell_{cell_id}_masks.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            # Determine RLE column
            rle_col = "rle_bf"
            if "rle_gfp" in df.columns and df["rle_gfp"].dropna().any():
                rle_col = "rle_gfp"

            H = int(df.iloc[0]["height"])
            W = int(df.iloc[0]["width"])

            # Compute septum scores for this cell
            sep_probs = _build_septum_scores(
                meta, cell_id, df, rle_col, T, channel_idx)

            # Enumerate frame pairs
            for t in range(T - 1):
                row_t  = df[df["time_point"] == t]
                row_t1 = df[df["time_point"] == t + 1]
                if row_t.empty or row_t1.empty:
                    continue

                rle_t  = str(row_t.iloc[0][rle_col]).strip()
                rle_t1 = str(row_t1.iloc[0][rle_col]).strip()
                if not rle_t or rle_t == "nan":
                    continue
                if not rle_t1 or rle_t1 == "nan":
                    continue

                mask_t  = rle_decode(rle_t,  (H, W)).astype(bool)
                mask_t1 = rle_decode(rle_t1, (H, W)).astype(bool)
                if not mask_t.any() or not mask_t1.any():
                    continue

                area_t  = float(mask_t.sum())
                area_t1 = float(mask_t1.sum())

                # Division label: comp == 'pair' OR area drops > 40%
                comp_col = "composition_bf" if "composition_bf" in df.columns else None
                comp_t1 = ""
                if comp_col:
                    comp_t1 = str(row_t1.iloc[0].get(comp_col, ""))
                div_label = int(
                    comp_t1 == "pair" or
                    (area_t > 0 and area_t1 / area_t < 0.60)
                )

                # Merge label: if there are 2+ seg.tif segments that together
                # cover mask_t1 well, but individually don't → probable false split
                mrg_label = 0
                seg_t1_path = masks_dir / f"{film}_t_{t+1:03d}_c_{channel_idx}_seg.tif"
                if seg_t1_path.exists():
                    try:
                        seg_t1 = to_labeled_current(load_segmentation(str(seg_t1_path)))
                        mrg_label = int(self._is_merge_frame(
                            mask_t1, seg_t1, sep_probs[t + 1]))
                    except Exception:
                        pass

                self.samples.append({
                    "film":        film,
                    "film_dir":    meta["film_dir"],
                    "frames_dir":  frames_dir,
                    "masks_dir":   masks_dir,
                    "channel_idx": channel_idx,
                    "H": H, "W": W,
                    "cell_id":     cell_id,
                    "t":           t,
                    "rle_t":       rle_t,
                    "rle_t1":      rle_t1,
                    "area_t":      area_t,
                    "area_t1":     area_t1,
                    "sep_t":       float(sep_probs[t]),
                    "sep_t1":      float(sep_probs[t + 1]),
                    "div_label":   div_label,
                    "mrg_label":   mrg_label,
                })

    @staticmethod
    def _is_merge_frame(mask_t1: np.ndarray, seg_lbl: np.ndarray,
                        sep_prob: float, sep_thr: float = 0.4) -> bool:
        """
        Heuristic: if septum prob is high AND the best single segment covers
        < 70% of mask_t1 but the best pair covers > 80% → likely false split.
        """
        if sep_prob < sep_thr:
            return False
        labels = [l for l in np.unique(seg_lbl) if l != 0]
        if len(labels) < 2:
            return False

        overlaps = [(compute_overlap(mask_t1, seg_lbl == l), l)
                    for l in labels]
        overlaps.sort(reverse=True)
        best_single = overlaps[0][0]
        if best_single > 0.70:
            return False   # single segment already good enough

        # Try best pair
        l0 = overlaps[0][1]
        l1 = overlaps[1][1]
        pair_mask = (seg_lbl == l0) | (seg_lbl == l1)
        best_pair = compute_overlap(mask_t1, pair_mask)
        return best_pair > 0.80

    # ── __len__ / __getitem__ ───────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # SSD cache setup
        cache_dir = Path(__file__).parent / "tracker_dataset_cache"
        cache_path = cache_dir / f"sample_{idx}.npz"

        if not cache_path.exists():
            film    = s["film"]
            fd      = s["frames_dir"]
            md      = s["masks_dir"]
            ch      = s["channel_idx"]
            t       = s["t"]
            H, W    = s["H"], s["W"]

            # ── load images ─────────────────────────────────────────────
            def load_frame(t_: int):
                p = fd / f"{film}_t_{t_:03d}_c_{ch}.tif"
                if p.exists():
                    return imread(str(p))
                return np.zeros((H, W), dtype=np.uint16)

            img_t  = load_frame(t)
            img_t1 = load_frame(t + 1)

            mask_t  = rle_decode(s["rle_t"],  (H, W)).astype(bool)
            mask_t1 = rle_decode(s["rle_t1"], (H, W)).astype(bool)

            # ── bounding box centred on mask_t ──────────────────────────
            r0, r1, c0, c1 = _padded_bbox(mask_t, H, W)

            bf_t  = _crop_and_resize(_norm_img(img_t),  r0, r1, c0, c1)
            bf_t1 = _crop_and_resize(_norm_img(img_t1), r0, r1, c0, c1)
            mk_t  = _crop_and_resize(mask_t.astype(np.float32), r0, r1, c0, c1)

            # img3ch: (3, CROP_SIZE, CROP_SIZE)
            img3ch = np.stack([bf_t, bf_t1, mk_t], axis=0).astype(np.float32)

            # ── candidate masks at t+1 ───────────────────────────────────
            seg_t1_path = md / f"{film}_t_{t+1:03d}_c_{ch}_seg.tif"
            candidates, adjacency = self._build_candidates(
                seg_t1_path, mask_t1, H, W, r0, r1, c0, c1)

            # ── area ratio ──────────────────────────────────────────────
            area_ratio = float(s["area_t1"]) / max(float(s["area_t"]), 1.0)

            # Save to SSD cache
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    img3ch=img3ch,
                    candidates=candidates,
                    adjacency=adjacency,
                    area_ratio=area_ratio
                )
            except Exception as e:
                # Fallback silently if save fails (e.g., write permission issue)
                pass
        else:
            # Load from SSD cache
            try:
                data = np.load(cache_path)
                img3ch = data["img3ch"]
                candidates = data["candidates"]
                adjacency = float(data["adjacency"])
                area_ratio = float(data["area_ratio"])
            except Exception:
                # Recompute if cache is corrupted
                film    = s["film"]
                fd      = s["frames_dir"]
                md      = s["masks_dir"]
                ch      = s["channel_idx"]
                t       = s["t"]
                H, W    = s["H"], s["W"]

                def load_frame(t_):
                    p = fd / f"{film}_t_{t_:03d}_c_{ch}.tif"
                    return imread(str(p)) if p.exists() else np.zeros((H, W), dtype=np.uint16)

                img_t  = load_frame(t)
                img_t1 = load_frame(t + 1)
                mask_t  = rle_decode(s["rle_t"],  (H, W)).astype(bool)
                mask_t1 = rle_decode(s["rle_t1"], (H, W)).astype(bool)
                r0, r1, c0, c1 = _padded_bbox(mask_t, H, W)
                bf_t  = _crop_and_resize(_norm_img(img_t), r0, r1, c0, c1)
                bf_t1 = _crop_and_resize(_norm_img(img_t1), r0, r1, c0, c1)
                mk_t  = _crop_and_resize(mask_t.astype(np.float32), r0, r1, c0, c1)
                img3ch = np.stack([bf_t, bf_t1, mk_t], axis=0).astype(np.float32)
                candidates, adjacency = self._build_candidates(
                    md / f"{film}_t_{t+1:03d}_c_{ch}_seg.tif", mask_t1, H, W, r0, r1, c0, c1)
                area_ratio = float(s["area_t1"]) / max(float(s["area_t"]), 1.0)

        # ── augmentation ────────────────────────────────────────────
        # Copy arrays to avoid mutating cached files or read-only numpy structures in memory
        img3ch = img3ch.copy()
        candidates = candidates.copy()
        if self.augment:
            img3ch, candidates = _augment(img3ch, candidates)

        return {
            "img3ch":    torch.from_numpy(img3ch),                   # (3,H,W)
            "candidates": torch.from_numpy(candidates),              # (K,1,H,W)
            "sep_t":     torch.tensor([s["sep_t"]],  dtype=torch.float32),
            "sep_t1":    torch.tensor([s["sep_t1"]], dtype=torch.float32),
            "area_ratio":torch.tensor([area_ratio],  dtype=torch.float32),
            "adjacency": torch.tensor([adjacency],   dtype=torch.float32),
            "div_label": torch.tensor([s["div_label"]], dtype=torch.float32),
            "mrg_label": torch.tensor([s["mrg_label"]], dtype=torch.float32),
            # gt candidate is always index 0
            "gt_idx":    torch.tensor(0, dtype=torch.long),
        }

    def _build_candidates(
        self,
        seg_path: Path,
        gt_mask: np.ndarray,
        H: int, W: int,
        r0: int, r1: int, c0: int, c1: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Build a (K, 1, CROP_SIZE, CROP_SIZE) array of candidate binary masks.
        Candidate 0 = ground-truth (corrected mask).
        Candidates 1..K = top negative Cellpose segments by overlap.
        Returns (candidates, adjacency_flag).
        """
        K = 1 + self.topk_neg

        # Ground-truth crop (always index 0)
        gt_crop = _crop_and_resize(gt_mask.astype(np.float32),
                                   r0, r1, c0, c1)
        candidates = np.zeros((K, 1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        candidates[0, 0] = gt_crop

        adjacency = 0.0

        if not seg_path.exists():
            # No negatives available — fill with zeros
            return candidates, adjacency

        try:
            seg = to_labeled_current(load_segmentation(str(seg_path)))
        except Exception:
            return candidates, adjacency

        labels = [l for l in np.unique(seg) if l != 0]
        if not labels:
            return candidates, adjacency

        # Rank by overlap with gt
        ranked = sorted(
            [(compute_overlap(gt_mask, seg == l), l) for l in labels],
            reverse=True,
        )

        # Adjacency: are the top-2 segments spatially adjacent?
        if len(ranked) >= 2:
            m0 = (seg == ranked[0][1])
            m1 = (seg == ranked[1][1])
            adjacency = float(_segments_adjacent(m0, m1))

        # Fill negatives: pick candidates that are NOT the GT mask
        neg_count = 0
        for ov, lbl in ranked:
            if neg_count >= self.topk_neg:
                break
            cm = (seg == lbl).astype(np.float32)
            # Skip if it's essentially the GT
            if ov > 0.90:
                continue
            crop = _crop_and_resize(cm, r0, r1, c0, c1)
            candidates[1 + neg_count, 0] = crop
            neg_count += 1

        return candidates, adjacency
