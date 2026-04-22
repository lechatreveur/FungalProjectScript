#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for septum alignment GUI.

Stable pieces that should not need frequent editing:
- TIFF/PNG IO
- RLE decoding (matches your tracking rle_encode/rle_decode exactly)
- mask bbox + cropping + tiling
- discovery/loading of cell mask CSVs
- tile cache path and generation
- composing contact sheet
- saving/loading GUI state (JSON) and exporting CSV
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


# -------------------------
# IO helpers (tif + png)
# -------------------------
def read_tif_gray(path: str) -> np.ndarray:
    """Read TIFF frame and return 2D grayscale array."""
    try:
        import tifffile
        img = tifffile.imread(path)
    except Exception:
        import imageio.v3 as iio
        img = iio.imread(path)

    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def write_png_gray(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = np.nanpercentile(a, [1, 99]) if np.isfinite(a).any() else (0.0, 1.0)
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        a = np.clip((a - lo) / (hi - lo), 0, 1)
        arr = (255 * a).astype(np.uint8)

    try:
        import imageio.v3 as iio
        iio.imwrite(path, arr)
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(path)


def read_png_gray(path: str) -> np.ndarray:
    try:
        import imageio.v3 as iio
        img = iio.imread(path)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path))

    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


# -------------------------
# RLE decoding (your exact tracking convention)
# -------------------------
def decode_rle_mask_tracking(rle, shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Decode start/len pairs with:
      - 1-based starts
      - Fortran order flattening (order='F')
    Matches your rle_encode/rle_decode exactly.
    Returns uint8 mask 0/1.
    """
    H, W = shape_hw
    if rle is None or rle == "" or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((H, W), dtype=np.uint8)

    s = str(rle).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return np.zeros((H, W), dtype=np.uint8)

    parts = s.split()
    if len(parts) < 2:
        return np.zeros((H, W), dtype=np.uint8)

    try:
        nums = list(map(int, parts))
    except Exception:
        nums = list(map(int, re.findall(r"-?\d+", s)))
        if len(nums) < 2:
            return np.zeros((H, W), dtype=np.uint8)

    starts = np.array(nums[0::2], dtype=np.int64) - 1
    lengths = np.array(nums[1::2], dtype=np.int64)
    ends = starts + lengths

    flat = np.zeros(H * W, dtype=np.uint8)
    n = flat.size
    for st, en in zip(starts, ends):
        if en <= 0 or st >= n:
            continue
        st = max(0, int(st))
        en = min(n, int(en))
        if en > st:
            flat[st:en] = 1

    return flat.reshape((H, W), order="F")


# -------------------------
# Cropping / tiling
# -------------------------
def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def crop_with_pad(img: np.ndarray, bbox: Tuple[int, int, int, int], pad: int) -> np.ndarray:
    H, W = img.shape[:2]
    y0, y1, x0, x1 = bbox
    y0 = max(0, y0 - pad)
    y1 = min(H - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(W - 1, x1 + pad)
    return img[y0:y1 + 1, x0:x1 + 1]


def to_tile(img: np.ndarray, tile_hw: Tuple[int, int]) -> np.ndarray:
    """
    Robust normalize -> center crop/pad into fixed tile size. No resizing.
    Returns uint8 tile.
    """
    Ht, Wt = tile_hw
    a = np.asarray(img)

    if a.dtype != np.uint8:
        af = a.astype(np.float32)
        lo, hi = np.nanpercentile(af, [1, 99]) if np.isfinite(af).any() else (0.0, 1.0)
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        af = np.clip((af - lo) / (hi - lo), 0, 1)
        a = (255 * af).astype(np.uint8)
    else:
        a = a.copy()

    h, w = a.shape[:2]

    # crop if too big
    if h > Ht:
        y0 = (h - Ht) // 2
        a = a[y0:y0 + Ht, :]
        h = Ht
    if w > Wt:
        x0 = (w - Wt) // 2
        a = a[:, x0:x0 + Wt]
        w = Wt

    # pad if too small
    out = np.zeros((Ht, Wt), dtype=np.uint8)
    y0 = (Ht - h) // 2
    x0 = (Wt - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    return out


# -------------------------
# Paths / discovery
# -------------------------
@dataclass(frozen=True)
class FilmPaths:
    film_dir: str
    frames_dir: str
    tracked_dir: str
    cache_img_dir: str
    label_dir: str
    json_path: str
    csv_path: str


def build_film_paths(working_dir: str, film_name: str) -> FilmPaths:
    film_dir = os.path.join(working_dir, film_name)
    frames_dir = os.path.join(film_dir, f"Frames_{film_name}")
    tracked_dir = os.path.join(film_dir, f"TrackedCells_{film_name}")
    cache_img_dir = os.path.join(tracked_dir, "cell_plots", "gui")
    label_dir = os.path.join(tracked_dir, "cell_plots", "gui_labels")
    json_path = os.path.join(label_dir, "global_septum_alignment.json")
    csv_path = os.path.join(label_dir, "septum_interval_per_cell.csv")
    return FilmPaths(
        film_dir=film_dir,
        frames_dir=frames_dir,
        tracked_dir=tracked_dir,
        cache_img_dir=cache_img_dir,
        label_dir=label_dir,
        json_path=json_path,
        csv_path=csv_path,
    )


def discover_cell_mask_csvs(tracked_dir: str) -> Tuple[List[int], Dict[int, str]]:
    cell_csvs = []
    for fn in os.listdir(tracked_dir):
        if fn.startswith("cell_") and fn.endswith("_masks.csv"):
            m = re.match(r"cell_(\d+)_masks\.csv$", fn)
            if m:
                cell_csvs.append((int(m.group(1)), os.path.join(tracked_dir, fn)))

    if not cell_csvs:
        raise FileNotFoundError(f"No cell_*_masks.csv found in: {tracked_dir}")

    cell_csvs.sort(key=lambda x: x[0])
    all_cell_ids = [cid for cid, _ in cell_csvs]
    cell_csv_map = {cid: path for cid, path in cell_csvs}
    return all_cell_ids, cell_csv_map


# -------------------------
# Mask tables cache
# -------------------------
class MaskTableCache:
    """
    Lazy in-memory cache for each cell's mask CSV.
    Keeps:
      - dfm: time_point + mask_col rows where mask exists
      - tps: list of available tps
      - min/max tp
    """
    def __init__(self, cell_csv_map: Dict[Tuple[str, int], str], time_col: str, mask_col: str):
        self.cell_csv_map = cell_csv_map
        self.time_col = time_col
        self.mask_col = mask_col
        self._df: Dict[Tuple[str, int], pd.DataFrame] = {}
        self._tps: Dict[Tuple[str, int], List[int]] = {}
        self._minmax: Dict[Tuple[str, int], Tuple[Optional[int], Optional[int]]] = {}

    def load(self, key: Tuple[str, int]) -> pd.DataFrame:
        if key in self._df:
            return self._df[key]

        path = self.cell_csv_map[key]
        df = pd.read_csv(path)
        if self.time_col not in df.columns:
            raise KeyError(f"{path} missing column '{self.time_col}'")
            
        actual_mask_col = self.mask_col
        if actual_mask_col not in df.columns:
            if "rle_gfp" in df.columns:
                actual_mask_col = "rle_gfp"
            elif "rle_bf" in df.columns:
                actual_mask_col = "rle_bf"
            else:
                raise KeyError(f"{path} missing column '{self.mask_col}' and no fallback found")

        df = df[[self.time_col, actual_mask_col]].copy()
        if actual_mask_col != self.mask_col:
            df = df.rename(columns={actual_mask_col: self.mask_col})
        df[self.time_col] = pd.to_numeric(df[self.time_col], errors="coerce").astype("Int64")
        df = df.dropna(subset=[self.time_col]).copy()
        df[self.time_col] = df[self.time_col].astype(int)

        def _has_rle(x) -> bool:
            if x is None:
                return False
            if isinstance(x, float) and np.isnan(x):
                return False
            s = str(x).strip()
            return (s != "" and s.lower() not in ("nan", "none"))

        df["_has"] = df[self.mask_col].apply(_has_rle)
        df_ok = df[df["_has"]].drop(columns=["_has"]).copy()
        df_ok = df_ok.sort_values(self.time_col).reset_index(drop=True)

        self._df[key] = df_ok
        tps = df_ok[self.time_col].tolist()
        self._tps[key] = tps
        self._minmax[key] = (min(tps), max(tps)) if tps else (None, None)
        return df_ok

    def tps_set(self, key: Tuple[str, int]) -> set:
        self.load(key)
        return set(self._tps.get(key, []))

    def minmax(self, key: Tuple[str, int]) -> Tuple[Optional[int], Optional[int]]:
        self.load(key)
        return self._minmax.get(key, (None, None))


# -------------------------
# Tile cache + sheet composition
# -------------------------
def frame_path(frames_dir: str, film_name: str, tp: int, channel_index: int) -> str:
    return os.path.join(frames_dir, f"{film_name}_t_{tp:03d}_c_{channel_index}.tif")


def cache_png_path(cache_img_dir: str, key: Tuple[str, int], tp: int) -> str:
    # key is (film_name, cid)
    _, cid = key
    return os.path.join(cache_img_dir, f"cell_{cid}_t_{tp:03d}.png")


def ensure_tile_for_cell_tp(
    *,
    cid: int,
    tp: int,
    film_name: str,
    frames_dir: str,
    cache_img_dir: str,
    masks: MaskTableCache,
    offsets: Dict[int, int],
    time_col: str,
    mask_col: str,
    pad: int,
    tile_size: int,
    channel_index: int,
    cache_force: bool,
) -> Optional[np.ndarray]:

    png_path = cache_png_path(cache_img_dir, (film_name, cid), tp)
    if (not cache_force) and os.path.isfile(png_path):
        try:
            return to_tile(read_png_gray(png_path), (tile_size, tile_size))
        except Exception:
            pass

    dfm = masks.load((film_name, cid))
    row = dfm[dfm[time_col] == tp]
    if row.empty:
        return None
    rle = row.iloc[0][mask_col]

    fp = frame_path(frames_dir, film_name, tp, channel_index)
    if not os.path.isfile(fp):
        return None

    img = read_tif_gray(fp)
    mask = decode_rle_mask_tracking(rle, img.shape[:2])
    bb = bbox_from_mask(mask)
    if bb is None:
        return None

    crop = crop_with_pad(img, bb, pad=pad)
    tile = to_tile(crop, (tile_size, tile_size))

    try:
        write_png_gray(png_path, tile)
    except Exception as e:
        print(f"[warn] could not save cache {png_path}: {e}")

    return tile


def compose_sheet(
    *,
    visible_cids: List[int],
    a_left: int,
    n_cols: int,
    tile_size: int,
    tile_gap: int,
    film_name: str,
    frames_dir: str,
    cache_img_dir: str,
    masks: MaskTableCache,
    offsets: Dict[int, int],
    time_col: str,
    mask_col: str,
    pad: int,
    channel_index: int,
    cache_force: bool,
) -> Tuple[np.ndarray, dict]:

    R = len(visible_cids)
    C = int(n_cols)
    Ht = int(tile_size)
    Wt = int(tile_size)

    sheet_h = R * Ht + max(0, R - 1) * tile_gap
    sheet_w = C * Wt + max(0, C - 1) * tile_gap
    sheet = np.zeros((sheet_h, sheet_w), dtype=np.uint8)

    aligned_cols = [int(a_left) + j for j in range(C)]

    for i, cid in enumerate(visible_cids):
        off = int(offsets.get(cid, 0))
        tps = masks.tps_set(cid)
        y0 = i * (Ht + tile_gap)

        for j, a in enumerate(aligned_cols):
            tp = int(a - off)
            if tp not in tps:
                continue

            tile = ensure_tile_for_cell_tp(
                cid=cid, tp=tp,
                film_name=film_name,
                frames_dir=frames_dir,
                cache_img_dir=cache_img_dir,
                masks=masks,
                offsets=offsets,
                time_col=time_col,
                mask_col=mask_col,
                pad=pad,
                tile_size=tile_size,
                channel_index=channel_index,
                cache_force=cache_force,
            )
            if tile is None:
                continue

            x0 = j * (Wt + tile_gap)
            sheet[y0:y0 + Ht, x0:x0 + Wt] = tile

    meta = {"row_cell_ids": list(visible_cids), "a_left": int(a_left), "aligned_cols": aligned_cols}
    return sheet, meta


# -------------------------
# Save/load GUI state + export CSV
# -------------------------
def load_state(json_path: str, film_name: str, all_cell_ids: List[int]):
    offsets = {cid: 0 for cid in all_cell_ids}
    global_interval = None
    row_start = 0
    a_left = 0
    order = list(all_cell_ids)   # default
    cell_intervals = {}          # NEW: cid -> dict(start_aligned,end_aligned,has_septum)

    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            if st.get("film_name") == film_name:
                # offsets
                off = st.get("offsets", {})
                for k, v in off.items():
                    try:
                        cid = int(k)
                        if cid in offsets:
                            offsets[cid] = int(v)
                    except Exception:
                        pass

                # interval (global)
                gi = st.get("global_interval", None)
                if isinstance(gi, dict) and "G0" in gi and "G1" in gi:
                    try:
                        global_interval = (int(gi["G0"]), int(gi["G1"]))
                    except Exception:
                        global_interval = None

                # view
                view = st.get("view", {})
                row_start = int(view.get("row_start", 0) or 0)
                a_left = int(view.get("a_left", 0) or 0)

                # order
                saved_order = st.get("cell_order", None)
                if isinstance(saved_order, list):
                    cleaned = []
                    seen = set()
                    for x in saved_order:
                        try:
                            cid = int(x)
                        except Exception:
                            continue
                        if cid in offsets and cid not in seen:
                            cleaned.append(cid)
                            seen.add(cid)
                    for cid in all_cell_ids:
                        if cid not in seen:
                            cleaned.append(cid)
                    order = cleaned

                # NEW: per-cell intervals
                ci = st.get("cell_intervals", {})
                if isinstance(ci, dict):
                    for k, v in ci.items():
                        try:
                            cid = int(k)
                        except Exception:
                            continue
                        if cid not in offsets:
                            continue
                        if not isinstance(v, dict):
                            continue
                        cell_intervals[cid] = {
                            "has_septum": bool(v.get("has_septum", False)),
                            "start_aligned": v.get("start_aligned", None),
                            "end_aligned": v.get("end_aligned", None),
                            "white_septum": bool(v.get("white_septum", False)),  # polarity flag
                        }

        except Exception as e:
            print(f"[warn] Could not load JSON resume ({json_path}): {e}")

    return offsets, global_interval, row_start, a_left, order, cell_intervals




def compute_per_cell_results(
    *,
    film_name: str,
    all_cell_ids: List[int],
    masks: MaskTableCache,
    offsets: Dict[int, int],
    G0: int,
    G1: int,
) -> pd.DataFrame:

    rows = []
    for cid in all_cell_ids:
        tp_min, tp_max = masks.minmax(cid)
        off = int(offsets.get(cid, 0))

        if tp_min is None or tp_max is None:
            rows.append({
                "film_name": film_name, "cell_id": cid, "offset": off,
                "tp_min": np.nan, "tp_max": np.nan,
                "G0": int(G0), "G1": int(G1),
                "has_septum": False,
                "start_aligned": np.nan, "end_aligned": np.nan,
                "start_tp": np.nan, "end_tp": np.nan
            })
            continue

        a_min = int(tp_min) + off
        a_max = int(tp_max) + off

        I0 = max(int(G0), a_min)
        I1 = min(int(G1), a_max)
        has = (I0 <= I1)

        if not has:
            start_a = None
            end_a = None
        else:
            # your semantics:
            # - if interval starts before cell exists => no start time
            # - if interval ends after cell disappears => no end time
            start_a = None if (int(G0) < a_min) else int(G0)
            end_a = None if (int(G1) > a_max) else int(G1)

        start_tp = (start_a - off) if start_a is not None else None
        end_tp = (end_a - off) if end_a is not None else None

        rows.append({
            "film_name": film_name, "cell_id": cid, "offset": off,
            "tp_min": int(tp_min), "tp_max": int(tp_max),
            "G0": int(G0), "G1": int(G1),
            "has_septum": bool(has),
            "start_aligned": (np.nan if start_a is None else int(start_a)),
            "end_aligned": (np.nan if end_a is None else int(end_a)),
            "start_tp": (np.nan if start_tp is None else int(start_tp)),
            "end_tp": (np.nan if end_tp is None else int(end_tp)),
        })

    return pd.DataFrame(rows)


def save_state_and_labels(
    *,
    working_dir: str,
    film_name: str,
    json_path: str,
    csv_path: str,
    offsets: Dict[int, int],
    all_cell_ids: List[int],
    global_interval: Optional[Tuple[int, int]],
    view_row_start: int,
    view_a_left: int,
    masks: MaskTableCache,
    cell_order: List[int],
    cell_intervals: Optional[Dict[int, dict]] = None,  # NEW
) -> None:


    st = {
        "working_dir": working_dir,
        "film_name": film_name,
        "cell_order": [int(x) for x in cell_order],
        "offsets": {str(cid): int(offsets.get(cid, 0)) for cid in all_cell_ids},
        "global_interval": (None if global_interval is None else {"G0": int(global_interval[0]), "G1": int(global_interval[1])}),
        "cell_intervals": ({} if not cell_intervals else {
            str(int(cid)): {
                "has_septum": bool(v.get("has_septum", False)),
                "start_aligned": (None if v.get("start_aligned", None) is None else float(v["start_aligned"])),
                "end_aligned": (None if v.get("end_aligned", None) is None else float(v["end_aligned"])),
                "white_septum": bool(v.get("white_septum", False)),  # polarity flag
            }
            for cid, v in cell_intervals.items()
        }),
        "view": {"row_start": int(view_row_start), "a_left": int(view_a_left)},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)

    print(f"[saved] JSON -> {json_path}")

    if global_interval is not None:
        G0, G1 = global_interval
        df_out = compute_per_cell_results(
            film_name=film_name,
            all_cell_ids=all_cell_ids,
            masks=masks,
            offsets=offsets,
            G0=int(G0), G1=int(G1),
        )
        df_out.to_csv(csv_path, index=False)
        print(f"[saved] CSV  -> {csv_path}")
    else:
        print("[note] Global interval not set yet; CSV not written.")
        
def cache_strip_path(cache_img_dir: str, key: Tuple[str, int], tile_size: int, channel_index: int) -> str:
    # include tile_size/channel to avoid collisions across GUI settings
    _, cid = key
    return os.path.join(cache_img_dir, f"cell_{cid}_strip_ts{int(tile_size)}_c{int(channel_index)}.npy")


def ensure_strip_for_cell(
    *,
    cid: int,
    film_name: str,
    frames_dir: str,
    cache_img_dir: str,
    masks: MaskTableCache,
    offsets: Dict[int, int],
    time_col: str,
    mask_col: str,
    pad: int,
    tile_size: int,
    channel_index: int,
    cache_force: bool,
) -> Tuple[np.ndarray, int]:
    """
    Returns (strip, tp0) where:
      strip shape = (tile_size, tile_size * L)
      tp0 = first tp (so index = tp - tp0)
    """
    key = (film_name, cid)
    strip_fp = cache_strip_path(cache_img_dir, key, tile_size=tile_size, channel_index=channel_index)

    tp0, tp1 = masks.minmax(key)
    if tp0 is None or tp1 is None:
        return np.zeros((tile_size, tile_size), np.uint8), 0

    if (not cache_force) and os.path.isfile(strip_fp):
        try:
            strip = np.load(strip_fp, mmap_mode="r")  # mmap keeps RAM low
    
            # validate cache matches current expected tile_size
            if strip.ndim != 2:
                raise ValueError("strip cache not 2D")
            if int(strip.shape[0]) != int(tile_size):
                raise ValueError(f"strip height {strip.shape[0]} != tile_size {tile_size}")
            if (strip.shape[1] % int(tile_size)) != 0:
                raise ValueError("strip width not divisible by tile_size")
    
            return strip, int(tp0)
        except Exception as e:
            # cache is stale / mismatched / corrupted -> rebuild
            try:
                os.remove(strip_fp)
            except Exception:
                pass


    L = int(tp1 - tp0 + 1)
    strip = np.zeros((tile_size, tile_size * L), dtype=np.uint8)

    tps = masks.tps_set(key)  # only those that exist
    for tp in range(int(tp0), int(tp1) + 1):
        if tp not in tps:
            continue
        tile = ensure_tile_for_cell_tp(
            cid=cid, tp=tp,
            film_name=film_name,
            frames_dir=frames_dir,
            cache_img_dir=cache_img_dir,
            masks=masks,
            offsets=offsets,
            time_col=time_col,
            mask_col=mask_col,
            pad=pad,
            tile_size=tile_size,
            channel_index=channel_index,
            cache_force=cache_force,
        )
        if tile is None:
            continue
        j = tp - int(tp0)
        x0 = j * tile_size
        strip[:, x0:x0 + tile_size] = tile

    # Save as .npy for fast reload
    try:
        np.save(strip_fp, strip)
    except Exception as e:
        print(f"[warn] could not save strip {strip_fp}: {e}")

    return strip, int(tp0)

def compose_sheet_from_strips(
    *,
    visible_keys: List[Tuple[str, int]],
    a_left_map: Dict[str, int],  # film_name -> a_left
    n_cols: int,
    tile_size: int,
    tile_gap: int,
    film_paths_map: Dict[str, FilmPaths], # film_name -> paths
    masks: MaskTableCache,
    offsets: Dict[Tuple[str, int], int],
    time_col: str,
    mask_col: str,
    pad: int,
    channel_index: int,
    cache_force: bool,
) -> Tuple[np.ndarray, dict]:

    R = len(visible_keys)
    C = int(n_cols)
    Ht = int(tile_size)
    Wt = int(tile_size)

    sheet_h = R * Ht + max(0, R - 1) * tile_gap
    sheet_w = C * Wt + max(0, C - 1) * tile_gap
    sheet = np.zeros((sheet_h, sheet_w), dtype=np.uint8)

    # aligned_cols depends on the row's film global interval
    # but for simplicity of the 'board', we keep a_left per film
    # and we render each row accordingly.

    for i, key in enumerate(visible_keys):
        fname, cid = key
        paths = film_paths_map[fname]
        a_left = a_left_map.get(fname, 0)
        aligned_cols = [int(a_left) + j for j in range(C)]

        off = int(offsets.get(key, 0))
        strip, tp0 = ensure_strip_for_cell(
            cid=cid,
            film_name=fname,
            frames_dir=paths.frames_dir,
            cache_img_dir=paths.cache_img_dir,
            masks=masks,
            offsets=offsets,
            time_col=time_col,
            mask_col=mask_col,
            pad=pad,
            tile_size=tile_size,
            channel_index=channel_index,
            cache_force=cache_force,
        )
        y0 = i * (Ht + tile_gap)

        # Fill each visible column by slicing strip (no IO)
        for j, a in enumerate(aligned_cols):
            tp = int(a - off)
            idx = tp - int(tp0)
            if idx < 0:
                continue
            x_strip0 = idx * Wt
            x_strip1 = x_strip0 + Wt
            if x_strip1 > strip.shape[1]:
                continue
            tile = strip[:, x_strip0:x_strip1]
            x0 = j * (Wt + tile_gap)
            sheet[y0:y0 + Ht, x0:x0 + Wt] = tile

    meta = {"row_keys": list(visible_keys), "a_left_map": a_left_map}
    return sheet, meta


def load_multi_state(film_paths_map: Dict[str, FilmPaths], film_cells_map: Dict[str, List[int]]):
    """
    Returns (offsets, a_left_map, cell_intervals, global_interval) for all films.
    """
    total_offsets = {}
    a_left_map = {}
    total_cell_intervals = {} # (fname, cid) -> dict
    global_intervals_map = {} # fname -> (G0, G1)

    for fname, paths in film_paths_map.items():
        all_cids = film_cells_map[fname]
        
        offsets = { (fname, cid): 0 for cid in all_cids }
        a_left = 0
        intervals = {}

        if os.path.isfile(paths.json_path):
            try:
                with open(paths.json_path, "r") as f:
                    js = json.load(f)
                
                # offsets
                offs_js = js.get("offsets", {})
                for cid_str, val in offs_js.items():
                    try:
                        cid = int(cid_str)
                        if (fname, cid) in offsets:
                            offsets[(fname, cid)] = int(val)
                    except: pass
                
                # a_left and global_interval map
                gi = js.get("global_interval")
                if gi:
                    a_left = int(gi.get("G0", 0))
                    if "G0" in gi and "G1" in gi:
                        global_intervals_map[fname] = (int(gi["G0"]), int(gi["G1"]))
                
                # cell_intervals
                ci_js = js.get("cell_intervals", {})
                for cid_str, val in ci_js.items():
                    try:
                        cid = int(cid_str)
                        intervals[(fname, cid)] = val
                    except: pass

            except Exception as e:
                print(f"[warn] Could not load JSON for {fname}: {e}")

        total_offsets.update(offsets)
        a_left_map[fname] = a_left
        total_cell_intervals.update(intervals)

    return total_offsets, a_left_map, total_cell_intervals, global_intervals_map


def save_multi_state(
    film_paths_map: Dict[str, FilmPaths],
    film_cells_map: Dict[str, List[int]],
    offsets: Dict[Tuple[str, int], int],
    a_left_map: Dict[str, int],
    cell_intervals: Dict[Tuple[str, int], dict],
    ordered_keys: List[Tuple[str, int]],
    global_intervals_map: Dict[str, Tuple[int, int]],
):
    """
    Splits the unified state and saves to each film's individually.
    """
    for fname, paths in film_paths_map.items():
        all_cids = film_cells_map[fname]
        # Filter unified state for this film
        # Note: cell_intervals strings keys were used in the original JSON structure.
        f_offsets = { str(cid): val for (f, cid), val in offsets.items() if f == fname }
        f_intervals = { str(cid): val for (f, cid), val in cell_intervals.items() if f == fname }
        f_order = [ cid for (f, cid) in ordered_keys if f == fname ]
        f_a_left = a_left_map.get(fname, 0)
        
        f_global_interval = global_intervals_map.get(fname, None)
        gi_dict = {"G0": f_global_interval[0], "G1": f_global_interval[1]} if f_global_interval else {"G0": f_a_left, "G1": f_a_left + 55}

        # Build JSON state
        js = {
            "working_dir": os.path.dirname(paths.film_dir),
            "film_name": fname,
            "cell_order": f_order,
            "offsets": f_offsets,
            "global_interval": gi_dict,
            "cell_intervals": f_intervals,
            "updated_at": datetime.now().isoformat(),
        }

        try:
            os.makedirs(paths.label_dir, exist_ok=True)
            with open(paths.json_path, "w") as f:
                json.dump(js, f, indent=2)

            # Export CSV if possible
            if f_a_left is not None:
                rows = []
                for cid in all_cids:
                    ci = f_intervals.get(str(cid), {})
                    rows.append({
                        "cell_id": cid,
                        "a_left": f_a_left,
                        "start_aligned": ci.get("start_aligned"),
                        "end_aligned": ci.get("end_aligned"),
                        "has": 1 if ci.get("has_septum") else 0,
                        "white_septum": 1 if ci.get("white_septum") else 0,
                    })
                df = pd.DataFrame(rows)
                df.to_csv(paths.csv_path, index=False)
        except Exception as e:
            print(f"[warn] Could not save state for {fname} (drive disconnected?): {e}")        


import re as _re
_FIELD_RE = _re.compile(r"_F(\d+)$")


def export_manifest_from_json_states(
    working_dir: str,
    film_names: List[str],
    out_relpath: str = "training_dataset/manifest.csv",
) -> pd.DataFrame:
    """
    Bridge function: reads the per-film JSON states saved by the upgraded
    multi-film alignment board GUI and regenerates training_dataset/manifest.csv
    in the schema expected by load_manifest() / run_field_sequence().

    Required manifest columns:
        film_name | cell_id | has | offset | field

    Call this immediately after the alignment board QC cells, before
    run_field_sequence(), so the downstream pipeline has an up-to-date manifest.

    Parameters
    ----------
    working_dir  : str  — experiment root (WORKING_DIR)
    film_names   : list — all film names for this experiment
                   (plain names without field suffix, OR full names with _F0 etc.)
    out_relpath  : str  — path relative to working_dir for the output CSV

    Returns
    -------
    pd.DataFrame  — the manifest that was written to disk
    """
    rows = []
    missing = []

    for fname in film_names:
        paths = build_film_paths(working_dir, fname)

        # Derive field label from film name
        mo = _FIELD_RE.search(fname)
        field = f"F{mo.group(1)}" if mo else None

        if not os.path.isfile(paths.json_path):
            missing.append(fname)
            continue

        try:
            with open(paths.json_path, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception as e:
            print(f"[warn] Could not read JSON for {fname}: {e}")
            continue

        offsets_js = js.get("offsets", {})
        intervals_js = js.get("cell_intervals", {})
        # a_left is stored as the G0 of the global_interval
        gi = js.get("global_interval", {})
        a_left = int(gi.get("G0", 0)) if gi else 0

        # Collect all cell IDs seen in offsets (complete population)
        all_cids = set()
        for cid_str in offsets_js:
            try:
                all_cids.add(int(cid_str))
            except ValueError:
                pass
        # Also include any cells that only appear in cell_intervals
        for cid_str in intervals_js:
            try:
                all_cids.add(int(cid_str))
            except ValueError:
                pass

        for cid in sorted(all_cids):
            cid_str = str(cid)
            offset = int(offsets_js.get(cid_str, 0))
            ci = intervals_js.get(cid_str, {})
            has_septum = bool(ci.get("has_septum", False)) if isinstance(ci, dict) else False
            rows.append({
                "film_name": fname,
                "cell_id": cid,
                "offset": offset if has_septum else float("nan"),
                "has": 1 if has_septum else 0,
                "field": field,
            })

    if missing:
        print(f"[warn] No JSON state found for {len(missing)} film(s): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    df = pd.DataFrame(rows, columns=["film_name", "cell_id", "has", "offset", "field"])
    df["cell_id"] = pd.to_numeric(df["cell_id"], errors="coerce").astype("Int64")
    df["has"] = df["has"].fillna(0).astype(int)
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")

    out_path = os.path.join(working_dir, out_relpath)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[manifest] Written {len(df)} rows ({df['has'].sum()} with septum) → {out_path}")

    return df

