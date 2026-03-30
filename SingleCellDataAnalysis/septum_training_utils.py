#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:01:27 2026

@author: user
"""
import tempfile
import pandas as pd
import numpy as np
import os
import csv
from typing import Optional, Tuple, Dict, List

def training_dataset_dir(label_dir: str) -> str:
    return os.path.join(label_dir, "training_dataset")

def _next_npz_path(ds_dir: str) -> str:
    os.makedirs(ds_dir, exist_ok=True)
    # Find next shard id
    existing = []
    for fn in os.listdir(ds_dir):
        if fn.startswith("samples_") and fn.endswith(".npz"):
            try:
                existing.append(int(fn.split("_")[1].split(".")[0]))
            except Exception:
                pass
    nxt = 0 if not existing else (max(existing) + 1)
    return os.path.join(ds_dir, f"samples_{nxt:04d}.npz")

def extract_window_from_strip(
    *,
    strip: np.ndarray,
    tp0: int,
    offset: int,
    a_left: int,
    n_cols: int,
    tile_size: int,
) -> np.ndarray:
    H = int(tile_size)
    W = int(tile_size) * int(n_cols)
    out = np.zeros((H, W), dtype=np.uint8)

    for j in range(int(n_cols)):
        a = int(a_left) + j
        tp = int(a - int(offset))
        idx = int(tp - int(tp0))
        if idx < 0:
            continue
        xs0 = idx * int(tile_size)
        xs1 = xs0 + int(tile_size)
        if xs1 > strip.shape[1]:
            continue
        x0 = j * int(tile_size)
        out[:, x0:x0 + int(tile_size)] = strip[:, xs0:xs1]

    return out

def export_training_windows_npz(
    *,
    ds_dir: str,
    film_name: str,
    samples: list,
) -> Tuple[str, int]:
    """
    samples: list of dict with keys:
      X (uint8 HxW), y_start (float or nan), y_end (float or nan), has (0/1),
      film_name, cell_id, a_left, offset, tp0
    Returns (npz_path, N)
    """
    if not samples:
        return ("", 0)

    npz_path = _next_npz_path(ds_dir)

    X = np.stack([s["X"] for s in samples], axis=0).astype(np.uint8)
    y_start = np.array([s["y_start"] for s in samples], dtype=np.float32)
    y_end   = np.array([s["y_end"] for s in samples], dtype=np.float32)
    has_arr = np.array([s["has"] for s in samples], dtype=np.uint8)

    film_id = np.array([s["film_name"] for s in samples], dtype=object)
    cell_id = np.array([int(s["cell_id"]) for s in samples], dtype=np.int32)
    a_left  = np.array([int(s["a_left"]) for s in samples], dtype=np.int32)
    offset  = np.array([int(s["offset"]) for s in samples], dtype=np.int32)
    tp0     = np.array([int(s["tp0"]) for s in samples], dtype=np.int32)

    np.savez_compressed(
        npz_path,
        X=X,
        y_start=y_start,
        y_end=y_end,
        has=has_arr,
        film_name=film_id,
        cell_id=cell_id,
        a_left=a_left,
        offset=offset,
        tp0=tp0,
    )

    return npz_path, X.shape[0]

def append_dataset_index_csv(
    *,
    ds_dir: str,
    npz_path: str,
    n_added: int,
    film_name: str,
    start_row_idx: int,
):
    """
    Write dataset_index.csv with one row per sample:
      npz_path, idx_in_npz, film_name, cell_id, a_left, offset, tp0, y_start, y_end, has
    start_row_idx: index offset inside this npz (usually 0)
    """
    idx_path = os.path.join(ds_dir, "dataset_index.csv")
    write_header = not os.path.isfile(idx_path)

    # Load lightweight metadata back from npz (no X needed)
    with np.load(npz_path, allow_pickle=True) as z:
        cell_id = z["cell_id"]
        a_left  = z["a_left"]
        offset  = z["offset"]
        tp0     = z["tp0"]
        y_start = z["y_start"]
        y_end   = z["y_end"]
        has_arr = z["has"]

    with open(idx_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",")
        if write_header:
            w.writerow(["npz_path", "idx_in_npz", "film_name", "cell_id", "a_left", "offset", "tp0", "y_start", "y_end", "has"])
        for i in range(n_added):
            j = start_row_idx + i
            w.writerow([
                os.path.basename(npz_path),
                int(j),
                film_name,
                int(cell_id[j]),
                int(a_left[j]),
                int(offset[j]),
                int(tp0[j]),
                float(y_start[j]) if np.isfinite(y_start[j]) else "",
                float(y_end[j]) if np.isfinite(y_end[j]) else "",
                int(has_arr[j]),
            ])



# =========================
# Central training dataset IO (WORKING_DIR/training_dataset)
# =========================

def dataset_root_dir(working_dir: str) -> str:
    return os.path.join(working_dir, "training_dataset")

def dataset_samples_dir(working_dir: str) -> str:
    return os.path.join(dataset_root_dir(working_dir), "samples")

def dataset_manifest_path(working_dir: str) -> str:
    return os.path.join(dataset_root_dir(working_dir), "manifest.csv")

def sample_npz_path(working_dir: str, film_name: str, cell_id: int) -> str:
    # stable filename => overwrite is automatic
    return os.path.join(
        dataset_samples_dir(working_dir),
        f"{film_name}__cell_{int(cell_id):06d}.npz"
    )

def _atomic_write_bytes(dst_path: str, write_fn) -> None:
    """
    Atomic write helper: write to temp file in same directory then os.replace.
    write_fn(tmp_path) should write the file.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    d = os.path.dirname(dst_path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, dst_path)  # atomic rename
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def atomic_save_npz(dst_path: str, **arrays) -> None:
    def _write(tmp_path: str):
        # np.savez_compressed adds ".npz" if tmp_path doesn't end with it
        # so ensure it ends with ".npz"
        tmp_npz = tmp_path if tmp_path.endswith(".npz") else (tmp_path + ".npz")
        np.savez_compressed(tmp_npz, **arrays)
        if tmp_npz != tmp_path:
            os.replace(tmp_npz, tmp_path)
    _atomic_write_bytes(dst_path, _write)

def upsert_manifest_rows(working_dir: str, new_rows: pd.DataFrame) -> None:
    """
    Enforces uniqueness by (film_name, cell_id). New rows overwrite old rows.
    """
    root = dataset_root_dir(working_dir)
    os.makedirs(root, exist_ok=True)
    manifest_fp = dataset_manifest_path(working_dir)

    # Read existing (if any)
    if os.path.isfile(manifest_fp):
        old = pd.read_csv(manifest_fp)
    else:
        old = pd.DataFrame()

    # Normalize columns
    if "film_name" not in new_rows.columns or "cell_id" not in new_rows.columns:
        raise ValueError("new_rows must contain film_name and cell_id")

    # If old empty, just write
    if old.empty:
        out = new_rows.copy()
    else:
        # Drop rows with same keys
        old["cell_id"] = pd.to_numeric(old["cell_id"], errors="coerce").astype("Int64")
        new_rows2 = new_rows.copy()
        new_rows2["cell_id"] = pd.to_numeric(new_rows2["cell_id"], errors="coerce").astype("Int64")

        old_idx = old.set_index(["film_name", "cell_id"]).index
        new_idx = new_rows2.set_index(["film_name", "cell_id"]).index
        keep_mask = ~old_idx.isin(new_idx)

        out = pd.concat([old.loc[keep_mask].copy(), new_rows2], ignore_index=True)

    # Stable order
    out = out.sort_values(["film_name", "cell_id"], kind="stable").reset_index(drop=True)

    def _write_csv(tmp_path: str):
        out.to_csv(tmp_path, index=False)

    _atomic_write_bytes(manifest_fp, _write_csv)



def export_cell_training_sample(
    *,
    working_dir: str,
    film_name: str,
    cell_id: int,
    strip: np.ndarray,   # (tile_size, tile_size * L)
    tp0: int,            # first raw tp index of strip
    offset: int,         # aligned = tp + offset
    start_idx: int,
    end_idx: int,
    label_source: str = "none",               # "cell" | "global" | "none"
    start_aligned: Optional[float] = None,    # audit only
    end_aligned: Optional[float] = None,      # audit only
) -> str:
    """
    Writes:
      WORKING_DIR/training_dataset/samples/<film>__cell_<id>.npz

    Overwrites by design.
    Also upserts one row into manifest.csv.
    Labels are stored in STRIP INDEX SPACE (tile indices).

    Conventions:
      - start_idx/end_idx are independent; each can be -1 (missing)
      - has is derived:
            has = 1 if (start_idx>=0 or end_idx>=0) else 0
      - If has==0, both indices are forced to -1 (defensive)
    """
    npz_fp = sample_npz_path(working_dir, film_name, cell_id)

    # Normalize indices
    s_idx = -1 if start_idx is None else int(start_idx)
    e_idx = -1 if end_idx is None else int(end_idx)

    # Normalize strip + infer tile_size/L
    strip = np.asarray(strip, dtype=np.uint8)
    if strip.ndim != 2:
        raise ValueError(f"strip must be 2D, got shape={strip.shape}")

    tile_size = int(strip.shape[0])  # assumes square tiles; height = tile_size
    if tile_size <= 0:
        raise ValueError(f"invalid tile_size inferred from strip height: {tile_size}")

    if strip.shape[1] % tile_size != 0:
        raise ValueError(
            f"strip width {strip.shape[1]} not divisible by inferred tile_size {tile_size}"
        )
    L = int(strip.shape[1] // tile_size)

    # Clip any provided indices into [0, L-1], else set to -1 if out of range
    if s_idx >= 0 and not (0 <= s_idx < L):
        s_idx = -1
    if e_idx >= 0 and not (0 <= e_idx < L):
        e_idx = -1

    # Only swap if BOTH exist
    if s_idx >= 0 and e_idx >= 0 and e_idx < s_idx:
        s_idx, e_idx = e_idx, s_idx

    # Derive has from indices
    has_i = int((s_idx >= 0) or (e_idx >= 0))
    if has_i == 0:
        s_idx, e_idx = -1, -1
        if not label_source:
            label_source = "none"

    atomic_save_npz(
        npz_fp,
        strip=strip,
        film_name=np.array([film_name], dtype=object),
        cell_id=np.array([int(cell_id)], dtype=np.int32),
        tp0=np.array([int(tp0)], dtype=np.int32),
        offset=np.array([int(offset)], dtype=np.int32),
        has=np.array([has_i], dtype=np.int8),
        start_idx=np.array([s_idx], dtype=np.int32),
        end_idx=np.array([e_idx], dtype=np.int32),
        label_source=np.array([str(label_source)], dtype=object),
        start_aligned=np.array(
            [np.nan if start_aligned is None else float(start_aligned)],
            dtype=np.float32,
        ),
        end_aligned=np.array(
            [np.nan if end_aligned is None else float(end_aligned)],
            dtype=np.float32,
        ),
    )

    row = pd.DataFrame([{
        "npz_path": os.path.relpath(npz_fp, start=working_dir),
        "film_name": film_name,
        "cell_id": int(cell_id),
        "tp0": int(tp0),
        "offset": int(offset),
        "has": has_i,
        "start_idx": int(s_idx),
        "end_idx": int(e_idx),
        "label_source": str(label_source),
        "start_aligned": ("" if start_aligned is None else float(start_aligned)),
        "end_aligned": ("" if end_aligned is None else float(end_aligned)),
        "L": int(L),
        "tile_size": int(tile_size),
    }])
    upsert_manifest_rows(working_dir, row)

    return npz_fp


def export_film_training_dataset(
    *,
    working_dir: str,
    film_name: str,
    all_cell_ids: List[int],
    offsets: Dict[int, int],
    global_interval: Optional[Tuple[int, int]],
    cell_intervals: Optional[Dict[int, dict]],
    # strip builder deps (same as GUI uses)
    frames_dir: str,
    cache_img_dir: str,
    masks,                 # MaskTableCache
    time_col: str,
    mask_col: str,
    pad: int,
    tile_size: int,
    channel_index: int,
    cache_force: bool = False,
    verbose_every: int = 200,
) -> Dict[str, int]:
    """
    Export per-cell training samples for an entire film using the same semantics as the GUI:
      - per-cell interval wins if present
      - else global_interval bootstrap if present
      - else skip
    Converts aligned endpoints -> strip indices and calls export_cell_training_sample().

    Returns counts: written/pos/neg/skipped.
    """
    from .septum_gui_utils import ensure_strip_for_cell  # keep GUI small; reuse canonical strip builder

    if global_interval is None and not cell_intervals:
        print("[export] No global interval and no per-cell intervals; nothing to export.")
        return {"written": 0, "pos": 0, "neg": 0, "skipped": 0}

    cell_intervals = cell_intervals or {}

    def aligned_to_idx(a_val: Optional[int], *, tp0: int, off: int, L: int) -> int:
        if a_val is None:
            return -1
        tp = int(a_val) - int(off)
        idx = int(tp) - int(tp0)
        return idx if (0 <= idx < int(L)) else -1

    n_written = n_pos = n_neg = n_skipped = 0

    for k, cid in enumerate(all_cell_ids):
        cid = int(cid)
        off = int(offsets.get(cid, 0))

        label_source = "none"
        start_a = None
        end_a = None

        # (1) per-cell label
        if cid in cell_intervals:
            v = cell_intervals[cid]
            label_source = "cell"
            if bool(v.get("has_septum", False)) is False:
                start_a = None
                end_a = None
            else:
                s = v.get("start_aligned", None)
                e = v.get("end_aligned", None)
                start_a = None if s is None else int(round(float(s)))
                end_a   = None if e is None else int(round(float(e)))

        # (2) global bootstrap
        elif global_interval is not None:
            label_source = "global"
            G0, G1 = map(int, global_interval)

            tp_min, tp_max = masks.minmax(cid)
            if tp_min is None or tp_max is None:
                n_skipped += 1
                continue

            a_min = int(tp_min) + off
            a_max = int(tp_max) + off

            if (a_max < G0) or (a_min > G1):
                start_a = None
                end_a = None
            else:
                start_a = int(G0) if (a_min <= G0) else None
                end_a   = int(G1) if (a_max >= G1) else None

        else:
            n_skipped += 1
            continue

        # build/load strip
        strip, tp0 = ensure_strip_for_cell(
            cid=cid,
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

        # skip degenerate/no-data cells
        tp_min, tp_max = masks.minmax(cid)
        if tp_min is None or tp_max is None:
            n_skipped += 1
            continue

        strip = np.asarray(strip)
        if strip.ndim != 2:
            n_skipped += 1
            continue
        if strip.shape[1] % int(tile_size) != 0:
            n_skipped += 1
            continue
        L = int(strip.shape[1] // int(tile_size))

        start_idx = aligned_to_idx(start_a, tp0=int(tp0), off=off, L=L)
        end_idx   = aligned_to_idx(end_a,   tp0=int(tp0), off=off, L=L)

        # swap only if both exist
        if start_idx >= 0 and end_idx >= 0 and end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        has = 1 if (start_idx >= 0 or end_idx >= 0) else 0
        if has == 1:
            n_pos += 1
        else:
            n_neg += 1

        export_cell_training_sample(
            working_dir=working_dir,
            film_name=film_name,
            cell_id=cid,
            strip=strip,
            tp0=int(tp0),
            offset=int(off),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            label_source=label_source,
            start_aligned=(None if start_a is None else float(start_a)),
            end_aligned=(None if end_a is None else float(end_a)),
        )
        n_written += 1

        if verbose_every and (n_written <= 10 or n_written % int(verbose_every) == 0):
            print(f"[export] {n_written}: cell {cid} has={has} s={start_idx} e={end_idx} src={label_source}")

    print(f"[export] DONE {film_name}: wrote {n_written} (pos={n_pos}, neg={n_neg}, skipped={n_skipped})")
    return {"written": n_written, "pos": n_pos, "neg": n_neg, "skipped": n_skipped}