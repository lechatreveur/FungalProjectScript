#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:07:40 2026

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

TIME_RE = re.compile(r"_t_(\d+)_")

def _find_time_from_name(name: str) -> Optional[int]:
    m = TIME_RE.search(name)
    return int(m.group(1)) if m else None

def discover_frames(frames_dir: Path) -> List[Path]:
    frames = []
    for ext in ("*.tif","*.tiff","*.png","*.jpg","*.jpeg"):
        frames.extend(frames_dir.glob(ext))
    frames = [p for p in frames if _find_time_from_name(p.name) is not None]
    frames.sort(key=lambda p: _find_time_from_name(p.name))
    return frames

def read_frame(path: Path) -> np.ndarray:
    # matplotlib can read tif/png/jpg via imread in most setups
    img = plt.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        # normalize float [0..1] or uint16 -> uint8
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        else:
            img = (img.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return img

def _detect_delimiter(sample_line: str) -> str:
    return "\t" if "\t" in sample_line and "," not in sample_line else ","

def load_cell_masks_csv(cell_masks_csv: Path, rle_key: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns: {t: {"pairs":[(start,length),...], "size":[H,W]}}
    Expects columns: time_point, height, width, and rle column.
    Your rle string format: "start length start length ..." (space-separated ints)
    """
    out = {}
    with open(cell_masks_csv, "r", newline="") as f:
        first = f.readline()
        if not first:
            return out
        delim = _detect_delimiter(first)
        f.seek(0)
        rdr = csv.DictReader(f, delimiter=delim)

        # allow fallbacks
        rle_keys = [rle_key, "rle", "rle_bf", "rle_gfp"]
        for row in rdr:
            tp = row.get("time_point") or row.get("t") or row.get("time")
            if tp is None or tp == "":
                continue
            try:
                t = int(tp)
            except Exception:
                continue

            H = int(row.get("height") or 0)
            W = int(row.get("width") or 0)

            rle_str = ""
            for k in rle_keys:
                v = row.get(k)
                if v and v.strip():
                    rle_str = v.strip()
                    break
            if not rle_str:
                continue

            try:
                nums = [int(x) for x in rle_str.split()]
            except Exception:
                continue
            if len(nums) % 2 != 0:
                continue
            pairs = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
            out[t] = {"pairs": pairs, "size": [H, W]}
    return out

def decode_pairs_to_mask(pairs_obj: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """
    pairs are in flat column-major indexing (order='F'), consistent with your older code.
    """
    total = H * W
    arr = np.zeros(total, dtype=np.uint8)
    pairs = pairs_obj.get("pairs", [])
    for s, L in pairs:
        s = int(s); L = int(L)
        if L <= 0 or s >= total:
            continue
        e = min(s + L, total)
        arr[s:e] = 1
    return arr.reshape((H, W), order="F").astype(bool)

def parse_cell_id_from_masks_filename(path: Path) -> Optional[int]:
    m = re.search(r"cell_(\d+)_masks", path.stem)
    return int(m.group(1)) if m else None

def build_cell_maps_from_trackedcells(trackedcells_dir: Path, rle_key: str) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Returns:
      cell_maps[cell_id][t] -> {"pairs":..., "size":[H,W]}
    """
    cell_maps = {}
    files = sorted([p for p in trackedcells_dir.iterdir()
                    if p.is_file() and "cell_" in p.name and "masks" in p.name and p.suffix.lower() in (".csv",".tsv")])
    for fp in files:
        cid = parse_cell_id_from_masks_filename(fp)
        if cid is None:
            continue
        time_to = load_cell_masks_csv(fp, rle_key=rle_key)
        if time_to:
            cell_maps[cid] = time_to
    return cell_maps

def overlay_masks_and_labels(
    img_rgb: np.ndarray,
    time_t: int,
    cell_maps: Dict[int, Dict[int, Dict[str, Any]]],
    local_to_global: Dict[int, int],
    alpha: float = 0.35,
) -> Tuple[np.ndarray, Dict[int, Tuple[float, float]]]:
    """
    Returns:
      out_img, centroids_global -> {global_id: (cx, cy)} for click selection
    """
    out = img_rgb.copy()
    H, W = out.shape[:2]
    overlay = np.zeros_like(out, dtype=np.uint8)
    centroids = {}

    for local_id, time_to in cell_maps.items():
        rec = time_to.get(time_t)
        if rec is None:
            continue
        size = rec.get("size", [H, W])
        h2, w2 = int(size[0]), int(size[1])
        # trust frame size if mismatch
        mask = decode_pairs_to_mask(rec, H, W) if (h2 != H or w2 != W) else decode_pairs_to_mask(rec, h2, w2)
        if not mask.any():
            continue

        gid = local_to_global.get(int(local_id), None)
        if gid is None:
            continue

        # deterministic pseudo-color
        rng = abs(hash(int(gid)))
        color = np.array([(rng % 255), ((rng//7) % 255), ((rng//49) % 255)], dtype=np.uint8)

        overlay[mask] = color

        ys, xs = np.where(mask)
        cx = float(xs.mean()); cy = float(ys.mean())
        centroids[int(gid)] = (cx, cy)

    # alpha blend
    out = (alpha * overlay.astype(np.float32) + (1.0 - alpha) * out.astype(np.float32)).clip(0,255).astype(np.uint8)

    # draw labels (matplotlib text is drawn in GUI layer; we return centroids here)
    return out, centroids

def review_population_movie_gui(
    working_dir: str,
    film_name: str,
    frames_subdir: Optional[str],
    trackedcells_subdir: Optional[str],
    local_to_global: Dict[int, int],
    rle_key: str,
    out_qc_csv: str,
    alpha: float = 0.35,
):
    """
    Matplotlib player:
      - Click near a cell to select that global_id (based on centroid)
      - Press:
          g = good, b = bad, u = unsure, d = delete label
          space = play/pause
          left/right arrows = step
      - Saves QC table to out_qc_csv whenever you press "Save"
    QC schema:
      global_id, status, film_name
    """
    base = Path(working_dir) / film_name
    frames_dir = base / (frames_subdir or f"Frames_{film_name}")
    tracked_dir = base / (trackedcells_subdir or f"TrackedCells_{film_name}")

    frames = discover_frames(frames_dir)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    cell_maps = build_cell_maps_from_trackedcells(tracked_dir, rle_key=rle_key)
    if not cell_maps:
        raise RuntimeError(f"No cell_*_masks.csv found / usable in {tracked_dir}")

    times = [ _find_time_from_name(p.name) for p in frames ]
    times = [t for t in times if t is not None]

    qc = {}  # global_id -> status
    selected_gid = [None]
    playing = [True]
    idx = [0]
    last_centroids = [{}]

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.12)

    img0 = read_frame(frames[0])
    disp = ax.imshow(img0)
    ax.set_title(f"{film_name}   (click cell; g/b/u to label; space play/pause)")
    ax.axis("off")

    # Save button
    ax_save = plt.axes([0.75, 0.02, 0.20, 0.06])
    btn_save = Button(ax_save, "Save QC")

    def save_qc(_=None):
        Path(out_qc_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_qc_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["global_id", "status", "film_name"])
            for gid, status in sorted(qc.items(), key=lambda x: int(x[0])):
                w.writerow([int(gid), status, film_name])
        print(f"[QC saved] {out_qc_csv}  n={len(qc)}")

    btn_save.on_clicked(save_qc)

    def update_frame():
        t = times[idx[0]]
        img = read_frame(frames[idx[0]])
        out_img, centroids = overlay_masks_and_labels(
            img, t,
            cell_maps=cell_maps,
            local_to_global=local_to_global,
            alpha=alpha
        )
        last_centroids[0] = centroids
        disp.set_data(out_img)

        gid = selected_gid[0]
        status = qc.get(gid, "") if gid is not None else ""
        ax.set_xlabel(f"t={t}   selected_global_id={gid}   status={status}")
        fig.canvas.draw_idle()

    def nearest_gid(x, y, centroids, max_dist=40.0):
        best = None
        bestd = 1e18
        for gid, (cx, cy) in centroids.items():
            d = (cx - x)**2 + (cy - y)**2
            if d < bestd:
                bestd = d
                best = gid
        if best is None:
            return None
        if np.sqrt(bestd) > max_dist:
            return None
        return best

    def on_click(event):
        if event.inaxes != ax:
            return
        centroids = last_centroids[0]
        gid = nearest_gid(event.xdata, event.ydata, centroids)
        selected_gid[0] = gid
        update_frame()

    def on_key(event):
        k = event.key
        if k == " ":
            playing[0] = not playing[0]
        elif k in ("right", "d"):
            idx[0] = min(idx[0] + 1, len(frames) - 1)
            update_frame()
        elif k == "left":
            idx[0] = max(idx[0] - 1, 0)
            update_frame()
        elif k in ("g", "b", "u"):
            gid = selected_gid[0]
            if gid is not None:
                qc[int(gid)] = {"g":"good","b":"bad","u":"unsure"}[k]
                update_frame()
        elif k == "backspace":
            gid = selected_gid[0]
            if gid is not None and int(gid) in qc:
                del qc[int(gid)]
                update_frame()
        elif k == "s":
            save_qc()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update_frame()

    # simple timer playback
    timer = fig.canvas.new_timer(interval=80)  # ~12.5 fps
    def tick():
        if playing[0]:
            idx[0] = (idx[0] + 1) % len(frames)
            update_frame()
    timer.add_callback(tick)
    timer.start()

    plt.show()

    # autosave on exit
    save_qc()
    return qc