#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:47:37 2026

@author: user
"""

import os
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- Optional OpenCV (cv2) ----
# We DO NOT hard-require cv2 at import time, because some helpers (e.g. pairing/global IDs)
# don't need it.
_CV2 = None
_CV2_IMPORT_ERROR = None
try:
    import cv2 as _cv2
    _CV2 = _cv2
except Exception as e:
    _CV2_IMPORT_ERROR = e

def _require_cv2():
    """
    Call this inside functions that actually need cv2.
    """
    if _CV2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for population movie / GUI functions.\n"
            "Install in your spyder-env:\n"
            "  conda activate spyder-env\n"
            "  conda install -c conda-forge opencv\n"
            "or:\n"
            "  conda activate spyder-env\n"
            "  python -m pip install opencv-python\n"
        ) from _CV2_IMPORT_ERROR
    return _CV2


# -----------------------------
# 1) Pairing -> Global IDs
# -----------------------------

def build_global_id_maps_from_pairings(
    field_seq: List[Tuple[str, str]],
    pair_mappings: List[Dict[str, Any]],
    anchor_film: Optional[str] = None,
) -> Dict[str, Dict[int, int]]:
    """
    Build film->(cell_id -> global_id) using pairing outputs across a sequence.

    field_seq example:
      [("gfp","GFP1"), ("bf","BF1"), ("gfp","GFP2"), ("bf","BF2"), ("gfp","GFP3")]

    pair_mappings: list of dicts, one per adjacent pair.
      Each item should contain:
        - "gfp_film"
        - "bf_film"
        - "mapping_gfp_to_bf": {gfp_id:int -> bf_id:int}
      (This matches what you put in map_out in run_field_sequence.)

    anchor_film:
      If provided, use IDs from that film (prefer BF1) as global_id when possible.
    """

    # --- union-find over nodes (film, cell_id) ---
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def find(x: Tuple[str, int]) -> Tuple[str, int]:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: Tuple[str, int], b: Tuple[str, int]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Add edges from each mapping GFP->BF
    for item in pair_mappings:
        gfp_film = item["gfp_film"]
        bf_film = item["bf_film"]
        mapping = item["mapping_gfp_to_bf"]
        for gfp_id, bf_id in mapping.items():
            if gfp_id is None or bf_id is None:
                continue
            union((gfp_film, int(gfp_id)), (bf_film, int(bf_id)))

    # Ensure all films/cells seen in mappings are registered
    for item in pair_mappings:
        gfp_film = item["gfp_film"]
        bf_film = item["bf_film"]
        mapping = item["mapping_gfp_to_bf"]
        for gfp_id, bf_id in mapping.items():
            find((gfp_film, int(gfp_id)))
            find((bf_film, int(bf_id)))

    # Group nodes by component root
    comps: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for node in list(parent.keys()):
        r = find(node)
        comps.setdefault(r, []).append(node)

    # Choose global_id for each component:
    # prefer anchor_film cell_id if present; else smallest cell_id
    comp_global: Dict[Tuple[str, int], int] = {}
    for root, nodes in comps.items():
        gid = None
        if anchor_film is not None:
            anchor_ids = [cid for (film, cid) in nodes if film == anchor_film]
            if anchor_ids:
                gid = int(sorted(anchor_ids)[0])
        if gid is None:
            gid = int(sorted([cid for (_, cid) in nodes])[0])
        comp_global[root] = gid

    # Build film->cell_id->global_id
    film_map: Dict[str, Dict[int, int]] = {}
    for root, nodes in comps.items():
        gid = comp_global[root]
        for film, cid in nodes:
            film_map.setdefault(film, {})[int(cid)] = int(gid)

    # Ensure every film in seq exists (even if empty)
    for _, film in field_seq:
        film_map.setdefault(film, {})

    return film_map


# -----------------------------
# 2) Frames + masks loader
# -----------------------------

TIME_RE = re.compile(r"_t_(\d+)_")

def find_time_from_name(name: str) -> Optional[int]:
    m = TIME_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def discover_frames(frames_dir: Path) -> List[Path]:
    candidates = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
        candidates.extend(frames_dir.glob(ext))
    frames = [p for p in candidates if find_time_from_name(p.name) is not None]
    frames.sort(key=lambda p: find_time_from_name(p.name))
    return frames

def normalize_to_uint8(img, gmin, gmax):
    imgf = img.astype(np.float32, copy=False)
    if imgf.max() <= 1.0:
        imgf = imgf * 255.0
    imgf = np.clip((imgf - gmin) * (255.0 / max(1e-6, (gmax - gmin))), 0.0, 255.0)
    return imgf.astype(np.uint8)

def compute_global_percentile_range(frame_paths, p_low=1.0, p_high=99.5, sample_stride=10):
    cv2 = _require_cv2()
    samples = []
    for p in frame_paths:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            continue
        gray = arr if arr.ndim == 2 else arr[..., 0]
        samples.extend(gray.ravel()[::sample_stride])
    if not samples:
        return 0.0, 255.0
    samples = np.asarray(samples)
    gmin = float(np.percentile(samples, p_low))
    gmax = float(np.percentile(samples, p_high))
    if gmax <= gmin:
        gmin, gmax = float(samples.min()), float(samples.max())
        if gmax <= gmin:
            gmin, gmax = 0.0, 255.0
    return gmin, gmax

def read_image_graceful(path: Path, global_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    cv2 = _require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    if global_range is not None:
        gmin, gmax = global_range
        img = normalize_to_uint8(img, gmin, gmax)
    else:
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        elif img.dtype in (np.float32, np.float64):
            m, M = float(np.nanmin(img)), float(np.nanmax(img))
            img = (img * 255.0 if M <= 1.0 else img).clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported image shape {img.shape} for {path}")

def id_to_color(seed_id: int) -> Tuple[int, int, int]:
    cv2 = _require_cv2()
    rng = abs(hash(int(seed_id)))
    h = rng % 180
    s = 200 + (rng // 180) % 56
    v = 220 + (rng // (180 * 56)) % 36
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_label(img: np.ndarray, text: str, x: int, y: int):
    cv2 = _require_cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

CELL_MASK_RE = re.compile(r"cell_(\d+)_masks", re.IGNORECASE)

def parse_cell_id_from_filename(path: Path) -> Optional[int]:
    m = CELL_MASK_RE.search(path.stem)
    return int(m.group(1)) if m else None

def _detect_delimiter(sample_line: str) -> str:
    return "\t" if "\t" in sample_line and "," not in sample_line else ","

def load_cell_masks_csv(path: Path, rle_key: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns {t: {"pairs": [(start,length),...], "size":[H,W]}}
    Expects rle stored as a space-separated "start length start length ..." string.
    """
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", newline="") as f:
        first = f.readline()
        if not first:
            return out
        delim = _detect_delimiter(first)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)

        fallback_keys = [rle_key, "rle", "rle_bf", "rle_gfp"]
        seen = set()
        ordered_keys = [k for k in fallback_keys if not (k in seen or seen.add(k))]

        for row in reader:
            tp = row.get("time_point") or row.get("t") or row.get("time")
            if not tp:
                continue
            try:
                t = int(tp)
            except Exception:
                continue

            H = int(row.get("height") or row.get("H") or 0)
            W = int(row.get("width")  or row.get("W") or 0)

            rle_str = ""
            for k in ordered_keys:
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

def decode_pairs_rle(pairs_obj: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """
    Decode {"pairs":[(start,length),...]} into boolean mask, assuming column-major ("F") indexing.
    """
    total = int(H) * int(W)
    arr = np.zeros(total, dtype=np.uint8)
    pairs = pairs_obj.get("pairs", [])
    for start, length in pairs:
        s = int(start)
        L = max(0, int(length))
        if s >= total or L == 0:
            continue
        e = min(s + L, total)
        arr[s:e] = 1
    return arr.reshape((int(H), int(W)), order="F").astype(bool)


# -----------------------------
# 3) Overlay + GUI review
# -----------------------------

def overlay_global_ids_on_frame(
    frame_bgr: np.ndarray,
    time_t: int,
    cell_maps: List[Tuple[int, Dict[int, Dict[str, Any]]]],
    cell_to_global: Dict[int, int],
    alpha: float = 0.35,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Overlays masks and draws global_id labels.
    Returns (frame_out, id_image) where id_image is an int32 image of global_id per pixel
    (used for mouse picking). Pixels with no cell are -1.
    """
    cv2 = _require_cv2()
    H, W = frame_bgr.shape[:2]
    overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
    id_img = np.full((H, W), -1, dtype=np.int32)

    for cell_id, time_to_obj in cell_maps:
        obj = time_to_obj.get(time_t)
        if obj is None:
            continue
        size = obj.get("size", [H, W])
        h2, w2 = int(size[0] or H), int(size[1] or W)
        if h2 != H or w2 != W:
            # ignore size mismatch; decode using frame size
            pass

        try:
            mask = decode_pairs_rle(obj, H, W)
        except Exception:
            continue
        if not mask.any():
            continue

        gid = cell_to_global.get(int(cell_id), None)
        if gid is None:
            continue

        overlay[mask] = id_to_color(gid)
        id_img[mask] = int(gid)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            draw_label(frame_bgr, str(gid), cx, max(12, cy - 5))

    blended = cv2.addWeighted(overlay, float(alpha), frame_bgr, 1.0, 0.0)
    return blended, id_img


def review_population_movie_gui(
    working_dir: str,
    film_name: str,
    cell_to_global: Dict[int, int],
    rle_key: str,
    frames_dirname: Optional[str] = None,
    tracked_dirname: Optional[str] = None,
    alpha: float = 0.35,
    fps: int = 10,
    save_csv: Optional[str] = None,
) -> str:
    """
    GUI reviewer for one film.

    Controls:
      - SPACE: play/pause
      - g: mark hovered global_id as GOOD
      - b: mark hovered global_id as BAD
      - u: mark hovered global_id as UNKNOWN
      - s: save now
      - q or ESC: quit (autosave)

    Returns path to saved CSV.
    """
    cv2 = _require_cv2()
    base = Path(working_dir) / film_name
    frames_dir = base / (frames_dirname or f"Frames_{film_name}")
    cells_dir  = base / (tracked_dirname or f"TrackedCells_{film_name}")

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"TrackedCells dir not found: {cells_dir}")

    frames = discover_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"No frames discovered in {frames_dir}")

    gmin, gmax = compute_global_percentile_range(frames, p_low=1.0, p_high=99.5, sample_stride=10)
    global_range = (gmin, gmax)

    # load all cell mask files
    cell_files = sorted([p for p in cells_dir.iterdir() if p.is_file() and "cell_" in p.name and "masks" in p.name])
    cell_maps: List[Tuple[int, Dict[int, Dict[str, Any]]]] = []
    for cf in cell_files:
        cid = parse_cell_id_from_filename(cf)
        if cid is None:
            continue
        if int(cid) not in cell_to_global:
            continue  # not in mapping => skip (or remove this if you want to display unmapped)
        if cf.suffix.lower() in (".csv", ".tsv"):
            tmap = load_cell_masks_csv(cf, rle_key=rle_key)
            if tmap:
                cell_maps.append((int(cid), tmap))

    if not cell_maps:
        raise RuntimeError("No cell masks loaded (maybe rle_key mismatch or mapping empty).")

    # statuses
    status_map: Dict[int, str] = {}  # global_id -> {"good","bad","unknown"}

    # mouse hover state
    hover_gid = {"val": -1}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            id_img = param.get("id_img", None)
            if id_img is None:
                hover_gid["val"] = -1
            else:
                if 0 <= y < id_img.shape[0] and 0 <= x < id_img.shape[1]:
                    hover_gid["val"] = int(id_img[y, x])
                else:
                    hover_gid["val"] = -1

    win = f"review: {film_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # set later each frame
    mouse_param = {"id_img": None}
    cv2.setMouseCallback(win, on_mouse, mouse_param)

    # playback
    playing = True
    delay_ms = max(1, int(1000 / max(1, fps)))

    # determine output CSV path
    if save_csv is None:
        save_csv = str((Path(working_dir) / "pipeline_outputs" / "tracking_review" / f"{film_name}__review.csv"))
    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)

    def save_now():
        # best effort metadata
        rows = []
        for gid, st in sorted(status_map.items(), key=lambda kv: kv[0]):
            rows.append({"global_id": int(gid), "status": st, "film_name": film_name})
        import pandas as pd
        pd.DataFrame(rows).to_csv(save_csv, index=False)

    # main loop
    i = 0
    while True:
        i = max(0, min(i, len(frames) - 1))
        fp = frames[i]
        t = find_time_from_name(fp.name)
        if t is None:
            t = i

        frame = read_image_graceful(fp, global_range=global_range)

        frame_out, id_img = overlay_global_ids_on_frame(
            frame,
            time_t=int(t),
            cell_maps=cell_maps,
            cell_to_global=cell_to_global,
            alpha=alpha,
        )

        mouse_param["id_img"] = id_img

        # HUD
        gid = hover_gid["val"]
        st = status_map.get(gid, "unknown") if gid != -1 else "none"
        hud = f"t={t}  frame={i+1}/{len(frames)}  hover_gid={gid}  status={st}  [SPACE play/pause | g good | b bad | u unk | s save | q quit]"
        cv2.putText(frame_out, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_out, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow(win, frame_out)
        key = cv2.waitKey(delay_ms if playing else 0) & 0xFF

        if key in (27, ord("q")):  # ESC / q
            save_now()
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("s"):
            save_now()
        elif key == ord("g"):
            if gid != -1:
                status_map[gid] = "good"
        elif key == ord("b"):
            if gid != -1:
                status_map[gid] = "bad"
        elif key == ord("u"):
            if gid != -1:
                status_map[gid] = "unknown"
        elif key == ord("a"):   # step back (handy)
            i -= 1
        elif key == ord("d"):   # step forward
            i += 1
        else:
            if playing:
                i += 1

        if i >= len(frames):
            i = len(frames) - 1
            playing = False

    cv2.destroyWindow(win)
    return save_csv