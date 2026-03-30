
#!/usr/bin/env python3
"""
Population-level single-cell tracking movie generator

Reads raw frames and per-cell mask files (with RLE under key `rle_bf`) and overlays
semi-transparent colored masks + cell IDs for each timeframe, then writes an MP4.

Assumptions / supported mask formats for each cell file:
- JSON, PKL, or NPZ files containing, for each time t, an object with a key "rle_bf".
- The value for "rle_bf" can be one of:
  1) COCO-style RLE dict: {"counts": [ints] OR "compressed string", "size": [H, W]}
     - If "counts" is a string (COCO-compressed), we try to use pycocotools if available.
     - If pycocotools is not installed, we fall back to a pure-Python decoder for the string.
  2) A plain list of RLE counts (COCO order: alternating zeros/ones runs).
  3) A dict with "idxs" or "indices" or "pixels" encoding pixel indices/coords.
     - idxs/indices: flat indices (column-major expected) of 1-valued pixels.
     - pixels: list of [row, col] pairs (0-based).
- The file structure can be one of:
  A) { "118": {"rle_bf": ...}, "119": {"rle_bf": ...}, ... }
  B) { "timepoints": { "118": {"rle_bf": ...}, ... } }
  C) [ {"t": 118, "rle_bf": ...}, {"t": 119, "rle_bf": ...}, ... ]

Frame discovery:
- Frames are read from a directory like "Frames_*" with filenames containing "t_{t}_"
  For example: A14_1TP1_BF_F1_t_118_c_0.tif
- You can customize the frame glob pattern with --frame_glob (use "{t}" placeholder).

Usage example:
python make_population_movie.py \
  --base_dir "/Volumes/Movies/2025_09_17/A14_1TP1_BF_F1" \
  --frames_dir "Frames_A14_1TP1_BF_F1" \
  --cells_dir "TrackedCells_A14_1TP1_BF_F1" \
  --out "A14_1TP1_BF_F1_population.mp4" --fps 10 --alpha 0.4

Tip: If you have many cells and the first run is slow, try --cell_limit to test on a subset.
"""

import argparse
import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except Exception as e:
    raise SystemExit("This script requires OpenCV (cv2). Please install with `pip install opencv-python`.") from e

# ----------------------- Utility: logging -----------------------

def log(msg: str):
    print(msg, flush=True)

# ----------------------- Frame utilities -----------------------

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
    # Accept common TIFF/TIF and other image extensions
    candidates = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
        candidates.extend(frames_dir.glob(ext))
    # Filter to those with "_t_<int>_" in name
    frames = [p for p in candidates if find_time_from_name(p.name) is not None]
    frames.sort(key=lambda p: find_time_from_name(p.name))
    return frames

def read_image_graceful(path: Path, global_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    # Normalize first (in native channels), then convert to BGR
    if global_range is not None:
        gmin, gmax = global_range
        img = normalize_to_uint8(img, gmin, gmax)
    else:
        # Force to uint8 if 16-bit to avoid dtype mismatch during blending
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        elif img.dtype in (np.float32, np.float64):
            m, M = float(np.nanmin(img)), float(np.nanmax(img))
            if M <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

    # Convert to BGR if needed
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_bgr = img
    elif img.ndim == 3 and img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        raise RuntimeError(f"Unsupported image shape {img.shape} for {path}")
    return img_bgr

# ----------------------- Contrast helpers -----------------------

def compute_global_percentile_range(frame_paths, p_low=1.0, p_high=99.5, sample_stride=10):
    """
    Sample pixels sparsely across all frames to estimate a global brightness range.
    Works with 8/16-bit grayscale or color; uses the first channel if multi-channel.
    """
    samples = []
    for p in frame_paths:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            continue
        if arr.ndim == 2:
            gray = arr
        elif arr.ndim == 3:
            gray = arr[..., 0]  # use first channel for range estimation
        else:
            continue
        samples.extend(gray.ravel()[::sample_stride])
    if not samples:
        # Fallback if reading somehow failed
        return 0.0, 255.0
    samples = np.asarray(samples)
    gmin = float(np.percentile(samples, p_low))
    gmax = float(np.percentile(samples, p_high))
    if gmax <= gmin:
        # avoid divide-by-zero later
        gmin, gmax = float(samples.min()), float(samples.max())
        if gmax <= gmin:
            gmin, gmax = 0.0, 255.0
    return gmin, gmax


def normalize_to_uint8(img, gmin, gmax):
    """
    Clip-and-scale img using [gmin,gmax] -> [0,255], return uint8.
    Works for grayscale or color and for uint8/uint16/float dtypes.
    """
    imgf = img.astype(np.float32, copy=False)
    # if the data looks like 0..1 floats, scale up before clipping
    if imgf.max() <= 1.0:
        imgf = imgf * 255.0
    imgf = np.clip((imgf - gmin) * (255.0 / max(1e-6, (gmax - gmin))), 0.0, 255.0)
    return imgf.astype(np.uint8)


# ----------------------- Cell color + label utilities -----------------------

def id_to_color(cell_id: int) -> Tuple[int, int, int]:
    """
    Deterministic color from cell_id in BGR (OpenCV). Uses HSV wheel hashing.
    """
    rng = abs(hash(int(cell_id)))  # deterministic per run in Python process
    h = rng % 180  # OpenCV HSV hue range [0,180)
    s = 200 + (rng // 180) % 56  # [200,255]
    v = 220 + (rng // (180 * 56)) % 36  # [220,255]
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_label(img: np.ndarray, text: str, x: int, y: int):
    """
    Draw cell ID with a light outline for readability.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 1
    # outline
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # text
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ----------------------- RLE decoders -----------------------

def _coco_rle_string_to_counts(s: str) -> List[int]:
    """
    Decode COCO's compressed RLE string to counts list.
    Pure-Python port of the algorithm used in pycocotools (rletools).

    Reference: https://github.com/cocodataset/cocoapi (mask API)
    """
    counts = []
    p = 0
    m = 0
    val = 0
    # decode each run
    i = 0
    n = len(s)
    while i < n:
        x = 0
        k = 0
        more = 1
        while more:
            c = ord(s[i]) - 48
            i += 1
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            k += 1
            if not more and (c & 0x10):
                x = -x
        if m == 0 and x == 0:
            # special case: sometimes first count is 0; ensure progress
            # but keep it (COCO allows leading zero for starting with ones)
            pass
        counts.append(x)
        m += 1
        val += x
    # COCO sometimes encodes alternating runs; ensure all positive
    counts = [int(c) for c in counts]
    return counts

def decode_rle_any(rle: Any, height: int, width: int) -> np.ndarray:
    """
    Decode multiple possible RLE representations into a boolean mask [H, W].
    Assumes COCO column-major ordering for RLE runs / flat indices.
    """
    H, W = int(height), int(width)
    total = H * W

    # Case: dict with counts/size (COCO)
    if isinstance(rle, dict) and ("counts" in rle or "size" in rle):
        counts = rle.get("counts")
        size = rle.get("size", [H, W])
        # If size mismatches, prefer provided H,W but warn
        h2, w2 = int(size[0]), int(size[1])
        if h2 != H or w2 != W:
            # We'll trust frame size (H,W), but COCO expects size consistency.
            # If mismatched, still proceed with (H,W).
            pass
        if isinstance(counts, str):
            # Try pycocotools if available
            try:
                from pycocotools import mask as maskUtils  # type: ignore
                m = maskUtils.decode({"counts": counts, "size": [H, W]})
                # maskUtils returns HxW (or HxWxN); ensure 2D
                if m.ndim == 3:
                    m = m[:, :, 0]
                return (m.astype(bool))
            except Exception:
                # Fallback: decode compressed string ourselves
                counts = _coco_rle_string_to_counts(counts)
        if isinstance(counts, list):
            # Uncompressed counts
            arr = np.zeros(total, dtype=np.uint8)
            idx = 0
            val = 0  # 0-run first
            for run in counts:
                run = int(run)
                if run <= 0:
                    continue
                if val == 1:
                    end = min(idx + run, total)
                    arr[idx:end] = 1
                    idx = end
                    val = 0
                else:
                    # zeros
                    idx = min(idx + run, total)
                    val = 1
                if idx >= total:
                    break
            mask = arr.reshape((H, W), order="F")
            return mask.astype(bool)

    # Case: plain list of counts
    if isinstance(rle, list) and all(isinstance(x, (int, np.integer)) for x in rle):
        counts = [int(x) for x in rle]
        arr = np.zeros(total, dtype=np.uint8)
        idx = 0
        val = 0
        for run in counts:
            run = int(run)
            if run <= 0:
                continue
            if val == 1:
                end = min(idx + run, total)
                arr[idx:end] = 1
                idx = end
                val = 0
            else:
                idx = min(idx + run, total)
                val = 1
            if idx >= total:
                break
        mask = arr.reshape((H, W), order="F")
        return mask.astype(bool)

    # Case: indices (flat) dict
    if isinstance(rle, dict):
        if "idxs" in rle or "indices" in rle:
            key = "idxs" if "idxs" in rle else "indices"
            idxs = rle[key]
            arr = np.zeros(total, dtype=np.uint8)
            arr[np.clip(np.asarray(idxs, dtype=np.int64), 0, total - 1)] = 1
            return arr.reshape((H, W), order="F").astype(bool)
        if "pixels" in rle:
            pix = np.asarray(rle["pixels"], dtype=np.int64)
            arr = np.zeros((H, W), dtype=np.uint8)
            # pixels assumed as [row, col]
            valid = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
            pix = pix[valid]
            arr[pix[:, 0], pix[:, 1]] = 1

            return arr.astype(bool)
    # Case: dict with start/length pairs in flat column-major indexing
    if isinstance(rle, dict) and "pairs" in rle:
        arr = np.zeros(total, dtype=np.uint8)
        pairs = rle["pairs"]
        # pairs can be list of tuples [(start, length), ...] or flat [s0, l0, s1, l1, ...]
        if isinstance(pairs, list) and pairs and isinstance(pairs[0], (list, tuple)):
            iterable = pairs
        else:
            # assume flat list
            flat = [int(x) for x in pairs]
            if len(flat) % 2 != 0:
                raise ValueError("pairs list must have even length (start,length,...).")
            iterable = [(flat[i], flat[i+1]) for i in range(0, len(flat), 2)]
        for start, length in iterable:
            s = int(start)
            L = max(0, int(length))
            if s >= total or L == 0:
                continue
            e = min(s + L, total)
            arr[s:e] = 1
        return arr.reshape((H, W), order="F").astype(bool)

    raise ValueError("Unsupported RLE format; expected COCO-style dict/list or indices/pixels.")

# ----------------------- Cell mask file loader -----------------------
def _detect_delimiter(sample_line: str) -> str:
    # simple heuristic
    return "\t" if "\t" in sample_line and "," not in sample_line else ","

def _load_csv_time_to_rle(path: Path, rle_key: str) -> Dict[int, Any]:
    """
    Load a cell_*_masks.csv/tsv file into a map: {t: {'pairs': [(start,length),...], 'size':[H,W]}}
    Expects headers including: time_point, width, height, and an RLE column (e.g. rle_bf or rle_gfp).
    """
    import csv
    m: Dict[int, Any] = {}
    with open(path, "r", newline="") as f:
        first = f.readline()
        if not first:
            return m
        delim = _detect_delimiter(first)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)

        # fallbacks if requested rle_key is absent
        fallback_keys = [rle_key, "rle", "rle_bf", "rle_gfp"]
        # preserve order but dedupe
        seen = set()
        ordered_keys = [k for k in fallback_keys if not (k in seen or seen.add(k))]

        for row in reader:
            tp = row.get("time_point") or row.get("t") or row.get("time")
            if tp is None or tp == "":
                continue
            try:
                t = int(tp)
            except Exception:
                continue

            H = int(row.get("height") or row.get("H") or row.get("h") or 0)
            W = int(row.get("width")  or row.get("W") or row.get("w") or 0)

            # pick the first available rle column
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
            m[t] = {"pairs": pairs, "size": [H, W]}
    return m


def parse_cell_id_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"cell_(\d+)_masks", path.stem)
    if m:
        return int(m.group(1))
    return None

def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_npz(path: Path) -> Any:
    data = np.load(path, allow_pickle=True)
    # Try common keys first
    if "arr_0" in data and data["arr_0"].shape == ():
        return data["arr_0"].item()
    # Otherwise, convert to dict
    return {k: data[k].item() if data[k].shape == () else data[k] for k in data.files}

def load_cell_data(path: Path) -> Any:
    # Try by extension
    if path.suffix.lower() in (".json",):
        return _load_json(path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        return _load_pickle(path)
    if path.suffix.lower() == ".npz":
        return _load_npz(path)

    # No/unknown extension; try JSON, then PKL
    try:
        return _load_json(path)
    except Exception:
        try:
            return _load_pickle(path)
        except Exception:
            # last resort, try NPZ with implicit extension
            try:
                return _load_npz(path.with_suffix(".npz"))
            except Exception as e:
                raise RuntimeError(f"Cannot load cell masks from {path} (tried JSON/PKL/NPZ).") from e

def build_time_to_rle_map(cell_obj: Any, rle_key: str) -> Dict[int, Any]:
    """
    Normalize various cell file schemas into {t_int: rle}.
    Looks first for rle under the provided rle_key, then falls back to 'rle', 'rle_bf', 'rle_gfp'.
    """
    out: Dict[int, Any] = {}

    def pick_rle(rec: Any) -> Optional[Any]:
        if not isinstance(rec, dict):
            return rec
        for k in (rle_key, "rle", "rle_bf", "rle_gfp"):
            if k in rec:
                return rec[k]
        return None

    def put(t, v):
        try:
            t_int = int(t)
            out[t_int] = v
        except Exception:
            pass

    if isinstance(cell_obj, dict):
        if "timepoints" in cell_obj and isinstance(cell_obj["timepoints"], dict):
            for t, rec in cell_obj["timepoints"].items():
                r = pick_rle(rec)
                if r is not None:
                    put(t, r)
        else:
            for k, v in cell_obj.items():
                if k.isdigit():
                    if isinstance(v, dict):
                        r = pick_rle(v)
                        if r is not None:
                            put(k, r)
                    else:
                        put(k, v)
    elif isinstance(cell_obj, list):
        for rec in cell_obj:
            if isinstance(rec, dict) and ("t" in rec or "time" in rec):
                t = rec.get("t", rec.get("time"))
                r = pick_rle(rec)
                if r is not None:
                    put(t, r)
    return out


# ----------------------- Main processing -----------------------

def overlay_masks_on_frame(
    frame_bgr: np.ndarray,
    time_t: int,
    H: int,
    W: int,
    cell_maps: List[Tuple[int, Dict[int, Any]]],
    alpha: float,
) -> np.ndarray:
    """
    cell_maps: list of (cell_id, time_to_rle_map)
    """
    overlay = np.zeros_like(frame_bgr, dtype=np.uint8)

    for cell_id, time_to_rle in cell_maps:
        rle = time_to_rle.get(time_t)
        if rle is None:
            continue
        try:
            mask = decode_rle_any(rle, H, W)
        except Exception as e:
            # Skip misformatted entries but continue others
            continue
        if not mask.any():
            continue

        color = id_to_color(cell_id)
        # color the overlay where mask is True
        overlay[mask] = color

        # centroid for label
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            # offset label slightly above centroid
            draw_label(frame_bgr, str(cell_id), cx, max(12, cy - 5))

    # blend once per frame (faster than per-cell)
    blended = cv2.addWeighted(overlay, float(alpha), frame_bgr, 1.0, 0.0)
    return blended

def main():
    parser = argparse.ArgumentParser(description="Overlay single-cell RLE masks onto frames and write an MP4.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for this dataset (e.g., /Volumes/Movies/2025_09_17/A14_1TP1_BF_F1)")
    parser.add_argument("--frames_dir", type=str, required=True, help="Subfolder containing frames (e.g., Frames_A14_1TP1_BF_F1)")
    parser.add_argument("--cells_dir", type=str, required=True, help="Subfolder containing cell mask files (e.g., TrackedCells_A14_1TP1_BF_F1)")
    parser.add_argument("--frame_glob", type=str, default="*t_{t}_c_0.tif*", help="Glob with {t} placeholder if needed. Default '*t_{t}_c_0.tif*'. Only used if direct discovery fails.")
    parser.add_argument("--out", type=str, required=True, help="Output MP4 path (e.g., A14_1TP1_BF_F1_population.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay opacity in [0,1]")
    parser.add_argument("--cell_limit", type=int, default=0, help="Limit number of cells for testing (0 = no limit)")
    parser.add_argument("--rle_key", type=str, default="rle_bf", help="Key in the cell files that stores RLE (e.g., 'rle_bf', 'rle_gfp'). Default: rle_bf")

    args = parser.parse_args()

    base = Path(args.base_dir)
    frames_dir = base / args.frames_dir
    cells_dir = base / args.cells_dir
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames_dir.is_dir():
        raise SystemExit(f"Frames directory not found: {frames_dir}")
    if not cells_dir.is_dir():
        raise SystemExit(f"Cells directory not found: {cells_dir}")

    # Discover frames and timepoints
    frames = discover_frames(frames_dir)
    if not frames:
        # fallback: build via glob pattern with {t} wildcard by scanning directory for all times
        # but we still need to find times; we will try any files and sort by time.
        candidates = list(frames_dir.glob(args.frame_glob.replace("{t}", "*")))
        frames = [p for p in candidates if find_time_from_name(p.name) is not None]
        frames.sort(key=lambda p: find_time_from_name(p.name))

    if not frames:
        raise SystemExit(f"No frames found in {frames_dir}. Check naming pattern and --frame_glob.")

    # Read first frame to get size
    first_frame = read_image_graceful(frames[0])
    H, W = first_frame.shape[:2]
    log(f"Frame size: {W}x{H}, total frames discovered: {len(frames)}")

    # Build list of times
    times = [find_time_from_name(p.name) for p in frames]
    times = [t for t in times if t is not None]
    if not times:
        raise SystemExit("Could not parse time indices from frame names.")
    log(f"Time range: {min(times)} .. {max(times)}")

    # Load cell mask files
    cell_files = sorted([p for p in cells_dir.iterdir() if p.is_file() and "cell_" in p.name and "masks" in p.name])
    log(f"Found {len(cell_files)} cell mask files.")

    if args.cell_limit > 0:
        cell_files = cell_files[:args.cell_limit]
        log(f"Limiting to first {len(cell_files)} cell files for testing.")

    cell_maps: List[Tuple[int, Dict[int, Any]]] = []
    for cf in cell_files:
        try:
            cell_id = parse_cell_id_from_filename(cf)
            if cell_id is None:
                continue

            if cf.suffix.lower() in (".csv", ".tsv"):
                time_to_rle = _load_csv_time_to_rle(cf, args.rle_key)
            else:
                obj = load_cell_data(cf)
                time_to_rle = build_time_to_rle_map(obj, args.rle_key)

            if time_to_rle:
                cell_maps.append((cell_id, time_to_rle))
            else:
                log(f"[WARN] No usable RLE in {cf.name}")
        except Exception as e:
            log(f"[WARN] Skipping {cf.name}: {e}")


    if not cell_maps:
        raise SystemExit("No usable cell masks found. Check file formats and 'rle_bf' presence.")

    log(f"Loaded masks for {len(cell_maps)} cells.")
    # Compute a global percentile range for contrast normalization
    gmin, gmax = compute_global_percentile_range(frames, p_low=1.0, p_high=99.5, sample_stride=10)
    log(f"Global intensity range (percentile 1.0–99.5): gmin={gmin:.2f}, gmax={gmax:.2f}")
    global_range = (gmin, gmax)

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (W, H), True)
    if not vw.isOpened():
        raise SystemExit("Failed to open VideoWriter. Try a different --out path or codec.")

    # Iterate frames by time
    for i, frame_path in enumerate(frames):
        t = times[i]
        frame_bgr = read_image_graceful(frame_path, global_range=global_range)
        frame_out = overlay_masks_on_frame(frame_bgr, t, H, W, cell_maps, alpha=args.alpha)
        vw.write(frame_out)
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            log(f"Rendered frame {i+1}/{len(frames)} (t={t})")

    vw.release()
    log(f"Done. Wrote {out_path}")

if __name__ == "__main__":
    main()
