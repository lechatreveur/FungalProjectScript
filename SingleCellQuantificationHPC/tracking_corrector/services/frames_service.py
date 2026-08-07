import re
import io
import cv2
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from skimage.io import imread
from PIL import Image
from ..config import Config
from ..errors import NotFoundError
from ..security import resolve_under_root
from ..repositories.mask_repository import MaskRepository
from ..schemas import validate_and_decode_rle

def find_time_from_name(filename: str) -> Optional[int]:
    m = re.search(r"_t_(\d+)_", filename)
    if m:
        return int(m.group(1))
    return None

def id_to_color(cell_id: int) -> Tuple[int, int, int]:
    val = (cell_id * 2654435761) % (2**32)
    h = (val % 360) / 360.0
    s = 0.8
    v = 0.95
    
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    i %= 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    
    return int(r * 255), int(g * 255), int(b * 255)


class FramesService:
    def __init__(self, config: Config, mask_repo: MaskRepository):
        self.config = config
        self.mask_repo = mask_repo
        self._path_cache: Dict[Tuple[str, str, str], Dict[int, Path]] = {}
        self._running_pregens: Set[Tuple[str, str]] = set()
        self._pregen_lock = threading.Lock()

    def get_film_frame_paths(self, exp: str, film: str, channel: str = "bf") -> Dict[int, Path]:
        key = (exp, film, channel)
        if key in self._path_cache:
            return self._path_cache[key]

        frames_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"Frames_{film}")
        cache = {}
        target_c = 1 if channel.lower() == "gfp" else 0

        if frames_dir.exists():
            for f in frames_dir.iterdir():
                if f.name.startswith(".") or not f.is_file():
                    continue
                t_val = find_time_from_name(f.name)
                m_c = re.search(r"_c_(\d+)\.", f.name)
                if t_val is not None:
                    c_num = int(m_c.group(1)) if m_c else 0
                    if t_val not in cache or c_num == target_c:
                        cache[t_val] = f

        self._path_cache[key] = cache
        return cache

    def get_frame_path(self, exp: str, film: str, t_val: int, channel: str = "bf") -> Path:
        paths = self.get_film_frame_paths(exp, film, channel)
        if t_val not in paths:
            paths = self.get_film_frame_paths(exp, film, "bf")
        if t_val not in paths:
            raise NotFoundError(f"Frame t={t_val} not found for {exp}/{film}")
        return paths[t_val]

    def render_frame_jpeg(
        self,
        exp: str,
        film: str,
        t_val: int,
        p_low: float = 1.0,
        p_high: float = 99.0,
        quality: int = 85
    ) -> bytes:
        file_path = self.get_frame_path(exp, film, t_val)
        img = imread(str(file_path))
        
        a = np.asarray(img, dtype=np.float32)
        if a.ndim > 2:
            a = a[..., 0]
            
        lo, hi = np.nanpercentile(a, [p_low, p_high]) if np.isfinite(a).any() else (0.0, 1.0)
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        
        norm = np.clip((a - lo) / (hi - lo), 0, 1)
        norm_u8 = (norm * 255).astype(np.uint8)
        
        im = Image.fromarray(norm_u8)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def get_septum_center(self, exp: str, film: str, cell_id: int, t_val: int) -> Optional[Tuple[int, int]]:
        csv_path = resolve_under_root(self.config.local_movie_root, exp, film, f"TrackedCells_{film}", f"cell_{cell_id}_data.csv")
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                df = pd.read_csv(csv_path)
                row = df[(df["time_point"] == t_val)]
                if not row.empty:
                    r = row.iloc[0]
                    cx = r.get("pattern_center_col")
                    cy = r.get("pattern_center_row")
                    if pd.notna(cx) and pd.notna(cy):
                        return int(cx), int(cy)
            except Exception:
                pass
        return None

    def trigger_pregeneration(self, exp: str, film: str):
        key = (exp, film)
        with self._pregen_lock:
            if key in self._running_pregens:
                return
            self._running_pregens.add(key)

        def worker():
            try:
                paths = self.get_film_frame_paths(exp, film)
                for t_val in sorted(paths.keys()):
                    cache_dir = resolve_under_root(self.config.cache_root, exp, film, f"PopulationFrames_{film}")
                    cache_file = cache_dir / f"frame_{t_val:03d}.jpg"
                    if not cache_file.exists():
                        try:
                            self._generate_population_frame_bytes(exp, film, t_val, cache_file)
                        except Exception:
                            pass
            finally:
                with self._pregen_lock:
                    self._running_pregens.discard(key)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _generate_population_frame_bytes(self, exp: str, film: str, t_val: int, cache_file: Optional[Path] = None, quality: int = 85) -> bytes:
        file_path = self.get_frame_path(exp, film, t_val)
        img = imread(str(file_path))
        
        a = np.asarray(img, dtype=np.float32)
        if a.ndim > 2: a = a[..., 0]
        
        lo, hi = np.nanpercentile(a, [1, 99]) if np.isfinite(a).any() else (0.0, 1.0)
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        norm = np.clip((a - lo) / (hi - lo), 0, 1)
        bg = (norm * 255).astype(np.uint8)
        
        img_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        H, W = bg.shape[:2]
        
        overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        alpha = 0.4
        
        septum_json_path = resolve_under_root(self.config.local_movie_root, exp, film, f"TrackedCells_{film}", "cell_plots", "gui_labels", "global_septum_alignment.json")
        septum_intervals = {}
        if septum_json_path.exists():
            try:
                import json
                with open(septum_json_path, 'r') as f:
                    js = json.load(f)
                septum_intervals = js.get("cell_intervals", {})
            except Exception:
                pass

        tracked_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"TrackedCells_{film}")
        for csv_file in tracked_dir.glob("cell_*_masks.csv"):
            if csv_file.name.startswith("."): continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_file.name)
            if not m: continue
            cid = int(m.group(1))
            
            try:
                df = pd.read_csv(csv_file)
                rows = df[df["time_point"] == t_val]
                if rows.empty: continue
                rle = str(rows.iloc[0].get("rle_bf", ""))
                if not rle or rle.strip() == "" or rle.lower() == "nan":
                    rle = str(rows.iloc[0].get("rle_gfp", ""))
                if not rle or rle.strip() == "" or rle.lower() == "nan": continue
                
                mask = validate_and_decode_rle(rle, H, W)
                if not mask.any(): continue
                
                b, g, r = id_to_color(cid)
                overlay[mask > 0] = (b, g, r)
                
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    text = str(cid)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img_bgr, text, (cx, cy), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img_bgr, text, (cx, cy), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                ci = septum_intervals.get(str(cid), {})
                if ci.get("has_septum"):
                    st = ci.get("start_aligned")
                    et = ci.get("end_aligned")
                    in_range = False
                    if st is not None and et is not None:
                        in_range = (st <= t_val <= et)
                    elif st is not None:
                        in_range = (t_val >= st)
                    elif et is not None:
                        in_range = (t_val <= et)
                        
                    if in_range and len(xs) > 0:
                        xmin, ymin = xs.min(), ys.min()
                        center = self.get_septum_center(exp, film, cid, t_val)
                        if center is not None:
                            local_cx, local_cy = center
                            cx_global = xmin + local_cx
                            cy_global = ymin + local_cy
                        else:
                            cx_global = cx
                            cy_global = cy
                            
                        half = 6
                        x1_box, x2_box = max(0, cx_global - half), min(W - 1, cx_global + half)
                        y1_box, y2_box = max(0, cy_global - half), min(H - 1, cy_global + half)
                        cv2.rectangle(img_bgr, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 255), 2)
            except Exception:
                continue

        blended = cv2.addWeighted(overlay, alpha, img_bgr, 1.0, 0.0)
        is_success, buffer = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        jpeg_bytes = buffer.tobytes()

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(jpeg_bytes)
            except Exception:
                pass

        return jpeg_bytes

    def render_population_frame_jpeg(
        self,
        exp: str,
        film: str,
        t_val: int,
        quality: int = 85
    ) -> bytes:
        cache_dir = resolve_under_root(self.config.cache_root, exp, film, f"PopulationFrames_{film}")
        cache_file = cache_dir / f"frame_{t_val:03d}.jpg"
        if cache_file.exists():
            try:
                return cache_file.read_bytes()
            except Exception:
                pass

        # Trigger background pre-generation of remaining frames for smooth playback
        self.trigger_pregeneration(exp, film)
        return self._generate_population_frame_bytes(exp, film, t_val, cache_file, quality)

    def render_frame_crop_jpeg(
        self,
        exp: str,
        film: str,
        cell_id: int,
        t_val: int,
        channel: str = "bf",
        crop_size: int = 100
    ) -> bytes:
        cache_dir = resolve_under_root(self.config.cache_root, exp, film, f"CellCrops_{film}")
        cache_file = cache_dir / f"cell_{cell_id}_t_{t_val:03d}_{channel}.jpg"
        if cache_file.exists():
            try:
                return cache_file.read_bytes()
            except Exception:
                pass

        try:
            file_path = self.get_frame_path(exp, film, t_val, channel)
            img = imread(str(file_path))
        except Exception:
            img = np.zeros((crop_size, crop_size), dtype=np.uint8)

        H, W = img.shape[:2]
        cy, cx = H // 2, W // 2

        try:
            df, _ = self.mask_repo.load_cell_masks(exp, film, cell_id)
            rows = df[df["time_point"] == t_val] if "time_point" in df.columns else (df.iloc[[t_val]] if t_val < len(df) else pd.DataFrame())
            if not rows.empty:
                rle_col = f"rle_{channel}"
                if rle_col not in rows.columns or not rows.iloc[0][rle_col]:
                    rle_col = "rle_gfp" if ("rle_gfp" in rows.columns and rows.iloc[0]["rle_gfp"]) else "rle_bf"
                    
                rle = str(rows.iloc[0].get(rle_col, ""))
                if rle and rle.strip() and rle.lower() != "nan":
                    mask = validate_and_decode_rle(rle, H, W)
                    ys, xs = np.where(mask > 0)
                    if len(ys) > 0:
                        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        except Exception:
            pass

        y0 = max(0, cy - crop_size // 2)
        y1 = min(H, cy + crop_size // 2)
        x0 = max(0, cx - crop_size // 2)
        x1 = min(W, cx + crop_size // 2)

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((crop_size, crop_size), dtype=np.uint8)

        af = crop.astype(np.float32)
        p_lo = np.percentile(af, 1.0) if np.isfinite(af).any() else 0.0
        p_hi = np.percentile(af, 99.5) if np.isfinite(af).any() else 255.0
        
        if p_hi > p_lo:
            img_scaled = np.clip((af - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_scaled = crop.astype(np.uint8)

        im = Image.fromarray(img_scaled)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(jpeg_bytes)
        except Exception:
            pass

        return jpeg_bytes

