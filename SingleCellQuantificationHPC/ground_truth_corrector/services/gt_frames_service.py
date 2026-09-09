import re
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from tifffile import imread
import pandas as pd
from skimage.measure import regionprops

from ..config import Config
from ..security import resolve_under_root
from ..errors import NotFoundError
from ..schemas import validate_and_decode_rle

# Colour rendered for a segmentation region that has no tracked cell behind it
# (BGR, for cv2). Unassigned / untracked segments are rendered in pure white.
UNTRACKED_COLOR = (255, 255, 255)

def find_time_from_name(name: str) -> Optional[int]:
    m = re.search(r"_t_(\d+)_", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"_t(\d+)_", name)
    if m2:
        return int(m2.group(1))
    return None

def fnv1a_32(s: str) -> int:
    """Deterministic 32-bit FNV-1a hash of an ASCII string. Ported byte-for-byte
    to static/js/color.js so a cell's colour matches between server and client.
    global_cell_id values are ASCII, so charCodeAt in JS lines up with these
    UTF-8 bytes."""
    h = 0x811C9DC5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) % (2 ** 32)
    return h

def stable_color_key(identity: Union[int, str, "np.integer"]) -> int:
    """Map a stable cell identity to the integer id_to_color expects.

    identity is either an int (a tracked local cell id, for a single-film cell)
    or a str (the global_cell_id, for a linked/multi-film cell). Strings are
    hashed deterministically so the same global cell keeps one colour across
    every film and keyframe of its sequence. See docs/FLASK_APPS.md
    "Shared UI conventions".
    """
    if isinstance(identity, (int, np.integer)):
        return int(identity)
    return fnv1a_32(str(identity))

def format_cell_display_label(raw_id: Any) -> str:
    """Format cell identity to match the UI button convention (e.g. 'Cell 50')."""
    s = str(raw_id)
    if "_cell_" in s:
        return f"Cell {s.split('_cell_')[1]}"
    if s.isdigit():
        return f"Cell {s}"
    return s

def id_to_color(cell_id: int) -> Tuple[int, int, int]:
    """Deterministic hue for a stable cell key. Returns BGR for cv2.

    Algorithm is kept identical to tracking_corrector.services.frames_service
    .id_to_color (the canonical copy per docs/FLASK_APPS.md P9); that one returns
    RGB, this returns BGR. Deduplicating the two is a tracked cleanup task.
    """
    val = (int(cell_id) * 2654435761) % (2 ** 32)
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

    return int(b * 255), int(g * 255), int(r * 255)


class GTFramesService:
    def __init__(self, config: Config):
        self.config = config
        self._path_cache: Dict[Tuple[str, str, str], Dict[int, Path]] = {}
        self._pop_cache: Dict[Tuple, bytes] = {}
        self._boundaries_cache: Dict[Tuple, bytes] = {}
        self._ident_cache: Dict[Tuple, Dict[int, Tuple[int, str]]] = {}
        self._film_rle_cache: Dict[Tuple[str, str, int], Dict[int, str]] = {}
        self._parsed_films: set = set()

    def get_film_cell_rles_at_t(self, exp: str, film: str, t_val: int) -> Dict[int, str]:
        """Returns {cid: rle_str} for a film at timepoint t_val, parsing all CSVs in film once."""
        cache_key = (exp, film, t_val)
        film_key = (exp, film)
        if film_key not in self._parsed_films:
            self._parsed_films.add(film_key)
            tracked_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"TrackedCells_{film}")
            if tracked_dir.exists():
                for csv_file in tracked_dir.glob("cell_*_masks.csv"):
                    if csv_file.name.startswith("."):
                        continue
                    m = re.match(r"^cell_(\d+)_masks\.csv$", csv_file.name)
                    if not m:
                        continue
                    cid = int(m.group(1))
                    try:
                        df = pd.read_csv(csv_file)
                        for _, row in df.iterrows():
                            t = int(row["time_point"])
                            rle = str(row.get("rle_bf", "") or "")
                            if not rle.strip() or rle.lower() == "nan":
                                rle = str(row.get("rle_gfp", "") or "")
                            if rle.strip() and rle.lower() != "nan":
                                tk = (exp, film, t)
                                if tk not in self._film_rle_cache:
                                    self._film_rle_cache[tk] = {}
                                self._film_rle_cache[tk][cid] = rle
                    except Exception:
                        continue
        return self._film_rle_cache.get(cache_key, {})

    def get_film_frame_paths(self, exp: str, film: str, channel: str = "bf") -> Dict[int, Path]:
        key = (exp, film, channel.lower())
        if key in self._path_cache:
            return self._path_cache[key]

        frames_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"Frames_{film}")
        cache: Dict[int, Path] = {}
        target_c = 1 if channel.lower() == "gfp" else 0

        valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        if frames_dir.exists():
            for f in frames_dir.iterdir():
                if f.name.startswith(".") or not f.is_file():
                    continue
                if f.suffix.lower() not in valid_exts:
                    continue
                if "_seg" in f.stem:
                    continue
                t_val = find_time_from_name(f.name)
                m_c = re.search(r"_c_(\d+)\.", f.name)
                if t_val is not None:
                    c_num = int(m_c.group(1)) if m_c else 0
                    if t_val not in cache or c_num == target_c:
                        cache[t_val] = f

        self._path_cache[key] = cache
        return cache

    def get_film_keyframes(self, exp: str, film: str) -> List[int]:
        """Return the 3 keyframes (first, middle, last) for a film."""
        paths = self.get_film_frame_paths(exp, film, "bf")
        if not paths:
            paths = self.get_film_frame_paths(exp, film, "gfp")
        if not paths:
            return [0]
        
        sorted_times = sorted(paths.keys())
        n = len(sorted_times)
        if n <= 3:
            return sorted_times
        first_t = sorted_times[0]
        mid_t = sorted_times[(n - 1) // 2]
        last_t = sorted_times[-1]
        return [first_t, mid_t, last_t]

    def get_sequence_keyframe_map(self, exp: str, sequence: str) -> List[Dict[str, Any]]:
        """Map linear sequence keyframes across all films in sequence."""
        from .linkage_service import LinkageService
        from ..repositories.linkage_repository import LinkageRepository
        
        repo = LinkageRepository(self.config.local_movie_root)
        link_svc = LinkageService(repo)
        seqs = link_svc.get_sequences(exp).get("sequences", {})
        
        if sequence in seqs:
            films = seqs[sequence].get("films", [sequence])
        else:
            films = [sequence]
            
        keyframe_map = []
        global_idx = 0
        for f_idx, film in enumerate(films):
            k_times = self.get_film_keyframes(exp, film)
            for k_idx, local_t in enumerate(k_times):
                if k_idx == 0:
                    pos = "First"
                elif k_idx == len(k_times) - 1:
                    pos = "Last"
                else:
                    pos = "Middle"
                keyframe_map.append({
                    "global_t": global_idx,
                    "film": film,
                    "film_idx": f_idx,
                    "local_t": local_t,
                    "keyframe_pos": pos,
                    "keyframe_idx": k_idx,
                    "total_film_keyframes": len(k_times)
                })
                global_idx += 1
        return keyframe_map

    def local_to_global_map(self, exp: str, film: str, sequence: Optional[str]) -> Dict[int, str]:
        """{local_cell_id: global_cell_id} for `film` within `sequence`.

        Empty when there is no sequence context (single-film view) or the film is
        not part of the sequence — callers then key colour on the local id.
        """
        if not sequence:
            return {}
        from .linkage_service import LinkageService
        from ..repositories.linkage_repository import LinkageRepository

        try:
            repo = LinkageRepository(self.config.local_movie_root)
            seqs = LinkageService(repo).get_sequences(exp).get("sequences", {})
        except Exception:
            return {}

        seq = seqs.get(sequence)
        if not seq:
            return {}
        films = seq.get("films", [])
        if film not in films:
            return {}
        f_idx = films.index(film)
        out: Dict[int, str] = {}
        for gid, local_ids in (seq.get("global_cells") or {}).items():
            if f_idx < len(local_ids):
                lid = local_ids[f_idx]
                if lid is not None and int(lid) != -1:
                    out[int(lid)] = str(gid)
        return out

    def seg_label_identity(
        self,
        exp: str,
        film: str,
        local_t: int,
        seg_lbl: "np.ndarray",
        H: int,
        W: int,
        local2global: Dict[int, str],
        sequence: Optional[str] = None,
    ) -> Dict[int, Tuple[int, str]]:
        """{seg_label: (stable_color_key, display_str)} by max mask overlap with
        the tracked cells for (film, local_t).

        A seg label absent from the result has no tracked cell behind it and must
        be drawn in UNTRACKED_COLOR, never coloured by the raw label.
        """
        cache_key = (exp, film, local_t, sequence)
        if cache_key in self._ident_cache:
            return self._ident_cache[cache_key]

        out: Dict[int, Tuple[int, str]] = {}
        rles_dict = self.get_film_cell_rles_at_t(exp, film, local_t)
        seg_flat = seg_lbl.flatten(order='F')

        for cid, rle in rles_dict.items():
            if sequence and cid not in local2global:
                continue
            try:
                nums = np.fromstring(rle.strip(), dtype=int, sep=' ')
                starts = nums[0::2] - 1
                ends = starts + nums[1::2]
                slices = [seg_flat[st:en] for st, en in zip(starts, ends) if en <= seg_flat.size]
                if not slices:
                    continue
                labels_here = np.concatenate(slices)
                labels_here = labels_here[labels_here > 0]
                if labels_here.size == 0:
                    continue
                vals, counts = np.unique(labels_here, return_counts=True)
                best_lbl = int(vals[np.argmax(counts)])
                identity = local2global[cid] if sequence else cid
                display = format_cell_display_label(identity)
                gid_val = local2global.get(cid) if sequence else None
                out.setdefault(best_lbl, (stable_color_key(identity), display, cid, gid_val))
            except Exception:
                continue
        self._ident_cache[cache_key] = out
        return out

    def get_frame_path(self, exp: str, film: str, t_val: int, channel: str = "bf") -> Path:
        cache = self.get_film_frame_paths(exp, film, channel)
        if t_val in cache:
            return cache[t_val]
        # Fallback to alternate channel if requested not found
        alt_channel = "gfp" if channel.lower() == "bf" else "bf"
        alt_cache = self.get_film_frame_paths(exp, film, alt_channel)
        if t_val in alt_cache:
            return alt_cache[t_val]
        raise NotFoundError(f"Frame t={t_val} not found in {exp}/{film}")

    def render_frame_jpeg(self, exp: str, film: str, t_val: int, channel: str = "bf", quality: int = 90) -> bytes:
        file_path = self.get_frame_path(exp, film, t_val, channel)
        img = imread(str(file_path))
        a = np.asarray(img, dtype=np.float32)
        if a.ndim > 2:
            a = a[..., 0]

        lo, hi = np.nanpercentile(a, [1, 99.5]) if np.isfinite(a).any() else (0.0, 1.0)
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        norm = np.clip((a - lo) / (hi - lo), 0, 1)
        u8 = (norm * 255).astype(np.uint8)

        is_success, buffer = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not is_success:
            raise RuntimeError("Failed to encode JPEG")
        return buffer.tobytes()

    def clear_population_cache(self, exp: Optional[str] = None, film: Optional[str] = None, t_val: Optional[int] = None):
        for cache_dict in (self._pop_cache, self._boundaries_cache, self._ident_cache):
            if exp is None:
                cache_dict.clear()
            else:
                keys_to_del = [k for k in cache_dict if k[0] == exp and (film is None or k[1] == film) and (t_val is None or k[2] == t_val)]
                for k in keys_to_del:
                    cache_dict.pop(k, None)
        if exp is None:
            self._film_rle_cache.clear()
            self._parsed_films.clear()
        else:
            rle_keys_to_del = [k for k in self._film_rle_cache if k[0] == exp and (film is None or k[1] == film) and (t_val is None or k[2] == t_val)]
            for k in rle_keys_to_del:
                self._film_rle_cache.pop(k, None)
            parsed_to_del = [k for k in self._parsed_films if k[0] == exp and (film is None or k[1] == film)]
            for k in parsed_to_del:
                self._parsed_films.discard(k)

    def render_population_frame_jpeg(self, exp: str, film: str, t_val: int, sequence: Optional[str] = None, quality: int = 85, force: bool = False) -> bytes:
        # Colour now depends on the sequence (local id -> global_cell_id), so the
        # sequence is part of the cache key.
        cache_key = (exp, film, t_val, sequence)
        if not force and cache_key in self._pop_cache:
            return self._pop_cache[cache_key]

        # Check on-disk cache if sequence directory exists
        disk_cache_file: Optional[Path] = None
        if sequence:
            seq_dir = resolve_under_root(self.config.local_movie_root, exp, sequence, f"GTPopulationFrames_{sequence}")
            seq_dir.mkdir(parents=True, exist_ok=True)
            disk_cache_file = seq_dir / f"{film}_t_{t_val:03d}.jpg"
            if not force and disk_cache_file.exists() and disk_cache_file.stat().st_size > 0:
                data = disk_cache_file.read_bytes()
                self._pop_cache[cache_key] = data
                return data

        local2global = self.local_to_global_map(exp, film, sequence)

        file_path = self.get_frame_path(exp, film, t_val)
        img = imread(str(file_path))
        a = np.asarray(img, dtype=np.float32)
        if a.ndim > 2:
            a = a[..., 0]

        lo, hi = np.nanpercentile(a, [1, 99]) if np.isfinite(a).any() else (0.0, 1.0)
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        norm = np.clip((a - lo) / (hi - lo), 0, 1)
        bg = (norm * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        H, W = bg.shape[:2]

        overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        alpha = 0.45

        def _put_label(cx: int, cy: int, text: str) -> None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness_outline = 3
            thickness_text = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness_outline)
            tx = int(cx - tw / 2)
            ty = int(cy + th / 2)
            cv2.putText(img_bgr, text, (tx, ty), font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
            cv2.putText(img_bgr, text, (tx, ty), font, scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

        # 1. Load updated individual cell masks from TrackedCells_<film>
        tracked_union = np.zeros((H, W), dtype=bool)
        cell_draw_list = []
        rles_dict = self.get_film_cell_rles_at_t(exp, film, t_val)
        is_seq_mode = bool(sequence)

        for cid, rle in rles_dict.items():
            try:
                mask = validate_and_decode_rle(rle, H, W)
                if not mask.any():
                    continue

                tracked_union |= (mask > 0)
                if is_seq_mode:
                    if cid not in local2global:
                        # Unlinked local cell in sequence mode -> render as UNTRACKED / WHITE
                        overlay[mask > 0] = (255, 255, 255)
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_bgr, contours, -1, (255, 255, 255), 1)
                        continue
                    identity = local2global[cid]
                else:
                    identity = cid

                # Global/Local cell color (BGR)
                b, g, r = id_to_color(stable_color_key(identity))
                display = format_cell_display_label(identity)
                ys, xs = np.where(mask > 0)
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cell_draw_list.append((mask, (b, g, r), (cx, cy), display, identity))
            except Exception:
                continue

        # 2. Layer 1: Render Untracked / Background Cellpose segments in pure WHITE (255, 255, 255)
        masks_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"Masks_{film}")
        seg_candidates = list(masks_dir.glob(f"*_t_{t_val:03d}_*_seg.tif")) or list(masks_dir.glob(f"*_t{t_val:03d}_*_seg.tif")) or list(masks_dir.glob(f"*_t_{t_val}_*_seg.tif"))
        if seg_candidates and seg_candidates[0].exists():
            try:
                seg = imread(str(seg_candidates[0]))
                if seg.ndim > 2:
                    seg = seg[..., 0]
                untracked_mask = (seg > 0) & (~tracked_union)
                if untracked_mask.any():
                    overlay[untracked_mask] = (255, 255, 255)  # Pure white fill
                    contours, _ = cv2.findContours(untracked_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_bgr, contours, -1, (255, 255, 255), 1)  # Thin white contour
            except Exception:
                pass

        # 3. Layer 2: Render Tracked Global Cell Masks & Outlines & Labels
        for mask, (b, g, r), (cx, cy), display, identity in cell_draw_list:
            overlay[mask > 0] = (b, g, r)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, contours, -1, (b, g, r), 2)
            _put_label(cx, cy, display)

        blended = cv2.addWeighted(overlay, alpha, img_bgr, 1.0, 0.0)

        target_w, target_h = 1000, 1000
        if W != target_w or H != target_h:
            interp = cv2.INTER_AREA if (W >= target_w and H >= target_h) else cv2.INTER_LINEAR
            blended = cv2.resize(blended, (target_w, target_h), interpolation=interp)

        is_success, buffer = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        jpeg_bytes = buffer.tobytes()
        self._pop_cache[cache_key] = jpeg_bytes
        if disk_cache_file:
            try:
                disk_cache_file.write_bytes(jpeg_bytes)
            except Exception:
                pass
        return jpeg_bytes

    def render_boundary_png(self, exp: str, film: str, t_val: int, sequence: Optional[str] = None, force: bool = False) -> bytes:
        cache_key = (exp, film, t_val, sequence)
        if not force and cache_key in self._boundaries_cache:
            return self._boundaries_cache[cache_key]

        disk_cache_file: Optional[Path] = None
        if sequence:
            seq_dir = resolve_under_root(self.config.local_movie_root, exp, sequence, f"GTPopulationFrames_{sequence}")
            seq_dir.mkdir(parents=True, exist_ok=True)
            disk_cache_file = seq_dir / f"{film}_t_{t_val:03d}_boundaries.png"
            if not force and disk_cache_file.exists() and disk_cache_file.stat().st_size > 0:
                data = disk_cache_file.read_bytes()
                self._boundaries_cache[cache_key] = data
                return data

        local2global = self.local_to_global_map(exp, film, sequence)

        sample_path = self.get_frame_path(exp, film, t_val)
        sample_img = imread(str(sample_path))
        H, W = sample_img.shape[:2]

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        tracked_union = np.zeros((H, W), dtype=bool)

        # 1. Tracked cells from updated individual CSV masks
        rles_dict = self.get_film_cell_rles_at_t(exp, film, t_val)
        is_seq_mode = bool(sequence)

        for cid, rle in rles_dict.items():
            try:
                mask = validate_and_decode_rle(rle, H, W)
                if not mask.any():
                    continue
                tracked_union |= (mask > 0)

                if is_seq_mode:
                    if cid not in local2global:
                        # Unlinked local cell in sequence mode -> render boundary in pure WHITE
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(rgba, contours, -1, (255, 255, 255, 180), 1)
                        continue
                    identity = local2global[cid]
                else:
                    identity = cid

                b, g, r = id_to_color(stable_color_key(identity))
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgba, contours, -1, (int(b), int(g), int(r), 255), 2)
            except Exception:
                continue

        # 2. Untracked Cellpose background/debris segments in pure WHITE (255, 255, 255, 180)
        masks_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"Masks_{film}")
        seg_files = list(masks_dir.glob(f"*_t_{t_val:03d}_*_seg.tif")) or list(masks_dir.glob(f"*_t{t_val:03d}_*_seg.tif"))
        if seg_files and seg_files[0].exists():
            try:
                seg = imread(str(seg_files[0]))
                if seg.ndim > 2:
                    seg = seg[..., 0]
                untracked_mask = (seg > 0) & (~tracked_union)
                if untracked_mask.any():
                    contours, _ = cv2.findContours(untracked_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rgba, contours, -1, (255, 255, 255, 180), 1)
            except Exception:
                pass

        is_success, buffer = cv2.imencode(".png", rgba)
        if not is_success:
            raise RuntimeError("Failed to encode boundaries PNG")
        png_bytes = buffer.tobytes()
        self._boundaries_cache[cache_key] = png_bytes
        if disk_cache_file:
            try:
                disk_cache_file.write_bytes(png_bytes)
            except Exception:
                pass
        return png_bytes
