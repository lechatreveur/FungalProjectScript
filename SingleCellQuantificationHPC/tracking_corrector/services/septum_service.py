import os
import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from ..repositories.qc_repository import QCRepository
from ..repositories.linkage_repository import LinkageRepository
from .audit_service import AuditService
from ..schemas import SaveSeptumRequest
from ..errors import ValidationError, NotFoundError
from datetime import datetime

# Add project path to sys.path for FungalInferenceCore
sys.path.append("/Users/user/Documents/Python_Scripts/FungalProjectScript")
try:
    from SingleCellDataAnalysis.inference_core import FungalInferenceCore
except ImportError:
    FungalInferenceCore = None

try:
    from SingleCellDataAnalysis.septum_lineage_dataset import (
        infer_modality,
        parse_acquisition_metadata,
    )
    from SingleCellDataAnalysis.septum_lineage_inference import (
        SeptumLineageInference,
    )
except ImportError:
    SeptumLineageInference = None

_model_cache: Dict[str, Any] = {}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LINEAGE_MODEL_DIR = _PROJECT_ROOT / "SingleCellDataAnalysis" / "lineage_model_v1"
_LINEAGE_MODEL_METRICS = {
    "held_out_experiment": "2026_04_30_M135",
    "state_balanced_accuracy": 0.7271147406629027,
    "endpoint_event_f1_at_5_min": 0.2397003745318352,
    "endpoint_median_absolute_error_min": 2.0,
}

def get_inference_runner(base_root: Path, exp: str) -> Optional[Any]:
    if exp in _model_cache:
        return _model_cache[exp]
    if FungalInferenceCore is None:
        return None

    chk_dir = base_root / exp / "training_dataset" / "checkpoints_binary"
    model_path = chk_dir / "model_best.pt"
    if not model_path.exists():
        model_path = chk_dir / "model_latest.pt"
    if not model_path.exists():
        return None
    try:
        runner = FungalInferenceCore(str(model_path), device="cpu")
        _model_cache[exp] = runner
        return runner
    except Exception:
        return None


def get_lineage_inference_runner() -> Optional[Any]:
    cache_key = "lineage_v1_review_only"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    if SeptumLineageInference is None:
        return None
    checkpoint = _LINEAGE_MODEL_DIR / "model_best.pt"
    if not checkpoint.is_file():
        return None
    try:
        runner = SeptumLineageInference(checkpoint, device="cpu")
        _model_cache[cache_key] = runner
        return runner
    except Exception as exc:
        print(f"Could not load lineage septum model: {exc}")
        return None


def _experiment_metadata_rules(exp: str) -> list[dict[str, str]]:
    config_path = (
        _PROJECT_ROOT
        / "SingleCellDataAnalysis"
        / "septum_lineage_experiments.json"
    )
    if not config_path.is_file():
        return []
    data = json.loads(config_path.read_text(encoding="utf-8"))
    for spec in data.get("experiments", []):
        if str(spec.get("id")) == str(exp):
            return list(spec.get("metadata_rules", []))
    return []


def _film_acquisition_metadata(
    base_root: Path,
    exp: str,
    film: str,
) -> Any:
    matches = [
        base_root / exp / rule["metadata"]
        for rule in _experiment_metadata_rules(exp)
        if re.search(rule["film_pattern"], film)
    ]
    if len(matches) != 1 or not matches[0].is_file():
        raise ValidationError(
            f"No unique acquisition metadata mapping is configured for {exp}/{film}."
        )
    return parse_acquisition_metadata(matches[0])


def aligned_to_local_frame(aligned: Optional[int], offset: int) -> Optional[int]:
    if aligned is None:
        return None
    return int(aligned) - int(offset)


def local_to_aligned_frame(local: Optional[int], offset: int) -> Optional[int]:
    if local is None:
        return None
    return int(local) + int(offset)


def resolve_local_endpoint(
    explicit_local: Optional[int],
    legacy_aligned: Optional[int],
    offset: int,
) -> Optional[int]:
    if explicit_local is not None:
        return int(explicit_local)
    return aligned_to_local_frame(legacy_aligned, offset)


def validate_local_interval(
    start: Optional[int],
    end: Optional[int],
    frame_count: int,
    label: str,
) -> None:
    for endpoint_name, value in (("start", start), ("end", end)):
        if value is not None and not (0 <= int(value) < int(frame_count)):
            raise ValidationError(
                f"{label} {endpoint_name} frame {value} is outside local film range "
                f"0–{max(0, int(frame_count) - 1)}."
            )
    if start is not None and end is not None and int(end) < int(start):
        raise ValidationError(f"{label} end frame {end} precedes start frame {start}.")


def endpoint_order_is_valid(
    start_candidate: Dict[str, Any],
    end_candidate: Dict[str, Any],
) -> bool:
    return float(end_candidate["time_min"]) >= float(start_candidate["time_min"])


def _get_cell_crop_tile(
    base_root: Path,
    exp: str,
    film: str,
    t: int,
    rle: str,
    pad: int = 10,
    tile_size: int = 96,
) -> Optional[np.ndarray]:
    try:
        from skimage.io import imread
        from Cell_tracking_functions import rle_decode

        frames_dir = base_root / exp / film / f"Frames_{film}"
        files = sorted(
            f for f in frames_dir.glob(f"{film}_t_{t:03d}_c_0.tif")
            if not f.name.startswith(".")
        )
        if not files:
            files = sorted(
                f for f in frames_dir.glob(f"*_t_{t:03d}_c_0.tif")
                if not f.name.startswith(".")
            )
        if not files:
            files = sorted(
                f for f in frames_dir.glob(f"{film}_t_{t:03d}_c_*.tif")
                if not f.name.startswith(".")
            )
        if not files:
            files = sorted(
                f for f in frames_dir.glob(f"*_t_{t:03d}_c_*.tif")
                if not f.name.startswith(".")
            )
        if not files:
            return None

        img = imread(str(files[0]))
        height, width = img.shape[:2]
        mask = rle_decode(rle, (height, width))
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = max(0, y0 - pad), min(height - 1, y1 + pad)
        x0, x1 = max(0, x0 - pad), min(width - 1, x1 + pad)
        crop = img[y0:y1 + 1, x0:x1 + 1]

        array = np.asarray(crop)
        if array.dtype != np.uint8:
            normalized = array.astype(np.float32)
            if np.isfinite(normalized).any():
                lo, hi = np.nanpercentile(normalized, [1, 99])
            else:
                lo, hi = 0.0, 1.0
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            normalized = np.clip((normalized - lo) / (hi - lo), 0, 1)
            array = (255 * normalized).astype(np.uint8)
        else:
            array = array.copy()

        target_h = target_w = int(tile_size)
        height, width = array.shape[:2]
        if height > target_h:
            y_start = (height - target_h) // 2
            array = array[y_start:y_start + target_h, :]
            height = target_h
        if width > target_w:
            x_start = (width - target_w) // 2
            array = array[:, x_start:x_start + target_w]
            width = target_w

        output = np.zeros((target_h, target_w), dtype=np.uint8)
        y_start = (target_h - height) // 2
        x_start = (target_w - width) // 2
        output[y_start:y_start + height, x_start:x_start + width] = array
        return output
    except Exception as exc:
        print(f"Error cropping cell at t={t}: {exc}")
        return None


def _rotation_angle_from_mask(mask: np.ndarray) -> float:
    """Return rotation angle (degrees, CCW) to align the cell's major axis vertically.

    ``skimage.measure.regionprops.orientation`` is the angle between the 0th
    axis (rows) and the major axis of the equivalent ellipse, measured CCW in
    radians, range [-π/2, π/2].

    Derivation
    ----------
    The major axis direction in (row, col) image coordinates for orientation θ
    is (cos θ, sin θ).  Rotating the *image* by angle α maps that vector to::

        (cos θ · cos α - sin θ · sin α,  cos θ · sin α + sin θ · cos α)

    Setting α = -θ yields (1, 0), i.e. the major axis points along rows
    (vertically).  Therefore ``rotation_deg = -degrees(θ)``.

    Spot-checks
    -----------
    * θ = 0   (already vertical)   → 0°  rotation (no-op)
    * θ = π/2 (horizontal cell)    → -90° CW rotation  → vertical ✓
    * θ = -π/2 (horizontal, other) → +90° CCW rotation → vertical ✓
    * θ = π/4 (45° diagonal)       → -45° rotation      → vertical ✓
    """
    try:
        from skimage.measure import regionprops, label as sk_label
        lab = sk_label((mask > 0).astype(np.uint8))
        props = regionprops(lab)
        if not props:
            return 0.0
        return float(-np.degrees(props[0].orientation))
    except Exception:
        return 0.0


def _rotate_point_in_crop(
    py: float,
    px: float,
    crop_h: int,
    crop_w: int,
    angle_deg: float,
) -> tuple[float, float]:
    """Rotate a (row, col) point about the crop centre by *angle_deg* (CCW).

    Matches the rotation applied to the image by ``scipy.ndimage.rotate`` with
    ``reshape=False``, which rotates about the image centre.
    """
    cy, cx = crop_h / 2.0, crop_w / 2.0
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dy, dx = py - cy, px - cx
    new_dy = dy * cos_a - dx * sin_a
    new_dx = dy * sin_a + dx * cos_a
    return cy + new_dy, cx + new_dx


class SeptumService:
    def __init__(
        self,
        base_movie_root: Path,
        qc_repo: QCRepository,
        audit_service: AuditService,
        al_service: Any = None,
    ):
        self.base_root = base_movie_root
        self.qc_repo = qc_repo
        self.audit_service = audit_service
        self.al_service = al_service


    def _find_sequence_linkage(self, exp: str, film: str, cell_id: int):
        linkage_path = self.base_root / exp / "sequence_linkage.json"
        if not linkage_path.is_file():
            return None, None
        try:
            with open(linkage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for seq_name, seq_data in data.items():
                films = seq_data.get("films", [])
                if film in films:
                    f_idx = films.index(film)
                    global_cells = seq_data.get("global_cells", {})
                    for global_cid, local_ids in global_cells.items():
                        if f_idx < len(local_ids) and local_ids[f_idx] == cell_id:
                            return films, local_ids
        except Exception:
            pass
        return None, None

    def _get_film_frame_count_and_size(self, exp: str, film: str) -> int:
        tracked_dir = self.base_root / exp / film / f"TrackedCells_{film}"
        if tracked_dir.exists():
            for cf in tracked_dir.iterdir():
                if cf.name.startswith("."): continue
                if cf.name.endswith("_masks.csv"):
                    try:
                        df = pd.read_csv(cf)
                        if not df.empty and "width" in df.columns:
                            return len(df)
                    except Exception:
                        continue
        frames_dir = self.base_root / exp / film / f"Frames_{film}"
        if frames_dir.exists():
            files = [f for f in frames_dir.glob("*.tif") if not f.name.startswith(".")]
            return len(files)
        return 0

    def _sequence_film_bounds(self, exp: str, films: list[str]) -> Dict[str, tuple[int, int]]:
        """
        Cumulative [b_start, b_end) frame-count bounds for each film in a
        linked sequence, in the SAME numbering the frontend's gallery/slider
        uses (state.filmBoundaries / getFilmSequenceBounds in state.js): film
        0 occupies [0, L0), film 1 occupies [L0, L0+L1), etc.

        This is THE authoritative sequence-wide coordinate space for a linked,
        multi-film cell. Both save_septum_label() (splitting one sequence-wide
        interval across films) and get_septum_alignment() (reassembling a
        sequence-wide interval from each film's own storage) must use this
        same helper so the two stay consistent with each other.
        """
        bounds: Dict[str, tuple[int, int]] = {}
        current_t = 0
        for f_name in films:
            length = self._get_film_frame_count_and_size(exp, f_name)
            bounds[f_name] = (current_t, current_t + length)
            current_t += length
        return bounds

    def get_septum_alignment(self, exp: str, film: str, cell_id: Optional[str] = None) -> Dict[str, Any]:
        data, rev = self.qc_repo.load_septum(exp, film)
        if cell_id is not None:
            cid_str = str(cell_id)
            offsets = data.get("offsets", {})
            cell_intervals = data.get("cell_intervals", {})

            offset = int(offsets.get(cid_str, 0))
            ci = cell_intervals.get(cid_str, {})

            # Check for sequence linkage to load merged values
            films, local_ids = self._find_sequence_linkage(exp, film, int(cell_id))
            if films and local_ids:
                # CROSS-FILM RECONSTRUCTION RULE
                #
                # A division can span a film boundary, so septum start/end may
                # live in two different films. Each film's own JSON stores its
                # own half of the interval as `start_aligned`/`end_aligned` =
                # local_frame_in_that_film + that_film's_own_offset - and
                # crucially, DIFFERENT films can have DIFFERENT stored offsets
                # (they're saved independently, per local cell id, per film).
                #
                # The old code decoded every film's aligned value using ONE
                # offset (whichever film happened to be the query's entry
                # point), which only produced correct numbers when all films'
                # offsets happened to match by coincidence - otherwise the
                # reassembled start/end would visibly change depending on
                # which film the frontend queried from. Fix: decode each
                # film's aligned value with THAT film's own offset, then place
                # it on the shared sequence-wide timeline via
                # _sequence_film_bounds() before combining. The result is a
                # true sequence frame number, independent of query entry
                # point, so we return "offset": 0 to tell the frontend it's
                # already fully resolved - no further offset math needed.
                film_bounds = self._sequence_film_bounds(exp, films)

                has_septum = False
                start_aligned = None
                end_aligned = None
                is_white_septum = False

                has_septum_2 = False
                start_aligned_2 = None
                end_aligned_2 = None
                is_white_septum_2 = False

                for f_name, lid in zip(films, local_ids):
                    if lid == -1:
                        continue
                    f_data, _ = self.qc_repo.load_septum(exp, f_name)
                    f_ci = f_data.get("cell_intervals", {}).get(str(lid), {})
                    f_offset = int(f_data.get("offsets", {}).get(str(lid), 0))
                    b_start, _b_end = film_bounds.get(f_name, (0, 0))

                    def _to_sequence_frame(aligned_value):
                        if aligned_value is None:
                            return None
                        return b_start + aligned_to_local_frame(aligned_value, f_offset)

                    if f_ci.get("has_septum"):
                        has_septum = True
                        s_a = _to_sequence_frame(f_ci.get("start_aligned"))
                        e_a = _to_sequence_frame(f_ci.get("end_aligned"))
                        if s_a is not None:
                            start_aligned = min(start_aligned, s_a) if start_aligned is not None else s_a
                        if e_a is not None:
                            end_aligned = max(end_aligned, e_a) if end_aligned is not None else e_a
                        if f_ci.get("white_septum"):
                            is_white_septum = True

                    if f_ci.get("has_septum_2"):
                        has_septum_2 = True
                        s_a2 = _to_sequence_frame(f_ci.get("start_aligned_2"))
                        e_a2 = _to_sequence_frame(f_ci.get("end_aligned_2"))
                        if s_a2 is not None:
                            start_aligned_2 = min(start_aligned_2, s_a2) if start_aligned_2 is not None else s_a2
                        if e_a2 is not None:
                            end_aligned_2 = max(end_aligned_2, e_a2) if end_aligned_2 is not None else e_a2
                        if f_ci.get("white_septum_2"):
                            is_white_septum_2 = True

                cell_data = {
                    "has_septum": has_septum,
                    "start_aligned": start_aligned,
                    "end_aligned": end_aligned,
                    "is_white_septum": is_white_septum,

                    "has_septum_2": has_septum_2,
                    "start_aligned_2": start_aligned_2,
                    "end_aligned_2": end_aligned_2,
                    "is_white_septum_2": is_white_septum_2,

                    # Already resolved to a sequence-wide frame above - tell
                    # the frontend not to subtract anything further.
                    "offset": 0,
                }
            else:
                cell_data = {
                    "has_septum": bool(ci.get("has_septum", False)),
                    "start_aligned": ci.get("start_aligned"),
                    "end_aligned": ci.get("end_aligned"),
                    "is_white_septum": bool(ci.get("white_septum", False)),
                    
                    "has_septum_2": bool(ci.get("has_septum_2", False)),
                    "start_aligned_2": ci.get("start_aligned_2"),
                    "end_aligned_2": ci.get("end_aligned_2"),
                    "is_white_septum_2": bool(ci.get("white_septum_2", False)),
                    
                    "offset": offset
                }
            # Check Local QC status for mistracked flag
            is_mistracked = False
            if cell_id is not None:
                from ..qc_schema import LocalCellQC
                rev_state = self.qc_repo.load_review_state(exp, film)
                if not rev_state and "_" in film:
                    parts = film.split('_')
                    seq_target = f"{parts[0]}_{parts[-1]}"
                    rev_state = self.qc_repo.load_review_state(exp, seq_target)
                cell_st = QCRepository.get_status_for_cell(rev_state, cell_id)
                is_mistracked = (cell_st == LocalCellQC.MISTRACKED.value)

            cell_data["mistracked"] = is_mistracked

            return {
                "status": "success",
                "data": cell_data,
                "revision": rev
            }
            
        return {
            "status": "success",
            "alignment": data,
            "revision": rev
        }

    def _save_single_film_label(self, req: SaveSeptumRequest, user: str = "anonymous") -> Dict[str, Any]:
        tracked_dir = (
            self.base_root / req.experiment / req.film / f"TrackedCells_{req.film}"
        )
        csv_path_cell = tracked_dir / f"cell_{req.cell_id}_masks.csv"
        if not csv_path_cell.exists():
            raise NotFoundError(f"Cell masks CSV not found: {csv_path_cell}")

        df_cell = pd.read_csv(csv_path_cell)
        frame_count = len(df_cell)
        if frame_count == 0:
            raise ValidationError(f"Cell mask sequence is empty: {csv_path_cell}")

        offset = int(req.offset)
        local_start = resolve_local_endpoint(req.start_frame, req.start_aligned, offset)
        local_end = resolve_local_endpoint(req.end_frame, req.end_aligned, offset)
        local_start_2 = resolve_local_endpoint(
            req.start_frame_2, req.start_aligned_2, offset
        )
        local_end_2 = resolve_local_endpoint(
            req.end_frame_2, req.end_aligned_2, offset
        )

        has_second = bool(req.has_septum_2)
        if not req.has_septum:
            local_start = local_end = None
        if not has_second:
            local_start_2 = local_end_2 = None

        validate_local_interval(local_start, local_end, frame_count, "Septum 1")
        validate_local_interval(local_start_2, local_end_2, frame_count, "Septum 2")

        start_aligned = local_to_aligned_frame(local_start, offset)
        end_aligned = local_to_aligned_frame(local_end, offset)
        start_aligned_2 = local_to_aligned_frame(local_start_2, offset)
        end_aligned_2 = local_to_aligned_frame(local_end_2, offset)

        data, old_rev = self.qc_repo.load_septum(req.experiment, req.film)
        cell_intervals = data.setdefault("cell_intervals", {})
        offsets = data.setdefault("offsets", {})
        cid_str = str(req.cell_id)

        # Check for AI prediction override/correction
        ci_old = cell_intervals.get(cid_str, {})
        if ci_old.get("label_source") == "ai":
            old_has = bool(ci_old.get("has_septum", False))
            old_start = ci_old.get("start_aligned")
            old_end = ci_old.get("end_aligned")
            
            new_has = bool(req.has_septum)
            new_start = start_aligned
            new_end = end_aligned

            if (old_has != new_has) or (old_start != new_start) or (old_end != new_end):
                try:
                    corrections_dir = self.base_root / ".tracking_corrector"
                    corrections_dir.mkdir(parents=True, exist_ok=True)
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "user": user,
                        "experiment": req.experiment,
                        "film": req.film,
                        "cell_id": cid_str,
                        "original_ai": {
                            "has_septum": old_has,
                            "start_aligned": old_start,
                            "end_aligned": old_end
                        },
                        "corrected_human": {
                            "has_septum": new_has,
                            "start_aligned": new_start,
                            "end_aligned": new_end
                        }
                    }
                    log_file = corrections_dir / "septum_corrections.jsonl"
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception as exc:
                    print(f"Error logging septum correction: {exc}")

        offsets[cid_str] = offset

        label_source = getattr(req, "label_source", None) or "cell"

        # Update pattern_center_row/col in cell_data.csv if provided
        if (
            getattr(req, "pattern_center_row", None) is not None
            and getattr(req, "pattern_center_col", None) is not None
            and local_start is not None
        ):
            data_csv = tracked_dir / f"cell_{req.cell_id}_data.csv"
            if data_csv.is_file():
                try:
                    df_data = pd.read_csv(data_csv)
                    if "time_point" in df_data.columns:
                        mask_dp = df_data["time_point"] == local_start
                        if mask_dp.any():
                            df_data.loc[mask_dp, "pattern_center_row"] = float(
                                req.pattern_center_row
                            )
                            df_data.loc[mask_dp, "pattern_center_col"] = float(
                                req.pattern_center_col
                            )
                            df_data.to_csv(data_csv, index=False)
                except Exception as exc:
                    print(f"Error updating pattern_center in cell_data.csv: {exc}")

        cell_intervals[cid_str] = {
            "has_septum": bool(req.has_septum),
            "start_aligned": start_aligned,
            "end_aligned": end_aligned,
            "white_septum": bool(req.is_white_septum),
            "has_septum_2": has_second,
            "start_aligned_2": start_aligned_2,
            "end_aligned_2": end_aligned_2,
            "white_septum_2": bool(req.is_white_septum_2),
            "label_source": label_source,
        }

        new_rev = self.qc_repo.save_septum(req.experiment, req.film, data)
        source_path = self.qc_repo.get_septum_json_path(req.experiment, req.film)

        compiled_csv = {"status": "ok"}
        try:
            import re
            label_dir = source_path.parent
            all_cids = []
            if tracked_dir.is_dir():
                for f in tracked_dir.iterdir():
                    if f.name.startswith("."):
                        continue
                    m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
                    if m:
                        all_cids.append(int(m.group(1)))
            all_cids.sort()

            gi = data.get("global_interval", {})
            a_left = int(gi.get("G0", 0))
            csv_path = label_dir / "septum_interval_per_cell.csv"
            rows = []
            for cid in all_cids:
                c_str = str(cid)
                ci = cell_intervals.get(c_str, {})
                rows.append({
                    "cell_id": cid,
                    "a_left": a_left,
                    "start_aligned": ci.get("start_aligned") if ci.get("start_aligned") is not None else "",
                    "end_aligned": ci.get("end_aligned") if ci.get("end_aligned") is not None else "",
                    "has": 1 if ci.get("has_septum") else 0,
                    "white_septum": 1 if ci.get("white_septum") else 0,
                    "start_aligned_2": ci.get("start_aligned_2") if ci.get("start_aligned_2") is not None else "",
                    "end_aligned_2": ci.get("end_aligned_2") if ci.get("end_aligned_2") is not None else "",
                    "has_2": 1 if ci.get("has_septum_2") else 0,
                    "white_septum_2": 1 if ci.get("white_septum_2") else 0,
                })
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            compiled_csv["path"] = str(csv_path)
        except Exception as exc:
            compiled_csv = {"status": "error", "message": str(exc)}
            print(f"Error compiling septum_interval_per_cell.csv: {exc}")

        training_export = {"status": "ok"}
        try:
            rle_col = "rle_bf"
            if "rle_gfp" in df_cell.columns and df_cell["rle_gfp"].dropna().any():
                rle_col = "rle_gfp"

            tiles = []
            for t in range(frame_count):
                rle = df_cell.iloc[t][rle_col]
                tile = None
                if isinstance(rle, str) and rle.strip():
                    tile = _get_cell_crop_tile(
                        self.base_root, req.experiment, req.film, t, rle
                    )
                if tile is None:
                    tile = np.zeros((96, 96), dtype=np.uint8)
                tiles.append(tile)
            strip = np.hstack(tiles)

            from SingleCellDataAnalysis.septum_training_utils import (
                export_cell_training_sample,
            )

            exported_path = export_cell_training_sample(
                working_dir=str(self.base_root / req.experiment),
                film_name=req.film,
                cell_id=int(req.cell_id),
                strip=strip,
                tp0=0,
                offset=offset,
                start_idx=local_start if req.has_septum and local_start is not None else -1,
                end_idx=local_end if req.has_septum and local_end is not None else -1,
                start_idx_2=local_start_2 if has_second and local_start_2 is not None else -1,
                end_idx_2=local_end_2 if has_second and local_end_2 is not None else -1,
                label_source=label_source,
                start_aligned=start_aligned,
                end_aligned=end_aligned,
                start_aligned_2=start_aligned_2,
                end_aligned_2=end_aligned_2,
                white_septum=bool(req.is_white_septum),
                white_septum_2=bool(req.is_white_septum_2),
            )
            training_export["path"] = str(exported_path)

        except Exception as exc:
            training_export = {"status": "error", "message": str(exc)}
            print(f"Error exporting training sample for cell {req.cell_id}: {exc}")

        self.audit_service.record_revision(
            user=user,
            experiment=req.experiment,
            film_or_sequence=req.film,
            cell_id=cid_str,
            operation="save_septum_label",
            source_file_path=source_path,
            old_checksum=old_rev,
            new_checksum=new_rev,
            changed_frames=[]
        )

        return {
            "status": "success",
            "film": req.film,
            "cell_id": cid_str,
            "new_revision": new_rev,
            "compiled_csv": compiled_csv,
            "training_export": training_export,
        }

    def save_septum_label(self, req: SaveSeptumRequest, user: str = "anonymous") -> Dict[str, Any]:
        films, local_ids = self._find_sequence_linkage(req.experiment, req.film, int(req.cell_id))

        if films and local_ids:
            # Same sequence-wide bounds get_septum_alignment() uses to
            # reassemble an interval on load - keep these on the shared
            # helper so save and load can never drift apart.
            film_bounds = self._sequence_film_bounds(req.experiment, films)

            last_res = {"status": "success"}
            for f_name, lid in zip(films, local_ids):
                if lid == -1:
                    continue
                b_start, b_end = film_bounds[f_name]
                
                f_has_septum = False
                f_local_start = None
                f_local_end = None
                
                if req.has_septum:
                    start_val = req.start_aligned if req.start_aligned is not None else b_start
                    end_val = req.end_aligned if req.end_aligned is not None else b_end
                    
                    if (start_val < b_end) and (end_val >= b_start):
                        f_has_septum = True
                        if req.start_aligned is not None and b_start <= req.start_aligned < b_end:
                            f_local_start = req.start_aligned - b_start
                        if req.end_aligned is not None and b_start <= req.end_aligned < b_end:
                            f_local_end = req.end_aligned - b_start
                            
                f_has_septum_2 = False
                f_local_start_2 = None
                f_local_end_2 = None
                
                if req.has_septum_2:
                    start_val_2 = req.start_aligned_2 if req.start_aligned_2 is not None else b_start
                    end_val_2 = req.end_aligned_2 if req.end_aligned_2 is not None else b_end
                    
                    if (start_val_2 < b_end) and (end_val_2 >= b_start):
                        f_has_septum_2 = True
                        if req.start_aligned_2 is not None and b_start <= req.start_aligned_2 < b_end:
                            f_local_start_2 = req.start_aligned_2 - b_start
                        if req.end_aligned_2 is not None and b_start <= req.end_aligned_2 < b_end:
                            f_local_end_2 = req.end_aligned_2 - b_start
                            
                film_req = SaveSeptumRequest(
                    experiment=req.experiment,
                    film=f_name,
                    cell_id=str(lid),
                    has_septum=f_has_septum,
                    start_frame=f_local_start,
                    end_frame=f_local_end,
                    start_aligned=local_to_aligned_frame(f_local_start, req.offset) if f_local_start is not None else None,
                    end_aligned=local_to_aligned_frame(f_local_end, req.offset) if f_local_end is not None else None,
                    is_white_septum=req.is_white_septum,
                    has_septum_2=f_has_septum_2,
                    start_frame_2=f_local_start_2,
                    end_frame_2=f_local_end_2,
                    start_aligned_2=local_to_aligned_frame(f_local_start_2, req.offset) if f_local_start_2 is not None else None,
                    end_aligned_2=local_to_aligned_frame(f_local_end_2, req.offset) if f_local_end_2 is not None else None,
                    is_white_septum_2=req.is_white_septum_2,
                    offset=req.offset,
                    sequence=req.sequence,
                    global_cell_id=req.global_cell_id,
                    film_index=films.index(f_name),
                    label_source=req.label_source,
                    pattern_center_row=req.pattern_center_row,
                    pattern_center_col=req.pattern_center_col,
                    note=req.note,
                    annotator=req.annotator
                )

                
                last_res = self._save_single_film_label(film_req, user)
                
            return last_res
        else:
            return self._save_single_film_label(req, user)

    def _prediction_segments(
        self,
        exp: str,
        film: str,
        cell_id: int,
        sequence: Optional[str],
        global_cell_id: Optional[str],
    ) -> list[tuple[str, int, int, int]]:
        if not sequence or global_cell_id is None:
            return [(film, int(cell_id), 0, 0)]

        linkage, _ = LinkageRepository(self.base_root).load_linkage(exp)
        sequences = linkage.get("sequences", linkage)
        if sequence not in sequences:
            raise NotFoundError(f"Sequence not found: {sequence}")
        sequence_data = sequences[sequence]
        films = list(sequence_data.get("films", []))
        local_ids = list(
            sequence_data.get("global_cells", {}).get(str(global_cell_id), [])
        )
        if not films:
            raise ValidationError(f"Sequence {sequence} contains no films.")

        result: list[tuple[str, int, int, int]] = []
        sequence_offset = 0
        for index, segment_film in enumerate(films):
            local_id = int(local_ids[index]) if index < len(local_ids) else -1
            tracked_dir = (
                self.base_root
                / exp
                / segment_film
                / f"TrackedCells_{segment_film}"
            )
            frame_count = 0
            candidate = tracked_dir / f"cell_{local_id}_masks.csv"
            if local_id >= 0 and candidate.is_file():
                frame_count = len(pd.read_csv(candidate))
                result.append((segment_film, local_id, sequence_offset, index))
            elif tracked_dir.is_dir():
                first_csv = next(
                    (
                        path
                        for path in sorted(tracked_dir.glob("cell_*_masks.csv"))
                        if not path.name.startswith(".")
                    ),
                    None,
                )
                if first_csv is not None:
                    frame_count = len(pd.read_csv(first_csv))
            sequence_offset += frame_count
        if not result:
            raise ValidationError(
                f"No linked cell observations found for {global_cell_id} in {sequence}."
            )
        return result

    def predict_septum(
        self,
        exp: str,
        film: str,
        cell_id: int,
        sequence: Optional[str] = None,
        global_cell_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        runner = get_lineage_inference_runner()
        if runner is None:
            raise ValidationError(
                "Review-only lineage model checkpoint not found or could not be loaded."
            )

        segments = self._prediction_segments(
            exp,
            film,
            cell_id,
            sequence,
            global_cell_id,
        )
        metadata = {
            segment_film: _film_acquisition_metadata(
                self.base_root,
                exp,
                segment_film,
            )
            for segment_film, _, _, _ in segments
        }
        experiment_t0 = min(item.acquired_at for item in metadata.values())
        tiles: list[np.ndarray] = []
        times: list[float] = []
        modalities: list[int] = []
        boundaries: list[float] = []
        frame_map: list[dict[str, Any]] = []

        for segment_film, local_id, sequence_offset, film_index in segments:
            csv_path = (
                self.base_root
                / exp
                / segment_film
                / f"TrackedCells_{segment_film}"
                / f"cell_{local_id}_masks.csv"
            )
            if not csv_path.is_file():
                raise NotFoundError(f"Cell masks CSV not found: {csv_path}")
            df = pd.read_csv(csv_path)
            modality = infer_modality(segment_film, film_index)
            preferred = "rle_gfp" if modality == "gfp" else "rle_bf"
            fallback = "rle_bf" if preferred == "rle_gfp" else "rle_gfp"
            rle_col = preferred if preferred in df.columns else fallback
            if rle_col not in df.columns:
                raise ValidationError(f"No RLE mask column found in {csv_path}")

            acquisition = metadata[segment_film]
            start_min = (
                acquisition.acquired_at - experiment_t0
            ).total_seconds() / 60.0
            for local_frame in range(len(df)):
                rle = df.iloc[local_frame][rle_col]
                tile = None
                if isinstance(rle, str) and rle.strip():
                    tile = _get_cell_crop_tile(
                        self.base_root,
                        exp,
                        segment_film,
                        local_frame,
                        rle,
                    )
                if tile is None:
                    tile = np.zeros((96, 96), dtype=np.uint8)
                tiles.append(tile)
                times.append(
                    start_min + local_frame * float(acquisition.interval_min)
                )
                modalities.append(1 if modality == "gfp" else 0)
                boundaries.append(1.0 if local_frame == 0 else 0.0)
                frame_map.append(
                    {
                        "sequence_frame": int(sequence_offset + local_frame),
                        "film": segment_film,
                        "film_index": film_index,
                        "local_cell_id": local_id,
                        "local_frame": local_frame,
                        "time_min": times[-1],
                        "modality": modality,
                    }
                )

        try:
            order = np.argsort(np.asarray(times), kind="stable")
            ordered_tiles = np.stack(tiles)[order]
            ordered_times = np.asarray(times, dtype=np.float32)[order]
            ordered_modalities = np.asarray(modalities, dtype=np.int64)[order]
            ordered_boundaries = np.asarray(boundaries, dtype=np.float32)[order]
            frame_map = [frame_map[int(index)] for index in order]
            output = runner.predict(
                ordered_tiles,
                ordered_times,
                ordered_modalities,
                ordered_boundaries,
            )
            state_probs = output["state"]
            start_probs = output["start"]
            end_probs = output["end"]
            peak_index = int(np.argmax(state_probs))
            start_index = int(np.argmax(start_probs))
            end_index = int(np.argmax(end_probs))
            raw_start = frame_map[start_index]
            raw_end = frame_map[end_index]
            endpoint_order_valid = endpoint_order_is_valid(raw_start, raw_end)
            warning = (
                "Suggestion only: held-out endpoint F1 is 0.240. "
                "Verify both endpoints before saving."
            )
            if not endpoint_order_valid:
                warning += (
                    " The raw endpoint peaks were temporally impossible, "
                    "so no interval suggestion is shown."
                )
            return {
                "status": "success",
                "review_only": True,
                "model_id": "lineage_v1",
                "warning": warning,
                "model_metrics": _LINEAGE_MODEL_METRICS,
                "scope": "full_sequence" if sequence else "single_film",
                "probs": state_probs.tolist(),
                "start_probs": start_probs.tolist(),
                "end_probs": end_probs.tolist(),
                "sequence_indices": [
                    int(item["sequence_frame"]) for item in frame_map
                ],
                "peak_t": int(frame_map[peak_index]["sequence_frame"]),
                "peak_prob": float(state_probs[peak_index]),
                "endpoint_order_valid": endpoint_order_valid,
                "suggested_start": raw_start if endpoint_order_valid else None,
                "suggested_end": raw_end if endpoint_order_valid else None,
                "raw_start_candidate": raw_start,
                "raw_end_candidate": raw_end,
                "start_confidence_uncalibrated": float(start_probs[start_index]),
                "end_confidence_uncalibrated": float(end_probs[end_index]),
                "has_event_confidence_uncalibrated": output["has_event"],
            }
        except Exception as e:
            raise ValidationError(f"Inference exception: {e}")

    def rotated_crop_with_prediction(
        self,
        exp: str,
        film: str,
        cell_id: int,
        frame: int,
        channel: str = "bf",
        crop_size: int = 160,
    ) -> Dict[str, Any]:
        """Return a rotation-normalised single-frame crop plus per-frame AI probabilities.

        The crop is rotated so the cell's major axis (from ``regionprops.orientation``
        computed on the RLE mask) runs along the image *height*.  The existing
        ``predict_septum()`` inference path is reused — no second model instance
        or separate forward pass is created.

        Parameters
        ----------
        exp : str
            Experiment ID (e.g. ``"M156"``).
        film : str
            Film name (e.g. ``"3_BF2_F0"``).
        cell_id : int
            Local cell identifier within this film.
        frame : int
            Zero-based local frame index (``time_point`` column in the masks CSV).
        channel : str
            ``"bf"`` or ``"gfp"`` — selects which frame file and RLE column to use.
        crop_size : int
            Side length (pixels) of the square crop before *and* after rotation.

        Returns
        -------
        dict with keys:
            ``image_b64``            base64-encoded JPEG of the rotated crop
            ``rotation_deg``         applied rotation (CCW degrees; negative = CW)
            ``crop_size``            actual crop side length used
            ``centroid_in_crop``     [row, col] of mask centroid in the padded crop
            ``state_prob``           per-frame septum-state probability (0–1) or null
            ``start_prob``           per-frame septum-start probability (0–1) or null
            ``end_prob``             per-frame septum-end probability (0–1) or null
            ``warning``              model confidence caveat string
            ``model_metrics``        dict of held-out benchmark values
            ``overlay``              sub-dict with ``septum_center_in_crop`` ([row, col] or null)
        """
        import base64
        import io as _io
        from PIL import Image as _PILImage
        from scipy.ndimage import rotate as _ndimage_rotate
        from skimage.io import imread as _sk_imread

        # ---- 1. Load masks CSV ----------------------------------------
        csv_path = (
            self.base_root / exp / film
            / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
        )
        if not csv_path.is_file():
            raise NotFoundError(f"Cell masks CSV not found: {csv_path}")
        df_masks = pd.read_csv(csv_path)

        # ---- 2. Find RLE for the requested frame ----------------------
        preferred_rle = "rle_gfp" if channel == "gfp" else "rle_bf"
        fallback_rle = "rle_bf" if preferred_rle == "rle_gfp" else "rle_gfp"
        rle_col = preferred_rle if preferred_rle in df_masks.columns else fallback_rle
        if rle_col not in df_masks.columns:
            raise ValidationError(f"No RLE column found in {csv_path}")

        if "time_point" in df_masks.columns:
            mask_rows = df_masks[df_masks["time_point"] == frame]
        elif frame < len(df_masks):
            mask_rows = df_masks.iloc[[frame]]
        else:
            mask_rows = pd.DataFrame()

        if mask_rows.empty:
            raise NotFoundError(
                f"Frame {frame} not found in {csv_path} "
                f"(total rows: {len(df_masks)})"
            )
        rle = str(mask_rows.iloc[0].get(rle_col, ""))

        # ---- 3. Locate and load the raw frame image -------------------
        c_num = 1 if channel == "gfp" else 0
        frames_dir = self.base_root / exp / film / f"Frames_{film}"
        frame_files = sorted(
            f for f in frames_dir.glob(f"*_t_{frame:03d}_c_{c_num}.tif")
            if not f.name.startswith(".")
        )
        if not frame_files:
            # Graceful fallback to channel 0
            frame_files = sorted(
                f for f in frames_dir.glob(f"*_t_{frame:03d}_c_0.tif")
                if not f.name.startswith(".")
            )
        if not frame_files:
            raise NotFoundError(
                f"Frame image not found for t={frame} in {frames_dir}"
            )
        img = _sk_imread(str(frame_files[0]))
        img_h, img_w = img.shape[:2]

        # ---- 4. Decode mask → centroid + rotation angle ---------------
        rotation_deg = 0.0
        cy, cx = img_h // 2, img_w // 2   # fallback if no mask

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if rle and rle.strip() and rle.lower() != "nan":
            try:
                from ..schemas import validate_and_decode_rle as _decode_rle
                mask = _decode_rle(rle, img_h, img_w)
                ys, xs = np.where(mask > 0)
                if ys.size > 0:
                    cy = int(np.mean(ys))
                    cx = int(np.mean(xs))
                    rotation_deg = _rotation_angle_from_mask(mask)
            except Exception:
                pass

        # ---- 5. Crop a square window centred on the mask centroid -----
        half = crop_size // 2
        y0 = max(0, cy - half)
        y1 = min(img_h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(img_w, cx + half)

        crop_raw = img[y0:y1, x0:x1]
        if crop_raw.size == 0:
            crop_raw = np.zeros((crop_size, crop_size), dtype=np.uint8)

        # Pad symmetrically to exactly crop_size × crop_size when the crop
        # was clipped at an image boundary (cell near the edge).
        padded = np.zeros((crop_size, crop_size), dtype=crop_raw.dtype)
        ph = min(crop_raw.shape[0], crop_size)
        pw = min(crop_raw.shape[1], crop_size)
        oy = (crop_size - ph) // 2
        ox = (crop_size - pw) // 2
        padded[oy : oy + ph, ox : ox + pw] = crop_raw[:ph, :pw]
        crop_img = padded

        # Track where the mask centroid ended up in the padded crop
        centroid_crop_y = float(cy - y0 + oy)
        centroid_crop_x = float(cx - x0 + ox)

        # ---- 6. Normalise intensity (p1–p99.5 stretch to uint8) -------
        af = crop_img.astype(np.float32)
        flat = af.ravel()
        valid = flat[flat > 0] if flat.any() else flat
        if valid.size > 1:
            p_lo = float(np.percentile(valid, 1.0))
            p_hi = float(np.percentile(valid, 99.5))
        else:
            p_lo, p_hi = 0.0, 255.0
        if p_hi > p_lo:
            crop_norm = np.clip(
                (af - p_lo) / (p_hi - p_lo) * 255.0, 0, 255
            ).astype(np.uint8)
        else:
            crop_norm = crop_img.astype(np.uint8)

        # ---- 7. Rotate crop around its centre -------------------------
        # scipy.ndimage.rotate rotates CCW for positive angles, about the
        # array centre, with reshape=False (output stays crop_size × crop_size).
        rotated_img = _ndimage_rotate(
            crop_norm,
            angle=rotation_deg,
            reshape=False,
            mode="constant",
            cval=0,
            order=1,   # bilinear — fast and artefact-free for uint8
        )

        # ---- 8. Encode to JPEG → base64 --------------------------------
        pil_img = _PILImage.fromarray(rotated_img)
        buf = _io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # ---- 9. Transform labeled septum centre into rotated crop ------
        overlay_septum = None
        data_csv = (
            self.base_root / exp / film
            / f"TrackedCells_{film}" / f"cell_{cell_id}_data.csv"
        )
        if data_csv.is_file():
            try:
                df_data = pd.read_csv(data_csv)
                if "time_point" in df_data.columns:
                    data_row = df_data[df_data["time_point"] == frame]
                elif frame < len(df_data):
                    data_row = df_data.iloc[[frame]]
                else:
                    data_row = pd.DataFrame()

                if not data_row.empty:
                    sc_col = data_row.iloc[0].get("pattern_center_col")
                    sc_row = data_row.iloc[0].get("pattern_center_row")
                    if pd.notna(sc_col) and pd.notna(sc_row):
                        # Full-image → crop → rotated-crop coordinates
                        sc_crop_y = float(sc_row) - y0 + oy
                        sc_crop_x = float(sc_col) - x0 + ox
                        rot_y, rot_x = _rotate_point_in_crop(
                            sc_crop_y, sc_crop_x,
                            crop_size, crop_size,
                            rotation_deg,
                        )
                        overlay_septum = [round(rot_y, 1), round(rot_x, 1)]
            except Exception:
                pass

        # ---- 10. Extract per-frame AI probabilities -------------------
        # Reuse the existing predict_septum() path — no second inference.
        # For single-film mode, sequence_indices[i] == local frame index.
        state_prob = start_prob = end_prob = None
        warning = None
        model_metrics = None
        try:
            pred = self.predict_septum(exp, film, cell_id)
            warning = pred.get("warning")
            model_metrics = pred.get("model_metrics")
            seq_indices: list[int] = pred.get("sequence_indices", [])
            probs_all: list[float] = pred.get("probs", [])
            start_probs_all: list[float] = pred.get("start_probs", [])
            end_probs_all: list[float] = pred.get("end_probs", [])
            if frame in seq_indices:
                i = seq_indices.index(frame)
                if i < len(probs_all):
                    state_prob = round(float(probs_all[i]), 4)
                if i < len(start_probs_all):
                    start_prob = round(float(start_probs_all[i]), 4)
                if i < len(end_probs_all):
                    end_prob = round(float(end_probs_all[i]), 4)
        except Exception as exc:
            warning = (warning or "") + f" [inference unavailable: {exc}]"

        return {
            "status": "success",
            "image_b64": image_b64,
            "rotation_deg": round(rotation_deg, 2),
            "crop_size": crop_size,
            "centroid_in_crop": [
                round(centroid_crop_y, 1),
                round(centroid_crop_x, 1),
            ],
            "state_prob": state_prob,
            "start_prob": start_prob,
            "end_prob": end_prob,
            "warning": warning,
            "model_metrics": model_metrics,
            "overlay": {
                "septum_center_in_crop": overlay_septum,
            },
        }

    def get_cached_ai_suggestion(
        self,
        exp: str,
        sequence: str,
        global_cell_id: str,
    ) -> Dict[str, Any]:
        """Look up a pre-computed AI suggestion from an offline batch run."""
        cache = self.qc_repo.load_ai_cache(exp, sequence)
        suggestions = cache.get("suggestions", {})
        entry = suggestions.get(str(global_cell_id))
        if entry is None:
            return {"status": "success", "cached": False}
        return {
            "status": "success",
            "cached": True,
            "generated_at": cache.get("generated_at"),
            "source": cache.get("source"),
            "data": entry,
        }

    # =========================================================================
    # Mobile Septum Review Tool Methods
    # =========================================================================

    def _get_mobile_septum_state_path(self, exp: str, film: str) -> Path:
        exp_dir = self.qc_repo._resolve_exp_dir(exp)
        return exp_dir / film / f"mobile_septum_review_state_{film}.json"

    def _load_mobile_septum_state(self, exp: str, film: str) -> Dict[str, Any]:
        path = self._get_mobile_septum_state_path(exp, film)
        if not path.is_file():
            return {"mobile_labels_saved_count": 0, "reviewed_frames": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                data.setdefault("mobile_labels_saved_count", 0)
                data.setdefault("reviewed_frames", {})
                return data
        except Exception:
            return {"mobile_labels_saved_count": 0, "reviewed_frames": {}}

    def _save_mobile_septum_state(self, exp: str, film: str, state_data: Dict[str, Any]) -> None:
        path = self._get_mobile_septum_state_path(exp, film)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(state_data, indent=2)
        from ..repositories.mask_repository import atomic_write_text
        atomic_write_text(path, content)

    def get_mobile_septum_stats(self, exp: str, film: str) -> Dict[str, Any]:
        state = self._load_mobile_septum_state(exp, film)
        return {
            "status": "success",
            "experiment": exp,
            "film": film,
            "mobile_labels_saved_count": int(state.get("mobile_labels_saved_count", 0)),
            "total_reviewed": len(state.get("reviewed_frames", {})),
        }

    def get_mobile_septum_batch(
        self,
        exp: str,
        film: str,
        count: int = 20,
        crop_size: int = 160,
    ) -> Dict[str, Any]:
        state = self._load_mobile_septum_state(exp, film)
        reviewed = state.get("reviewed_frames", {})

        tracked_dir = (
            self.qc_repo._resolve_exp_dir(exp) / film / f"TrackedCells_{film}"
        )
        if not tracked_dir.is_dir():
            return {
                "status": "success",
                "experiment": exp,
                "film": film,
                "items": [],
                "mobile_labels_saved_count": state.get("mobile_labels_saved_count", 0),
            }

        items = []
        cell_files = sorted(
            [f for f in tracked_dir.glob("cell_*_masks.csv") if not f.name.startswith(".")],
            key=lambda p: int(re.search(r"cell_(\d+)_masks", p.name).group(1)) if re.search(r"cell_(\d+)_masks", p.name) else 0
        )

        for cf in cell_files:
            if len(items) >= count:
                break
            m = re.search(r"cell_(\d+)_masks", cf.name)
            if not m:
                continue
            cid = int(m.group(1))
            try:
                df = pd.read_csv(cf)
                if df.empty:
                    continue
                frames = list(range(0, len(df), max(1, len(df) // 3)))[:3]
                for f_idx in frames:
                    key = f"{cid}_{f_idx}"
                    if key in reviewed:
                        continue
                    try:
                        crop_data = self.rotated_crop_with_prediction(
                            exp, film, cid, f_idx, channel="bf", crop_size=crop_size
                        )
                        crop_data["experiment"] = exp
                        crop_data["film"] = film
                        crop_data["cell_id"] = cid
                        crop_data["frame"] = f_idx
                        crop_data["item_key"] = key
                        items.append(crop_data)
                        if len(items) >= count:
                            break
                    except Exception:
                        continue
            except Exception:
                continue

        return {
            "status": "success",
            "experiment": exp,
            "film": film,
            "items": items,
            "mobile_labels_saved_count": int(state.get("mobile_labels_saved_count", 0)),
        }

    def save_mobile_septum_review(self, data: Dict[str, Any], user: str = "anonymous") -> Dict[str, Any]:
        from skimage.io import imread as _sk_imread
        exp = data.get("experiment")
        film = data.get("film")
        cell_id_val = data.get("cell_id")
        frame_val = data.get("frame")
        has_septum = bool(data.get("has_septum", False))
        stroke_center = data.get("stroke_center_in_crop")  # [y_rot, x_rot]
        crop_size = int(data.get("crop_size", 160))

        if not exp or not film or cell_id_val is None or frame_val is None:
            raise ValidationError("experiment, film, cell_id, and frame are required")

        cell_id = int(cell_id_val)
        frame = int(frame_val)

        pattern_row = None
        pattern_col = None

        if has_septum and stroke_center and len(stroke_center) == 2:
            try:
                y_rot, x_rot = float(stroke_center[0]), float(stroke_center[1])
                csv_path = (
                    self.qc_repo._resolve_exp_dir(exp) / film
                    / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
                )
                df_masks = pd.read_csv(csv_path)
                rle_col = "rle_bf" if "rle_bf" in df_masks.columns else ("rle_gfp" if "rle_gfp" in df_masks.columns else None)
                if rle_col:
                    rle = str(df_masks.iloc[frame][rle_col])
                    frames_dir = self.qc_repo._resolve_exp_dir(exp) / film / f"Frames_{film}"
                    f_files = sorted([f for f in frames_dir.glob("*_t_*_c_*.tif") if not f.name.startswith(".")])
                    if f_files:
                        img = _sk_imread(str(f_files[0]))
                        h, w = img.shape[:2]
                        from ..schemas import validate_and_decode_rle as _decode_rle
                        mask = _decode_rle(rle, h, w)
                        rotation_deg = _rotation_angle_from_mask(mask)
                        ys, xs = np.where(mask > 0)
                        if ys.size > 0:
                            cy, cx = int(np.mean(ys)), int(np.mean(xs))
                            half = crop_size // 2
                            y0 = max(0, cy - half)
                            x0 = max(0, cx - half)
                            ph = min(h - y0, crop_size)
                            pw = min(w - x0, crop_size)
                            oy = (crop_size - ph) // 2
                            ox = (crop_size - pw) // 2

                            # Inverse rotate by -rotation_deg
                            y_crop, x_crop = _rotate_point_in_crop(
                                y_rot, x_rot, crop_size, crop_size, -rotation_deg
                            )
                            pattern_row = round(y_crop - oy + y0, 1)
                            pattern_col = round(x_crop - ox + x0, 1)
            except Exception as exc:
                print(f"Inverse rotation transform error: {exc}")

        req = SaveSeptumRequest(
            experiment=exp,
            film=film,
            cell_id=str(cell_id),
            has_septum=has_septum,
            start_frame=frame if has_septum else None,
            end_frame=frame if has_septum else None,
            label_source="mobile_septum_review",
            pattern_center_row=pattern_row,
            pattern_center_col=pattern_col,
            note=f"Labeled via mobile septum review tool (frame={frame})",
            annotator=user,
        )

        res = self.save_septum_label(req, user=user)

        state = self._load_mobile_septum_state(exp, film)
        key = f"{cell_id}_{frame}"
        reviewed = state.get("reviewed_frames", {})
        if key not in reviewed:
            state["mobile_labels_saved_count"] = int(state.get("mobile_labels_saved_count", 0)) + 1
        reviewed[key] = {
            "has_septum": has_septum,
            "pattern_center": [pattern_row, pattern_col] if (pattern_row and pattern_col) else None,
            "timestamp": datetime.now().isoformat(),
            "label_source": "mobile_septum_review",
        }
        state["reviewed_frames"] = reviewed
        self._save_mobile_septum_state(exp, film, state)

        al_info = {}
        if self.al_service:
            try:
                al_info = self.al_service.record_label_saved()
            except Exception as exc:
                print(f"Active learning record label error: {exc}")

        res["mobile_labels_saved_count"] = state["mobile_labels_saved_count"]
        res["al_info"] = al_info
        return res


