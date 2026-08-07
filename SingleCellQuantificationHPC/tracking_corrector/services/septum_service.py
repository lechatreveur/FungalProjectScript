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


class SeptumService:
    def __init__(self, base_movie_root: Path, qc_repo: QCRepository, audit_service: AuditService):
        self.base_root = base_movie_root
        self.qc_repo = qc_repo
        self.audit_service = audit_service

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

        cell_intervals[cid_str] = {
            "has_septum": bool(req.has_septum),
            "start_aligned": start_aligned,
            "end_aligned": end_aligned,
            "white_septum": bool(req.is_white_septum),
            "has_septum_2": has_second,
            "start_aligned_2": start_aligned_2,
            "end_aligned_2": end_aligned_2,
            "white_septum_2": bool(req.is_white_septum_2),
            "label_source": "cell",
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
                label_source="cell",
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

    def get_cached_ai_suggestion(
        self,
        exp: str,
        sequence: str,
        global_cell_id: str,
    ) -> Dict[str, Any]:
        """Look up a pre-computed AI suggestion from an offline batch run,
        if one exists. This is intentionally separate from predict_septum():
        that endpoint always runs the model live and never persists its
        result (review_only), so a saved cell can never be silently treated
        as AI-confirmed. This method only ever READS a file that some
        external batch process wrote - the app itself never writes to it -
        so there's no risk of the live "Run AI" flow and this cached-lookup
        flow stepping on each other.
        """
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
