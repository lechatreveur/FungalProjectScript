import os
import re
import time
import math
import io
import json
import random
import base64
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from flask import Blueprint, jsonify, request, current_app, render_template
from PIL import Image

from ..repositories.mask_repository import atomic_write_text
from ..qc_schema import (
    LocalCellQC,
    GlobalCellQC,
    validate_local_qc_status,
    validate_global_qc_status,
    InvalidQCStatusError,
)

mistrack_review_bp = Blueprint("mistrack_review", __name__)

STRIPS_DIR = Path("/Volumes/X10 Pro/FungalProject_Outputs/video_ae/vertical_strips/")
FRAME_HEIGHT = 32  # height of each frame in pixels
DEFAULT_WINDOW_FRAMES = 15  # default frames per window

_state_lock = threading.Lock()


@mistrack_review_bp.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@mistrack_review_bp.route("/mobile", methods=["GET"])
@mistrack_review_bp.route("/mobile_review", methods=["GET"])
def mobile_review():
    return render_template("mobile_review.html")


def _get_qc_repo():
    qc_service = current_app.config.get("QC_SERVICE")
    if qc_service and hasattr(qc_service, "qc_repo"):
        return qc_service.qc_repo
    app_cfg = current_app.config.get("APP_CONFIG")
    if app_cfg:
        from ..repositories.qc_repository import QCRepository
        return QCRepository(app_cfg.local_movie_root)
    raise RuntimeError("QCRepository not available")


def _get_review_state_path(qc_repo, exp: str, target: str) -> Path:
    qc_path = qc_repo.get_qc_json_path(exp, target)
    return qc_path.parent / f"mistrack_review_state_{target}.json"


def _load_review_state(qc_repo, exp: str, target: str) -> Dict[str, Any]:
    path = _get_review_state_path(qc_repo, exp, target)
    if not path.exists():
        return {}
    with _state_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            current_app.logger.warning(f"Failed to load review state file at {path}: {e}")
            return {}


def _save_review_state(qc_repo, exp: str, target: str, state_data: Dict[str, Any]) -> None:
    path = _get_review_state_path(qc_repo, exp, target)
    content = json.dumps(state_data, indent=2)
    with _state_lock:
        atomic_write_text(path, content)


def _find_strip_path(global_cell_id: str, exp: str, target: str) -> Optional[Path]:
    if not STRIPS_DIR.exists():
        return None

    canonical_path = STRIPS_DIR / f"{global_cell_id}.png"
    if canonical_path.exists():
        return canonical_path

    candidates = [
        STRIPS_DIR / f"{target}_{global_cell_id}.png",
        STRIPS_DIR / f"{exp}_{global_cell_id}.png",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


_strip_glob_cache: Dict[str, Tuple[float, List[str]]] = {}

def _get_candidate_cell_ids(qc_repo, exp: str, target: str, review_state: Dict[str, Any]) -> List[str]:
    cell_set = set(review_state.keys())

    # Add cells from QC file
    data, _ = qc_repo.load_qc(exp, target)
    if isinstance(data, dict):
        records = data.get("records", data) if "records" in data else data
        for k in records.keys():
            cell_set.add(k)

    # Add cells from global septum alignment if present
    try:
        sep_data, _ = qc_repo.load_septum(exp, target)
        if isinstance(sep_data, dict):
            for c in sep_data.get("cell_order", []):
                cell_set.add(str(c))
            for k in sep_data.get("offsets", {}).keys():
                cell_set.add(str(k))
    except Exception:
        pass

    # Add strip files matching target prefix
    cache_key = f"{exp}_{target}"
    now = time.time()
    if cache_key in _strip_glob_cache and (now - _strip_glob_cache[cache_key][0]) < 30.0:
        cached_stems = _strip_glob_cache[cache_key][1]
    else:
        cached_stems = []
        if STRIPS_DIR.exists():
            target_prefix = f"{target}_"
            for p in STRIPS_DIR.glob(f"{target_prefix}*.png"):
                cached_stems.append(p.stem)
        _strip_glob_cache[cache_key] = (now, cached_stems)

    for stem in cached_stems:
        cell_set.add(stem)

    filtered = []
    for cid in cell_set:
        s_cid = str(cid)
        if s_cid.startswith(target) or s_cid.startswith(f"{target}_"):
            filtered.append(s_cid)

    return filtered


def _crop_window_base64(strip_path: Path, window_index: int, window_frames: int = DEFAULT_WINDOW_FRAMES) -> Tuple[str, int]:
    with Image.open(strip_path) as im:
        img_w, img_h = im.size
        win_h = window_frames * FRAME_HEIGHT
        total_windows = max(1, math.ceil(img_h / win_h))

        y0 = window_index * win_h
        y1 = min(img_h, (window_index + 1) * win_h)

        crop = im.crop((0, y0, img_w, y1))

        if crop.mode in ("RGBA", "LA", "P"):
            crop = crop.convert("RGB")

        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64_str, total_windows


@mistrack_review_bp.route("/api/mistrack_review/batch", methods=["GET", "OPTIONS"])
def get_batch():
    if request.method == "OPTIONS":
        return "", 200

    exp = request.args.get("experiment") or request.args.get("exp")
    target = request.args.get("target") or request.args.get("sequence") or request.args.get("film")
    if not exp or not target:
        return jsonify({"status": "error", "message": "experiment and target required"}), 400

    try:
        count = int(request.args.get("count", 10))
    except ValueError:
        count = 10

    try:
        window_frames = int(request.args.get("window_frames", DEFAULT_WINDOW_FRAMES))
    except ValueError:
        window_frames = DEFAULT_WINDOW_FRAMES

    exclude_qc_arg = request.args.get("exclude_qc")
    include_qc_arg = request.args.get("include_qc")

    exclude_qc_set = set(x.strip().lower() for x in exclude_qc_arg.split(",") if x.strip()) if exclude_qc_arg else set()
    include_qc_set = set(x.strip().lower() for x in include_qc_arg.split(",") if x.strip()) if include_qc_arg else set()

    qc_repo = _get_qc_repo()
    qc_data, _ = qc_repo.load_qc(exp, target)
    qc_records = qc_data.get("records", qc_data) if isinstance(qc_data, dict) else {}

    review_state = _load_review_state(qc_repo, exp, target)
    candidate_ids = _get_candidate_cell_ids(qc_repo, exp, target, review_state)

    state_changed = False
    eligible_cells = []

    for cid in candidate_ids:
        raw_qc_status = qc_records.get(cid)
        qc_status = str(raw_qc_status or "unlabeled").lower()

        if exclude_qc_set and qc_status in exclude_qc_set:
            continue

        if include_qc_set and qc_status not in include_qc_set:
            continue

        cell_info = review_state.get(cid)
        if cell_info is None:
            strip_path = _find_strip_path(cid, exp, target)
            if not strip_path:
                continue
            try:
                with Image.open(strip_path) as im:
                    img_h = im.size[1]
                win_h = window_frames * FRAME_HEIGHT
                tot_win = max(1, math.ceil(img_h / win_h))
            except Exception:
                continue

            cell_info = {
                "total_windows": tot_win,
                "shown_windows": [],
                "status": LocalCellQC.PENDING.value,
                "reviewed": False,
            }
            review_state[cid] = cell_info
            state_changed = True

        status = cell_info.get("status", LocalCellQC.PENDING.value)
        is_reviewed = cell_info.get("reviewed", False) or status == "exhausted"

        if status == LocalCellQC.MISTRACKED.value or is_reviewed:
            continue

        tot_win = cell_info.get("total_windows", 1)
        shown = set(cell_info.get("shown_windows", []))
        unshown = [w for w in range(tot_win) if w not in shown]

        if not unshown:
            cell_info["status"] = LocalCellQC.PENDING.value
            cell_info["reviewed"] = True
            state_changed = True
            continue

        eligible_cells.append((cid, unshown, tot_win))

    if state_changed:
        _save_review_state(qc_repo, exp, target, review_state)

    selected_cells = random.sample(eligible_cells, min(count, len(eligible_cells))) if eligible_cells else []

    items = []
    for cid, unshown, tot_win in selected_cells:
        w_idx = random.choice(unshown)
        strip_path = _find_strip_path(cid, exp, target)
        if not strip_path:
            continue
        try:
            b64_img, confirmed_tot = _crop_window_base64(strip_path, w_idx, window_frames)
            items.append({
                "global_cell_id": cid,
                "window_index": w_idx,
                "total_windows": confirmed_tot,
                "image_base64": b64_img,
            })
        except Exception as e:
            current_app.logger.error(f"Failed to crop window {w_idx} for cell {cid}: {e}")

    remaining_pending = len(eligible_cells) - len(items)

    return jsonify({
        "status": "success",
        "items": items,
        "remaining_pending": max(0, remaining_pending),
        "total_eligible_cells": len(eligible_cells),
    })


@mistrack_review_bp.route("/api/mistrack_review/result", methods=["POST", "OPTIONS"])
def post_result():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json() or {}
    exp = data.get("experiment") or data.get("exp")
    target = data.get("target") or data.get("sequence") or data.get("film")
    cell_id = data.get("global_cell_id") or data.get("cell_id")
    if cell_id is not None:
        cell_id = str(cell_id)

    w_idx_val = data.get("window_index")
    swipe = str(data.get("swipe") or "").lower().strip()

    if not exp or not target or not cell_id or w_idx_val is None or swipe not in ("left", "right"):
        return jsonify({
            "status": "error",
            "message": "experiment, target, global_cell_id, window_index, and swipe ('left'|'right') required",
        }), 400

    try:
        w_idx = int(w_idx_val)
    except ValueError:
        return jsonify({"status": "error", "message": "window_index must be an integer"}), 400

    qc_repo = _get_qc_repo()
    review_state = _load_review_state(qc_repo, exp, target)
    cell_info = review_state.get(cell_id)

    if cell_info is None:
        strip_path = _find_strip_path(cell_id, exp, target)
        tot_win = 1
        if strip_path:
            try:
                with Image.open(strip_path) as im:
                    tot_win = max(1, math.ceil(im.size[1] / (DEFAULT_WINDOW_FRAMES * FRAME_HEIGHT)))
            except Exception:
                pass
        cell_info = {
            "total_windows": tot_win,
            "shown_windows": [],
            "status": LocalCellQC.PENDING.value,
            "reviewed": False,
        }
        review_state[cell_id] = cell_info

    if swipe == "left":
        cell_info["status"] = LocalCellQC.MISTRACKED.value
        cell_info["reviewed"] = True
        new_cell_status = LocalCellQC.MISTRACKED.value

    elif swipe == "right":
        shown = set(cell_info.get("shown_windows", []))
        shown.add(w_idx)
        cell_info["shown_windows"] = sorted(list(shown))

        tot_win = cell_info.get("total_windows", 1)
        if len(shown) >= tot_win:
            cell_info["status"] = LocalCellQC.PENDING.value
            cell_info["reviewed"] = True
            new_cell_status = LocalCellQC.PENDING.value
        else:
            if cell_info.get("status") != LocalCellQC.MISTRACKED.value:
                cell_info["status"] = LocalCellQC.PENDING.value
                cell_info["reviewed"] = False
                new_cell_status = LocalCellQC.PENDING.value

    _save_review_state(qc_repo, exp, target, review_state)

    return jsonify({
        "status": "success",
        "global_cell_id": cell_id,
        "window_index": w_idx,
        "cell_status": new_cell_status,
        "reviewed": cell_info.get("reviewed", False),
        "shown_windows": cell_info.get("shown_windows", []),
    })


@mistrack_review_bp.route("/api/mistrack_review/undo", methods=["POST", "OPTIONS"])
def post_undo():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json() or {}
    exp = data.get("experiment") or data.get("exp")
    target = data.get("target") or data.get("sequence") or data.get("film")
    cell_id = data.get("global_cell_id") or data.get("cell_id")
    if cell_id is not None:
        cell_id = str(cell_id)

    w_idx_val = data.get("window_index")
    if not exp or not target or not cell_id or w_idx_val is None:
        return jsonify({"status": "error", "message": "experiment, target, global_cell_id, and window_index required"}), 400

    try:
        w_idx = int(w_idx_val)
    except ValueError:
        return jsonify({"status": "error", "message": "window_index must be an integer"}), 400

    qc_repo = _get_qc_repo()
    review_state = _load_review_state(qc_repo, exp, target)
    cell_info = review_state.get(cell_id)
    if cell_info is not None:
        shown = set(cell_info.get("shown_windows", []))
        if w_idx in shown:
            shown.remove(w_idx)
        cell_info["shown_windows"] = sorted(list(shown))
        cell_info["status"] = LocalCellQC.PENDING.value
        cell_info["reviewed"] = False
        _save_review_state(qc_repo, exp, target, review_state)

    return jsonify({
        "status": "success",
        "global_cell_id": cell_id,
        "window_index": w_idx,
        "cell_status": LocalCellQC.PENDING.value,
        "reviewed": False,
    })



@mistrack_review_bp.route("/api/mistrack_review/coverage", methods=["GET", "OPTIONS"])
def get_coverage():
    if request.method == "OPTIONS":
        return "", 200

    exp = request.args.get("experiment") or request.args.get("exp") or "M156"
    target_req = request.args.get("target") or request.args.get("sequence") or request.args.get("film")

    qc_repo = _get_qc_repo()
    all_targets = ["3_F0", "3_F1", "3_F2"]

    field_coverage = {}
    total_all = 0
    pending_all = 0
    exhausted_all = 0
    mistracked_all = 0

    for t in all_targets:
        review_state = _load_review_state(qc_repo, exp, t)
        qc_data, _ = qc_repo.load_qc(exp, t)
        records = qc_data.get("records", qc_data) if isinstance(qc_data, dict) else {}

        all_cells = set(review_state.keys())
        for k in records.keys():
            if k.startswith(t):
                all_cells.add(k)

        t_total = len(all_cells)
        t_pending = 0
        t_exhausted = 0
        t_mistracked = 0

        for cid in all_cells:
            st_info = review_state.get(cid, {})
            st = st_info.get("status") if isinstance(st_info, dict) else LocalCellQC.PENDING.value
            is_reviewed = st_info.get("reviewed", False) if isinstance(st_info, dict) else False

            if st == LocalCellQC.MISTRACKED.value:
                t_mistracked += 1
            elif is_reviewed or st == "exhausted":
                t_exhausted += 1
            else:
                t_pending += 1


        pct = round(((t_exhausted + t_mistracked) / t_total * 100), 1) if t_total > 0 else 0.0

        field_coverage[t] = {
            "total_cells": t_total,
            "pending": t_pending,
            "exhausted": t_exhausted,
            "mistracked": t_mistracked,
            "percent_reviewed": pct,
        }

        total_all += t_total
        pending_all += t_pending
        exhausted_all += t_exhausted
        mistracked_all += t_mistracked

    overall_pct = round(((exhausted_all + mistracked_all) / total_all * 100), 1) if total_all > 0 else 0.0

    selected_target = target_req if target_req and target_req in field_coverage else "3_F0"

    return jsonify({
        "status": "success",
        "experiment": exp,
        "target": target_req,
        "coverage": field_coverage.get(selected_target, {}),
        "overall": {
            "total_cells": total_all,
            "pending": pending_all,
            "exhausted": exhausted_all,
            "mistracked": mistracked_all,
            "percent_reviewed": overall_pct,
        },
        "fields": field_coverage,
    })
