import re
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, current_app
from pydantic import ValidationError as PydanticValidationError
from ..schemas import SaveMasksRequest, validate_and_decode_rle

masks_bp = Blueprint("masks", __name__)

def _encode_rle(mask):
    """Encode a binary mask using the column-major RLE format used by cell CSVs."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    padded = np.concatenate(([0], pixels, [0]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    transitions[1::2] -= transitions[::2]
    return " ".join(str(value) for value in transitions)


@masks_bp.route("/api/list_cells", methods=["GET"])
def list_cells():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment parameter required"}), 400
        
    tracking_service = current_app.config["TRACKING_SERVICE"]
    
    if "sequence" in request.args:
        seq = request.args.get("sequence")
        res = tracking_service.list_sequence_cells(exp, seq)
        return jsonify(res)
    else:
        film = request.args.get("film")
        if not film:
            return jsonify({"status": "error", "message": "sequence or film parameter required"}), 400
        cells = tracking_service.list_film_cells(exp, film)
        return jsonify({"cells": cells})

@masks_bp.route("/api/cell_masks", methods=["GET"])
def get_cell_masks():
    exp = request.args.get("experiment")
    cell_id_str = request.args.get("cell_id")
    
    if not exp or not cell_id_str:
        return jsonify({"status": "error", "message": "experiment and cell_id required"}), 400
        
    masks_service = current_app.config["MASKS_SERVICE"]
    
    if "sequence" in request.args and request.args.get("sequence"):
        seq = request.args.get("sequence")
        res = masks_service.get_sequence_masks(exp, seq, cell_id_str)
        return jsonify(res)
    else:
        film = request.args.get("film")
        if not film:
            return jsonify({"status": "error", "message": "film or sequence required"}), 400
        try:
            cid = int(cell_id_str)
        except ValueError:
            cid = 1
        res = masks_service.get_cell_masks(exp, film, cid)
        return jsonify(res)

@masks_bp.route("/api/save_masks", methods=["POST"])
def save_masks():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "JSON body required"}), 400
        
    try:
        req = SaveMasksRequest(**data)
    except PydanticValidationError as e:
        return jsonify({"status": "error", "code": "VALIDATION_ERROR", "message": str(e)}), 422
        
    film = req.film or req.sequence or "film1"
    try:
        cell_id = int(req.cell_id)
    except ValueError:
        cell_id = 1
        
    masks_service = current_app.config["MASKS_SERVICE"]
    res = masks_service.save_cell_masks(req, film, cell_id)
    
    cache_service = current_app.config["CACHE_SERVICE"]
    cache_service.clear_all_caches_for_film(req.experiment, film)

    return jsonify(res)

@masks_bp.route("/api/create_new_cell", methods=["POST"])
def create_new_cell():
    data = request.get_json() or {}
    exp = data.get("experiment")
    film = data.get("film") or data.get("sequence")
    num_frames = data.get("num_frames", 100)
    
    if not exp or not film:
        return jsonify({"status": "error", "message": "experiment and film required"}), 400
        
    tracking_service = current_app.config["TRACKING_SERVICE"]
    new_id = tracking_service.create_new_cell(exp, film, num_frames)
    return jsonify({"status": "success", "new_cell_id": new_id})

@masks_bp.route("/api/identify_cell", methods=["POST", "GET"])
def identify_cell():
    if request.method == "POST":
        data = request.get_json() or {}
    else:
        data = request.args.to_dict()
        
    exp = data.get("experiment")
    sequence = data.get("sequence")
    film = data.get("film")
    t_str = data.get("t", "0")
    try:
        t = int(t_str)
    except ValueError:
        t = 0
        
    try:
        x, y = int(data.get("x")), int(data.get("y"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Coordinates x and y must be integers"}), 400

    base_root = current_app.config["APP_CONFIG"].local_movie_root
    
    local_t = t
    if sequence and not film:
        from .frames_bp import resolve_global_t
        film, _, local_t = resolve_global_t(exp, sequence, "", t, base_root)
    elif not film and sequence:
        film = sequence
        
    if not film:
        return jsonify({"status": "error", "message": "film or sequence parameter required"}), 400

    tracked_dir = base_root / exp / film / f"TrackedCells_{film}"
    if not tracked_dir.exists():
        return jsonify({"status": "error", "message": f"TrackedCells directory not found for {film}"}), 404

    t_candidates = [local_t]
    for delta in [1, -1, 2, -2, 3, -3]:
        t_cand = local_t + delta
        if t_cand >= 0:
            t_candidates.append(t_cand)

    for cf in tracked_dir.iterdir():
        if cf.name.startswith("."): continue
        m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
        if not m: continue
        cid = int(m.group(1))
        
        try:
            df = pd.read_csv(cf)
            if df.empty: continue
            H, W = int(df.iloc[0]["height"]), int(df.iloc[0]["width"])
            if x < 0 or y < 0 or x >= W or y >= H: continue
            
            found = False
            for t_check in t_candidates:
                rows = df[df["time_point"] == t_check]
                if rows.empty: continue
                for rle_col in ["rle_bf", "rle_gfp"]:
                    if rle_col in df.columns:
                        rle = str(rows.iloc[0].get(rle_col, ""))
                        if rle and rle.strip() and rle.lower() != "nan":
                            mask = validate_and_decode_rle(rle, H, W)
                            if mask[y, x] > 0:
                                found = True
                                global_cid = str(cid)
                                if sequence:
                                    linkage_svc = current_app.config["LINKAGE_SERVICE"]
                                    seq_res = linkage_svc.get_sequences(exp)
                                    seq_info = seq_res.get("sequences", {}).get(sequence, {})
                                    films_list = seq_info.get("films", [])
                                    global_cells = seq_info.get("global_cells", {})
                                    if film in films_list:
                                        film_idx = films_list.index(film)
                                        for g_id, loc_ids in global_cells.items():
                                            if film_idx < len(loc_ids) and loc_ids[film_idx] == cid:
                                                global_cid = g_id
                                                break
                                return jsonify({
                                    "status": "success",
                                    "cell_id": global_cid,
                                    "local_cell_id": cid,
                                    "film": film
                                })
                if found:
                    break
        except Exception:
            continue

    masks_dir = base_root / exp / film / f"Masks_{film}"
    if masks_dir.is_dir():
        files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if files:
            try:
                from skimage.io import imread
                seg = imread(str(files[0]))
                if 0 <= y < seg.shape[0] and 0 <= x < seg.shape[1]:
                    label_val = int(seg[y, x])
                    if label_val > 0:
                        return jsonify({
                            "status": "untracked",
                            "label_id": label_val,
                            "film": film
                        })
            except Exception:
                pass

    return jsonify({"status": "error", "message": f"No cell mask found at ({x}, {y}) on frame t={t}"}), 404

@masks_bp.route("/api/click_segment", methods=["POST"])
def click_segment():
    data = request.get_json() or {}
    exp = data.get("experiment")
    t = int(data.get("t", 0))
    x, y = int(data.get("x", 0)), int(data.get("y", 0))
    
    base_root = current_app.config["APP_CONFIG"].local_movie_root

    if "sequence" in data and data.get("sequence"):
        seq = data.get("sequence")
        gid = data.get("cell_id")
        from .frames_bp import resolve_global_t
        film, local_cid, local_t = resolve_global_t(exp, seq, gid, t, base_root)
    else:
        film = data.get("film")
        try:
            local_cid = int(data.get("cell_id", -1))
        except (ValueError, TypeError):
            local_cid = -1
        local_t = t

    if not film or local_cid == -1:
        return jsonify({"status": "error", "message": "Cannot select segment for an unassigned cell mapping."})

    masks_dir = base_root / exp / film / f"Masks_{film}"
    files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files:
        files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files:
        return jsonify({"status": "error", "message": "Segmentation file not found"}), 404

    try:
        import sys
        from pathlib import Path
        routes_dir = Path(__file__).resolve().parent
        workspace_root = routes_dir.parent.parent.parent
        if str(workspace_root) not in sys.path:
            sys.path.append(str(workspace_root))
        from Cell_tracking_functions import load_segmentation, rle_encode
        from skimage.measure import label

        seg = load_segmentation(str(files[0]))
        seg_lbl = label(seg) if seg.dtype == bool else seg
        H, W = seg_lbl.shape
        if y >= H or x >= W or y < 0 or x < 0:
            return jsonify({"status": "error", "message": "Click coordinates out of range"}), 400

        lbl = seg_lbl[y, x]
        if lbl == 0:
            return jsonify({"status": "success", "rle": ""})

        segment_mask = (seg_lbl == lbl)
        rle = rle_encode(segment_mask)
        return jsonify({"status": "success", "rle": rle})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to compute click segment: {str(e)}"}), 500


@masks_bp.route("/api/autofix_masks", methods=["POST"])
def autofix_masks():
    """Snap existing masks to overlapping labels in the raw segmentation files."""
    data = request.get_json(silent=True) or {}
    exp = data.get("experiment")
    cell_id = data.get("cell_id")
    film = data.get("film") or data.get("local_film")
    sequence = data.get("sequence")
    channel = data.get("channel", "bf")

    if not exp or cell_id is None:
        return jsonify({"status": "error", "message": "experiment and cell_id are required"}), 400
    if not film and not sequence:
        return jsonify({"status": "error", "message": "film or sequence is required"}), 400
    if channel not in {"bf", "gfp"}:
        return jsonify({"status": "error", "message": "channel must be 'bf' or 'gfp'"}), 400

    try:
        start_t = int(data.get("start_t"))
        end_t = int(data.get("end_t"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "start_t and end_t must be integers"}), 400
    if start_t < 0 or end_t < start_t:
        return jsonify({"status": "error", "message": "Invalid frame range"}), 400

    base_root = current_app.config["APP_CONFIG"].local_movie_root
    mask_repo = current_app.config["MASKS_SERVICE"].mask_repo
    cache_service = current_app.config["CACHE_SERVICE"]
    loaded = {}
    result_masks = {}
    fixed_count = 0

    try:
        from skimage.io import imread
        from skimage.measure import label
        from .frames_bp import resolve_global_t

        for global_t in range(start_t, end_t + 1):
            if sequence:
                actual_film, local_cell_id, local_t = resolve_global_t(
                    exp, sequence, str(cell_id), global_t, base_root
                )
            else:
                actual_film, local_t = film, global_t
                try:
                    local_cell_id = int(cell_id)
                except (TypeError, ValueError):
                    return jsonify({
                        "status": "error",
                        "message": "Local cell_id must be an integer",
                    }), 400

            if not actual_film or local_cell_id in {-1, None}:
                continue

            key = (actual_film, int(local_cell_id))
            if key not in loaded:
                df, revision = mask_repo.load_cell_masks(exp, key[0], key[1])
                loaded[key] = [df, revision, False]
            df = loaded[key][0]

            matching_rows = df.index[df["time_point"] == local_t].tolist()
            if not matching_rows:
                continue
            row_idx = matching_rows[0]

            preferred_col = f"rle_{channel}"
            if preferred_col in df.columns:
                rle_col = preferred_col
            elif "rle_bf" in df.columns:
                rle_col = "rle_bf"
            elif "rle_gfp" in df.columns:
                rle_col = "rle_gfp"
            else:
                continue

            existing_rle = df.at[row_idx, rle_col]
            if not isinstance(existing_rle, str) or not existing_rle.strip():
                continue
            height = int(df.at[row_idx, "height"])
            width = int(df.at[row_idx, "width"])
            existing_mask = validate_and_decode_rle(existing_rle, height, width).astype(bool)
            if not existing_mask.any():
                continue

            masks_dir = base_root / exp / actual_film / f"Masks_{actual_film}"
            seg_files = sorted(
                path for path in masks_dir.glob(
                    f"{actual_film}_t_{local_t:03d}_c_*_seg.tif"
                ) if not path.name.startswith(".")
            )
            if not seg_files:
                seg_files = sorted(
                    path for path in masks_dir.glob(
                        f"*_t_{local_t:03d}_c_*_seg.tif"
                    ) if not path.name.startswith(".")
                )
            if not seg_files:
                continue

            segmentation = np.squeeze(imread(str(seg_files[0])))
            if segmentation.ndim != 2 or segmentation.shape != existing_mask.shape:
                continue
            segmentation = label(segmentation) if segmentation.dtype == bool else segmentation

            labels, overlap_counts = np.unique(
                segmentation[existing_mask], return_counts=True
            )
            selected_labels = []
            best_label = 0
            best_iou = 0.0
            existing_area = int(existing_mask.sum())
            for label_id, overlap in zip(labels, overlap_counts):
                if label_id == 0:
                    continue
                raw_area = int(np.count_nonzero(segmentation == label_id))
                if raw_area == 0:
                    continue
                if overlap / raw_area >= 0.4:
                    selected_labels.append(label_id)
                iou = overlap / (existing_area + raw_area - overlap)
                if iou > best_iou:
                    best_iou = iou
                    best_label = label_id

            if not selected_labels and best_label:
                selected_labels.append(best_label)
            if not selected_labels:
                continue

            fixed_mask = np.isin(segmentation, selected_labels)
            new_rle = _encode_rle(fixed_mask)
            df.at[row_idx, rle_col] = new_rle
            source_col = "source_bf" if rle_col == "rle_bf" else "source_gfp"
            area_col = "area_bf" if rle_col == "rle_bf" else "area_gfp"
            if source_col not in df.columns:
                df[source_col] = ""
            df.at[row_idx, source_col] = "manual"
            if area_col in df.columns:
                df.at[row_idx, area_col] = int(fixed_mask.sum())

            loaded[key][2] = True
            result_masks[global_t] = new_rle
            fixed_count += 1

        for (actual_film, local_cell_id), (df, revision, modified) in loaded.items():
            if modified:
                mask_repo.save_cell_masks(
                    exp, actual_film, local_cell_id, df, expected_revision=revision
                )
                cache_service.clear_all_caches_for_film(exp, actual_film)

        return jsonify({
            "status": "success",
            "fixed_count": fixed_count,
            "masks": result_masks,
        })
    except Exception as exc:
        current_app.logger.exception("Auto-fix failed")
        return jsonify({
            "status": "error",
            "message": f"Auto-fix failed: {exc}",
        }), 500
