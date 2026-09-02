import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread
from skimage.measure import label
from flask import Blueprint, jsonify, request, current_app

from typing import Optional, Dict, Any, List
from ..schemas import SaveMasksRequest, validate_and_decode_rle, encode_mask_to_rle
from ..security import resolve_under_root
from .frames_bp import resolve_gt_keyframe

masks_bp = Blueprint("masks", __name__)

@masks_bp.route("/api/cell_masks", methods=["GET"])
def get_cell_masks():
    exp = request.args.get("experiment")
    cell_id = request.args.get("cell_id")
    film = request.args.get("film")
    sequence = request.args.get("sequence")

    if not exp or cell_id is None:
        return jsonify({"status": "error", "message": "experiment and cell_id required"}), 400

    frames_svc = current_app.extensions["gt_frames_service"]
    mask_repo = current_app.extensions["mask_repository"]
    linkage_repo = current_app.extensions["linkage_repository"]
    linkage_svc = current_app.extensions["linkage_service"]

    # 1. Build Keyframe Map
    if sequence:
        keyframe_map = frames_svc.get_sequence_keyframe_map(exp, sequence)
        seq_res = linkage_svc.get_sequences(exp)
        seq_info = seq_res.get("sequences", {}).get(sequence, {})
        films = seq_info.get("films", [sequence])
        global_cells = seq_info.get("global_cells", {})
        track = global_cells.get(str(cell_id)) or global_cells.get(f"{sequence}_cell_{cell_id}") or [-1] * len(films)
    else:
        target_film = film or exp
        k_times = frames_svc.get_film_keyframes(exp, target_film)
        keyframe_map = [
            {
                "global_t": idx,
                "film": target_film,
                "film_idx": 0,
                "local_t": t_val,
                "keyframe_pos": "First" if idx == 0 else ("Last" if idx == len(k_times) - 1 else "Middle"),
                "keyframe_idx": idx
            }
            for idx, t_val in enumerate(k_times)
        ]
        films = [target_film]
        try:
            cid_int = int(cell_id)
        except ValueError:
            cid_int = 0
        track = [cid_int]

    # 2. Extract masks for each keyframe
    masks_result = []
    w, h = 2000, 2000
    track_channel = "bf"

    # Preload dataframes for all involved films
    film_dfs = {}
    for f_idx, f_name in enumerate(films):
        local_cid = track[f_idx] if f_idx < len(track) else -1
        if local_cid != -1:
            try:
                df, _ = mask_repo.load_cell_masks(exp, f_name, local_cid)
                film_dfs[f_name] = df
                if not df.empty and "width" in df.columns:
                    w = int(df.iloc[0]["width"])
                    h = int(df.iloc[0]["height"])
                if not df.empty and "rle_gfp" in df.columns and df["rle_gfp"].dropna().any():
                    track_channel = "gfp"
            except Exception:
                film_dfs[f_name] = None
        else:
            film_dfs[f_name] = None

    for item in keyframe_map:
        f_name = item["film"]
        local_t = item["local_t"]
        df = film_dfs.get(f_name)
        
        rle_str = ""
        if df is not None and not df.empty:
            rows = df[df["time_point"] == local_t]
            if not rows.empty:
                if "FL" in f_name:
                    rle_col = "rle_gfp" if "rle_gfp" in rows.columns and str(rows.iloc[0].get("rle_gfp", "")).strip() not in ("", "nan") else "rle_bf"
                else:
                    rle_col = "rle_bf" if "rle_bf" in rows.columns and str(rows.iloc[0].get("rle_bf", "")).strip() not in ("", "nan") else "rle_gfp"
                rle_str = str(rows.iloc[0].get(rle_col, ""))
                if rle_str == "nan":
                    rle_str = ""
        masks_result.append(rle_str)

    film_boundaries = []
    # Compute film transition boundaries for UI
    cur_film = None
    for idx, item in enumerate(keyframe_map):
        if item["film"] != cur_film:
            cur_film = item["film"]
            film_boundaries.append({
                "frame": idx,
                "film": cur_film,
                "local_start": item["local_t"]
            })

    return jsonify({
        "status": "success",
        "cell_id": cell_id,
        "num_frames": len(keyframe_map),
        "masks": masks_result,
        "width": w,
        "height": h,
        "track_channel": track_channel,
        "keyframes": keyframe_map,
        "film_boundaries": film_boundaries,
        "linkage_details": {
            "films": films,
            "local_ids": [int(x) for x in track]
        }
    })


@masks_bp.route("/api/save_mask", methods=["POST"])
def save_mask():
    data = request.get_json() or {}
    try:
        req = SaveMasksRequest(**data)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Validation error: {e}"}), 400

    frames_svc = current_app.extensions["gt_frames_service"]
    mask_repo = current_app.extensions["mask_repository"]
    export_svc = current_app.extensions["gt_export_service"]
    linkage_svc = current_app.extensions["linkage_service"]

    # 1. Resolve target film and local cell ID
    target_film = req.film
    local_cid = None
    
    if req.sequence and not target_film:
        seq_res = linkage_svc.get_sequences(req.experiment)
        seq_info = seq_res.get("sequences", {}).get(req.sequence, {})
        films = seq_info.get("films", [req.sequence])
        global_cells = seq_info.get("global_cells", {})
        track = global_cells.get(str(req.cell_id)) or global_cells.get(f"{req.sequence}_cell_{req.cell_id}") or [-1] * len(films)
        
        # If changes specified, find the film for the timepoint
        if req.changes and len(req.changes) > 0:
            gt_t = req.changes[0].time_point
            k_res = resolve_gt_keyframe(req.experiment, req.sequence, None, gt_t, frames_svc)
            target_film = k_res["film"]
            f_idx = films.index(target_film) if target_film in films else k_res.get("film_idx", 0)
            local_cid = track[f_idx] if f_idx < len(track) else -1
        else:
            target_film = films[0] if films else req.sequence
            local_cid = track[0] if track else -1
            
    if local_cid is None or local_cid == -1:
        try:
            local_cid = int(str(req.cell_id).split("_cell_")[-1])
        except ValueError:
            local_cid = 1

    # Auto-sync track linkage if slot was unlinked or changed
    if req.sequence and 'f_idx' in locals() and f_idx < len(track) and track[f_idx] != local_cid:
        track[f_idx] = local_cid
        linkage_repo = current_app.extensions.get("linkage_repository")
        if linkage_repo:
            try:
                raw_data, rev = linkage_repo.load_linkage(req.experiment)
                if req.sequence in raw_data and "global_cells" in raw_data[req.sequence]:
                    cell_key = str(req.cell_id) if str(req.cell_id) in raw_data[req.sequence]["global_cells"] else f"{req.sequence}_cell_{req.cell_id}"
                    raw_data[req.sequence]["global_cells"][cell_key] = track
                    linkage_repo.save_linkage(req.experiment, raw_data)
            except Exception as e:
                current_app.logger.warning(f"Could not auto-sync linkage: {e}")

    if not target_film:
        target_film = req.sequence or req.experiment

    # 2. Update TrackedCells CSV
    try:
        df, rev = mask_repo.load_cell_masks(req.experiment, target_film, local_cid)
    except Exception:
        # Create empty template if cell csv doesn't exist yet
        k_times = frames_svc.get_film_keyframes(req.experiment, target_film)
        all_paths = frames_svc.get_film_frame_paths(req.experiment, target_film)
        max_t = max(all_paths.keys()) if all_paths else 100
        rows = []
        for t in range(max_t + 1):
            rows.append({
                "time_point": t,
                "width": 2000,
                "height": 2000,
                "rle_bf": "",
                "rle_gfp": ""
            })
        df = pd.DataFrame(rows)
        rev = "init"

    rle_col = "rle_gfp" if "FL" in target_film else "rle_bf"
    if rle_col not in df.columns:
        df[rle_col] = ""
    df[rle_col] = df[rle_col].astype(object)

    local_t_synced = None
    if req.changes:
        for chg in req.changes:
            gt_t = chg.time_point
            rle_val = getattr(chg, "new_rle", getattr(chg, "rle", ""))
            k_res = resolve_gt_keyframe(req.experiment, req.sequence, req.film, gt_t, frames_svc)
            local_t = k_res["local_t"]
            local_t_synced = local_t
            
            mask_rows = df[df["time_point"] == local_t]
            if not mask_rows.empty:
                df.at[mask_rows.index[0], rle_col] = str(rle_val)
            else:
                # Append row if missing
                alt_col = "rle_bf" if rle_col == "rle_gfp" else "rle_gfp"
                new_row = {
                    "time_point": local_t,
                    "width": 2000,
                    "height": 2000,
                    rle_col: str(rle_val),
                    alt_col: ""
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    elif req.masks:
        # Array of masks aligned to keyframes
        k_map = frames_svc.get_sequence_keyframe_map(req.experiment, req.sequence) if req.sequence else None
        if k_map:
            for idx, rle_val in enumerate(req.masks):
                if idx < len(k_map) and k_map[idx]["film"] == target_film:
                    loc_t = k_map[idx]["local_t"]
                    local_t_synced = loc_t
                    mask_rows = df[df["time_point"] == loc_t]
                    if not mask_rows.empty:
                        df.at[mask_rows.index[0], rle_col] = rle_val

    new_rev = mask_repo.save_cell_masks(req.experiment, target_film, local_cid, df)
    frames_svc.clear_population_cache(req.experiment, target_film, local_t_synced)

    # 3. Live-sync instance mask into Cellpose training dataset folder
    training_sync_res = {}
    if local_t_synced is not None:
        try:
            training_sync_res = export_svc.sync_keyframe_to_training(req.experiment, target_film, local_t_synced)
        except Exception as e:
            training_sync_res = {"status": "warning", "message": f"Training sync failed: {e}"}

    return jsonify({
        "status": "success",
        "revision": new_rev,
        "film": target_film,
        "cell_id": local_cid,
        "training_sync": training_sync_res
    })


@masks_bp.route("/api/segment_at_coords", methods=["GET"])
def segment_at_coords():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    sequence = request.args.get("sequence")
    t_str = request.args.get("t", "0")
    x_str = request.args.get("x", "0")
    y_str = request.args.get("y", "0")

    if not exp:
        return jsonify({"error": "experiment required"}), 400

    try:
        t_val = int(t_str)
        x_val = int(round(float(x_str)))
        y_val = int(round(float(y_str)))
    except ValueError:
        return jsonify({"error": "Invalid coordinates"}), 400

    frames_svc = current_app.extensions["gt_frames_service"]
    res = resolve_gt_keyframe(exp, sequence, film, t_val, frames_svc)
    target_film = res["film"]
    local_t = res["local_t"]

    masks_dir = frames_svc.config.local_movie_root / exp / target_film / f"Masks_{target_film}"
    files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files:
        files = sorted([f for f in masks_dir.glob(f"*_t{local_t:03d}_*_seg.tif") if not f.name.startswith(".")])

    if not files:
        return jsonify({"rle": "", "label": 0})

    try:
        seg = imread(str(files[0]))
        seg_lbl = label(seg) if seg.dtype == bool else seg
        H, W = seg_lbl.shape[:2]
        
        if 0 <= y_val < H and 0 <= x_val < W:
            hit_label = seg_lbl[y_val, x_val]
            if hit_label > 0:
                mask = (seg_lbl == hit_label).astype(np.uint8)
                rle = encode_mask_to_rle(mask)
                return jsonify({"rle": rle, "label": int(hit_label)})
    except Exception:
        pass

    return jsonify({"rle": "", "label": 0})


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
        t_val = int(t_str)
    except ValueError:
        t_val = 0

    try:
        x_val = int(round(float(data.get("x", 0))))
        y_val = int(round(float(data.get("y", 0))))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid coordinates"}), 400

    if not exp:
        return jsonify({"status": "error", "message": "experiment required"}), 400

    frames_svc = current_app.extensions["gt_frames_service"]
    linkage_svc = current_app.extensions["linkage_service"]
    res = resolve_gt_keyframe(exp, sequence, film, t_val, frames_svc)
    target_film = res["film"]
    local_t = res["local_t"]

    base_root = frames_svc.config.local_movie_root
    local_cid = None

    # 1. Check _seg.tif directly (fastest path)
    masks_dir = base_root / exp / target_film / f"Masks_{target_film}"
    if masks_dir.exists():
        files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in masks_dir.glob(f"*_t{local_t:03d}_*_seg.tif") if not f.name.startswith(".")])
        if files:
            try:
                seg = imread(str(files[0]))
                if seg.ndim > 2: seg = seg[..., 0]
                H, W = seg.shape[:2]
                if 0 <= y_val < H and 0 <= x_val < W:
                    hit_lbl = int(seg[y_val, x_val])
                    if hit_lbl > 0:
                        local_cid = hit_lbl
            except Exception:
                pass

    # 2. Check TrackedCells CSVs fallback
    if local_cid is None:
        tracked_dir = base_root / exp / target_film / f"TrackedCells_{target_film}"
        if tracked_dir.exists():
            for csv_file in tracked_dir.glob("cell_*_masks.csv"):
                if csv_file.name.startswith("."): continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", csv_file.name)
                if not m: continue
                cid = int(m.group(1))

                try:
                    df = pd.read_csv(csv_file)
                    rows = df[df["time_point"] == local_t]
                    if rows.empty: continue
                    rle = str(rows.iloc[0].get("rle_bf", ""))
                    if not rle or rle.strip() == "" or rle.lower() == "nan":
                        rle = str(rows.iloc[0].get("rle_gfp", ""))
                    if not rle or rle.strip() == "" or rle.lower() == "nan": continue

                    H, W = int(rows.iloc[0].get("height", 2000)), int(rows.iloc[0].get("width", 2000))
                    if 0 <= y_val < H and 0 <= x_val < W:
                        mask = validate_and_decode_rle(rle, H, W)
                        if mask[y_val, x_val] > 0:
                            local_cid = cid
                            break
                except Exception:
                    continue

    if local_cid is None:
        return jsonify({"status": "not_found", "message": "No cell found at coordinates"}), 404

    # 3. Map to global cell if in a sequence
    global_cid = str(local_cid)
    if sequence:
        seq_res = linkage_svc.get_sequences(exp)
        seq_info = seq_res.get("sequences", {}).get(sequence, {})
        films_list = seq_info.get("films", [])
        global_cells = seq_info.get("global_cells", {})
        if target_film in films_list:
            f_idx = films_list.index(target_film)
            for gid, track in global_cells.items():
                if f_idx < len(track) and track[f_idx] == local_cid:
                    global_cid = gid
                    break

    return jsonify({
        "status": "success",
        "cell_id": global_cid,
        "local_cell_id": local_cid,
        "film": target_film
    })

