import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread
from skimage.measure import label
from flask import Blueprint, jsonify, request, current_app


import shutil
from typing import Optional, Dict, Any, List, Tuple
from ..schemas import SaveMasksRequest, validate_and_decode_rle, encode_mask_to_rle
from ..security import resolve_under_root
from .frames_bp import resolve_gt_keyframe

masks_bp = Blueprint("masks", __name__)


def normalize_cell_key(global_cells: Dict[str, Any], cell_id: str, sequence: Optional[str] = None) -> str:
    s_id = str(cell_id).strip()
    if s_id in global_cells:
        return s_id
    if sequence and f"{sequence}_cell_{s_id}" in global_cells:
        return f"{sequence}_cell_{s_id}"
    for k in global_cells:
        if k == s_id or k.endswith(f"cell_{s_id}") or (s_id.isdigit() and k.endswith(f"_{s_id}")):
            return k
    return s_id


def check_and_duplicate_shared_local_cell(
    movie_root: Path,
    exp: str,
    sequence: str,
    target_film: str,
    f_idx: int,
    current_cell_id: str,
    local_cid: int,
    tool: str,
    films: List[str],
    global_cells: Dict[str, List[int]]
) -> Tuple[int, Optional[List[int]], bool]:
    """
    Checks if local_cid in target_film is shared by another global cell that is
    adjacent/sister to current_cell_id. If tool is 'brush' or 'eraser', duplicates
    the local cell file and updates sequence linkage for current_cell_id.
    
    Returns: (effective_local_cid, updated_track, was_duplicated)
    """
    cur_key = normalize_cell_key(global_cells, current_cell_id, sequence)
    cur_track = global_cells.get(cur_key, [])

    if tool not in ["brush", "eraser"]:
        return local_cid, cur_track, False

    if local_cid is None or local_cid <= 0:
        return local_cid, cur_track, False

    if not cur_track:
        return local_cid, cur_track, False

    # 1. Find all other global cells in this sequence sharing local_cid at f_idx
    sharing_cells = []
    for other_id, other_track in global_cells.items():
        if other_id != cur_key:
            if f_idx < len(other_track) and other_track[f_idx] == local_cid:
                sharing_cells.append((other_id, other_track))

    if not sharing_cells:
        return local_cid, cur_track, False

    # 2. Check if at least one sharing cell is adjacent/sister
    is_adjacent = False
    for other_id, other_track in sharing_cells:
        diverged = False
        check_indices = list(range(f_idx + 1, min(len(cur_track), len(other_track)))) + list(range(f_idx - 1, -1, -1))
        for test_idx in check_indices:
            t_cid1 = cur_track[test_idx] if test_idx < len(cur_track) else -1
            t_cid2 = other_track[test_idx] if test_idx < len(other_track) else -1
            if t_cid1 > 0 and t_cid2 > 0:
                if t_cid1 != t_cid2:
                    diverged = True
                    test_film = films[test_idx]
                    t_dir = resolve_under_root(movie_root, Path(exp) / test_film / f"TrackedCells_{test_film}")
                    p1 = t_dir / f"cell_{t_cid1}_masks.csv" if t_dir else None
                    p2 = t_dir / f"cell_{t_cid2}_masks.csv" if t_dir else None
                    if p1 and p2 and p1.exists() and p2.exists():
                        try:
                            df1 = pd.read_csv(p1)
                            df2 = pd.read_csv(p2)
                            col1 = 'rle_gfp' if 'FL' in test_film and 'rle_gfp' in df1.columns else 'rle_bf'
                            col2 = 'rle_gfp' if 'FL' in test_film and 'rle_gfp' in df2.columns else 'rle_bf'
                            
                            c1_x, c1_y, c2_x, c2_y = None, None, None, None
                            for _, r in df1.iterrows():
                                if pd.notna(r.get(col1)) and str(r.get(col1)).strip():
                                    m1 = validate_and_decode_rle(str(r[col1]), int(r['height']), int(r['width']))
                                    ys, xs = m1.nonzero()
                                    if len(xs) > 0:
                                        c1_x, c1_y = float(xs.mean()), float(ys.mean())
                                        break
                            for _, r in df2.iterrows():
                                if pd.notna(r.get(col2)) and str(r.get(col2)).strip():
                                    m2 = validate_and_decode_rle(str(r[col2]), int(r['height']), int(r['width']))
                                    ys, xs = m2.nonzero()
                                    if len(xs) > 0:
                                        c2_x, c2_y = float(xs.mean()), float(ys.mean())
                                        break
                            if c1_x is not None and c2_x is not None:
                                dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
                                if dist <= 300.0:
                                    is_adjacent = True
                                    break
                        except Exception:
                            pass
                    is_adjacent = True
                    break
        if is_adjacent:
            break
        if not diverged:
            is_adjacent = True
            break

    if not is_adjacent:
        return local_cid, cur_track, False

    # 3. Duplicate local cell in target_film
    t_dir = resolve_under_root(movie_root, Path(exp) / target_film / f"TrackedCells_{target_film}")
    if not t_dir or not t_dir.exists():
        return local_cid, cur_track, False

    max_cid = 0
    for p in t_dir.glob("cell_*_masks.csv"):
        if p.name.startswith("._"): continue
        m = re.match(r"^cell_(\d+)_masks\.csv$", p.name)
        if m:
            max_cid = max(max_cid, int(m.group(1)))

    new_local_cid = max_cid + 1
    src_csv = t_dir / f"cell_{local_cid}_masks.csv"
    dst_csv = t_dir / f"cell_{new_local_cid}_masks.csv"

    if src_csv.exists():
        shutil.copy2(src_csv, dst_csv)

    new_track = list(cur_track)
    if f_idx < len(new_track):
        new_track[f_idx] = new_local_cid

    return new_local_cid, new_track, True

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


def find_matching_local_cell(movie_root: Path, exp: str, film: str, local_t: int, user_rle: str = "", click_x: int = None, click_y: int = None) -> Optional[int]:
    """Finds the local cell ID in TrackedCells_<film> that matches the user selected mask or coordinate."""
    film_dir = resolve_under_root(movie_root, Path(exp) / film / f"TrackedCells_{film}")
    if not film_dir or not film_dir.exists():
        return None
        
    r_col = "rle_gfp" if "FL" in film else "rle_bf"
    
    target_c = None
    user_mask = None
    if user_rle and user_rle != "nan":
        try:
            user_mask = validate_and_decode_rle(user_rle, 2000, 2000)
            ys, xs = np.where(user_mask > 0)
            if len(ys) > 0:
                target_c = (float(np.mean(xs)), float(np.mean(ys)))
        except Exception:
            pass
    elif click_x is not None and click_y is not None:
        target_c = (float(click_x), float(click_y))
        
    if target_c is None:
        return None
        
    best_cid = None
    best_score = 0
    
    for p in film_dir.glob("cell_*_masks.csv"):
        if p.name.startswith("._"):
            continue
        m = re.match(r"^cell_(\d+)_masks\.csv$", p.name)
        if not m:
            continue
        cid = int(m.group(1))
        
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
            
        if r_col not in df.columns:
            continue
        row = df[df["time_point"] == local_t]
        if row.empty:
            continue
            
        cand_rle = str(row.iloc[0].get(r_col, ""))
        if not cand_rle or cand_rle == "nan":
            continue
            
        if user_mask is not None:
            try:
                cand_mask = validate_and_decode_rle(cand_rle, 2000, 2000)
                inter = int(np.sum(user_mask & cand_mask))
                if inter > best_score:
                    best_score = inter
                    best_cid = cid
            except Exception:
                pass
        elif click_x is not None and click_y is not None:
            if _point_in_rle(cand_rle, int(click_x), int(click_y)):
                return cid
                
    if user_mask is not None and best_score > 0:
        return best_cid
    return None


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
        # Unlinked slot: find overlapping existing local cell or allocate new unique cid in target_film
        found_cid = None
        cfg = current_app.extensions["config"]
        if req.changes and len(req.changes) > 0:
            gt_t = req.changes[0].time_point
            rle_val = getattr(req.changes[0], "new_rle", getattr(req.changes[0], "rle", ""))
            k_res = resolve_gt_keyframe(req.experiment, req.sequence, req.film, gt_t, frames_svc)
            local_t = k_res.get("frame_idx", 0)
            
            if rle_val:
                found_cid = find_matching_local_cell(cfg.local_movie_root, req.experiment, target_film, local_t, user_rle=rle_val)
                
        if found_cid is not None:
            local_cid = found_cid
        else:
            # Allocate next available local ID in target_film
            t_dir = resolve_under_root(cfg.local_movie_root, Path(req.experiment) / target_film / f"TrackedCells_{target_film}")
            max_cid = 0
            if t_dir and t_dir.exists():
                for p in t_dir.glob("cell_*_masks.csv"):
                    if p.name.startswith("._"): continue
                    m = re.match(r"^cell_(\d+)_masks\.csv$", p.name)
                    if m:
                        max_cid = max(max_cid, int(m.group(1)))
            local_cid = max_cid + 1

    # Auto-sync track linkage in sequence_linkage.json: update -1 to local_cid
    if req.sequence and 'f_idx' in locals() and f_idx < len(track) and track[f_idx] != local_cid:
        track[f_idx] = local_cid
        linkage_repo = current_app.extensions.get("linkage_repository")
        if linkage_repo:
            try:
                raw_data, rev = linkage_repo.load_linkage(req.experiment)
                seq_obj = raw_data.get("sequences", raw_data)
                if req.sequence in seq_obj and "global_cells" in seq_obj[req.sequence]:
                    cell_key = normalize_cell_key(seq_obj[req.sequence]["global_cells"], req.cell_id, req.sequence)
                    seq_obj[req.sequence]["global_cells"][cell_key] = track
                    linkage_repo.save_linkage(req.experiment, raw_data)
                    current_app.logger.info(f"Auto-linked {req.sequence} global cell {req.cell_id} on film {target_film} -> local #{local_cid}")
            except Exception as e:
                current_app.logger.warning(f"Could not auto-sync linkage: {e}")

    # Check for auto-duplication when brush/eraser modifies a shared adjacent local cell
    if req.sequence and 'f_idx' in locals() and local_cid and local_cid > 0:
        tool_used = getattr(req, "tool", "select") or "select"
        if tool_used in ["brush", "eraser"]:
            cfg = current_app.extensions["config"]
            new_cid, updated_track, was_dup = check_and_duplicate_shared_local_cell(
                cfg.local_movie_root,
                req.experiment,
                req.sequence,
                target_film,
                f_idx,
                str(req.cell_id),
                local_cid,
                tool_used,
                films,
                global_cells
            )
            if was_dup:
                local_cid = new_cid
                track = updated_track
                linkage_repo = current_app.extensions.get("linkage_repository")
                if linkage_repo:
                    try:
                        raw_data, rev = linkage_repo.load_linkage(req.experiment)
                        seq_obj = raw_data.get("sequences", raw_data)
                        if req.sequence in seq_obj and "global_cells" in seq_obj[req.sequence]:
                            ck = normalize_cell_key(seq_obj[req.sequence]["global_cells"], req.cell_id, req.sequence)
                            seq_obj[req.sequence]["global_cells"][ck] = track
                            linkage_repo.save_linkage(req.experiment, raw_data)
                            current_app.logger.info(f"Auto-duplicated shared local cell in {req.sequence} global cell {req.cell_id} on film {target_film} -> local #{local_cid}")
                    except Exception as e:
                        current_app.logger.warning(f"Could not auto-sync linkage after duplication: {e}")

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
        "track": track if req.sequence else None,
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


def _point_in_rle(rle_str: str, x: int, y: int, H: int = 2000, W: int = 2000) -> bool:
    """Check if (x, y) pixel is within 1-based Fortran column-major RLE intervals."""
    if not rle_str or str(rle_str) == "nan" or not isinstance(rle_str, str):
        return False
    idx_1 = x * H + y + 1
    parts = list(map(int, rle_str.strip().split()))
    for s, l in zip(parts[0::2], parts[1::2]):
        if s <= idx_1 < s + l:
            return True
    return False


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

    # Search at local_t and slight neighbor frames for robustness
    t_candidates = [local_t]
    for delta in [1, -1, 2, -2, 3, -3]:
        tc = local_t + delta
        if tc >= 0:
            t_candidates.append(tc)

    # 1. Fast path: check TrackedCells CSVs directly via analytic RLE intervals (<1ms)
    tracked_dir = base_root / exp / target_film / f"TrackedCells_{target_film}"
    if tracked_dir.exists():
        for csv_file in sorted(tracked_dir.glob("cell_*_masks.csv")):
            if csv_file.name.startswith("."):
                continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_file.name)
            if not m:
                continue
            cid = int(m.group(1))

            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue
                H = int(df.iloc[0].get("height", 2000))
                W = int(df.iloc[0].get("width", 2000))
                if x_val < 0 or y_val < 0 or x_val >= W or y_val >= H:
                    continue

                found = False
                for t_cand in t_candidates:
                    rows = df[df["time_point"] == t_cand]
                    if rows.empty:
                        continue
                    for rle_col in ["rle_gfp", "rle_bf"]:
                        if rle_col in df.columns:
                            val = str(rows.iloc[0].get(rle_col, "")).strip()
                            if val and val != "nan":
                                if _point_in_rle(val, x_val, y_val, H, W):
                                    local_cid = cid
                                    found = True
                                    break
                    if found:
                        break
                if found:
                    break
            except Exception:
                continue

    # 2. Fallback: check _seg.tif directly and resolve via seg_label_identity
    if local_cid is None:
        masks_dir = base_root / exp / target_film / f"Masks_{target_film}"
        if masks_dir.exists():
            files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
            if not files:
                files = sorted([f for f in masks_dir.glob(f"*_t{local_t:03d}_*_seg.tif") if not f.name.startswith(".")])
            if files:
                try:
                    seg = imread(str(files[0]))
                    if seg.ndim > 2:
                        seg = seg[..., 0]
                    H, W = seg.shape[:2]
                    if 0 <= y_val < H and 0 <= x_val < W:
                        hit_lbl = int(seg[y_val, x_val])
                        if hit_lbl > 0:
                            local2global = frames_svc.local_to_global_map(exp, target_film, sequence)
                            ident = frames_svc.seg_label_identity(exp, target_film, local_t, seg, H, W, local2global, sequence=sequence)
                            if hit_lbl in ident:
                                local_cid = hit_lbl
                            else:
                                local_cid = hit_lbl
                except Exception:
                    pass

    if local_cid is None:
        return jsonify({"status": "not_found", "message": "No cell found at coordinates"}), 404

    # 3. Map local_cid to global cell ID if in a sequence
    global_cid = str(local_cid)
    if sequence:
        seq_res = linkage_svc.get_sequences(exp)
        seq_info = seq_res.get("sequences", {}).get(sequence, {})
        films_list = seq_info.get("films", [])
        global_cells = seq_info.get("global_cells", {})
        if target_film in films_list:
            f_idx = films_list.index(target_film)
            
            # If current_cell_id already matches this local_cid, keep it!
            current_cell_id = data.get("current_cell_id")
            if current_cell_id and current_cell_id in global_cells:
                cur_track = global_cells[current_cell_id]
                if f_idx < len(cur_track) and cur_track[f_idx] == local_cid:
                    global_cid = current_cell_id
                else:
                    for gid, track in global_cells.items():
                        if f_idx < len(track) and track[f_idx] == local_cid:
                            global_cid = gid
                            break
            else:
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



