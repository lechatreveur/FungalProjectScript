import cv2
import numpy as np
from pathlib import Path
from tifffile import imread
from skimage.measure import label
from flask import Blueprint, jsonify, request, current_app, Response

from typing import Optional, Dict, Any, List
from ..security import resolve_under_root
from ..schemas import validate_and_decode_rle

frames_bp = Blueprint("frames", __name__)

def resolve_gt_keyframe(exp: str, sequence: Optional[str], film: Optional[str], global_t: int, frames_service) -> dict:
    """Resolve global keyframe index to target film and exact local timepoint."""
    if film:
        k_times = frames_service.get_film_keyframes(exp, film)
        clamped_idx = max(0, min(global_t, len(k_times) - 1))
        local_t = k_times[clamped_idx] if k_times else 0
        pos = "First" if clamped_idx == 0 else ("Last" if clamped_idx == len(k_times) - 1 else "Middle")
        return {
            "film": film,
            "film_idx": 0,
            "local_t": local_t,
            "keyframe_idx": clamped_idx,
            "keyframe_pos": pos,
            "total_keyframes": len(k_times)
        }
    elif sequence:
        k_map = frames_service.get_sequence_keyframe_map(exp, sequence)
        if not k_map:
            return {"film": sequence, "film_idx": 0, "local_t": 0, "keyframe_idx": 0, "keyframe_pos": "First", "total_keyframes": 1}
        clamped_idx = max(0, min(global_t, len(k_map) - 1))
        item = k_map[clamped_idx]
        return {
            "film": item["film"],
            "film_idx": item.get("film_idx", 0),
            "local_t": item["local_t"],
            "keyframe_idx": clamped_idx,
            "keyframe_pos": item["keyframe_pos"],
            "total_keyframes": len(k_map)
        }
    return {"film": "", "film_idx": 0, "local_t": 0, "keyframe_idx": 0, "keyframe_pos": "First", "total_keyframes": 1}


@frames_bp.route("/api/keyframes_info", methods=["GET"])
def keyframes_info():
    exp = request.args.get("experiment")
    sequence = request.args.get("sequence")
    film = request.args.get("film")
    
    if not exp:
        return jsonify({"error": "experiment is required"}), 400

    frames_svc = current_app.extensions["gt_frames_service"]
    if sequence:
        k_map = frames_svc.get_sequence_keyframe_map(exp, sequence)
        return jsonify({"keyframes": k_map, "total": len(k_map)})
    elif film:
        k_times = frames_svc.get_film_keyframes(exp, film)
        k_map = [
            {
                "global_t": idx,
                "film": film,
                "local_t": t_val,
                "keyframe_pos": "First" if idx == 0 else ("Last" if idx == len(k_times) - 1 else "Middle"),
                "keyframe_idx": idx
            }
            for idx, t_val in enumerate(k_times)
        ]
        return jsonify({"keyframes": k_map, "total": len(k_map)})
    return jsonify({"keyframes": [], "total": 0})


@frames_bp.route("/api/frame_image", methods=["GET"])
def get_frame_image():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    sequence = request.args.get("sequence")
    t_str = request.args.get("t", "0")
    channel = request.args.get("channel", "bf")
    
    if not exp:
        return jsonify({"error": "experiment required"}), 400
        
    try:
        t_val = int(t_str)
    except ValueError:
        t_val = 0

    frames_svc = current_app.extensions["gt_frames_service"]
    res = resolve_gt_keyframe(exp, sequence, film, t_val, frames_svc)
    target_film = res["film"]
    local_t = res["local_t"]

    try:
        jpeg_bytes = frames_svc.render_frame_jpeg(exp, target_film, local_t, channel=channel)
        resp = Response(jpeg_bytes, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception as e:
        return jsonify({"error": f"Failed to render frame: {e}"}), 500


@frames_bp.route("/api/population_frame", methods=["GET"])
def get_population_frame():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    sequence = request.args.get("sequence")
    t_str = request.args.get("t", "0")
    
    if not exp:
        return jsonify({"error": "experiment required"}), 400
        
    try:
        t_val = int(t_str)
    except ValueError:
        t_val = 0

    frames_svc = current_app.extensions["gt_frames_service"]
    res = resolve_gt_keyframe(exp, sequence, film, t_val, frames_svc)
    target_film = res["film"]
    local_t = res["local_t"]

    try:
        jpeg_bytes = frames_svc.render_population_frame_jpeg(exp, target_film, local_t, sequence=sequence)
        resp = Response(jpeg_bytes, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception as e:
        return jsonify({"error": f"Failed to render population frame: {e}"}), 500


@frames_bp.route("/api/frame_boundaries", methods=["GET"])
def get_frame_boundaries():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    sequence = request.args.get("sequence")
    t_str = request.args.get("t", "0")
    
    if not exp:
        return jsonify({"error": "experiment required"}), 400
        
    try:
        t_val = int(t_str)
    except ValueError:
        t_val = 0

    frames_svc = current_app.extensions["gt_frames_service"]
    base_root = frames_svc.config.local_movie_root
    res = resolve_gt_keyframe(exp, sequence, film, t_val, frames_svc)
    target_film = res["film"]
    local_t = res["local_t"]

    cache_key = (exp, target_film, local_t, sequence)
    if cache_key in frames_svc._boundaries_cache:
        resp = Response(frames_svc._boundaries_cache[cache_key], mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    masks_dir = base_root / exp / target_film / f"Masks_{target_film}"
    files = []
    if masks_dir.exists():
        files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in masks_dir.glob(f"*_t{local_t:03d}_*_seg.tif") if not f.name.startswith(".")])
            
    if not files:
        # Transparent 1x1 fallback PNG
        empty_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        resp = Response(empty_png, mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    try:
        from ..services.gt_frames_service import id_to_color, UNTRACKED_COLOR

        seg = imread(str(files[0]))
        seg_lbl = (label(seg) if seg.dtype == bool else seg).copy()

        # Build boundary outline overlay image (BGRA, as cv2.imencode expects).
        H, W = seg_lbl.shape[:2]
        rgba = np.zeros((H, W, 4), dtype=np.uint8)

        # Colour each cell's outline by its stable identity, so a cell keeps one
        # colour across keyframes / films (matches the population view).
        local2global = frames_svc.local_to_global_map(exp, target_film, sequence)
        ident = frames_svc.seg_label_identity(exp, target_film, local_t, seg_lbl, H, W, local2global, sequence=sequence)
        for lbl in np.unique(seg_lbl):
            if lbl == 0:
                continue
            hit = ident.get(int(lbl))
            b, g, r = id_to_color(hit[0]) if hit is not None else UNTRACKED_COLOR
            contours, _ = cv2.findContours(
                (seg_lbl == lbl).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(rgba, contours, -1, (int(b), int(g), int(r), 255), 1)

        is_success, buffer = cv2.imencode(".png", rgba)
        if not is_success:
            raise RuntimeError("Failed to encode boundaries PNG")
        png_bytes = buffer.tobytes()
        frames_svc._boundaries_cache[cache_key] = png_bytes
        resp = Response(png_bytes, mimetype="image/png")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception as e:
        return jsonify({"error": f"Failed to render boundaries: {e}"}), 500
