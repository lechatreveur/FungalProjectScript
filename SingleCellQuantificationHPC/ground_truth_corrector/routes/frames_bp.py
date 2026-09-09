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
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
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
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
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

    try:
        png_bytes = frames_svc.render_boundary_png(exp, target_film, local_t, sequence=sequence)
        resp = Response(png_bytes, mimetype="image/png")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    except Exception as e:
        return jsonify({"error": f"Failed to render boundaries: {e}"}), 500
