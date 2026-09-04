from flask import Blueprint, jsonify, request, current_app
from pydantic import ValidationError as PydanticValidationError
from ..schemas import SaveSeptumRequest

septum_bp = Blueprint("septum", __name__)

@septum_bp.route("/api/get_septum_label", methods=["GET"])
def get_septum_label():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    cell_id = request.args.get("cell_id")
    if not exp or not film:
        return jsonify({"status": "error", "message": "experiment and film required"}), 400
        
    septum_service = current_app.config["SEPTUM_SERVICE"]
    res = septum_service.get_septum_alignment(exp, film, cell_id)
    return jsonify(res)

@septum_bp.route("/api/save_septum_label", methods=["POST"])
def save_septum_label():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "JSON body required"}), 400
        
    try:
        req = SaveSeptumRequest(**data)
    except PydanticValidationError as e:
        log_file = "/Users/user/.gemini/antigravity-ide/brain/1f0254e9-f034-45b1-a6e8-d72f128eab8b/flask_error.log"
        with open(log_file, "a") as f:
            f.write(f"PydanticValidationError: {e}\n\n")
        print(f"[ERROR] save_septum_label ValidationError: {e}")
        return jsonify({"status": "error", "code": "VALIDATION_ERROR", "message": str(e)}), 422
        
    try:
        septum_service = current_app.config["SEPTUM_SERVICE"]
        res = septum_service.save_septum_label(req)
    except Exception as e:
        import traceback
        log_file = "/Users/user/.gemini/antigravity-ide/brain/1f0254e9-f034-45b1-a6e8-d72f128eab8b/flask_error.log"
        with open(log_file, "a") as f:
            f.write(f"Custom ValidationError: {e}\n{traceback.format_exc()}\n\n")
        raise e
    
    # Invalidate cache
    cache_service = current_app.config["CACHE_SERVICE"]
    cache_service.clear_all_caches_for_film(req.experiment, req.film)
    
    return jsonify(res)

@septum_bp.route("/api/predict_septum", methods=["POST"])
def predict_septum():
    data = request.get_json() or {}
    exp = data.get("experiment")
    film = data.get("film")
    cell_id_str = data.get("cell_id")
    if not exp or not film or cell_id_str is None:
        return jsonify({"status": "error", "message": "experiment, film, and cell_id required"}), 400
        
    cell_id = int(cell_id_str)
    septum_service = current_app.config["SEPTUM_SERVICE"]
    res = septum_service.predict_septum(
        exp,
        film,
        cell_id,
        sequence=data.get("sequence"),
        global_cell_id=data.get("global_cell_id"),
    )
    return jsonify(res)

@septum_bp.route("/api/get_septum_ai_cache", methods=["GET"])
def get_septum_ai_cache():
    """Cheap, file-only lookup of a pre-computed (offline batch) AI
    suggestion, keyed by global_cell_id within a sequence. Unlike
    /api/predict_septum, this never runs the model - it just reads whatever
    an external batch job already wrote, so it's safe to call automatically
    whenever a cell is opened instead of requiring a manual "Run AI" click.
    Returns {"cached": false} (not an error) when no sequence/global_cell_id
    was given or nothing has been cached for that cell yet.
    """
    exp = request.args.get("experiment")
    sequence = request.args.get("sequence")
    global_cell_id = request.args.get("global_cell_id")
    if not exp or not sequence or not global_cell_id:
        return jsonify({"status": "success", "cached": False})

    septum_service = current_app.config["SEPTUM_SERVICE"]
    res = septum_service.get_cached_ai_suggestion(exp, sequence, global_cell_id)
    return jsonify(res)


@septum_bp.route("/api/septum_review/crop", methods=["GET"])
def septum_review_crop():
    """Return a rotation-normalised single-frame cell crop with AI overlay data.

    Query parameters
    ----------------
    experiment  str   required
    film        str   required
    cell_id     int   required
    frame       int   required  (zero-based local frame index within *film*)
    channel     str   optional  ``bf`` (default) or ``gfp``
    crop_size   int   optional  side length in pixels, clamped to [32, 512], default 160

    Response JSON
    -------------
    image_b64              base64-encoded JPEG of the rotation-normalised crop
    rotation_deg           CCW degrees applied (negative = CW); 0 = already vertical
    crop_size              actual crop side used
    centroid_in_crop       [row, col] of the mask centroid in the output image
    state_prob             per-frame septum-state probability, 0–1, or null
    start_prob             per-frame septum-start probability, 0–1, or null
    end_prob               per-frame septum-end probability, 0–1, or null
    warning                model confidence caveat (held-out F1=0.240...)
    model_metrics          dict of held-out benchmark values from evaluation.json
    overlay.septum_center_in_crop  [row, col] of the labeled septum centre (rotated),
                           or null when cell_data.csv has no entry for this frame
    """
    exp = request.args.get("experiment")
    film = request.args.get("film")
    cell_id_str = request.args.get("cell_id")
    frame_str = request.args.get("frame")
    channel = (request.args.get("channel") or "bf").lower().strip()
    try:
        crop_size = int(request.args.get("crop_size") or 160)
    except (TypeError, ValueError):
        crop_size = 160

    if not exp or not film or cell_id_str is None or frame_str is None:
        return jsonify({
            "status": "error",
            "message": "experiment, film, cell_id, and frame are required",
        }), 400

    try:
        cell_id = int(cell_id_str)
        frame = int(frame_str)
    except (TypeError, ValueError):
        return jsonify({
            "status": "error",
            "message": "cell_id and frame must be integers",
        }), 400

    if channel not in ("bf", "gfp"):
        channel = "bf"
    crop_size = max(32, min(512, crop_size))

    septum_service = current_app.config["SEPTUM_SERVICE"]
    try:
        result = septum_service.rotated_crop_with_prediction(
            exp, film, cell_id, frame, channel, crop_size
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@septum_bp.route("/mobile_septum", methods=["GET"])
@septum_bp.route("/mobile_septum_review", methods=["GET"])
def mobile_septum_review_page():
    from flask import render_template
    return render_template("mobile_septum_review.html")


@septum_bp.route("/api/septum_review/batch", methods=["GET"])
def septum_review_batch():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    if not exp or not film:
        return jsonify({"status": "error", "message": "experiment and film are required"}), 400

    try:
        count = int(request.args.get("count") or 20)
    except (TypeError, ValueError):
        count = 20

    septum_service = current_app.config["SEPTUM_SERVICE"]
    res = septum_service.get_mobile_septum_batch(exp, film, count=count)
    return jsonify(res)


@septum_bp.route("/api/septum_review/save", methods=["POST"])
def septum_review_save():
    data = request.get_json() or {}
    if not data:
        return jsonify({"status": "error", "message": "JSON body required"}), 400

    septum_service = current_app.config["SEPTUM_SERVICE"]
    try:
        res = septum_service.save_mobile_septum_review(data)
        return jsonify(res)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@septum_bp.route("/api/septum_review/stats", methods=["GET"])
def septum_review_stats():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    if not exp or not film:
        return jsonify({"status": "error", "message": "experiment and film are required"}), 400

    septum_service = current_app.config["SEPTUM_SERVICE"]
    res = septum_service.get_mobile_septum_stats(exp, film)
    return jsonify(res)


@septum_bp.route("/api/septum_review/al_status", methods=["GET"])
def septum_review_al_status():
    al_service = current_app.config.get("ACTIVE_LEARNING_SERVICE")
    if not al_service:
        return jsonify({"status": "error", "message": "Active learning service unavailable"}), 500
    state = al_service.get_state()
    return jsonify({"status": "success", "al_state": state})


@septum_bp.route("/api/septum_review/retrain_history", methods=["GET"])
def septum_review_retrain_history():
    al_service = current_app.config.get("ACTIVE_LEARNING_SERVICE")
    if not al_service:
        return jsonify({"status": "error", "message": "Active learning service unavailable"}), 500
    state = al_service.get_state()
    history = state.get("retrain_history", [])
    return jsonify({
        "status": "success",
        "current_round": state.get("current_round", 1),
        "current_live_checkpoint": state.get("current_live_checkpoint"),
        "current_baseline_metrics": state.get("current_baseline_metrics"),
        "history": history,
    })


@septum_bp.route("/api/septum_review/trigger_retrain", methods=["POST"])
def septum_review_trigger_retrain():
    data = request.get_json() or {}
    epochs = int(data.get("epochs", 30))
    test_mode = bool(data.get("test_mode", False))
    al_service = current_app.config.get("ACTIVE_LEARNING_SERVICE")
    if not al_service:
        return jsonify({"status": "error", "message": "Active learning service unavailable"}), 500
    res = al_service.trigger_manual_retrain(epochs=epochs, test_mode=test_mode)
    return jsonify(res)



