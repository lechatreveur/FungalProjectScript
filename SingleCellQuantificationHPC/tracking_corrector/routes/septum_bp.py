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
