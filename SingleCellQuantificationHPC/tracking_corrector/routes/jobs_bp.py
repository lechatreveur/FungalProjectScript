from flask import Blueprint, jsonify, request, current_app

jobs_bp = Blueprint("jobs", __name__)

@jobs_bp.route("/api/quantify_on_hpc", methods=["POST"])
def quantify_on_hpc():
    data = request.get_json() or {}
    exp = data.get("experiment")
    film = data.get("film")
    cell_id = data.get("cell_id")
    
    if not exp or not film or cell_id is None:
        return jsonify({"status": "error", "message": "experiment, film, and cell_id required"}), 400
        
    quant_service = current_app.config["QUANTIFICATION_SERVICE"]
    res = quant_service.trigger_quantification(exp, film, int(cell_id))
    return jsonify(res)

@jobs_bp.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    jobs_service = current_app.config["JOBS_SERVICE"]
    st = jobs_service.get_job_status(job_id)
    if not st:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify(st)
