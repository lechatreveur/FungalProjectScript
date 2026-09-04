from flask import Blueprint, jsonify, request, current_app
from ..qc_schema import InvalidQCStatusError

qc_bp = Blueprint("qc", __name__)

@qc_bp.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@qc_bp.route("/api/get_qc", methods=["GET", "OPTIONS"])
def get_qc():
    if request.method == "OPTIONS":
        return "", 200
    exp = request.args.get("experiment")
    target = request.args.get("sequence") or request.args.get("film")
    level = request.args.get("level", "global")
    if not exp or not target:
        return jsonify({"status": "error", "message": "experiment and sequence/film required"}), 400
        
    qc_service = current_app.config["QC_SERVICE"]
    data = qc_service.get_qc_records(exp, target, level=level)
    records = data.get("qc_records", {})
    return jsonify({"status": "success", "qc": records})

@qc_bp.route("/api/save_qc", methods=["POST", "OPTIONS"])
def save_qc():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json() or {}
    exp = data.get("experiment")
    target = data.get("sequence") or data.get("film")
    cell_id = str(data.get("cell_id"))
    status = data.get("status")
    level = data.get("level", "global")
    
    if not exp or not target or not cell_id or status is None:
        return jsonify({"status": "error", "message": "experiment, sequence/film, cell_id, and status required"}), 400
        
    qc_service = current_app.config["QC_SERVICE"]
    try:
        res = qc_service.save_qc_record_simple(exp, target, cell_id, status, level=level)
        return jsonify(res)
    except InvalidQCStatusError as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@qc_bp.route("/api/suspicious_cells", methods=["GET"])
def suspicious_cells():
    exp = request.args.get("experiment")
    target = request.args.get("sequence") or request.args.get("film")
    if not exp or not target:
        return jsonify({"status": "error", "message": "experiment and sequence/film required"}), 400
        
    suspicious_service = current_app.config["SUSPICIOUS_SERVICE"]
    res = suspicious_service.analyze_suspicious_cells(exp, target)
    return jsonify({"suspicious": res.get("suspicious", {})})
