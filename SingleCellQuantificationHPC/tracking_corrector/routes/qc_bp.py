from flask import Blueprint, jsonify, request, current_app

qc_bp = Blueprint("qc", __name__)

@qc_bp.route("/api/get_qc", methods=["GET"])
def get_qc():
    exp = request.args.get("experiment")
    target = request.args.get("sequence") or request.args.get("film")
    if not exp or not target:
        return jsonify({"status": "error", "message": "experiment and sequence/film required"}), 400
        
    qc_service = current_app.config["QC_SERVICE"]
    data = qc_service.get_qc_records(exp, target)
    records = data.get("qc_records", {})
    return jsonify({"status": "success", "qc": records})

@qc_bp.route("/api/save_qc", methods=["POST"])
def save_qc():
    data = request.get_json() or {}
    exp = data.get("experiment")
    target = data.get("sequence") or data.get("film")
    cell_id = str(data.get("cell_id"))
    status = data.get("status")
    
    if not exp or not target or not cell_id or status is None:
        return jsonify({"status": "error", "message": "experiment, sequence/film, cell_id, and status required"}), 400
        
    qc_service = current_app.config["QC_SERVICE"]
    res = qc_service.save_qc_record_simple(exp, target, cell_id, status)
    return jsonify(res)

@qc_bp.route("/api/suspicious_cells", methods=["GET"])
def suspicious_cells():
    exp = request.args.get("experiment")
    target = request.args.get("sequence") or request.args.get("film")
    if not exp or not target:
        return jsonify({"status": "error", "message": "experiment and sequence/film required"}), 400
        
    suspicious_service = current_app.config["SUSPICIOUS_SERVICE"]
    res = suspicious_service.analyze_suspicious_cells(exp, target)
    return jsonify({"suspicious": res.get("suspicious", {})})
