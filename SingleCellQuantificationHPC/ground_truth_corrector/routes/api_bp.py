from flask import Blueprint, jsonify, request, current_app
from ..schemas import SaveQCRequest, ExportTrainingDataRequest

api_bp = Blueprint("api", __name__)

@api_bp.route("/api/list_experiments", methods=["GET"])
def list_experiments():
    exp_svc = current_app.extensions["experiments_service"]
    experiments = exp_svc.list_experiments()
    return jsonify({"experiments": experiments})


@api_bp.route("/api/list_films_and_sequences", methods=["GET"])
def list_films_and_sequences():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"error": "experiment is required"}), 400
    exp_svc = current_app.extensions["experiments_service"]
    data = exp_svc.list_films_and_sequences(exp)
    return jsonify(data)


@api_bp.route("/api/list_cells", methods=["GET"])
def list_cells():
    exp = request.args.get("experiment")
    sequence = request.args.get("sequence")
    film = request.args.get("film")
    
    if not exp:
        return jsonify({"error": "experiment required"}), 400
        
    tracking_svc = current_app.extensions["tracking_service"]
    if sequence:
        data = tracking_svc.list_cells_for_sequence(exp, sequence)
        return jsonify(data)
    elif film:
        cells = tracking_svc.list_cells_for_film(exp, film)
        return jsonify({"cells": cells, "lineage": {}})
        
    return jsonify({"cells": [], "lineage": {}})


@api_bp.route("/api/get_qc", methods=["GET"])
def get_qc():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    sequence = request.args.get("sequence")
    
    if not exp:
        return jsonify({"error": "experiment required"}), 400
        
    qc_repo = current_app.extensions["qc_repository"]
    if film:
        return jsonify({"qc": qc_repo.load_qc(exp, film)})
    elif sequence:
        combined = qc_repo.load_sequence_qc(exp, sequence)
        linkage_svc = current_app.extensions["linkage_service"]
        seq_res = linkage_svc.get_sequences(exp)
        seq_info = seq_res.get("sequences", {}).get(sequence, {})
        films = seq_info.get("films", [sequence])
        global_cells = seq_info.get("global_cells", {})
        
        film_qcs = {f: qc_repo.load_qc(exp, f) for f in films}
        for gid, track in global_cells.items():
            if gid in combined:
                continue
            for f_idx, cid in enumerate(track):
                if cid != -1 and f_idx < len(films):
                    f_name = films[f_idx]
                    cid_str = str(cid)
                    if cid_str in film_qcs[f_name]:
                        combined[gid] = film_qcs[f_name][cid_str]
                        break
        return jsonify({"qc": combined})
    return jsonify({"qc": {}})


@api_bp.route("/api/save_qc", methods=["POST"])
def save_qc():
    data = request.get_json() or {}
    try:
        req = SaveQCRequest(**data)
    except Exception as e:
        return jsonify({"error": f"Invalid request: {e}"}), 400

    qc_repo = current_app.extensions["qc_repository"]
    if req.sequence:
        qc_repo.save_sequence_qc_entry(
            exp=req.experiment,
            sequence=req.sequence,
            global_id=req.cell_id,
            status=req.status,
            reasons=req.reasons,
            note=req.note,
            reviewer=req.reviewer
        )

    film = req.film
    if not film and req.sequence:
        linkage_svc = current_app.extensions["linkage_service"]
        seq_res = linkage_svc.get_sequences(req.experiment)
        seq_info = seq_res.get("sequences", {}).get(req.sequence, {})
        films = seq_info.get("films", [req.sequence])
        film = films[0] if films else req.sequence

    if film:
        qc_repo.save_qc_entry(
            exp=req.experiment,
            film=film,
            cell_id=req.cell_id,
            status=req.status,
            reasons=req.reasons,
            note=req.note,
            reviewer=req.reviewer
        )
    return jsonify({"status": "success"})


@api_bp.route("/api/export_training_data", methods=["POST"])
def export_training_data():
    data = request.get_json() or {}
    try:
        req = ExportTrainingDataRequest(**data)
    except Exception as e:
        return jsonify({"error": f"Invalid request: {e}"}), 400

    export_svc = current_app.extensions["gt_export_service"]
    res = export_svc.export_all_keyframes(
        exp=req.experiment,
        sequence=req.sequence,
        custom_subfolder=req.subfolder
    )
    return jsonify(res)


@api_bp.route("/api/sync_keyframe_training", methods=["POST"])
def sync_keyframe_training():
    data = request.get_json() or {}
    exp = data.get("experiment")
    film = data.get("film")
    local_t = data.get("time_point", 0)
    subfolder = data.get("subfolder")

    if not exp or not film:
        return jsonify({"error": "experiment and film required"}), 400

    export_svc = current_app.extensions["gt_export_service"]
    res = export_svc.sync_keyframe_to_training(exp, film, int(local_t), subfolder)
    return jsonify(res)


@api_bp.route("/api/update_linkage", methods=["POST"])
def update_linkage():
    data = request.get_json() or {}
    from ..schemas import UpdateLinkageRequest
    try:
        req = UpdateLinkageRequest(**data)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Validation error: {e}"}), 400

    linkage_service = current_app.extensions["linkage_service"]
    res = linkage_service.update_linkage(req)
    return jsonify(res)

