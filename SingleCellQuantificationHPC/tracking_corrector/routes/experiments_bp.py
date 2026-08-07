from flask import Blueprint, jsonify, request, current_app

experiments_bp = Blueprint("experiments", __name__)

@experiments_bp.route("/api/list_experiments", methods=["GET"])
def list_experiments():
    exp_service = current_app.config["EXP_SERVICE"]
    exps = exp_service.list_experiments()
    exp_ids = [e["id"] for e in exps if e.get("discovered", True)]
    return jsonify({"experiments": exp_ids})

@experiments_bp.route("/api/list_films_and_sequences", methods=["GET"])
def list_films_and_sequences():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment parameter required"}), 400
        
    exp_service = current_app.config["EXP_SERVICE"]
    linkage_service = current_app.config["LINKAGE_SERVICE"]
    frames_service = current_app.config.get("FRAMES_SERVICE")
    
    films = exp_service.discover_films(exp)
    seq_data = linkage_service.get_sequences(exp)
    sequences = list(seq_data.get("sequences", {}).keys())
    
    if frames_service is not None:
        for film in films:
            frames_service.trigger_pregeneration(exp, film)
            
    return jsonify({
        "films": films,
        "sequences": sequences
    })
