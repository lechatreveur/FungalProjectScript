from flask import Blueprint, render_template, jsonify, request, current_app

board_bp = Blueprint("board", __name__)

@board_bp.route("/")
def index():
    return render_template("index.html")

@board_bp.route("/api/list_cells", methods=["GET"])
def list_cells():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment parameter required"}), 400
        
    seq = request.args.get("sequence")
    if not seq:
        film = request.args.get("film")
        if film:
            seq = film
        else:
            return jsonify({"status": "error", "message": "sequence parameter required"}), 400

    tracking_service = current_app.config["TRACKING_SERVICE"]
    res = tracking_service.list_sequence_cells(exp, seq)
    return jsonify(res)
