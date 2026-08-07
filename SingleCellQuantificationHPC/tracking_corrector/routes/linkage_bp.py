from flask import Blueprint, jsonify, request, current_app
from pydantic import ValidationError as PydanticValidationError
from ..schemas import UpdateLinkageRequest

linkage_bp = Blueprint("linkage", __name__)

@linkage_bp.route("/api/update_linkage", methods=["POST"])
def update_linkage():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "JSON body required"}), 400
        
    try:
        req = UpdateLinkageRequest(**data)
    except PydanticValidationError as e:
        return jsonify({"status": "error", "code": "VALIDATION_ERROR", "message": str(e)}), 422
        
    linkage_service = current_app.config["LINKAGE_SERVICE"]
    res = linkage_service.update_linkage(req)
    return jsonify(res)
