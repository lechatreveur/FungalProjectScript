import shutil
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__)

@health_bp.route("/api/health", methods=["GET"])
def health_check():
    cfg = current_app.config["APP_CONFIG"]
    local_root = cfg.local_movie_root
    return jsonify({
        "status": "healthy" if local_root.exists() else "degraded",
        "movie_root_exists": local_root.exists(),
        "host": cfg.host,
        "port": cfg.port
    })

@health_bp.route("/api/health/storage", methods=["GET"])
def health_storage():
    cfg = current_app.config["APP_CONFIG"]
    local_root = cfg.local_movie_root
    
    if not local_root.exists():
        return jsonify({"status": "error", "message": "Local movie root path does not exist"}), 404
        
    usage = shutil.disk_usage(local_root)
    return jsonify({
        "status": "ok",
        "path": str(local_root),
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2)
    })
