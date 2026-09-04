from pathlib import Path
from typing import Optional
from flask import Flask, render_template, jsonify
from .config import Config, config as default_config
from .errors import TrackingCorrectorError, handle_app_error
from .repositories.mask_repository import MaskRepository
from .repositories.linkage_repository import LinkageRepository
from .repositories.qc_repository import QCRepository
from .services.audit_service import AuditService
from .services.experiments_service import ExperimentsService
from .services.frames_service import FramesService
from .services.masks_service import MasksService
from .services.linkage_service import LinkageService
from .services.qc_service import QCService
from .services.septum_service import SeptumService
from .services.suspicious_service import SuspiciousService
from .services.quantification_service import QuantificationService
from .services.cache_service import CacheService
from .services.sync_service import SyncService
from .services.jobs_service import JobsService
from .services.tracking_service import TrackingService

from .routes.experiments_bp import experiments_bp
from .routes.frames_bp import frames_bp
from .routes.masks_bp import masks_bp
from .routes.linkage_bp import linkage_bp
from .routes.qc_bp import qc_bp
from .routes.septum_bp import septum_bp
from .routes.jobs_bp import jobs_bp
from .routes.health_bp import health_bp
from .routes.mistrack_review_bp import mistrack_review_bp

def create_app(cfg: Optional[Config] = None) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    
    app_cfg = cfg or default_config
    app.config["SECRET_KEY"] = app_cfg.data.get("server", {}).get("secret_key", "dev-secret-key")
    app.config["APP_CONFIG"] = app_cfg
    
    # Repositories
    mask_repo = MaskRepository(app_cfg.local_movie_root)
    linkage_repo = LinkageRepository(app_cfg.local_movie_root)
    qc_repo = QCRepository(app_cfg.local_movie_root)
    
    # Services
    audit_service = AuditService(app_cfg.local_movie_root)
    exp_service = ExperimentsService(app_cfg)
    frames_service = FramesService(app_cfg, mask_repo)
    masks_service = MasksService(mask_repo, audit_service)
    linkage_service = LinkageService(linkage_repo, audit_service)
    qc_service = QCService(qc_repo, audit_service)
    from .services.active_learning_service import ActiveLearningService
    al_service = ActiveLearningService(app_cfg.local_movie_root)
    septum_service = SeptumService(app_cfg.local_movie_root, qc_repo, audit_service, al_service=al_service)
    suspicious_service = SuspiciousService(mask_repo)

    quant_service = QuantificationService(app_cfg.local_movie_root)
    cache_service = CacheService(
        app_cfg.cache_root,
        max_bytes=app_cfg.cache_max_bytes,
        max_files=app_cfg.cache_max_files,
    )
    sync_service = SyncService(app_cfg.local_movie_root, app_cfg.nas_movie_root)
    jobs_service = JobsService()
    tracking_service = TrackingService(app_cfg.local_movie_root)
    
    # DI Injection
    app.config["EXP_SERVICE"] = exp_service
    app.config["FRAMES_SERVICE"] = frames_service
    app.config["MASKS_SERVICE"] = masks_service
    app.config["LINKAGE_SERVICE"] = linkage_service
    app.config["QC_SERVICE"] = qc_service
    app.config["SEPTUM_SERVICE"] = septum_service
    app.config["ACTIVE_LEARNING_SERVICE"] = al_service
    app.config["SUSPICIOUS_SERVICE"] = suspicious_service

    app.config["QUANTIFICATION_SERVICE"] = quant_service
    app.config["CACHE_SERVICE"] = cache_service
    app.config["SYNC_SERVICE"] = sync_service
    app.config["JOBS_SERVICE"] = jobs_service
    app.config["TRACKING_SERVICE"] = tracking_service
    
    # Defensive render-cache cap: prune oldest CellCrops_*/PopulationFrames_*
    # files on startup if the combined cache exceeds cache_max_gb. Cheap,
    # one-shot check (not a background watcher) -- see CacheService.enforce_cache_limit.
    try:
        cache_service.enforce_cache_limit()
    except Exception:
        app.logger.exception("Render cache eviction check failed at startup (non-fatal)")

    # Error Handler
    app.register_error_handler(TrackingCorrectorError, handle_app_error)
    
    # Blueprints
    app.register_blueprint(experiments_bp)
    app.register_blueprint(frames_bp)
    app.register_blueprint(masks_bp)
    app.register_blueprint(linkage_bp)
    app.register_blueprint(qc_bp)
    app.register_blueprint(septum_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(mistrack_review_bp)

    from septum_alignment_board.routes.board_bp import board_bp
    app.register_blueprint(board_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/septum_board")
    def septum_board():
        return render_template("septum_board.html")

    return app
