import os
from pathlib import Path
from flask import Flask, render_template

from .config import Config, config as default_config
from .repositories.mask_repository import MaskRepository
from .repositories.qc_repository import QCRepository
from .repositories.linkage_repository import LinkageRepository
from .services.gt_frames_service import GTFramesService
from .services.gt_export_service import GTExportService
from .services.tracking_service import TrackingService
from .services.linkage_service import LinkageService
from .services.experiments_service import ExperimentsService

from .routes.api_bp import api_bp
from .routes.frames_bp import frames_bp
from .routes.masks_bp import masks_bp

def create_app(cfg: Config = None) -> Flask:
    if cfg is None:
        cfg = default_config

    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )

    # Initialize Repositories
    mask_repo = MaskRepository(cfg.local_movie_root)
    qc_repo = QCRepository(cfg.local_movie_root)
    linkage_repo = LinkageRepository(cfg.local_movie_root)

    # Initialize Services
    frames_service = GTFramesService(cfg)
    export_service = GTExportService(cfg, frames_service)
    linkage_service = LinkageService(linkage_repo)
    tracking_service = TrackingService(cfg, linkage_repo, mask_repo, qc_repo)
    experiments_service = ExperimentsService(cfg, linkage_repo)

    # Attach to Flask extensions
    app.extensions["config"] = cfg
    app.extensions["mask_repository"] = mask_repo
    app.extensions["qc_repository"] = qc_repo
    app.extensions["linkage_repository"] = linkage_repo
    app.extensions["gt_frames_service"] = frames_service
    app.extensions["gt_export_service"] = export_service
    app.extensions["linkage_service"] = linkage_service
    app.extensions["tracking_service"] = tracking_service
    app.extensions["experiments_service"] = experiments_service

    # Register Blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(frames_bp)
    app.register_blueprint(masks_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app
