import sys
from pathlib import Path
from typing import Optional
from flask import Flask

# Insert parent directory (SingleCellQuantificationHPC) onto sys.path so tracking_corrector is importable
package_parent = str(Path(__file__).resolve().parent.parent)
if package_parent not in sys.path:
    sys.path.insert(0, package_parent)

from tracking_corrector.config import Config, config as default_config
from tracking_corrector.errors import TrackingCorrectorError, handle_app_error
from tracking_corrector.repositories.mask_repository import MaskRepository
from tracking_corrector.repositories.linkage_repository import LinkageRepository
from tracking_corrector.repositories.qc_repository import QCRepository
from tracking_corrector.services.audit_service import AuditService
from tracking_corrector.services.experiments_service import ExperimentsService
from tracking_corrector.services.frames_service import FramesService
from tracking_corrector.services.linkage_service import LinkageService
from tracking_corrector.services.cache_service import CacheService
from tracking_corrector.services.septum_service import SeptumService
from tracking_corrector.services.tracking_service import TrackingService

from tracking_corrector.routes.experiments_bp import experiments_bp
from tracking_corrector.routes.frames_bp import frames_bp
from tracking_corrector.routes.septum_bp import septum_bp
from septum_alignment_board.routes.board_bp import board_bp

def create_app(cfg: Optional[Config] = None) -> Flask:
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static")
    )
    
    app_cfg = cfg or default_config
    app.config["SECRET_KEY"] = app_cfg.data.get("server", {}).get("secret_key", "dev-septum-board-secret-key")
    app.config["APP_CONFIG"] = app_cfg
    
    # Instantiating Repositories & Services from tracking_corrector directly
    mask_repo = MaskRepository(app_cfg.local_movie_root)
    linkage_repo = LinkageRepository(app_cfg.local_movie_root)
    qc_repo = QCRepository(app_cfg.local_movie_root)
    
    audit_service = AuditService(app_cfg.local_movie_root)
    exp_service = ExperimentsService(app_cfg)
    frames_service = FramesService(app_cfg, mask_repo)
    linkage_service = LinkageService(linkage_repo, audit_service)
    septum_service = SeptumService(app_cfg.local_movie_root, qc_repo, audit_service)
    cache_service = CacheService(app_cfg.local_movie_root)
    tracking_service = TrackingService(app_cfg.local_movie_root)
    
    # Inject into Flask app config for blueprint routes to consume
    app.config["EXP_SERVICE"] = exp_service
    app.config["FRAMES_SERVICE"] = frames_service
    app.config["LINKAGE_SERVICE"] = linkage_service
    app.config["SEPTUM_SERVICE"] = septum_service
    app.config["CACHE_SERVICE"] = cache_service
    app.config["TRACKING_SERVICE"] = tracking_service
    
    # Register error handler
    app.register_error_handler(TrackingCorrectorError, handle_app_error)
    
    # Register blueprints
    app.register_blueprint(experiments_bp)
    app.register_blueprint(frames_bp)
    app.register_blueprint(septum_bp)
    app.register_blueprint(board_bp)
    
    return app
