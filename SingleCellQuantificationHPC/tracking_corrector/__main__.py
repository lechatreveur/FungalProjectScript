import argparse
import sys
import logging
from pathlib import Path
from .config import config
from .app import create_app
from .services.sync_service import SyncService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fungal Cell Tracking Corrector Server & Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run Flask web application server")
    serve_parser.add_argument("--host", type=str, default=config.host, help="Bind host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=config.port, help="Bind port number (default: 5001)")
    serve_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Synchronize local SSD data with NAS")
    sync_parser.add_argument("action", choices=["pull", "push", "status"], help="Sync action")
    sync_parser.add_argument("--experiment", type=str, required=True, help="Target experiment name")
    
    args = parser.parse_args()
    
    if args.command == "serve" or args.command is None:
        host = getattr(args, "host", config.host)
        port = getattr(args, "port", config.port)
        debug = getattr(args, "debug", False)
        
        if host == "0.0.0.0":
            logger.warning("⚠️  WARNING: Binding to 0.0.0.0 exposes the server to external network access without authentication!")
            
        app = create_app()
        logger.info("🚀 Starting Tracking Corrector at http://%s:%s", host, port)
        app.run(host=host, port=port, debug=debug)
        
    elif args.command == "sync":
        sync_service = SyncService(config.local_movie_root, config.nas_movie_root)
        if args.action == "pull":
            logger.info("Pulling data from NAS for experiment '%s'...", args.experiment)
            res = sync_service.pull_experiment(args.experiment)
            logger.info("✅ Pull completed: %s", res)
        elif args.action == "push":
            logger.info("Pushing data to NAS for experiment '%s'...", args.experiment)
            res = sync_service.push_experiment(args.experiment)
            logger.info("✅ Push completed: %s", res)

if __name__ == "__main__":
    main()
