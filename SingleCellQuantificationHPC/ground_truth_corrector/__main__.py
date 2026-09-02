#!/usr/bin/env python3
import sys
import argparse
import logging
from pathlib import Path

from .config import config
from .app import create_app

def main():
    parser = argparse.ArgumentParser(description="Start Ground-Truth Cell Tracking Tool (3-Keyframe Curator & Cellpose Exporter)")
    parser.add_argument("--host", type=str, default=config.host, help=f"Host to listen on (default: {config.host})")
    parser.add_argument("--port", type=int, default=config.port, help=f"Port to listen on (default: {config.port})")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("ground_truth_corrector")
    
    app = create_app(config)
    
    logger.info(f"🎯 Starting Ground-Truth Cell Tracking Tool at http://{args.host}:{args.port}")
    logger.info(f"📁 Movie root: {config.local_movie_root}")
    logger.info(f"📦 Cellpose training root: {config.cellpose_training_root}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
