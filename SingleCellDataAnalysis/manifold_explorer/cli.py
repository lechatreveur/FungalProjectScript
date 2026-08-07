import argparse
import sys
import logging
from .pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Fungal Manifold Explorer Command Line Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the manifold explorer dashboard")
    build_parser.add_argument("config", type=str, help="Path to config.yaml file")
    build_parser.add_argument("--strict", action="store_true", help="Fail build on warning validation errors")
    build_parser.add_argument("--output-dir", type=str, help="Override output directory")
    build_parser.add_argument("--mode", type=str, choices=["single-html", "static-site"], help="Override output mode")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    if args.command == "build":
        try:
            run_pipeline(args.config, strict=args.strict)
        except Exception as e:
            logging.exception(f"❌ Build failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
