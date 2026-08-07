import os
import sys
from pathlib import Path

# Add SingleCellQuantificationHPC directory to python path
package_parent = str(Path(__file__).resolve().parent.parent)
if package_parent not in sys.path:
    sys.path.insert(0, package_parent)

from tracking_corrector.config import Config
from septum_alignment_board.app import create_app

def main():
    cfg = Config()
    host = os.environ.get("SEPTUM_BOARD_HOST", cfg.data.get("server", {}).get("host", "127.0.0.1"))
    port = int(os.environ.get("SEPTUM_BOARD_PORT", "5002"))
    
    app = create_app(cfg)
    print(f"Starting Septum Alignment Board server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
