from pathlib import Path
from typing import Dict, Any
from ..security import safe_subprocess_run, resolve_under_root
from ..errors import ValidationError

class QuantificationService:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root

    def trigger_quantification(
        self,
        exp: str,
        film: str,
        cell_id: int,
        target_hpc: bool = True
    ) -> Dict[str, Any]:
        exp_dir = resolve_under_root(self.base_root, exp)
        script_path = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/run_ui_ordered_pipeline.sh")
        
        if not script_path.exists():
            raise ValidationError(f"Quantification script not found at '{script_path}'")

        cmd = [str(script_path), str(exp_dir), film, str(cell_id)]
        
        # Execute securely using argument list (shell=False)
        proc = safe_subprocess_run(cmd, check=False)
        
        return {
            "status": "submitted" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "command": " ".join(cmd),
            "experiment": exp,
            "film": film,
            "cell_id": cell_id
        }
