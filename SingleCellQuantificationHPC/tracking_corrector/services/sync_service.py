from pathlib import Path
from typing import List, Dict, Any
from ..security import safe_subprocess_run, resolve_under_root
from ..errors import ValidationError

class SyncService:
    def __init__(self, local_root: Path, nas_root: Path):
        self.local_root = local_root
        self.nas_root = nas_root

    def pull_experiment(self, exp: str) -> Dict[str, Any]:
        src_dir = resolve_under_root(self.nas_root, exp)
        dst_dir = resolve_under_root(self.local_root, exp)
        
        if not src_dir.exists():
            raise ValidationError(f"NAS directory '{src_dir}' does not exist.")
            
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "rsync", "-avz", "--update",
            "--exclude=__pycache__", "--exclude=*.ims",
            f"{src_dir}/", f"{dst_dir}/"
        ]
        
        proc = safe_subprocess_run(cmd, check=True)
        return {
            "status": "success",
            "direction": "pull",
            "src": str(src_dir),
            "dst": str(dst_dir)
        }

    def push_experiment(self, exp: str) -> Dict[str, Any]:
        src_dir = resolve_under_root(self.local_root, exp)
        dst_dir = resolve_under_root(self.nas_root, exp)
        
        if not src_dir.exists():
            raise ValidationError(f"Local directory '{src_dir}' does not exist.")
            
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "rsync", "-avz", "--update",
            "--exclude=__pycache__", "--exclude=*.ims",
            f"{src_dir}/", f"{dst_dir}/"
        ]
        
        proc = safe_subprocess_run(cmd, check=True)
        return {
            "status": "success",
            "direction": "push",
            "src": str(src_dir),
            "dst": str(dst_dir)
        }
