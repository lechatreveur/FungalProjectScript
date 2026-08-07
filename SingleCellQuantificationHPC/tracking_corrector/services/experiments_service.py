import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from ..config import Config, ExperimentConfig
from ..errors import NotFoundError
from ..security import resolve_under_root, validate_path_component

class ExperimentsService:
    def __init__(self, config: Config):
        self.config = config

    def list_experiments(self) -> List[Dict[str, Any]]:
        registered = self.config.get_experiments()
        base_root = self.config.local_movie_root
        
        discovered_dirs = set()
        if base_root.exists():
            for item in base_root.iterdir():
                if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("_"):
                    # Check if it contains any film subfolder with TrackedCells
                    for sub in item.iterdir():
                        if sub.is_dir() and not sub.name.startswith("."):
                            tracked_dir = sub / f"TrackedCells_{sub.name}"
                            if tracked_dir.is_dir():
                                discovered_dirs.add(item.name)
                                break

        result = []
        all_exp_ids = sorted(list(set(registered.keys()) | discovered_dirs))
        
        for exp_id in all_exp_ids:
            cfg = registered.get(exp_id, ExperimentConfig(id=exp_id, display_name=exp_id))
            if not cfg.enabled:
                continue
                
            exp_dir = base_root / exp_id
            discovered = exp_dir.is_dir()
            readable = discovered and os.access(exp_dir, os.R_OK)
            writable = discovered and os.access(exp_dir, os.W_OK)
            
            result.append({
                "id": exp_id,
                "display_name": cfg.display_name or exp_id,
                "discovered": discovered,
                "registered": exp_id in registered,
                "readable": readable,
                "writable": writable,
                "editing_enabled": writable,
                "channels": cfg.channels
            })
        return result

    def get_experiment_cfg(self, exp_id: str) -> ExperimentConfig:
        validate_path_component(exp_id, "experiment")
        registered = self.config.get_experiments()
        if exp_id in registered:
            return registered[exp_id]
            
        exp_dir = self.config.local_movie_root / exp_id
        if exp_dir.is_dir():
            return ExperimentConfig(id=exp_id, display_name=exp_id)
            
        raise NotFoundError(f"Experiment '{exp_id}' not found.")

    def discover_films(self, exp_id: str) -> List[str]:
        cfg = self.get_experiment_cfg(exp_id)
        exp_dir = resolve_under_root(self.config.local_movie_root, exp_id)
        if not exp_dir.exists():
            return []
            
        films = []
        for item in sorted(exp_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                tracked_dir = item / cfg.tracked_dir_template.format(film=item.name)
                if tracked_dir.is_dir():
                    films.append(item.name)
        return films
