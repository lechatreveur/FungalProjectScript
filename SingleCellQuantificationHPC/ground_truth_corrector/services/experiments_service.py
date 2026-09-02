import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..config import Config, ExperimentConfig
from ..security import resolve_under_root
from .linkage_service import LinkageService
from ..repositories.linkage_repository import LinkageRepository

class ExperimentsService:
    def __init__(self, config: Config, linkage_repo: LinkageRepository):
        self.config = config
        self.linkage_repo = linkage_repo
        self.linkage_service = LinkageService(linkage_repo)

    def list_experiments(self) -> List[Dict[str, Any]]:
        configured_exps = self.config.get_all_experiments()
        base_root = self.config.local_movie_root

        result = []
        if base_root.exists():
            for item in sorted(base_root.iterdir()):
                if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("_"):
                    exp_id = item.name
                    cfg = self.config.get_experiment_config(exp_id)
                    
                    has_ims = any(f.suffix == ".ims" for f in item.iterdir() if f.is_file())
                    has_subdirs = any(d.is_dir() for d in item.iterdir() if not d.name.startswith("."))
                    
                    if has_ims or has_subdirs or (cfg and cfg.enabled):
                        channels = cfg.channels if cfg else ["bf"]
                        display_name = cfg.display_name if cfg else exp_id
                        result.append({
                            "id": exp_id,
                            "display_name": display_name,
                            "channels": channels,
                            "training_subfolder": cfg.training_subfolder if cfg else "train"
                        })
        return result

    def list_films_and_sequences(self, exp: str) -> Dict[str, Any]:
        seq_res = self.linkage_service.get_sequences(exp)
        sequences = seq_res.get("sequences", {})
        
        exp_dir = resolve_under_root(self.config.local_movie_root, exp)
        films = []
        if exp_dir.exists():
            for item in sorted(exp_dir.iterdir()):
                if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("_"):
                    film_name = item.name
                    frames = item / f"Frames_{film_name}"
                    tracked = item / f"TrackedCells_{film_name}"
                    if frames.is_dir() or tracked.is_dir():
                        films.append(film_name)

        return {
            "sequences": sorted(list(sequences.keys())),
            "films": films
        }
