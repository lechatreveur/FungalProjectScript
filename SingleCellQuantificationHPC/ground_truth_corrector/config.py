import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    id: str
    enabled: bool = True
    display_name: str = ""
    linkage_file: str = "sequence_linkage.json"
    tracked_dir_template: str = "TrackedCells_{film}"
    frames_dir_template: str = "Frames_{film}"
    masks_dir_template: str = "Masks_{film}"
    channels: List[str] = field(default_factory=lambda: ["bf"])
    training_subfolder: str = "train"


class Config:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
            
        self.config_path = config_path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f) or {}
        else:
            self.data = {}

    @property
    def host(self) -> str:
        return os.environ.get("GT_TRACKING_HOST", self.data.get("server", {}).get("host", "0.0.0.0"))

    @property
    def port(self) -> int:
        return int(os.environ.get("GT_TRACKING_PORT", self.data.get("server", {}).get("port", 5002)))

    @property
    def local_movie_root(self) -> Path:
        p = os.environ.get("LOCAL_MOVIE_ROOT", self.data.get("movie_roots", {}).get("local", "/Volumes/X10 Pro/Movies"))
        return Path(p)

    @property
    def nas_movie_root(self) -> Path:
        p = os.environ.get("NAS_MOVIE_ROOT", self.data.get("movie_roots", {}).get("nas", "/Volumes/Movies"))
        return Path(p)

    @property
    def cellpose_training_root(self) -> Path:
        p = os.environ.get(
            "CELLPOSE_TRAINING_ROOT",
            self.data.get("movie_roots", {}).get("cellpose_training", "/Volumes/X10 Pro/Movies/cellpose_training_data")
        )
        return Path(p)

    @property
    def cache_root(self) -> Path:
        p = os.environ.get(
            "GT_CORRECTOR_CACHE_ROOT",
            str(self.local_movie_root.parent / "_gt_tracking_cache")
            if self.local_movie_root.parent.exists()
            else str(self.local_movie_root / "_gt_tracking_cache")
        )
        return Path(p)

    @property
    def cache_max_gb(self) -> float:
        val = self.data.get("server", {}).get("cache_max_gb", 2.0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return 2.0

    @property
    def cache_max_files(self) -> int:
        val = self.data.get("server", {}).get("cache_max_files", 20000)
        try:
            return int(val)
        except (ValueError, TypeError):
            return 20000

    def get_experiment_config(self, exp_id: str) -> Optional[ExperimentConfig]:
        exps = self.data.get("experiments", [])
        for e in exps:
            if isinstance(e, dict) and e.get("id") == exp_id:
                return ExperimentConfig(
                    id=e["id"],
                    enabled=e.get("enabled", True),
                    display_name=e.get("display_name", e["id"]),
                    linkage_file=e.get("linkage_file", "sequence_linkage.json"),
                    channels=e.get("channels", ["bf"]),
                    training_subfolder=e.get("training_subfolder", "train")
                )
        return None

    def get_all_experiments(self) -> List[ExperimentConfig]:
        exps = self.data.get("experiments", [])
        res = []
        for e in exps:
            if isinstance(e, dict):
                res.append(ExperimentConfig(
                    id=e["id"],
                    enabled=e.get("enabled", True),
                    display_name=e.get("display_name", e["id"]),
                    linkage_file=e.get("linkage_file", "sequence_linkage.json"),
                    channels=e.get("channels", ["bf"]),
                    training_subfolder=e.get("training_subfolder", "train")
                ))
        return res


config = Config()
