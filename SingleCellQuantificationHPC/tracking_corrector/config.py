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
        return os.environ.get("TRACKING_CORRECTOR_HOST", self.data.get("server", {}).get("host", "127.0.0.1"))

    @property
    def port(self) -> int:
        return int(os.environ.get("TRACKING_CORRECTOR_PORT", self.data.get("server", {}).get("port", 5001)))

    @property
    def local_movie_root(self) -> Path:
        p = os.environ.get("LOCAL_MOVIE_ROOT", self.data.get("movie_roots", {}).get("local", "/Volumes/X10 Pro/Movies"))
        return Path(p)

    @property
    def nas_movie_root(self) -> Path:
        p = os.environ.get("NAS_MOVIE_ROOT", self.data.get("movie_roots", {}).get("nas", "/Volumes/Movies"))
        return Path(p)

    @property
    def cache_root(self) -> Path:
        """
        Where CellCrops_*/PopulationFrames_* render caches are written.

        Deliberately NOT nested inside local_movie_root's per-experiment/film
        folders by default: those caches are pure, fully-regenerable derived
        images (nothing else reads them). By default this points at a single
        consolidated folder at the X10 Pro drive's root, sibling to Movies/ --
        kept on X10 Pro per user preference (not the Mac's internal disk), but
        out of the Movies/ tree so it can't accidentally get swept up by any
        rsync-to-NAS backup step. Measured block size on this drive is large
        (512KB-1MB per file via `stat -f`, well above typical exFAT defaults),
        so the real risk here is FILE COUNT, not raw byte size -- see
        cache_max_files. Override via TRACKING_CORRECTOR_CACHE_ROOT or
        config.yaml's server.cache_root.
        """
        default = str(self.local_movie_root.parent / "_tracking_corrector_cache")
        p = os.environ.get(
            "TRACKING_CORRECTOR_CACHE_ROOT",
            self.data.get("server", {}).get("cache_root", default),
        )
        path = Path(p)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_max_bytes(self) -> int:
        """Soft cap for the combined render cache LOGICAL size, in bytes. 0 disables enforcement.
        Secondary to cache_max_files on this drive -- see cache_root docstring."""
        default_gb = 2.0
        v = os.environ.get(
            "TRACKING_CORRECTOR_CACHE_MAX_GB",
            self.data.get("server", {}).get("cache_max_gb", default_gb),
        )
        return int(float(v) * 1e9)

    @property
    def cache_max_files(self) -> int:
        """
        Soft cap on the TOTAL NUMBER of cached files, regardless of their
        logical size. This is the cap that actually matters on X10 Pro: its
        measured filesystem block size is 512KB-1MB per file, so e.g. 20,000
        small cache files could occupy 10-20GB of real disk space even though
        their logical content is only ~50MB. 0 disables enforcement.
        """
        default_n = 20_000
        v = os.environ.get(
            "TRACKING_CORRECTOR_CACHE_MAX_FILES",
            self.data.get("server", {}).get("cache_max_files", default_n),
        )
        return int(v)

    def get_experiments(self) -> Dict[str, ExperimentConfig]:
        res = {}
        raw_exp = self.data.get("experiments", {})
        for exp_key, exp_data in raw_exp.items():
            if not isinstance(exp_data, dict):
                continue
            exp_id = str(exp_key)
            cfg = ExperimentConfig(
                id=exp_id,
                enabled=exp_data.get("enabled", True),
                display_name=str(exp_data.get("display_name", exp_id)),
                linkage_file=exp_data.get("linkage_file", "sequence_linkage.json"),
                tracked_dir_template=exp_data.get("tracked_dir_template", "TrackedCells_{film}"),
                frames_dir_template=exp_data.get("frames_dir_template", "Frames_{film}"),
                masks_dir_template=exp_data.get("masks_dir_template", "Masks_{film}"),
                channels=exp_data.get("channels", ["bf"])
            )
            res[exp_id] = cfg
        return res

config = Config()
