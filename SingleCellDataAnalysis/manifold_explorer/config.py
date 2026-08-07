import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class SourceConfig:
    film_name: str
    midpoint_min: float
    time_res_min: float
    start_time_min: float
    bf_indices: list[int] = field(default_factory=list)
    modality: str = "GFP"

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    root: Path
    stacked_csv: Path
    id_map_csv: Path | None = None
    linkage_json: Path | None = None
    qc_jsons: list[Path] = field(default_factory=list)
    cycle_length_min: float = 300.0
    reference: bool = False
    expected_frames: int | None = 101
    adapter_type: str = "generic"
    sources: dict[str, SourceConfig] = field(default_factory=dict)

@dataclass(frozen=True)
class GlobalConfig:
    output_dir: Path
    output_mode: str  # "single-html" or "static-site"
    output_filename: str
    model_path: Path
    reference_experiment: str
    umap_seed: int = 42
    experiments: dict[str, ExperimentConfig] = field(default_factory=dict)

def load_config(config_path: str) -> GlobalConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    out_data = data.get("output", {})
    output_dir = Path(out_data.get("directory", ""))
    output_mode = out_data.get("mode", "single-html")
    output_filename = out_data.get("filename", "explorer.html")

    model_data = data.get("model", {})
    model_path = Path(model_data.get("path", ""))
    ref_exp = model_data.get("reference_experiment", "Sept17")
    umap_seed = model_data.get("umap_seed", 42)

    exps = {}
    exp_data = data.get("experiments", {})
    for name, item in exp_data.items():
        root = Path(item.get("root", ""))
        stacked = Path(item.get("stacked_csv", ""))
        
        id_map_val = item.get("id_map_csv")
        id_map_csv = Path(id_map_val) if id_map_val else None
        
        linkage_val = item.get("linkage_json")
        linkage_json = Path(linkage_val) if linkage_val else None
        
        qc_jsons = [Path(p) for p in item.get("qc_jsons", [])]
        cycle_len = float(item.get("cycle_length_min", 300.0))
        ref = bool(item.get("reference", False))
        expected_frames = item.get("expected_frames", 101)
        adapter = item.get("adapter_type", "generic")
        
        sources = {}
        source_data = item.get("sources", {})
        for s_name, s_item in source_data.items():
            sources[s_name] = SourceConfig(
                film_name=s_item.get("film_name", ""),
                midpoint_min=float(s_item.get("midpoint_min", 0.0)),
                time_res_min=float(s_item.get("time_res_min", 0.2)),
                start_time_min=float(s_item.get("start_time_min", 0.0)),
                bf_indices=list(s_item.get("bf_indices", [])),
                modality=s_item.get("modality", "GFP")
            )
            
        exps[name] = ExperimentConfig(
            name=name,
            root=root,
            stacked_csv=stacked,
            id_map_csv=id_map_csv,
            linkage_json=linkage_json,
            qc_jsons=qc_jsons,
            cycle_length_min=cycle_len,
            reference=ref,
            expected_frames=expected_frames,
            adapter_type=adapter,
            sources=sources
        )

    return GlobalConfig(
        output_dir=output_dir,
        output_mode=output_mode,
        output_filename=output_filename,
        model_path=model_path,
        reference_experiment=ref_exp,
        umap_seed=umap_seed,
        experiments=exps
    )
