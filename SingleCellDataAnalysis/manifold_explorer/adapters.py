import os
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
from .config import ExperimentConfig
from .schemas import validate_id_map_schema

class ExperimentAdapter(ABC):
    @abstractmethod
    def load_metadata(self, config: ExperimentConfig) -> pd.DataFrame:
        """
        Loads and returns a normalized metadata DataFrame with columns:
        ['local_cell_id', 'original_cell_id', 'global_cell_id', 'field', 'source', 'film']
        """
        pass

    @abstractmethod
    def load_qc(self, config: ExperimentConfig) -> dict[str, str]:
        """
        Loads and returns a QC map: global_cell_id -> status ('good', 'corrected', 'bad', etc.)
        """
        pass


class GenericAdapter(ExperimentAdapter):
    def load_metadata(self, config: ExperimentConfig) -> pd.DataFrame:
        if not config.id_map_csv or not os.path.exists(config.id_map_csv):
            raise FileNotFoundError(f"Missing required ID map CSV for generic experiment {config.name}")
            
        df = pd.read_csv(config.id_map_csv)
        validate_id_map_schema(df, str(config.id_map_csv), config.name)
        
        # Standard ID map has columns: new_cell_id, orig_str_id, field, source
        # Let's normalize it to our canonical schema
        records = []
        for _, row in df.iterrows():
            cid = int(row['new_cell_id'])
            field = str(row['field'])
            source = str(row['source'])
            
            # Map source/field/experiment configuration to film folder name
            source_cfg = config.sources.get(source)
            if source_cfg:
                # E.g. A14_FL1_{field}
                film = source_cfg.film_name.replace("{field}", field)
            else:
                # Default film folder fallback
                if source == 'GFP1': film = f"A14_FL1_{field}"
                elif source == 'GFP2': film = f"A14_FL2_{field}"
                else: film = f"A14_FL3_{field}"
                
            orig_str = str(row.get('orig_str_id', ''))
            if ':' in orig_str:
                orig_id = int(orig_str.split(':')[1])
            else:
                orig_id = int(row.get('local_fl_id', cid))
                
            gcid = str(row.get('global_cell_id', f"A14_{field}_cell_{orig_id}"))
            
            records.append({
                'local_cell_id': cid,
                'original_cell_id': orig_id,
                'global_cell_id': gcid,
                'field': field,
                'source': source,
                'film': film
            })
            
        return pd.DataFrame(records)

    def load_qc(self, config: ExperimentConfig) -> dict[str, str]:
        qc = {}
        for path in config.qc_jsons:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    qc.update(json.load(f))
        return qc


class Sept17Adapter(ExperimentAdapter):
    def load_metadata(self, config: ExperimentConfig) -> pd.DataFrame:
        # Sept17 reads metadata directly from stacked CSV because it has fields
        # like 'orig_gfp_id', 'field', 'global_cell_id' in each row!
        df_stacked = pd.read_csv(config.stacked_csv)
        df_unique = df_stacked.drop_duplicates('cell_id')
        
        records = []
        for _, row in df_unique.iterrows():
            cid = int(row['cell_id'])
            field = str(row['field'])
            source = str(row['source'])
            orig_id = int(row['orig_gfp_id'])
            gcid = str(row['global_cell_id'])
            
            # Sept17 timing film names: A14_1TP1_F1, A14_1TP2_F1
            tp = int(row['tp'])
            film = f"A14_1TP1_{field}" if tp == 1 else f"A14_1TP2_{field}"
            
            records.append({
                'local_cell_id': cid,
                'original_cell_id': orig_id,
                'global_cell_id': gcid,
                'field': field,
                'source': source,
                'film': film
            })
            
        return pd.DataFrame(records)

    def load_qc(self, config: ExperimentConfig) -> dict[str, str]:
        qc = {}
        for path in config.qc_jsons:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    qc.update(json.load(f))
        return qc


def get_adapter(adapter_type: str) -> ExperimentAdapter:
    if adapter_type == "sept17":
        return Sept17Adapter()
    return GenericAdapter()
