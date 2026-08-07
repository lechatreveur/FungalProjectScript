import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from .config import load_config, SourceConfig, ExperimentConfig
from .schemas import validate_stacked_schema, validate_trajectory
from .qc import apply_qc_rules, make_qc_report
from .adapters import GenericAdapter, Sept17Adapter

class TestManifoldExplorer(unittest.TestCase):

    def test_trajectory_validation(self):
        # Valid trajectory
        df_valid = pd.DataFrame({
            "pol1_int_corr": np.random.normal(5, 1, 101),
            "pol2_int_corr": np.random.normal(3, 1, 101)
        })
        err = validate_trajectory(df_valid, "test_cell", 101)
        self.assertIsNone(err)

        # Invalid length
        df_short = pd.DataFrame({
            "pol1_int_corr": np.random.normal(5, 1, 50),
            "pol2_int_corr": np.random.normal(3, 1, 50)
        })
        err = validate_trajectory(df_short, "test_cell", 101)
        self.assertIn("length", err)

        # Zero variance
        df_zero = pd.DataFrame({
            "pol1_int_corr": np.ones(101) * 2.0,
            "pol2_int_corr": np.ones(101) * 1.5
        })
        err = validate_trajectory(df_zero, "test_cell", 101)
        self.assertIn("variance", err)

    def test_qc_reporting(self):
        cells = pd.DataFrame([
            {"observation_id": "c1", "trajectory_p1": [1]*101, "included": True, "exclusion_reason": None, "qc_status": "good", "time_to_division": -10.0},
            {"observation_id": "c2", "trajectory_p1": None, "included": False, "exclusion_reason": "Missing trajectory data", "qc_status": "good", "time_to_division": None},
            {"observation_id": "c3", "trajectory_p1": [1]*101, "included": False, "exclusion_reason": "Excluded by manual QC curation", "qc_status": "bad", "time_to_division": -5.0}
        ]).set_index("observation_id")

        report = make_qc_report(cells)
        self.assertEqual(report["loaded_cells"], 3)
        self.assertEqual(report["included_cells"], 1)
        self.assertEqual(report["excluded_cells"], 2)
        self.assertEqual(report["missing_trajectory"], 1)

    def test_generic_adapter_normalization(self):
        # Mocking generic adapter mapping load
        adapter = GenericAdapter()
        config = ExperimentConfig(
            name="test_exp",
            root=Path("/mock/root"),
            stacked_csv=Path("/mock/stacked.csv"),
            id_map_csv=Path("/mock/id_map.csv"),
            sources={
                "GFP1": SourceConfig(film_name="A14_TP1_{field}", midpoint_min=10, time_res_min=0.2, start_time_min=0.0)
            }
        )

        id_map_df = pd.DataFrame([
            {"new_cell_id": 1, "field": "F0", "source": "GFP1", "orig_str_id": "c:12"}
        ])

        with patch("pandas.read_csv", return_value=id_map_df):
            with patch("os.path.exists", return_value=True):
                meta = adapter.load_metadata(config)
                self.assertEqual(len(meta), 1)
                self.assertEqual(meta.iloc[0]["film"], "A14_TP1_F0")
                self.assertEqual(meta.iloc[0]["original_cell_id"], 12)
                self.assertEqual(meta.iloc[0]["global_cell_id"], "A14_F0_cell_12")

if __name__ == "__main__":
    unittest.main()
