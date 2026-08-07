import tempfile
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tracking_corrector.errors import ValidationError
from tracking_corrector.schemas import SaveSeptumRequest
from tracking_corrector.services.septum_service import (
    aligned_to_local_frame,
    endpoint_order_is_valid,
    local_to_aligned_frame,
    resolve_local_endpoint,
    validate_local_interval,
)
from SingleCellDataAnalysis.septum_training_utils import export_cell_training_sample
from SingleCellDataAnalysis.septum_lineage_dataset import (
    pair_censored_boundaries,
    parse_acquisition_metadata,
    parse_duration_minutes,
)
from SingleCellDataAnalysis.septum_train_lineage import (
    average_precision,
    choose_physical_window,
    grouped_split,
    match_endpoint_times,
    physical_time_weights,
    temporal_peaks,
)


class TestSeptumCoordinates(unittest.TestCase):
    def test_request_offset_defaults_to_zero(self):
        request = SaveSeptumRequest(
            experiment="exp",
            film="film",
            cell_id="1",
            has_septum=True,
        )
        self.assertEqual(request.offset, 0)

    def test_aligned_local_round_trip(self):
        self.assertEqual(aligned_to_local_frame(27, 7), 20)
        self.assertEqual(local_to_aligned_frame(20, 7), 27)
        self.assertIsNone(aligned_to_local_frame(None, 7))
        self.assertIsNone(local_to_aligned_frame(None, 7))

    def test_explicit_local_coordinate_wins_over_legacy_value(self):
        self.assertEqual(resolve_local_endpoint(12, 99, 4), 12)
        self.assertEqual(resolve_local_endpoint(None, 19, 4), 15)

    def test_interval_validation_rejects_cross_film_coordinate(self):
        with self.assertRaises(ValidationError):
            validate_local_interval(9, 120, 120, "Septum 1")

    def test_interval_validation_rejects_reversed_interval(self):
        with self.assertRaises(ValidationError):
            validate_local_interval(30, 20, 120, "Septum 1")

    def test_review_suggestion_rejects_end_before_start(self):
        self.assertFalse(
            endpoint_order_is_valid(
                {"time_min": 30.0},
                {"time_min": 10.0},
            )
        )
        self.assertTrue(
            endpoint_order_is_valid(
                {"time_min": 10.0},
                {"time_min": 30.0},
            )
        )


class TestSeptumTrainingExport(unittest.TestCase):
    def test_export_preserves_both_intervals(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            strip = np.zeros((8, 8 * 12), dtype=np.uint8)
            npz_path = export_cell_training_sample(
                working_dir=tmp_dir,
                film_name="film",
                cell_id=3,
                strip=strip,
                tp0=0,
                offset=2,
                start_idx=1,
                end_idx=4,
                start_idx_2=7,
                end_idx_2=10,
                label_source="cell",
                start_aligned=3,
                end_aligned=6,
                start_aligned_2=9,
                end_aligned_2=12,
                white_septum=False,
                white_septum_2=True,
            )

            with np.load(npz_path, allow_pickle=True) as sample:
                self.assertEqual(int(sample["start_idx"][0]), 1)
                self.assertEqual(int(sample["end_idx"][0]), 4)
                self.assertEqual(int(sample["has_2"][0]), 1)
                self.assertEqual(int(sample["start_idx_2"][0]), 7)
                self.assertEqual(int(sample["end_idx_2"][0]), 10)

            manifest = pd.read_csv(
                Path(tmp_dir) / "training_dataset" / "manifest.csv"
            )
            self.assertEqual(int(manifest.loc[0, "has_2"]), 1)
            self.assertEqual(int(manifest.loc[0, "start_idx_2"]), 7)
            self.assertEqual(int(manifest.loc[0, "end_idx_2"]), 10)
            self.assertTrue(bool(manifest.loc[0, "white_septum_2"]))


class TestPhysicalTimeLineageIndex(unittest.TestCase):
    def test_metadata_duration_units(self):
        self.assertAlmostEqual(parse_duration_minutes("12 s"), 0.2)
        self.assertAlmostEqual(parse_duration_minutes("2 min"), 2.0)
        self.assertAlmostEqual(parse_duration_minutes("500 ms"), 0.5 / 60.0)

    def test_metadata_parser_uses_active_time_series_interval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "movie_metadata.txt"
            metadata_path.write_text(
                "\n".join(
                    [
                        "DateAndTime=2026-04-30 16:08:09",
                        "RepeatCount=101",
                        "ActualInterval=12 s",
                        "RepeatCount=1",
                        "ActualInterval=472400 µs",
                    ]
                ),
                encoding="utf-8",
            )
            metadata = parse_acquisition_metadata(metadata_path)
            self.assertEqual(metadata.repeat_count, 101)
            self.assertAlmostEqual(metadata.interval_min, 0.2)

    def test_boundary_pairing_joins_adjacent_films(self):
        events = pair_censored_boundaries(
            [
                {"kind": "start", "time_min": 19.8, "film": "gfp"},
                {"kind": "end", "time_min": 28.0, "film": "bf"},
            ]
        )
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["left_censored"])
        self.assertFalse(events[0]["right_censored"])
        self.assertEqual(events[0]["start_source"]["film"], "gfp")
        self.assertEqual(events[0]["end_source"]["film"], "bf")

    def test_unmatched_boundaries_remain_censored(self):
        events = pair_censored_boundaries(
            [
                {"kind": "end", "time_min": 4.0, "film": "bf1"},
                {"kind": "start", "time_min": 10.0, "film": "gfp2"},
            ]
        )
        self.assertEqual(len(events), 2)
        self.assertTrue(events[0]["left_censored"])
        self.assertTrue(events[1]["right_censored"])

    def test_physical_window_never_falls_entirely_in_acquisition_gap(self):
        lineage = {"events": []}
        observed = np.asarray([0.0, 1.0, 100.0, 101.0], dtype=np.float32)
        for seed in range(20):
            start, end = choose_physical_window(
                lineage,
                observed,
                window_minutes=20.0,
                rng=np.random.default_rng(seed),
            )
            self.assertTrue(np.any((observed >= start) & (observed <= end)))

    def test_split_holds_out_complete_experiment_and_sequence_groups(self):
        rows = [
            {
                "lineage_key": f"a:{i}",
                "experiment": "a",
                "sequence": "large",
            }
            for i in range(20)
        ]
        rows += [
            {
                "lineage_key": f"b:{i}",
                "experiment": "b",
                "sequence": "small",
            }
            for i in range(3)
        ]
        rows += [
            {
                "lineage_key": f"test:{i}",
                "experiment": "test",
                "sequence": "held-out",
            }
            for i in range(5)
        ]
        train, validation, test = grouped_split(
            rows, test_experiment="test", val_fraction=0.2, seed=123
        )
        self.assertEqual({row["experiment"] for row in test}, {"test"})
        train_groups = {(row["experiment"], row["sequence"]) for row in train}
        validation_groups = {
            (row["experiment"], row["sequence"]) for row in validation
        }
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertEqual(len(validation), 3)

    def test_endpoint_metrics_support_multiple_events_in_minutes(self):
        times = np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        probabilities = np.asarray([0.1, 0.9, 0.1, 0.1, 0.8, 0.1])
        indices = temporal_peaks(
            probabilities, times, threshold=0.5, min_separation_min=2.0
        )
        self.assertEqual(indices, [1, 4])
        matched, false_positive, false_negative, errors = match_endpoint_times(
            [1.0, 11.0], [1.5, 10.0], tolerance_min=2.0
        )
        self.assertEqual((matched, false_positive, false_negative), (2, 0, 0))
        self.assertEqual(errors, [0.5, 1.0])

    def test_average_precision_rewards_correct_ranking(self):
        targets = np.asarray([0, 1, 0, 1])
        self.assertAlmostEqual(
            average_precision(targets, np.asarray([0.1, 0.9, 0.2, 0.8])),
            1.0,
        )

    def test_physical_time_weights_are_frame_rate_invariant(self):
        fast = physical_time_weights(
            np.asarray([0.0, 0.1, 0.2, 0.3]),
            np.asarray([1, 1, 1, 1]),
            np.asarray([1, 0, 0, 0]),
        )
        slow = physical_time_weights(
            np.asarray([0.0, 2.0, 4.0, 6.0]),
            np.asarray([1, 1, 1, 1]),
            np.asarray([1, 0, 0, 0]),
        )
        np.testing.assert_allclose(fast, slow, atol=1e-6)
        self.assertAlmostEqual(float(fast.sum()), 1.0, places=6)

    def test_lineage_validation_keeps_experiment_test_locked(self):
        rows = [
            {
                "lineage_key": f"dev:{index}",
                "experiment": "development",
                "sequence": "sequence",
            }
            for index in range(30)
        ] + [
            {
                "lineage_key": f"test:{index}",
                "experiment": "test",
                "sequence": "held-out",
            }
            for index in range(5)
        ]
        train, validation, test = grouped_split(
            rows,
            test_experiment="test",
            val_fraction=0.2,
            seed=123,
            validation_mode="lineage",
        )
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertEqual({row["experiment"] for row in test}, {"test"})
        self.assertTrue(
            {row["lineage_key"] for row in train}.isdisjoint(
                row["lineage_key"] for row in validation
            )
        )


class TestSeptumCorrectionLogging(unittest.TestCase):
    def test_logs_correction_when_user_overrides_ai(self):
        from unittest.mock import MagicMock
        from tracking_corrector.services.septum_service import SeptumService
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            qc_repo = MagicMock()
            audit_service = MagicMock()
            
            existing_septum_data = {
                "cell_intervals": {
                    "42": {
                        "has_septum": True,
                        "start_aligned": 10,
                        "end_aligned": 20,
                        "label_source": "ai"
                    }
                },
                "offsets": {"42": 0}
            }
            qc_repo.load_septum.return_value = (existing_septum_data, "old_rev")
            qc_repo.get_septum_json_path.return_value = tmp_path / "global_septum_alignment.json"
            
            service = SeptumService(tmp_path, qc_repo, audit_service)
            
            req = SaveSeptumRequest(
                experiment="test_exp",
                film="test_film",
                cell_id="42",
                has_septum=True,
                start_frame=15,
                end_frame=20,
                start_aligned=15,
                end_aligned=20,
                offset=0
            )
            
            import pandas as pd
            original_read_csv = pd.read_csv
            pd.read_csv = MagicMock(return_value=pd.DataFrame([0]*30))
            
            tracked_dir = tmp_path / "test_exp" / "test_film" / "TrackedCells_test_film"
            tracked_dir.mkdir(parents=True, exist_ok=True)
            (tracked_dir / "cell_42_masks.csv").touch()
            
            try:
                service.save_septum_label(req, user="test_user")
            finally:
                pd.read_csv = original_read_csv
                
            log_file = tmp_path / ".tracking_corrector" / "septum_corrections.jsonl"
            self.assertTrue(log_file.exists())
            
            with open(log_file, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            
            log_entry = json.loads(lines[0])
            self.assertEqual(log_entry["user"], "test_user")
            self.assertEqual(log_entry["cell_id"], "42")
            self.assertEqual(log_entry["original_ai"]["start_aligned"], 10)
            self.assertEqual(log_entry["corrected_human"]["start_aligned"], 15)


if __name__ == "__main__":
    unittest.main()
