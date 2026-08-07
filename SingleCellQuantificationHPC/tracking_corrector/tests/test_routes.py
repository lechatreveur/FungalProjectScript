import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock
from tifffile import imwrite
from tracking_corrector.app import create_app
from tracking_corrector.config import Config
from tracking_corrector.schemas import validate_and_decode_rle

class TestRoutes(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        self.app = create_app(cfg)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)

    def test_list_experiments_endpoint(self):
        res = self.client.get("/api/list_experiments")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("experiments", data)
        self.assertIsInstance(data["experiments"], list)

    def test_autofix_endpoint_returns_json_validation_error(self):
        res = self.client.post("/api/autofix_masks", json={})
        self.assertEqual(res.status_code, 400)
        self.assertTrue(res.is_json)
        self.assertEqual(res.get_json()["status"], "error")

    def test_predict_septum_passes_full_sequence_identity_without_saving(self):
        service = Mock()
        service.predict_septum.return_value = {
            "status": "success",
            "review_only": True,
        }
        self.app.config["SEPTUM_SERVICE"] = service
        res = self.client.post(
            "/api/predict_septum",
            json={
                "experiment": "exp",
                "film": "film_2",
                "cell_id": 0,
                "sequence": "sequence_a",
                "global_cell_id": "global_7",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["review_only"])
        service.predict_septum.assert_called_once_with(
            "exp",
            "film_2",
            0,
            sequence="sequence_a",
            global_cell_id="global_7",
        )

    def test_autofix_endpoint_snaps_and_saves_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exp = "experiment"
            film = "film"
            tracked_dir = root / exp / film / f"TrackedCells_{film}"
            masks_dir = root / exp / film / f"Masks_{film}"
            tracked_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)

            csv_path = tracked_dir / "cell_1_masks.csv"
            pd.DataFrame({
                "time_point": [0],
                "width": [4],
                "height": [4],
                "rle_bf": ["6 1"],
                "source_bf": ["ai"],
                "area_bf": [1],
            }).to_csv(csv_path, index=False)

            segmentation = np.zeros((4, 4), dtype=np.uint16)
            segmentation[1:3, 1:3] = 7
            imwrite(masks_dir / f"{film}_t_000_c_0_seg.tif", segmentation)

            cfg = Config()
            cfg.data = {"movie_roots": {"local": str(root), "nas": str(root)}}
            client = create_app(cfg).test_client()
            res = client.post("/api/autofix_masks", json={
                "experiment": exp,
                "film": film,
                "cell_id": 1,
                "channel": "bf",
                "start_t": 0,
                "end_t": 0,
            })

            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json()["fixed_count"], 1)
            saved = pd.read_csv(csv_path)
            fixed = validate_and_decode_rle(saved.loc[0, "rle_bf"], 4, 4)
            self.assertEqual(int(fixed.sum()), 4)
            self.assertEqual(saved.loc[0, "source_bf"], "manual")

if __name__ == "__main__":
    unittest.main()
