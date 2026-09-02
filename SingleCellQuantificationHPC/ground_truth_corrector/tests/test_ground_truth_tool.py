import os
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import tifffile

from ground_truth_corrector.config import Config
from ground_truth_corrector.app import create_app
from ground_truth_corrector.schemas import validate_and_decode_rle, encode_mask_to_rle
from ground_truth_corrector.services.gt_frames_service import GTFramesService
from ground_truth_corrector.services.gt_export_service import GTExportService

class TestGroundTruthTool(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.movie_root = Path(self.temp_dir) / "Movies"
        self.training_root = Path(self.temp_dir) / "cellpose_training_data"
        self.movie_root.mkdir(parents=True)
        self.training_root.mkdir(parents=True)

        # Create mock experiment and film
        self.exp = "2026_08_28_M160"
        self.film = "5_1_N1_FL1_F0"
        self.film_dir = self.movie_root / self.exp / self.film
        self.frames_dir = self.film_dir / f"Frames_{self.film}"
        self.masks_dir = self.film_dir / f"Masks_{self.film}"
        self.tracked_dir = self.film_dir / f"TrackedCells_{self.film}"
        
        self.frames_dir.mkdir(parents=True)
        self.masks_dir.mkdir(parents=True)
        self.tracked_dir.mkdir(parents=True)

        # Create mock TIFF frames (t=0..100)
        # Create t=0, t=50, t=100
        for t in [0, 50, 100]:
            img = np.full((100, 100), 500, dtype=np.uint16)
            tifffile.imwrite(str(self.frames_dir / f"{self.film}_t_{t:03d}_c_0.tif"), img)
            
            # Mask seg
            seg = np.zeros((100, 100), dtype=np.uint16)
            seg[10:30, 10:30] = 1
            tifffile.imwrite(str(self.masks_dir / f"{self.film}_t_{t:03d}_c_0_seg.tif"), seg)

        # Create mock cell masks CSV (cell 1)
        m = np.zeros((100, 100), dtype=np.uint8)
        m[10:30, 10:30] = 1
        rle = encode_mask_to_rle(m)
        
        rows = [
            {"time_point": 0, "width": 100, "height": 100, "rle_bf": rle, "rle_gfp": ""},
            {"time_point": 50, "width": 100, "height": 100, "rle_bf": rle, "rle_gfp": ""},
            {"time_point": 100, "width": 100, "height": 100, "rle_bf": rle, "rle_gfp": ""}
        ]
        pd.DataFrame(rows).to_csv(self.tracked_dir / "cell_1_masks.csv", index=False)

        # Config
        self.cfg = Config()
        self.cfg.data = {
            "server": {"port": 5002},
            "movie_roots": {
                "local": str(self.movie_root),
                "cellpose_training": str(self.training_root)
            },
            "experiments": [
                {
                    "id": self.exp,
                    "display_name": "M160",
                    "channels": ["bf", "gfp"],
                    "training_subfolder": "NeonGreenGFP"
                }
            ]
        }
        self.app = create_app(self.cfg)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_keyframe_calculation(self):
        frames_svc = GTFramesService(self.cfg)
        k_times = frames_svc.get_film_keyframes(self.exp, self.film)
        self.assertEqual(k_times, [0, 50, 100])

    def test_rle_encode_decode(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:50] = 1
        rle = encode_mask_to_rle(mask)
        decoded = validate_and_decode_rle(rle, 100, 100)
        np.testing.assert_array_equal(mask, decoded)

    def test_export_service(self):
        frames_svc = GTFramesService(self.cfg)
        export_svc = GTExportService(self.cfg, frames_svc)
        
        res = export_svc.sync_keyframe_to_training(self.exp, self.film, 0)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["cell_count"], 1)
        
        # Verify file written to training directory
        dest_mask = self.training_root / "NeonGreenGFP" / f"{self.film}_t000_masks.tif"
        self.assertTrue(dest_mask.exists())
        
        mask_read = tifffile.imread(str(dest_mask))
        self.assertEqual(mask_read.shape, (100, 100))
        self.assertEqual(mask_read[15, 15], 1)

    def test_routes(self):
        # 1. Index page
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Ground-Truth Tracking Tool", res.data)

        # 2. List experiments
        res_exp = self.client.get("/api/list_experiments")
        self.assertEqual(res_exp.status_code, 200)
        data_exp = res_exp.get_json()
        self.assertTrue(any(e["id"] == self.exp for e in data_exp["experiments"]))

        # 3. List cells
        res_cells = self.client.get(f"/api/list_cells?experiment={self.exp}&film={self.film}")
        self.assertEqual(res_cells.status_code, 200)
        data_cells = res_cells.get_json()
        self.assertEqual(len(data_cells["cells"]), 1)
        self.assertEqual(data_cells["cells"][0]["id"], 1)

        # 4. Cell masks (3 keyframes)
        res_masks = self.client.get(f"/api/cell_masks?experiment={self.exp}&film={self.film}&cell_id=1")
        self.assertEqual(res_masks.status_code, 200)
        data_masks = res_masks.get_json()
        self.assertEqual(data_masks["num_frames"], 3)
        self.assertEqual(len(data_masks["masks"]), 3)
        self.assertEqual(len(data_masks["keyframes"]), 3)

        # 5. Frame image
        res_img = self.client.get(f"/api/frame_image?experiment={self.exp}&film={self.film}&t=0&channel=bf")
        self.assertEqual(res_img.status_code, 200)
        self.assertEqual(res_img.mimetype, "image/jpeg")

        # 6. Population frame (1000x1000)
        res_pop = self.client.get(f"/api/population_frame?experiment={self.exp}&film={self.film}&t=0")
        self.assertEqual(res_pop.status_code, 200)
        self.assertEqual(res_pop.mimetype, "image/jpeg")
        pop_im = Image.open(io.BytesIO(res_pop.data))
        self.assertEqual(pop_im.size, (1000, 1000))

        # 7. Save mask & verify training live sync
        new_m = np.zeros((100, 100), dtype=np.uint8)
        new_m[40:60, 40:60] = 1
        new_rle = encode_mask_to_rle(new_m)
        
        save_payload = {
            "experiment": self.exp,
            "film": self.film,
            "cell_id": "1",
            "channel": "bf",
            "changes": [{"time_point": 0, "new_rle": new_rle}]
        }
        res_save = self.client.post("/api/save_mask", json=save_payload)
        self.assertEqual(res_save.status_code, 200)
        
        # Verify training mask updated
        dest_mask = self.training_root / "NeonGreenGFP" / f"{self.film}_t000_masks.tif"
        self.assertTrue(dest_mask.exists())
        mask_read = tifffile.imread(str(dest_mask))
        self.assertEqual(mask_read[45, 45], 1)
        self.assertEqual(mask_read[15, 15], 0) # old position cleared

        # 8. Export all keyframes
        res_export = self.client.post("/api/export_training_data", json={"experiment": self.exp})
        self.assertEqual(res_export.status_code, 200)
        data_exp_all = res_export.get_json()
        self.assertEqual(data_exp_all["status"], "success")
        self.assertEqual(data_exp_all["total_keyframes_exported"], 3)


if __name__ == "__main__":
    unittest.main()
