import unittest
import tempfile
from pathlib import Path
import pandas as pd
from tracking_corrector.repositories.mask_repository import atomic_write_text, MaskRepository
from tracking_corrector.errors import RevisionConflict

class TestAtomicWrites(unittest.TestCase):
    def test_atomic_write_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.txt"
            atomic_write_text(file_path, "hello world")
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.read_text(), "hello world")

    def test_mask_repository_revision_conflict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo = MaskRepository(tmp_path)
            
            df = pd.DataFrame({"time_point": [0, 1], "rle_bf": ["5 3", "10 2"]})
            initial_rev = repo.save_cell_masks("2026_04_30_M135", "A14_BF1_F0", 1, df)
            
            # Second write with correct expected revision succeeds
            df["rle_bf"] = ["5 3", "10 5"]
            new_rev = repo.save_cell_masks("2026_04_30_M135", "A14_BF1_F0", 1, df, expected_revision=initial_rev)
            self.assertNotEqual(new_rev, initial_rev)
            
            # Third write with stale revision raises RevisionConflict
            with self.assertRaises(RevisionConflict):
                repo.save_cell_masks("2026_04_30_M135", "A14_BF1_F0", 1, df, expected_revision=initial_rev)

if __name__ == "__main__":
    unittest.main()
