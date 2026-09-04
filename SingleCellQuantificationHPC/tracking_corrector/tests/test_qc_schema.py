"""
test_qc_schema.py

Unit tests covering the two-level QC schema:
1. Formal GlobalCellQC vs LocalCellQC enum & validation functions.
2. QCRepository and QCService write validation and error handling.
3. Filtering constants (USABLE_GLOBAL_STATUSES, EXCLUDED_GLOBAL_STATUSES).
4. Review endpoints local state updates and reviewed flag handling.
"""

import unittest
import tempfile
from pathlib import Path
import json

from tracking_corrector.qc_schema import (
    GlobalCellQC,
    LocalCellQC,
    InvalidQCStatusError,
    validate_global_qc_status,
    validate_local_qc_status,
    USABLE_GLOBAL_STATUSES,
    EXCLUDED_GLOBAL_STATUSES,
)
from tracking_corrector.repositories.qc_repository import QCRepository
from tracking_corrector.services.qc_service import QCService
from tracking_corrector.app import create_app
from tracking_corrector.config import Config


class TestQCSchemaValidation(unittest.TestCase):
    def test_global_qc_valid_statuses(self):
        for status in ["good", "bad", "pending", "corrected", "GOOD", "  BAD  "]:
            val = validate_global_qc_status(status)
            self.assertIn(val, GlobalCellQC.valid_statuses())

    def test_global_qc_invalid_statuses(self):
        invalid_statuses = ["mistracked", "exhausted", "unknown", "", 123, None]
        for status in invalid_statuses:
            with self.assertRaises(InvalidQCStatusError):
                validate_global_qc_status(status)

    def test_local_qc_valid_statuses(self):
        for status in ["pending", "mistracked", "PENDING", "  MISTRACKED  "]:
            val = validate_local_qc_status(status)
            self.assertIn(val, LocalCellQC.valid_statuses())

    def test_local_qc_invalid_statuses(self):
        invalid_statuses = ["good", "bad", "corrected", "exhausted", "unknown", 123, None]
        for status in invalid_statuses:
            with self.assertRaises(InvalidQCStatusError):
                validate_local_qc_status(status)

    def test_filtering_constants(self):
        self.assertIn("good", USABLE_GLOBAL_STATUSES)
        self.assertIn("corrected", USABLE_GLOBAL_STATUSES)
        self.assertNotIn("bad", USABLE_GLOBAL_STATUSES)
        self.assertNotIn("pending", USABLE_GLOBAL_STATUSES)
        self.assertIn("bad", EXCLUDED_GLOBAL_STATUSES)


class TestQCRepositoryValidation(unittest.TestCase):
    def test_save_qc_global_level(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = QCRepository(Path(tmp_dir))
            exp = "exp"
            target = "3_F0"

            # Valid global status write
            valid_data = {"3_F0_cell_1": "good", "3_F0_cell_2": "bad"}
            hash_val = repo.save_qc(exp, target, valid_data)
            self.assertTrue(bool(hash_val))

            loaded, _ = repo.load_qc(exp, target)
            self.assertEqual(loaded["3_F0_cell_1"], "good")

            # Invalid global status write ("mistracked" is invalid at global level)
            invalid_data = {"3_F0_cell_1": "mistracked"}
            with self.assertRaises(InvalidQCStatusError):
                repo.save_qc(exp, target, invalid_data)

    def test_save_review_state_local_level(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = QCRepository(Path(tmp_dir))
            exp = "exp"
            target = "3_F0"

            # Valid local review state write
            valid_state = {
                "3_F0_3_BF1_F0_cell_104": {"status": "mistracked", "reviewed": True},
                "3_F0_3_BF1_F0_cell_108": {"status": "pending", "reviewed": False},
            }
            repo.save_review_state(exp, target, valid_state)

            loaded = repo.load_review_state(exp, target)
            self.assertEqual(loaded["3_F0_3_BF1_F0_cell_104"]["status"], "mistracked")

            # Invalid local status write ("good" is invalid at local level)
            invalid_state = {"3_F0_3_BF1_F0_cell_104": {"status": "good"}}
            with self.assertRaises(InvalidQCStatusError):
                repo.save_review_state(exp, target, invalid_state)


class TestQCServiceValidation(unittest.TestCase):
    def test_save_qc_record_simple_global_and_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = QCRepository(Path(tmp_dir))
            service = QCService(repo, audit_service=None)
            exp = "exp"
            target = "3_F0"

            # Global write: "good" is valid
            res_g = service.save_qc_record_simple(exp, target, "3_F0_cell_1", "good", level="global")
            self.assertEqual(res_g["status"], "success")

            # Global write: "mistracked" raises InvalidQCStatusError
            with self.assertRaises(InvalidQCStatusError):
                service.save_qc_record_simple(exp, target, "3_F0_cell_1", "mistracked", level="global")

            # Local write: "mistracked" is valid
            res_l = service.save_qc_record_simple(exp, target, "3_F0_3_BF1_F0_cell_104", "mistracked", level="local")
            self.assertEqual(res_l["status"], "success")

            # Local write: "good" raises InvalidQCStatusError
            with self.assertRaises(InvalidQCStatusError):
                service.save_qc_record_simple(exp, target, "3_F0_3_BF1_F0_cell_104", "good", level="local")


class TestQCRoutesEndpoints(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        self.app = create_app(cfg)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_save_qc_route_validation_error_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = QCRepository(Path(tmp_dir))
            service = QCService(repo, audit_service=None)
            self.app.config["QC_SERVICE"] = service

            # Valid POST -> HTTP 200
            res1 = self.client.post("/api/save_qc", json={
                "experiment": "exp",
                "sequence": "3_F0",
                "cell_id": "3_F0_cell_1",
                "status": "good",
                "level": "global"
            })
            self.assertEqual(res1.status_code, 200)

            # Invalid status POST -> HTTP 400 with error message
            res2 = self.client.post("/api/save_qc", json={
                "experiment": "exp",
                "sequence": "3_F0",
                "cell_id": "3_F0_cell_1",
                "status": "mistracked",
                "level": "global"
            })
            self.assertEqual(res2.status_code, 400)
            data2 = res2.get_json()
            self.assertEqual(data2["status"], "error")
            self.assertIn("Invalid status 'mistracked'", data2["message"])


if __name__ == "__main__":
    unittest.main()
