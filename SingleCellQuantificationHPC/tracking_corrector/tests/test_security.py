import unittest
import tempfile
from pathlib import Path
from tracking_corrector.security import resolve_under_root, validate_path_component, safe_subprocess_run

class TestSecurity(unittest.TestCase):
    def test_resolve_under_root_valid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sub = tmp_path / "valid_dir"
            sub.mkdir()
            res = resolve_under_root(tmp_path, "valid_dir")
            self.assertEqual(res, sub.resolve())

    def test_resolve_under_root_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with self.assertRaises(ValueError):
                resolve_under_root(tmp_path, "../outside")

    def test_validate_path_component_invalid(self):
        with self.assertRaises(ValueError):
            validate_path_component("..")
        with self.assertRaises(ValueError):
            validate_path_component("foo/bar")

    def test_safe_subprocess_run(self):
        proc = safe_subprocess_run(["echo", "hello"], check=True, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("hello", proc.stdout)

if __name__ == "__main__":
    unittest.main()
