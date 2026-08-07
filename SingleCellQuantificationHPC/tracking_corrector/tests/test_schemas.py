import unittest
import numpy as np
from tracking_corrector.schemas import validate_and_decode_rle

class TestSchemas(unittest.TestCase):
    def test_validate_and_decode_rle_valid(self):
        mask = validate_and_decode_rle("5 3", 10, 10)
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask.sum(), 3)
        flat = mask.ravel(order='F')
        self.assertEqual(flat[4], 1)
        self.assertEqual(flat[5], 1)
        self.assertEqual(flat[6], 1)

    def test_validate_and_decode_rle_empty(self):
        mask = validate_and_decode_rle("", 10, 10)
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask.sum(), 0)

    def test_validate_and_decode_rle_out_of_bounds(self):
        with self.assertRaises(ValueError):
            validate_and_decode_rle("95 10", 10, 10)

    def test_validate_and_decode_rle_odd_length(self):
        with self.assertRaises(ValueError):
            validate_and_decode_rle("5 3 10", 10, 10)

if __name__ == "__main__":
    unittest.main()
