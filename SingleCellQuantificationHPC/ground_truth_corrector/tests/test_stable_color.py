import unittest

from ground_truth_corrector.services.gt_frames_service import (
    fnv1a_32,
    id_to_color,
    stable_color_key,
)


class TestStableColor(unittest.TestCase):
    def test_id_to_color_deterministic_and_bounded(self):
        for cid in (0, 1, 7, 42, 313, 99999):
            c1 = id_to_color(cid)
            c2 = id_to_color(cid)
            self.assertEqual(c1, c2)
            self.assertEqual(len(c1), 3)
            self.assertTrue(all(0 <= ch <= 255 for ch in c1))

    def test_same_identity_same_color(self):
        # int identity (single-film tracked local id)
        self.assertEqual(id_to_color(stable_color_key(12)), id_to_color(stable_color_key(12)))
        # str identity (global_cell_id) is stable across calls
        gid = "3_F0_cell_79"
        self.assertEqual(
            id_to_color(stable_color_key(gid)),
            id_to_color(stable_color_key(gid)),
        )

    def test_global_id_color_invariant_to_resolution_path(self):
        # A linked cell resolved from any film/keyframe must land on one colour:
        # the colour is a pure function of the global_cell_id string.
        gid = "5_1_N1_F1_cell_4"
        colors = {id_to_color(stable_color_key(gid)) for _ in range(5)}
        self.assertEqual(len(colors), 1)

    def test_distinct_identities_usually_differ(self):
        colors = {id_to_color(stable_color_key(f"seq_F0_cell_{n}")) for n in range(40)}
        # Not a guarantee of zero collisions, but a flat palette would fail this.
        self.assertGreater(len(colors), 30)

    def test_fnv1a_32_matches_reference_vectors(self):
        # Reference values for the JS port (static/js/color.js) to match.
        self.assertEqual(fnv1a_32(""), 0x811C9DC5)
        self.assertEqual(fnv1a_32("a"), 0xE40C292C)
        self.assertEqual(fnv1a_32("foobar"), 0xBF9CF968)

    def test_int_key_is_identity(self):
        self.assertEqual(stable_color_key(5), 5)
        self.assertEqual(stable_color_key(0), 0)


if __name__ == "__main__":
    unittest.main()
