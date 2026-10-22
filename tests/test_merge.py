"""Merge modes: erase (historical) vs additive with a confidence threshold."""

import unittest

import numpy as np

from src.polar.merge import merge_polar_with_nnunet


def _probs(H=3, W=3, Z=1):
    return np.zeros((5, H, W, Z), dtype=np.float32)


class MergeTests(unittest.TestCase):
    def test_erase_and_additive_differ_for_stage1_mvo(self):
        pred = np.full((3, 3, 1), 2, dtype=np.uint8)   # myocardium
        pred[1, 1, 0] = 4                              # stage-1 MVO
        probs = _probs()
        probs[2, 1, 1, 0] = 1.0                        # polar says myo
        covered = np.array([True])

        erased = merge_polar_with_nnunet(pred, probs, covered, dilation=0,
                                        mode="erase")
        added = merge_polar_with_nnunet(pred, probs, covered, dilation=0,
                                        mode="additive")
        self.assertEqual(erased[1, 1, 0], 2)
        self.assertEqual(added[1, 1, 0], 4)

    def test_probability_threshold_gates_paint(self):
        pred = np.full((3, 3, 1), 2, dtype=np.uint8)   # stage-1 myo
        probs = _probs()
        probs[3, 0, 0, 0] = 0.6
        probs[2, 0, 0, 0] = 0.4
        covered = np.array([True])

        painted = merge_polar_with_nnunet(pred, probs, covered, dilation=0,
                                          mode="additive", prob_thresh=0.5)
        blocked = merge_polar_with_nnunet(pred, probs, covered, dilation=0,
                                          mode="additive", prob_thresh=0.7)
        self.assertEqual(painted[0, 0, 0], 3)
        self.assertEqual(blocked[0, 0, 0], 2)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            merge_polar_with_nnunet(np.zeros((2, 2, 1), np.uint8),
                                    _probs(2, 2), np.array([True]), mode="x")


if __name__ == "__main__":
    unittest.main()
