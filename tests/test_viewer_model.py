"""Headless tests for the Qt viewer's data/rendering layer (no Qt imports)."""

import unittest

import numpy as np

from src.viewer.model import (
    bootstrap_ci,
    contour_mask,
    derived_channel,
    error_rgba,
    label_rgba,
    montage_cols,
    probability_rgba,
)


COLORS = {
    1: (0.2, 0.4, 1.0, 0.8),
    2: (0.1, 0.9, 0.3, 0.6),
}


class TestOverlays(unittest.TestCase):
    def test_label_rgba_respects_visibility(self):
        labels = np.zeros((3, 3), dtype=np.uint8)
        labels[0, 0] = 1
        labels[1, 1] = 2
        rgba = label_rgba(labels, {1}, COLORS, alpha=0.5)
        self.assertEqual(rgba.shape, (3, 3, 4))
        self.assertAlmostEqual(float(rgba[0, 0, 3]), 0.4, places=6)
        self.assertEqual(float(rgba[1, 1, 3]), 0.0)

    def test_error_rgba_marks_tp_fp_fn(self):
        gt = np.zeros((1, 3), dtype=np.uint8)
        pred = np.zeros((1, 3), dtype=np.uint8)
        gt[0, 0] = 1          # true positive
        pred[0, 0] = 1
        pred[0, 1] = 1        # false positive
        gt[0, 2] = 1          # false negative
        rgba = error_rgba(gt, pred, {1}, COLORS, alpha=1.0)
        self.assertAlmostEqual(float(rgba[0, 0, 3]), 0.8 * 0.45, places=6)
        self.assertAlmostEqual(float(rgba[0, 1, 3]), 0.75, places=6)
        self.assertAlmostEqual(float(rgba[0, 2, 3]), 0.80, places=6)
        self.assertGreater(float(rgba[0, 1, 0]), 0.9)  # orange

    def test_contour_mask_is_a_border(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        edge = contour_mask(mask)
        self.assertFalse(edge[3, 3])
        self.assertTrue(edge[2, 2])
        self.assertEqual(int(edge.sum()), 8)
        self.assertFalse(contour_mask(np.zeros((2, 2), dtype=bool)).any())

    def test_probability_rgba_lifts_low_values(self):
        prob = np.array([[0.0, 0.01, 0.5, 1.0]], dtype=np.float32)
        rgba = probability_rgba(prob, (1.0, 0.0, 0.0), alpha=1.0, gamma=0.5)
        self.assertEqual(float(rgba[0, 0, 3]), 0.0)
        self.assertGreater(float(rgba[0, 1, 3]), 0.0)
        self.assertLess(float(rgba[0, 2, 3]), float(rgba[0, 3, 3]))
        self.assertAlmostEqual(float(rgba[0, 3, 3]), 1.0, places=6)


class TestChannels(unittest.TestCase):
    def test_all_channels_are_normalized_and_finite(self):
        rng = np.random.default_rng(0)
        image = rng.normal(2000, 200, size=(16, 18, 4)).astype(np.float32)
        image[8:12, 8:12, :] += 800
        for kind in ("lge", "window", "blur", "dog", "log", "std"):
            out = derived_channel(image, kind)
            self.assertEqual(out.shape, image.shape, kind)
            self.assertTrue(np.isfinite(out).all(), kind)
            self.assertGreaterEqual(float(out.min()), 0.0, kind)
            self.assertLessEqual(float(out.max()), 1.0, kind)

    def test_unknown_channel_raises(self):
        with self.assertRaises(KeyError):
            derived_channel(np.zeros((2, 2, 2)), "nope")


class TestHelpers(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic_and_brackets_mean(self):
        values = [0.1, 0.5, 0.6, 0.7, 0.9]
        first = bootstrap_ci(values, n_boot=200)
        second = bootstrap_ci(values, n_boot=200)
        self.assertEqual(first, second)
        mean, lo, hi = first
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)

    def test_bootstrap_ci_empty(self):
        mean, lo, hi = bootstrap_ci([])
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(lo))
        self.assertTrue(np.isnan(hi))

    def test_montage_columns_are_sensible(self):
        self.assertEqual(montage_cols(4), 2)
        self.assertEqual(montage_cols(7), 4)
        self.assertEqual(montage_cols(9), 4)
        self.assertEqual(montage_cols(12), 5)


if __name__ == "__main__":
    unittest.main()
