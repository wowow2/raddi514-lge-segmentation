import unittest

import numpy as np

from src.cascade import (
    CropBox,
    crop_array,
    foreground_bbox,
    largest_component,
    shifted_affine,
)


class CascadeGeometryTest(unittest.TestCase):
    def test_cropbox_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            CropBox(5, 5, 0, 10)
        with self.assertRaises(ValueError):
            CropBox(-1, 10, 0, 10)
        with self.assertRaises(ValueError):
            CropBox(0, 10, 5, 5)

    def test_largest_component_ignores_distant_false_positive(self):
        mask = np.zeros((80, 80, 5), dtype=bool)
        mask[30:45, 30:45, :] = True
        mask[2:6, 70:74, 0] = True
        out = largest_component(mask)
        xs, ys, _ = np.nonzero(out)
        self.assertGreaterEqual(xs.min(), 30)
        self.assertLessEqual(xs.max(), 44)
        self.assertGreaterEqual(ys.min(), 30)
        self.assertLessEqual(ys.max(), 44)
        self.assertFalse(out[2:6, 70:74, 0].any())

    def test_bbox_is_square_padded_and_inside_image(self):
        mask = np.zeros((100, 120, 6), dtype=bool)
        mask[20:50, 70:100, :] = True
        box = foreground_bbox(mask, pad=16, min_size=64, multiple=16)
        self.assertEqual(box.width, box.height)
        self.assertEqual(box.width % 16, 0)
        self.assertGreaterEqual(box.width, 64)
        self.assertGreaterEqual(box.x0, 0)
        self.assertGreaterEqual(box.y0, 0)
        self.assertLessEqual(box.x1, 100)
        self.assertLessEqual(box.y1, 120)
        cropped = crop_array(mask, box)
        self.assertGreater(cropped.sum(), 0)

    def test_bbox_handles_empty_mask(self):
        box = foreground_bbox(np.zeros((50, 60, 4), dtype=bool))
        self.assertEqual(box.width, box.height)
        self.assertLessEqual(box.x1, 50)
        self.assertLessEqual(box.y1, 60)

    def test_crop_array_3d_and_4d(self):
        arr = np.arange(6 * 7 * 2).reshape(6, 7, 2)
        box = CropBox(1, 5, 2, 6)
        self.assertEqual(crop_array(arr, box).shape, (4, 4, 2))
        volumes = np.stack([arr, arr])
        self.assertEqual(crop_array(volumes, box).shape, (2, 4, 4, 2))

    def test_shifted_affine_moves_origin(self):
        affine = np.eye(4)
        affine[0, 0] = 2.0
        affine[1, 1] = 3.0
        box = CropBox(4, 10, 5, 11)
        out = shifted_affine(affine, box)
        np.testing.assert_allclose(out[:3, 3], [8.0, 15.0, 0.0])
        np.testing.assert_allclose(out[:3, :3], affine[:3, :3])


if __name__ == "__main__":
    unittest.main()
