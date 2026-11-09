import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.scarnet.checkpoint import load_model
from src.scarnet.loss import CombinedLoss, DiceLoss, FocalTverskyLoss
from src.scarnet.trainer import ScarNetTrainer


class TinyModel(nn.Module):
    """Minimal stand-in for an injected ScarNet architecture."""

    def __init__(self, num_classes: int = 5, out_size: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.out_size = out_size
        self.net = nn.Conv2d(1, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)


class DummySlices(Dataset):
    def __init__(self, n: int = 6, size: int = 32):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.rand(1, self.size, self.size),
            "mask": torch.randint(0, 5, (self.size, self.size)),
        }


class ScarNetLossTest(unittest.TestCase):
    def test_components_are_finite_and_backward(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 16, 16, requires_grad=True)
        target = torch.randint(0, 5, (2, 16, 16))
        loss = CombinedLoss()(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_gt_class_does_not_nan(self):
        logits = torch.randn(1, 5, 8, 8)
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        loss = CombinedLoss()(logits, target)
        self.assertTrue(torch.isfinite(loss))

    def test_focal_and_dice_losses_include_only_present_classes(self):
        logits = torch.zeros(1, 5, 4, 4)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[:, 0, 0] = 3
        self.assertTrue(torch.isfinite(FocalTverskyLoss()(logits, target)))
        self.assertTrue(torch.isfinite(DiceLoss()(logits, target)))


class ScarNetTrainerTest(unittest.TestCase):
    def test_one_epoch_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ScarNetTrainer(
                fold=0,
                epochs=1,
                batch_size=2,
                device="cpu",
                save_dir=tmp,
                pred_dir=None,
                train_ds=DummySlices(4),
                val_ds=DummySlices(4),
                model=TinyModel(5, 32),
                model_class="tests.test_scarnet:TinyModel",
                model_params={"num_classes": 5, "out_size": 32},
                out_size=32,
                amp=False,
                num_workers=0,
                select_metric="mean",
                augment=False,
                write_metadata=False,
            )
            best = trainer.fit()
            self.assertTrue(best.is_file())
            self.assertTrue((Path(tmp) / "fold_0" / "history.json").is_file())

            model, config = load_model(best, device="cpu")
            self.assertEqual(type(model).__name__, "TinyModel")
            self.assertEqual(
                config["model"]["class"], "tests.test_scarnet:TinyModel")
            self.assertEqual(config["model"]["params"]["num_classes"], 5)



class ScarNetMergeTest(unittest.TestCase):
    def test_merge_slice_predictions_pastes_at_the_crop_box(self):
        import numpy as np

        from src.cascade.geometry import CropBox
        from src.scarnet.merge import merge_slice_predictions

        box = CropBox(x0=2, x1=6, y0=3, y1=9)  # height 6, width 4
        full_shape = (10, 12, 3)
        preds = {1: np.full((4, 4), 3, dtype=np.uint8)}
        merged = merge_slice_predictions(preds, box, full_shape)

        self.assertEqual(merged.shape, full_shape)
        self.assertTrue((merged[2:6, 3:9, 1] == 3).all())
        self.assertEqual(int(merged[:, :, 0].sum()), 0)
        self.assertEqual(int(merged[:, :, 2].sum()), 0)
        self.assertEqual(int(merged[0, 0, 1]), 0)

    def test_resize_label_preserves_labels(self):
        import numpy as np

        from src.scarnet.merge import resize_label

        out = resize_label(np.array([[0, 3], [3, 0]], dtype=np.uint8), (4, 4))
        self.assertEqual(set(np.unique(out).tolist()), {0, 3})


if __name__ == "__main__":
    unittest.main()
