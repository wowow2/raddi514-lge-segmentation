"""Resume metadata: optimizer/scheduler/epoch survive a restart."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.polar.trainer import PolarTrainer


class _EmptyDataset:
    """Enough of a dataset for DataLoader construction; never iterated."""

    R = 64
    Theta = 256

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise IndexError(index)


class ResumeTests(unittest.TestCase):
    def test_resume_restores_epoch_score_and_config(self):
        with tempfile.TemporaryDirectory() as td:
            common = dict(fold=0, epochs=5, save_dir=td, num_workers=0,
                          train_ds=_EmptyDataset(), val_ds=_EmptyDataset(),
                          write_metadata=False, seed=0)
            first = PolarTrainer(**common)
            self.assertEqual(first.start_epoch, 0)
            first.best_score = 0.42
            first.history["train_loss"] = [1.0]
            torch.save(first._checkpoint_payload(2), first.last_ckpt_path)

            resumed = PolarTrainer(**common, resume=True)
            self.assertEqual(resumed.start_epoch, 3)
            self.assertAlmostEqual(resumed.best_score, 0.42)
            self.assertEqual(resumed.history["train_loss"], [1.0])

            ckpt = torch.load(str(resumed.last_ckpt_path),
                              map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["epoch"], 2)
            self.assertEqual(ckpt["config"]["model"]["name"], "polar_unet")
            self.assertIn("optimizer", ckpt)
            self.assertIn("scheduler", ckpt)
            self.assertTrue(Path(resumed.best_ckpt_path).parent.exists())


if __name__ == "__main__":
    unittest.main()
