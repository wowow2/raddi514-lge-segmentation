"""Exercise rendered SLURM workflows without a cluster or ML dependencies."""

import os
import json
import shlex
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class FirCheckpointTests(unittest.TestCase):
    def _cluster_env(self, root):
        binary = root / "bin"
        binary.mkdir()
        for name, body in {
            "module": "exit 0\n", "squeue": "exit 0\n", "lfs": "exit 0\n",
            "sbatch": 'cp "${@: -1}" "$TEST_ROOT/job.sh"\n',
            "python": 'exec "$TEST_PYTHON" "$@"\n',
            "nnUNetv2_predict": 'printf "%s\\n" "$*" >> "$TEST_ROOT/predictions"\n',
        }.items():
            path = binary / name
            path.write_text("#!/bin/bash\n" + body)
            path.chmod(0o755)
        (root / "jobtmp").mkdir()
        return dict(os.environ, TEST_ROOT=str(root), FIR_BASE=str(root / "work"),
                    RADDI_CODE=str(root), FIR_VENV="/dev/null", RADDI_ROOTS_FILE="",
                    SLURM_TMPDIR=str(root / "jobtmp"), TEST_PYTHON=sys.executable,
                    USER="test", PATH=str(binary) + os.pathsep + os.environ["PATH"])

    def test_probability_prediction_stages_only_original_held_out_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self._cluster_env(root)
            native = root / "work/emidec_nnunet"
            task = "Dataset500_EMIDEC"
            images = native / "nnUNet_raw" / task / "imagesTr"
            images.mkdir(parents=True)
            original = REPO / "data/splits/emidec-1c67734e21eda69fd2a1e4de053c3923e72997d554f5d071b0a7ce1a16c7bb8b.json"
            folds = json.loads(original.read_text())
            splits = native / "nnUNet_preprocessed" / task / "splits_final.json"
            splits.parent.mkdir(parents=True)
            splits.write_bytes(original.read_bytes())
            for fold in folds:
                for case in fold["val"]:
                    for channel in (0, 1):
                        (images / f"{case}_{channel:04}.nii.gz").touch()
            command = ["bash", str(REPO / "fir/predict.sh"), "500", "--save-probabilities",
                       "--tr=nnUNetTrainer_probabilisticOversampling"]
            subprocess.run(command, env=env, check=True, capture_output=True)
            job = root / "job.sh"
            self.assertIn("#SBATCH --time=02:00:00", job.read_text())
            subprocess.run(["bash", str(job)], env=env, check=True, capture_output=True)
            calls = (root / "predictions").read_text().splitlines()
            self.assertEqual(len(calls), 5)
            for call in calls:
                args = shlex.split(call)
                fold = int(args[args.index("-f") + 1])
                inputs = Path(args[args.index("-i") + 1])
                expected = {f"{case}_{c:04}.nii.gz" for case in folds[fold]["val"] for c in (0, 1)}
                self.assertEqual({p.name for p in inputs.iterdir()}, expected)
                self.assertIn("--save_probabilities", args)
                self.assertNotIn("--continue_prediction", args)
                self.assertIn("-chk checkpoint_best.pth", call)
                self.assertIn("/probs_nnUNetResEncUNetMPlans_2d_nnUNetTrainer_probabilisticOversampling/", call)
            # A changed fold order must fail before invoking inference.
            (root / "predictions").unlink()
            splits.write_text(json.dumps(list(reversed(folds))))
            result = subprocess.run(["bash", str(job)], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "predictions").exists())

    def test_status_detects_custom_trainers_and_refuses_incomplete_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self._cluster_env(root)
            native = root / "work/emidec_nnunet"
            task = "Dataset503_EMIDEC_Cascade"
            (native / "nnUNet_raw" / task).mkdir(parents=True)
            trainer = "nnUNet_PrimusV3S_Trainer"
            plan = "nnUNetPrimusV3SPlans"
            weights = native / "nnUNet_results" / task / f"{trainer}__{plan}__3d_fullres"
            for fold in range(4):
                path = weights / f"fold_{fold}/checkpoint_best.pth"
                path.parent.mkdir(parents=True)
                path.touch()
            command = ["bash", str(REPO / "fir/status.sh"), "503"]
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            self.assertIn(trainer, result.stdout)
            pack = command + [f"--tr={trainer}", f"--plans={plan}", "--config=3d_fullres", "--pack"]
            result = subprocess.run(pack, env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            archive = root / "work" / f"{task}_results.tar.gz"
            self.assertFalse(archive.exists())
            last = weights / "fold_4/checkpoint_best.pth"
            last.parent.mkdir()
            last.touch()
            subprocess.run(pack, env=env, capture_output=True, check=True)
            self.assertTrue(archive.is_file())

    def test_new_cascade_variants_use_distinct_model_and_result_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self._cluster_env(root)
            variants = [("nnUNetTrainer_probabilisticOversampling", "nnUNetResEncUNetMPlans", "503_over_best"),
                        ("nnUNet_PrimusV3S_Trainer", "nnUNetPrimusV3SPlans", "503_primus_best")]
            outputs = []
            for trainer, plan, name in variants:
                subprocess.run(["bash", str(REPO / "fir/evaluate.sh"), "503",
                                f"--tr={trainer}", f"--plans={plan}", f"--run-id={name}"],
                               env=env, capture_output=True, check=True)
                text = (root / "job.sh").read_text()
                self.assertIn(f"{trainer}__{plan}__3d_fullres", text)
                self.assertIn(f'--run-id "{name}_pp"', text)
                self.assertIn(f"{name}_raw.json", text)
                self.assertIn(f"-p {plan} -tr {trainer}", text)
                self.assertNotIn("503_3d", text)
                outputs.append(next(line for line in text.splitlines() if line.startswith("PRED=")))
            self.assertNotEqual(*outputs)

    def test_prediction_uses_same_loop_locally_and_in_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "bin"
            binary.mkdir()
            (root / "work/emidec_nnunet/nnUNet_raw/Dataset501_Regions").mkdir(parents=True)
            for name, body in {
                "module": "exit 0\n",
                "squeue": "exit 0\n",
                "sbatch": 'cp "${@: -1}" "$TEST_ROOT/job.sh"\n',
                "nnUNetv2_predict": 'printf "%s\\n" "$*" >> "$TEST_ROOT/predictions"\n',
            }.items():
                path = binary / name
                path.write_text("#!/bin/bash\n" + body)
                path.chmod(0o755)
            env = dict(os.environ, TEST_ROOT=str(root), FIR_BASE=str(root / "work"),
                       FIR_VENV="/dev/null", RADDI_ROOTS_FILE="", USER="test",
                       PATH=str(binary) + os.pathsep + os.environ["PATH"])
            command = ["bash", str(REPO / "fir/predict.sh"), "501", "--cpu", "--parts=2"]
            subprocess.run(command, env=env, check=True, capture_output=True)
            job = root / "job.sh"
            text = job.read_text()
            self.assertLess(text.index("#SBATCH --cpus-per-task=8"), text.index("set -euo pipefail"))
            self.assertIn("#SBATCH --mem=32G", text)
            subprocess.run(["bash", str(job)], env=env, check=True, capture_output=True)
            scheduled = sorted((root / "predictions").read_text().splitlines())
            self.assertEqual(len(scheduled), 10)
            (root / "predictions").unlink()
            job.unlink()
            subprocess.run(command + ["--local"], env=env, check=True, capture_output=True)
            self.assertFalse(job.exists())  # local mode never calls sbatch
            self.assertEqual(sorted((root / "predictions").read_text().splitlines()), scheduled)
            self.assertTrue(all("-chk checkpoint_best.pth" in call for call in scheduled))

    def test_failed_submission_dry_run_never_submits(self):
        command = '''
module() { :; }
source "$1"
sbatch() { printf '%s\\n' "$*"; return 17; }
squeue() { echo unexpected-queue; }
submit_job /tmp/example-job.sh
'''
        env = dict(os.environ, FIR_VENV="/dev/null", RADDI_ROOTS_FILE="", SLURM_JOB_ID="test")
        result = subprocess.run(["bash", "-c", command, "test", str(REPO / "fir/env.sh")],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(result.stdout.strip(), "--test-only /tmp/example-job.sh")

    def test_training_and_evaluation_select_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fir = root / "fir"
            fir.mkdir()
            binary = root / "bin"
            binary.mkdir()
            (root / "work").mkdir()
            (root / "jobtmp").mkdir()
            shutil.copy2(REPO / "fir/env.sh", fir / "env.sh")
            (root / "work/emidec_nnunet/nnUNet_raw/Dataset503_EMIDEC_Cascade").mkdir(parents=True)
            stubs = {
                "sbatch": 'cp "${@: -1}" "$TEST_ROOT/rendered.sh"\n',
                "squeue": "exit 0\n",
                "module": "exit 0\n",
                "nnUNetv2_predict":
                    'printf "predict %s\\n" "$*" >> "$TEST_ROOT/calls"\n',
                "python":
                    'printf "python %s\\n" "$*" >> "$TEST_ROOT/calls"\n'
                    'if [[ "$1" == "-" ]]; then cat >/dev/null; fi\n',
            }
            for name, body in stubs.items():
                path = binary / name
                path.write_text("#!/bin/bash\n" + body)
                path.chmod(0o755)
            env = dict(os.environ, TEST_ROOT=str(root), RADDI_CODE=str(root), USER="test",
                       SLURM_TMPDIR=str(root / "jobtmp"), FIR_BASE=str(root / "work"),
                       FIR_VENV="/dev/null", RADDI_ROOTS_FILE="",
                       PATH=str(binary) + os.pathsep + os.environ["PATH"])
            for script in ("submit.sh", "evaluate.sh"):
                shutil.copy2(REPO / "fir" / script, fir / script)
                subprocess.run(["bash", str(fir / script), "503"],
                               cwd=root, env=env, check=True, capture_output=True)
                rendered = root / "rendered.sh"
                subprocess.run(["bash", "-n", str(rendered)], check=True)
                if script == "submit.sh":
                    self.assertIn("--val_best", rendered.read_text())
                    continue

                # Stub only external commands; run the actual generated shell.
                trainer = (root / "work/emidec_nnunet/nnUNet_results/Dataset503_EMIDEC_Cascade/"
                           "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")
                for fold in range(5):
                    checkpoint = trainer / f"fold_{fold}/checkpoint_best.pth"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.touch()
                results = root / "work/results"
                results.mkdir(parents=True)
                for name in ("503_best_raw.json", "503_best_pp.json"):
                    (results / name).touch()
                subprocess.run(["bash", str(rendered)], env=env, check=True,
                               capture_output=True)
                calls = (root / "calls").read_text().splitlines()
                predictions = [c for c in calls if c.startswith("predict ")]
                self.assertEqual(len(predictions), 5)
                for fold, call in enumerate(predictions):
                    self.assertIn("-chk checkpoint_best.pth", call)
                    self.assertIn(f"-f {fold} ", call)
                    self.assertIn(f"cascade_best_inputs/fold_{fold}", call)
                    self.assertNotIn("--continue_prediction", call)
                evaluations = [c for c in calls if "scripts/evaluate_nnunet_cv.py" in c]
                self.assertEqual(len(evaluations), 2)
                for call in evaluations:
                    self.assertIn("--checkpoint best", call)
                    self.assertIn("--stage stage2", call)
                    self.assertNotIn("503_3d", call)


if __name__ == "__main__":
    unittest.main()
