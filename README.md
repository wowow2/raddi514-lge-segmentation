# RADD I 514 — LGE Cardiac MRI Segmentation

Segmentation of late gadolinium enhancement (LGE) cardiac MRI on the EMIDEC dataset
(5 classes: background, LV cavity, myocardium, scar, MVO/no-reflow).

A two-stage cascade: an nnU-Net v2 backbone feeds a custom polar-coordinate U-Net,
followed by anatomical post-processing. Evaluated with 5-fold cross-validation
(Dice and 95% Hausdorff distance).

## Setup

Python 3.12 (virtualenv at `.venv`). Main dependencies: `torch`, `nnunetv2`,
`nibabel`, `surface_distance`, `scikit-learn`, `scipy`, `matplotlib`. Tests use
only the standard library (`unittest`); `pytest` is not required.

Data locations default to `var/` under the repository root and can be
overridden in `.env` (loaded by `src/config.py`):

| variable | default |
|---|---|
| `EMIDEC_GT_ROOT` | `var/emidec_dataset/train` |
| `NNUNET_PRED_DIR` | `var/nnUNet_validation` |
| `NNUNET_SPLITS_PATH` | `<NNUNET_PRED_DIR>/splits_final.json` |
| `NNUNET_CV_OUT` | `var/results/nnunet_2d_cv_metrics.json` |

All path arguments are resolved relative to the repository root, so scripts run
the same from anywhere.

## Layout

- `src/datautils/` — EMIDEC loading, splits/validation (`splits.py`), shared
  prediction lookup (`predictions.py`), metrics, post-processing, nnU-Net export
- `src/runs.py` — run manifests, discovery, filtering, completeness checks
- `src/results.py` — tidy per-case/per-class results and run comparison
- `src/analysis/probs.py` — out-of-fold probability diagnostics
- `src/polar/` — polar transform, model, training, checkpoint helpers, prediction, merge
- `src/viewer/` — interactive case browser
- `fir/` — SLURM scripts for training and prediction on the Fir cluster
- `scripts/` — entry points and the `runs.py` query CLI
- `tests/` — stdlib `unittest` coverage (no pytest required)

See [`NOTES.md`](NOTES.md) for dataset details and results, and
[`READINGS.md`](READINGS.md) for references.

## Experiment metadata

Training, prediction, post-processing, and evaluation write a `run.json`
manifest next to their output. Manifests for the same `run_id` are merged, so
each step contributes what it knows:

- `model` — architecture name + constructor parameters, input channels, label scheme
- `data` — dataset/gt root, splits path and content hash, fold/case counts
- `training` — seed, epochs, batch size, lr, weight decay, loss, augmentation
- `inference` — checkpoint choice, device, merge mode/dilation/tau
- `postprocess` — ordered operations with parameters
- `parent_run` / `source_run` — producing training run and stage-1 run
- `paths` — predictions, probabilities, checkpoints, metrics
- `status` — expected outputs and per-fold prediction coverage
- `provenance` — git commit/dirty flag, command, package versions, host, time

Unknown values are stored as `null`, never guessed. Directories created before
this change are still usable: `RunStore` discovers prediction trees and
`var/results/*.json` read-only, marks them `provenance.legacy`, and treats
missing fields as unknown. Attach explicit metadata when you know it:

```bash
python scripts/runs.py register 501_regions_pp \
  --predictions var/nnUNet_validation_501_pp \
  --metrics var/results/nnunet_501_cv_metrics_pp.json \
  --model nnunet --stage stage1 --tags regions,baseline \
  --description "ResEnc-M, region labels, scar/MVO gates"
```

Older metrics JSON (`per_case` with nested `dice`/`hd95`) remain readable and
are converted to rows on load; fields that cannot be recovered (TP/FP/FN,
predicted volumes) stay unknown.

## Finding and comparing runs

```bash
python scripts/runs.py list
python scripts/runs.py list --tag baseline --model nnunet
python scripts/runs.py show 501_regions_pp
python scripts/runs.py validate --all              # missing cases/folds, conflicts
python scripts/runs.py results --all --class-name Scar --gt-present
python scripts/runs.py results --all --pred-empty  # empty predictions (score failures)
python scripts/runs.py compare --all --metric dice --require-same-cases
python scripts/runs.py compare --all --metric hd95 --case-set intersection --csv /tmp/cmp.csv
python scripts/runs.py cases --out var/cache/case_metadata.json
```

`compare` defaults to the intersection of case sets and refuses runs whose
per-case folds disagree. The `missing` column counts selected cases absent from
a run; `excluded` counts cases the run has but that fall outside the comparison
set. Use `--require-same-cases` to reject runs with non-identical case sets,
and `--allow-mixed-splits` only when you intentionally ignore differing
`data.splits_sha256` values. Conflicting manifests are rejected by `compare`
and flagged by `validate`/`show`.

`results` prints tidy rows (one per case, fold, and class). Besides
`--class-name`/`--gt-present` it supports `--pred-empty` (rows with exactly zero
predicted voxels; runs whose metrics cannot answer this are excluded, unknown
never counts), `--fp-min`/`--fn-min` for false-positive/negative volume checks,
and `--fold`/`--group` to narrow by fold or dataset group. Routes go to
`--csv <file>` (with per-run row counts); `--limit`/`--print-all` control the
terminal view.

`cases` builds a case metadata cache (`src/datautils/case_metadata.py`):
per-case image/spacing geometry and ground-truth class volumes read directly
from the NIfTI files, for use in overlays and reports without full evaluation.

Python access:

```python
from src.runs import RunStore

store = RunStore()
run = store.get("nnUNet_validation_501_pp")       # or any discovered run id
run.manifest.get("postprocess.ops")
run.manifest.get("data.splits_sha256")

index = run.prediction_index()
pred = index.load("Case_P064")                    # NIfTI only when needed
metrics = run.load_metrics()                      # ResultsTable, no volumes
scar = metrics.filter(class_id=3, gt_present=True)
scar.summary("dice")
```

## Workflow

1. Export to nnU-Net format and train on Fir (`scripts/export_nnunet.py`,
   `fir/submit.sh`). `submit.sh` stores its own nnU-Net results; metadata for
   the prediction run is recorded at evaluation time.
2. Predict (`fir/predict.sh`), then post-process:

   ```bash
   python scripts/apply_postprocess.py \
     --pred-dir var/nnUNet_validation_501 \
     --out-dir var/nnUNet_validation_501_pp \
     --run-id 501_pp --parent-run 501 --tag gated
   ```

3. Evaluate; this writes tidy metrics and `run.json`:

   ```bash
   python scripts/evaluate_nnunet_cv.py \
     --pred-dir var/nnUNet_validation_501_pp \
     --out var/results/501_pp.json --run-id 501_pp \
     --model nnunet --trainer nnUNetTrainer --plans nnUNetResEncUNetMPlans \
     --config-name 2d --label-scheme regions --channels 1 \
     --checkpoints var/nnUNet_results/Dataset501_EMIDEC_Regions
   ```

4. Stage 2 (polar). Training writes per-fold `config.json`, checkpoints, and
   `run.json`; checkpoints embed the same config so evaluation never guesses
   model flags.
   Prediction and scoring are separate so gates can change without re-inference:

   ```bash
   python scripts/predict_polar.py --model-dir var/polar_models \
     --out-dir var/polar_preds_raw --stage1-dir var/nnUNet_validation_501_pp \
     --mode additive --tau 0.7 --dilation 0 --run-id polar_add_raw

   python scripts/apply_postprocess.py --pred-dir var/polar_preds_raw \
     --out-dir var/polar_preds_pp --run-id polar_add_pp \
     --parent-run polar_add_raw --scar-grow-px 0

   python scripts/evaluate_nnunet_cv.py --pred-dir var/polar_preds_pp \
     --out var/results/polar_add_pp.json --run-id polar_add_pp
   ```

   `scripts/evaluate_polar.py --raw` is the one-shot variant that predicts and
   scores ungated outputs; without `--raw` it preserves the historical
   gated-output behaviour. `--mode additive --tau ...` reproduces the
   documented additive merge variant.

## Adding a model

The trainer accepts an injected model, datasets, and loss, and records
`model.name`/`model.params` in the manifest:

```python
from src.polar.trainer import PolarTrainer
from src.datautils.splits import CVSplits

trainer = PolarTrainer(
    fold=0, save_dir="var/runs/my_model", run_id="my_model_f0",
    model=MyModel(in_channels=1), train_ds=my_train_ds, val_ds=my_val_ds,
    criterion=my_loss, model_name="my_model",
    model_params={"in_channels": 1},
    tags=["my_model"],
)
trainer.fit()
```

The default `PolarDataset` emits one channel per slice. A multi-channel model
needs its own dataset/preprocessing (or a `NeighborStacker`-based dataset) and a
matching prediction path; only model construction is generic today. From the
CLI: `python scripts/train_polar.py --model-class mypkg.models:MyUNet
--model-params '{"in_channels": 1}'`.

For it to be usable by the rest of the pipeline, inference must write the same
`fold_<k>/validation/<EMIDEC_id>.nii.gz` layout (see
`src/polar/predict.py`) and a `run.json`. Post-processing, scoring, the viewer,
and `runs.py` then work unchanged.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/runs.py validate --all
```
