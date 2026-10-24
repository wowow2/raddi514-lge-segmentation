# LGE Cardiac MRI Segmentation

RADD I 514 course project: segmentation of late gadolinium enhancement (LGE)
cardiac MRI on the EMIDEC dataset. Five classes: background, LV cavity,
myocardium, scar, and MVO (no-reflow zone).

The main approach is a two-stage cascade: an nnU-Net v2 backbone feeds a custom
polar-coordinate U-Net, followed by anatomical post-processing. All variants
are evaluated with 5-fold cross-validated Dice and 95th-percentile Hausdorff
distance.

## Setup

Python 3.12, virtualenv at `.venv`. Main dependencies: `torch`, `nnunetv2`,
`nibabel`, `surface_distance`, `scikit-learn`, `scipy`, `matplotlib`, `PyQt5`. Tests use
only the standard library (`unittest`); `pytest` is not required.

Data locations default to `var/` under the repository root and can be
overridden in `.env`, which `src/config.py` loads at import:

| variable | default |
|---|---|
| `EMIDEC_GT_ROOT` | `var/emidec_dataset/train` |
| `NNUNET_PRED_DIR` | `var/nnUNet_validation` |
| `NNUNET_SPLITS_PATH` | `<NNUNET_PRED_DIR>/splits_final.json` |
| `NNUNET_CV_OUT` | `var/results/nnunet_2d_cv_metrics.json` |

All path arguments are resolved relative to the repository root, so scripts
behave the same no matter where they are launched from.

## Layout

- `src/datautils/`: EMIDEC loading, splits and validation (`splits.py`), shared
  prediction lookup (`predictions.py`), metrics, post-processing, nnU-Net export
- `src/runs.py`: run manifests, discovery, filtering, completeness checks
- `src/results.py`: tidy per-case/per-class results and run comparison
- `src/analysis/probs.py`: out-of-fold probability diagnostics
- `src/polar/`: polar transform, model, training, checkpoint helpers,
  prediction, and merge
- `src/viewer/`: PyQt5 case explorer (Explore/Metrics/Compare)
- `fir/`: SLURM scripts for training and prediction on the Fir cluster
- `scripts/`: entry points and the `runs.py` query CLI
- `tests/`: standard-library `unittest` coverage (no pytest required)

See [`NOTES.md`](NOTES.md) for dataset details and results, and
[`READINGS.md`](READINGS.md) for references.

## Experiment metadata

Training, prediction, post-processing, and evaluation each write a `run.json`
manifest next to their output. Manifests with the same `run_id` are merged, so
every step contributes what it knows:

- `model`: architecture name and constructor parameters, input channels, label
  scheme
- `data`: dataset and gt roots, splits path, splits content hash, fold and case
  counts
- `training`: seed, epochs, batch size, lr, weight decay, loss, augmentation
- `inference`: checkpoint choice, device, merge mode, dilation, tau
- `postprocess`: ordered operations with parameters
- `parent_run` / `source_run`: the training run that produced this output and
  the stage-1 run it builds on
- `paths`: predictions, probabilities, checkpoints, metrics
- `status`: expected outputs and per-fold prediction coverage
- `provenance`: git commit and dirty flag, command, package versions, host,
  timestamp

Unknown values are stored as `null` and never guessed. Directories created
before this metadata system existed still work: `RunStore` discovers their
prediction trees and `var/results/*.json` files read-only, marks them
`provenance.legacy`, and treats missing fields as unknown. Attach explicit
metadata whenever you know it:

```bash
python scripts/runs.py register 501_regions_pp \
  --predictions var/nnUNet_validation_501_pp \
  --metrics var/results/nnunet_501_cv_metrics_pp.json \
  --model nnunet --stage stage1 --tags regions,baseline \
  --description "ResEnc-M, region labels, scar/MVO gates"
```

Older metrics files (`per_case` lists with nested `dice`/`hd95`) stay readable;
they are converted to rows on load, and fields that cannot be recovered (TP,
FP, FN, predicted volumes) remain unknown.

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

Both `results` and `compare` share a flag vocabulary for choosing runs: `--tag`,
`--model`, `--stage`, and `--parent` are selectors; `--all` sweeps every
selected run.

`compare` defaults to the intersection of case sets and refuses runs whose
per-case folds disagree. In its output, `missing` counts selected cases that a
run lacks, and `excluded` counts cases the run has but that fall outside the
comparison set. `--require-same-cases` rejects runs with non-identical case
sets. `--allow-mixed-splits` is only for when you intentionally ignore
differing `data.splits_sha256` values. Conflicting manifests are rejected by
`compare` and flagged by `validate` and `show`.

`results` prints one row per case, fold, and class. Beyond
`--class-name`/`--gt-present` it takes:

- `--pred-empty`: rows with exactly zero predicted voxels, i.e. cases where the
  model predicted an empty structure. Runs whose metrics cannot answer this
  are excluded; an unknown count never satisfies the filter.
- `--fp-min` / `--fn-min`: false-positive and false-negative volume checks
- `--fold`, `--group`: narrow by fold or dataset group

Write the full row set with `--csv <file>`; `--limit` and `--print-all`
control the terminal view.

`cases` builds a case metadata cache (`src/datautils/case_metadata.py`):
per-case image and spacing geometry plus ground-truth class volumes, read
directly from the NIfTI files. Useful for overlays and reports without a full
evaluation.

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

## Viewer

`scripts/view.py` launches a PyQt5 explorer with three tabs:

```bash
python scripts/view.py                                 # all cases, every run
python scripts/view.py --split train --runs nnUNet_validation_501_pp  # restrict runs
python scripts/view.py --no-preds                      # images + GT only
```

- **Explore**: follow one case through slices or a whole-volume montage.
  Overlays are GT, prediction, both (GT fill + prediction contour), a
  TP/FP/FN error map, or a stage-1 probability heatmap; masks and contours
  have independent styles. Display-only derived channels (local contrast,
  LoG, local standard deviation, percentile windows) preview candidate model
  inputs without exporting a dataset. The side panel shows clinical metadata,
  GT and prediction volumes, per-class Dice/HD95/recall/precision, and the
  GT-to-prediction breakdown for the focus class.
- **Metrics**: per-class distributions and mean scores with 95% bootstrap
  CIs for any set of runs, plus a scorecard and a failure-triage table.
  Double-clicking a case opens it in Explore; *Review failures* loads every
  case below the threshold as an Explore queue.
- **Compare**: two linked panes, each with its own run, for the same case and
  slice (single or montage).

Explore also previews the challenge gates: dashed outlines mark prediction
components the locked postprocessing would remove (scar < 50 vox, MVO < 10,
MVO without scar) and the side panel reports the post-gate Dice.

The shortcut set is intentionally small: arrow keys move slice/case, Space
cycles mask/contour/both, `Ctrl+F` searches, `Ctrl+S` exports the current tab,
and Esc leaves the search box. Everything else is on screen; the Help dialog
lists the bindings and controls.

## Workflow

1. Export to nnU-Net format and train on Fir (`scripts/export_nnunet.py`,
   `fir/submit.sh`). `submit.sh` keeps its own nnU-Net results; metadata for
   the prediction run is recorded at evaluation time.
2. Predict (`fir/predict.sh`), then post-process:

   ```bash
   python scripts/apply_postprocess.py \
     --pred-dir var/nnUNet_validation_501 \
     --out-dir var/nnUNet_validation_501_pp \
     --run-id 501_pp --parent-run 501 --tag gated
   ```

3. Evaluate; this writes tidy metrics and a `run.json`:

   ```bash
   python scripts/evaluate_nnunet_cv.py \
     --pred-dir var/nnUNet_validation_501_pp \
     --out var/results/501_pp.json --run-id 501_pp \
     --model nnunet --trainer nnUNetTrainer --plans nnUNetResEncUNetMPlans \
     --config-name 2d --label-scheme regions --channels 1 \
     --checkpoints var/nnUNet_results/Dataset501_EMIDEC_Regions
   ```

4. Stage 2, the polar model. Training writes per-fold `config.json`,
   checkpoints, and a `run.json`; checkpoints embed the same config so
   evaluation never has to guess model flags. Prediction and scoring are
   separate steps so gates can change without re-running inference:

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

   `scripts/evaluate_polar.py --raw` is the one-shot variant: it predicts and
   scores ungated outputs in a single command. Without `--raw` it keeps the
   historical behaviour of reading pre-gated outputs. `--mode additive
    --tau ...` reproduces the documented additive merge variant.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/runs.py validate --all
```
