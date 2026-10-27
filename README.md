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

Canonical storage roots are selected by `config/roots.json`. Environment
overrides in `.env` are loaded by `src/config.py`:

| variable | default |
|---|---|
| `EMIDEC_GT_ROOT` | `data/source/emidec/train` |
| `NNUNET_PRED_DIR` | baseline output's canonical prediction path |
| `NNUNET_SPLITS_PATH` | original hash-named file under `data/splits/` |
| `NNUNET_CV_OUT` | baseline output's canonical metrics path |

All path arguments are resolved relative to the repository root, so scripts
behave the same no matter where they are launched from.

## Layout

- `src/datautils/`: EMIDEC loading, splits and validation (`splits.py`), shared
  prediction lookup (`predictions.py`), metrics, post-processing, nnU-Net export
- `src/runs.py`: run manifests, discovery, filtering, completeness checks
- `src/results.py`: tidy per-case/per-class results and run comparison
- `src/analysis/probs.py`: out-of-fold probability loading for the cascade and viewer
- `src/polar/`: polar transform, model, training, checkpoint helpers,
  prediction, and merge
- `src/viewer/`: PyQt5 case explorer (Explore/Metrics/Compare)
- `fir/`: SLURM scripts for training and prediction on the Fir cluster
- `scripts/`: entry points and the `runs.py` query CLI
- `tests/`: standard-library `unittest` coverage (no pytest required)

Generate the dataset preamble and all experiment results with:

```bash
.venv/bin/python scripts/report_cv.py --html
```

Open `reports/experiments.html`. It includes every canonical output, metric
sample counts, GT-present Dice, and checkpoint provenance, without plots.
Initially only experiment defaults are visible; uncheck “Show default outputs
only” to see the other retained views.
Regenerate after updating metadata or metrics. Reports and viewer exports go
to ignored `reports/`; culled artifacts go to ignored `data/archive/`.
See [`READINGS.md`](READINGS.md) for references.

## Experiment metadata

After migration, `experiments/<experiment>/experiment.json` and each
`outputs/<output>/output.json` are authoritative. Pipeline writers update
canonical metadata directly; training-only metadata goes to `training.json`.
Every step contributes what it knows:

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

Unknown values are stored as `null` and never guessed. Discovery reads only
canonical experiment/output JSON. Local `var/` paths and migration/reconstruction
commands have been retired. Unverified historical files are retained in ignored
`data/archive/`; Fir's existing launch paths remain intact. Pipeline steps
write their own `run.json`/`output.json`; there is no separate registration
command.

Older metrics files (`per_case` lists with nested `dice`/`hd95`) stay readable;
they are converted to rows on load, and fields that cannot be recovered (TP,
FP, FN, predicted volumes) remain unknown.

## Finding and comparing runs

```bash
python scripts/runs.py list
python scripts/runs.py list --tag baseline --model nnunet
python scripts/runs.py show nnUNet_validation_501_pp
python scripts/runs.py validate                   # split hashes + missing folds/cases
python scripts/runs.py results --all --class-name Scar --gt-present
python scripts/runs.py results --all --pred-empty  # empty predictions (score failures)
python scripts/runs.py compare --all --metric dice --require-same-cases
python scripts/runs.py compare --all --metric hd95 --case-set intersection --csv /tmp/cmp.csv
```

Both `results` and `compare` share a flag vocabulary for choosing runs: `--tag`,
`--model`, `--stage`, and `--parent` are selectors; `--all` sweeps every
selected run.

`compare` defaults to the intersection of case sets and refuses runs whose
per-case folds disagree. In its output, `missing` counts selected cases that a
run lacks, and `excluded` counts cases the run has but that fall outside the
comparison set. `--require-same-cases` rejects runs with non-identical case
sets. `--allow-mixed-splits` is only for when you intentionally ignore
differing `data.splits_sha256` values; otherwise `compare` refuses them.

`results` prints one row per case, fold, and class. Beyond
`--class-name`/`--gt-present` it takes:

- `--pred-empty`: rows with exactly zero predicted voxels, i.e. cases where the
  model predicted an empty structure. Runs whose metrics cannot answer this
  are excluded; an unknown count never satisfies the filter.
- `--fp-min` / `--fn-min`: false-positive and false-negative volume checks
- `--fold`, `--group`: narrow by fold or dataset group

Write the full row set with `--csv <file>`; `--limit` and `--print-all`
control the terminal view.

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

Use the best checkpoint throughout: `fir/submit.sh` passes `--val_best`, and
nnU-Net prediction explicitly passes `-chk checkpoint_best.pth`. Polar
prediction uses `checkpoint_best.pt`. Historical validation folders created
without `--val_best` contain final-model predictions and must not be relabelled
as best-checkpoint outputs.

For Dataset503, the current baseline is **`503_best_pp`**.
The viewer prefers this run when available. Historical final-checkpoint views
(`503_3d`, `503_3d_pp`, `503_3d_raw`) and the dangling `polar_preds` alias are
archived under `data/archive/`. `fir/evaluate.sh 503`
regenerates best-checkpoint OOF predictions in a separate directory, scores raw
and gated outputs, and records `inference.checkpoint = "best"`. This runs CPU
inference with an eight-hour allocation; existing evaluated local best outputs
do not need to be regenerated just to update the viewer.

Fir utility commands:

```bash
# Held-out predictions and probabilities (replaces predict_probs.sh).
bash fir/predict.sh 500 --save-probabilities --tr=nnUNetTrainer_probabilisticOversampling
# Held-out labels only; --local runs the same loop on CPU without submitting.
bash fir/predict.sh 501 --oof --cpu
# Check a selected model and pack its dataset results (replaces collect.sh).
bash fir/status.sh 503 --config=3d_fullres --tr=nnUNetTrainer_probabilisticOversampling --pack
# Evaluate new Dataset503 models without reusing the baseline's output paths.
bash fir/evaluate.sh 503 --tr=nnUNetTrainer_probabilisticOversampling --run-id=503_over_best
bash fir/evaluate.sh 503 --plans=nnUNetPrimusV3SPlans --tr=nnUNet_PrimusV3S_Trainer --run-id=503_primus_best
```

New training jobs write native nnU-Net directories selected by dataset, trainer,
plans and configuration. They do not appear automatically in the local report.
After completion, import their outputs locally, restore crop-space labels,
evaluate raw and postprocessed predictions, and record each model as a separate
canonical experiment with its own default output. Regenerate the HTML report
after registration; preserve the existing BEST/FINAL baseline outputs.

1. Export to nnU-Net format and train on Fir (`scripts/export_nnunet.py`,
   `fir/submit.sh`). `submit.sh` keeps its own nnU-Net results; metadata for
   the prediction run is recorded at evaluation time.
2. Predict (`fir/predict.sh`), then post-process:

   ```bash
   python scripts/apply_postprocess.py \
     --pred-dir experiments://nnUNet_validation_501/outputs/nnUNet_validation_501/predictions \
     --out-dir experiments://nnUNet_validation_501/outputs/nnUNet_validation_501_pp/predictions \
     --run-id nnUNet_validation_501_pp --parent-run nnUNet_validation_501 --tag gated
   ```

3. Evaluate; this writes tidy metrics and canonical output metadata:

   ```bash
   python scripts/evaluate_nnunet_cv.py \
     --pred-dir experiments://nnUNet_validation_501/outputs/nnUNet_validation_501_pp/predictions \
     --out experiments://nnUNet_validation_501/outputs/nnUNet_validation_501_pp/metrics.json --run-id nnUNet_validation_501_pp \
     --model nnunet --trainer nnUNetTrainer --plans nnUNetResEncUNetMPlans \
     --config-name 2d --label-scheme regions --channels 1 \
     --checkpoints nnunet://nnUNet_results/Dataset501_EMIDEC_Regions/nnUNetTrainer__nnUNetResEncUNetMPlans__2d --checkpoint best
   ```

4. Stage 2, the polar model. Training writes per-fold `config.json`,
   checkpoints, and canonical training metadata; checkpoints embed the same config so
   evaluation never has to guess model flags. Prediction and scoring are
   separate steps so gates can change without re-running inference:

   ```bash
   python scripts/predict_polar.py --model-dir experiments://polar/checkpoints \
     --out-dir experiments://polar/outputs/polar_preds_raw/predictions --stage1-dir experiments://nnUNet_validation_501/outputs/nnUNet_validation_501_pp/predictions \
     --mode erase --dilation 0 --run-id polar_preds_raw

   python scripts/apply_postprocess.py --pred-dir experiments://polar/outputs/polar_preds_raw/predictions \
     --out-dir experiments://polar/outputs/polar_preds_erase/predictions --run-id polar_preds_erase \
      --parent-run polar_preds_raw

   python scripts/evaluate_nnunet_cv.py --pred-dir experiments://polar/outputs/polar_preds_erase/predictions \
     --out experiments://polar/outputs/polar_preds_erase/metrics.json --run-id polar_preds_erase
   ```

   `scripts/evaluate_polar.py --raw` is the one-shot variant: it predicts and
   scores ungated outputs in a single command. Without `--raw` it keeps the
   historical behaviour of reading pre-gated outputs. The retained main Polar experiment uses erase merging.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/runs.py validate
```
