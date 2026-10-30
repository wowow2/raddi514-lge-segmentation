# LGE Cardiac MRI Segmentation

RADDI 514 course project: segment late gadolinium enhancement (LGE) cardiac MRI
from the EMIDEC dataset into five classes: background, LV cavity, myocardium,
scar, and MVO. Models are evaluated with 5-fold out-of-fold Dice and HD95.

## Pipeline

1. nnU-Net v2 baselines: 2D (`500`), region labels (`501`), and 2.5D (`502`).
2. `503` ROI cascade: crop around the stage-1 foreground and train on LGE plus
   stage-1 class probabilities.
3. Optional polar-coordinate stage-2 U-Net for scar/MVO refinement.
4. Anatomical postprocessing: small-lesion gates, scar growth, and MVO
   containment.

Current best OOF run: `503_best_pp`.

## Repository layout

```text
src/
  config.py          storage roots and environment paths
  datautils/         EMIDEC loading, splits, predictions, metrics, postprocessing,
                     nnU-Net export
  cascade/           ROI geometry, channel export, cascade merge
  polar/             polar transform, U-Net, training, prediction, merge
  analysis/          out-of-fold probability loading
  plotter/           report plots
  viewer/            PyQt5 case explorer (Explore, Metrics, Compare)
  experiment_registry.py  artifact roots and catalog views
  layout.py          artifact paths, IDs, metadata catalog, output checks
  runs.py            experiment/output discovery and run queries
  results.py         tidy per-case/per-class metrics and run comparison
  reporting.py       HTML report generation
scripts/             CLI entry points (export, merge, train/predict, evaluate, report)
fir/                 SLURM scripts for the Fir cluster
experiments/         experiment.json plus outputs/<output>/output.json
data/                local datasets and splits; large artifacts ignored
tests/               standard-library unittest suite
```

Canonical artifacts live under `experiments/<experiment>/outputs/<output>/`.
New runs are discovered from their `experiment.json` and `output.json`; the
HTML report is generated from that metadata.

## Quick start

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/report_cv.py --html   # reports/experiments.html
.venv/bin/python scripts/view.py               # PyQt5 viewer
```

See [`READINGS.md`](READINGS.md) for references.
