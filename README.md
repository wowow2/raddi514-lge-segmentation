# RADD I 514 — LGE Cardiac MRI Segmentation

Segmentation of late gadolinium enhancement (LGE) cardiac MRI on the EMIDEC dataset
(5 classes: background, LV cavity, myocardium, scar, MVO/no-reflow).

A two-stage cascade: an nnU-Net v2 backbone feeds a custom polar-coordinate U-Net,
followed by anatomical post-processing. Evaluated with 5-fold cross-validation
(Dice and 95% Hausdorff distance).

## Layout

- `src/datautils/` — EMIDEC loading, nnU-Net export, metrics, splits, post-processing
- `src/polar/` — polar transform, model, training, and merge into nnU-Net predictions
- `src/viewer/` — interactive case browser
- `fir/` — SLURM scripts for training and prediction on the Fir cluster
- `scripts/` — training, evaluation, and reporting entry points

See [`NOTES.md`](NOTES.md) for dataset details and results, and
[`READINGS.md`](READINGS.md) for references.
