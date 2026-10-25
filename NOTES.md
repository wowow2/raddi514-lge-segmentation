## Dataset: EMIDEC

* **Cases:** 100 training (67 Pathological `Case_P*`, 33 Normal `Case_N*`), 50 blind testing.
* **Modality:** Short-axis LGE Cardiac MRI.

### Geometry & Slices
* **Matrix:** Median $240 \times 256$ in-plane ($1.46 \times 1.46$ mm). It is not consistent.
* **Slices ($Z$):** Median 7 slices per patient (range 4–10).
* **Slice Gap:** $10.0$ mm thickness ($\sim 7\times$ coarser than in-plane $\rightarrow$ use 2D models).

### Classes & Voxel Distribution
* **0 - Background:** $97.2\%$
* **1 - LV Cavity:** $1.41\%$
* **2 - Normal Myocardium:** $1.21\%$ (nulled signal $\approx 2256$)
* **3 - Infarct / Scar:** $0.19\%$ (bright signal $\approx 2551$, near-identical to cavity)
* **4 - MVO / No-Reflow:** $0.02\%$ (in $40/67$ infarct cases)

### Validation Plan
* 5-fold cross-validation stratified across Pathological and Normal cases.
* Metrics: Dice (DSC), 95% Hausdorff Distance (HD95).
* Report metrics from test set

### What is LGE-MRI?
* **Concept:** Late Gadolinium Enhancement (LGE) is the clinical gold standard for imaging myocardial damage after a heart attack.
* **Mechanism:** The patient gets an IV injection of a gadolinium contrast agent and waits 10–15 minutes ("Late")

### nnUNet baseline

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.935 | 0.019 | 6.16 | 18.04 |
| Myo | 0.836 | 0.048 | 4.57 | 7.18 |
| Scar | 0.551 | 0.318 | 18.17 | 19.87 |
| Scar (gated) | 0.656 | 0.288 | 18.03 | 20.29 |
| MVO | 0.695 | 0.394 | 18.82 | 11.45 |
| **All (gated)** | **0.780** | **0.270** | **9.43** | — |

![plain per-case Dice](var/results/dice_plain.png)

### nnUNet ResEnc-M, flat labels

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.932 | 0.024 | 9.43 | 29.66 |
| Myo | 0.835 | 0.048 | 6.72 | 22.76 |
| Scar | 0.470 | 0.318 | 20.16 | 28.51 |
| Scar (gated) | 0.668 | 0.275 | 18.34 | 24.82 |
| MVO | 0.721 | 0.378 | 16.51 | 9.60 |
| **All (gated)** | **0.789** | **0.257** | **11.17** | — |

![resenc per-case Dice](var/results/dice_resenc.png)

### nnUNet ResEnc-M, region labels (Dataset501: whole_heart > myocardium > infarction > no_reflow)

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.932 | 0.023 | 9.97 | 27.32 |
| Myo | 0.833 | 0.052 | 7.05 | 21.24 |
| Scar | 0.491 | 0.321 | 19.05 | 23.31 |
| Scar (gated) | 0.677 | 0.273 | 17.22 | 18.53 |
| MVO | 0.696 | 0.398 | 17.36 | 9.65 |
| **All (gated)** | **0.784** | **0.264** | **11.21** | — |

![regions per-case Dice](var/results/dice_regions.png)

### nnUNet ResEnc-M, region labels + 2.5D input (Dataset502: neighbor stack z-1/z/z+1)

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.935 | 0.021 | 6.57 | 19.59 |
| Myo | 0.833 | 0.051 | 5.85 | 16.81 |
| Scar | 0.536 | 0.311 | 16.69 | 22.69 |
| Scar (gated) | 0.683 | 0.259 | 16.35 | 22.93 |
| MVO | 0.667 | 0.425 | 18.97 | 9.01 |
| **All (gated)** | **0.779** | **0.274** | **9.42** | — |

![2.5d per-case Dice](var/results/dice_2.5d.png)

### nnUNet ResEnc-M + probabilistic oversampling (Dataset500, foreground-emphasis trainer)

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.933 | 0.021 | 9.14 | 27.62 |
| Myo | 0.835 | 0.049 | 6.02 | 16.82 |
| Scar | 0.462 | 0.319 | 19.82 | 28.41 |
| Scar (gated) | 0.690 | 0.264 | 17.84 | 25.25 |
| MVO | 0.703 | 0.388 | 16.33 | 9.22 |
| **All (gated)** | **0.790** | **0.256** | **10.63** | — |

![oversampling per-case Dice](var/results/dice_over.png)

### ROI 3D cascade (Dataset503: LGE + 5 OOF probability channels, ResEnc-M 3d fullres)

* **Export:** `scripts/export_cascade.py` builds square in-plane ROI crops from the largest 3D component of the stage-1 foreground (pad 16, min 64, size multiple of 16, Z untouched), 6 input channels = LGE + the Dataset500 probabilistic-oversampling out-of-fold softmax probabilities, full 5-class labels; each case bbox is recorded in `cascade_manifest.json`.
* **Train:** nnU-Net ResEnc-M `3d_fullres`, 5 folds, ~23 h/fold on Fir H100. Internal validation (ungated, ROI-crop space): cavity 0.939, myo 0.833, scar 0.577, MVO 0.307; fold 3 is the weak scar fold (0.448).
* **Evaluate:** `src/cascade/merge.py` pastes each fold validation prediction into a full-volume label map at the recorded bbox (`fir/evaluate.sh`); sorted under the same gates as the other runs.

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.939 | 0.018 | 2.40 | 0.91 |
| Myo | 0.833 | 0.050 | 3.19 | 2.70 |
| Scar (raw) | 0.710 | 0.246 | 12.87 | 11.90 |
| Scar (gated) | 0.718 | 0.241 | 12.76 | 11.97 |
| MVO (raw) | 0.700 | 0.387 | 15.45 | 10.07 |
| MVO (gated) | 0.705 | 0.390 | 15.06 | 10.12 |
| All (raw) | 0.796 | 0.251 | 6.27 | — |
| **All (gated)** | **0.799** | **0.250** | **6.09** | — |

![cascade per-case Dice](var/results/dice_cascade.png)

* **Why it wins:** scar 0.718 (gated) beats every stage-1 run (best 0.690) and MVO edges ahead too; HD95 drops from ~10.6 to 6.09. The OOF probability channels + ROI input carry the edge/low-signal information the stage-2 model needs.
* **Gates barely bite** (scar raw→gated +0.008, all-class +0.003): the 3D model produces few normal-case specks, so the speck-driven gate story is largely a stage-1 property.

### Polar cascade (stage-2 strip U-Net on 501-regions preds, gated)

| Variant | Cavity | Myo | Scar | MVO | **All** |
|---|---|---|---|---|---|
| erase, d=0 | 0.931 | 0.832 | 0.663 | 0.526 | **0.738** |
| additive, d=0, τ=0.7 | 0.932 | 0.835 | 0.695 | 0.689 | **0.788** |

### Post-processing: scar + MVO gates

* **Scar rule:** zero the scar label when fewer than 50 scar voxels are predicted (`PostProcessor`, `scripts/apply_postprocess.py`). Threshold picked by validation sweep (25–100 flat, peak at 50) and frozen; needs only predictions, so the same code runs on blind test data.
* **Why:** under the challenge convention (both-empty → 1.0) one speck on a scar-free case scores 0.0. Stratified analysis showed pathological scar tied ~0.54 across all runs — the raw gaps came from normal-case specks, not backbone quality. Containment filtering tested +0.000 (specks sit inside predicted cardiac territory); dropped.
* **MVO rule:** zero the MVO label when fewer than 10 MVO voxels are predicted (same class, `--min-mvo-voxels`). Peak uniform across all four runs (plain/resenc/regions/2.5d); stage-1 small MVOs are partly real so the gate bites past 10, while merged (polar) small MVOs are overwhelmingly added specks and gain monotonically to 20. Locked at 10 for stage-1 outputs.
* **Effect:**

| Run | Scar raw | Scar gated | MVO ungated | MVO gated |
|---|---|---|---|---|
| Plain (500) | 0.551 | 0.656 | 0.657 | 0.695 |
| ResEnc-M flat (500) | 0.470 | 0.668 | 0.668 | 0.721 |
| ResEnc-M regions (501) | 0.491 | 0.677 | 0.688 | 0.696 |
| ResEnc-M regions + 2.5D (502) | 0.536 | 0.683 | 0.642 | 0.667 |
| ResEnc-M + prob. oversampling (500) | 0.462 | 0.690 | 0.694 | 0.703 |

### Probability diagnostic (Dataset500 plain ResEnc, flat 5-class, out-of-fold softmax)

* Scar voxels the model labels cavity are confidently wrong, not borderline: on the hard cases (P064, P088, P011, P066, P051, P003) P(cavity) is 0.955-0.986 and P(scar) is 0.001-0.020. Recall at P(scar) > 0.05 is only 3-25%.
* Scar threshold sweep (override cavity/myo to scar when P(scar) > t, then the 50-voxel gate): scar-present Dice 0.550 (argmax) -> 0.567 at t=0.05 but normal-case FPs go 3 -> 7 of 33 (all-case 0.640); t=0.2 gives +0.009 scar-present with no extra FPs. Not adopted (within noise, tuned on the same folds). Conclusion: probabilities carry no extra signal at the errors, so a probability-input refiner was not pursued.

### Where the scar error is (same model, 67 pathological cases)

* Pooled scar recall 0.535, precision 0.688; predicted/GT scar volume ratio median 0.77 (IQR 0.53-0.93). The model is conservative.
* GT scar voxels go to: scar 0.535, myocardium 0.192, cavity 0.167, background 0.098, MVO 0.008. Predicted scar voxels are really: scar 0.688, cavity 0.126, myocardium 0.115, background 0.041, MVO 0.030. Errors occur on all sides, not only the cavity edge.
* GT scar voxels lost to cavity have intensity index (0 = myocardium, 1 = blood pool) median 1.16 vs 1.20 for correctly found scar, i.e. they look like scar, not blood. Median depth inside the predicted cavity is 1.0 px (51% within 1 px, 83% within 2 px; 59 cases with >= 20 lost voxels).
* GT-only simulation: shifting the endocardial scar layer by 1/2/3 px gives Dice 0.903/0.779/0.637 (mean), so a pure edge offset does not by itself explain 0.550.
* Persistent failures: P051 (GT contour partly outside the predicted heart), P064, P088 (scar-present Dice ~0 to 0.13 in every run); nine cases with GT MVO of 13-39 voxels (P003, P010, P015, P039, P045, P063, P064, P074, P092) get MVO Dice 0 in all runs.



