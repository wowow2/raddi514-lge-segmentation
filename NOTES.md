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
* Metrics: Dice (DSC), 95% Hausdorff Distance (HD95), Reliability curves.
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
| MVO | 0.657 | 0.410 | 19.17 | 11.00 |
| **All (gated)** | **0.771** | **0.279** | **9.56** | — |

![plain per-case Dice](var/results/dice_plain.png)

### nnUNet ResEnc-M, flat labels

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.932 | 0.024 | 9.43 | 29.66 |
| Myo | 0.835 | 0.048 | 6.72 | 22.76 |
| Scar | 0.470 | 0.318 | 20.16 | 28.51 |
| Scar (gated) | 0.668 | 0.275 | 18.34 | 24.82 |
| MVO | 0.668 | 0.398 | 16.92 | 9.41 |
| **All (gated)** | **0.776** | **0.268** | **11.29** | — |

![resenc per-case Dice](var/results/dice_resenc.png)

### nnUNet ResEnc-M, region labels (Dataset501: whole_heart > myocardium > infarction > no_reflow)

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.932 | 0.023 | 9.97 | 27.32 |
| Myo | 0.833 | 0.052 | 7.05 | 21.24 |
| Scar | 0.491 | 0.321 | 19.05 | 23.31 |
| Scar (gated) | 0.677 | 0.273 | 17.22 | 18.53 |
| MVO | 0.688 | 0.392 | 17.66 | 9.38 |
| **All (gated)** | **0.782** | **0.263** | **11.34** | — |

![regions per-case Dice](var/results/dice_regions.png)

### nnUNet ResEnc-M, region labels + 2.5D input (Dataset502: neighbor stack z-1/z/z+1)

| Class | Dice mean | Dice std | HD95 mean | HD95 std |
|---|---|---|---|---|
| Cavity | 0.935 | 0.021 | 6.57 | 19.59 |
| Myo | 0.833 | 0.051 | 5.85 | 16.81 |
| Scar | 0.536 | 0.311 | 16.69 | 22.69 |
| Scar (gated) | 0.683 | 0.259 | 16.35 | 22.93 |
| MVO | 0.642 | 0.430 | 20.75 | 10.04 |
| **All (gated)** | **0.773** | **0.279** | **9.74** | — |

![2.5d per-case Dice](var/results/dice_2.5d.png)

### Post-processing: scar gate

* **Rule:** zero the scar label when fewer than 50 scar voxels are predicted (`PostProcessor`, `scripts/apply_postprocess.py`). Threshold picked by validation sweep (25–100 flat, peak at 50) and frozen; needs only predictions, so the same code runs on blind test data.
* **Why:** under the challenge convention (both-empty → 1.0) one speck on a scar-free case scores 0.0. Stratified analysis showed pathological scar tied ~0.54 across all runs — the raw gaps came from normal-case specks, not backbone quality. Containment filtering tested +0.000 (specks sit inside predicted cardiac territory); dropped.
* **Effect (scar Dice, MVO untouched):**

| Run | Scar raw | Scar gated | MVO |
|---|---|---|---|
| Plain (500) | 0.551 | 0.656 | 0.657 |
| ResEnc-M flat (500) | 0.470 | 0.668 | 0.668 |
| ResEnc-M regions (501) | 0.491 | 0.677 | 0.688 |
| ResEnc-M regions + 2.5D (502) | 0.536 | 0.683 | 0.642 |

