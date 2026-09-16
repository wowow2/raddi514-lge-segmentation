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
| MVO | 0.657 | 0.410 | 19.17 | 11.00 |
| **All** | **0.744** | **0.301** | **9.68** | — |

