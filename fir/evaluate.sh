#!/usr/bin/env bash
# Evaluate the Dataset503 ROI-cascade 3d_fullres ResEnc-M run on Fir (CPU job).
# Assembles the evaluate layout from the fold validation trees produced by the
# training runs, scores ungated and post-gated (scar/MVO gates), and writes
# metrics JSON under $FIR_BASE/results.
#
#   ./evaluate.sh 503
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE=/scratch/aarizvi2/raddi514code
DATASET="${1:?Usage: evaluate.sh <dataset-id>}"

SBATCH_FILE="$(mktemp /tmp/eval_${DATASET}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_${DATASET}
#SBATCH --output=/scratch/aarizvi2/raddi514/logs/eval_${DATASET}_%j.out
set -euo pipefail
source "$HERE/env.sh"
cd "$CODE"

NAME="\$(resolve_ds $DATASET)"
D0="\$nnUNet_results/\$NAME/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
PRED="\$nnUNet_results/\$NAME/preds_nnUNetResEncUNetMPlans_3d_fullres"
GT="\$FIR_BASE/polar_data/emidec_train"
SPLITS="\$nnUNet_raw/\$NAME/splits_final.json"
OUTDIR="\$FIR_BASE/results"
mkdir -p "\$OUTDIR"

# Assemble the evaluation layout (fresh copy, dereferenced).
rm -rf "\$PRED" "\${PRED}_pp"
for f in 0 1 2 3 4; do
  mkdir -p "\$PRED/fold_\$f"
  cp -rL "\$D0/fold_\$f/validation" "\$PRED/fold_\$f/validation"
done
echo "prediction files: \$(find "\$PRED" -name '*.nii.gz' | wc -l) (expect 100)"

# ROI space -> full volume space.
PRED_FULL="\$nnUNet_results/\$NAME/preds_roicascade_full"
rm -rf "\$PRED_FULL"
python scripts/merge_cascade.py \
  --pred-dir "\$PRED" --manifest "\$nnUNet_raw/\$NAME/cascade_manifest.json" \
  --gt-root "\$GT" --out-dir "\$PRED_FULL"

# Ungated.
python scripts/evaluate_nnunet_cv.py \
  --pred-dir "\$PRED_FULL" --gt-root "\$GT" --splits "\$SPLITS" \
  --out "\$OUTDIR/${DATASET}_3d_raw.json" --run-id "${DATASET}_3d" \
  --model nnunet --trainer nnUNetTrainer --plans nnUNetResEncUNetMPlans \
  --config-name 3d_fullres --label-scheme cascade --channels 6 \
  --description "ROI cascade stage-2, LGE + 5 OOF probability channels, merged to full space" \
  --checkpoints "\$D0"

# Post gates.
python scripts/apply_postprocess.py \
  --pred-dir "\$PRED_FULL" --out-dir "\${PRED_FULL}_pp" \
  --run-id "${DATASET}_3d_pp" --parent-run "${DATASET}_3d" \
  --splits "\$SPLITS" \
  --tag gated

python scripts/evaluate_nnunet_cv.py \
  --pred-dir "\${PRED_FULL}_pp" --gt-root "\$GT" --splits "\$SPLITS" \
  --out "\$OUTDIR/${DATASET}_3d_pp.json" --run-id "${DATASET}_3d_pp" \
  --model nnunet --trainer nnUNetTrainer --plans nnUNetResEncUNetMPlans \
  --config-name 3d_fullres --label-scheme cascade --channels 6 \
  --description "ROI cascade stage-2, post gates" \
  --checkpoints "\$D0" --tag gated

echo "== results =="
ls -l "\$OUTDIR"/${DATASET}_3d*.json || true
EOF

chmod +x "$0"

sbatch --test-only "$SBATCH_FILE"
sbatch "$SBATCH_FILE"
squeue -u "$USER"
