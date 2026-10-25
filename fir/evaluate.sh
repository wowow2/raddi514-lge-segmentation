#!/usr/bin/env bash
# Evaluate the Dataset503 ROI-cascade 3d_fullres ResEnc-M run on Fir (CPU job).
# Runs fresh OOF inference from checkpoint_best.pth, then scores raw and gated
# outputs. Never reuses historical training validation trees (final checkpoint).
# Preserves final results and writes distinct best-checkpoint outputs.
#
#   ./evaluate.sh 503
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="${RADDI_CODE:-/scratch/aarizvi2/raddi514code}"
DATASET="${1:?Usage: evaluate.sh <dataset-id>}"

SBATCH_FILE="$(mktemp /tmp/eval_${DATASET}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_${DATASET}
#SBATCH --output=/scratch/aarizvi2/raddi514/logs/eval_${DATASET}_%j.out
set -euo pipefail
source "$HERE/env.sh"
cd "$CODE"

NAME="\$(resolve_ds $DATASET)"
D0="\$nnUNet_results/\$NAME/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
PRED="\$nnUNet_results/\$NAME/probs_nnUNetResEncUNetMPlans_3d_fullres_best"
GT="\$FIR_BASE/polar_data/emidec_train"
SPLITS="\$nnUNet_raw/\$NAME/splits_final.json"
OUTDIR="\$FIR_BASE/results"
mkdir -p "\$OUTDIR"

# Per-fold validation inputs only. SLURM_TMPDIR is unique to this job.
INPUT_ROOT="\$SLURM_TMPDIR/cascade_best_inputs"
export OMP_NUM_THREADS=4
python - "\$SPLITS" "\$nnUNet_raw/\$NAME/imagesTr" "\$INPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

splits_path, source, target = map(Path, sys.argv[1:])
for fold, split in enumerate(json.loads(splits_path.read_text())):
    folder = target / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    for case in split["val"]:
        channels = sorted(source.glob(f"{case}_*.nii.gz"))
        if not channels:
            raise FileNotFoundError(f"No channels for {case} in {source}")
        for channel in channels:
            (folder / channel.name).symlink_to(channel.resolve())
PY

# Explicit checkpoint; no --continue_prediction, so stale labels are replaced.
for f in 0 1 2 3 4; do
  test -f "\$D0/fold_\$f/checkpoint_best.pth"
  nnUNetv2_predict -d ${DATASET} -i "\$INPUT_ROOT/fold_\$f" \\
    -o "\$PRED/fold_\$f/validation" -f "\$f" \\
    -c 3d_fullres -p nnUNetResEncUNetMPlans -tr nnUNetTrainer \\
    -chk checkpoint_best.pth -device cpu --save_probabilities --disable_progress_bar
done
echo "prediction files: \$(find "\$PRED" -name '*.nii.gz' | wc -l) (expect 100)"

# ROI space -> full volume space.
PRED_FULL="\$nnUNet_results/\$NAME/preds_roicascade_best_full"
python scripts/merge_cascade.py \
  --pred-dir "\$PRED" --manifest "\$nnUNet_raw/\$NAME/cascade_manifest.json" \
  --gt-root "\$GT" --out-dir "\$PRED_FULL"

# Ungated.
python scripts/evaluate_nnunet_cv.py \
  --pred-dir "\$PRED_FULL" --gt-root "\$GT" --splits "\$SPLITS" \
  --out "\$OUTDIR/${DATASET}_best_raw.json" --run-id "${DATASET}_best" \
  --model nnunet --stage stage2 --trainer nnUNetTrainer --plans nnUNetResEncUNetMPlans \
  --config-name 3d_fullres --label-scheme cascade --channels 6 \
  --description "ROI cascade stage-2, LGE + 5 OOF probability channels, merged to full space" \
  --checkpoints "\$D0" --checkpoint best --probabilities "\$PRED"

# Post gates.
python scripts/apply_postprocess.py \
  --pred-dir "\$PRED_FULL" --out-dir "\${PRED_FULL}_pp" \
  --run-id "${DATASET}_best_pp" --parent-run "${DATASET}_best" \
  --splits "\$SPLITS" \
  --tag gated

python scripts/evaluate_nnunet_cv.py \
  --pred-dir "\${PRED_FULL}_pp" --gt-root "\$GT" --splits "\$SPLITS" \
  --out "\$OUTDIR/${DATASET}_best_pp.json" --run-id "${DATASET}_best_pp" \
  --model nnunet --stage stage2 --trainer nnUNetTrainer --plans nnUNetResEncUNetMPlans \
  --config-name 3d_fullres --label-scheme cascade --channels 6 \
  --description "ROI cascade stage-2, post gates" \
  --checkpoints "\$D0" --checkpoint best --probabilities "\$PRED" --tag gated

echo "== results =="
ls -l "\$OUTDIR"/${DATASET}_best*.json
EOF

sbatch --test-only "$SBATCH_FILE"
sbatch "$SBATCH_FILE"
squeue -u "$USER"
