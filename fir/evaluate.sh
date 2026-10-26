#!/usr/bin/env bash
# Evaluate the Dataset503 ROI-cascade 3d_fullres ResEnc-M run on Fir (CPU job).
# Runs fresh OOF inference from checkpoint_best.pth, then scores raw and gated
# outputs. Never reuses historical training validation trees (final checkpoint).
# Preserves final results and writes distinct best-checkpoint outputs.
#
#   ./evaluate.sh 503
#   ./evaluate.sh 503 --tr=nnUNetTrainer_probabilisticOversampling --run-id=503_over_best
#   ./evaluate.sh 503 --plans=nnUNetPrimusV3SPlans --tr=nnUNet_PrimusV3S_Trainer --run-id=503_primus_best
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
CODE="$RADDI_CODE"
DATASET="${1:?Usage: evaluate.sh <dataset-id> [--plans=X] [--tr=Trainer] [--run-id=ID]}"
PLAN="nnUNetResEncUNetMPlans"
TR="nnUNetTrainer"
RUN_ID=""
for arg in "${@:2}"; do
  case "$arg" in
    --plans=*) PLAN="${arg#--plans=}" ;;
    --tr=*) TR="${arg#--tr=}" ;;
    --run-id=*) RUN_ID="${arg#--run-id=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done
SUFFIX=""
if [[ "$PLAN" != "nnUNetResEncUNetMPlans" || "$TR" != "nnUNetTrainer" ]]; then
  SUFFIX="_${TR}__${PLAN}"
fi
RUN_ID="${RUN_ID:-${DATASET}${SUFFIX}_best}"
for value in "$PLAN" "$TR" "$RUN_ID"; do
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid model/run ID: $value" >&2; exit 1; }
done

SBATCH_FILE="$(mktemp /tmp/eval_${RUN_ID}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_${RUN_ID}
#SBATCH --output=${FIR_BASE}/logs/eval_${RUN_ID}_%j.out
set -euo pipefail
source "$HERE/env.sh"
cd "$CODE"

NAME="\$(resolve_ds $DATASET)"
D0="\$nnUNet_results/\$NAME/${TR}__${PLAN}__3d_fullres"
PRED="\$nnUNet_results/\$NAME/probs_${PLAN}_3d_fullres_best${SUFFIX}"
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
    -c 3d_fullres -p ${PLAN} -tr ${TR} \\
    -chk checkpoint_best.pth -device cpu --save_probabilities --disable_progress_bar
done
echo "prediction files: \$(find "\$PRED" -name '*.nii.gz' | wc -l) (expect 100)"

# ROI space -> full volume space.
PRED_FULL="\$nnUNet_results/\$NAME/preds_roicascade_best_full${SUFFIX}"
python scripts/merge_cascade.py \
  --pred-dir "\$PRED" --manifest "\$nnUNet_raw/\$NAME/cascade_manifest.json" \
  --gt-root "\$GT" --out-dir "\$PRED_FULL"

# Ungated.
python scripts/evaluate_nnunet_cv.py \
  --pred-dir "\$PRED_FULL" --gt-root "\$GT" --splits "\$SPLITS" \
  --out "\$OUTDIR/${RUN_ID}_raw.json" --run-id "${RUN_ID}" \
  --model nnunet --stage stage2 --trainer ${TR} --plans ${PLAN} \
  --config-name 3d_fullres --label-scheme cascade --channels 6 \
  --description "ROI cascade stage-2, LGE + 5 OOF probability channels, merged to full space" \
  --checkpoints "\$D0" --checkpoint best --probabilities "\$PRED"

# Post gates.
python scripts/apply_postprocess.py \
  --pred-dir "\$PRED_FULL" --out-dir "\${PRED_FULL}_pp" \
  --run-id "${RUN_ID}_pp" --parent-run "${RUN_ID}" \
  --splits "\$SPLITS" \
  --tag gated

python scripts/evaluate_nnunet_cv.py \
  --pred-dir "\${PRED_FULL}_pp" --gt-root "\$GT" --splits "\$SPLITS" \
  --out "\$OUTDIR/${RUN_ID}_pp.json" --run-id "${RUN_ID}_pp" \
  --model nnunet --stage stage2 --trainer ${TR} --plans ${PLAN} \
  --config-name 3d_fullres --label-scheme cascade --channels 6 \
  --description "ROI cascade stage-2, post gates" \
  --checkpoints "\$D0" --checkpoint best --probabilities "\$PRED" --tag gated

echo "== results =="
ls -l "\$OUTDIR"/${RUN_ID}*.json
EOF

mkdir -p "$FIR_BASE/logs"
submit_job "$SBATCH_FILE"
