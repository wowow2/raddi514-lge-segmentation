#!/usr/bin/env bash
# CPU-only polar evaluation on Fir: OOF merge + gate + score, all 5 folds.
# Runs where the data already lives (polar_data/) so only the tiny metrics
# JSON ever needs to come back down. Use rsync (below) for that, not scp.
#
#   ./submit_eval_polar.sh                          # baseline models
#   ./submit_eval_polar.sh radius                   # radius-variant models
#   ./submit_eval_polar.sh radius --dependency=afterok:62969811
#                                                   # chain: run only if training exits 0
#                                                   # (dependency is forwarded to the
#                                                   #  inner sbatch, not this wrapper)
#
# Then locally:
#   rsync -avP aarizvi2@fir.alliancecan.ca:/scratch/aarizvi2/raddi514/polar_data/polar_cv_metrics.json var/results/
# And optionally the merged preds for the viewer (small: label maps only):
#   rsync -avP aarizvi2@fir.alliancecan.ca:/scratch/aarizvi2/raddi514/polar_data/polar_preds/ var/polar_preds/
#
# Runs sbatch --test-only first and aborts if the dry run fails.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$FIR_BASE"

REPO="${POLAR_REPO:-$RADDI_CODE}"
DATA="$FIR_BASE/polar_data"

VARIANT="${1:-base}"
TAG=""; MODEL_SUBDIR="polar_models"
case "$VARIANT" in
  base) ;;
  radius) TAG="_radius"; MODEL_SUBDIR="polar_models_radius" ;;
  *) echo "Unknown variant: $VARIANT (use 'base' or 'radius')" >&2; exit 1 ;;
esac
DEPENDENCY=""
for arg in "${@:2}"; do
  case "$arg" in
    --dependency=*) DEPENDENCY="${arg#--dependency=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done
DEPLINE=""
if [[ -n "$DEPENDENCY" ]]; then DEPLINE="#SBATCH --dependency=${DEPENDENCY}"; fi

for p in "$DATA/emidec_train" "$DATA/splits_final.json" "$DATA/501_pp" "$DATA/$MODEL_SUBDIR"; do
  [[ -e "$p" ]] || { echo "missing bundle path: $p" >&2; exit 1; }
done
missing=0
for f in 0 1 2 3 4; do
  [[ -f "$DATA/$MODEL_SUBDIR/fold_$f/checkpoint_best.pt" ]] \
    || { echo "missing checkpoint: $MODEL_SUBDIR/fold_$f" >&2; missing=1; }
done
if [[ "$missing" -ne 0 ]]; then
  if [[ -n "$DEPENDENCY" ]]; then
    echo "note: checkpoints missing now, but continuing (training job may still be running)" >&2
  else
    exit 1
  fi
fi

JOB_NAME="polar_eval${TAG}"
OUT="logs/polar_eval${TAG}_%j.out"

SBATCH_FILE="$(mktemp /tmp/submit_polar_eval_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUT}
${DEPLINE}

source "${HERE}/env.sh"
cd "\$FIR_BASE"

# polar OOF eval (${VARIANT}) | paint-over + scar gate | challenge Dice
python "${REPO}/scripts/evaluate_polar.py" --device cpu \
  --model-dir "${DATA}/${MODEL_SUBDIR}" --stage1-dir "${DATA}/501_pp" \
  --pred-out-dir "${DATA}/polar_preds${TAG}" --out "${DATA}/polar_cv_metrics${TAG}.json" \
  --gt-root "${DATA}/emidec_train" --splits "${DATA}/splits_final.json"
EOF

mkdir -p logs
submit_job "$SBATCH_FILE"
