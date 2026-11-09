#!/usr/bin/env bash
# Submit the ScarNet OOF evaluation (CPU). Requires all 5 fold checkpoints.
#   ./evaluate_scarnet.sh
#   ./evaluate_scarnet.sh --dependency=afterok:12345678
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$RADDI_CODE"

DEPENDENCY=""
for arg in "$@"; do
  case "$arg" in
    --dependency=*) DEPENDENCY="${arg#--dependency=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

DATA="$FIR_BASE/polar_data/emidec_train"
if [[ -f "$FIR_BASE/polar_data/splits_final.json" ]]; then
  SPLITS="$FIR_BASE/polar_data/splits_final.json"
else
  SPLITS="$nnUNet_raw/Dataset500_EMIDEC/splits_final.json"
fi
PRIOR="$nnUNet_results/Dataset500_EMIDEC/probs_nnUNetResEncUNetMPlans_2d_nnUNetTrainer_probabilisticOversampling"
SAVE_DIR="$FIR_BASE/scarnet/checkpoints"
PRED_OUT="$FIR_BASE/scarnet/outputs/scarnet_oof/predictions"
PROB_OUT="$FIR_BASE/scarnet/outputs/scarnet_oof/probabilities"
METRICS="$FIR_BASE/scarnet/outputs/scarnet_oof/metrics.json"

for path in "$RADDI_CODE" "$DATA" "$SPLITS" "$PRIOR"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
if ! python -c "import segment_anything" >/dev/null 2>&1; then
  echo "segment_anything is not installed in $FIR_VENV" >&2
  exit 1
fi
if [[ -z "$DEPENDENCY" ]]; then
  for f in 0 1 2 3 4; do
    [[ -f "$SAVE_DIR/fold_$f/checkpoint_best.pt" ]] \
      || { echo "Missing checkpoint: $SAVE_DIR/fold_$f" >&2; exit 1; }
  done
fi
mkdir -p "$FIR_BASE/logs" "$PRED_OUT"

DEPLINE=""
[[ -n "$DEPENDENCY" ]] && DEPLINE="#SBATCH --dependency=${DEPENDENCY}"
EVAL_SBATCH="$(mktemp /tmp/eval_scarnet_XXXX.sh)"
cat > "$EVAL_SBATCH" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=scarnet_eval
#SBATCH --output=${FIR_BASE}/logs/scarnet_eval_%j.out
${DEPLINE}
set -euo pipefail
source "${HERE}/env.sh"
cd "${RADDI_CODE}"
export OMP_NUM_THREADS=4

python "${RADDI_CODE}/scripts/evaluate_scarnet.py" \\
  --model-dir "${SAVE_DIR}" \\
  --pred-out-dir "${PRED_OUT}" \\
  --out "${METRICS}" \\
  --prob-out-dir "${PROB_OUT}" \\
  --save-probabilities \\
  --stage1-dir "${PRIOR}" \\
  --gt-root "${DATA}" \\
  --splits "${SPLITS}" \\
  --device cpu \\
  --batch-size 8 \\
  --run-id scarnet_oof \\
  --description "ScarNet OOF eval" \\
  --tag scarnet
EOF

sbatch --test-only "$EVAL_SBATCH" || exit 1
JOB_ID="$(sbatch --parsable "$EVAL_SBATCH" | cut -d';' -f1)"
echo "Submitted ScarNet eval: $JOB_ID"
squeue -u "$USER"
