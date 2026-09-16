#!/usr/bin/env bash
# Run 5-fold predictions on Fir for ANY dataset ID, then pack for download.
# Output layout matches scripts/evaluate_nnunet_cv.py:
#   preds_<plans>_<config>/fold_<F>/validation/<nnunet_id>.nii.gz
#
#   ./predict.sh 501            # submit short GPU job (defaults: ResEncM, 2d)
#   ./predict.sh 501 --cpu      # same, but CPU-only job (~1-2h instead of minutes)
#   ./predict.sh 501 --pack     # after the job finishes: tar preds, print scp line
#   ./predict.sh 500 --plans=nnUNetPlans --config=2d
#
# Uses checkpoint_best.pth (nnUNetv2_predict default) — the correct checkpoint
# when a run stopped at the time wall without checkpoint_final.pth.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$FIR_BASE"

DATASET="${1:?Usage: predict.sh <dataset-id> [--pack] [--cpu] [--plans=X] [--config=Y]}"
PLAN="nnUNetResEncUNetMPlans"
CONFIG="2d"
PACK=0
RES_LINES="#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00"
DEVICE_ARGS=""
EXTRA_LINES=""
for arg in "${@:2}"; do
  case "$arg" in
    --pack) PACK=1 ;;
    --cpu)
      RES_LINES="#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00"
      DEVICE_ARGS="--device cpu"
      EXTRA_LINES='export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK' ;;
    --plans=*) PLAN="${arg#--plans=}" ;;
    --config=*) CONFIG="${arg#--config=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

NAME="$(resolve_ds "$DATASET")"
PREDROOT="$nnUNet_results/$NAME/preds_${PLAN}_${CONFIG}"

if [[ "$PACK" -eq 1 ]]; then
  n=$(find "$PREDROOT" -name '*.nii.gz' 2>/dev/null | wc -l)
  echo "prediction files: $n (expect 500 = 100 cases x 5 folds)"
  tar czf "$FIR_BASE/${NAME}_preds.tar.gz" -C "$PREDROOT" .
  ls -lh "$FIR_BASE/${NAME}_preds.tar.gz"
  echo "download with: scp aarizvi2@fir:${FIR_BASE}/${NAME}_preds.tar.gz ."
  exit 0
fi

SBATCH_FILE="$(mktemp /tmp/predict_${DATASET}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
${RES_LINES}
#SBATCH --job-name=pred_${DATASET}
#SBATCH --output=logs/predict_${DATASET}_%j.out

source "${HERE}/env.sh"
cd "\$FIR_BASE"
mkdir -p logs
${EXTRA_LINES}

# ${NAME} | ${PLAN} | ${CONFIG} | checkpoint_best (default)
for f in 0 1 2 3 4; do
  mkdir -p "${PREDROOT}/fold_\$f/validation"
  nnUNetv2_predict -d ${DATASET} -i "\$nnUNet_raw/${NAME}/imagesTr" \\
    -o "${PREDROOT}/fold_\$f/validation" -f \$f -c ${CONFIG} -p ${PLAN} \\
    ${DEVICE_ARGS} --continue_prediction --disable_progress_bar
done
EOF

mkdir -p logs
sbatch --test-only "$SBATCH_FILE"
sbatch "$SBATCH_FILE"
squeue -u "$USER"
