#!/usr/bin/env bash
# Submit (or resume) a 5-fold training array on Nibi for any dataset ID.
#
#   ./submit.sh 503 --config=3d_fullres
#   ./submit.sh 503 --config=3d_fullres --tr=nnUNetTrainer_probabilisticOversampling --name=503_over
#   ./submit.sh 503 --config=3d_fullres --plans=nnUNetPrimusV3SPlans --tr=nnUNet_PrimusV3S_Trainer --name=503_primus
#   ./submit.sh 503 --config=3d_fullres --continue
#
# Runs sbatch --test-only first and aborts if the dry run fails.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$NIBI_BASE"

DATASET="${1:?Usage: submit.sh <dataset-id> [--continue] [--time=HH:MM:SS] [--plans=X] [--config=Y] [--tr=Trainer] [--name=NAME] [--folds=0-4]}"
CONTINUE_FLAG=""
TR_FLAG=""
TIME="23:00:00"
PLAN="nnUNetResEncUNetMPlans"
CONFIG="2d"
NAME_OVERRIDE=""
FOLDS="0-4"
for arg in "${@:2}"; do
  case "$arg" in
    --continue) CONTINUE_FLAG="--c"; TIME="12:00:00" ;;
    --time=*) TIME="${arg#--time=}" ;;
    --plans=*) PLAN="${arg#--plans=}" ;;
    --config=*) CONFIG="${arg#--config=}" ;;
    --tr=*) TR_FLAG="-tr ${arg#--tr=}" ;;
    --name=*) NAME_OVERRIDE="${arg#--name=}" ;;
    --folds=*) FOLDS="${arg#--folds=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

NAME="$(resolve_ds "$DATASET")"
JOB_NAME="${NAME_OVERRIDE:-nnunet_${DATASET}}"
OUT="logs/train_${JOB_NAME}_%a_%j.out"

SBATCH_FILE="$(mktemp /tmp/submit_${DATASET}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=${NIBI_ACCOUNT}
#SBATCH --partition=${NIBI_PARTITION}
#SBATCH --gpus-per-node=${NIBI_GPU}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUT}
#SBATCH --array=${FOLDS}

set -euo pipefail
source "${HERE}/env.sh"
cd "\$NIBI_BASE"

mkdir -p "\$SLURM_TMPDIR/nn/nnUNet_preprocessed" "\$SLURM_TMPDIR/nn/nnUNet_raw"
cp -r "\$nnUNet_preprocessed/${NAME}" \
      "\$SLURM_TMPDIR/nn/nnUNet_preprocessed/"
cp -r "\$nnUNet_raw/${NAME}" \
      "\$SLURM_TMPDIR/nn/nnUNet_raw/"
export nnUNet_preprocessed="\$SLURM_TMPDIR/nn/nnUNet_preprocessed"
export nnUNet_raw="\$SLURM_TMPDIR/nn/nnUNet_raw"

nnUNetv2_train ${DATASET} ${CONFIG} \$SLURM_ARRAY_TASK_ID -p ${PLAN} ${TR_FLAG} ${CONTINUE_FLAG} --val_best
EOF

mkdir -p logs
submit_job "$SBATCH_FILE"
