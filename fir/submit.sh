#!/usr/bin/env bash
# Submit (or resume) a 5-fold training array on Fir for ANY dataset ID.
#
#   ./submit.sh 501                                # fresh 23h run (defaults: ResEncM, 2d)
#   ./submit.sh 501 --continue                     # resume from checkpoint_latest, 12h
#   ./submit.sh 502 --plans=nnUNetPlans --config=3d_fullres
#   ./submit.sh 501 --continue --time=06:00:00
#
# Runs sbatch --test-only first and aborts if the dry run fails.
# Never submit --continue while the old array is still running (checkpoint race).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$FIR_BASE"

DATASET="${1:?Usage: submit.sh <dataset-id> [--continue] [--time=HH:MM:SS] [--plans=X] [--config=Y]}"
CONTINUE_FLAG=""
TIME="23:00:00"
PLAN="nnUNetResEncUNetMPlans"
CONFIG="2d"
for arg in "${@:2}"; do
  case "$arg" in
    --continue) CONTINUE_FLAG="--c"; TIME="12:00:00" ;;
    --time=*) TIME="${arg#--time=}" ;;
    --plans=*) PLAN="${arg#--plans=}" ;;
    --config=*) CONFIG="${arg#--config=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

NAME="$(resolve_ds "$DATASET")"
JOB_NAME="nnunet_${DATASET}"
OUT="logs/train_${DATASET}_%a_%j.out"

SBATCH_FILE="$(mktemp /tmp/submit_${DATASET}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUT}
#SBATCH --array=0-4

source "${HERE}/env.sh"
cd "\$FIR_BASE"
mkdir -p logs "\$SLURM_TMPDIR/nn"
cp -r "\$FIR_BASE/emidec_nnunet/nnUNet_preprocessed" "\$SLURM_TMPDIR/nn/"
cp -r "\$FIR_BASE/emidec_nnunet/nnUNet_raw" "\$SLURM_TMPDIR/nn/"
export nnUNet_preprocessed="\$SLURM_TMPDIR/nn/nnUNet_preprocessed"
export nnUNet_raw="\$SLURM_TMPDIR/nn/nnUNet_raw"

# ${NAME} | ${PLAN} | ${CONFIG}
nnUNetv2_train ${DATASET} ${CONFIG} \$SLURM_ARRAY_TASK_ID -p ${PLAN} ${CONTINUE_FLAG}
EOF

mkdir -p logs
sbatch --test-only "$SBATCH_FILE"
sbatch "$SBATCH_FILE"
squeue -u "$USER"
