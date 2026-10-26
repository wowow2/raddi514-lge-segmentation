#!/usr/bin/env bash
# BEST-checkpoint prediction, locally or as a Slurm job.
#   ./predict.sh 501 --cpu --parts=4
#   ./predict.sh 500 --oof --save-probabilities --tr=nnUNetTrainer_probabilisticOversampling
#   ./predict.sh 501 --local --oof
#   ./predict.sh 501 --pack
# --save-probabilities implies --oof and preserves the former probs_* paths.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$FIR_BASE"

DATASET="${1:?Usage: predict.sh <dataset-id> [--oof] [--save-probabilities] [--cpu|--local] [--parts=N] [--pack] [--plans=X] [--config=Y] [--tr=Trainer] [--time=HH:MM:SS]}"
PLAN="nnUNetResEncUNetMPlans"
CONFIG="2d"
TR=""
PACK=0
LOCAL=0
PARTS=1
CPU=0
OOF=0
SAVE_PROBS=0
TIME=""
set -- "${@:2}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pack) PACK=1; shift ;;
    --local) LOCAL=1; CPU=1; shift ;;
    --cpu) CPU=1; shift ;;
    --oof) OOF=1; shift ;;
    --save-probabilities) SAVE_PROBS=1; OOF=1; shift ;;
    --parts) PARTS="${2:?--parts needs a value}"; shift 2 ;;
    --parts=*) PARTS="${1#--parts=}"; shift ;;
    --plans=*) PLAN="${1#--plans=}"; shift ;;
    --config=*) CONFIG="${1#--config=}"; shift ;;
    --tr=*) TR="${1#--tr=}"; shift ;;
    --time=*) TIME="${1#--time=}"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
[[ "$PARTS" =~ ^[1-9][0-9]*$ ]] || { echo "--parts must be a positive integer" >&2; exit 1; }

RES_LINES="#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G"
DEVICE_ARGS=""
EXTRA_LINES=""
if [[ "$CPU" -eq 1 ]]; then
  CPUS=$((PARTS * 4)); [[ "$CPUS" -lt 8 ]] && CPUS=8
  MEM=$((PARTS * 8)); [[ "$MEM" -lt 32 ]] && MEM=32
  RES_LINES="#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}G"
  TIME="${TIME:-08:00:00}"
  DEVICE_ARGS="-device cpu"
  EXTRA_LINES='export OMP_NUM_THREADS=4'
  if [[ "$LOCAL" -eq 1 ]]; then EXTRA_LINES='export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"'; fi
elif [[ "$SAVE_PROBS" -eq 1 ]]; then
  TIME="${TIME:-02:00:00}"
else
  TIME="${TIME:-04:00:00}"
fi

NAME="$(resolve_ds "$DATASET")"
TSUFFIX=""
TR_FLAG=""
if [[ -n "$TR" ]]; then TSUFFIX="_${TR}"; TR_FLAG="-tr $TR"; fi
PREFIX="preds"
SUFFIX=""
PROB_ARGS=""
CONTINUE_ARGS="--continue_prediction"
if [[ "$SAVE_PROBS" -eq 1 ]]; then
  PREFIX="probs"; PROB_ARGS="--save_probabilities"; CONTINUE_ARGS=""
elif [[ "$OOF" -eq 1 ]]; then
  SUFFIX="_oof"
fi
PREDROOT="$nnUNet_results/$NAME/${PREFIX}_${PLAN}_${CONFIG}${TSUFFIX}${SUFFIX}"

if [[ "$PACK" -eq 1 && "$LOCAL" -eq 0 ]]; then
  TAG="${NAME}__${TR:-nnUNetTrainer}__${PLAN}__${CONFIG}_${PREFIX}${SUFFIX}"
  tar czf "$FIR_BASE/${TAG}.tar.gz" -C "$PREDROOT" .
  echo "download with: scp fir:${FIR_BASE}/${TAG}.tar.gz ."
  exit 0
fi

SBATCH_FILE="$(mktemp /tmp/predict_${DATASET}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
${RES_LINES}
#SBATCH --time=${TIME}
#SBATCH --job-name=pred_${DATASET}
#SBATCH --output=logs/predict_${DATASET}_%j.out

set -euo pipefail
source "${HERE}/env.sh"
cd "\$FIR_BASE"
${EXTRA_LINES}
INPUT_ROOT="\$nnUNet_raw/${NAME}/imagesTr"
if [[ ${OOF} -eq 1 ]]; then
  SPLITS="\$nnUNet_preprocessed/${NAME}/splits_final.json"
  [[ -f "\$SPLITS" ]] || SPLITS="\$FIR_BASE/polar_data/splits_final.json"
  INPUT_ROOT=\$(mktemp -d "\${SLURM_TMPDIR:-\${TMPDIR:-/tmp}}/raddi_oof_${DATASET}_XXXXXX")
  python - "\$SPLITS" "\$nnUNet_raw/${NAME}/imagesTr" "\$INPUT_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
split_path, source, target = map(Path, sys.argv[1:])
splits = json.loads(split_path.read_text())
expected = '1c67734e21eda69fd2a1e4de053c3923e72997d554f5d071b0a7ce1a16c7bb8b'
digest = hashlib.sha256(json.dumps(splits, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if digest != expected:
    raise ValueError(f'Original split identity differs: {split_path}')
for fold, split in enumerate(splits):
    folder = target / f'fold_{fold}'
    folder.mkdir()
    for case in split['val']:
        channels = sorted(source.glob(f'{case}_*.nii.gz'))
        if not channels:
            raise FileNotFoundError(f'No input channels for {case}')
        for channel in channels:
            (folder / channel.name).symlink_to(channel.resolve())
PY
fi
for f in 0 1 2 3 4; do
  INPUT="\$INPUT_ROOT"
  if [[ ${OOF} -eq 1 ]]; then INPUT="\$INPUT_ROOT/fold_\$f"; fi
  mkdir -p "${PREDROOT}/fold_\$f/validation"
  pids=()
  for p in \$(seq 0 $((PARTS - 1))); do
    nnUNetv2_predict -d ${DATASET} -i "\$INPUT" \\
      -o "${PREDROOT}/fold_\$f/validation" -f \$f -c ${CONFIG} -p ${PLAN} \\
      ${TR_FLAG} -chk checkpoint_best.pth ${DEVICE_ARGS} ${CONTINUE_ARGS} ${PROB_ARGS} --disable_progress_bar \\
      -num_parts ${PARTS} -part_id \$p &
    pids+=(\$!)
  done
  for pid in "\${pids[@]}"; do wait "\$pid"; done
done
EOF

mkdir -p logs
if [[ "$LOCAL" -eq 1 ]]; then
  bash "$SBATCH_FILE"
else
  submit_job "$SBATCH_FILE"
fi
