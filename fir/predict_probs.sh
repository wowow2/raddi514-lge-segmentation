#!/usr/bin/env bash
# Save nnU-Net softmax probabilities for the *validation* cases of each fold.
#
#   ./predict_probs.sh 500 --tr=nnUNetTrainer_probabilisticOversampling
#   ./predict_probs.sh 500
#   ./predict_probs.sh 501 --plans=nnUNetResEncUNetMPlans
#
# Writes <out>/fold_F/validation/<id>.nii.gz + <id>.npz, where <out> is
#   $nnUNet_results/<dataset>/probs_<plans>_<config>[_<trainer>]
# Only the fold's validation cases are predicted (100 npz total across folds,
# not 500), so the output is entirely out-of-fold and cheap to keep.
#
# The .npz stores a 4D array under the key `probabilities`, shape
# (C, X, Y, Z), transposed back to the original NIfTI orientation. Verify with
#   argmax(probabilities, 0) == nii.get_fdata()
# before relying on the axis order.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$FIR_BASE"

DATASET="${1:?Usage: predict_probs.sh <dataset-id> [--plans=X] [--config=Y] [--tr=Trainer] [--time=HH:MM:SS]}"
PLAN="nnUNetResEncUNetMPlans"
CONFIG="2d"
TR=""
TIME="02:00:00"
for arg in "${@:2}"; do
  case "$arg" in
    --plans=*) PLAN="${arg#--plans=}" ;;
    --config=*) CONFIG="${arg#--config=}" ;;
    --tr=*) TR="${arg#--tr=}" ;;
    --time=*) TIME="${arg#--time=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

NAME="$(resolve_ds "$DATASET")"
SUF=""
TR_FLAG=""
if [[ -n "$TR" ]]; then SUF="_${TR}"; TR_FLAG="-tr $TR"; fi
OUT="$nnUNet_results/$NAME/probs_${PLAN}_${CONFIG}${SUF}"
INPUT_ROOT="$FIR_BASE/probs_inputs/$NAME"

# Splits live with the preprocessed data; fall back to the polar bundle copy.
SPLITS="$nnUNet_preprocessed/$NAME/splits_final.json"
[[ -f "$SPLITS" ]] || SPLITS="$FIR_BASE/polar_data/splits_final.json"
[[ -f "$SPLITS" ]] || { echo "missing splits_final.json for $NAME" >&2; exit 1; }

# Build one input folder per fold containing symlinks to that fold's val cases.
python - "$SPLITS" "$nnUNet_raw/$NAME/imagesTr" "$INPUT_ROOT" <<'PY'
import glob
import json
import os
import sys

splits_path, src, root = sys.argv[1:4]
with open(splits_path) as f:
    splits = json.load(f)
for fold, split in enumerate(splits):
    dst_dir = os.path.join(root, f"fold_{fold}")
    os.makedirs(dst_dir, exist_ok=True)
    for nid in split["val"]:
        for p in glob.glob(os.path.join(src, f"{nid}_*.nii.gz")):
            dst = os.path.join(dst_dir, os.path.basename(p))
            if not os.path.exists(dst):
                os.symlink(p, dst)
print(f"symlinked validation inputs under {root}")
PY

SBATCH_FILE="$(mktemp /tmp/probs_${DATASET}_XXXX.sh)"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=${TIME}
#SBATCH --job-name=probs_${DATASET}
#SBATCH --output=logs/probs_${DATASET}_%j.out

source "${HERE}/env.sh"
cd "\$FIR_BASE"
for f in 0 1 2 3 4; do
  mkdir -p "${OUT}/fold_\$f/validation"
  nnUNetv2_predict -d ${DATASET} -i "${INPUT_ROOT}/fold_\$f" \\
    -o "${OUT}/fold_\$f/validation" -f \$f -c ${CONFIG} -p ${PLAN} \\
    ${TR_FLAG} -chk checkpoint_best.pth --save_probabilities --disable_progress_bar
done
EOF

mkdir -p logs
sbatch --test-only "$SBATCH_FILE"
sbatch "$SBATCH_FILE"
squeue -u "$USER"
