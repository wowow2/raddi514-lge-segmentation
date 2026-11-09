#!/usr/bin/env bash
# Submit the ScarNet 5-fold training array on Nibi.
#   ./submit_scarnet.sh
#   ./submit_scarnet.sh --epochs=150 --batch-size=4 --time=10:00:00
#   ./submit_scarnet.sh --sam-input-size=128 --batch-size=16
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$RADDI_CODE"

EPOCHS=150
BATCH=4
TIME="10:00:00"
SAM_INPUT=1024
UNFREEZE=1
RESUME=""
for arg in "$@"; do
  case "$arg" in
    --epochs=*) EPOCHS="${arg#--epochs=}" ;;
    --batch-size=*) BATCH="${arg#--batch-size=}" ;;
    --time=*) TIME="${arg#--time=}" ;;
    --sam-input-size=*) SAM_INPUT="${arg#--sam-input-size=}" ;;
    --unfreeze-last-n=*) UNFREEZE="${arg#--unfreeze-last-n=}" ;;
    --resume) RESUME="--resume" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

DATA="$NIBI_BASE/polar_data/emidec_train"
if [[ -f "$NIBI_BASE/polar_data/splits_final.json" ]]; then
  SPLITS="$NIBI_BASE/polar_data/splits_final.json"
else
  SPLITS="$nnUNet_raw/Dataset500_EMIDEC/splits_final.json"
fi
PRIOR="$nnUNet_results/Dataset500_EMIDEC/probs_nnUNetResEncUNetMPlans_2d_nnUNetTrainer_probabilisticOversampling"
MEDSAM="$NIBI_BASE/checkpoints/medsam_vit_b.pth"
SAVE_DIR="$NIBI_BASE/scarnet/checkpoints"

for path in "$RADDI_CODE" "$DATA" "$SPLITS" "$PRIOR" "$MEDSAM"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
if ! python -c "import segment_anything" >/dev/null 2>&1; then
  echo "segment_anything is not installed in $NIBI_VENV" >&2
  echo "  pip install git+https://github.com/facebookresearch/segment-anything.git" >&2
  exit 1
fi
mkdir -p "$NIBI_BASE/logs" "$SAVE_DIR"

MODEL_PARAMS="{\"num_classes\":5,\"medsam_ckpt\":\"${MEDSAM}\",\"unfreeze_last_n\":${UNFREEZE},\"sam_input_size\":${SAM_INPUT}}"
TRAIN_SBATCH="$(mktemp /tmp/submit_scarnet_nibi_XXXX.sh)"
cat > "$TRAIN_SBATCH" <<EOF
#!/bin/bash
#SBATCH --account=${NIBI_ACCOUNT}
#SBATCH --partition=${NIBI_PARTITION}
#SBATCH --gpus-per-node=${NIBI_GPU}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --job-name=scarnet_fold
#SBATCH --output=${NIBI_BASE}/logs/scarnet_train_%a_%j.out
#SBATCH --array=0-4
set -euo pipefail
source "${HERE}/env.sh"
cd "${RADDI_CODE}"

python "${RADDI_CODE}/scripts/train_scarnet.py" \\
  --fold "\$SLURM_ARRAY_TASK_ID" \\
  --epochs ${EPOCHS} \\
  --batch-size ${BATCH} \\
  --grad-accum 1 \\
  --lr 1e-3 \\
  --lr-encoder 1e-5 \\
  --out-size 128 \\
  --device cuda \\
  --num-workers 4 \\
  --model-class src.scarnet.model:ScarNet \\
  --model-params '${MODEL_PARAMS}' \\
  --pred-dir "${PRIOR}" \\
  --gt-root "${DATA}" \\
  --splits "${SPLITS}" \\
  --save-dir "${SAVE_DIR}" \\
  --select-metric lesion \\
  --tag scarnet \\
  --description "ScarNet Nibi fold \${SLURM_ARRAY_TASK_ID}" \\
  --no-metadata ${RESUME}
EOF

sbatch --test-only "$TRAIN_SBATCH" || exit 1
TRAIN_ID="$(sbatch --parsable "$TRAIN_SBATCH" | cut -d';' -f1)"
echo "Submitted ScarNet Nibi training array: $TRAIN_ID"
squeue -u "$USER"
