#!/usr/bin/env bash
# Single-GPU polar training on Fir. NOT an array: 5 folds x 60 epochs of a
# ~100k-param strip U-Net fits one short job.
#
#   ./submit_polar.sh 3       # fold-3 pilot (default, ~1.5h)
#   ./submit_polar.sh all     # all 5 folds sequentially (~4h)
#
# Expects the upload bundle unpacked at $FIR_BASE/polar_data/:
#   polar_data/emidec_train/  (Case_*/Images + Contours)
#   polar_data/splits_final.json
#   polar_data/501_pp/fold_*/validation/*.nii.gz
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$FIR_BASE"

FOLD="${1:-3}"
if [[ "$FOLD" == "all" ]]; then TIME="04:00:00"; else TIME="01:30:00"; fi

REPO="${POLAR_REPO:-/scratch/aarizvi2/raddi514code}"
DATA="$FIR_BASE/polar_data"
for p in "$DATA/emidec_train" "$DATA/splits_final.json" "$DATA/501_pp"; do
  [[ -e "$p" ]] || { echo "missing bundle path: $p" >&2; exit 1; }
done

JOB_NAME="polar_${FOLD}"
OUT="logs/polar_${FOLD}_%j.out"
mkdir -p logs "$DATA/polar_models"

sbatch --account=def-punithak \
  --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1 \
  --cpus-per-task=4 --mem=16G --time="$TIME" \
  --job-name="$JOB_NAME" --output="$OUT" \
  --wrap="source '$HERE/env.sh' && cd '\$FIR_BASE' && python '$REPO/scripts/train_polar.py' \
    --fold '$FOLD' --epochs 60 --batch-size 16 \
    --gt-root '$DATA/emidec_train' --splits '$DATA/splits_final.json' \
    --pred-dir '$DATA/501_pp' --save-dir '$DATA/polar_models' --num-workers 2"
