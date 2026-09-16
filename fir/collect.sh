#!/usr/bin/env bash
# Verify fold checkpoints for ANY dataset ID; optionally pack results for download.
#
#   ./collect.sh 501                 # verify only (defaults: ResEncM, 2d)
#   ./collect.sh 501 --pack          # verify + tar results dir for scp
#   ./collect.sh 502 --plans=nnUNetPlans --config=3d_fullres --pack
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

DATASET="${1:?Usage: collect.sh <dataset-id> [--pack] [--plans=X] [--config=Y]}"
PLAN="nnUNetResEncUNetMPlans"
CONFIG="2d"
PACK=0
for arg in "${@:2}"; do
  case "$arg" in
    --pack) PACK=1 ;;
    --plans=*) PLAN="${arg#--plans=}" ;;
    --config=*) CONFIG="${arg#--config=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

NAME="$(resolve_ds "$DATASET")"
DDIR="$nnUNet_results/$NAME/nnUNetTrainer__${PLAN}__${CONFIG}"

missing=0
for f in 0 1 2 3 4; do
  if [[ -f "$DDIR/fold_$f/checkpoint_best.pth" ]]; then
    size=$(du -h "$DDIR/fold_$f/checkpoint_best.pth" | cut -f1)
    final="final:no"; [[ -f "$DDIR/fold_$f/checkpoint_final.pth" ]] && final="final:yes"
    echo "fold $f: best OK ($size) $final"
  else
    echo "fold $f: MISSING checkpoint_best.pth"
    missing=$((missing + 1))
  fi
done

if [[ "$PACK" -eq 1 ]]; then
  if [[ "$missing" -gt 0 ]]; then echo "Refusing to pack with missing folds." >&2; exit 1; fi
  echo "results size: $(du -sh "$nnUNet_results/$NAME" | cut -f1)"
  tar czf "$FIR_BASE/${NAME}_results.tar.gz" -C "$nnUNet_results" "$NAME"
  ls -lh "$FIR_BASE/${NAME}_results.tar.gz"
  echo "download with: scp aarizvi2@fir:${FIR_BASE}/${NAME}_results.tar.gz ."
fi
