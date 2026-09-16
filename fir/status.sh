#!/usr/bin/env bash
# One-shot health check for training arrays on Fir.
# Discovers every Dataset*/nnUNetTrainer* in nnUNet_results — no hardcoded IDs.
#
#   ./status.sh          # everything
#   ./status.sh 501      # one dataset (numeric ID, resolved via nnUNet_raw)
#   ./status.sh 500 501
#
# Shows: queue, project usage, per-fold latest epoch / epoch time /
# pseudo dice, and checkpoint_best presence. Replaces the tail-grep loop.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

echo "=== queue ==="
squeue -u "$USER"
echo
echo "=== project usage (kbytes used; 0 limit = uncapped) ==="
lfs quota -u "$USER" "$FIR_BASE" 2>/dev/null | tail -n +3 \
  || df -h "$FIR_BASE" | tail -1
echo

fold_status() { # $1 = trainer dir, $2 = label
  local ddir="$1"
  echo "=== $2 ==="
  for f in 0 1 2 3 4; do
    local log="" best="best:no" epoch="-" etime="-" dice="-"
    log=$(ls -t "$ddir/fold_$f"/training_log_*.txt 2>/dev/null | head -1 || true)
    [[ -f "$ddir/fold_$f/checkpoint_best.pth" ]] && best="best:yes"
    if [[ -n "$log" ]]; then
      epoch=$(grep -oE 'Epoch [0-9]+' "$log" | tail -1 || true)
      etime=$(grep -oE 'Epoch time: [0-9.]+ s' "$log" | tail -1 || true)
      dice=$(grep -oE 'Pseudo dice \[.*\]' "$log" | tail -1 || true)
    else
      epoch="no log yet"
    fi
    echo "  fold $f: $epoch | $etime | $dice | $best"
  done
}

if [[ $# -gt 0 ]]; then
  for id in "$@"; do
    name="$(resolve_ds "$id")"
    for tdir in "$nnUNet_results/$name"/nnUNetTrainer__*/; do
      [[ -d "$tdir" ]] || continue
      fold_status "${tdir%/}" "$name / $(basename "${tdir%/}")"
    done
  done
else
  found=0
  for tdir in "$nnUNet_results"/Dataset*/nnUNetTrainer__*/; do
    [[ -d "$tdir" ]] || continue
    found=1
    rel="${tdir#"$nnUNet_results"/}"
    fold_status "${tdir%/}" "${rel%/}"
  done
  [[ "$found" -eq 0 ]] && echo "(no trained datasets in nnUNet_results yet)"
fi

echo
echo "=== predictions (expect 500 = 100 cases x 5 folds when done) ==="
found_p=0
for pdir in "$nnUNet_results"/Dataset*/preds_*/; do
  [[ -d "$pdir" ]] || continue
  found_p=1
  n=$(find "$pdir" -name '*.nii.gz' 2>/dev/null | wc -l)
  echo "  ${pdir#"$nnUNet_results"/}: $n / 500 files"
done
[[ "$found_p" -eq 0 ]] && echo "  (none yet)"
