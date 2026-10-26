#!/usr/bin/env bash
# One-shot health check for training arrays on Fir.
# Discovers every Dataset*/nnUNetTrainer* in nnUNet_results — no hardcoded IDs.
#
#   ./status.sh          # everything
#   ./status.sh 501      # one dataset (numeric ID, resolved via nnUNet_raw)
#   ./status.sh 500 501
#   ./status.sh 503 --config=3d_fullres --tr=nnUNetTrainer_probabilisticOversampling --pack
#
# Shows: queue, project usage, per-fold latest epoch / epoch time /
# pseudo dice, and checkpoint_best presence. Replaces the tail-grep loop.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

# Model selectors reproduce the former collect.sh checkpoint/pack command.
IDS=()
PLAN="nnUNetResEncUNetMPlans"
CONFIG="2d"
TR="nnUNetTrainer"
SELECTED=0
PACK=0
for arg in "$@"; do
  case "$arg" in
    --pack) PACK=1; SELECTED=1 ;;
    --plans=*) PLAN="${arg#--plans=}"; SELECTED=1 ;;
    --config=*) CONFIG="${arg#--config=}"; SELECTED=1 ;;
    --tr=*) TR="${arg#--tr=}"; SELECTED=1 ;;
    --*) echo "Unknown arg: $arg" >&2; exit 1 ;;
    *) IDS+=("$arg") ;;
  esac
done
if [[ "$SELECTED" -eq 1 && "${#IDS[@]}" -ne 1 ]]; then
  echo "Model selection/packing requires exactly one dataset ID." >&2; exit 1
fi
MISSING=0

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
    local log="" best="best:no" final="final:no" epoch="-" etime="-" dice="-"
    log=$(ls -t "$ddir/fold_$f"/training_log_*.txt 2>/dev/null | head -1 || true)
    if [[ -f "$ddir/fold_$f/checkpoint_best.pth" ]]; then
      best="best:yes"
    else
      MISSING=$((MISSING + 1))
    fi
    [[ -f "$ddir/fold_$f/checkpoint_final.pth" ]] && final="final:yes"
    if [[ -n "$log" ]]; then
      epoch=$(grep -oE 'Epoch [0-9]+' "$log" | tail -1 || true)
      etime=$(grep -oE 'Epoch time: [0-9.]+ s' "$log" | tail -1 || true)
      dice=$(grep -oE 'Pseudo dice \[.*\]' "$log" | tail -1 || true)
    else
      epoch="no log yet"
    fi
    echo "  fold $f: $epoch | $etime | $dice | $best | $final"
  done
}

if [[ "$SELECTED" -eq 1 ]]; then
  name="$(resolve_ds "${IDS[0]}")" || exit 1
  fold_status "$nnUNet_results/$name/${TR}__${PLAN}__${CONFIG}" "$name / $TR / $PLAN / $CONFIG"
  if [[ "$PACK" -eq 1 ]]; then
    [[ "$MISSING" -eq 0 ]] || { echo "Refusing to pack with missing folds." >&2; exit 1; }
    tar czf "$FIR_BASE/${name}_results.tar.gz" -C "$nnUNet_results" "$name" || exit 1
    echo "download with: scp fir:${FIR_BASE}/${name}_results.tar.gz ."
  fi
elif [[ "${#IDS[@]}" -gt 0 ]]; then
  for id in "${IDS[@]}"; do
    name="$(resolve_ds "$id")" || exit 1
    for tdir in "$nnUNet_results/$name"/*__*__*; do
      [[ -d "$tdir" ]] || continue
      fold_status "$tdir" "$name / $(basename "$tdir")"
    done
  done
else
  found=0
  for tdir in "$nnUNet_results"/Dataset*/*__*__*; do
    [[ -d "$tdir" ]] || continue
    found=1
    fold_status "$tdir" "${tdir#"$nnUNet_results"/}"
  done
  [[ "$found" -eq 0 ]] && echo "(no trained datasets in nnUNet_results yet)"
fi

echo
echo "=== prediction and probability files (counts depend on full vs OOF mode) ==="
found_p=0
for pdir in "$nnUNet_results"/Dataset*/{preds,probs}_*/; do
  [[ -d "$pdir" ]] || continue
  found_p=1
  n=$(find "$pdir" -name '*.nii.gz' 2>/dev/null | wc -l)
  echo "  ${pdir#"$nnUNet_results"/}: $n prediction files"
done
if [[ "$found_p" -eq 0 ]]; then echo "  (none yet)"; fi
