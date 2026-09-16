#!/usr/bin/env bash
# fir/env.sh — shared Fir environment. Source it, don't run it:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u

export FIR_BASE=/project/def-punithak/aarizvi2/raddi514
export FIR_VENV=/scratch/aarizvi2/env_raddi514/bin/activate

export nnUNet_raw=$FIR_BASE/emidec_nnunet/nnUNet_raw
export nnUNet_preprocessed=$FIR_BASE/emidec_nnunet/nnUNet_preprocessed
export nnUNet_results=$FIR_BASE/emidec_nnunet/nnUNet_results
export nnUNet_compile=false

module load StdEnv/2023 python/3.11 cuda 2>/dev/null \
  || module load StdEnv/2023 python/3.11
# shellcheck disable=SC1091
source "$FIR_VENV"

# resolve_ds <numeric-id> — print the raw dataset dir name (e.g.
# Dataset501_EMIDEC_Regions) for a numeric nnU-Net ID, or fail if the
# match isn't unique. No hardcoded ID list; works for any future dataset.
resolve_ds() {
  local matches=( "$nnUNet_raw"/Dataset"$1"_* )
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Cannot uniquely resolve Dataset$1_* in $nnUNet_raw" >&2
    return 1
  fi
  basename "${matches[0]}"
}
