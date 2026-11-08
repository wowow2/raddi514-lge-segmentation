#!/usr/bin/env bash
# nibi/env.sh — shared Nibi environment. Source it, don't run it:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u

export NIBI_BASE="${NIBI_BASE:-/scratch/aarizvi2/raddi514}"
export NIBI_VENV="${NIBI_VENV:-/scratch/aarizvi2/env_raddi514/bin/activate}"
export RADDI_CODE="${RADDI_CODE:-/scratch/aarizvi2/raddi514code}"

export nnUNet_raw=$NIBI_BASE/emidec_nnunet/nnUNet_raw
export nnUNet_preprocessed=$NIBI_BASE/emidec_nnunet/nnUNet_preprocessed
export nnUNet_results=$NIBI_BASE/emidec_nnunet/nnUNet_results
export nnUNet_compile=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export NIBI_ACCOUNT="${NIBI_ACCOUNT:-def-punithak_gpu}"
export NIBI_PARTITION="${NIBI_PARTITION:-gpubase_bygpu_b3}"
export NIBI_GPU="${NIBI_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"

module load StdEnv/2023 python/3.11 cuda/13.2 2>/dev/null \
  || module load StdEnv/2023 python/3.11
# shellcheck disable=SC1091
source "$NIBI_VENV"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "nibi env: NIBI_BASE=$NIBI_BASE" >&2
fi

resolve_ds() {
  local matches=( "$nnUNet_raw"/Dataset"$1"_* )
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Cannot uniquely resolve Dataset$1_* in $nnUNet_raw" >&2
    return 1
  fi
  basename "${matches[0]}"
}

submit_job() {
  sbatch --test-only "$1" || return
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry run: $1" >&2
    return 0
  fi
  sbatch "$1" || return
  squeue -u "$USER"
}
