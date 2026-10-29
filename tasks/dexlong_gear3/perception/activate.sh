#!/usr/bin/env bash
# Source this file: source tasks/dexlong_gear3/perception/activate.sh
_fp_nounset=0
if [[ $- == *u* ]]; then
  _fp_nounset=1
  set +u
fi
_fp_conda_base="$(conda info --base)"
source "${_fp_conda_base}/etc/profile.d/conda.sh"
conda activate foundationpose
if [[ "${_fp_nounset}" == 1 ]]; then
  set -u
fi
unset _fp_conda_base _fp_nounset
