#!/usr/bin/env bash
set -euo pipefail
CONTROL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
mkdir -p "${CONTROL_DIR}/runtime"
if ! conda env list --json | python -c 'import json,sys; sys.exit(not any(p.endswith("/gear3_control") for p in json.load(sys.stdin)["envs"]))'; then
  if command -v mamba >/dev/null 2>&1; then
    mamba env create -y -f "${CONTROL_DIR}/environment.yml"
  else
    conda env create -f "${CONTROL_DIR}/environment.yml"
  fi
fi
set +u
conda activate gear3_control
set -u
python -m pip install -r "${CONTROL_DIR}/requirements.txt"
# Only the native SDK is used; its Open3D/OpenCV/GUI demo dependencies are unnecessary.
python -m pip install --no-deps pyorbbecsdk2==2.1.2
python -m pip install --no-deps -e "${CONTROL_DIR}/vendor/libgx7"
python -m pip freeze > "${CONTROL_DIR}/runtime/pip-freeze.txt"
printf 'Ready: bash %s/run.sh\n' "${CONTROL_DIR}"
