#!/usr/bin/env bash
set -euo pipefail
CONTROL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate gear3_control
set -u
exec python -u "${CONTROL_DIR}/app.py" "$@"
