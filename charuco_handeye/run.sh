#!/usr/bin/env bash
set -euo pipefail
CALIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate gear3_control
set -u
case "${1:-}" in
  calibrate) shift; exec python -u "${CALIB_DIR}/solve_session.py" "$@" ;;
  view) shift; exec python -u "${CALIB_DIR}/view_result.py" "$@" ;;
  *) echo "Usage: bash $0 {calibrate|view} [options]" >&2; exit 2 ;;
esac
