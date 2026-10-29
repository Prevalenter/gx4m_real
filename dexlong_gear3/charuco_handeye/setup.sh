#!/usr/bin/env bash
set -euo pipefail
CALIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate gear3_control
set -u
python -m pip install -r "${CALIB_DIR}/requirements.txt"
python -c 'import cv2; assert hasattr(cv2.aruco, "CharucoDetector"); import viser, yourdfpy'
