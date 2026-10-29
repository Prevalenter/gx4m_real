#!/usr/bin/env bash
# Install only. This script deliberately does not run inference or GPU tests.
set -euo pipefail
PERCEPTION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${PERCEPTION_DIR}/runtime/logs" "${PERCEPTION_DIR}/runtime/downloads"
exec > >(tee -a "${PERCEPTION_DIR}/runtime/logs/setup.log") 2>&1
rm -f "${PERCEPTION_DIR}/runtime/installation.json"

if ! conda env list --json | python -c 'import json,sys; sys.exit(not any(p.endswith("/foundationpose") for p in json.load(sys.stdin)["envs"]))'; then
  if command -v mamba >/dev/null 2>&1; then
    mamba env create -y -f "${PERCEPTION_DIR}/environment.yml"
  else
    conda env create -y -f "${PERCEPTION_DIR}/environment.yml"
  fi
fi
source "${PERCEPTION_DIR}/activate.sh"
conda env config vars set -n foundationpose \
  CUDA_HOME="${CONDA_PREFIX}" CUDA_PATH="${CONDA_PREFIX}" \
  CUDAHOSTCXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" \
  CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include" \
  CMAKE_PREFIX_PATH="${CONDA_PREFIX}" TORCH_CUDA_ARCH_LIST=8.6 \
  MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  TORCH_EXTENSIONS_DIR="${CONDA_PREFIX}/var/torch_extensions"
set +u  # Conda's compiler activation hooks reference optional unset variables.
conda deactivate
conda activate foundationpose
set -u

# CUDA compilation targets 3090 explicitly, without probing or using the GPUs.
export CUDA_VISIBLE_DEVICES=""
export FORCE_CUDA=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL="${FOUNDATIONPOSE_PIP_INDEX_URL:-https://pypi.org/simple}"

install_python() {
  if command -v uv >/dev/null 2>&1; then
    if [[ -f "${PERCEPTION_DIR}/runtime/python-downloads.json" ]]; then
      uv pip install --python "${CONDA_PREFIX}/bin/python" --offline --no-build-isolation \
        --find-links "${PERCEPTION_DIR}/runtime/wheels" "$@"
    else
      uv pip install --python "${CONDA_PREFIX}/bin/python" --index-url "${PIP_INDEX_URL}" "$@"
    fi
  else
    python -m pip install --find-links "${PERCEPTION_DIR}/runtime/wheels" "$@"
  fi
}

download_source() {
  local name="$1" repo="$2" revision="$3"
  local source_dir="${PERCEPTION_DIR}/runtime/${name}"
  local archive="${PERCEPTION_DIR}/runtime/downloads/${name}.tar.gz"
  if [[ -f "${source_dir}/.source-revision" ]]; then
    [[ "$(cat "${source_dir}/.source-revision")" == "${revision}" ]] || {
      echo "Source revision mismatch: ${source_dir}" >&2; return 1;
    }
    return
  fi
  if [[ ! -f "${archive}" ]] || ! tar -tzf "${archive}" >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 \
      "https://codeload.github.com/${repo}/tar.gz/${revision}" -o "${archive}.part"
    mv "${archive}.part" "${archive}"
  fi
  mkdir -p "${source_dir}"
  tar -xzf "${archive}" --strip-components=1 -C "${source_dir}"
  printf '%s\n' "${revision}" > "${source_dir}/.source-revision"
}

download_source FoundationPose NVlabs/FoundationPose a1b694b83e633c2cb6115b9063d940a687759392
download_source pytorch3d facebookresearch/pytorch3d 33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
download_source nvdiffrast NVlabs/nvdiffrast 729261dc64c4241ea36efda84fbf532cc8b425b8

mkdir -p "${PERCEPTION_DIR}/runtime/wheels"
TORCH_WHEEL="${PERCEPTION_DIR}/runtime/wheels/torch-2.6.0-cp311-cp311-manylinux1_x86_64.whl"
if [[ -f "${TORCH_WHEEL}" ]]; then
  # PyPI's Linux torch 2.6.0 wheel is also the CUDA 12.4 build.
  install_python "${TORCH_WHEEL}" torchvision==0.21.0
else
  python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --find-links "${PERCEPTION_DIR}/runtime/wheels" \
    --index-url https://download.pytorch.org/whl/cu124
fi
install_python -r "${PERCEPTION_DIR}/requirements.txt"
python -m pip install -r "${PERCEPTION_DIR}/viser-requirements.txt"
python -m pip install --no-build-isolation "${PERCEPTION_DIR}/runtime/pytorch3d"
# Upstream clears TORCH_CUDA_ARCH_LIST unconditionally. Preserve our explicit
# target so JIT compilation works without accessing any GPU.
python - "${PERCEPTION_DIR}/runtime/nvdiffrast/nvdiffrast/torch/ops.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
original = "os.environ['TORCH_CUDA_ARCH_LIST'] = ''"
patched = "os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '')"
source = path.read_text()
if source.count(original) == 1:
    path.write_text(source.replace(original, patched))
elif patched not in source:
    raise RuntimeError('Unexpected nvdiffrast source: cannot apply architecture patch')
PY
python -m pip install --no-build-isolation "${PERCEPTION_DIR}/runtime/nvdiffrast"
# Compile/load the shared library only: no rasterizer, CUDA context or kernels.
python -c 'from nvdiffrast.torch.ops import _get_plugin; _get_plugin(gl=False)'

cmake -S "${PERCEPTION_DIR}/runtime/FoundationPose/mycpp" \
  -B "${PERCEPTION_DIR}/runtime/FoundationPose/mycpp/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DPython_EXECUTABLE="${CONDA_PREFIX}/bin/python" \
  -DPython3_EXECUTABLE="${CONDA_PREFIX}/bin/python" \
  -DPYBIND11_PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
cmake --build "${PERCEPTION_DIR}/runtime/FoundationPose/mycpp/build" --parallel 4
python - "${PERCEPTION_DIR}" <<'PY'
from pathlib import Path
import sys
import sysconfig
root = Path(sys.argv[1]) / 'runtime' / 'FoundationPose'
(Path(sysconfig.get_path('purelib')) / 'foundationpose-local.pth').write_text(
    f'{root}\n{root / "mycpp" / "build"}\n'
)
PY

conda env export -n foundationpose > "${PERCEPTION_DIR}/runtime/environment.resolved.yml"
conda list -n foundationpose --explicit > "${PERCEPTION_DIR}/runtime/conda-explicit.txt"
python -m pip freeze > "${PERCEPTION_DIR}/runtime/pip-freeze.txt"
sha256sum "${PERCEPTION_DIR}"/runtime/downloads/*.tar.gz > "${PERCEPTION_DIR}/runtime/source-sha256.txt"
python - "${PERCEPTION_DIR}" <<'PY'
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import sys
root = Path(sys.argv[1])
record = {
    'installed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'environment': 'foundationpose',
    'installed': True,
    'gpu_tests_run': False,
    'demo_run': False,
    'weights_downloaded': False,
    'sources': json.loads((root / 'sources.json').read_text()),
    'packages': {name: version(name) for name in (
        'torch', 'torchvision', 'numpy', 'pytorch3d', 'nvdiffrast', 'warp-lang', 'open3d'
    )},
}
(root / 'runtime/installation.json').write_text(json.dumps(record, indent=2) + '\n')
PY
printf '\nEnvironment installation complete. No GPU tests or demos were run.\n'
