#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the exact GPU build environment that works with the notebook.
# Usage:
#   bash bootstrap_lob_env.sh
#
# This script expects to run on a GPU node where environment modules are available.
# It loads a modern GCC and CUDA toolchain, then installs the CUDA extensions
# that otherwise fail with the system GCC 8 / missing nvcc setup.

module purge
module load gcc/13.3.1-p20240614
module load cuda/12.4.0

export CUDA_HOME=/opt/packages/cuda/v12.4.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Force extension builds to use GCC 13 instead of the system compiler.
export CC=/opt/packages/gcc/v13.3.1-p20240614/b2rm/bin/gcc
export CXX=/opt/packages/gcc/v13.3.1-p20240614/b2rm/bin/g++
export CUDAHOSTCXX="$CXX"

# Keep builds small and stable on shared nodes.
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1

# If you want a clean env from scratch, create it from environment-lob.yml first.
# The notebook currently expects the env to be named `lob`.
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in this shell. Load your conda setup first."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lob

python -m pip install --upgrade pip setuptools wheel ninja einops packaging wandb ipywidgets
python -m pip uninstall -y causal-conv1d mamba-ssm || true
python -m pip install --no-build-isolation --no-cache-dir --no-binary=:all: causal-conv1d==1.6.1
python -m pip install --no-build-isolation --no-cache-dir --no-binary=:all: mamba-ssm==2.3.1

python - <<'PY'
import torch
import causal_conv1d
import mamba_ssm
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("causal_conv1d:", causal_conv1d.__version__)
print("mamba_ssm:", mamba_ssm.__version__)
PY

echo "Environment bootstrap complete. Restart the notebook kernel before rerunning the mamba cells."
