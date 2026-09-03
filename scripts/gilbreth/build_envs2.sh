#!/bin/bash
# Build the remaining 2 envs (stardist-tf, metrics). NO `set -u`: conda's Qt activation hook
# (pulled into torch-cell by micro_sam) references unbound vars and aborts under nounset.
#   nohup bash scripts/gilbreth/build_envs2.sh > logs/build_envs2.log 2>&1 &
set -o pipefail

CONDA=/apps/external/anaconda/2025.06
export PATH="$CONDA/bin:$PATH"
source "$CONDA/etc/profile.d/conda.sh"
SCRATCH=/scratch/gilbreth/$USER
ENVS="$SCRATCH/envs"
export PIP_CACHE_DIR="$SCRATCH/.pipcache"
export CONDA_PKGS_DIRS="$SCRATCH/.condapkgs"

echo "===== stardist-tf : StarDist + TensorFlow2 ====="
conda create -y -p "$ENVS/stardist-tf" python=3.10 || echo "FAIL create stardist-tf"
conda activate "$ENVS/stardist-tf"
python -m pip install -q --upgrade pip
python -m pip install -q "tensorflow[and-cuda]==2.15.*" stardist csbdeep || echo "FAIL stardist install"
python - <<'PY' || echo "FAIL import stardist"
import tensorflow as tf, stardist
print("STARDIST_OK tf", tf.__version__, "stardist", stardist.__version__)
PY
conda deactivate

echo "===== metrics : framework-agnostic scoring ====="
conda create -y -p "$ENVS/metrics" python=3.10 numpy scipy scikit-image matplotlib pandas || echo "FAIL create metrics"
conda activate "$ENVS/metrics"
python -m pip install -q --upgrade pip
python -m pip install -q stardist monai tifffile || echo "FAIL metrics install"
python - <<'PY' || echo "FAIL import metrics"
import numpy, scipy, skimage, monai, stardist
from stardist.matching import matching
print("METRICS_OK monai", monai.__version__)
PY
conda deactivate
echo "ALL_BUILDS2_DONE"
