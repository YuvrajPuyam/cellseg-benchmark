#!/bin/bash
# Build the 3 conda envs for cajal on Gilbreth. Paths hard-coded (module/$RCAC_SCRATCH are NOT
# set in non-interactive shells). Run detached:
#   nohup bash scripts/gilbreth/build_envs.sh > logs/build_envs.log 2>&1 &
# Markers in the log: TORCHCELL_OK / STARDIST_OK / METRICS_OK / ALL_BUILDS_DONE (+ FAIL ... lines).
set -uo pipefail

CONDA=/apps/external/anaconda/2025.06
export PATH="$CONDA/bin:$PATH"
source "$CONDA/etc/profile.d/conda.sh"

SCRATCH=/scratch/gilbreth/$USER
ENVS="$SCRATCH/envs"
mkdir -p "$ENVS"
export PIP_CACHE_DIR="$SCRATCH/.pipcache"        # keep pip cache off the 25 GB home
export CONDA_PKGS_DIRS="$SCRATCH/.condapkgs"      # and the conda pkg cache too

echo "===== [1/3] torch-cell : cellpose (+micro_sam attempt), PyTorch/CUDA12.1 ====="
conda create -y -p "$ENVS/torch-cell" python=3.10 || echo "FAIL create torch-cell"
conda activate "$ENVS/torch-cell"
python -m pip install -q --upgrade pip
python -m pip install -q torch==2.5.* torchvision --index-url https://download.pytorch.org/whl/cu121 || echo "FAIL torch"
python -m pip install -q cellpose || echo "FAIL cellpose"
python -m pip install -q micro_sam || echo "FAIL micro_sam (optional; pip pkg may be unavailable)"
python - <<'PY' || echo "FAIL import torch-cell"
import torch, cellpose
print("TORCHCELL_OK torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "cellpose", getattr(cellpose, "version", "?"))
try:
    import micro_sam; print("  micro_sam present", getattr(micro_sam, "__version__", "?"))
except Exception as e:
    print("  micro_sam absent:", type(e).__name__)
PY
conda deactivate

echo "===== [2/3] stardist-tf : StarDist + TensorFlow2 (separate env) ====="
conda create -y -p "$ENVS/stardist-tf" python=3.10 || echo "FAIL create stardist-tf"
conda activate "$ENVS/stardist-tf"
python -m pip install -q --upgrade pip
python -m pip install -q "tensorflow[and-cuda]==2.15.*" stardist csbdeep || echo "FAIL stardist install"
python - <<'PY' || echo "FAIL import stardist"
import tensorflow as tf, stardist
print("STARDIST_OK tf", tf.__version__, "gpus", len(tf.config.list_physical_devices("GPU")),
      "stardist", stardist.__version__)
PY
conda deactivate

echo "===== [3/3] metrics : framework-agnostic scoring ====="
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

echo "ALL_BUILDS_DONE"
