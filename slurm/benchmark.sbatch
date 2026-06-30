#!/bin/bash
#SBATCH --job-name=cajal-bench
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G                     # Gilbreth requires explicit memory; 60G/GPU is the a30 default
#SBATCH --time=04:00:00               # default walltime is only 30 min — always set
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=/scratch/gilbreth/gupta596/MotionGen/HOI/cajal
cd "$REPO"
source /apps/external/anaconda/2025.06/etc/profile.d/conda.sh
conda activate "/scratch/gilbreth/gupta596/envs/${ENV:-torch-cell}"   # stardist-tf for stardist

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY' || true
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
except ModuleNotFoundError:
    pass
try:
    import tensorflow as tf
    print("tf", tf.__version__, "gpus", len(tf.config.list_physical_devices("GPU")))
except ModuleNotFoundError:
    pass
PY

# infer one model/task; weights are pre-cached so no internet is needed on the compute node.
#   sbatch --export=ALL,ENV=torch-cell,MODEL=cellpose,TASK=wholecell  slurm/benchmark.sbatch
#   sbatch --export=ALL,ENV=stardist-tf,MODEL=stardist,TASK=nuclear   slurm/benchmark.sbatch
python -m src.eval.run_benchmark infer \
    --model "${MODEL:?set MODEL}" --task "${TASK:?set TASK}" --split "${SPLIT:-test}" \
    ${LIMIT:+--limit ${LIMIT}} ${TAG:+--tag ${TAG}} --out "$REPO/results/masks"
echo "BENCH_JOB_DONE"
