#!/bin/bash
#SBATCH --job-name=cajal-bench
#SBATCH --account=csml                 # csml allocation (A30 only)
#SBATCH --partition=a30                # GPU model selected by partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1              # == --gres=gpu:1 (RCAC-preferred form)
#SBATCH --cpus-per-task=8             # A30 nodes have 24 cpu; 8 is plenty per job
#SBATCH --time=04:00:00               # default is only 30 min — ALWAYS set
#SBATCH --output=cajal-bench-%j.out
set -euo pipefail

REPO=/scratch/gilbreth/gupta596/MotionGen/HOI/cajal
cd "$REPO"

module purge
module load anaconda                  # confirm exact ver via `module spider anaconda`
module use "$HOME/privatemodules"
module load conda-env/"${ENVNAME:?set ENVNAME, e.g. torch-cell-py3.10}"   # built with conda-env-mod

nvidia-smi                            # log hardware/driver into results for reproducibility
python -c "import torch;print('torch',torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))" 2>/dev/null || true

# one model per submission (TF and PyTorch can't share a GPU env): pass MODEL + ENVNAME via --export
#   sbatch --export=ALL,MODEL=cellpose,ENVNAME=torch-cell-py3.10  slurm/benchmark.sbatch
python -m src.eval.run_benchmark \
    --model "${MODEL:?set MODEL (cellpose|microsam|stardist)}" \
    --split "${SPLIT:-test}" \
    --seed 42 \
    --out "$REPO/results/$SLURM_JOB_ID"
