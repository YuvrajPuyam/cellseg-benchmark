#!/bin/bash
#SBATCH --job-name=cajal-finetune
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G                     # Gilbreth requires explicit memory
#SBATCH --time=12:00:00               # fine-tune + sweep is long; raise if your QOS caps allow
#SBATCH --output=cajal-finetune-%j.out
set -euo pipefail

REPO=/scratch/gilbreth/$USER/cellseg-benchmark
cd "$REPO"
source /apps/external/anaconda/2025.06/etc/profile.d/conda.sh
conda activate /scratch/gilbreth/$USER/envs/torch-cell   # fine-tuning stays in the PyTorch env

nvidia-smi --query-gpu=name --format=csv,noheader

# fine-tune cellpose-SAM on a TissueNet whole-cell subset.
#   sbatch --export=ALL,LR=1e-5,FRACTION=1.0,EPOCHS=20,MAXN=200  slurm/finetune.sbatch
python -m src.train.finetune \
    --lr "${LR:-1e-5}" --fraction "${FRACTION:-1.0}" --epochs "${EPOCHS:-20}" \
    ${MAXN:+--max-n ${MAXN}} --seed 42 \
    --out "$REPO/results/checkpoints"
