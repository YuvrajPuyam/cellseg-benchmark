#!/bin/bash
#SBATCH --job-name=cajal-finetune
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00               # fine-tune + sweep is long; raise if your QOS caps allow
#SBATCH --output=cajal-finetune-%j.out
set -euo pipefail

REPO=/scratch/gilbreth/gupta596/MotionGen/HOI/cajal
cd "$REPO"

module purge
module load anaconda
module use "$HOME/privatemodules"
module load conda-env/"${ENVNAME:-torch-cell-py3.10}"   # fine-tuning stays in the PyTorch env

nvidia-smi

# fine-tune one PyTorch model (cellpose via train_seg, or microsam via train_sam).
# sweep is driven inside finetune.py from configs/default.yaml (lr_sweep x data_fractions).
#   sbatch --export=ALL,BASE_MODEL=cellpose,ENVNAME=torch-cell-py3.10  slurm/finetune.sbatch
python -m src.train.finetune \
    --base-model "${BASE_MODEL:-cellpose}" \
    --config configs/default.yaml \
    --seed 42 \
    --out "$REPO/results/finetune/$SLURM_JOB_ID"
