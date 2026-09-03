#!/bin/bash
#SBATCH --job-name=cajal-fullft
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=cajal-fullft-%j.out
# Full fine-tune study (val-based LR selection + 3-seed test eval). ~2.5-3.5 GPU-hrs.
cd /scratch/gilbreth/$USER/cellseg-benchmark
bash scripts/gilbreth/finetune_full.sh
