#!/bin/bash
#SBATCH --job-name=cajal-rerun
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=cajal-rerun-%j.out
cd /scratch/gilbreth/$USER/cellseg-benchmark
bash scripts/gilbreth/rerun_v2.sh
