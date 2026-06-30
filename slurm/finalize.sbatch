#!/bin/bash
#SBATCH --job-name=cajal-finalize
#SBATCH --account=csml
#SBATCH --partition=a30
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=cajal-finalize-%j.out
# Runs the full finalize (stardist CPU infer + score all + report + delta + figures) as a
# scheduler-managed job — immune to login-node SSH drops. Poll cajal-finalize-<jobid>.out.
cd /scratch/gilbreth/gupta596/MotionGen/HOI/cajal
bash scripts/gilbreth/finalize.sh
