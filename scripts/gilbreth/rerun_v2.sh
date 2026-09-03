#!/bin/bash
# Re-run with the hardened methodology: (1) μSAM whole-cell with the FAIR RGB=[membrane,nuclear,
# membrane] input (the old channel-mean strawman), (2) re-score every mask with bootstrap 95% CIs,
# (3) regenerate the tables (now "mean ±ci"). Same first-300 test images as the existing cellpose
# results, so the head-to-head stays comparable. Run via slurm/rerun_v2.sbatch (scheduler-managed).
set -o pipefail
R=/scratch/gilbreth/$USER/cellseg-benchmark; cd "$R"
source /apps/external/anaconda/2025.06/etc/profile.d/conda.sh

echo "[rerun] (1) μSAM whole-cell, fair RGB input"
conda activate /scratch/gilbreth/$USER/envs/torch-cell
python -m src.eval.run_benchmark infer --model microsam --task wholecell --split test --limit 300 \
    --out results/masks 2>&1 | tail -3
conda deactivate

echo "[rerun] (2) re-score all masks with bootstrap CIs"
conda activate /scratch/gilbreth/$USER/envs/metrics
for f in results/masks/*_test.npz; do
  echo "  scoring $(basename "$f")"
  python -m src.eval.run_benchmark score --pred-npz "$f" 2>&1 | grep -E "SCORE_DONE|Error|Traceback" | tail -1
done

echo "[rerun] (3) report (mean ±ci)"
python -m src.eval.run_benchmark report > /dev/null 2>&1
echo "----- benchmark_tables.md -----"; cat results/benchmark_tables.md
echo "RERUN_DONE"
