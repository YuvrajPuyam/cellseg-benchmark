#!/bin/bash
# Robust, detached finalize: stardist CPU infer (if needed) + score ALL masks + report +
# fine-tune delta + regenerate nuclear figures. Survives ssh drops (run via nohup, poll the log):
#   nohup bash scripts/gilbreth/finalize.sh > logs/finalize.log 2>&1 &
set -o pipefail
R=/scratch/gilbreth/gupta596/MotionGen/HOI/cajal; cd "$R"
source /apps/external/anaconda/2025.06/etc/profile.d/conda.sh

if [ ! -f results/masks/stardist_nuclear_test.npz ]; then
  echo "[finalize] stardist nuclear CPU inference (no GPU → avoids the TF segfault)"
  conda activate /scratch/gilbreth/gupta596/envs/stardist-tf
  CUDA_VISIBLE_DEVICES="" python -m src.eval.run_benchmark infer \
      --model stardist --task nuclear --split test --limit 300 --out results/masks 2>&1 | tail -3
  conda deactivate
fi

conda activate /scratch/gilbreth/gupta596/envs/metrics
for f in results/masks/*_test*.npz; do
  echo "[finalize] scoring $(basename "$f")"
  python -m src.eval.run_benchmark score --pred-npz "$f" 2>&1 | grep -E "SCORE_DONE|Error|Traceback|x not in" | tail -2
done

echo "[finalize] report"
python -m src.eval.run_benchmark report > /dev/null 2>&1
echo "----- benchmark_tables.md -----"; cat results/benchmark_tables.md

python - <<'PY' > results/finetune_delta.md 2>/dev/null
import json, glob, os
b = json.load(open("results/cellpose_wholecell_test_agg.json"))
def row(name, a):
    return f"| {name} | {a['f1@0.5_mean']:.3f} | {a['aji_plus_mean']:.3f} | {a['pq_mean']:.3f} | {a['boundary_f1_mean']:.3f} | {a['dice_mean']:.3f} |"
print("# Fine-tuning study — Cellpose-SAM, whole-cell (TissueNet test, N=300)\n")
print("Fine-tuned on a 200-image TissueNet train subset, 20 epochs. Delta vs zero-shot.\n")
print("| model | F1@0.5 | AJI+ | PQ | boundary-F1 | Dice |")
print("|---|---|---|---|---|---|")
print(row("cellpose zero-shot", b))
for f in sorted(glob.glob("results/cellpose_wholecell_test_*_agg.json")):
    a = json.load(open(f)); t = os.path.basename(f).replace("cellpose_wholecell_test_", "").replace("_agg.json", "")
    print(row("cellpose ft " + t, a))
print()
for f in sorted(glob.glob("results/cellpose_wholecell_test_*_agg.json")):
    a = json.load(open(f)); t = os.path.basename(f).replace("cellpose_wholecell_test_", "").replace("_agg.json", "")
    print(f"- **{t}**: ΔF1@0.5 = {a['f1@0.5_mean']-b['f1@0.5_mean']:+.3f}, ΔAJI+ = {a['aji_plus_mean']-b['aji_plus_mean']:+.3f}")
PY
echo "----- finetune_delta.md -----"; cat results/finetune_delta.md

python -m src.viz.plots --task nuclear --split test --indices 0,2,5,8,14,20 > /dev/null 2>&1 && echo "[finalize] nuclear figures regenerated (3 models)"
echo "FINALIZE_DONE"
