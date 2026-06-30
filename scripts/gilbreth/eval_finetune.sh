#!/bin/bash
# Autonomously evaluate fine-tuned cellpose checkpoints: wait for fine-tune jobs → submit tagged
# whole-cell eval infer with CAJAL_CELLPOSE_CKPT → score → print delta vs zero-shot.
#   nohup bash scripts/gilbreth/eval_finetune.sh > logs/eval_finetune.log 2>&1 &
set -o pipefail
R=/scratch/gilbreth/gupta596/MotionGen/HOI/cajal; cd "$R"
source /apps/external/anaconda/2025.06/etc/profile.d/conda.sh

echo "waiting for fine-tune checkpoints + zero-shot baseline..."
for i in $(seq 1 320); do
  ftrun=$(squeue -u gupta596 -h -n cajal-finetune -o %T 2>/dev/null | wc -l)
  nck=$(ls results/checkpoints/models/ 2>/dev/null | grep -c "cpsam_ft_.*_n200")
  base=0; [ -f results/cellpose_wholecell_test_agg.json ] && base=1
  [ "$ftrun" -eq 0 ] && [ "$nck" -ge 1 ] && [ "$base" -eq 1 ] && { echo "ready after ~$((i*15))s ($nck ckpts)"; break; }
  sleep 15
done
echo "=== checkpoints ==="; ls -1 results/checkpoints/models/ 2>&1 | grep cpsam_ft

declare -a JIDS
for ckpt in results/checkpoints/models/cpsam_ft_*_n200; do
  [ -e "$ckpt" ] || continue
  tag=$(basename "$ckpt" | sed "s/cpsam_ft_//;s/_frac1.0_n200//;s/\\./p/g")
  jid=$(sbatch --parsable --export=ALL,ENV=torch-cell,MODEL=cellpose,TASK=wholecell,LIMIT=300,TAG=$tag,CAJAL_CELLPOSE_CKPT=$R/$ckpt slurm/benchmark.sbatch)
  echo "eval $tag -> job $jid"; JIDS+=("$jid")
done
JCSV=$(IFS=,; echo "${JIDS[*]}")
echo "waiting for eval infer jobs: $JCSV"
for i in $(seq 1 160); do
  r=$(squeue -u gupta596 -h -j "$JCSV" -o %T 2>/dev/null | wc -l)
  [ "$r" -eq 0 ] && { echo "eval infer done after ~$((i*15))s"; break; }
  sleep 15
done

conda activate /scratch/gilbreth/gupta596/envs/metrics
for f in results/masks/cellpose_wholecell_test_*.npz; do
  python -m src.eval.run_benchmark score --pred-npz "$f" 2>&1 | grep -E "SCORE_DONE|Error|Traceback" | tail -1
done

echo "=== FINE-TUNE DELTA (whole-cell, vs zero-shot cellpose) ==="
python - <<'PY'
import json, glob, os
base = json.load(open("results/cellpose_wholecell_test_agg.json"))
def fmt(a): return f"f1@0.5={a.get('f1@0.5_mean',0):.3f} aji+={a.get('aji_plus_mean',0):.3f} pq={a.get('pq_mean',0):.3f} dice={a.get('dice_mean',0):.3f}"
print("zero-shot          :", fmt(base))
for f in sorted(glob.glob("results/cellpose_wholecell_test_*_agg.json")):
    a = json.load(open(f)); tag = os.path.basename(f).replace("cellpose_wholecell_test_","").replace("_agg.json","")
    d = a.get('f1@0.5_mean',0) - base.get('f1@0.5_mean',0)
    print(f"ft {tag:15}:", fmt(a), f"   Δf1@0.5={d:+.3f}")
PY
echo "EVAL_FINETUNE_DONE"
