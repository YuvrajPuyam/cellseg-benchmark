#!/bin/bash
# Full fine-tune study (the "do better" run): Cellpose-SAM on the FULL 2580-image train split.
#   Phase 1 — LR sweep on VAL (few epochs) → pick best LR (no LR-on-test).
#   Phase 2 — fine-tune at best LR for FINAL_EPOCHS across 3 seeds → eval on TEST (first-300, same
#             images as the zero-shot baseline) → mean ± seed-sd delta vs zero-shot.
# Switches between the torch-cell (train/infer) and metrics (score) conda envs. Run via
# slurm/finetune_full.sbatch; poll cajal-fullft-<jobid>.out.
set -o pipefail
R=/scratch/gilbreth/gupta596/MotionGen/HOI/cajal; cd "$R"
CONDA=/apps/external/anaconda/2025.06/etc/profile.d/conda.sh
SCRATCH=/scratch/gilbreth/gupta596
LRS="1e-5 3e-5"
SWEEP_EPOCHS=6
FINAL_EPOCHS=20
SEEDS="0 1 2"
LIM=300

echo "===== PHASE 1: LR sweep on VAL (full train data, ${SWEEP_EPOCHS} epochs) ====="
source "$CONDA"; conda activate "$SCRATCH/envs/torch-cell"
for LR in $LRS; do
  echo "--- sweep fine-tune lr=$LR ---"
  ck=$(python -m src.train.finetune --lr "$LR" --fraction 1.0 --epochs "$SWEEP_EPOCHS" --seed 0 \
        --out results/checkpoints 2>&1 | tee /dev/stderr | grep '^FT_CKPT' | awk '{print $2}')
  echo "sweep ckpt lr=$LR -> $ck"
  CAJAL_CELLPOSE_CKPT="$ck" python -m src.eval.run_benchmark infer --model cellpose --task wholecell \
        --split val --limit "$LIM" --tag "sweep_$LR" --out results/masks 2>&1 | grep -E 'INFER_DONE|Error' | tail -1
done
conda deactivate

echo "===== score VAL → pick best LR ====="
conda activate "$SCRATCH/envs/metrics"
for LR in $LRS; do
  python -m src.eval.run_benchmark score --pred-npz "results/masks/cellpose_wholecell_val_sweep_$LR.npz" \
        --out results 2>&1 | grep -E 'SCORE_DONE|Error' | tail -1
done
python - <<PY
import json
best=None; bf=-1.0
for lr in "$LRS".split():
    try: f=json.load(open(f"results/cellpose_wholecell_val_sweep_{lr}_agg.json"))["f1@0.5_mean"]
    except Exception: f=-1.0
    print(f"  VAL lr={lr}  f1@0.5={f:.4f}")
    if f>bf: bf, best = f, lr
print(f"  BEST_LR={best} (val f1={bf:.4f})")
open(".best_lr","w").write(best or "1e-5")
PY
conda deactivate
BEST=$(cat .best_lr); echo "selected LR = $BEST"

echo "===== PHASE 2: final fine-tune lr=$BEST, ${FINAL_EPOCHS} epochs, seeds {$SEEDS} → TEST ====="
conda activate "$SCRATCH/envs/torch-cell"
for S in $SEEDS; do
  echo "--- final fine-tune lr=$BEST seed=$S ---"
  ck=$(python -m src.train.finetune --lr "$BEST" --fraction 1.0 --epochs "$FINAL_EPOCHS" --seed "$S" \
        --out results/checkpoints 2>&1 | tee /dev/stderr | grep '^FT_CKPT' | awk '{print $2}')
  echo "final ckpt seed=$S -> $ck"
  CAJAL_CELLPOSE_CKPT="$ck" python -m src.eval.run_benchmark infer --model cellpose --task wholecell \
        --split test --limit "$LIM" --tag "fullft_s$S" --out results/masks 2>&1 | grep -E 'INFER_DONE|Error' | tail -1
done
conda deactivate

echo "===== score TEST + delta vs zero-shot ====="
conda activate "$SCRATCH/envs/metrics"
for S in $SEEDS; do
  python -m src.eval.run_benchmark score --pred-npz "results/masks/cellpose_wholecell_test_fullft_s$S.npz" \
        --out results 2>&1 | grep -E 'SCORE_DONE|Error' | tail -1
done
python - <<PY
import json, statistics as st
base=json.load(open("results/cellpose_wholecell_test_agg.json"))["f1@0.5_mean"]
ba=json.load(open("results/cellpose_wholecell_test_agg.json"))["aji_plus_mean"]
f1=[]; aji=[]
for s in "$SEEDS".split():
    try:
        a=json.load(open(f"results/cellpose_wholecell_test_fullft_s{s}_agg.json"))
        f1.append(a["f1@0.5_mean"]); aji.append(a["aji_plus_mean"])
    except Exception as e: print("  miss seed", s, e)
if f1:
    mf=sum(f1)/len(f1); sf=st.pstdev(f1) if len(f1)>1 else 0.0
    ma=sum(aji)/len(aji); sa=st.pstdev(aji) if len(aji)>1 else 0.0
    print(f"  FULL-FT test  F1@0.5={mf:.4f} ±{sf:.4f}  AJI+={ma:.4f} ±{sa:.4f}  (seeds={[round(x,4) for x in f1]})")
    print(f"  ZERO-SHOT     F1@0.5={base:.4f}            AJI+={ba:.4f}")
    print(f"  DELTA         F1@0.5={mf-base:+.4f}        AJI+={ma-ba:+.4f}")
    open("results/finetune_full.md","w").write(
      "# Full fine-tune — Cellpose-SAM, whole-cell (full 2580-img train split)\n\n"
      f"Val-selected LR = {open('.best_lr').read().strip()} · {$FINAL_EPOCHS} epochs · {len(f1)} seeds · test N=297.\n\n"
      "| | F1@0.5 | AJI+ |\n|---|---|---|\n"
      f"| zero-shot (200-img earlier was +1.5) | {base:.3f} | {ba:.3f} |\n"
      f"| **full fine-tune** | **{mf:.3f} ±{sf:.3f}** | **{ma:.3f} ±{sa:.3f}** |\n"
      f"| **Δ** | **{mf-base:+.3f}** | **{ma-ba:+.3f}** |\n")
else:
    print("  no fine-tuned test aggs found")
PY
conda deactivate
echo "FULL_FT_DONE"
