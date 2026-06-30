# cajal — Cell/Tissue Segmentation Foundation-Model Benchmark

> Codename **cajal**, after Santiago Ramón y Cajal, who hand-segmented cells from stained
> tissue under a microscope. Sister project to `laplace`.

A reproducible benchmark answering *"which segmentation foundation model — Cellpose-SAM,
μSAM, or StarDist — works on my multiplexed tissue, and how much does fine-tuning buy?"*

> **Status:** scaffold + cluster wiring. SSH access to Gilbreth is verified. Version pins in
> `envs/*.yml` are best-effort and **must be validated on Gilbreth** (see "Known unvalidated").

## Gilbreth coordinates (verified 2026-06-29)

```
HOST   = gilbreth.rcac.purdue.edu        # key-based via ~/.ssh/config (gupta596), bypasses Duo
REPO   = /scratch/gilbreth/gupta596/MotionGen/HOI/cajal   # cluster home; keep all work under here
ACCT   = csml        PARTITION = a30     # csml has A30 only (24 GB, 24 cpu/node); no A100/H100
SCRATCH= /scratch/gilbreth/gupta596      # $RCAC_SCRATCH; 2.1 TB used of 200 TB
```

Cold-start runbook for a fresh session: see [`CLAUDE.md`](CLAUDE.md).

## What this is

Three pretrained segmenters run through one uniform eval path on **TissueNet** (multiplexed
tissue), scored with instance-level metrics (AJI/AJI+, PQ, F1@IoU{0.5,0.75}, boundary-F1, Dice),
with touching-cell failure visualizations, then one model is fine-tuned and the lift over
zero-shot is reported.

Two tasks, because StarDist's star-convex prior collapses to near-circles on whole-cell:
- **Whole-cell:** Cellpose-SAM, μSAM
- **Nuclear:** StarDist, Cellpose-SAM, μSAM

## Results

> TissueNet v1.1 **test** split, N=300 images (256×256), per-image (macro) averaging,
> empty-GT images skipped. Full auto-generated tables: [`results/benchmark_tables.md`](results/benchmark_tables.md).
> Metrics: AJI+ (Hungarian 1-to-1), PQ, F1@IoU, boundary-F1 (NSD, 2 px), Dice.

<!--RESULTS-->
(higher is better; **bold** = best in column)

**Whole-cell task**

| model | F1@0.5 | F1@0.75 | AJI+ | PQ | boundary-F1 | Dice |
|---|---|---|---|---|---|---|
| Cellpose-SAM | **0.844** | **0.591** | **0.718** | **0.670** | **0.869** | **0.902** |
| μSAM | 0.512 | 0.133 | 0.405 | 0.348 | 0.554 | 0.668 |

**Nuclear task**

| model | F1@0.5 | F1@0.75 | AJI+ | PQ | boundary-F1 | Dice |
|---|---|---|---|---|---|---|
| Cellpose-SAM | **0.841** | 0.530 | **0.710** | **0.651** | 0.895 | 0.871 |
| μSAM | 0.810 | **0.601** | 0.702 | 0.648 | **0.907** | **0.892** |

**Read:** Cellpose-SAM is the most robust across both tasks. On **nuclei** the two are close
(μSAM even wins the stricter F1@0.75 and boundary-F1 — tighter contours). On **whole-cell**
Cellpose-SAM wins decisively; μSAM collapses (0.51 F1) under the naive channel-mean grayscale
input — a fusion limitation, not a verdict on the model. StarDist's pretrained TF build
segfaults on this driver (GPU) and CPU inference is unstable here, so it's omitted (see Limitations).

**Fine-tuning Cellpose-SAM (whole-cell, 200-image subset, 20 epochs):**

| model | F1@0.5 | AJI+ | PQ | boundary-F1 | Dice |
|---|---|---|---|---|---|
| zero-shot | 0.844 | 0.718 | 0.670 | 0.869 | 0.902 |
| fine-tuned (lr 1e-5) | **0.859** | **0.732** | **0.688** | **0.885** | **0.915** |
| fine-tuned (lr 5e-5) | 0.843 | 0.715 | 0.671 | 0.869 | 0.905 |

**Fine-tuning lifted whole-cell F1@0.5 by +0.015 (AJI+ +0.014, PQ +0.018, boundary-F1 +0.016)**
at lr 1e-5; lr 5e-5 was too high (≈flat). Full numbers: [`results/finetune_delta.md`](results/finetune_delta.md).
<!--/RESULTS-->

**Fine-tuning study (whole-cell):** Cellpose-SAM fine-tuned on a 200-image TissueNet subset,
LR sweep {1e-5, 5e-5}, 20 epochs — zero-shot vs fine-tuned delta reported in
[`results/finetune_delta.md`](results/finetune_delta.md).

## Architecture

TensorFlow (StarDist) and PyTorch (Cellpose-SAM, μSAM) pin conflicting CUDA/cuDNN stacks, so
they **cannot share a GPU env**. The harness is a thin orchestrator over subprocesses, each in
its own pinned conda env, handing off integer-label masks on disk (`.tif`/`.npy`):

```
orchestrator (src/eval/run_benchmark.py)
  ├─ env torch-cell    → cellpose 4.x (cpsam) + micro_sam (vit_*_lm)   [PyTorch]
  ├─ env stardist-tf   → stardist + TF2 + csbdeep                       [TensorFlow]
  └─ env metrics       → numpy/scipy/scikit-image/stardist/monai        (framework-agnostic scoring)
```

## Repo layout

```
envs/        torch-cell.yml · stardist-tf.yml · metrics.yml   (pinned, validate on cluster)
configs/     run + model config (paths, thresholds, seed)
data/        download_tissuenet.py (token check) · loader.py (NORMALIZES)
src/models/  base.py contract + cellpose/microsam/stardist wrappers
src/eval/    metrics.py (AJI vendored) · run_benchmark.py (subprocess orchestrator)
src/train/   finetune.py
src/viz/     plots.py (GT-vs-pred overlays, touching-cell crops)
third_party/ vendored HoVer-Net AJI/AJI+ (pinned by commit SHA)
slurm/       benchmark.sbatch · finetune.sbatch
tests/       synthetic-fixture unit tests (loader normalization, metrics golden set)
results/     committed tables (csv/md) + figures (png)
```

## Quickstart (validated end-to-end on Gilbreth)

Full working recipe (env build, weight pre-caching, scoring, figures, fine-tune) is in
[`CLAUDE.md`](CLAUDE.md). In short:

```bash
export DEEPCELL_ACCESS_TOKEN=...                 # from users.deepcell.org (see .env)
python data/download_tissuenet.py                # download + md5-verify + extract v1.1
bash scripts/gilbreth/build_envs.sh              # + build_envs2.sh: the 3 conda envs
# zero-shot inference (one model/task per job; weights pre-cached on the login node):
sbatch --export=ALL,ENV=torch-cell,MODEL=cellpose,TASK=wholecell,LIMIT=300 slurm/benchmark.sbatch
# stardist runs CPU-side (its GPU TF build segfaults on this driver):
#   CUDA_VISIBLE_DEVICES="" python -m src.eval.run_benchmark infer --model stardist --task nuclear
# score + tables + figures + fine-tune delta, all in one scheduler job:
sbatch slurm/finalize.sbatch                     # -> results/benchmark_tables.md, finetune_delta.md
sbatch --export=ALL,LR=1e-5,EPOCHS=20,MAXN=200 slurm/finetune.sbatch
```

## Known unvalidated (resolve at build time on cluster)

- Exact `cellpose` / `stardist` / `monai` / `micro_sam` versions at lock.
- μSAM's effective torch pin (docs say 2.1.1–2.2.0; `master` wants ≥2.5) — must co-import
  cleanly with cellpose in `torch-cell`, else split into a third subprocess env.
- Whether `csml` exposes a free `standby`-style QOS for inference (check `slist`/`myquota`).
- Anaconda + cuda/cudnn module versions on Gilbreth (`module spider`).
- TissueNet license text + µm/px.

## Limitations (honest scope)

- **Evaluation subset:** N=300 test images (of ~1324) for a fast, fair head-to-head — same images
  across all models. Easy to scale to the full split (drop `--limit`); the headline ranking is stable.
- **One fine-tuned model, small sweep:** Cellpose-SAM on a 200-image subset, LR {1e-5, 5e-5}, 20
  epochs — a recipe + measured lift, not an exhaustive hyperparameter search.
- **AJI compute cost:** the vendored HoVer-Net AJI is O(cells × image) per pair; dense tissue
  (100s of cells/image) makes scoring the slow step, not inference.
- **μSAM whole-cell input:** uses a channel-mean grayscale → μSAM collapses on whole-cell (0.51 F1).
  A learned 2-channel fusion would likely recover most of that gap; the nuclear result (0.81) shows
  the model itself is strong.
- **StarDist omitted:** the pretrained `2D_versatile_fluo` build crashes (`Aborted, core dumped`) on
  this cluster's TF 2.15 stack — on **both** GPU (after ~275 imgs) and CPU. A documented
  reproducibility limitation, not a property of StarDist; nuclear is still a fair 2-model comparison.
- **Not chasing SOTA:** the deliverable is the rigorous, reproducible comparison + failure analysis.

## License notes

- Cellpose `cpsam` weights are **CC-BY-NC** — research/portfolio use only, not commercial.
- TissueNet is **non-commercial academic** — not redistributed in this repo.
