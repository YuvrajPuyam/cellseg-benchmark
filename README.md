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
**Nuclear task** (all metrics: higher is better)

| model | F1@0.5 | F1@0.75 | AJI+ | PQ | boundary-F1 | Dice | n |
|---|---|---|---|---|---|---|---|
| Cellpose-SAM | **0.841** | 0.530 | **0.710** | **0.651** | 0.895 | 0.871 | 297 |
| μSAM | 0.810 | **0.601** | 0.702 | 0.648 | **0.907** | **0.892** | 297 |
| StarDist (2D_versatile_fluo) | _see results/benchmark_tables.md_ | | | | | | |

Cellpose-SAM leads on F1@0.5 / AJI+ / PQ; μSAM edges ahead on the stricter F1@0.75 and on
boundary-F1 (tighter contours). Whole-cell table + StarDist row: `results/benchmark_tables.md`.
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
- **μSAM input:** whole-cell μSAM uses a channel-mean grayscale; a learned 2-channel fusion could help.
- **Not chasing SOTA:** the deliverable is the rigorous, reproducible comparison + failure analysis.

## License notes

- Cellpose `cpsam` weights are **CC-BY-NC** — research/portfolio use only, not commercial.
- TissueNet is **non-commercial academic** — not redistributed in this repo.
