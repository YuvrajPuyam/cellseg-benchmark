# Cell/Tissue Segmentation Foundation-Model Benchmark

A reproducible benchmark answering *"which segmentation foundation model — Cellpose-SAM,
μSAM, or StarDist — works on my multiplexed tissue, and how much does fine-tuning buy?"*

> **Status:** scaffold. Code is being written ahead of cluster access. Version pins in
> `envs/*.yml` are best-effort and **must be validated on Gilbreth** (see "Known unvalidated").

## What this is

Three pretrained segmenters run through one uniform eval path on **TissueNet** (multiplexed
tissue), scored with instance-level metrics (AJI/AJI+, PQ, F1@IoU{0.5,0.75}, boundary-F1, Dice),
with touching-cell failure visualizations, then one model is fine-tuned and the lift over
zero-shot is reported.

Two tasks, because StarDist's star-convex prior collapses to near-circles on whole-cell:
- **Whole-cell:** Cellpose-SAM, μSAM
- **Nuclear:** StarDist, Cellpose-SAM, μSAM

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

## Quickstart (target — not yet runnable end-to-end)

```bash
# 0. Prereqs YOU provide: DEEPCELL_ACCESS_TOKEN, Gilbreth -A account, SSH access
export DEEPCELL_ACCESS_TOKEN=...            # from users.deepcell.org
python data/download_tissuenet.py --check   # verify token, pull a v1.1 sample

# 1. Build envs on Gilbreth (conda-env-mod; datasets/envs in $RCAC_SCRATCH)
# 2. Zero-shot benchmark (fits free standby QOS)
sbatch slurm/benchmark.sbatch
# 3. Fine-tune + LR x data-fraction sweep (needs your account queue)
sbatch slurm/finetune.sbatch
```

## Known unvalidated (resolve at build time on cluster)

- Exact `cellpose` / `stardist` / `monai` / `micro_sam` versions at lock.
- μSAM's effective torch pin (docs say 2.1.1–2.2.0; `master` wants ≥2.5) — must co-import
  cleanly with cellpose in `torch-cell`, else split into a third subprocess env.
- Gilbreth partition / module names (`slist`, `module spider`).
- TissueNet license text + µm/px.

## License notes

- Cellpose `cpsam` weights are **CC-BY-NC** — research/portfolio use only, not commercial.
- TissueNet is **non-commercial academic** — not redistributed in this repo.
