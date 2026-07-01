# Handoff - Cell/Tissue Segmentation Foundation-Model Benchmark

**Owner:** Yuvraj Puyam
**Created:** 2026-06-28
**Read this first, then read [`project-spec.md`](project-spec.md)** (the full, research-hardened build plan in this same folder).

---

## Objective (one paragraph)

Build a **reproducible benchmark** that evaluates the three leading cell/nucleus segmentation foundation models - **Cellpose-SAM**, **μSAM (micro-sam)**, and **StarDist** - on multiplexed tissue imaging (**TissueNet**). Quantify where each model wins and fails (especially on **touching/overlapping cells**), then **fine-tune one model** on TissueNet and measure the lift over zero-shot. The output is a clear "which model, when, and how much does fine-tuning help" guide - benchmark tables, failure-mode visualizations, and a fine-tuning recipe - packaged as a one-command-reproducible GitHub repo.

## Why we're building it (the strategic purpose)

This is a **portfolio / research-assistantship artifact** for outreach to **computational-biology / bioimage-analysis / spatial-omics labs** (e.g. Purdue RA applications). Segmentation is the load-bearing first step of every spatial-biology pipeline, and rigorous apples-to-apples comparisons on multiplexed tissue are scarce - so a finished, verifiable answer to "which segmenter for our tissue?" is genuinely useful to those labs.

**What gets pitched** (not the repo itself): a **one-page result sheet** - the two benchmark tables, one killer touching-cell failure figure, and the headline "fine-tuning lifted F1 by +X pts" number - backed by the reproducible repo and a 2-minute demo. Tone of the pitch must be **curious, not claiming** (share a finding + a question; don't offer to "fix their pipeline"). Target a *cluster* of imaging/omics labs, not one professor, so it reads as a portfolio piece, not flattery.

> Strategic caveat carried over: this artifact is sharp for **imaging/omics** labs specifically. If the target professors are digital-twin / non-imaging, the sibling project (SpatialVLM, in `D:\career-ops\projects\p1-spatial-vlm-spec.md`) may be the better-matched calling card. Gut-check the target labs before committing.

---

## The 7 research-verified facts that shape the build

(All verified against primary sources on 2026-06-27 - full citations in `project-spec.md`.)

1. **No single environment.** StarDist is **TensorFlow**; Cellpose-SAM + μSAM are **PyTorch**. TF and PyTorch pin conflicting CUDA/cuDNN - they cannot share a GPU env. → **Two model envs + a metrics env; models run as subprocesses and hand off masks via disk** (`.tif`/`.npy`).
2. **StarDist can't do whole-cell fairly** on TissueNet (star-convex prior → near-circles). → **Two tasks:** whole-cell (Cellpose-SAM, μSAM) and nuclear (all three).
3. **"Cellpose-SAM" = the `cpsam` checkpoint inside `cellpose` 4.x** (pip `cellpose`). No `channels=` arg in 4.x (auto-uses 3 channels). Returns a 3-tuple. Weights are **CC-BY-NC** (research OK, not commercial).
4. **TissueNet is gated** - free account → token from `users.deepcell.org` → `DEEPCELL_ACCESS_TOKEN` → `deepcell.datasets.TissueNet(version='1.1')`. Auth has been flaky. **#1 project risk.**
5. **It's low-single-digit GB** (not 1.3 TB), 2-ch images (0=nuclear, 1=membrane), 2-ch masks (whole-cell + nuclear). **Not pre-normalized - must normalize in the loader.**
6. **Metrics: no one library does all.** Anchor on `stardist.matching` (PQ, F1@IoU, precision/recall - but NOT AJI). Vendor HoVer-Net `stats_utils.py` for AJI/AJI+ (pin by commit SHA). MONAI for Dice/IoU + boundary-F1 (NSD). `torchmetrics.Dice` was removed in v1.7. **Document per-image vs pooled averaging** - it's the #1 reproducibility trap.
7. **Gilbreth (Purdue Slurm):** per-PI account (`-A` from `slist` is REQUIRED), partitions per-GPU-model (`a100-40gb`), free `standby` QOS (4h cap - good for inference), default walltime 30 min, datasets in `$RCAC_SCRATCH` (home is 25 GB), envs via `conda-env-mod`.

---

## START HERE - Day 0 (do this before writing any pipeline code)

Both highest risks get retired up front:

1. **Prove TissueNet access.** Register at `users.deepcell.org`, get the token, set `DEEPCELL_ACCESS_TOKEN`, and successfully pull a TissueNet v1.1 sample. If the gate is broken, fall back to the Cellpose-relabeled figshare subset (link in spec).
2. **Build & GPU-validate the two conda envs** (`torch-cell` for cellpose + micro_sam; `stardist-tf` for StarDist + TF2 + csbdeep). Confirm `torch.cuda.is_available()` and `tf.config.list_physical_devices('GPU')` each return GPU in their own env. (On Windows, StarDist GPU needs WSL2 or just run it on Gilbreth - native-Windows GPU TF was dropped after 2.10.)

Only after both pass: build `loader.py` (with normalization) → one model wrapper end-to-end → all three wrappers + metrics → zero-shot tables → fine-tune + sweep → writeup. Full timeline (~5-7 focused days) and repo structure are in `project-spec.md` §4-§8.

## Deliverables checklist

- [ ] One-command-reproducible GitHub repo (pinned per-env lockfiles)
- [ ] Two benchmark tables (whole-cell, nuclear): AJI/AJI+, PQ, F1@IoU{0.5,0.75}, boundary-F1, Dice
- [ ] Touching-cell failure visualizations (GT vs pred overlays)
- [ ] Fine-tuning study: zero-shot vs fine-tuned deltas (LR × data-fraction sweep)
- [ ] README writeup: "which model when, what fine-tuning buys"
- [ ] One-page result sheet + 2-min demo (the pitch artifacts)

## Working notes for the next session

- The full plan with inline citations, verified API call snippets, repo tree, annotated sbatch, metrics code, and risk table is in **[`project-spec.md`](project-spec.md)** - read it before starting.
- This is a fresh repo. First real coding step after Day 0 is scaffolding: the two `envs/*.yml`, the `segment(image) -> mask` wrapper contract, and `data/download_tissuenet.py` with the token check.
- Re-confirm volatile facts at build time: latest `cellpose`/`stardist`/`monai`/`micro_sam` versions; μSAM's effective torch pin; exact Gilbreth partition/module names via `slist` / `module spider`.
