# Project Spec — Cell/Tissue Segmentation Foundation-Model Benchmark

**Owner:** Yuvraj Puyam
**Date:** 2026-06-27
**Status:** Proposed (BUILD) — plan hardened against primary-source research
**One-liner:** A reproducible benchmark answering "which segmentation foundation model — Cellpose-SAM, μSAM, or StarDist — works on *my* multiplexed tissue, and how much does fine-tuning buy?" with metrics, failure-mode visualizations, and a fine-tuning recipe a wet-lab could adopt.

> This spec supersedes the original plan. Every technical claim below was verified against primary sources (model source code, dataset docs, RCAC docs) on 2026-06-27. **Seven findings changed the plan** — see §0. Citations are inline.

---

## 0. What the research changed (read this first)

| # | Original assumption | Verified reality | Plan impact |
|---|---|---|---|
| 1 | "Env built — 3 models + CUDA coexisting" (Day 0.5) | StarDist is **TensorFlow**; Cellpose-SAM + μSAM are **PyTorch**. TF and PyTorch pin **conflicting CUDA/cuDNN** stacks — they do **not** coexist with GPU in one env. ([TF install](https://www.tensorflow.org/install/pip)) | **Two conda envs** + a third metrics env. Models run as **subprocesses**, hand off masks via disk. This is now the backbone of the architecture (§4). |
| 2 | StarDist is one of three equal whole-cell competitors | StarDist fits **star-convex polygons** → great for nuclei, but on TissueNet **whole-cell it collapses to near-circles** (documented). ([benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10862744/)) | **Split the benchmark into two tasks**: whole-cell (Cellpose-SAM, μSAM) and nuclear (all three). StarDist competes only where it's fair. Richer story, not a weakness. |
| 3 | "Cellpose-SAM" is a distinct package | It's the **`cpsam` checkpoint inside `cellpose` 4.x** (pip `cellpose`, latest 4.2.1.1, Jun 2026). **`channels=` is dropped** — it auto-uses the first 3 channels, order-invariant. Returns a **3-tuple**. ([repo](https://github.com/MouseLand/cellpose), [models.py](https://raw.githubusercontent.com/MouseLand/cellpose/main/cellpose/models.py)) | Wrapper is simpler than expected (no channel plumbing). **Weights are CC-BY-NC** — fine for a research/portfolio benchmark, not for any commercial use. |
| 4 | TissueNet = "download it" | **Gated.** Needs a free account → API token from `users.deepcell.org` → `DEEPCELL_ACCESS_TOKEN` → `deepcell.datasets.TissueNet`. Auth has a **history of flakiness** ([#694](https://github.com/vanvalenlab/deepcell-tf/issues/694)). License = **non-commercial academic**. | **Day 0 = verify your token works** before building anything on it. This is the #1 project risk. Fallback: [Cellpose-relabeled subset on figshare](https://janelia.figshare.com/articles/dataset/Human-in-the-loop_labelled_TissueNet_data_Cellpose_2_0_/20510016) (derivative). |
| 5 | "1.3 TB dataset" / big compute for data | That 1.3 TB is a **different** (deepcell-types) corpus. **TissueNet itself is low-single-digit GB.** 2-ch images (0=nuclear, 1=membrane), 2-ch masks (whole-cell + nuclear), 512×512. **Not pre-normalized** — you must normalize. Use **v1.1**. ([docs](https://deepcell.readthedocs.io/en/master/data-gallery/tissuenet.html), [#618](https://github.com/vanvalenlab/deepcell-tf/issues/618)) | Trivial storage. Normalization is a real footgun — bake it into the loader. |
| 6 | "use established metric impls" | **No single lib does all of them.** `stardist.matching` gives PQ + F1@IoU + precision/recall (Hungarian, optimal 1-to-1) but **NOT AJI**. AJI/AJI+ → vendor HoVer-Net `stats_utils.py` (no pip release). Dice/IoU + boundary-F1(NSD) → MONAI. `torchmetrics.Dice` was **removed in v1.7**. | Pinned 3-source metrics stack (§5). Anchor on `stardist.matching`; vendor HoVer-Net by commit SHA; MONAI for the rest. |
| 7 | Generic Slurm sbatch | Gilbreth is **per-PI account** (`-A` from `slist` is **required**), partitions are **per-GPU-model** (`a100-80gb` etc.), free **`standby` QOS** (4 h cap, no charge), default walltime **30 min**, datasets go in **`$RCAC_SCRATCH`** (home is 25 GB), env via **`conda-env-mod`**. ([queues](https://www.rcac.purdue.edu/knowledge/gilbreth/run/slurm/queues), [conda](https://www.rcac.purdue.edu/knowledge/gilbreth/run/examples/apps/python/conda)) | Real sbatch recipe in §7. Inference fits in free `standby`; fine-tune needs your account queue. |

---

## 1. Goal — the résumé/portfolio bullet you want to *earn*

> "Built an open, reproducible benchmark of three segmentation foundation models (Cellpose-SAM, μSAM, StarDist) on multiplexed tissue (TissueNet), with instance-level metrics (AJI/PQ/F1@IoU/boundary-F1), touching-cell failure analysis, and a **fine-tuning recipe that lifted whole-cell F1 by +{X} pts over zero-shot** — one-command reproducible on a Slurm A100."

Three verifiable claims from one repo: (1) rigorous multi-model evaluation others can rerun, (2) honest error analysis on the hard case (touching cells), (3) a measured fine-tuning gain. **Strategic fit:** this is your biomedical-imaging / spatial-omics lane — complements (doesn't duplicate) the AI-infra story in [p1-spatial-vlm-spec.md](p1-spatial-vlm-spec.md). Strong artifact for a Purdue bio-imaging / digital-twin RA pitch.

## 2. What it does

Downloads TissueNet, runs three pretrained segmenters through **one uniform eval path**, scores them with established instance metrics, visualizes *where and why* they fail (especially touching-cell boundaries), then fine-tunes one model on a TissueNet subset and reports the lift over zero-shot. Deliverable is a "which model, when, and how much does fine-tuning help" guide — table + figures + recipe.

## 3. Deliverables

- GitHub repo, one-command setup, reproducible runs (pinned per-env lockfiles).
- **Two benchmark tables** (whole-cell task; nuclear task): AJI/AJI+, PQ, F1@IoU{0.5,0.75}, boundary-F1, Dice.
- Qualitative side-by-side overlays focused on touching-cell boundary failures.
- Fine-tuning study: one model fine-tuned on TissueNet + small LR × data-fraction sweep, zero-shot vs fine-tuned deltas.
- README writeup: what worked, when to use which model, what fine-tuning buys.

---

## 4. Architecture — environments & orchestration (the load-bearing decision)

Because TF (StarDist) and PyTorch (Cellpose-SAM, μSAM) can't share a GPU env, the harness is a **thin orchestrator over subprocesses**, each in its own pinned env, exchanging **instance-label masks on disk** (`.tif` via `tifffile`, or `.npy`). All three models emit integer-ID masks, so the handoff contract is clean.

```
                    ┌─────────────────────────────────────────┐
                    │  orchestrator (run_benchmark.py)         │
                    │  loops models × splits, calls subprocess │
                    └───────────────┬─────────────────────────┘
        conda run -n <env> python -m src.models.<wrapper> ...   (one model at a time on the GPU)
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                           ▼
 env: torch-cell           env: torch-cell             env: stardist-tf
 cellpose 4.x (cpsam)      micro_sam (vit_*_lm)         stardist + TF2 + csbdeep
 [PyTorch ≥2.5/CUDA12.1]   [PyTorch, same env]          [TensorFlow 2.x]
        └──────────── writes masks/{model}/{split}/*.tif ───────┘
                                   ▼
                    env: metrics (numpy/scipy/scikit-image/stardist/monai)
                    reads GT + pred masks → metrics tables + figures
```

**Why this shape:** (a) sidesteps the cuDNN clash entirely; (b) makes each model independently reproducible (one `environment.yml` per env, pinned); (c) keeps scoring framework-agnostic so metric numbers never depend on which DL framework loaded. Cellpose-SAM and μSAM are *both* PyTorch and share `torch-cell` — but reconcile their torch pins (cellpose floor `>=1.6`; μSAM docs tested 2.1.1–2.2.0 / CUDA 12.1, but `master` wants `>=2.5`) and validate both import before locking. If they fight, split them too — the subprocess design makes that a config change, not a rewrite.

**Windows note:** native-Windows GPU TensorFlow was dropped after TF 2.10, so you can't run GPU StarDist natively on your Windows box — develop StarDist CPU-only locally or in WSL2, and run all GPU work on Gilbreth (Linux), where it's a non-issue.

### Repo structure
```
cell-seg-benchmark/
  README.md
  envs/
    torch-cell.yml          # cellpose + micro_sam, pinned
    stardist-tf.yml         # stardist + TF2 + csbdeep, pinned
    metrics.yml             # numpy/scipy/scikit-image/stardist/monai
  configs/                  # model + run configs (yaml): paths, thresholds, seed
  data/
    download_tissuenet.py   # token check → deepcell.datasets.TissueNet(version='1.1')
    loader.py               # yields (image, wholecell_mask, nuclear_mask); NORMALIZES
  src/
    models/
      base.py               # contract: segment(image) -> instance_mask (uint)
      cellpose_wrapper.py    # CellposeModel(gpu=True).eval(img) -> masks (3-tuple)
      microsam_wrapper.py     # automatic_instance_segmentation(predictor, segmenter, img)
      stardist_wrapper.py     # StarDist2D.from_pretrained(...).predict_instances(normalize(img))
    eval/
      metrics.py            # AJI/AJI+ (vendored), PQ, F1@IoU, boundary-F1, Dice
      run_benchmark.py      # orchestrator: subprocess per (model, env), disk handoff
    train/finetune.py       # fine-tune cellpose OR micro_sam on a TissueNet subset
    viz/plots.py            # GT-vs-pred overlays, touching-cell crops
  third_party/
    hover_net_stats_utils.py # vendored AJI/AJI+/fast_pq, pinned to a commit SHA
  slurm/
    benchmark.sbatch
    finetune.sbatch
  results/                  # committed: tables (csv/md), figures (png)
```

### Per-model wrapper contract (verified API calls)
```python
# base.py — every wrapper implements this
def segment(image) -> "np.ndarray[uint]":  # HxW integer instance-label mask, 0=bg
    ...

# cellpose_wrapper.py  (env: torch-cell)  — no channels arg in 4.x; auto-uses 3 channels
from cellpose import models
m = models.CellposeModel(gpu=True)                 # downloads cpsam from HF on first use
masks, flows, styles = m.eval(img, flow_threshold=0.4, cellprob_threshold=0.0)  # 3-tuple

# microsam_wrapper.py  (env: torch-cell)  — _lm model for light microscopy/tissue
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm")  # InstanceSegWithDecoder
mask = automatic_instance_segmentation(predictor, segmenter, img, ndim=2)   # numpy instance mask

# stardist_wrapper.py  (env: stardist-tf)  — nuclear task only on TissueNet
from stardist.models import StarDist2D
from csbdeep.utils import normalize
m = StarDist2D.from_pretrained("2D_versatile_fluo")   # fluorescence nuclei
labels, details = m.predict_instances(normalize(img)) # (label_mask, dict)
```

---

## 5. Metrics — pinned stack

Anchor instance matching on **`stardist.matching`** (pip-installable, optimal Hungarian 1-to-1, audited source). Fill the two gaps it has: **AJI** (vendor HoVer-Net) and **boundary-F1** (MONAI NSD).

```
stardist  (latest at lock) → PQ, F1@IoU{0.5,0.75}, precision/recall, SQ — one call
monai     (latest at lock) → semantic Dice/IoU; boundary-F1 via SurfaceDiceMetric (NSD)
third_party/hover_net_stats_utils.py @ <commit SHA> → get_fast_aji / get_fast_aji_plus
```
```python
from stardist.matching import matching
r50, r75 = matching(y_true, y_pred, thresh=[0.5, 0.75], criterion='iou')
pq, f1_50, f1_75 = r50.panoptic_quality, r50.f1, r75.f1
from third_party.hover_net_stats_utils import get_fast_aji_plus, remap_label
aji_plus = get_fast_aji_plus(remap_label(y_true), remap_label(y_pred))   # prefer AJI+ (1-to-1)
```
**Reproducibility rules (the #1 cause of irreproducible cell-seg numbers):**
- **Document the averaging convention** explicitly: per-image (macro) vs dataset-pooled (micro). `stardist.matching_dataset(by_image=...)` and MONAI `reduction`/`ignore_empty` must be pinned and stated in the README.
- Prefer **AJI+** over classic AJI (classic uses greedy 1-to-many → over-penalizes; documented in HoVer-Net source).
- `remap_label()` to contiguous IDs before AJI. Decide empty-image handling (skip vs 0) and write it down.
- Reference point: Mesmer (TissueNet paper) reports ~**0.67 F1@IoU0.5 nuclear**; whole-cell ~0.77–0.89 per platform (search-extracted from the paywalled Nature paper — treat as ballpark, not gospel).

---

## 6. Methodology

1. **Data.** `download_tissuenet.py` checks `DEEPCELL_ACCESS_TOKEN`, pulls **v1.1**, caches to `$RCAC_SCRATCH`. `loader.py` yields `(image[512,512,2], wholecell_mask, nuclear_mask)` and **normalizes** (per-image 1st–99.5th percentile; arcsinh optional) — TissueNet is **not** pre-normalized.
2. **Zero-shot benchmark — two tasks:**
   - **Whole-cell:** Cellpose-SAM, μSAM (StarDist excluded as architecturally unfit, or shown once as a labeled "known-limited" baseline).
   - **Nuclear:** StarDist (`2D_versatile_fluo` on the nuclear channel), Cellpose-SAM, μSAM.
   Run all via the subprocess harness → per-image + aggregated tables.
3. **Error analysis + viz.** Overlay pred vs GT; isolate touching-cell / boundary failures; show where each model under/over-merges adjacent cells (the boundary-F1 story made visual).
4. **Fine-tune.** Pick the best-tunable PyTorch model (**Cellpose-SAM** via `cellpose.train.train_seg`, or **μSAM** via `micro_sam.training.train_sam`) — keeps fine-tuning inside `torch-cell`. Fine-tune on a **train subset**; sweep **LR × train-data fraction** (2–3 runs). Export checkpoint.
5. **Report.** Zero-shot vs fine-tuned deltas + "which model when" guidance + honest limitations (StarDist whole-cell, CC-BY-NC weights, normalization sensitivity).

---

## 7. Compute plan (Gilbreth / Slurm)

GPU: A100 (`-p a100-40gb` is plenty; `a100-80gb` if you batch large). Budget **~20–35 GPU-hrs**: inference ~1–2 (3 short jobs), fine-tune + sweep ~15–25 (one job/run). **Inference fits the free `standby` QOS** (4 h cap, no charge); fine-tune jobs that exceed 4 h must use your account queue.

**Day-0 cluster checklist (values vary per account — confirm on-cluster):** `slist` (your `-A` account + walltime caps) · `module spider anaconda cuda cudnn` · `echo $RCAC_SCRATCH` · `myquota`.

```bash
#!/bin/bash
#SBATCH --job-name=cellseg-bench
#SBATCH --account=<your_account>      # REQUIRED — from `slist`
#SBATCH --partition=a100-40gb         # GPU model selected by partition
#SBATCH --qos=standby                 # free idle queue, 4h cap (inference only)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1             # RCAC-preferred (== --gres=gpu:1)
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00               # default is only 30 min — always set
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$RCAC_SCRATCH/cell-seg-benchmark"
module purge
module load anaconda/<ver>
module use $HOME/privatemodules
module load conda-env/<env>-py<ver>   # built with conda-env-mod
nvidia-smi                            # log hardware/driver into results for reproducibility
python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
python -m src.eval.run_benchmark --model "$MODEL" --split test --seed 42 \
       --out "$RCAC_SCRATCH/cell-seg-benchmark/results/$SLURM_JOB_ID"
```
Env build (RCAC pattern, datasets/envs in scratch — home is 25 GB): `module load anaconda/<ver>; conda-env-mod create -p $RCAC_SCRATCH/envs/torch-cell --jupyter`. Debug interactively: `sinteractive -A <acct> -p a100-40gb --gpus-per-node=1 -t 02:00:00 -c 8`. View figures via Open OnDemand at `gateway.gilbreth.rcac.purdue.edu`.

---

## 8. Revised timeline (realistic)

The original 3–4 day estimate under-counts the gated dataset + dual-env setup + metric vendoring. Budget **~5–7 focused days** (≈1.5 weeks part-time); compute overlaps coding.

| Day | Milestone | De-risks |
|-----|-----------|----------|
| **0** | **Verify `users.deepcell.org` token works** + download a TissueNet sample. Two `conda-env-mod` envs build & GPU-validate (`torch.cuda.is_available()`, `tf.config.list_physical_devices('GPU')`). | The two biggest risks (gated data, env clash) — *before* writing pipeline code. |
| 1 | `loader.py` (with normalization) + **one** wrapper running end-to-end on the cluster. | The data + one-model path. |
| 2 | All 3 wrappers + subprocess orchestrator + metrics module (vendored AJI). | The eval engine. |
| 3 | Zero-shot tables (whole-cell + nuclear tasks) + touching-cell visualizations. | The headline result. |
| 4–5 | Fine-tune (Cellpose-SAM or μSAM) + LR×fraction sweep → deltas. | The "fine-tuning buys you X" claim. |
| 6 | README writeup, repo polish, one-command repro check, results committed. | Shippable. |

## 9. Scope guardrails (explicitly OUT)

- NOT chasing SOTA or beating a paper — the credential is the **rigorous, reproducible comparison + honest failure analysis**.
- One fine-tuned model, one small sweep. Don't rabbit-hole on accuracy; if the lift is modest, the comparison + recipe still stand.
- No new architecture, no 3D, no promptable-μSAM UI (that's a stretch goal).
- StarDist is **not** forced onto whole-cell — fairness over a fuller table.

## 10. Top risks & mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **TissueNet auth gate fails** (login flaky, token never arrives) | Med | Register + verify **Day 0**. Fallback: Cellpose-relabeled figshare subset (derivative; note provenance). |
| Cellpose-SAM & μSAM torch pins conflict in one env | Low–Med | Validate both import on a single pinned torch ≥2.5/CUDA12.1; if not, split into a third subprocess env (config change only). |
| TF/PyTorch CUDA clash if co-installed | High *if* attempted | **Never co-install** — separate envs is the design, not a workaround. |
| Normalization mistakes tank all scores silently | Med | Normalize in the loader; unit-test on the `tissuenet-sample.npz`; eyeball overlays early. |
| Metric numbers non-reproducible (per-image vs pooled) | Med | Pin & document averaging conventions; vendor HoVer-Net by commit SHA; commit a tiny golden-set regression test. |
| Gilbreth queue/partition names changed (2025 modernization) | Med | Confirm via `slist`/`sinfo` Day 0; don't hardcode partition names from this doc. |
| CC-BY-NC Cellpose weights | Low | Fine for research/portfolio; state non-commercial limitation in README. |

## 11. Stretch (only if time)
MoNuSeg (H&E nuclei, ~150–250 MB, [grand-challenge](https://monuseg.grand-challenge.org/Data/)) for a second-modality generality claim · promptable μSAM / napari demo · chain segmented centroids into a toy cell-neighborhood spatial analysis (ties to your spatial-omics framing).

---

### Verdict
**BUILD.** The plan is sound; the research de-risked the four places it would have broken (env coexistence, dataset gating, StarDist fairness, metric vendoring). Start at **Day 0**: prove the token and the two envs before writing a line of pipeline code.

*Volatile facts to re-confirm at build time: exact Gilbreth partition/module versions (`slist`/`module spider`); μSAM's effective torch pin (`environment.yaml` vs docs); latest `cellpose`/`stardist`/`monai`/`micro_sam` versions at lock; full TissueNet modified-Apache license text; TissueNet µm/px (paywalled).*
