# cajal — cold-start runbook for Gilbreth

Sister project to `laplace`. Everything a fresh session needs to operate the cell/tissue
segmentation benchmark on Purdue's Gilbreth cluster. Copy-pasteable. Paths are hard-coded.

---

## 0. The one fact block

```
HOST   = gilbreth.rcac.purdue.edu        # ~/.ssh/config resolves to gupta596, key-based, bypasses Duo
REPO   = /scratch/gilbreth/gupta596/MotionGen/HOI/cajal      # ALL work stays under here (containment)
ACCT   = csml        PARTITION = a30     # csml has A30 ONLY (24 GB, 24 cpu/node); no A100/H100
SCRATCH= /scratch/gilbreth/gupta596      # $RCAC_SCRATCH; home is 25 GB so keep data/envs in scratch
```

Containment: only write under `REPO`. Never `$HOME`, never `HOI/laplace`, never other scratch.
`cajal` is a sibling of `HOI/laplace` — a separate project, not laplace work.

## 1. SSH (non-interactive, key-based)

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 gilbreth.rcac.purdue.edu "<remote cmd>"
# strip login banner noise:
... 2>&1 | grep -avE "OneDrive|Welcome|Last login|^\*\*\*|BoilerKey|SSH keys is Required|^####"
```
Never hold an interactive session. Submit detached via `sbatch`, poll with short `squeue`/`grep`.
`scp` single files (no `rsync` in Git Bash).

## 2. Day-0 status (what's done / what's blocked)

- [x] SSH access verified (2026-06-29): logs in as `gupta596`, scratch has room.
- [x] `csml` partitions confirmed via `slist`: **A30 only**.
- [x] Repo scaffold present (envs, wrapper contract, configs, metrics/loader stubs).
- [ ] **TissueNet token — the #1 remaining human blocker.** Register at `users.deepcell.org`,
      `export DEEPCELL_ACCESS_TOKEN=...`, then `python data/download_tissuenet.py --check`.
- [ ] Conda envs not yet built on Gilbreth (`conda-env-mod`, see §4).
- [ ] Wrappers / loader / metrics not yet validated against real packages.

## 3. Update code on Gilbreth

```bash
scp -o BatchMode=yes -o ConnectTimeout=20 D:/cajal/<path> \
  gilbreth.rcac.purdue.edu:/scratch/gilbreth/gupta596/MotionGen/HOI/cajal/<path>
```

## 4. Build the conda envs (datasets/envs in scratch — home is 25 GB)

```bash
# on Gilbreth, under REPO:
module load anaconda                                  # confirm exact ver via `module spider anaconda`
conda-env-mod create -p $RCAC_SCRATCH/envs/torch-cell --jupyter   # cellpose + micro_sam (PyTorch)
conda-env-mod create -p $RCAC_SCRATCH/envs/stardist-tf --jupyter  # stardist + TF2 (separate; CUDA clash)
conda-env-mod create -p $RCAC_SCRATCH/envs/metrics --jupyter      # scoring (framework-agnostic)
# then pip-install per envs/*.yml inside each, and GPU-validate:
#   torch-cell : python -c "import cellpose, micro_sam, torch; print(torch.cuda.is_available())"
#   stardist-tf: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 5. Run the benchmark

```bash
# inference (zero-shot) — one model per job, GPU handed off between them
sbatch slurm/benchmark.sbatch        # wired for --account=csml --partition=a30
# fine-tune + LR x data-fraction sweep
sbatch slurm/finetune.sbatch
# poll:
ssh ... 'squeue -u gupta596'
```

## 6. Gotchas

- `csml` = A30 only. Don't request a100/h100 partitions — the job will never start.
- Default walltime on Gilbreth is 30 min — **always set `--time`** in the sbatch.
- A30 = 24 GB. Cellpose/μSAM at 512×512 fit comfortably; batch modestly for fine-tune.
- TF (StarDist) and PyTorch (Cellpose/μSAM) must stay in separate envs — never co-install.
- Confirm anaconda + cuda/cudnn module versions with `module spider` before building envs.

## 7. The plan

Full research-hardened spec: [`project-spec.md`](project-spec.md). Original handoff:
[`handoff.md`](handoff.md). Both predate the cluster wiring; coordinates here supersede theirs.
