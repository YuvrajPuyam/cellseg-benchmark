# cajal - cold-start runbook for Gilbreth

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
`cajal` is a sibling of `HOI/laplace` - a separate project, not laplace work.

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
- [x] Repo scaffold present (envs, wrapper contract, configs, loader, downloader).
- [x] **TissueNet token validated** (2026-06-29) and **v1.1 downloaded + md5-verified on cluster**
      at `data/tissuenet/tissuenet_v1-1/{train,val,test}.npz` (2580×512² / 3118×256² / 1324×256²,
      X=[nuclear,whole-cell], y=[whole-cell,nuclear]). Token lives only in local `.env` (never on
      the cluster); the download used a presigned URL resolved locally.
- [x] `data/loader.py` written + unit-tested (percentile normalization validated on real sample).
- [ ] Conda envs not yet built on Gilbreth (`conda-env-mod`, see §4).
- [ ] Model wrappers / metrics not yet validated against real packages.
- [ ] Zero-shot benchmark + fine-tune not yet run.

## 3. Update code on Gilbreth

```bash
scp -o BatchMode=yes -o ConnectTimeout=20 D:/cajal/<path> \
  gilbreth.rcac.purdue.edu:/scratch/gilbreth/gupta596/MotionGen/HOI/cajal/<path>
```

## 4. Build the conda envs  (the ACTUAL working recipe - `module`/`conda-env-mod` are flaky non-interactively)

`module` and `$RCAC_SCRATCH` are NOT set in non-interactive ssh. conda is on PATH directly at
`/apps/external/anaconda/2025.06`. Build with plain conda (see `scripts/gilbreth/build_envs*.sh`):

```bash
source /apps/external/anaconda/2025.06/etc/profile.d/conda.sh
SCRATCH=/scratch/gilbreth/gupta596 ; export PIP_CACHE_DIR=$SCRATCH/.pipcache CONDA_PKGS_DIRS=$SCRATCH/.condapkgs
conda create -y -p $SCRATCH/envs/torch-cell python=3.10
conda activate $SCRATCH/envs/torch-cell
pip install torch==2.5.* torchvision --index-url https://download.pytorch.org/whl/cu121
pip install cellpose micro_sam            # micro_sam pulls Qt → do NOT `set -u` (qt hook = unbound var)
# stardist-tf:  pip install "tensorflow[and-cuda]==2.15.*" stardist csbdeep
# metrics    :  conda create ... numpy scipy scikit-image matplotlib pandas ; pip install stardist monai tifffile
```

Built envs live at `$SCRATCH/envs/{torch-cell,stardist-tf,metrics}`.

**GPU gotchas (hard-won):**
- The **login node GPU is MPS-gated** → `torch.cuda.is_available()` is **False** there (Error 805).
  Run all GPU work via `sbatch` on a compute node (there it's True). Driver 590 / CUDA 13.1.
- **Compute nodes have no internet** → pre-cache model weights on the login node first:
  cpsam → `~/.cellpose/models/cpsam_v2`; micro_sam vit_b_lm via `get_predictor_and_segmenter(..., device="cpu")`.
- Gilbreth requires explicit `--mem` (use `--mem=60G` per a30 GPU).

## 5. Run the benchmark

```bash
# infer (one model/task per job; weights pre-cached). torch-cell for cellpose/microsam, stardist-tf for stardist:
sbatch --export=ALL,ENV=torch-cell,MODEL=cellpose,TASK=wholecell,LIMIT=300 slurm/benchmark.sbatch
sbatch --export=ALL,ENV=torch-cell,MODEL=microsam,TASK=nuclear,LIMIT=300  slurm/benchmark.sbatch
sbatch --export=ALL,ENV=stardist-tf,MODEL=stardist,TASK=nuclear,LIMIT=300 slurm/benchmark.sbatch
# score (metrics env, CPU - runs fine on the login node, no GPU/MPS needed):
conda activate $SCRATCH/envs/metrics
python -m src.eval.run_benchmark score --pred-npz results/masks/cellpose_wholecell_test.npz
python -m src.eval.run_benchmark report          # collate → results/benchmark_tables.md
# figures:
python -m src.viz.plots --task nuclear --split test --indices 3,7,12
# fine-tune (small) then eval the checkpoint via CAJAL_CELLPOSE_CKPT (see src/train/finetune.py):
sbatch --export=ALL,LR=1e-5,FRACTION=1.0,EPOCHS=20,MAXN=200 slurm/finetune.sbatch
```

## 6. Gotchas

- `csml` = A30 only. Don't request a100/h100 partitions - the job will never start.
- Default walltime on Gilbreth is 30 min - **always set `--time`** in the sbatch.
- A30 = 24 GB. Cellpose/μSAM at 512×512 fit comfortably; batch modestly for fine-tune.
- TF (StarDist) and PyTorch (Cellpose/μSAM) must stay in separate envs - never co-install.
- Confirm anaconda + cuda/cudnn module versions with `module spider` before building envs.

## 7. The plan

Full research-hardened spec: [`project-spec.md`](project-spec.md). Original handoff:
[`handoff.md`](handoff.md). Both predate the cluster wiring; coordinates here supersede theirs.
