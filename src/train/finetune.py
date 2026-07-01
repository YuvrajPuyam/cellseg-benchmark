"""Fine-tune Cellpose-SAM on a TissueNet whole-cell subset (env: torch-cell).

Produces a checkpoint per (lr, data-fraction). Evaluate a checkpoint by running the normal
benchmark with CAJAL_CELLPOSE_CKPT=<path>:
    CAJAL_CELLPOSE_CKPT=results/checkpoints/<name> \
      python -m src.eval.run_benchmark infer --model cellpose --task wholecell --split test
then `score` as usual → compare to the zero-shot row.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.loader import load_npz  # noqa: E402


def load_train_subset(fraction: float, seed: int = 42, max_n: int = 0):
    samples = list(load_npz(ROOT / "data" / "tissuenet" / "tissuenet_v1-1" / "train.npz", normalize=False))
    idx = np.random.default_rng(seed).permutation(len(samples))
    k = int(len(samples) * fraction)
    if max_n:
        k = min(k, max_n)
    chosen = idx[:k]
    imgs = [samples[i].image for i in chosen]
    labels = [samples[i].wholecell for i in chosen]
    return imgs, labels


def _seed_everything(seed: int):
    """Seed numpy + torch (incl. CUDA) so a given --seed gives a reproducible fine-tune.

    Without this, only the train-subset *selection* was seeded; the optimisation itself
    (weight init deltas, dropout, data shuffling inside cellpose) was nondeterministic, so
    "3 seeds" produced near-identical runs. torch is imported lazily to keep this module
    importable in the metrics/local env.
    """
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def run(lr: float, fraction: float, epochs: int, out: str, seed: int = 42, max_n: int = 0):
    from cellpose import models, train

    _seed_everything(seed)
    # train subset is drawn ONLY from train.npz (never val/test) - the held-out splits are
    # untouched here; LR selection happens against val in src/train/finetune_study.py.
    imgs, labels = load_train_subset(fraction, seed, max_n)
    print(f"[ft] {len(imgs)} train images, lr={lr}, epochs={epochs}, seed={seed}", flush=True)
    m = models.CellposeModel(gpu=True)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the legacy name for the default seed so existing tooling that globs `*_n{N}`
    # (e.g. scripts/gilbreth/eval_finetune.sh) still matches; tag non-default seeds so the
    # multi-seed study writes distinct checkpoints instead of overwriting one another.
    name = f"cpsam_ft_lr{lr}_frac{fraction}_n{len(imgs)}"
    if seed != 42:
        name += f"_seed{seed}"
    t0 = time.time()
    path = train.train_seg(
        m.net, train_data=imgs, train_labels=labels,
        channel_axis=2, normalize=True, n_epochs=epochs,
        learning_rate=lr, weight_decay=0.1, save_path=str(out_dir), model_name=name,
    )
    # FT_CKPT line is machine-parseable: finetune_study.py reads it to locate the checkpoint
    # to hand to CAJAL_CELLPOSE_CKPT for evaluation.
    print(f"FT_DONE {name} in {time.time() - t0:.0f}s -> {path}", flush=True)
    print(f"FT_CKPT {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-n", type=int, default=0, help="cap #train images (quick run)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "checkpoints"))
    args = ap.parse_args()
    run(args.lr, args.fraction, args.epochs, args.out, args.seed, args.max_n)


if __name__ == "__main__":
    main()
