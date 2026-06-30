"""Qualitative overlays: GT vs pred boundaries, focused on touching-cell failures. Metrics env.

Builds a side-by-side panel per image: the nuclear-channel image with GT boundaries (green),
then each model's prediction boundaries (magenta) over the same image. Touching-cell over/under-
merging shows up as missing/extra boundaries between adjacent cells.

CLI:
  python -m src.viz.plots --task nuclear --split test --indices 3,7,12 \
      --masks results/masks --out results/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.loader import load_npz  # noqa: E402


def _draw(ax, img, mask, color, title):
    ax.imshow(img, cmap="gray")
    ov = np.zeros((*img.shape, 4))
    ov[find_boundaries(mask, mode="outer")] = color
    ax.imshow(ov)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def panel(img, gt, preds: dict, out_path: Path):
    n = 2 + len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("image", fontsize=10)
    axes[0].axis("off")
    _draw(axes[1], img, gt, [0, 1, 0, 1], "ground truth")
    for ax, (name, pred) in zip(axes[2:], preds.items()):
        _draw(ax, img, pred, [1, 0, 1, 1], name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["wholecell", "nuclear"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--indices", default="0,1,2", help="comma-separated image indices")
    ap.add_argument("--masks", default=str(ROOT / "results" / "masks"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()

    samples = list(load_npz(ROOT / "data" / "tissuenet" / "tissuenet_v1-1" / f"{args.split}.npz"))
    masks = {}
    for f in sorted(Path(args.masks).glob(f"*_{args.task}_{args.split}.npz")):
        model = f.stem.split("_")[0]
        masks[model] = np.load(f)["pred"]
    if not masks:
        raise SystemExit(f"no mask npz found in {args.masks} for {args.task}/{args.split}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for idx in [int(x) for x in args.indices.split(",")]:
        s = samples[idx]
        img = s.image[..., 0]  # nuclear channel for display
        gt = s.wholecell if args.task == "wholecell" else s.nuclear
        preds = {m: masks[m][idx] for m in masks}
        p = out / f"{args.task}_{args.split}_img{idx}.png"
        panel(img, gt, preds, p)
        print(f"saved {p}")


if __name__ == "__main__":
    main()
