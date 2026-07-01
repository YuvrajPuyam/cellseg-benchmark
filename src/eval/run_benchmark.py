"""Benchmark orchestrator. Three subcommands, each meant to run in the appropriate conda env:

  infer  (model env)   : run a model over a split, write masks/{model}_{task}_{split}.npz (pred+gt)
  score  (metrics env) : read a pred npz, compute metrics → results/{stem}_agg.json + _per_image.csv
  report (metrics env) : collate all *_agg.json into the two benchmark tables (whole-cell, nuclear)

Model→task eligibility: whole-cell = cellpose, microsam; nuclear = stardist, cellpose, microsam.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.loader import load_npz, sample_indices  # noqa: E402

WRAPPERS = {
    "cellpose": "src.models.cellpose_wrapper",
    "microsam": "src.models.microsam_wrapper",
    "stardist": "src.models.stardist_wrapper",
}
ELIGIBLE = {"wholecell": ("cellpose", "microsam"), "nuclear": ("stardist", "cellpose", "microsam")}
REPORT_METRICS = ["aji_plus", "aji", "pq", "f1@0.5", "f1@0.75", "boundary_f1", "dice"]


def _split_path(split: str) -> Path:
    return ROOT / "data" / "tissuenet" / "tissuenet_v1-1" / f"{split}.npz"


def cmd_infer(args):
    if args.model not in ELIGIBLE[args.task]:
        raise SystemExit(f"{args.model} is not eligible for task {args.task} (eligible: {ELIGIBLE[args.task]})")
    wrap = importlib.import_module(WRAPPERS[args.model])
    preds, gts = [], []
    t0 = time.time()
    keep = None  # seeded-random subset of image indices (overrides first-N --limit when set)
    if getattr(args, "sample", 0):
        n_total = int(np.load(_split_path(args.split), allow_pickle=True)["X"].shape[0])
        keep = set(int(j) for j in sample_indices(n_total, args.sample, args.seed))
        print(f"  sampling {len(keep)}/{n_total} images (seed={args.seed})", flush=True)
    for i, s in enumerate(load_npz(_split_path(args.split), normalize=False)):
        if keep is not None:
            if i not in keep:
                continue
        elif args.limit and i >= args.limit:
            break
        gt = s.wholecell if args.task == "wholecell" else s.nuclear
        if gt is None:
            continue
        mask = wrap.segment(s.image, task=args.task)
        preds.append(np.asarray(mask, np.int32))
        gts.append(np.asarray(gt, np.int32))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} images in {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if getattr(args, "tag", "") else ""
    f = out / f"{args.model}_{args.task}_{args.split}{tag}.npz"
    np.savez_compressed(f, pred=np.stack(preds), gt=np.stack(gts))
    print(f"INFER_DONE {f} n={len(preds)} time={time.time() - t0:.0f}s")


def cmd_score(args):
    from src.eval.metrics import score_split

    d = np.load(args.pred_npz)
    preds, gts = list(d["pred"]), list(d["gt"])
    records, agg = score_split(gts, preds, skip_empty=not args.keep_empty, boundary=not args.no_boundary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.pred_npz).stem
    if records:
        with open(out / f"{stem}_per_image.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
    json.dump(agg, open(out / f"{stem}_agg.json", "w"), indent=2)
    print(f"SCORE_DONE {stem}: f1@0.5={agg.get('f1@0.5_mean'):.4f} "
          f"aji_plus={agg.get('aji_plus_mean'):.4f} n={agg['n_images']}")


def _fmt_metric(agg, k):
    """Render one metric cell as 'mean ±ci' (half-width of the bootstrap 95% CI) when the CI is
    present, else the plain mean, else an em dash. The plain mean stays available in the JSON."""
    v = agg.get(f"{k}_mean")
    if not isinstance(v, (int, float)):
        return "-"
    lo, hi = agg.get(f"{k}_ci_lo"), agg.get(f"{k}_ci_hi")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and np.isfinite(lo) and np.isfinite(hi):
        return f"{v:.3f} ±{(hi - lo) / 2:.3f}"
    return f"{v:.3f}"


def cmd_report(args):
    results = Path(args.results)
    rows = {}  # (task) -> list of (model, agg)
    for f in sorted(results.glob("*_agg.json")):
        parts = f.stem.replace("_agg", "").split("_")  # model_task_split (zero-shot only)
        if len(parts) != 3:
            continue  # skip tagged (fine-tuned) aggs - those go in the fine-tune delta, not the main table
        agg = json.load(open(f))
        model, task = parts[0], parts[1]
        rows.setdefault(task, []).append((model, agg))

    lines = ["# cajal benchmark results\n"]
    for task in ("wholecell", "nuclear"):
        if task not in rows:
            continue
        lines.append(f"\n## {task} task\n")
        header = "| model | " + " | ".join(REPORT_METRICS) + " | n |"
        sep = "|" + "---|" * (len(REPORT_METRICS) + 2)
        lines += [header, sep]
        for model, agg in sorted(rows[task]):
            cells = [_fmt_metric(agg, k) for k in REPORT_METRICS]
            lines.append(f"| {model} | " + " | ".join(cells) + f" | {agg.get('n_images', '?')} |")
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nREPORT_DONE -> {out}")


def main():
    ap = argparse.ArgumentParser(description="cajal benchmark orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("infer")
    pi.add_argument("--model", required=True, choices=list(WRAPPERS))
    pi.add_argument("--task", required=True, choices=list(ELIGIBLE))
    pi.add_argument("--split", default="test")
    pi.add_argument("--limit", type=int, default=0, help="first-N images (0 = all)")
    pi.add_argument("--sample", type=int, default=0,
                    help="seeded-random N images instead of first-N --limit (0 = off)")
    pi.add_argument("--seed", type=int, default=0, help="rng seed for --sample")
    pi.add_argument("--tag", default="", help="suffix for output npz (e.g. ft_lr1e-5)")
    pi.add_argument("--out", default=str(ROOT / "results" / "masks"))
    pi.set_defaults(func=cmd_infer)

    ps = sub.add_parser("score")
    ps.add_argument("--pred-npz", required=True)
    ps.add_argument("--out", default=str(ROOT / "results"))
    ps.add_argument("--no-boundary", action="store_true")
    ps.add_argument("--keep-empty", action="store_true")
    ps.set_defaults(func=cmd_score)

    pr = sub.add_parser("report")
    pr.add_argument("--results", default=str(ROOT / "results"))
    pr.add_argument("--out", default=str(ROOT / "results" / "benchmark_tables.md"))
    pr.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
