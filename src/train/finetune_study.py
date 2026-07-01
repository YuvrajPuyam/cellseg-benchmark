"""Rigorous fine-tune LR study - select the LR on VAL, then report the lift on TEST.

Why this file exists
--------------------
The previous fine-tune workflow (``scripts/gilbreth/eval_finetune.sh``) fine-tuned at several
learning rates and then compared every checkpoint *directly on the test split*, reporting the
best one. That selects the LR on the test set - the reported test lift is optimistically biased
(you peeked at the test set to choose a hyper-parameter). This driver removes that leak:

    1.  SWEEP   : for each LR in --lrs, fine-tune on the TRAIN subset (one seed) and evaluate
                  on the **val** split only.
    2.  SELECT  : pick the LR with the best val score (default metric: f1@0.5).
    3.  CONFIRM : re-fine-tune *that* LR across >=3 seeds and evaluate each on the **test** split.
    4.  REPORT  : write results/finetune_study.md with the val-selected LR and the test
                  mean +/- 95% CI delta vs the zero-shot baseline.

The test split is touched ONLY in step 3, and never influences which LR is chosen.

It does not run anything itself except subprocesses; it shells out to the EXISTING interfaces:

    fine-tune:  python -m src.train.finetune --lr <LR> --fraction <F> --epochs <E>
                    --max-n <N> --seed <S> --out <CKPT_DIR>
    evaluate :  CAJAL_CELLPOSE_CKPT=<ckpt> python -m src.eval.run_benchmark infer
                    --model cellpose --task wholecell --split <val|test> --tag <TAG> --limit <L>
                python -m src.eval.run_benchmark score --pred-npz <masks/...npz> --out <RESULTS>

The fine-tune step needs the GPU env (torch-cell) and the score step needs the metrics env, so
in production these run as separate cluster jobs. This driver is the single-env reference
orchestration (e.g. an interactive GPU node with both stardist excluded); the exact commands it
issues are printed (and logged) so they can also be lifted into sbatch scripts. Nothing here is
specific to the local machine - it only assembles + runs argv lists.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Fine-tune is whole-cell cellpose only (the only model that consumes CAJAL_CELLPOSE_CKPT).
MODEL = "cellpose"
TASK = "wholecell"
# Match cellpose.train.train_seg's save layout: save_path/models/<model_name>.
CKPT_SUBDIR = "models"
FT_CKPT_RE = re.compile(r"^FT_CKPT\s+(.+)$", re.MULTILINE)


def _run(argv, env=None, capture=False, dry=False):
    """Run a subprocess (argv list). Returns CompletedProcess. Prints the exact command first."""
    shown = " ".join(argv)
    if env:
        extra = {k: v for k, v in env.items() if os.environ.get(k) != v}
        if extra:
            shown = " ".join(f"{k}={v}" for k, v in extra.items()) + " " + shown
    print(f"[study] $ {shown}", flush=True)
    if dry:
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    full_env = {**os.environ, **(env or {})}
    res = subprocess.run(
        argv, env=full_env, cwd=str(ROOT),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
    )
    if capture and res.stdout:
        print(res.stdout, flush=True)
    if res.returncode != 0:
        raise SystemExit(f"[study] command failed (rc={res.returncode}): {shown}")
    return res


def _finetune(lr, fraction, epochs, max_n, seed, ckpt_dir, py, dry):
    """Fine-tune one (lr, seed); return the checkpoint path parsed from the FT_CKPT line."""
    argv = [*py, "-m", "src.train.finetune",
            "--lr", repr_lr(lr), "--fraction", str(fraction), "--epochs", str(epochs),
            "--max-n", str(max_n), "--seed", str(seed), "--out", str(ckpt_dir)]
    res = _run(argv, capture=True, dry=dry)
    if dry:
        # Reconstruct the deterministic path finetune.py would write (kept in sync with its naming).
        n = max_n if max_n else "ALL"
        name = f"cpsam_ft_lr{repr_lr(lr)}_frac{fraction}_n{n}"
        if seed != 42:  # mirror finetune.py: default seed keeps the legacy (un-suffixed) name
            name += f"_seed{seed}"
        return str(Path(ckpt_dir) / CKPT_SUBDIR / name)
    m = FT_CKPT_RE.search(res.stdout or "")
    if not m:
        raise SystemExit("[study] could not find 'FT_CKPT <path>' in finetune.py output")
    return m.group(1).strip()


def _eval_checkpoint(ckpt, split, tag, limit, results_dir, masks_dir, py, dry):
    """Infer (with CAJAL_CELLPOSE_CKPT=ckpt) then score on `split`; return the agg dict path."""
    infer = [*py, "-m", "src.eval.run_benchmark", "infer",
             "--model", MODEL, "--task", TASK, "--split", split,
             "--tag", tag, "--out", str(masks_dir)]
    if limit:
        infer += ["--limit", str(limit)]
    _run(infer, env={"CAJAL_CELLPOSE_CKPT": ckpt}, capture=True, dry=dry)

    masks_npz = Path(masks_dir) / f"{MODEL}_{TASK}_{split}_{tag}.npz"
    score = [*py, "-m", "src.eval.run_benchmark", "score",
             "--pred-npz", str(masks_npz), "--out", str(results_dir)]
    _run(score, capture=True, dry=dry)
    return Path(results_dir) / f"{masks_npz.stem}_agg.json"


def repr_lr(lr) -> str:
    """Format an LR the way the CLI/checkpoint names expect (e.g. 1e-05 -> '1e-05')."""
    if isinstance(lr, str):
        return lr
    return f"{lr:g}"


def _load_mean(agg_path, metric, dry):
    if dry or not Path(agg_path).exists():
        return float("nan")
    agg = json.load(open(agg_path))
    return float(agg.get(f"{metric}_mean", float("nan")))


def _mean_ci(values, alpha=0.05):
    """Normal-approx mean +/- CI across the per-seed means (small N → t-ish via 1.96 fallback).

    Returns (mean, lo, hi, sem). With <2 finite values the CI is (nan, nan).
    """
    vals = [v for v in values if v is not None and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = sum(vals) / n
    if n < 2:
        return (mean, float("nan"), float("nan"), float("nan"))
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sem = math.sqrt(var / n)
    # 95% two-sided t critical values for small dof; fall back to 1.96 for larger n.
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    half = tcrit * sem
    return (mean, mean - half, mean + half, sem)


def main():
    ap = argparse.ArgumentParser(
        description="Rigorous fine-tune LR study: select LR on val, report lift on test.")
    ap.add_argument("--lrs", default="1e-5,5e-5,1e-4",
                    help="comma-separated LR sweep evaluated on val (default 1e-5,5e-5,1e-4)")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated seeds for the test confirmation of the selected LR "
                         "(>=3 recommended; default 0,1,2)")
    ap.add_argument("--sweep-seed", type=int, default=0,
                    help="seed used for the (single) val sweep fine-tunes (default 0)")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-n", type=int, default=200, help="cap #train images (0 = all)")
    ap.add_argument("--metric", default="f1@0.5",
                    help="agg metric used for val selection AND the reported delta (default f1@0.5)")
    ap.add_argument("--val-limit", type=int, default=0, help="cap val images during the sweep (0=all)")
    ap.add_argument("--test-limit", type=int, default=0, help="cap test images (0=all)")
    ap.add_argument("--baseline-agg", default=str(ROOT / "results" / f"{MODEL}_{TASK}_test_agg.json"),
                    help="zero-shot test agg json to diff against (must exist before reporting)")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "results" / "checkpoints"))
    ap.add_argument("--masks-dir", default=str(ROOT / "results" / "masks"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "finetune_study.md"))
    ap.add_argument("--python", default=sys.executable,
                    help="python used for the subprocesses (default: current interpreter)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact commands without executing (for review/sbatch lifting)")
    args = ap.parse_args()

    py = args.python.split()
    lrs = [s.strip() for s in args.lrs.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    dry = args.dry_run
    t0 = time.time()

    # ---- 1+2. SWEEP on val, SELECT best LR -------------------------------------------------
    print(f"[study] === SWEEP: {len(lrs)} LRs on val (seed={args.sweep_seed}) ===", flush=True)
    sweep = []  # list of (lr, val_score, ckpt, agg_path)
    for lr in lrs:
        ckpt = _finetune(lr, args.fraction, args.epochs, args.max_n, args.sweep_seed,
                         args.ckpt_dir, py, dry)
        tag = f"ftsweep_lr{repr_lr(lr)}_seed{args.sweep_seed}".replace(".", "p")
        agg = _eval_checkpoint(ckpt, "val", tag, args.val_limit,
                              args.results_dir, args.masks_dir, py, dry)
        score = _load_mean(agg, args.metric, dry)
        print(f"[study] val {args.metric} @ lr={lr}: {score:.4f}", flush=True)
        sweep.append((lr, score, ckpt, str(agg)))

    if dry:
        best_lr = lrs[0]  # arbitrary in dry-run; commands for the rest are still shown below
        print(f"[study] (dry-run) would SELECT best-val LR from {lrs}; using {best_lr} for the plan",
              flush=True)
    else:
        finite = [s for s in sweep if math.isfinite(s[1])]
        if not finite:
            raise SystemExit("[study] no finite val scores - cannot select an LR")
        best_lr, best_val, _, _ = max(finite, key=lambda s: s[1])
        print(f"[study] === SELECTED LR={best_lr} (val {args.metric}={best_val:.4f}) ===", flush=True)

    # ---- 3. CONFIRM selected LR on test across seeds ---------------------------------------
    print(f"[study] === CONFIRM: LR={best_lr} on test over seeds {seeds} ===", flush=True)
    test_scores, test_aggs = [], []
    for seed in seeds:
        ckpt = _finetune(best_lr, args.fraction, args.epochs, args.max_n, seed,
                         args.ckpt_dir, py, dry)
        tag = f"ftbest_lr{repr_lr(best_lr)}_seed{seed}".replace(".", "p")
        agg = _eval_checkpoint(ckpt, "test", tag, args.test_limit,
                              args.results_dir, args.masks_dir, py, dry)
        score = _load_mean(agg, args.metric, dry)
        print(f"[study] test {args.metric} @ seed={seed}: {score:.4f}", flush=True)
        test_scores.append(score)
        test_aggs.append(str(agg))

    # ---- 4. REPORT -------------------------------------------------------------------------
    base_mean = _load_mean(args.baseline_agg, args.metric, dry)
    ft_mean, ft_lo, ft_hi, ft_sem = _mean_ci(test_scores)
    # Per-seed deltas vs the single zero-shot baseline; CI over seeds.
    deltas = [s - base_mean for s in test_scores] if math.isfinite(base_mean) else []
    d_mean, d_lo, d_hi, _ = _mean_ci(deltas) if deltas else (float("nan"),) * 4

    _write_report(args, lrs, seeds, sweep, best_lr, test_scores, test_aggs,
                  base_mean, ft_mean, ft_lo, ft_hi, d_mean, d_lo, d_hi, dry, time.time() - t0)
    print(f"[study] STUDY_DONE -> {args.out} ({time.time() - t0:.0f}s)", flush=True)


def _write_report(args, lrs, seeds, sweep, best_lr, test_scores, test_aggs,
                  base_mean, ft_mean, ft_lo, ft_hi, d_mean, d_lo, d_hi, dry, secs):
    metric = args.metric

    def f(x):
        return "-" if x is None or not math.isfinite(x) else f"{x:.4f}"

    L = ["# Fine-tune LR study (val-selected, test-reported)\n",
         "_LR is chosen on the **val** split; the **test** split is used only to report the "
         "final lift, across multiple seeds. This avoids selecting the LR on the test set._\n",
         f"- model/task: `{MODEL}` / `{TASK}`",
         f"- train subset: fraction={args.fraction}, max_n={args.max_n}, epochs={args.epochs}",
         f"- selection metric: `{metric}`  (val sweep seed={args.sweep_seed})",
         f"- zero-shot baseline agg: `{args.baseline_agg}`",
         f"- wall time: {secs:.0f}s" + ("  _(dry-run: no commands executed)_" if dry else ""),
         "",
         "## 1. LR sweep on val\n",
         f"| lr | val {metric} | checkpoint |",
         "|---|---|---|"]
    for lr, score, ckpt, _agg in sweep:
        mark = " **(selected)**" if str(lr) == str(best_lr) else ""
        L.append(f"| {lr}{mark} | {f(score)} | `{ckpt}` |")

    L += ["",
          f"## 2. Selected LR = `{best_lr}` confirmed on test ({len(seeds)} seeds)\n",
          f"| seed | test {metric} | delta vs zero-shot | agg |",
          "|---|---|---|---|"]
    for seed, score, agg in zip(seeds, test_scores, test_aggs):
        dv = (score - base_mean) if (math.isfinite(score) and math.isfinite(base_mean)) else float("nan")
        L.append(f"| {seed} | {f(score)} | {f(dv)} | `{Path(agg).name}` |")

    L += ["",
          "## 3. Headline\n",
          f"- zero-shot test {metric}: **{f(base_mean)}**",
          f"- fine-tuned (LR={best_lr}) test {metric}: **{f(ft_mean)}**  "
          f"95% CI [{f(ft_lo)}, {f(ft_hi)}]  (n={len(seeds)} seeds)",
          f"- **delta vs zero-shot: {f(d_mean)}  95% CI [{f(d_lo)}, {f(d_hi)}]**",
          ""]
    sig = ("excludes 0 → significant lift" if (math.isfinite(d_lo) and math.isfinite(d_hi)
            and (d_lo > 0 or d_hi < 0)) else "includes 0 → not significant at 95%")
    if math.isfinite(d_lo):
        L.append(f"- CI {sig}.")
    L.append("\n_CI is a t-interval over the per-seed means (small N). The per-image bootstrap CI "
             "for any single run lives in its `*_agg.json` (`{metric}_ci_lo/hi`)._")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
