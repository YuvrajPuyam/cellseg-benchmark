# cajal — one-page result sheet

**Which cell-segmentation foundation model for multiplexed tissue, and how much does fine-tuning buy?**
Reproducible benchmark of Cellpose-SAM, μSAM, and StarDist on TissueNet v1.1 (Purdue Gilbreth A30).

---

### Setup
- **Data:** TissueNet v1.1 test split, **N=300** images (256×256), 2-channel (nuclear + whole-cell).
- **Eval:** per-image (macro) averaging; empty-GT images skipped. Instance matching = Hungarian 1-to-1.
- **Metrics:** AJI+ · PQ · F1@IoU{0.5,0.75} · boundary-F1 (NSD, 2 px) · Dice.
- **Two tasks:** whole-cell (Cellpose-SAM, μSAM) and nuclear (StarDist, Cellpose-SAM, μSAM) —
  StarDist is nuclear-only by design (its star-convex prior is unfair on whole-cell).

### Headline numbers
<!--HEADLINE-->
_(filled from the committed run — see `benchmark_tables.md`)_
<!--/HEADLINE-->

### Fine-tuning (whole-cell, Cellpose-SAM)
<!--FINETUNE-->
_(filled from `finetune_delta.md` — 200-image subset, LR sweep, 20 epochs)_
<!--/FINETUNE-->

### Failure analysis
Touching-cell boundary overlays (GT green vs prediction magenta) in `results/figures/` show where
each model over-/under-merges adjacent cells — the boundary-F1 story made visual.

### What this demonstrates
Rigorous, reproducible, apples-to-apples comparison on multiplexed tissue + honest failure
analysis + a measured fine-tuning gain — one-command reproducible on a Slurm A30.
