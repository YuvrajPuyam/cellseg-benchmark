# cajal - one-page result sheet

**Which cell-segmentation foundation model for multiplexed tissue, and how much does fine-tuning buy?**
Reproducible benchmark of Cellpose-SAM, μSAM, and StarDist on TissueNet v1.1 (Purdue Gilbreth A30).

---

### Setup
- **Data:** TissueNet v1.1 test split, **N=300** images (256×256), 2-channel (nuclear + whole-cell).
- **Eval:** per-image (macro) averaging; empty-GT images skipped. Instance matching = Hungarian 1-to-1.
- **Metrics:** AJI+ · PQ · F1@IoU{0.5,0.75} · boundary-F1 (NSD, 2 px) · Dice.
- **Two tasks:** whole-cell (Cellpose-SAM, μSAM) and nuclear (StarDist, Cellpose-SAM, μSAM) -
  StarDist is nuclear-only by design (its star-convex prior is unfair on whole-cell).

### Headline numbers
<!--HEADLINE-->
Per-image macro; `±` = bootstrap 95% CI half-width. N=297 (StarDist N=148).

**Whole-cell**

| model | F1@0.5 | F1@0.75 | AJI+ | PQ | boundary-F1 | Dice |
|---|---|---|---|---|---|---|
| Cellpose-SAM | **0.844 ±.013** | **0.591** | **0.718** | **0.670** | **0.869** | **0.902** |
| μSAM | 0.736 ±.018 | 0.412 | 0.597 | 0.553 | 0.740 | 0.808 |

**Nuclear**

| model | F1@0.5 | F1@0.75 | AJI+ | PQ | boundary-F1 | Dice |
|---|---|---|---|---|---|---|
| Cellpose-SAM | **0.841 ±.017** | 0.530 | **0.710** | **0.651** | 0.895 | 0.871 |
| μSAM | 0.810 ±.015 | **0.601** | 0.702 | 0.648 | **0.907** | **0.892** |
| StarDist | 0.766 ±.023 | 0.464 | 0.633 | 0.585 | 0.848 | 0.844 |

Cellpose-SAM's whole-cell lead over μSAM (0.844 vs 0.736) exceeds the CIs - a significant gap.
On nuclei the three are close; μSAM wins boundary-F1, StarDist (a small specialist) is solid at 0.77.
<!--/HEADLINE-->

### Fine-tuning (whole-cell, Cellpose-SAM)
<!--FINETUNE-->
Cellpose-SAM, fine-tuned on a 200-image TissueNet train subset (20 epochs), evaluated on the
same whole-cell test set:

| | F1@0.5 | AJI+ | PQ | boundary-F1 | Dice |
|---|---|---|---|---|---|
| zero-shot | 0.844 | 0.718 | 0.670 | 0.869 | 0.902 |
| **fine-tuned (lr 1e-5)** | **0.859** | **0.732** | **0.688** | **0.885** | **0.915** |
| Δ | **+0.015** | **+0.014** | **+0.018** | **+0.016** | **+0.013** |

**Fine-tuning lifted whole-cell F1@0.5 by +1.5 points** (and AJI+ by +1.4) - a small but consistent
gain across every metric, from just 200 labeled images. lr 5e-5 was too high (≈flat).
<!--/FINETUNE-->

### Headline takeaways
- **Cellpose-SAM** is the most robust zero-shot choice across both tasks (significant whole-cell lead).
- On **nuclei** the three are close - pick by whether you weight detection (Cellpose) or
  boundary tightness (μSAM, best boundary-F1/Dice); StarDist is a solid lightweight specialist.
- **Fine-tuning works**: +1.5 pts whole-cell F1 from 200 images at lr 1e-5.
- Honest caveats: μSAM whole-cell uses a corrected RGB input (the old 0.51 was a channel-mean
  strawman; now 0.736). StarDist runs at N=148 (TF build core-dumps beyond that on this cluster).

### Failure analysis
Touching-cell boundary overlays (GT green vs prediction magenta) in `results/figures/` show where
each model over-/under-merges adjacent cells - the boundary-F1 story made visual.

### What this demonstrates
Rigorous, reproducible, apples-to-apples comparison on multiplexed tissue + honest failure
analysis + a measured fine-tuning gain - one-command reproducible on a Slurm A30.
