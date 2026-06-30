# cajal benchmark results

TissueNet v1.1 test split · per-image (macro) averaging · `mean ±` half-width of a percentile
bootstrap 95% CI (2000 resamples). StarDist runs at N=148 (the largest stable subset before its
TF build core-dumps on this cluster); Cellpose-SAM and μSAM at N=297.

## wholecell task

| model | aji_plus | aji | pq | f1@0.5 | f1@0.75 | boundary_f1 | dice | n |
|---|---|---|---|---|---|---|---|---|
| cellpose | 0.718 ±0.012 | 0.676 ±0.012 | 0.670 ±0.014 | 0.844 ±0.013 | 0.591 ±0.024 | 0.869 ±0.014 | 0.902 ±0.010 | 297 |
| microsam | 0.597 ±0.015 | 0.561 ±0.015 | 0.553 ±0.015 | 0.736 ±0.018 | 0.412 ±0.020 | 0.740 ±0.020 | 0.808 ±0.015 | 297 |

## nuclear task

| model | aji_plus | aji | pq | f1@0.5 | f1@0.75 | boundary_f1 | dice | n |
|---|---|---|---|---|---|---|---|---|
| cellpose | 0.710 ±0.014 | 0.691 ±0.014 | 0.651 ±0.015 | 0.841 ±0.017 | 0.530 ±0.026 | 0.895 ±0.014 | 0.871 ±0.011 | 297 |
| microsam | 0.702 ±0.012 | 0.661 ±0.013 | 0.648 ±0.013 | 0.810 ±0.015 | 0.601 ±0.019 | 0.907 ±0.008 | 0.892 ±0.006 | 297 |
| stardist | 0.633 ±0.017 | 0.596 ±0.017 | 0.585 ±0.020 | 0.766 ±0.023 | 0.464 ±0.025 | 0.848 ±0.014 | 0.844 ±0.012 | 148 |

_Note: μSAM whole-cell uses the corrected RGB=[membrane, nuclear, membrane] input (the earlier
channel-mean grayscale gave 0.512 — a harness strawman, now 0.736)._
