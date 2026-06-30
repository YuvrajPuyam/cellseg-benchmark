"""Instance + semantic segmentation metrics (runs in the `metrics` conda env).

Anchored on stardist.matching (Hungarian 1-to-1 → PQ, SQ, F1@IoU, precision/recall). Gaps filled:
AJI/AJI+ and boundary-F1 (NSD). Both are implemented in O(image) form here (contingency table +
distance transforms) because the reference HoVer-Net AJI and MONAI NSD are O(cells × image) /
very slow on dense tissue (100s of cells/image → ~80 s/image). The fast AJI is numerically
identical to the vendored HoVer-Net version (cross-checked in tests against third_party/).

Averaging convention (pinned, stated in the README):
  * per-image (macro) is the HEADLINE — mean over images of each per-image metric.
  * dataset-pooled F1/PQ also reported via stardist.matching_dataset(by_image=False).
Empty-GT images are skipped by default. Masks are integer instance labels (0 = background).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → third_party importable
from third_party.hover_net_stats_utils import remap_label as _hovernet_remap  # noqa: E402

IOU_THRESHOLDS = (0.5, 0.75)
N_BOOTSTRAP = 2000
CI_SEED = 0


def bootstrap_ci(values, n_boot: int = N_BOOTSTRAP, seed: int = CI_SEED, alpha: float = 0.05):
    """Percentile bootstrap (lo, hi) for the MEAN of a per-image metric array.

    Resamples images with replacement ``n_boot`` times (seeded → reproducible) and returns the
    (alpha/2, 1-alpha/2) percentiles of the bootstrap mean distribution. NaNs are dropped first;
    fewer than 2 finite values yields (nan, nan).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_bootstrap_delta_ci(values_a, values_b, n_boot: int = N_BOOTSTRAP, seed: int = CI_SEED,
                              alpha: float = 0.05):
    """Paired bootstrap 95% CI for the mean per-image difference ``a - b`` (e.g. fine-tune lift).

    ``values_a`` and ``values_b`` are aligned per-image metric arrays (same image order, same
    length). Resampling shares the image index across both models so the pairing — and thus the
    correlation between models on the same image — is preserved. Returns
    ``(mean_delta, lo, hi)``; a CI excluding 0 indicates a significant difference at ``alpha``.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    d = a - b
    d = d[np.isfinite(d)]
    if d.size < 2:
        return (float(np.nan if d.size == 0 else d.mean()), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi))


def remap_label(m):
    """Contiguous relabel, robust to masks with no background pixel (where HoVer-Net's crashes)."""
    m = np.asarray(m).astype(np.int32)
    if (m == 0).any():
        return _hovernet_remap(m)
    ids = np.unique(m)                      # fully-covered mask: relabel every id to 1..N
    out = np.zeros_like(m)
    for i, v in enumerate(ids, start=1):
        out[m == v] = i
    return out


def _iou_areas(true: np.ndarray, pred: np.ndarray):
    """Return (inter, union, area_t, area_p) for contiguous-labelled masks, O(image)."""
    nt, npd = int(true.max()), int(pred.max())
    area_t = np.bincount(true.ravel(), minlength=nt + 1).astype(np.float64)
    area_p = np.bincount(pred.ravel(), minlength=npd + 1).astype(np.float64)
    C = np.zeros((nt + 1, npd + 1), dtype=np.int64)
    np.add.at(C, (true.ravel(), pred.ravel()), 1)   # contingency over all pixels
    inter = C[1:, 1:].astype(np.float64)            # drop background row/col
    union = area_t[1:, None] + area_p[None, 1:] - inter
    return inter, union, area_t, area_p


def fast_aji_plus(true: np.ndarray, pred: np.ndarray) -> float:
    """AJI+ (Hungarian 1-to-1). Identical result to HoVer-Net get_fast_aji_plus, O(image)."""
    nt, npd = int(true.max()), int(pred.max())
    if nt == 0 or npd == 0:
        return 0.0
    inter, union, area_t, area_p = _iou_areas(true, pred)
    iou = inter / (union + 1e-6)
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-iou)
    sel = iou[ri, ci] > 0.0
    pt, pp = ri[sel], ci[sel]
    overall_inter = inter[pt, pp].sum()
    overall_union = union[pt, pp].sum()
    overall_union += area_t[1:][~np.isin(np.arange(nt), pt)].sum()
    overall_union += area_p[1:][~np.isin(np.arange(npd), pp)].sum()
    return float(overall_inter / overall_union)


def fast_aji(true: np.ndarray, pred: np.ndarray) -> float:
    """Classic AJI (greedy 1-to-many). Identical to HoVer-Net get_fast_aji, O(image)."""
    nt, npd = int(true.max()), int(pred.max())
    if nt == 0 or npd == 0:
        return 0.0
    inter, union, area_t, area_p = _iou_areas(true, pred)
    iou = inter / (union + 1e-6)
    best_p = iou.argmax(axis=1)
    best_iou = iou.max(axis=1)
    pt = np.nonzero(best_iou > 0.0)[0]
    pp = best_p[pt]
    overall_inter = inter[pt, pp].sum()
    overall_union = union[pt, pp].sum()
    overall_union += area_t[1:][~np.isin(np.arange(nt), pt)].sum()
    overall_union += area_p[1:][~np.isin(np.arange(npd), np.unique(pp))].sum()
    return float(overall_inter / overall_union)


def dice_semantic(y_true, y_pred) -> float:
    a = np.asarray(y_true) > 0
    b = np.asarray(y_pred) > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def boundary_f1_nsd(y_true, y_pred, tol: float = 2.0) -> float:
    """Normalized Surface Dice (boundary-F1) on instance boundaries, tolerance `tol` px. O(image)."""
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import find_boundaries

    gtb = find_boundaries(np.asarray(y_true), mode="inner")   # incl. boundaries between touching cells
    prb = find_boundaries(np.asarray(y_pred), mode="inner")
    ng, npx = int(gtb.sum()), int(prb.sum())
    if ng == 0 and npx == 0:
        return 1.0
    if ng == 0 or npx == 0:
        return 0.0
    d_to_gt = distance_transform_edt(~gtb)
    d_to_pr = distance_transform_edt(~prb)
    pr_close = int((d_to_gt[prb] <= tol).sum())
    gt_close = int((d_to_pr[gtb] <= tol).sum())
    return float((pr_close + gt_close) / (npx + ng))


def per_image_metrics(y_true, y_pred, iou_thresholds=IOU_THRESHOLDS, boundary=True) -> dict:
    from stardist.matching import matching

    yt = remap_label(np.asarray(y_true).astype(np.int32))
    yp = remap_label(np.asarray(y_pred).astype(np.int32))
    out: dict = {"n_true": int(yt.max()), "n_pred": int(yp.max())}

    for thr in iou_thresholds:
        m = matching(yt, yp, thresh=thr, criterion="iou")
        out[f"f1@{thr}"] = float(m.f1)
        out[f"precision@{thr}"] = float(m.precision)
        out[f"recall@{thr}"] = float(m.recall)
        if thr == 0.5:
            out["pq"] = float(m.panoptic_quality)
            out["sq"] = float(m.mean_matched_score)  # mean IoU over matched pairs

    out["aji"] = fast_aji(yt, yp)
    out["aji_plus"] = fast_aji_plus(yt, yp)
    out["dice"] = dice_semantic(yt, yp)
    if boundary:
        out["boundary_f1"] = boundary_f1_nsd(yt, yp)
    return out


def score_split(gt_list, pred_list, iou_thresholds=IOU_THRESHOLDS, skip_empty=True, boundary=True):
    """Return (per_image_records, aggregate_dict). Macro mean over non-empty-GT images."""
    records = []
    for i, (gt, pred) in enumerate(zip(gt_list, pred_list)):
        if skip_empty and np.asarray(gt).max() == 0:
            continue
        rec = per_image_metrics(gt, pred, iou_thresholds, boundary)
        rec["idx"] = i
        records.append(rec)

    metric_keys = [k for k in records[0] if k not in ("idx", "n_true", "n_pred")] if records else []
    agg = {f"{k}_mean": float(np.nanmean([r[k] for r in records])) for k in metric_keys}
    agg.update({f"{k}_std": float(np.nanstd([r[k] for r in records])) for k in metric_keys})
    for k in metric_keys:                          # per-image bootstrap 95% CI of the mean
        lo, hi = bootstrap_ci([r[k] for r in records])
        agg[f"{k}_ci_lo"], agg[f"{k}_ci_hi"] = lo, hi
    agg["n_images"] = len(records)

    try:
        from stardist.matching import matching_dataset
        gts = [remap_label(np.asarray(g).astype(np.int32)) for g, p in zip(gt_list, pred_list)
               if not (skip_empty and np.asarray(g).max() == 0)]
        prs = [remap_label(np.asarray(p).astype(np.int32)) for g, p in zip(gt_list, pred_list)
               if not (skip_empty and np.asarray(g).max() == 0)]
        md = matching_dataset(gts, prs, thresh=0.5, criterion="iou", by_image=False, show_progress=False)
        agg["pooled_f1@0.5"] = float(md.f1)
        agg["pooled_pq"] = float(md.panoptic_quality)
    except Exception as e:
        agg["pooled_error"] = str(e)
    return records, agg
