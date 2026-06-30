"""Instance + semantic segmentation metrics (runs in the `metrics` conda env).

Anchored on stardist.matching (Hungarian 1-to-1 → PQ, SQ, F1@IoU, precision/recall). Gaps filled:
AJI/AJI+ from vendored HoVer-Net (third_party/), boundary-F1 (NSD) from MONAI SurfaceDiceMetric.

Averaging convention (pinned, and stated in the README):
  * per-image (macro) is the HEADLINE — mean over images of each per-image metric.
  * dataset-pooled F1/PQ also reported via stardist.matching_dataset(by_image=False).
Empty-GT images are skipped by default. All masks are integer instance labels (0 = background).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → third_party importable
from third_party.hover_net_stats_utils import get_fast_aji, get_fast_aji_plus, remap_label  # noqa: E402

IOU_THRESHOLDS = (0.5, 0.75)


def dice_semantic(y_true, y_pred) -> float:
    a = np.asarray(y_true) > 0
    b = np.asarray(y_pred) > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def boundary_f1_nsd(y_true, y_pred, tol: float = 2.0) -> float:
    """Normalized Surface Dice (boundary-F1) on the binary foreground, tolerance `tol` px."""
    try:
        import torch
        from monai.metrics import compute_surface_dice

        def onehot(m):
            fg = (np.asarray(m) > 0).astype(np.float32)
            return torch.from_numpy(np.stack([1.0 - fg, fg])[None])  # (1, 2, H, W)

        nsd = compute_surface_dice(onehot(y_pred), onehot(y_true),
                                   class_thresholds=[tol], include_background=False)
        return float(np.nan_to_num(nsd.numpy(), nan=0.0).mean())
    except Exception:
        return float("nan")


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

    both = yt.max() > 0 and yp.max() > 0
    out["aji"] = float(get_fast_aji(yt, yp)) if both else 0.0
    out["aji_plus"] = float(get_fast_aji_plus(yt, yp)) if both else 0.0
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
    agg["n_images"] = len(records)

    # dataset-pooled F1/PQ (secondary convention)
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
