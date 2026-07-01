"""Vendored AJI / AJI+ from HoVer-Net (the metric stardist.matching does not provide).

Source: https://github.com/vqdang/hover_net  metrics/stats_utils.py  (Apache-2.0).
Only get_fast_aji, get_fast_aji_plus, and remap_label are vendored here, verbatim, with the
cv2/matplotlib imports dropped (AJI needs neither). Commit pinned in third_party/SOURCES.md.

AJI+ uses Hungarian 1-to-1 pairing (preferred); classic AJI uses greedy 1-to-many and
over-penalises - kept only for reference. Always remap_label() to contiguous IDs first.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_fast_aji(true, pred):
    """Classic AJI (MoNuSeg). Greedy 1-to-many → over-penalises. Prefer get_fast_aji_plus."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))
    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    pairwise_inter = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    paired_true = list(paired_true + 1)
    paired_pred = list(paired_pred + 1)
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    return overall_inter / overall_union


def get_fast_aji_plus(true, pred):
    """AJI+ : maximal unique (Hungarian) 1-to-1 pairing. Preferred over classic AJI."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))
    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    pairwise_inter = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    paired_iou = pairwise_iou[paired_true, paired_pred]
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    return overall_inter / overall_union


def remap_label(pred, by_size=False):
    """Rename instance ids to be contiguous [1..N]. Call before AJI. by_size: big nuclei get low id."""
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred
    if by_size:
        pred_size = [(pred == inst_id).sum() for inst_id in pred_id]
        pair_list = sorted(zip(pred_id, pred_size), key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)
    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
