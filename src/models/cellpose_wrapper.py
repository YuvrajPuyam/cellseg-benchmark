"""Cellpose-SAM (cpsam) wrapper - env: torch-cell. Handles whole-cell and nuclear.

cellpose 4.x ships the cpsam checkpoint; no `channels=` arg (auto-uses up to 3 channels,
order-invariant) and eval() returns a 3-tuple. cpsam normalizes internally (normalize=True).
"""
import os

import numpy as np

_model = None


def _get_model():
    """Return the cpsam model, or a fine-tuned checkpoint if CAJAL_CELLPOSE_CKPT is set."""
    global _model
    if _model is None:
        from cellpose import models
        ckpt = os.environ.get("CAJAL_CELLPOSE_CKPT")
        if ckpt:
            _model = models.CellposeModel(gpu=True, pretrained_model=ckpt)
        else:
            _model = models.CellposeModel(gpu=True)  # downloads cpsam from HF on first use
    return _model


def segment(raw_image, task="wholecell", flow_threshold=0.4, cellprob_threshold=0.0):
    """raw_image: (H,W,2) raw [nuclear, membrane]. Returns (H,W) uint32 instance mask."""
    m = _get_model()
    img = np.asarray(raw_image, dtype=np.float32)
    if task == "nuclear":
        masks, _, _ = m.eval(img[..., 0], flow_threshold=flow_threshold,
                             cellprob_threshold=cellprob_threshold)
    else:  # whole-cell: feed both channels; cpsam auto-uses them
        masks, _, _ = m.eval(img, channel_axis=2, flow_threshold=flow_threshold,
                             cellprob_threshold=cellprob_threshold)
    return np.asarray(masks, dtype=np.uint32)
