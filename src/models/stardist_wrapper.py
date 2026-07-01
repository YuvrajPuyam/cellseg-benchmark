"""StarDist 2D_versatile_fluo wrapper - env: stardist-tf. Nuclear task only.

StarDist's star-convex prior collapses to near-circles on whole-cell, so it competes only on
the nuclear channel (fair-comparison decision). Uses csbdeep percentile normalization, the
canonical preprocessing for the pretrained fluorescence model.
"""
import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from stardist.models import StarDist2D
        _model = StarDist2D.from_pretrained("2D_versatile_fluo")
    return _model


def segment(raw_image, task="nuclear"):
    """raw_image: (H,W,2) raw [nuclear, membrane]. Returns (H,W) uint32 nuclear instance mask."""
    from csbdeep.utils import normalize
    m = _get_model()
    nuc = np.asarray(raw_image, dtype=np.float32)[..., 0]  # nuclear channel
    labels, _ = m.predict_instances(normalize(nuc, 1, 99.8))
    return np.asarray(labels, dtype=np.uint32)
