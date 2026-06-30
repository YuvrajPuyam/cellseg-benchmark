"""micro_sam (μSAM) wrapper — env: torch-cell. vit_b_lm (light-microscopy). Whole-cell + nuclear.

Optional model: if micro_sam fails to import, the orchestrator skips it (it's an extra PyTorch
package that can be awkward to install). Input is normalized to [0,1] grayscale, which the _lm
automatic instance segmentation expects.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → data.loader importable
from data.loader import normalize_image  # noqa: E402

_predictor = None
_segmenter = None


def _get():
    global _predictor, _segmenter
    if _predictor is None:
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        _predictor, _segmenter = get_predictor_and_segmenter(model_type="vit_b_lm")
    return _predictor, _segmenter


def segment(raw_image, task="wholecell"):
    """raw_image: (H,W,2) raw [nuclear, membrane]. Returns (H,W) uint32 instance mask."""
    from micro_sam.automatic_segmentation import automatic_instance_segmentation

    img = np.asarray(raw_image, dtype=np.float32)
    if task == "nuclear":
        x = normalize_image(img[..., 0:1])[..., 0]            # nuclear grayscale
    else:
        x = normalize_image(img).mean(axis=-1).astype(np.float32)  # combine channels → grayscale
    predictor, segmenter = _get()
    mask = automatic_instance_segmentation(predictor, segmenter, x, ndim=2)
    return np.asarray(mask, dtype=np.uint32)
