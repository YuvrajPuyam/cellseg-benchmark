"""micro_sam (μSAM) wrapper — env: torch-cell. vit_b_lm (light-microscopy). Whole-cell + nuclear.

Optional model: if micro_sam fails to import, the orchestrator skips it (it's an extra PyTorch
package that can be awkward to install). Inputs are percentile-normalized to [0,1] float32, which
the _lm automatic instance segmentation expects: nuclear-task uses a single nuclear channel;
whole-cell uses a 3-channel RGB = [membrane, nuclear, membrane] that preserves the membrane
boundary signal μSAM needs (see segment() for why a channel-mean grayscale fails here).
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
        # Whole-cell: μSAM segments cells by their MEMBRANE boundary, so the input must
        # preserve that signal. A channel-mean grayscale (the previous behavior) averages the
        # membrane channel against the nuclear channel and washes out the boundaries, which is
        # why μSAM badly under-performed Cellpose on whole-cell. Instead compose an RGB image
        # whose R and B carry the membrane (boundary) channel and whose G carries the nuclear
        # channel: RGB = [membrane, nuclear, membrane]. This keeps membrane dominant (2 of 3
        # channels) while still giving the nuclear interior signal, matching how the _lm model
        # was trained on multi-channel light-microscopy data. Each channel is independently
        # percentile-normalized to [0, 1] float32 — the dtype/range micro_sam's automatic
        # instance segmentation expects (it normalizes/upsamples internally from there).
        norm = normalize_image(img)                           # (H,W,2) per-channel → [0,1]
        nuclear, membrane = norm[..., 0], norm[..., 1]
        x = np.stack([membrane, nuclear, membrane], axis=-1).astype(np.float32)  # (H,W,3) RGB
    predictor, segmenter = _get()
    mask = automatic_instance_segmentation(predictor, segmenter, x, ndim=2)
    return np.asarray(mask, dtype=np.uint32)
