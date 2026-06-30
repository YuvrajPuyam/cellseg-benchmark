"""Wrapper contract shared by every segmentation model.

Each model runs in its own conda env as a subprocess (TF and PyTorch can't co-exist on one
GPU), so the only thing that crosses the env boundary is an integer instance-label mask written
to disk. Every wrapper implements `segment(image) -> mask` and the orchestrator handles I/O.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SegModel(ABC):
    """A segmentation model that maps one image to an instance-label mask.

    Contract:
      - input  `image`: HxW (single channel) or HxWxC float/array. Wrappers are responsible
                        for selecting/normalizing the channels their model expects.
      - output `mask` : HxW array of uint32, 0 = background, 1..N = distinct instances.
                        IDs need not be contiguous (metrics layer remaps before AJI).
    """

    #: short id used in output paths: masks/{name}/{split}/*.tif
    name: str = "base"
    #: which TissueNet task this model is scored on: "wholecell", "nuclear", or "both"
    task: str = "both"

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return an HxW uint32 instance-label mask (0 = background)."""
        raise NotImplementedError

    @staticmethod
    def _as_uint32(mask: np.ndarray) -> np.ndarray:
        """Normalize any integer-label output to a contiguous-safe uint32 mask."""
        return np.asarray(mask, dtype=np.uint32)
