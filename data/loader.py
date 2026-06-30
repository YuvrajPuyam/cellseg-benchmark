"""TissueNet loader + normalization.

TissueNet is **not** pre-normalized — pixels are raw intensity (the sample's nuclear channel
ranges ~0.4–27). This loader applies per-image, per-channel percentile normalization (the #1
silent-failure footgun in cell-seg benchmarks) before any model sees an image.

Data layout (TissueNet v1.1, extracted from tissuenet_v1-1.zip):
  X: (N, 512, 512, 2) float  — channel 0 = nuclear, channel 1 = membrane / whole-cell marker
  y: (N, 512, 512, 2) int    — channel 0 = whole-cell labels, channel 1 = nuclear labels
The public single-image sample (tissuenet-sample.npz) carries y of shape (1, 512, 512, 1).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

NUCLEAR_CH = 0
MEMBRANE_CH = 1
WHOLECELL_MASK = 0
NUCLEAR_MASK = 1


def normalize_image(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Per-channel percentile-normalize an HxWxC float image to [0, 1].

    Clips each channel to its [p_low, p_high] percentiles, then rescales. A flat/empty channel
    (hi <= lo) maps to zeros instead of dividing by ~0.
    """
    img = np.asarray(img, dtype=np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        lo, hi = np.percentile(ch, [p_low, p_high])
        if hi <= lo:
            continue
        out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out


@dataclass
class Sample:
    image: np.ndarray                 # (H, W, 2) normalized float32
    wholecell: Optional[np.ndarray]   # (H, W) int32 instance labels, 0 = bg (or None)
    nuclear: Optional[np.ndarray]     # (H, W) int32 instance labels, 0 = bg (or None)


def _split_masks(y_img: np.ndarray):
    """y_img: (H,W) or (H,W,C) → (wholecell, nuclear) as int32 arrays or None."""
    y = np.asarray(y_img)
    if y.ndim == 2:
        return y.astype(np.int32), None
    c = y.shape[-1]
    wc = y[..., WHOLECELL_MASK].astype(np.int32) if c >= 1 else None
    nuc = y[..., NUCLEAR_MASK].astype(np.int32) if c >= 2 else None
    return wc, nuc


def load_npz(path, normalize: bool = True, p_low: float = 1.0, p_high: float = 99.5) -> Iterator[Sample]:
    """Yield a Sample per image from a TissueNet-style npz containing arrays ``X`` and ``y``."""
    d = np.load(path, allow_pickle=True)
    X, y = d["X"], d["y"]
    for i in range(X.shape[0]):
        img = normalize_image(X[i], p_low, p_high) if normalize else X[i].astype(np.float32)
        wc, nuc = _split_masks(y[i])
        yield Sample(image=img, wholecell=wc, nuclear=nuc)


def _selftest(sample_path: str) -> None:
    n = 0
    for s in load_npz(sample_path):
        assert s.image.ndim == 3 and s.image.shape[-1] == 2, s.image.shape
        assert s.image.min() >= 0.0 and s.image.max() <= 1.0 + 1e-6, (s.image.min(), s.image.max())
        masks = [m for m in (s.wholecell, s.nuclear) if m is not None]
        ncells = max(int(m.max()) for m in masks) if masks else 0
        n += 1
    print(f"[selftest] {n} image(s); normalized to [0,1] OK; last image has ~{ncells} instances")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect/normalize a TissueNet npz.")
    ap.add_argument("npz", help="path to a TissueNet-style .npz (X, y)")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()
    _selftest(args.npz)
