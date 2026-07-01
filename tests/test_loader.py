"""Unit tests for data.loader - runnable with pytest or `python tests/test_loader.py`.

Uses synthetic arrays so it needs neither the gated dataset nor a GPU. Channel conventions
are pinned against TissueNet v1.1 metadata.yaml: X=[nuclear, whole-cell], y=[whole-cell, nuclear].
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import _split_masks, load_npz, normalize_image  # noqa: E402


def test_normalize_to_unit_range():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.5, 30.0, size=(64, 64, 2)).astype(np.float64)
    out = normalize_image(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_normalize_flat_channel_maps_to_zero():
    img = np.full((16, 16, 2), 5.0, dtype=np.float64)  # flat -> hi<=lo
    out = normalize_image(img)
    assert np.all(out == 0.0)


def test_split_masks_two_channel():
    y = np.zeros((8, 8, 2), dtype=np.int32)
    y[0, 0, 0] = 3   # whole-cell channel
    y[1, 1, 1] = 7   # nuclear channel
    wc, nuc = _split_masks(y)
    assert wc[0, 0] == 3 and nuc[1, 1] == 7


def test_split_masks_single_channel():
    y = np.zeros((8, 8, 1), dtype=np.int32)
    y[2, 2, 0] = 5
    wc, nuc = _split_masks(y)
    assert wc[2, 2] == 5 and nuc is None


def test_load_npz_roundtrip(tmp_path=None):
    import tempfile
    d = Path(tmp_path or tempfile.mkdtemp())
    p = d / "mini.npz"
    X = np.random.default_rng(1).uniform(0.4, 27.0, size=(2, 32, 32, 2))
    y = np.zeros((2, 32, 32, 2), dtype=np.int32)
    y[0, :4, :4, 0] = 1
    np.savez(p, X=X, y=y)
    samples = list(load_npz(p))
    assert len(samples) == 2
    assert samples[0].image.shape == (32, 32, 2)
    assert samples[0].image.max() <= 1.0 + 1e-6
    assert samples[0].wholecell.max() == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
