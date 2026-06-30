"""Metrics tests — fast AJI must equal the vendored HoVer-Net reference, and remap must be
robust to fully-covered masks. Runnable with numpy/scipy (no GPU/stardist needed for these)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_party.hover_net_stats_utils import get_fast_aji, get_fast_aji_plus  # noqa: E402
from third_party.hover_net_stats_utils import remap_label as hovernet_remap  # noqa: E402
from src.eval.metrics import fast_aji, fast_aji_plus, remap_label  # noqa: E402


def _make(n, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros((128, 128), np.int32)
    c = 1
    for _ in range(n):
        y, x = int(rng.integers(6, 122)), int(rng.integers(6, 122))
        r = int(rng.integers(3, 7))
        m[y - r:y + r, x - r:x + r] = c
        c += 1
    return m


def test_fast_aji_matches_vendored():
    for s in range(8):
        gt = hovernet_remap(_make(40, s))
        pr = hovernet_remap(_make(40, s + 100))
        assert abs(fast_aji_plus(gt, pr) - get_fast_aji_plus(gt, pr)) < 1e-9
        assert abs(fast_aji(gt, pr) - get_fast_aji(gt, pr)) < 1e-9


def test_remap_handles_no_background():
    m = np.ones((16, 16), np.int32)         # fully covered, no 0 — HoVer-Net's remove(0) would crash
    out = remap_label(m)
    assert out.min() == 1 and out.max() == 1


def test_fast_aji_bounds():
    a = hovernet_remap(_make(30, 1))
    assert abs(fast_aji_plus(a, a) - 1.0) < 1e-9          # perfect match
    empty = np.zeros((128, 128), np.int32)
    assert fast_aji_plus(a, empty) == 0.0                 # nothing predicted


if __name__ == "__main__":
    for fn in [test_fast_aji_matches_vendored, test_remap_handles_no_background, test_fast_aji_bounds]:
        fn()
        print("PASS", fn.__name__)
    print("ok")
