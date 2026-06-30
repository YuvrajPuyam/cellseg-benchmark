"""Metrics tests — fast AJI must equal the vendored HoVer-Net reference, and remap must be
robust to fully-covered masks. Runnable with numpy/scipy (no GPU/stardist needed for these)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_party.hover_net_stats_utils import get_fast_aji, get_fast_aji_plus  # noqa: E402
from third_party.hover_net_stats_utils import remap_label as hovernet_remap  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    bootstrap_ci,
    fast_aji,
    fast_aji_plus,
    paired_bootstrap_delta_ci,
    remap_label,
)
from data.loader import sample_indices  # noqa: E402


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


def test_sample_indices_deterministic():
    a = sample_indices(100, 10, seed=42)
    b = sample_indices(100, 10, seed=42)
    assert np.array_equal(a, b)                              # same (n_total, n, seed) -> same picks
    assert len(a) == 10 and len(set(a.tolist())) == 10      # no replacement
    assert (a == np.sort(a)).all()                          # sorted -> yielded order matches disk
    assert a.min() >= 0 and a.max() < 100
    assert not np.array_equal(a, sample_indices(100, 10, seed=43))  # seed actually matters
    assert not np.array_equal(a, np.arange(10))             # not just first-N (distinct from --limit)


def test_sample_indices_edge_cases():
    assert np.array_equal(sample_indices(5, 10, seed=0), np.arange(5))  # n >= total -> all
    assert np.array_equal(sample_indices(5, 0, seed=0), np.arange(5))   # n <= 0   -> all


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(7)
    v = rng.normal(0.8, 0.05, size=200)
    lo, hi = bootstrap_ci(v, seed=0)
    assert lo < v.mean() < hi                                # CI brackets the sample mean
    assert 0.0 <= lo <= hi
    lo2, hi2 = bootstrap_ci(v, seed=0)
    assert (lo, hi) == (lo2, hi2)                            # seeded -> reproducible


def test_bootstrap_ci_degenerate():
    lo, hi = bootstrap_ci([0.5] * 30, seed=0)               # zero variance -> zero-width CI at mean
    assert abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9
    lo, hi = bootstrap_ci([0.5], seed=0)                    # <2 finite values -> nan
    assert np.isnan(lo) and np.isnan(hi)


def test_paired_delta_ci_detects_lift():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.4, 0.6, size=100)
    ft = base + 0.1                                          # uniform fine-tune lift of +0.1
    mean_d, lo, hi = paired_bootstrap_delta_ci(ft, base, seed=0)
    assert abs(mean_d - 0.1) < 1e-9
    assert lo > 0.0                                          # CI excludes 0 -> significant lift


def test_paired_delta_ci_no_difference():
    rng = np.random.default_rng(11)
    x = rng.normal(0.7, 0.1, size=150)
    mean_d, lo, hi = paired_bootstrap_delta_ci(x, x, seed=0)  # identical -> zero delta, zero-width CI
    assert abs(mean_d) < 1e-9 and abs(lo) < 1e-9 and abs(hi) < 1e-9


def test_paired_delta_ci_shape_mismatch():
    try:
        paired_bootstrap_delta_ci(np.zeros(5), np.zeros(6))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on misaligned paired arrays")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ok")
