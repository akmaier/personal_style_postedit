"""Sanity checks for the statistical helpers."""
import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_style.stats import benjamini_hochberg, hedges_g, perm_test_paired, perm_test_unpaired


def test_hedges_g_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = rng.normal(0, 1, 200)
    assert abs(hedges_g(x, y, paired=False)) < 0.2


def test_hedges_g_large_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1, 200)
    y = rng.normal(0.0, 1, 200)
    g = hedges_g(x, y, paired=False)
    assert 0.7 < g < 1.3


def test_perm_paired_detects_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 100)
    y = rng.normal(0.0, 1, 100)
    p = perm_test_paired(x, y, n_perm=2000, rng=rng)
    assert p < 0.01


def test_perm_unpaired_no_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    p = perm_test_unpaired(x, y, n_perm=2000, rng=rng)
    assert 0.05 < p


def test_bh_basic():
    p = np.array([0.001, 0.01, 0.02, 0.6, 0.9])
    adj, reject = benjamini_hochberg(p, q=0.05)
    assert reject.tolist() == [True, True, True, False, False]
    assert (adj <= 1).all() and (adj >= 0).all()
