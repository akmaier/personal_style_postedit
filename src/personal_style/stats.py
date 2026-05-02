"""Statistical helpers: permutation tests, Hedges' g + bootstrap CI, BH-FDR, rmcorr."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def hedges_g(x: np.ndarray, y: np.ndarray, paired: bool) -> float:
    """Hedges' g (small-sample-corrected standardized mean difference).

    Following Hedges (1981), the SMD uses the *pooled* SD of the two samples even
    for paired data; this is the convention used by Baumler et al. (paper §6).
    For paired data we still pair the means (so n is the number of pairs).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    d_val = (x.mean() - y.mean()) / pooled
    df = (nx + ny - 2) if not paired else (nx - 1)
    j = 1 - 3.0 / (4 * df - 1) if df > 0 else 1.0
    return float(d_val * j)


def bootstrap_ci_g(
    x: np.ndarray,
    y: np.ndarray,
    paired: bool,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng(20260101)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gs = np.empty(n_boot)
    n = len(x)
    if paired:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            gs[b] = hedges_g(x[idx], y[idx], paired=True)
    else:
        ny = len(y)
        for b in range(n_boot):
            ix = rng.integers(0, n, size=n)
            iy = rng.integers(0, ny, size=ny)
            gs[b] = hedges_g(x[ix], y[iy], paired=False)
    lo, hi = np.nanpercentile(gs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def perm_test_paired(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sided paired permutation test on the mean difference (sign flips)."""
    rng = rng or np.random.default_rng(20260101)
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    extreme = 1  # +1 for the observed
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((d * signs).mean()) >= obs - 1e-15:
            extreme += 1
    return float(extreme / (n_perm + 1))


def perm_test_unpaired(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sided unpaired permutation test on the difference of means."""
    rng = rng or np.random.default_rng(20260101)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    pooled = np.concatenate([x, y])
    obs = abs(x.mean() - y.mean())
    extreme = 1
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:nx].mean() - pooled[nx:].mean()) >= obs - 1e-15:
            extreme += 1
    return float(extreme / (n_perm + 1))


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (adjusted p, reject) at FDR level q (BH 1995)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    reject = out <= q
    return out, reject


@dataclass(frozen=True)
class TestResult:
    name: str
    n: int
    paired: bool
    mean_x: float
    mean_y: float
    g: float
    ci_low: float
    ci_high: float
    p: float
    n_perm: int


def run_test(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    paired: bool,
    n_perm: int = 10000,
    n_boot: int = 1000,
    seed: int | None = None,
) -> TestResult:
    rng_p = np.random.default_rng(seed if seed is not None else hash(("perm", name)) & 0xFFFFFFFF)
    rng_b = np.random.default_rng(seed + 1 if seed is not None else hash(("boot", name)) & 0xFFFFFFFF)
    p = (
        perm_test_paired(x, y, n_perm=n_perm, rng=rng_p)
        if paired
        else perm_test_unpaired(x, y, n_perm=n_perm, rng=rng_p)
    )
    g = hedges_g(x, y, paired=paired)
    lo, hi = bootstrap_ci_g(x, y, paired=paired, n_boot=n_boot, rng=rng_b)
    return TestResult(
        name=name,
        n=int(len(x)),
        paired=paired,
        mean_x=float(np.mean(x)),
        mean_y=float(np.mean(y)),
        g=float(g),
        ci_low=lo,
        ci_high=hi,
        p=p,
        n_perm=n_perm,
    )
