"""Bootstrap percentile intervals for IIA from per-pair correctness.

Three resampling schemes, all written as random weights on pairs (IIA* = sum w c / sum w):

* ``iid``:     pairs resampled with replacement (multinomial counts over pairs).  Ignores that
               pairs share inputs.
* ``base``:    base inputs resampled with replacement; a pair's weight is its base's count
               (one-way cluster bootstrap, cf. Cameron, Gelbach & Miller 2008).
* ``twoway``:  base inputs and source inputs resampled independently; a pair's weight is
               (count of its base) x (product of the counts of its sources).  This is the
               "pigeonhole" product bootstrap for crossed factors (Owen 2007), extended to pairs
               with two sources by multiplying both source counts.  Owen shows it is mildly
               conservative (it over-estimates the variance by the small "interaction" terms).
"""

from __future__ import annotations

import numpy as np

METHODS = ("iid", "base", "twoway")


def _cluster_counts(ids: np.ndarray, B: int, rng: np.random.Generator):
    """Multinomial counts for each distinct id; returns (B, n_ids) counts and inverse index."""
    uniq, inv = np.unique(ids, return_inverse=True)
    m = len(uniq)
    counts = rng.multinomial(m, np.full(m, 1.0 / m), size=B)
    return counts, inv


def bootstrap_weights(method: str, pair_base: np.ndarray, pair_src: np.ndarray, B: int,
                      rng: np.random.Generator) -> np.ndarray:
    """(B, n_pairs) nonnegative weights."""
    n = len(pair_base)
    if method == "iid":
        return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    wb, inv_b = _cluster_counts(pair_base, B, rng)
    w = wb[:, inv_b].astype(float)
    if method == "base":
        return w
    if method != "twoway":
        raise ValueError(method)
    used = pair_src[pair_src >= 0]
    ws, _ = _cluster_counts(used, B, rng)
    uniq = np.unique(used)
    for i in range(pair_src.shape[1]):
        col = pair_src[:, i]
        has = col >= 0
        idx = np.searchsorted(uniq, col[has])
        w[:, has] *= ws[:, idx]
    return w


def bootstrap_iia(correct: np.ndarray, pair_base: np.ndarray, pair_src: np.ndarray,
                  method: str, B: int, rng: np.random.Generator) -> np.ndarray:
    """B bootstrap replicates of IIA (replicates with total weight 0 are dropped)."""
    w = bootstrap_weights(method, pair_base, pair_src, B, rng)
    tot = w.sum(1)
    ok = tot > 0
    return (w[ok] @ correct.astype(float)) / tot[ok]


def percentile_interval(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))
