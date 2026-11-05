"""Bootstrap weights: structure and sanity of the three schemes."""

import numpy as np

from cas.bootstrap import bootstrap_iia, bootstrap_weights, percentile_interval


def _design(n_base=100, uses=4, seed=0):
    rng = np.random.default_rng(seed)
    pair_base = np.repeat(np.arange(n_base), uses)
    n = len(pair_base)
    pair_src = rng.integers(0, 2 * n_base, size=(n, 2))
    pair_src[rng.random(n) < 0.33, 0] = -1
    pair_src[(pair_src[:, 0] >= 0) & (rng.random(n) < 0.5), 1] = -1
    return pair_base, pair_src


def test_weight_structure():
    pb, ps = _design()
    rng = np.random.default_rng(1)
    w = bootstrap_weights("iid", pb, ps, 50, rng)
    assert w.shape == (50, len(pb)) and np.allclose(w.sum(1), len(pb))
    w = bootstrap_weights("base", pb, ps, 50, rng)
    for b in range(5):  # constant within a base cluster
        assert np.unique(w[:, pb == b], axis=1).shape[1] == 1
    assert np.allclose(w.sum(1) / 4, 100)  # cluster counts sum to the number of bases
    w2 = bootstrap_weights("twoway", pb, ps, 50, rng)
    assert (w2 >= 0).all() and w2.shape == w.shape


def test_cluster_correlated_data_gives_wider_cluster_intervals():
    pb, ps = _design(n_base=100, uses=8, seed=2)
    rng = np.random.default_rng(3)
    # correctness determined by the base input only: strong within-base correlation
    base_ok = rng.random(100) < 0.7
    correct = base_ok[pb]
    widths = {}
    for m in ("iid", "base", "twoway"):
        reps = bootstrap_iia(correct, pb, ps, m, 2000, np.random.default_rng(4))
        assert abs(reps.mean() - correct.mean()) < 0.01
        lo, hi = percentile_interval(reps)
        widths[m] = hi - lo
    assert widths["base"] > 2 * widths["iid"]  # design effect ~ sqrt(8)
    assert widths["twoway"] >= 0.9 * widths["base"]


def test_twoway_counts_a_shared_source_once():
    """If both variables take the same source input, its count multiplies the weight once."""
    pair_base = np.array([0, 1, 2])
    pair_src = np.array([[0, 0], [1, 2], [3, -1]])  # pair 0: one source used for V1 and V2
    rng = np.random.default_rng(5)
    w = bootstrap_weights("twoway", pair_base, pair_src, 200, rng)
    # replay the same draws to get the base and source counts
    rng = np.random.default_rng(5)
    wb = rng.multinomial(3, np.full(3, 1 / 3), size=200)
    ws = rng.multinomial(4, np.full(4, 1 / 4), size=200)
    assert np.allclose(w[:, 0], wb[:, 0] * ws[:, 0])  # not ws[:, 0] ** 2
    assert np.allclose(w[:, 1], wb[:, 1] * ws[:, 1] * ws[:, 2])
    assert np.allclose(w[:, 2], wb[:, 2] * ws[:, 3])
