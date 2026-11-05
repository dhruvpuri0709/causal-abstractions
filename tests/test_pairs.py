"""Pair generation: disjoint pools, usage caps, stratification, correct labels."""

import numpy as np

from cas.pairs import load_splits, make_splits, save_splits
from cas.tasks import ALGORITHMS, TASKS, bits_from_inputs

DL = TASKS["distributive_law"]


def _splits(seed=0, hyp="and_or_and"):
    return make_splits(DL, hyp, {"fit": 800, "select": 400, "test": 400},
                       np.random.default_rng(seed), base_uses=2, source_uses=2)


def test_pools_are_disjoint_across_splits():
    sp = _splits()
    rows = {}
    for name, s in sp.items():
        rows[name] = {r.tobytes() for r in np.concatenate([s.base_x, s.src_x])}
    assert not (rows["fit"] & rows["select"]) and not (rows["fit"] & rows["test"])
    assert not (rows["select"] & rows["test"])


def test_usage_caps_and_stratification():
    for s in _splits(1).values():
        assert np.bincount(s.pair_base).max() <= 2
        used = s.pair_src[s.pair_src >= 0]
        assert np.bincount(used).max() <= 2
        lab = s.labels["and_or_and"]
        changed = lab != s.base_labels
        for v in (0, 1):
            for c in (False, True):
                frac = np.mean((lab == v) & (changed == c))
                assert abs(frac - 0.25) < 0.01
        # every pair intervenes on at least one variable; types roughly 1/3 each
        m = s.mask()
        assert m.any(1).all()
        assert abs(np.mean(m.all(1)) - 1 / 3) < 0.1


def test_labels_match_vectors_for_all_algorithms():
    s = _splits(2, hyp="and_or")["fit"]
    bb = bits_from_inputs(DL, s.base_x)[s.pair_base]
    sb = bits_from_inputs(DL, s.src_x)[np.clip(s.pair_src, 0, None)]
    for a in DL.algorithms:
        assert (ALGORITHMS[a].counterfactual(bb, sb, s.mask()) == s.labels[a]).all()
    assert (DL.label(bb) == s.base_labels).all()


def test_save_load_roundtrip(tmp_path):
    sp = _splits(3)
    save_splits(sp, tmp_path / "p.npz")
    back = load_splits(tmp_path / "p.npz")
    for k in sp:
        assert (back[k].pair_src == sp[k].pair_src).all()
        assert (back[k].labels["and_or"] == sp[k].labels["and_or"]).all()


def test_deterministic():
    a, b = _splits(5)["test"], _splits(5)["test"]
    assert (a.base_x == b.base_x).all() and (a.pair_src == b.pair_src).all()
