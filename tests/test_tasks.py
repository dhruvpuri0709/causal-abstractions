"""Causal models: labels and counterfactual labels on hand-worked examples."""

import itertools

import numpy as np

from cas.tasks import (ALGORITHMS, TASKS, bits_from_inputs, make_inputs, sample_bits,
                       sample_interchange_batch)

HE, DL = TASKS["hierarchical_equality"], TASKS["distributive_law"]


def cf(alg, base, s1=None, s2=None):
    """Counterfactual label for one example; s1/s2 None means 'not intervened'."""
    n_eq = len(base)
    src = np.zeros((1, 2, n_eq), dtype=bool)
    mask = np.array([[s1 is not None, s2 is not None]])
    if s1 is not None:
        src[0, 0] = s1
    if s2 is not None:
        src[0, 1] = s2
    return bool(ALGORITHMS[alg].counterfactual(np.array([base], bool), src, mask)[0])


def test_hierarchical_equality_labels():
    # AABB -> True, ABCD -> True, AABC -> False, ABCC -> False (Geiger et al. 2024 §4)
    bits = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=bool)
    assert HE.label(bits).tolist() == [True, True, False, False]


def test_hierarchical_equality_counterfactuals():
    # base AABC: WX=T, YZ=F -> F.  Set WX from a source with WX=F -> (F == F) = T.
    assert cf("both_equality", [1, 0], s1=[0, 1]) is True
    # same base, WX source has WX=T -> unchanged F
    assert cf("both_equality", [1, 0], s1=[1, 0]) is False
    # set YZ from source ABCC (YZ=T): (T == T) = T
    assert cf("both_equality", [1, 0], s2=[0, 1]) is True
    # both: WX from ABCD (F), YZ from AABB (T) -> F
    assert cf("both_equality", [1, 0], s1=[0, 0], s2=[1, 1]) is False
    # the pyvene tutorial example: base BBCD (WX=T, YZ=F), WX from source ABCC (WX=F) -> T
    assert cf("both_equality", [1, 0], s1=[0, 1]) is True


def test_distributive_law_algorithms_agree_on_task():
    pats = np.array(list(itertools.product([False, True], repeat=3)))
    a = ALGORITHMS["and_or_and"].run(pats)
    b = ALGORITHMS["and_or"].run(pats)
    truth = (pats[:, 0] & pats[:, 1]) | (pats[:, 1] & pats[:, 2])
    assert (a == truth).all() and (b == truth).all()


def test_distributive_law_counterfactuals():
    # base e = (1,1,0): AOA V1 = e0&e1 = 1, V2 = e1&e2 = 0 -> 1
    assert cf("and_or_and", [1, 1, 0]) is True
    # V1 from source (0,1,1) where V1 = 0 -> V1'=0, V2=0 -> 0
    assert cf("and_or_and", [1, 1, 0], s1=[0, 1, 1]) is False
    # V2 from the same source (V2 = 1) -> 1 or 1 = 1
    assert cf("and_or_and", [1, 1, 0], s2=[0, 1, 1]) is True
    # And-Or on base (1,1,0): V1 = e1 = 1, V2 = e0|e2 = 1 -> 1
    assert cf("and_or", [1, 1, 0]) is True
    # V2 from source (0,1,0) (V2 = 0) -> 0
    assert cf("and_or", [1, 1, 0], s2=[0, 1, 0]) is False
    # V1 from source (1,0,1) (V1 = e1 = 0) -> 0
    assert cf("and_or", [1, 1, 0], s1=[1, 0, 1]) is False
    # the two algorithms disagree on interventions: base (0,0,0), source (1,1,1) on V1
    assert cf("and_or_and", [0, 0, 0], s1=[1, 1, 1]) is True   # V1'=1 -> 1
    assert cf("and_or", [0, 0, 0], s1=[1, 1, 1]) is False      # V1'=1, V2=0 -> 0


def test_inputs_realise_bits_and_balance():
    rng = np.random.default_rng(0)
    for task in (HE, DL):
        bits = sample_bits(task, 4000, rng)
        x = make_inputs(task, bits, rng)
        assert x.shape == (4000, task.input_dim) and x.dtype == np.float32
        assert (bits_from_inputs(task, x) == bits).all()
        assert abs(task.label(bits).mean() - 0.5) < 0.03
        assert x.min() >= -0.5 and x.max() <= 0.5


def test_interchange_batch_labels_match_algorithm():
    rng = np.random.default_rng(1)
    xb, xs, mask, y = sample_interchange_batch(DL, ALGORITHMS["and_or"], 500, rng)
    bb = bits_from_inputs(DL, xb)
    sb = np.stack([bits_from_inputs(DL, xs[:, i]) for i in range(2)], axis=1)
    assert (ALGORITHMS["and_or"].counterfactual(bb, sb, mask) == y).all()
