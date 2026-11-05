"""DAS correctness checks: handcrafted network, interchange algebra, random network."""

import numpy as np
import torch

from cas.das import (FitSettings, Rotation, fit_rotation, interchange, pair_activations,
                     predict)
from cas.models import MLP, handcrafted_hierarchical_equality
from cas.pairs import make_split
from cas.tasks import TASKS, Task, sample_task_data

HE2 = Task("hierarchical_equality", "he", 4, ("both_equality",), emb_dim=2)


def _acts(model, layer, sp, hyp):
    return pair_activations(model, layer, torch.tensor(sp.base_x), torch.tensor(sp.src_x),
                            sp.pair_base, sp.pair_src, sp.labels[hyp])


def test_handcrafted_solves_task():
    m = handcrafted_hierarchical_equality()
    x, y = sample_task_data(HE2, 5000, np.random.default_rng(0))
    assert (m(torch.tensor(x)).argmax(1).numpy() == y).all()


def test_handcrafted_neuron_alignment_iia_is_one():
    """Neurons 0..3 of layer 0 = WX, 4..7 = YZ (pyvene tutorial): IIA must be exactly 1."""
    m = handcrafted_hierarchical_equality()
    sp = make_split(HE2, "both_equality", 2000, np.random.default_rng(1))
    acts = _acts(m, 0, sp, "both_equality")
    pred = predict(m, 0, acts, None, k=4)
    assert (pred == sp.labels["both_equality"]).mean() == 1.0
    # the same alignment expressed as an (identity) rotation gives the same result
    pred_r = predict(m, 0, acts, torch.eye(8), k=4)
    assert (pred_r == pred).all()


def test_handcrafted_wrong_alignment_is_not_perfect():
    """Swapping the neuron blocks (WX <-> YZ) must break the alignment."""
    m = handcrafted_hierarchical_equality()
    sp = make_split(HE2, "both_equality", 2000, np.random.default_rng(2))
    perm = torch.eye(8)[:, [4, 5, 6, 7, 0, 1, 2, 3]]  # rotated dims 0..3 = neurons 4..7
    pred = predict(m, 0, _acts(m, 0, sp, "both_equality"), perm, k=4)
    assert (pred == sp.labels["both_equality"]).mean() < 0.9


def test_interchange_algebra():
    g = torch.Generator().manual_seed(0)
    n, w, k = 50, 10, 3
    hb, hs = torch.randn(n, w, generator=g), torch.randn(n, 2, w, generator=g)
    R = Rotation(w, 0).R.detach()
    assert torch.allclose(R @ R.T, torch.eye(w), atol=1e-5)
    none = torch.zeros(n, 2, dtype=torch.bool)
    assert torch.allclose(interchange(hb, hs, none, R, k), hb, atol=1e-5)
    both = torch.ones(n, 2, dtype=torch.bool)
    out = interchange(hb, hs, both, R, k)
    rb, r0, r1, ro = hb @ R, hs[:, 0] @ R, hs[:, 1] @ R, out @ R
    assert torch.allclose(ro[:, :k], r0[:, :k], atol=1e-5)
    assert torch.allclose(ro[:, k:2 * k], r1[:, k:2 * k], atol=1e-5)
    assert torch.allclose(ro[:, 2 * k:], rb[:, 2 * k:], atol=1e-5)
    # intervening on everything with k = w/2 and the same source twice gives the source
    same = torch.stack([hs[:, 0], hs[:, 0]], 1)
    assert torch.allclose(interchange(hb, same, both, R, w // 2), hs[:, 0], atol=1e-5)


def test_random_network_iia_near_chance():
    """A random network whose outputs actually vary (larger init than PyTorch's default, which
    makes most random networks predict one constant class and IIA exactly 0.5): DAS fits the
    training pairs above chance (overfitting) but held-out IIA stays near chance."""
    task = TASKS["hierarchical_equality"]
    torch.manual_seed(0)
    m = MLP(task.input_dim, 16, 3).eval()
    with torch.no_grad():
        for lin in list(m.layers) + [m.head]:
            torch.nn.init.normal_(lin.weight, std=3 / np.sqrt(lin.in_features))
            torch.nn.init.normal_(lin.bias, std=0.1)
    for p in m.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(3)
    fit = make_split(task, "both_equality", 512, rng)
    test = make_split(task, "both_equality", 512, rng)
    res = fit_rotation(m, 0, 8, _acts(m, 0, fit, "both_equality"), 0,
                       FitSettings(max_steps=500))
    pred = predict(m, 0, _acts(m, 0, test, "both_equality"), res.R, 8)
    assert 0.1 < pred.mean() < 0.9  # not a constant predictor, so the test can fail
    assert 0.4 < (pred == test.labels["both_equality"]).mean() < 0.6
