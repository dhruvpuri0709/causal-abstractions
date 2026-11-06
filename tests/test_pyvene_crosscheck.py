"""Slow cross-check: our interchange vs pyvene 0.1.8's RotatedSpaceIntervention.

Run with `uv run pytest -m slow`.  We train a small IIT network, fit a DAS rotation with our
code, load the same rotation into pyvene's intervention and recompute IIA with it.  "Both"
interventions are applied as two linked pyvene interventions sharing the rotation (as in the
pyvene tutorial: first V1's dims from source 1, then V2's dims from source 2).
"""

import numpy as np
import pytest
import torch

from cas.das import FitSettings, fit_rotation, pair_activations, predict
from cas.pairs import make_split
from cas.tasks import TASKS
from cas.train import new_network, train_iit


@pytest.mark.slow
def test_iia_matches_pyvene_rotated_space_intervention():
    from pyvene import RotatedSpaceIntervention

    task, hyp, w, layer, k = TASKS["hierarchical_equality"], "both_equality", 16, 1, 4
    m = new_network(task, w, 3, 0)
    s = dict(lr=1e-3, batch_size=1024, max_steps=3000, eval_every=250, eval_n=2000,
             target_acc=0.99, target_iit_acc=0.98)
    train_iit(m, task, hyp, layer, k, np.random.default_rng(0), s)
    for p in m.parameters():
        p.requires_grad_(False)
    sp = make_split(task, hyp, 1024, np.random.default_rng(1))
    acts = pair_activations(m, layer, torch.tensor(sp.base_x), torch.tensor(sp.src_x),
                            sp.pair_base, sp.pair_src, sp.labels[hyp])
    R = fit_rotation(m, layer, k, acts, 0, FitSettings(max_steps=600)).R
    ours = predict(m, layer, acts, R, k)

    iv = RotatedSpaceIntervention(embed_dim=w)
    with torch.no_grad():
        iv.rotate_layer.weight = R  # orthogonal parametrization: right_inverse keeps R
    assert torch.allclose(iv.rotate_layer.weight, R, atol=1e-5)
    h = acts.h_base.clone()
    with torch.no_grad():
        for var, dims in ((0, list(range(0, k))), (1, list(range(k, 2 * k)))):
            rows = acts.mask[:, var]
            n = int(rows.sum())
            if n:
                h[rows] = iv(h[rows], acts.h_src[rows, var], subspaces=[dims] * n)
        theirs = m.from_hidden(h, layer).argmax(1).numpy()
    assert (theirs == ours).mean() > 0.999
    iia = (ours == sp.labels[hyp]).mean()
    assert iia > 0.9  # IIT network at its site
