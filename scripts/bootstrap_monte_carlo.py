"""Monte Carlo check of the bootstrap SDs (docs/code/procedures_and_statistics.md).

Run: uv run python scripts/bootstrap_monte_carlo.py   (~20 s).  For each of four units with IIA
well away from 0.5 and 1, fix the fitted rotation R, draw 300 fresh test splits (1024 pairs), and
compare the SD of IIA across them with bootstrap SDs computed on one split.
"""
import numpy as np, torch
torch.set_num_threads(4)
from cas.tasks import TASKS
from cas.train import new_network, train_iit, train_standard
from cas.pairs import make_split
from cas.das import FitSettings, fit_rotation, predict, pair_activations
from cas.bootstrap import bootstrap_iia
dl, he = TASKS["distributive_law"], TASKS["hierarchical_equality"]
S = dict(lr=1e-3, batch_size=1024, max_steps=20000, eval_every=250, eval_n=4000, target_acc=0.995, target_iit_acc=0.99)
units = []
for seed in (1, 2):
    m = new_network(dl, 24, 3, seed); train_iit(m, dl, "and_or_and", 1, 6, np.random.default_rng(seed), S)
    units.append((f"DL IIT-AOA net{seed} vs and_or, L1 k6", m, dl, "and_or", 1, 6))
m = new_network(he, 16, 3, 7); train_standard(m, he, np.random.default_rng(7), S)
units.append(("HE standard net, L0 k4", m, he, "both_equality", 0, 4))
units.append(("HE standard net, L1 k8", m, he, "both_equality", 1, 8))
for name, m, task, hyp, L, k in units:
    for p in m.parameters(): p.requires_grad_(False)
    A = lambda sp: pair_activations(m, L, torch.tensor(sp.base_x), torch.tensor(sp.src_x), sp.pair_base, sp.pair_src, sp.labels[hyp])
    fit = make_split(task, hyp, 512, np.random.default_rng(100))
    R = fit_rotation(m, L, k, A(fit), 0, FitSettings()).R
    iias = []
    for r in range(300):
        sp = make_split(task, hyp, 1024, np.random.default_rng(1000 + r))
        c = predict(m, L, A(sp), R, k) == sp.labels[hyp]
        iias.append(c.mean())
        if r == 0:
            boots = {meth: np.std(bootstrap_iia(c, sp.pair_base, sp.pair_src, meth, 2000, np.random.default_rng(9))) for meth in ("iid", "base", "twoway")}
    print(f"{name}: mean IIA {np.mean(iias):.3f}, Monte Carlo SD {np.std(iias, ddof=1):.4f}; bootstrap SD on one split: " + ", ".join(f"{k} {v:.4f}" for k, v in boots.items()), flush=True)
