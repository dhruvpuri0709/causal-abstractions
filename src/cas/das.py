"""Minimal distributed alignment search (DAS) for MLPs.

Interchange semantics (Geiger et al. 2024 Def. 2; pyvene ``RotatedSpaceIntervention``): for a
hidden layer with activations h in R^w and an orthogonal matrix R (w x w),

    r_b = h(base) R,   r_i = h(source_i) R
    r'  = r_b, except r'[dims_i] = r_i[dims_i] for every intervened variable i
    h'  = r' R^T          (R^T = R^{-1})

and the network continues from h'.  Variable i (i = 0, 1) is aligned with rotated dims
[i k, (i + 1) k).  With R = I this is an interchange of neurons, which is what IIT training and
the handcrafted-network check use.  IIA is the share of pairs where argmax of the intervened
network's logits equals the algorithm's counterfactual label.

The network is frozen, so activations of every input at the target layer are computed once and
only R and the layers above are evaluated during the fit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from .models import MLP


def interchange(h_base: torch.Tensor, h_src: torch.Tensor, mask: torch.Tensor,
                R: torch.Tensor | None, k: int) -> torch.Tensor:
    """Interchange intervention in the rotated basis.

    h_base: (n, w); h_src: (n, n_vars, w); mask: (n, n_vars) bool; R: (w, w) or None (identity).
    Variable i uses rotated dims [i k, (i+1) k).
    """
    n, n_vars, w = h_src.shape
    assert n_vars * k <= w, f"{n_vars} variables x k={k} do not fit in width {w}"
    r_b = h_base if R is None else h_base @ R
    r_s = h_src if R is None else h_src @ R
    dims = torch.zeros(n_vars, w, dtype=h_base.dtype, device=h_base.device)
    for i in range(n_vars):
        dims[i, i * k:(i + 1) * k] = 1.0
    sel = mask.to(h_base.dtype)[:, :, None] * dims[None]  # (n, n_vars, w), 1 = take from source
    r_new = r_b + (sel * (r_s - r_b[:, None, :])).sum(dim=1)
    return r_new if R is None else r_new @ R.T


class Rotation(nn.Module):
    """Orthogonal w x w matrix via torch's orthogonal parametrization, random orthogonal init."""

    def __init__(self, w: int, seed: int):
        super().__init__()
        lin = nn.Linear(w, w, bias=False)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            nn.init.orthogonal_(lin.weight, generator=g)
        self.lin = orthogonal(lin)

    @property
    def R(self) -> torch.Tensor:
        return self.lin.weight


@dataclass
class PairActivations:
    """Activations at one layer for one split's pairs, ready for interchange."""
    h_base: torch.Tensor  # (n, w)
    h_src: torch.Tensor  # (n, 2, w); rows of non-intervened variables are the base activations
    mask: torch.Tensor  # (n, 2) bool
    labels: torch.Tensor  # (n,) int64

    def __len__(self):
        return self.h_base.shape[0]

    def with_labels(self, labels: torch.Tensor) -> "PairActivations":
        return PairActivations(self.h_base, self.h_src, self.mask, labels)


def pair_activations(model: MLP, layer: int, x_base: torch.Tensor, x_src: torch.Tensor,
                     pair_base: np.ndarray, pair_src: np.ndarray,
                     labels: np.ndarray) -> PairActivations:
    """Gather layer activations for pairs from pools of base and source inputs.

    x_base: (n_b, d) base pool; x_src: (n_s, d) source pool; pair_base: (n,) index into the base
    pool; pair_src: (n, 2) index into the source pool, -1 when the variable is not intervened.
    """
    dev = x_base.device
    with torch.no_grad():
        hb_pool = model.hidden(x_base, layer)
        hs_pool = model.hidden(x_src, layer)
    pb = torch.as_tensor(pair_base, device=dev)
    ps = torch.as_tensor(pair_src, device=dev)
    mask = ps >= 0
    h_base = hb_pool[pb]
    h_src = hs_pool[ps.clamp(min=0)]
    h_src = torch.where(mask[:, :, None], h_src, h_base[:, None, :])
    return PairActivations(h_base, h_src, mask, torch.as_tensor(labels, device=dev).long())


@torch.no_grad()
def predict(model: MLP, layer: int, acts: PairActivations, R: torch.Tensor | None, k: int,
            batch_size: int = 8192) -> np.ndarray:
    preds = []
    for i in range(0, len(acts), batch_size):
        sl = slice(i, i + batch_size)
        h = interchange(acts.h_base[sl], acts.h_src[sl], acts.mask[sl], R, k)
        preds.append(model.from_hidden(h, layer).argmax(dim=1))
    return torch.cat(preds).cpu().numpy().astype(np.int8)


@dataclass
class FitSettings:
    lr: float = 0.01
    batch_size: int = 128
    max_steps: int = 4000
    window: int = 50  # steps per loss window (mean training loss over the window)
    patience: int = 5  # windows without improvement (by > tol) before stopping
    tol: float = 1e-3  # absolute improvement in window-mean training loss
    min_loss: float = 1e-3  # stop (converged) once the window-mean loss is below this


@dataclass
class FitResult:
    R: torch.Tensor
    losses: list = field(default_factory=list)  # window-mean training losses
    steps: int = 0
    converged: bool = False
    stop_reason: str = ""
    seconds: float = 0.0


def fit_rotation(model: MLP, layer: int, k: int, train: PairActivations, seed: int,
                 settings: FitSettings) -> FitResult:
    """Train R by cross-entropy between intervened logits and counterfactual labels.

    Minibatches cycle through random permutations of the fitting pairs, so the step budget does
    not depend on the number of pairs.  Every ``window`` steps the mean training loss over the
    window is recorded.  Convergence (recorded; non-converged fits are excluded downstream): the
    window loss stopped improving by more than ``tol`` for ``patience`` consecutive windows, or
    fell below ``min_loss``.  Reaching ``max_steps`` first, or a non-finite loss, counts as not
    converged.  The final R is returned (not the best window's).
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    dev = train.h_base.device
    w = train.h_base.shape[1]
    rot = Rotation(w, seed).to(dev)
    opt = torch.optim.Adam(rot.parameters(), lr=settings.lr)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed + 1)
    n = len(train)
    bs = min(settings.batch_size, n)
    best, since_best, window_total = float("inf"), 0, 0.0
    res = FitResult(R=None)
    perm, pos = torch.randperm(n, generator=g).to(dev), 0
    for step in range(1, settings.max_steps + 1):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g).to(dev), 0
        idx = perm[pos:pos + bs]
        pos += bs
        h = interchange(train.h_base[idx], train.h_src[idx], train.mask[idx], rot.R, k)
        loss = loss_fn(model.from_hidden(h, layer), train.labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        window_total += loss.item()
        res.steps = step
        if step % settings.window:
            continue
        wl = window_total / settings.window
        window_total = 0.0
        res.losses.append(wl)
        if not np.isfinite(wl):
            res.stop_reason = "non-finite loss"
            break
        if wl < settings.min_loss:
            res.converged, res.stop_reason = True, "loss below min_loss"
            break
        if wl < best - settings.tol:
            best, since_best = wl, 0
        else:
            since_best += 1
            if since_best >= settings.patience:
                res.converged, res.stop_reason = True, "plateau"
                break
    else:
        res.stop_reason = "max_steps"
    with torch.no_grad():
        res.R = rot.R.detach().clone()
    res.seconds = time.perf_counter() - t0
    return res
