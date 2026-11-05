"""Building the networks: frozen random, standard (task loss only) and IIT-trained.

* ``random``: PyTorch's default initialisation, never trained (Geiger et al. 2024 §4.4).  These
  are at chance on the task; that is expected.
* ``standard``: trained on the task label only.  Whether they implement an algorithm is unknown.
* ``iit``: interchange intervention training (Geiger et al. 2022) toward one algorithm.  With the
  identity alignment, V1 lives in neurons [0, k*) and V2 in neurons [k*, 2k*) of hidden layer
  L*.  Each batch mixes factual examples and interchange examples on V1, V2 or both (1/4 each,
  Sutter et al. App. I.3); the loss is cross-entropy against the algorithm's counterfactual label.
  A network IIT-trained to algorithm A is a known positive for A at (L*, k*) and, for the
  distributive law, a ``wrong_algorithm`` negative for the other algorithm.

All training data are fresh random vectors drawn on the fly, so no input is ever reused, and the
DAS pairs (drawn later from other seeds) are disjoint from the training inputs.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .das import interchange
from .models import MLP
from .tasks import ALGORITHMS, Task, sample_interchange_batch, sample_task_data


def _t(a, device):
    return torch.as_tensor(a, device=device)


@torch.no_grad()
def task_accuracy(model: MLP, task: Task, n: int, rng: np.random.Generator,
                  device="cpu") -> float:
    x, y = sample_task_data(task, n, rng)
    return float((model(_t(x, device)).argmax(1).cpu().numpy() == y).mean())


@torch.no_grad()
def iit_accuracy(model: MLP, task: Task, alg_name: str, layer: int, k: int, n: int,
                 rng: np.random.Generator, device="cpu") -> float:
    """Neuron-aligned interchange accuracy on interventions only (V1, V2 or both, 1/3 each)."""
    xb, xs, mask, y = sample_interchange_batch(task, ALGORITHMS[alg_name], n, rng,
                                               p_types=(0, 1, 1, 1))
    xb, xs = _t(xb, device), _t(xs, device)
    hb = model.hidden(xb, layer)
    hs = model.hidden(xs.reshape(-1, xs.shape[-1]), layer).reshape(n, 2, -1)
    h = interchange(hb, hs, _t(mask, device), None, k)
    return float((model.from_hidden(h, layer).argmax(1).cpu().numpy() == y).mean())


def new_network(task: Task, width: int, n_layers: int, seed: int) -> MLP:
    torch.manual_seed(seed)
    return MLP(task.input_dim, width, n_layers)


def train_standard(model: MLP, task: Task, rng: np.random.Generator, s: dict,
                   device="cpu", log=None) -> dict:
    """Adam on cross-entropy of the task label.  Stops when held-out accuracy >= target."""
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"])
    loss_fn = nn.CrossEntropyLoss()
    t0, step, acc = time.perf_counter(), 0, 0.0
    model.train()
    while step < s["max_steps"]:
        x, y = sample_task_data(task, s["batch_size"], rng)
        loss = loss_fn(model(_t(x, device)), _t(y, device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % s["eval_every"] == 0:
            model.eval()
            acc = task_accuracy(model, task, s["eval_n"], rng, device)
            model.train()
            if log:
                log(f"step {step} loss {loss.item():.4f} task acc {acc:.4f}")
            if acc >= s["target_acc"]:
                break
    model.eval()
    return {"steps": step, "train_seconds": time.perf_counter() - t0}


def train_iit(model: MLP, task: Task, alg_name: str, layer: int, k: int,
              rng: np.random.Generator, s: dict, device="cpu", log=None) -> dict:
    """Interchange intervention training with the identity alignment at (layer, k)."""
    alg = ALGORITHMS[alg_name]
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"])
    loss_fn = nn.CrossEntropyLoss()
    t0, step = time.perf_counter(), 0
    model.train()
    n = s["batch_size"]
    while step < s["max_steps"]:
        xb, xs, mask, y = sample_interchange_batch(task, alg, n, rng)
        xb, xs = _t(xb, device), _t(xs, device)
        hb = model.hidden(xb, layer)
        hs = model.hidden(xs.reshape(-1, xs.shape[-1]), layer).reshape(n, 2, -1)
        h = interchange(hb, hs, _t(mask, device), None, k)
        loss = loss_fn(model.from_hidden(h, layer), _t(y, device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % s["eval_every"] == 0:
            model.eval()
            acc = task_accuracy(model, task, s["eval_n"], rng, device)
            iacc = iit_accuracy(model, task, alg_name, layer, k, s["eval_n"], rng, device)
            model.train()
            if log:
                log(f"step {step} loss {loss.item():.4f} task acc {acc:.4f} iit acc {iacc:.4f}")
            if acc >= s["target_acc"] and iacc >= s["target_iit_acc"]:
                break
    model.eval()
    return {"steps": step, "train_seconds": time.perf_counter() - t0}
