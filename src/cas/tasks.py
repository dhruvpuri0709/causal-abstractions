"""Synthetic tasks and the high-level algorithms (causal models) that solve them.

Every task input is a concatenation of ``n_objects`` vectors in R^emb_dim, grouped into
consecutive pairs (x0, x1), (x2, x3), ...  Each pair is either *equal* (the second vector is a
copy of the first) or *different* (two independent draws).  The only thing an algorithm needs
from an input is the vector of equality bits ``e[j] = (x_{2j} == x_{2j+1})``, so we sample bits
first and then draw vectors that realise them.  We keep the bits alongside the vectors so that
labels never have to be recomputed from floats.

Tasks (the definitions follow Geiger et al. 2024 §4 and Sutter et al. 2025 App. I.3):

* ``hierarchical_equality``: 4 objects, label = (x0 == x1) == (x2 == x3).
  Algorithm ``both_equality``: V1 = e0, V2 = e1, out = (V1 == V2).
* ``distributive_law``: 6 objects, label = (e0 and e1) or (e1 and e2).
  Algorithm ``and_or_and``: V1 = e0 and e1, V2 = e1 and e2, out = V1 or V2.
  Algorithm ``and_or``:     V1 = e1,         V2 = e0 or e2,  out = V1 and V2.
  The two algorithms compute the same function (distributive law) but have different
  intermediate variables.

Every algorithm here has exactly two intermediate variables (V1, V2).  An interchange intervention
sets V_i of the base input to the value V_i takes on source input s_i, for the variables in a mask;
the counterfactual label is the algorithm's output after that intervention.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

EMB_DIM = 4
LOW, HIGH = -0.5, 0.5  # range of the uniform object vectors (as in Sutter et al.)
N_VARS = 2  # every algorithm in this module has two intermediate variables


@dataclass(frozen=True)
class Algorithm:
    name: str
    var_names: tuple[str, str]
    variables: Callable[[np.ndarray], np.ndarray]  # bits (n, n_eq) -> (n, 2) bool
    output: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (V1, V2) -> (n,) bool

    def run(self, bits: np.ndarray) -> np.ndarray:
        v = self.variables(bits)
        return self.output(v[:, 0], v[:, 1])

    def counterfactual(self, base_bits: np.ndarray, source_bits: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
        """Output after interchange interventions.

        base_bits: (n, n_eq); source_bits: (n, 2, n_eq), source_bits[:, i] is the source for V_i;
        mask: (n, 2) bool, mask[:, i] says whether V_i is intervened on.  Rows of source_bits
        for non-intervened variables are ignored.
        """
        v = self.variables(base_bits)
        for i in range(N_VARS):
            v_src = self.variables(source_bits[:, i])[:, i]
            v[:, i] = np.where(mask[:, i], v_src, v[:, i])
        return self.output(v[:, 0], v[:, 1])


def _both_equality_vars(e):
    return np.stack([e[:, 0], e[:, 1]], axis=1).astype(bool)


def _and_or_and_vars(e):
    return np.stack([e[:, 0] & e[:, 1], e[:, 1] & e[:, 2]], axis=1).astype(bool)


def _and_or_vars(e):
    return np.stack([e[:, 1], e[:, 0] | e[:, 2]], axis=1).astype(bool)


ALGORITHMS = {
    "both_equality": Algorithm("both_equality", ("WX", "YZ"), _both_equality_vars,
                               lambda a, b: a == b),
    "and_or_and": Algorithm("and_or_and", ("(e0&e1)", "(e1&e2)"), _and_or_and_vars,
                            lambda a, b: a | b),
    "and_or": Algorithm("and_or", ("e1", "(e0|e2)"), _and_or_vars,
                        lambda a, b: a & b),
}


@dataclass(frozen=True)
class Task:
    name: str
    short: str
    n_objects: int
    algorithms: tuple[str, ...]  # first one defines the task label (all agree)
    emb_dim: int = EMB_DIM

    @property
    def n_eq(self) -> int:
        return self.n_objects // 2

    @property
    def input_dim(self) -> int:
        return self.n_objects * self.emb_dim

    def label(self, bits: np.ndarray) -> np.ndarray:
        return ALGORITHMS[self.algorithms[0]].run(bits)

    def patterns_by_label(self) -> dict[bool, np.ndarray]:
        pats = np.array(list(itertools.product([False, True], repeat=self.n_eq)), dtype=bool)
        lab = self.label(pats)
        return {False: pats[~lab], True: pats[lab]}


TASKS = {
    "hierarchical_equality": Task("hierarchical_equality", "he", 4, ("both_equality",)),
    "distributive_law": Task("distributive_law", "dl", 6, ("and_or_and", "and_or")),
}


def sample_bits(task: Task, n: int, rng: np.random.Generator) -> np.ndarray:
    """Equality bits with a balanced task label; uniform over patterns within each label.

    For hierarchical equality this is the uniform distribution over the four patterns (as in the
    pyvene tutorial); for the distributive law it matches Sutter et al.'s rejection sampler.
    """
    pats = task.patterns_by_label()
    lab = rng.random(n) < 0.5
    out = np.empty((n, task.n_eq), dtype=bool)
    for value in (False, True):
        idx = np.flatnonzero(lab == value)
        p = pats[value]
        out[idx] = p[rng.integers(len(p), size=len(idx))]
    return out


def make_inputs(task: Task, bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw float32 input vectors (n, input_dim) that realise the given equality bits."""
    n = bits.shape[0]
    first = rng.uniform(LOW, HIGH, size=(n, task.n_eq, task.emb_dim))
    other = rng.uniform(LOW, HIGH, size=(n, task.n_eq, task.emb_dim))
    second = np.where(bits[:, :, None], first, other)
    x = np.stack([first, second], axis=2)  # (n, n_eq, 2, emb)
    return x.reshape(n, task.input_dim).astype(np.float32)


def bits_from_inputs(task: Task, x: np.ndarray) -> np.ndarray:
    """Recover equality bits from vectors (used only in tests and sanity checks)."""
    v = x.reshape(x.shape[0], task.n_eq, 2, task.emb_dim)
    return np.all(v[:, :, 0] == v[:, :, 1], axis=-1)


def sample_task_data(task: Task, n: int, rng: np.random.Generator):
    """Factual (x, y) for ordinary training: balanced labels."""
    bits = sample_bits(task, n, rng)
    return make_inputs(task, bits, rng), task.label(bits).astype(np.int64)


def sample_interchange_batch(task: Task, alg: Algorithm, n: int, rng: np.random.Generator,
                             p_types=(0.25, 0.25, 0.25, 0.25)):
    """Random interchange examples for interchange intervention training (IIT).

    Intervention type is drawn from (none, V1, V2, both) with probabilities p_types
    (Sutter et al. App. I.3: 1/4 each).  Returns base x, source x (n, 2, d), mask (n, 2), labels.
    """
    base_bits = sample_bits(task, n, rng)
    src_bits = np.stack([sample_bits(task, n, rng) for _ in range(N_VARS)], axis=1)
    t = rng.choice(4, size=n, p=np.asarray(p_types) / np.sum(p_types))
    mask = np.stack([(t == 1) | (t == 3), (t == 2) | (t == 3)], axis=1)
    y = alg.counterfactual(base_bits, src_bits, mask).astype(np.int64)
    xb = make_inputs(task, base_bits, rng)
    xs = np.stack([make_inputs(task, src_bits[:, i], rng) for i in range(N_VARS)], axis=1)
    return xb, xs, mask, y
