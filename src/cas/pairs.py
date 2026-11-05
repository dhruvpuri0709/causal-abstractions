"""Interchange-intervention pairs with disjoint input pools and recorded ids.

A *split* (fit, select or test) has its own pool of base inputs and its own pool of source
inputs, drawn fresh, so different splits share no inputs (the vectors are continuous draws; the
chance of an accidental repeat is zero).  A pair is (base, source for V1 or -1, source for V2 or
-1).  The intervention type is V1 only, V2 only or both, uniformly (Geiger et al. 2024 App. A.1;
Sutter et al. App. I.3).

Each base input is used in at most ``base_uses`` pairs and each source input in at most
``source_uses`` pairs, so a pair's base and sources are recorded clusters for the two-way
bootstrap.  Pairs are stratified so that the four cells of
(counterfactual label, label changed relative to the base label) each hold a quarter of the pairs.
That makes the majority-class guess and the "ignore the intervention" guess both score 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tasks import ALGORITHMS, N_VARS, Task, make_inputs, sample_bits

SPLITS = ("fit", "select", "test")


@dataclass
class PairSplit:
    base_x: np.ndarray  # (n_b, d) float32
    base_bits: np.ndarray  # (n_b, n_eq) bool
    src_x: np.ndarray  # (n_s, d)
    src_bits: np.ndarray  # (n_s, n_eq)
    pair_base: np.ndarray  # (n,) int index into the base pool
    pair_src: np.ndarray  # (n, 2) int index into the source pool, -1 = variable not intervened
    labels: dict  # algorithm name -> (n,) int8 counterfactual labels, for every task algorithm
    base_labels: np.ndarray  # (n,) int8 factual label of the base

    def __len__(self):
        return len(self.pair_base)

    def mask(self) -> np.ndarray:
        return self.pair_src >= 0


def counterfactual_labels(task: Task, alg_name: str, base_bits, src_bits, pair_base, pair_src):
    alg = ALGORITHMS[alg_name]
    mask = pair_src >= 0
    sb = src_bits[np.clip(pair_src, 0, None)]  # (n, 2, n_eq); rows with mask False are ignored
    return alg.counterfactual(base_bits[pair_base], sb, mask)


def make_split(task: Task, hypothesis: str, n_pairs: int, rng: np.random.Generator,
               base_uses: int = 2, source_uses: int = 2, slack: float = 1.5) -> PairSplit:
    """Generate one split's pools and pairs, stratified on (label, changed)."""
    n_b = int(np.ceil(slack * n_pairs / base_uses))
    n_s = int(np.ceil(slack * n_pairs * (4 / 3) / source_uses))  # 4/3 sources per pair on average
    base_bits = sample_bits(task, n_b, rng)
    src_bits = sample_bits(task, n_s, rng)
    alg = ALGORITHMS[hypothesis]
    base_lab = task.label(base_bits)

    quota = np.full(4, n_pairs // 4)
    quota[: n_pairs % 4] += 1
    filled = np.zeros(4, dtype=int)
    b_left = np.full(n_b, base_uses)
    s_left = np.full(n_s, source_uses)
    pb_out, ps_out = [], []
    for _round in range(200):
        m = 8 * n_pairs
        cb = rng.integers(n_b, size=m)
        cs = rng.integers(n_s, size=(m, N_VARS))
        t = rng.integers(3, size=m)  # 0: V1, 1: V2, 2: both
        cs[t == 1, 0] = -1
        cs[t == 0, 1] = -1
        lab = alg.counterfactual(base_bits[cb], src_bits[np.clip(cs, 0, None)], cs >= 0)
        stratum = 2 * lab.astype(int) + (lab != base_lab[cb]).astype(int)
        for j in range(m):
            st = stratum[j]
            if filled[st] >= quota[st]:
                continue
            b, s0, s1 = cb[j], cs[j, 0], cs[j, 1]
            if b_left[b] == 0 or (s0 >= 0 and s_left[s0] == 0) or (s1 >= 0 and s_left[s1] == 0):
                continue
            if s0 >= 0 and s0 == s1:
                if s_left[s0] < 2:
                    continue
            b_left[b] -= 1
            if s0 >= 0:
                s_left[s0] -= 1
            if s1 >= 0:
                s_left[s1] -= 1
            filled[st] += 1
            pb_out.append(b)
            ps_out.append((s0, s1))
            if filled.sum() == n_pairs:
                break
        if filled.sum() == n_pairs:
            break
    else:
        raise RuntimeError(f"could not fill strata {filled} of {quota}; increase slack")

    order = rng.permutation(n_pairs)
    pair_base = np.asarray(pb_out, dtype=np.int64)[order]
    pair_src = np.asarray(ps_out, dtype=np.int64)[order]
    labels = {a: counterfactual_labels(task, a, base_bits, src_bits, pair_base, pair_src)
              .astype(np.int8) for a in task.algorithms}
    return PairSplit(
        base_x=make_inputs(task, base_bits, rng), base_bits=base_bits,
        src_x=make_inputs(task, src_bits, rng), src_bits=src_bits,
        pair_base=pair_base, pair_src=pair_src, labels=labels,
        base_labels=base_lab[pair_base].astype(np.int8))


def make_splits(task: Task, hypothesis: str, sizes: dict, rng: np.random.Generator,
                base_uses: int = 2, source_uses: int = 2) -> dict[str, PairSplit]:
    return {s: make_split(task, hypothesis, sizes[s], rng, base_uses, source_uses)
            for s in SPLITS}


def save_splits(splits: dict[str, PairSplit], path, fingerprint: str = "") -> None:
    """Save to a path or an open binary file; ``fingerprint`` identifies the settings."""
    arrays = {"fingerprint": np.array(fingerprint)}
    for s, sp in splits.items():
        for name in ("base_x", "base_bits", "src_x", "src_bits", "pair_base", "pair_src",
                     "base_labels"):
            arrays[f"{s}/{name}"] = getattr(sp, name)
        for a, lab in sp.labels.items():
            arrays[f"{s}/label/{a}"] = lab
    np.savez_compressed(path, **arrays)


def load_fingerprint(path) -> str:
    z = np.load(path)
    return str(z["fingerprint"]) if "fingerprint" in z.files else ""


def load_splits(path) -> dict[str, PairSplit]:
    z = np.load(path)
    out = {}
    for s in SPLITS:
        labels = {k.split("/")[-1]: z[k] for k in z.files if k.startswith(f"{s}/label/")}
        out[s] = PairSplit(**{n: z[f"{s}/{n}"] for n in (
            "base_x", "base_bits", "src_x", "src_bits", "pair_base", "pair_src", "base_labels")},
            labels=labels)
    return out
