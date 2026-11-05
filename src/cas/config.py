"""Experiment configuration (YAML) and the deterministic enumeration of networks and analyses.

Everything downstream is a pure function of the config: network ids, training seeds, DAS pair
seeds and rotation seeds are all derived from ``seed`` and the object's id with SHA-256, so the
same config gives the same results regardless of worker count or job order.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from .tasks import TASKS

DEFAULTS = {
    "name": "unnamed",
    "seed": 0,
    "results_dir": "results",
    "tasks": {"hierarchical_equality": {"widths": [16]}},
    "network": {"n_layers": 3, "iit_layer": 1, "iit_k_frac": 0.25},
    "networks": {"eval": {"iit": 4, "random": 4, "standard": 2},
                 "calibration": {"random": 19, "iit": 19}},
    "train": {
        "standard": {"lr": 1e-3, "batch_size": 1024, "max_steps": 20000, "eval_every": 250,
                     "eval_n": 4000, "target_acc": 0.995, "pass_acc": 0.99},
        "iit": {"lr": 1e-3, "batch_size": 1024, "max_steps": 20000, "eval_every": 250,
                "eval_n": 4000, "target_acc": 0.995, "target_iit_acc": 0.99,
                "pass_acc": 0.99, "pass_iit_acc": 0.98},
    },
    "das": {
        "layers": [0, 1, 2], "k_fracs": [0.125, 0.25, 0.5], "seeds": 2,
        "n_fit": [1024], "n_select": 1024, "n_test": 1024,
        "base_uses": 2, "source_uses": 2,
        "fit": {"lr": 0.01, "batch_size": 128, "max_steps": 4000, "window": 50,
                "patience": 5, "tol": 1e-3, "min_loss": 1e-3},
    },
    "null": {"shuffle_B": 19},
    "analysis": {"threshold": 0.9,
                 "thresholds": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
                 "alpha": 0.05, "bootstrap_B": 2000},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (over or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict) and key != "tasks":
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def derive_seed(*parts) -> int:
    """A 31-bit seed from any sequence of strings/ints (stable across runs and machines)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def k_values(width: int, k_fracs) -> list[int]:
    return sorted({max(1, round(f * width)) for f in k_fracs})


def iit_k(width: int, cfg: dict) -> int:
    return max(1, round(cfg["network"]["iit_k_frac"] * width))


@dataclass(frozen=True)
class NetworkSpec:
    net_id: str
    task: str
    width: int
    kind: str  # random | standard | iit
    alg: str | None  # algorithm an IIT network was trained toward
    role: str  # eval | cal
    idx: int
    seed: int


@dataclass(frozen=True)
class UnitSpec:
    """One DAS analysis: a network, a hypothesised algorithm and a fitting-set size."""
    unit_id: str
    net_id: str
    task: str
    width: int
    kind: str
    alg: str | None
    role: str
    hypothesis: str
    n_fit: int
    truth: str  # positive | negative | unknown
    neg_type: str | None  # random | wrong_algorithm (for negatives and calibration units)
    pair_seed: int


class Config(dict):
    @classmethod
    def load(cls, path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls(_merge(DEFAULTS, raw))
        cfg["config_path"] = str(path)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls(_merge(DEFAULTS, d))
        cfg.validate()
        return cfg

    def validate(self):
        for t in self["tasks"]:
            if t not in TASKS:
                raise ValueError(f"unknown task {t}; known: {list(TASKS)}")
        n_layers = self["network"]["n_layers"]
        if self["network"]["iit_layer"] >= n_layers:
            raise ValueError("iit_layer must be < n_layers")
        if any(layer >= n_layers for layer in self["das"]["layers"]):
            raise ValueError("das.layers must be < n_layers")
        if self["network"]["iit_layer"] not in self["das"]["layers"]:
            raise ValueError("das.layers must contain iit_layer (the pre-registered site)")
        if max(self["das"]["k_fracs"]) > 0.5 or self["network"]["iit_k_frac"] > 0.5:
            raise ValueError("k fractions must be <= 0.5 (two variables per layer)")

    @property
    def out_dir(self) -> Path:
        return Path(self["results_dir"]) / self["name"]

    # ---- design --------------------------------------------------------------------------
    def widths(self, task: str) -> list[int]:
        return list(self["tasks"][task]["widths"])

    def ks(self, width: int) -> list[int]:
        """DAS subspace sizes for this width; always includes the pre-registered k*."""
        return sorted(set(k_values(width, self["das"]["k_fracs"])) | {iit_k(width, self)})

    def fixed_config(self, width: int) -> tuple[int, int, int]:
        """Pre-registered (layer, k, seed_index): the IIT site, first rotation seed."""
        return self["network"]["iit_layer"], iit_k(width, self), 0

    def das_configs(self, width: int) -> list[tuple[int, int, int]]:
        return [(layer, k, s) for layer in self["das"]["layers"] for k in self.ks(width)
                for s in range(self["das"]["seeds"])]

    def networks(self) -> list[NetworkSpec]:
        out = []
        for task in self["tasks"]:
            t = TASKS[task]
            for width in self.widths(task):
                for role, counts in (("eval", self["networks"]["eval"]),
                                     ("cal", self["networks"]["calibration"])):
                    for kind in ("iit", "random", "standard"):
                        n = counts.get(kind, 0)
                        if kind == "iit":
                            algs = t.algorithms
                            if role == "cal" and len(algs) < 2:
                                continue  # wrong-algorithm null needs a second algorithm
                        else:
                            algs = (None,)
                        for alg in algs:
                            for i in range(n):
                                tag = kind if alg is None else f"{kind}-{alg}"
                                net_id = f"{t.short}_w{width}_{tag}_{role}{i:02d}"
                                out.append(NetworkSpec(net_id, task, width, kind, alg, role, i,
                                                       derive_seed(self["seed"], "net", net_id)))
        return out

    def units(self) -> list[UnitSpec]:
        out = []
        for net in self.networks():
            for hyp in TASKS[net.task].algorithms:
                if net.kind == "iit":
                    if hyp == net.alg:
                        if net.role == "cal":
                            continue  # a positive is never a null
                        truth, neg = "positive", None
                    else:
                        truth, neg = "negative", "wrong_algorithm"
                elif net.kind == "random":
                    truth, neg = "negative", "random"
                else:
                    if net.role == "cal":
                        continue
                    truth, neg = "unknown", None
                for n_fit in self["das"]["n_fit"]:
                    uid = f"{net.net_id}__{hyp}__n{n_fit}"
                    out.append(UnitSpec(uid, net.net_id, net.task, net.width, net.kind, net.alg,
                                        net.role, hyp, n_fit, truth, neg,
                                        derive_seed(self["seed"], "pairs", uid)))
        return out
