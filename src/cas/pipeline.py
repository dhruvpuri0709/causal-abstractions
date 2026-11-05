"""The ``train`` and ``fit`` stages.  Both are resumable: finished outputs are skipped.

Layout under results/<name>/:

    networks/<net_id>.pt|.json        weights + metadata (accuracies, pass/fail)
    fits/<unit_id>/pairs.npz          the unit's fit/select/test pools and pairs (ids, labels)
    fits/<unit_id>/L{l}_k{k}_s{s}.npz one DAS fit: per-pair predictions and correctness per split
    fits/<unit_id>/shuffle_b{b}.npz   one shuffled-label null fit at the pre-registered config
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import Config, NetworkSpec, UnitSpec, derive_seed, iit_k
from .das import FitSettings, fit_rotation, pair_activations, predict
from .models import load_network, save_network
from .pairs import SPLITS, load_splits, make_splits, save_splits
from .parallel import run_jobs
from .tasks import TASKS
from .train import iit_accuracy, new_network, task_accuracy, train_iit, train_standard

FINAL_EVAL_N = 10000


def network_path(cfg: Config, net_id: str) -> Path:
    return cfg.out_dir / "networks" / net_id


# ---- train -------------------------------------------------------------------------------
def build_network(job) -> dict:
    cfg, spec, device = Config(job["cfg"]), NetworkSpec(**job["spec"]), job["device"]
    path = network_path(cfg, spec.net_id)
    if path.with_suffix(".json").exists():
        return json.loads(path.with_suffix(".json").read_text())
    task = TASKS[spec.task]
    layer, k = cfg["network"]["iit_layer"], iit_k(spec.width, cfg)
    model = new_network(task, spec.width, cfg["network"]["n_layers"], spec.seed).to(device)
    rng = np.random.default_rng(spec.seed)
    info = {"steps": 0, "train_seconds": 0.0}
    if spec.kind == "standard":
        info = train_standard(model, task, rng, cfg["train"]["standard"], device)
    elif spec.kind == "iit":
        info = train_iit(model, task, spec.alg, layer, k, rng, cfg["train"]["iit"], device)
    for p in model.parameters():
        p.requires_grad_(False)
    eval_rng = np.random.default_rng(derive_seed(spec.seed, "final-eval"))
    meta = asdict(spec) | info | {"iit_layer": layer, "iit_k": k}
    meta["task_acc"] = task_accuracy(model, task, FINAL_EVAL_N, eval_rng, device)
    for alg in task.algorithms:  # neuron-aligned IIA at the IIT site, for every algorithm
        meta[f"neuron_iia_{alg}"] = iit_accuracy(model, task, alg, layer, k, FINAL_EVAL_N,
                                                 eval_rng, device)
    if spec.kind == "random":
        meta["passed"] = True
    elif spec.kind == "standard":
        meta["passed"] = meta["task_acc"] >= cfg["train"]["standard"]["pass_acc"]
    else:
        s = cfg["train"]["iit"]
        meta["passed"] = bool(meta["task_acc"] >= s["pass_acc"]
                              and meta[f"neuron_iia_{spec.alg}"] >= s["pass_iit_acc"])
    save_network(model.cpu(), meta, path)
    return meta


def train_all(cfg: Config, workers: int, device: str) -> list[dict]:
    specs = cfg.networks()
    jobs = [{"cfg": dict(cfg), "spec": asdict(s), "device": device} for s in specs]
    # IIT/standard training is the slow part; start it first so workers stay busy
    jobs.sort(key=lambda j: j["spec"]["kind"] == "random")
    metas = run_jobs(build_network, jobs, workers, "train networks")
    failed = [m["net_id"] for m in metas if not m["passed"]]
    print(f"{len(metas)} networks ready under {cfg.out_dir / 'networks'}; "
          f"{len(failed)} failed their training targets" + (f": {failed}" if failed else ""))
    return metas


# ---- fit ---------------------------------------------------------------------------------
def _save_npz(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def unit_dir(cfg: Config, unit_id: str) -> Path:
    return cfg.out_dir / "fits" / unit_id


def get_pairs(cfg: Config, unit: UnitSpec):
    path = unit_dir(cfg, unit.unit_id) / "pairs.npz"
    if path.exists():
        return load_splits(path)
    d = cfg["das"]
    sizes = {"fit": unit.n_fit, "select": d["n_select"], "test": d["n_test"]}
    splits = make_splits(TASKS[unit.task], unit.hypothesis, sizes,
                         np.random.default_rng(unit.pair_seed), d["base_uses"], d["source_uses"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("pairs.tmp.npz")
    save_splits(splits, tmp)
    os.replace(tmp, path)
    return splits


def fit_unit(job) -> dict:
    cfg, unit, device = Config(job["cfg"]), UnitSpec(**job["unit"]), job["device"]
    meta_path = network_path(cfg, unit.net_id).with_suffix(".json")
    if not json.loads(meta_path.read_text())["passed"]:
        return {"unit_id": unit.unit_id, "skipped": "network failed training targets",
                "fits": 0}
    d = unit_dir(cfg, unit.unit_id)
    configs = cfg.das_configs(unit.width)
    fixed = cfg.fixed_config(unit.width)
    shuffle_B = cfg["null"]["shuffle_B"] if unit.role == "eval" else 0
    todo = [c for c in configs if not (d / f"L{c[0]}_k{c[1]}_s{c[2]}.npz").exists()]
    todo_shuffle = [b for b in range(shuffle_B) if not (d / f"shuffle_b{b:03d}.npz").exists()]
    if not todo and not todo_shuffle:
        return {"unit_id": unit.unit_id, "fits": 0}

    splits = get_pairs(cfg, unit)
    model, _ = load_network(network_path(cfg, unit.net_id), device)
    settings = FitSettings(**cfg["das"]["fit"])
    acts_cache = {}

    def acts(layer, split):
        if (layer, split) not in acts_cache:
            sp = splits[split]
            acts_cache[(layer, split)] = pair_activations(
                model, layer, torch.as_tensor(sp.base_x, device=device),
                torch.as_tensor(sp.src_x, device=device), sp.pair_base, sp.pair_src,
                sp.labels[unit.hypothesis])
        return acts_cache[(layer, split)]

    n_done = 0
    for layer, k, s in todo:
        seed = derive_seed(cfg["seed"], "rot", unit.unit_id, layer, k, s)
        res = fit_rotation(model, layer, k, acts(layer, "fit"), seed, settings)
        out = {}
        for split in SPLITS:
            pred = predict(model, layer, acts(layer, split), res.R, k)
            out[f"pred_{split}"] = pred
            out[f"correct_{split}"] = pred == splits[split].labels[unit.hypothesis]
        meta = {"unit_id": unit.unit_id, "layer": layer, "k": k, "seed_index": s, "seed": seed,
                "converged": res.converged, "stop_reason": res.stop_reason,
                "steps": res.steps, "seconds": res.seconds}
        _save_npz(d / f"L{layer}_k{k}_s{s}.npz", losses=np.asarray(res.losses),
                  meta=np.array(json.dumps(meta)), **out)
        n_done += 1

    layer, k, _ = fixed
    for b in todo_shuffle:
        prng = np.random.default_rng(derive_seed(cfg["seed"], "perm", unit.unit_id, b))
        lab_fit = prng.permutation(splits["fit"].labels[unit.hypothesis])
        lab_test = prng.permutation(splits["test"].labels[unit.hypothesis])
        seed = derive_seed(cfg["seed"], "shuffle-rot", unit.unit_id, b)
        res = fit_rotation(model, layer, k,
                           acts(layer, "fit").with_labels(torch.as_tensor(lab_fit).long()),
                           seed, settings)
        pred = predict(model, layer, acts(layer, "test"), res.R, k)
        meta = {"unit_id": unit.unit_id, "layer": layer, "k": k, "b": b, "seed": seed,
                "converged": res.converged, "stop_reason": res.stop_reason,
                "steps": res.steps, "seconds": res.seconds}
        _save_npz(d / f"shuffle_b{b:03d}.npz", pred_test=pred, labels_test=lab_test,
                  labels_fit=lab_fit, correct_test=pred == lab_test,
                  losses=np.asarray(res.losses), meta=np.array(json.dumps(meta)))
        n_done += 1
    return {"unit_id": unit.unit_id, "fits": n_done}


def fit_all(cfg: Config, workers: int, device: str) -> list[dict]:
    units = cfg.units()
    missing = [u.net_id for u in units
               if not network_path(cfg, u.net_id).with_suffix(".json").exists()]
    if missing:
        raise SystemExit(f"{len(set(missing))} networks are missing; run `cas train` first "
                         f"(e.g. {missing[0]})")
    jobs = [{"cfg": dict(cfg), "unit": asdict(u), "device": device} for u in units]
    jobs.sort(key=lambda j: -j["unit"]["width"])  # widest (slowest) first
    res = run_jobs(fit_unit, jobs, workers, "DAS fits (units)")
    n = sum(r["fits"] for r in res)
    skipped = [r["unit_id"] for r in res if "skipped" in r]
    print(f"{len(units)} analysis units; {n} new fits written under {cfg.out_dir / 'fits'}; "
          f"{len(skipped)} units skipped because their network failed training")
    return res
