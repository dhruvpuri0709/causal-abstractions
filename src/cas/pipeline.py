"""The ``train`` and ``fit`` stages.  Both are resumable: finished outputs are skipped.

Layout under results/<name>/:

    networks/<net_id>.pt|.json        weights + metadata (accuracies, pass/fail, fingerprint)
    fits/<unit_id>/unit.json          the unit's fingerprint
    fits/<unit_id>/pairs.npz          the unit's fit/select/test pools and pairs (ids, labels)
    fits/<unit_id>/L{l}_k{k}_s{s}.npz one DAS fit: per-pair predictions and correctness per split
    fits/<unit_id>/shuffle_b{b}.npz   one shuffled-label null fit at the pre-registered config

**Fingerprints.**  Every network, unit directory, pair file and fit records a hash of the config
settings that produced it (``Config.network_fingerprint`` / ``Config.unit_fingerprint``; a unit's
fingerprint includes its network's).  Reusing an output whose fingerprint differs from the
current config is refused with an error; ``--force`` rebuilds the mismatched outputs, or use a
new ``name``.  A unit directory without a readable ``unit.json`` is stale; inside a unit, the
pair file and each fit are checked against their own fingerprint.  Pass thresholds are not
fingerprinted: ``passed`` is recomputed from stored accuracies (``network_passed``).  Changing
only the grid (layers, k_fracs, seeds, shuffle_B) keeps fingerprints: the missing fits are added.
Partially written files end in ``.part`` and are never read.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import Config, NetworkSpec, UnitSpec, derive_seed, iit_k
from .das import FitSettings, fit_rotation, pair_activations, predict
from .models import load_network, save_network
from .pairs import SPLITS, load_fingerprint, load_splits, make_splits, save_splits
from .parallel import run_jobs
from .tasks import TASKS
from .train import iit_accuracy, new_network, task_accuracy, train_iit, train_standard

FINAL_EVAL_N = 10000


class StaleOutputError(SystemExit):
    pass


def read_json(path: Path) -> dict | None:
    """Parsed JSON object, or None if the file is missing, truncated or not valid JSON."""
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def network_passed(cfg: Config, meta: dict) -> bool:
    """Pass/fail from the stored accuracies and the *current* pass thresholds."""
    if meta["kind"] == "random":
        return True
    s = cfg["train"][meta["kind"]]
    if meta["kind"] == "standard":
        return bool(meta["task_acc"] >= s["pass_acc"])
    return bool(meta["task_acc"] >= s["pass_acc"]
                and meta[f"neuron_iia_{meta['alg']}"] >= s["pass_iit_acc"])


def load_network_meta(cfg: Config, net_id: str) -> dict | None:
    """Stored metadata with ``passed`` recomputed for the current config (None if unreadable)."""
    meta = read_json(network_path(cfg, net_id).with_suffix(".json"))
    if meta is not None:
        meta["passed"] = network_passed(cfg, meta)
    return meta


def network_path(cfg: Config, net_id: str) -> Path:
    return cfg.out_dir / "networks" / net_id


def unit_dir(cfg: Config, unit_id: str) -> Path:
    return cfg.out_dir / "fits" / unit_id


def fit_name(layer: int, k: int, s: int) -> str:
    return f"L{layer}_k{k}_s{s}.npz"


def shuffle_name(b: int) -> str:
    return f"shuffle_b{b:03d}.npz"


def _stale(what: str, items: list[str], cfg: Config):
    ex = ", ".join(items[:3]) + (" ..." if len(items) > 3 else "")
    raise StaleOutputError(
        f"{len(items)} existing {what} under {cfg.out_dir} were made with different settings "
        f"(fingerprint mismatch), e.g. {ex}.\nUse a new `name` in the config, or rerun with "
        f"--force to rebuild exactly those outputs.")


# ---- train -------------------------------------------------------------------------------
def build_network(job) -> dict:
    cfg, spec, device = Config(job["cfg"]), NetworkSpec(**job["spec"]), job["device"]
    path = network_path(cfg, spec.net_id)
    fp = cfg.network_fingerprint(spec)
    meta = load_network_meta(cfg, spec.net_id)
    if meta is not None and meta.get("fingerprint") == fp:
        return meta
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
    meta = asdict(spec) | info | {"iit_layer": layer, "iit_k": k, "fingerprint": fp}
    meta["task_acc"] = task_accuracy(model, task, FINAL_EVAL_N, eval_rng, device)
    for alg in task.algorithms:  # neuron-aligned IIA at the IIT site, for every algorithm
        meta[f"neuron_iia_{alg}"] = iit_accuracy(model, task, alg, layer, k, FINAL_EVAL_N,
                                                 eval_rng, device)
    meta["passed"] = network_passed(cfg, meta)  # recomputed on every load
    save_network(model.cpu(), meta, path)
    return meta


def network_status(cfg: Config) -> dict[str, str]:
    """net_id -> 'ok' | 'missing' | 'stale' for the current config."""
    out = {}
    for spec in cfg.networks():
        p = network_path(cfg, spec.net_id).with_suffix(".json")
        if not p.exists():
            out[spec.net_id] = "missing"
        else:  # unreadable or truncated JSON counts as stale, so --force can rebuild it
            meta = read_json(p)
            fp = meta.get("fingerprint") if meta else None
            out[spec.net_id] = "ok" if fp == cfg.network_fingerprint(spec) else "stale"
    return out


def train_all(cfg: Config, workers: int, device: str, force: bool = False) -> list[dict]:
    status = network_status(cfg)
    stale = [n for n, s in status.items() if s == "stale"]
    if stale and not force:
        _stale("networks", stale, cfg)
    if stale:
        print(f"--force: rebuilding {len(stale)} networks with stale fingerprints")
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
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def expected_unit_fingerprints(cfg: Config) -> dict[str, str]:
    """unit_id -> fingerprint, using the current config's network fingerprints."""
    nets = {s.net_id: s for s in cfg.networks()}
    return {u.unit_id: cfg.unit_fingerprint(u, cfg.network_fingerprint(nets[u.net_id]))
            for u in cfg.units()}


def read_unit_fingerprint(cfg: Config, unit_id: str) -> str | None:
    meta = read_json(unit_dir(cfg, unit_id) / "unit.json")
    return meta.get("fingerprint") if meta else None


def unit_state(cfg: Config, unit_id: str, fp: str) -> str:
    """'absent' (no directory or an empty one), 'ok', or 'stale' (a directory whose unit.json
    is missing, unreadable or has another fingerprint)."""
    d = unit_dir(cfg, unit_id)
    if not d.exists() or not any(d.iterdir()):
        return "absent"
    return "ok" if read_unit_fingerprint(cfg, unit_id) == fp else "stale"


def _file_fingerprint(path: Path) -> str | None:
    """Fingerprint stored in a fit file's meta, or None if missing/unreadable."""
    try:
        with np.load(path) as z:
            return json.loads(str(z["meta"])).get("fingerprint")
    except Exception:  # noqa: BLE001 - truncated or foreign file: treat as stale
        return None


def get_pairs(cfg: Config, unit: UnitSpec, fp: str):
    path = unit_dir(cfg, unit.unit_id) / "pairs.npz"
    if path.exists():
        try:
            if load_fingerprint(path) == fp:
                return load_splits(path)
        except Exception:  # noqa: BLE001 - unreadable: regenerate
            pass
    d = cfg["das"]
    sizes = {"fit": unit.n_fit, "select": d["n_select"], "test": d["n_test"]}
    splits = make_splits(TASKS[unit.task], unit.hypothesis, sizes,
                         np.random.default_rng(unit.pair_seed), d["base_uses"], d["source_uses"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("pairs.npz.part")
    with open(tmp, "wb") as f:
        save_splits(splits, f, fingerprint=fp)
    os.replace(tmp, path)
    return splits


def fit_unit(job) -> dict:
    cfg, unit, device = Config(job["cfg"]), UnitSpec(**job["unit"]), job["device"]
    fp = job["fingerprint"]
    if not load_network_meta(cfg, unit.net_id)["passed"]:
        return {"unit_id": unit.unit_id, "skipped": "network failed training targets",
                "fits": 0}
    d = unit_dir(cfg, unit.unit_id)
    if unit_state(cfg, unit.unit_id, fp) == "stale":
        shutil.rmtree(d)  # only reached with --force (fit_all refuses stale units otherwise)
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "unit.json", {"unit_id": unit.unit_id, "fingerprint": fp})
    configs = cfg.das_configs(unit.width)
    fixed = cfg.fixed_config(unit.width)
    shuffle_B = cfg["null"]["shuffle_B"] if unit.role == "eval" else 0
    # decide per file by its stored fingerprint, not by existence
    todo = [c for c in configs if _file_fingerprint(d / fit_name(*c)) != fp]
    todo_shuffle = [b for b in range(shuffle_B) if _file_fingerprint(d / shuffle_name(b)) != fp]
    if not todo and not todo_shuffle:
        return {"unit_id": unit.unit_id, "fits": 0}

    splits = get_pairs(cfg, unit, fp)
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
                "steps": res.steps, "seconds": res.seconds, "fingerprint": fp}
        _save_npz(d / fit_name(layer, k, s), losses=np.asarray(res.losses),
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
                "steps": res.steps, "seconds": res.seconds, "fingerprint": fp}
        _save_npz(d / shuffle_name(b), pred_test=pred, labels_test=lab_test,
                  labels_fit=lab_fit, correct_test=pred == lab_test,
                  losses=np.asarray(res.losses), meta=np.array(json.dumps(meta)))
        n_done += 1
    return {"unit_id": unit.unit_id, "fits": n_done}


def fit_all(cfg: Config, workers: int, device: str, force: bool = False) -> list[dict]:
    status = network_status(cfg)
    bad = [n for n, s in status.items() if s != "ok"]
    if bad:
        raise SystemExit(f"{len(bad)} networks are missing or were trained with different "
                         f"settings (e.g. {bad[0]}: {status[bad[0]]}); run `cas train` first")
    units = cfg.units()
    expected = expected_unit_fingerprints(cfg)
    passed = {n: load_network_meta(cfg, n)["passed"] for n in status}
    # units of networks that fail are never fitted or analyzed, so their directories are left
    # alone (and do not block a plain `cas fit`)
    stale = [u.unit_id for u in units if passed[u.net_id]
             and unit_state(cfg, u.unit_id, expected[u.unit_id]) == "stale"]
    if stale and not force:
        _stale("fit directories", stale, cfg)
    if stale:
        print(f"--force: refitting {len(stale)} units with stale fingerprints")
    jobs = [{"cfg": dict(cfg), "unit": asdict(u), "device": device,
             "fingerprint": expected[u.unit_id]} for u in units]
    jobs.sort(key=lambda j: -j["unit"]["width"])  # widest (slowest) first
    res = run_jobs(fit_unit, jobs, workers, "DAS fits (units)")
    n = sum(r["fits"] for r in res)
    skipped = [r["unit_id"] for r in res if "skipped" in r]
    print(f"{len(units)} analysis units; {n} new fits written under {cfg.out_dir / 'fits'}; "
          f"{len(skipped)} units skipped because their network failed training")
    return res
