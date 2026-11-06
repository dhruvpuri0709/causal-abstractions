"""End to end on a small design that converges: train -> fit -> analyze, plus the guards.

One module-scoped run (~12 s) is shared; tests that modify outputs work on a copy.
"""

import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from cas.analyze import IncompleteError, analyze
from cas.config import Config
import json

from cas.pipeline import StaleOutputError, fit_all, train_all, unit_dir

SMOKE = {
    "name": "smoke", "seed": 3,
    "tasks": {"hierarchical_equality": {"widths": [16]}, "distributive_law": {"widths": [16]}},
    "networks": {"eval": {"iit": 1, "random": 1, "standard": 0},
                 "calibration": {"random": 2, "iit": 1}},
    "train": {"iit": {"batch_size": 256, "lr": 0.003, "max_steps": 4000, "eval_every": 100,
                      "eval_n": 1000, "target_acc": 0.98, "target_iit_acc": 0.97,
                      "pass_acc": 0.95, "pass_iit_acc": 0.93}},
    "das": {"layers": [0, 1], "k_fracs": [0.25], "seeds": 1, "n_fit": [256],
            "n_select": 256, "n_test": 256,
            "fit": {"max_steps": 600, "window": 20, "patience": 3, "lr": 0.02}},
    "null": {"shuffle_B": 2}, "analysis": {"bootstrap_B": 50},
}


def _cfg(root, **over):
    d = copy.deepcopy(SMOKE)
    d["results_dir"] = str(root)
    for path, val in over.items():
        node = d
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return Config.from_dict(d)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("res")
    cfg = _cfg(root)
    train_all(cfg, workers=1, device="cpu")
    fit_all(cfg, workers=1, device="cpu")
    analyze(cfg)
    return root


def _copy(smoke, tmp_path):
    shutil.copytree(smoke / "smoke", tmp_path / "smoke")
    return tmp_path


def test_decisions_are_real(smoke):
    dec = pd.read_csv(smoke / "smoke" / "tables" / "decisions.csv")
    main = dec[dec.scope == "all"]
    thr = main[main.procedure.isin(["P1_naive", "P2_heldout", "F_heldout_fixed", "P4_split"])]
    assert thr.certified.notna().all()
    assert (thr[thr.truth_class == "positive"].certified == 1).all()
    assert (thr[thr.truth_class == "negative_random"].certified == 0).all()
    assert (smoke / "smoke" / "summary.md").exists()
    # two layers: max procedures see 2 configs in scope `all`, 1 in scope `site`
    us = pd.read_csv(smoke / "smoke" / "tables" / "unit_statistics.csv")
    assert set(us[us.scope == "all"].n_configs) == {2}
    assert set(us[us.scope == "site"].n_configs) == {1}
    assert (smoke / "smoke" / "tables" / "bootstrap_widths.csv").exists()


def test_resume_writes_nothing_new(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    cfg = _cfg(root)
    n_files = len(list((root / "smoke" / "fits").rglob("*.npz")))
    res = fit_all(cfg, workers=1, device="cpu")
    assert sum(r["fits"] for r in res) == 0
    assert len(list((root / "smoke" / "fits").rglob("*.npz"))) == n_files


def test_changed_das_settings_are_refused_then_forced(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    cfg = _cfg(root, **{"das.n_test": 128})
    with pytest.raises(StaleOutputError):
        fit_all(cfg, workers=1, device="cpu")
    with pytest.raises(StaleOutputError):
        analyze(cfg)
    fit_all(cfg, workers=1, device="cpu", force=True)
    analyze(cfg)
    ev = pd.read_csv(root / "smoke" / "tables" / "bootstrap_intervals.csv")
    assert (ev.n_pairs == 128).all()


def test_changed_training_budget_is_refused_then_forced(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    cfg = _cfg(root, **{"train.iit.max_steps": 3000})
    with pytest.raises(StaleOutputError):
        train_all(cfg, workers=1, device="cpu")
    with pytest.raises(SystemExit):  # fit refuses to run on stale networks
        fit_all(cfg, workers=1, device="cpu")
    metas = train_all(cfg, workers=1, device="cpu", force=True)
    assert all(m["fingerprint"] for m in metas)
    with pytest.raises(StaleOutputError):  # the fits now belong to the old networks
        fit_all(cfg, workers=1, device="cpu")


def test_grid_filter_temp_files_and_incomplete_units(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    cfg = _cfg(root)
    unit = cfg.units()[0]
    d = unit_dir(cfg, unit.unit_id)
    shutil.copy(d / "L1_k4_s0.npz", d / "L2_k99_s7.npz")  # outside the grid: ignored
    (d / "pairs.npz.part").write_bytes(b"garbage")  # leftover temp file: ignored
    analyze(cfg)
    fits = pd.read_csv(root / "smoke" / "tables" / "fits.csv")
    assert not ((fits.layer == 2) & (fits.k == 99)).any()
    (d / "L1_k4_s0.npz").unlink()
    with pytest.raises(IncompleteError):
        analyze(cfg)
    analyze(cfg, allow_incomplete=True)
    assert "DROPPED" in (root / "smoke" / "summary.md").read_text()


def test_same_results_with_one_and_two_workers(tmp_path):
    small = {"tasks": {"hierarchical_equality": {"widths": [16]}},
             "networks.eval": {"iit": 1, "random": 1, "standard": 0},
             "networks.calibration": {"random": 2, "iit": 0}}
    tables = []
    for workers in (1, 2):
        cfg = _cfg(tmp_path / f"w{workers}", **small)
        train_all(cfg, workers=workers, device="cpu")
        fit_all(cfg, workers=workers, device="cpu")
        analyze(cfg)
        t = tmp_path / f"w{workers}" / "smoke" / "tables"
        tables.append({
            "fits": pd.read_csv(t / "fits.csv").drop(columns=["seconds"])
            .sort_values(["unit_id", "layer", "k"]).reset_index(drop=True),
            "shuffle": pd.read_csv(t / "shuffle_fits.csv").drop(columns=["seconds"])
            .sort_values(["unit_id", "b"]).reset_index(drop=True),
            "decisions": pd.read_csv(t / "decisions.csv")
            .sort_values(["unit_id", "scope", "procedure"]).reset_index(drop=True)})
    for name in ("fits", "shuffle", "decisions"):
        pd.testing.assert_frame_equal(tables[0][name], tables[1][name])
    assert np.isfinite(tables[0]["fits"].iia_test).all() and len(tables[0]["shuffle"]) > 0


def test_force_repairs_unit_without_unit_json(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    cfg = _cfg(root)
    unit = next(u for u in cfg.units() if u.kind == "iit" and u.role == "eval")
    (unit_dir(cfg, unit.unit_id) / "unit.json").unlink()
    with pytest.raises(StaleOutputError):  # a directory without unit.json is stale
        fit_all(cfg, workers=1, device="cpu")
    res = fit_all(cfg, workers=1, device="cpu", force=True)
    assert {r["unit_id"]: r["fits"] for r in res}[unit.unit_id] > 0
    analyze(cfg)


def test_truncated_unit_json_counts_as_stale(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    cfg = _cfg(root)
    unit = cfg.units()[0]
    (unit_dir(cfg, unit.unit_id) / "unit.json").write_text('{"unit_id": ')
    with pytest.raises(StaleOutputError):
        fit_all(cfg, workers=1, device="cpu")
    fit_all(cfg, workers=1, device="cpu", force=True)
    analyze(cfg)


def test_tampered_fit_meta_is_refused_by_analyze(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    cfg = _cfg(root)
    unit = cfg.units()[0]
    f = unit_dir(cfg, unit.unit_id) / "L1_k4_s0.npz"
    z = dict(np.load(f))
    meta = json.loads(str(z["meta"]))
    meta["fingerprint"] = "0" * 16
    z["meta"] = np.array(json.dumps(meta))
    with open(f, "wb") as fh:
        np.savez_compressed(fh, **z)
    with pytest.raises(StaleOutputError):
        analyze(cfg)
    res = fit_all(cfg, workers=1, device="cpu")  # the per-file check refits just that file
    assert sum(r["fits"] for r in res) == 1
    analyze(cfg)


def test_changing_pass_threshold_needs_no_retraining(smoke, tmp_path):
    root = _copy(smoke, tmp_path)
    before = {m["net_id"]: m["passed"] for m in train_all(_cfg(root), 1, "cpu")}
    assert all(before.values())
    # also leave a stale directory behind for an IIT unit (M2): it must not block `fit`
    cfg0 = _cfg(root)
    unit = next(u for u in cfg0.units() if u.kind == "iit")
    (unit_dir(cfg0, unit.unit_id) / "unit.json").unlink()
    cfg = _cfg(root, **{"train.iit.pass_iit_acc": 0.9999})
    after = {m["net_id"]: m["passed"] for m in train_all(cfg, 1, "cpu")}  # not refused
    iit = [n for n in after if "_iit-" in n]
    assert iit and not any(after[n] for n in iit)
    assert all(after[n] for n in after if n not in iit)
    res = fit_all(cfg, workers=1, device="cpu")  # not refused, nothing to refit
    assert sum(r["fits"] for r in res) == 0
    analyze(cfg)
    units = pd.read_csv(root / "smoke" / "tables" / "decisions.csv")
    assert "positive" not in set(units.truth_class)  # IIT networks now excluded
