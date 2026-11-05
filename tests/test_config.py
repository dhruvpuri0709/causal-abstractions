"""Design enumeration: truth assignment, disjoint calibration networks, stable seeds."""

from collections import Counter

from cas.config import Config, derive_seed


def _cfg():
    return Config.from_dict({
        "name": "t", "tasks": {"hierarchical_equality": {"widths": [16]},
                               "distributive_law": {"widths": [24]}},
        "networks": {"eval": {"iit": 2, "random": 2, "standard": 1},
                     "calibration": {"random": 3, "iit": 3}},
        "das": {"n_fit": [100, 200]}})


def test_truth_assignment():
    units = _cfg().units()
    c = Counter((u.task, u.kind, u.role, u.truth, u.neg_type) for u in units)
    # HE: 2 IIT positives x 2 n_fit; no wrong-algorithm networks for HE
    assert c[("hierarchical_equality", "iit", "eval", "positive", None)] == 4
    assert not any(u.neg_type == "wrong_algorithm" for u in units
                   if u.task == "hierarchical_equality")
    # DL: 2 per algorithm, each tested against both hypotheses
    assert c[("distributive_law", "iit", "eval", "positive", None)] == 2 * 2 * 2
    assert c[("distributive_law", "iit", "eval", "negative", "wrong_algorithm")] == 2 * 2 * 2
    # calibration IIT networks only ever serve as wrong-algorithm nulls
    assert c[("distributive_law", "iit", "cal", "negative", "wrong_algorithm")] == 3 * 2 * 2
    assert not any(u.role == "cal" and u.truth != "negative" for u in units)
    for u in units:
        if u.kind == "iit" and u.truth == "negative":
            assert u.alg != u.hypothesis


def test_calibration_networks_are_disjoint_from_eval():
    nets = _cfg().networks()
    ids = [n.net_id for n in nets]
    seeds = [n.seed for n in nets]
    assert len(set(ids)) == len(ids) and len(set(seeds)) == len(seeds)


def test_fixed_config_in_grid():
    cfg = _cfg()
    for w in (16, 24, 48):
        layer, k, s = cfg.fixed_config(w)
        assert (layer, k, s) in cfg.das_configs(w)


def test_derive_seed_stable():
    assert derive_seed(1, "a", 2) == derive_seed(1, "a", 2) != derive_seed(1, "a", 3)
    assert 0 <= derive_seed("x") < 2**31
