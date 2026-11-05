"""Procedures on a synthetic fits table with known answers."""

import numpy as np
import pandas as pd
import pytest

from cas.procedures import add_null_pvalues, decisions, perm_pvalue, unit_statistics


def test_perm_pvalue():
    assert perm_pvalue(0.9, np.array([0.5, 0.6, 0.95])) == pytest.approx(2 / 4)
    assert perm_pvalue(0.9, np.array([0.9])) == pytest.approx(1.0)  # ties count against
    assert perm_pvalue(1.0, np.full(19, 0.5)) == pytest.approx(0.05)
    assert np.isnan(perm_pvalue(0.9, np.array([])))


def _fits(unit, iia_fit, iia_select, iia_test, conv=None):
    rows = []
    i = 0
    for layer in (0, 1):
        for k in (2, 4):
            rows.append({"unit_id": unit, "layer": layer, "k": k, "seed_index": 0,
                         "converged": True if conv is None else conv[i],
                         "iia_fit": iia_fit[i], "iia_select": iia_select[i],
                         "iia_test": iia_test[i]})
            i += 1
    return rows


def test_unit_statistics_and_decisions():
    # configs in order: (0,2), (0,4), (1,2), (1,4); fixed = (1, 4, 0)
    fits = pd.DataFrame(
        _fits("u", [0.95, 0.7, 0.8, 0.85], [0.6, 0.9, 0.7, 0.8], [0.6, 0.88, 0.7, 0.83])
        + _fits("c1", [0.6] * 4, [0.6] * 4, [0.5, 0.55, 0.6, 0.84])
        + _fits("c2", [0.6] * 4, [0.6] * 4, [0.5, 0.55, 0.6, 0.5],
                conv=[True, True, True, False]))
    units = pd.DataFrame({
        "unit_id": ["u", "c1", "c2"], "width": 8, "fixed_layer": 1, "fixed_k": 4,
        "role": ["eval", "cal", "cal"], "task": "t", "hypothesis": "h", "n_fit": 100,
        "neg_type": [None, "random", "random"], "truth": ["positive", "negative", "negative"]})
    s = unit_statistics(fits, units, {8: [2, 4]})
    u = s[(s.unit_id == "u") & (s.scope == "all")].iloc[0]
    assert u.P1_naive == 0.95 and u.P1_config == "L0_k2_s0"
    assert u.P2_heldout == 0.88 and u.P2_config == "L0_k4_s0"
    assert u.P4_split == 0.88  # selected (0,4) on the select split
    assert u.F_insample_fixed == 0.85 and u.F_heldout_fixed == 0.83
    uk2 = s[(s.unit_id == "u") & (s.scope == "k=2")].iloc[0]
    assert uk2.P1_naive == 0.95 and uk2.F_heldout_fixed == 0.7  # fixed at (1, 2, 0)
    c2 = s[(s.unit_id == "c2") & (s.scope == "all")].iloc[0]
    assert np.isnan(c2.F_heldout_fixed) and c2.n_configs == 3  # non-converged fit dropped

    shuffle = pd.DataFrame({"unit_id": "u", "b": range(3), "converged": True,
                            "iia_test": [0.5, 0.9, 0.4]})
    ev, sizes = add_null_pvalues(s, units, shuffle)
    e = ev[ev.scope == "all"].iloc[0]
    # P3 random: stat 0.83 vs null {c1: 0.84} (c2 fixed missing) -> (1+1)/(1+1) = 1
    assert e.P3_null_random == pytest.approx(1.0) and e.B_P3_null_random == 1
    # P4 maxT random: stat 0.88 vs null of max {0.84, 0.6} -> 1/3
    assert e.P4_maxT_random == pytest.approx(1 / 3)
    # shuffled: 0.83 vs {0.5, 0.9, 0.4} -> 2/4
    assert e.P3_null_shuffled == pytest.approx(0.5)
    assert np.isnan(ev[ev.scope == "k=2"].iloc[0].P3_null_shuffled)  # only at k*
    d = decisions(ev, c=0.86, alpha=0.05)
    d = d[d.scope == "all"].set_index("procedure")
    assert d.loc["P1_naive", "certified"] == 1 and d.loc["F_heldout_fixed", "certified"] == 0
    assert d.loc["P3_null_shuffled", "certified"] == 0
