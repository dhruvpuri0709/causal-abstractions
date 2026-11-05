"""Reporting procedures: from per-fit IIA to a certification decision for each analysis unit.

A *configuration* is (layer, k, rotation seed).  A *scope* is the set of configurations a
procedure may look at:

* ``all``:  every layer x every k x every seed (the main analysis);
* ``k=K``:  every layer x every seed at subspace size K (used for the regression on k/width).

The pre-registered ("fixed") configuration of a scope is (L*, k*, seed 0) where L* is the IIT
layer and k* the IIT subspace size (for scope ``all``) or K (for scope ``k=K``).  Random and
standard networks use the same nominal L*, k* as the IIT networks of the same width.

Procedures (statistic -> decision); names P1..P4 are the proposal's, F_* are the extra cells of
the {in-sample, held-out} x {max, fixed} factorial:

  P1_naive          max over scope of IIA on the fit split (in-sample)   certify if >= c
  P2_heldout        max over scope of IIA on the test split              certify if >= c
  F_insample_fixed  IIA on the fit split at the fixed config             certify if >= c
  F_heldout_fixed   IIA on the test split at the fixed config            certify if >= c
  P4_split          config chosen by max IIA on the select split; its test-split IIA  >= c
  P3_null_<null>    F_heldout_fixed statistic vs. a null sample:  p = (1 + #{null >= T}) / (1 + B)
                    certify if p <= alpha.  <null> is random | wrong_algorithm (calibration
                    networks of that type, same task/hypothesis/width/n_fit) or shuffled (the
                    same network refit on permuted counterfactual labels; fixed config only).
  P4_maxT_<null>    P2_heldout statistic vs. the null distribution of the same max statistic on
                    calibration networks (Westfall & Young 1993 single-step maxT); p as above.

Non-converged fits are dropped before anything else; a max statistic uses the converged configs,
and a fixed-config statistic is missing (NaN) if that one fit did not converge.  Ties in argmax
go to the first configuration in (layer, k, seed) order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD_PROCS = ["P1_naive", "P2_heldout", "F_insample_fixed", "F_heldout_fixed", "P4_split"]
NULL_TYPES = ["random", "wrong_algorithm", "shuffled"]
PVALUE_PROCS = ([f"P3_null_{t}" for t in NULL_TYPES]
                + [f"P4_maxT_{t}" for t in ("random", "wrong_algorithm")])
ALL_PROCS = THRESHOLD_PROCS + PVALUE_PROCS

PROC_INFO = {
    "P1_naive": ("in-sample", "max", "proposal (1) naive"),
    "P2_heldout": ("held-out", "max", "proposal (2) held-out"),
    "F_insample_fixed": ("in-sample", "fixed", "factorial cell: in-sample, pre-registered"),
    "F_heldout_fixed": ("held-out", "fixed", "factorial cell: held-out, pre-registered"),
    "P4_split": ("held-out", "selected on select split", "proposal (4) selection split"),
    "P3_null_random": ("held-out", "fixed", "proposal (3) null: random calibration nets"),
    "P3_null_wrong_algorithm": ("held-out", "fixed",
                                "proposal (3) null: wrong-algorithm calibration nets"),
    "P3_null_shuffled": ("held-out", "fixed", "proposal (3) null: shuffled labels, same net"),
    "P4_maxT_random": ("held-out", "max", "proposal (4) maxT null: random calibration nets"),
    "P4_maxT_wrong_algorithm": ("held-out", "max",
                                "proposal (4) maxT null: wrong-algorithm calibration nets"),
}

CELL = ["task", "hypothesis", "width", "n_fit"]


def scopes_for(ks: list[int]) -> list[tuple[str, int | None]]:
    return [("all", None)] + [(f"k={k}", k) for k in ks]


def unit_statistics(fits: pd.DataFrame, units: pd.DataFrame, ks_by_width: dict) -> pd.DataFrame:
    """One row per (unit, scope) with every procedure's statistic (before any null).

    fits: unit_id, layer, k, seed_index, converged, iia_fit, iia_select, iia_test.
    units: unit_id, width, fixed_layer, fixed_k (+ anything else, carried through).
    """
    fits = fits.sort_values(["unit_id", "layer", "k", "seed_index"])
    conv = fits[fits["converged"]]
    by_unit = dict(tuple(conv.groupby("unit_id")))
    n_all = fits.groupby("unit_id").size()
    rows = []
    for u in units.itertuples(index=False):
        f = by_unit.get(u.unit_id, conv.iloc[:0])
        for scope, k in scopes_for(ks_by_width[u.width]):
            fs = f if k is None else f[f["k"] == k]
            k_fixed = u.fixed_k if k is None else k
            fx = fs[(fs["layer"] == u.fixed_layer) & (fs["k"] == k_fixed)
                    & (fs["seed_index"] == 0)]
            row = {"unit_id": u.unit_id, "scope": scope, "scope_k": k_fixed,
                   "n_configs": len(fs), "n_fits_total": int(n_all.get(u.unit_id, 0))}
            if len(fs):
                i1 = fs["iia_fit"].to_numpy().argmax()
                i2 = fs["iia_test"].to_numpy().argmax()
                i4 = fs["iia_select"].to_numpy().argmax()
                row.update(
                    P1_naive=fs["iia_fit"].iloc[i1], P1_config=_cfg(fs.iloc[i1]),
                    P2_heldout=fs["iia_test"].iloc[i2], P2_config=_cfg(fs.iloc[i2]),
                    P4_split=fs["iia_test"].iloc[i4], P4_config=_cfg(fs.iloc[i4]))
            if len(fx):
                row.update(F_insample_fixed=fx["iia_fit"].iloc[0],
                           F_heldout_fixed=fx["iia_test"].iloc[0])
            rows.append(row)
    out = pd.DataFrame(rows)
    for c in THRESHOLD_PROCS:
        if c not in out:
            out[c] = np.nan
    return out


def _cfg(r) -> str:
    return f"L{int(r['layer'])}_k{int(r['k'])}_s{int(r['seed_index'])}"


def perm_pvalue(stat: float, null: np.ndarray) -> float:
    """(1 + #{null >= stat}) / (1 + B); NaN if stat is NaN or the null is empty."""
    null = np.asarray(null, dtype=float)
    null = null[~np.isnan(null)]
    if np.isnan(stat) or len(null) == 0:
        return float("nan")
    return (1 + np.sum(null >= stat)) / (1 + len(null))


def add_null_pvalues(stats: pd.DataFrame, units: pd.DataFrame,
                     shuffle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add p-values for P3_* and P4_maxT_* to the eval-unit rows.

    Returns (eval rows with p-value columns and B_* null sizes, null-size summary table).
    """
    s = stats.merge(units, on="unit_id")
    cal, ev = s[s["role"] == "cal"], s[s["role"] == "eval"].copy()
    null_sizes = []
    groups = {key: g for key, g in cal.groupby(CELL + ["scope", "neg_type"])}
    shuf = (shuffle[shuffle["converged"]].groupby("unit_id")["iia_test"].apply(np.asarray)
            if len(shuffle) else pd.Series(dtype=object))
    for col in PVALUE_PROCS + [f"B_{p}" for p in PVALUE_PROCS]:
        ev[col] = np.nan
    for i, r in ev.iterrows():
        for nt in ("random", "wrong_algorithm"):
            g = groups.get((r.task, r.hypothesis, r.width, r.n_fit, r.scope, nt))
            if g is None:
                continue
            for proc, statcol in ((f"P3_null_{nt}", "F_heldout_fixed"),
                                  (f"P4_maxT_{nt}", "P2_heldout")):
                null = g[statcol].dropna().to_numpy()
                ev.at[i, proc] = perm_pvalue(r[statcol], null)
                ev.at[i, f"B_{proc}"] = len(null)
        if r.scope_k == r.fixed_k and r.scope in ("all", f"k={r.fixed_k}"):
            null = shuf.get(r.unit_id, np.array([]))
            ev.at[i, "P3_null_shuffled"] = perm_pvalue(r["F_heldout_fixed"], null)
            ev.at[i, "B_P3_null_shuffled"] = len(null)
    for key, g in groups.items():
        null_sizes.append(dict(zip(CELL + ["scope", "neg_type"], key),
                               B_fixed=int(g["F_heldout_fixed"].notna().sum()),
                               B_max=int(g["P2_heldout"].notna().sum())))
    return ev, pd.DataFrame(null_sizes)


def decisions(ev: pd.DataFrame, c: float, alpha: float) -> pd.DataFrame:
    """Long table: one row per (eval unit, scope, procedure) with statistic and decision."""
    keep = [col for col in ev.columns if col not in ALL_PROCS and not col.startswith("B_")]
    rows = []
    for proc in ALL_PROCS:
        d = ev[keep].copy()
        d["procedure"] = proc
        if proc in THRESHOLD_PROCS:
            d["statistic"] = ev[proc]
            d["p_value"] = np.nan
            d["B"] = np.nan
            d["certified"] = np.where(ev[proc].isna(), np.nan, ev[proc] >= c)
        else:
            d["statistic"] = ev["F_heldout_fixed" if proc.startswith("P3") else "P2_heldout"]
            d["p_value"] = ev[proc]
            d["B"] = ev[f"B_{proc}"]
            d["certified"] = np.where(ev[proc].isna(), np.nan, ev[proc] <= alpha)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)
