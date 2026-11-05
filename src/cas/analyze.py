"""The ``analyze`` stage: per-pair fit files -> procedures -> statistics, tables and figures.

Everything here is computed offline from results/<name>/fits and results/<name>/networks; rerun
it as often as you like.  Outputs go to results/<name>/tables, results/<name>/figures and
results/<name>/summary.md.

Only fits in the configured grid are read (layers x k x seeds, shuffle b < shuffle_B), and every
file's fingerprint must match the current config.  Units with missing fits are an error unless
``allow_incomplete`` is set, in which case they are dropped and listed in summary.md.

**Independence.**  A network contributes one analysis unit per (hypothesis, n_fit), so rates and
tests are computed within a cell (task, hypothesis, n_fit) — pooled over widths, where every
network appears exactly once — or within (task, hypothesis, width, n_fit).  Summaries pooled over
hypotheses or n_fit use a cluster bootstrap over networks, never a binomial interval.
"""

from __future__ import annotations

import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import stats as st
from .bootstrap import METHODS, bootstrap_iia, percentile_interval
from .config import Config, derive_seed
from .pairs import load_fingerprint, load_splits
from .pipeline import (StaleOutputError, expected_unit_fingerprints, fit_name,
                       read_unit_fingerprint, shuffle_name, unit_dir)
from .procedures import (ALL_PROCS, CELL, PROC_INFO, THRESHOLD_PROCS, add_null_pvalues,
                         decisions, unit_statistics)

CLASSES = ["positive", "negative_random", "negative_wrong_algorithm", "unknown_standard"]
EVAL_CLASSES = CLASSES[:3]  # standard networks are descriptive only, never in alpha / power
COLORS = {"positive": "#2a78d6", "negative_random": "#eb6834",
          "negative_wrong_algorithm": "#1baf7a", "unknown_standard": "#8a8986"}
X_COLS = ["log2_width", "k_frac", "log_n_fit"]
HCELL = ["task", "hypothesis", "n_fit"]  # headline cell: pooled over widths, 1 row per network
PRIMARY = THRESHOLD_PROCS  # Cochran's Q is restricted to these
ORACLE = {"P3_null_wrong_algorithm", "P4_maxT_wrong_algorithm"}


class IncompleteError(SystemExit):
    pass


# ---- loading ------------------------------------------------------------------------------
def load_units(cfg: Config) -> pd.DataFrame:
    rows = []
    for u in cfg.units():
        meta_path = cfg.out_dir / "networks" / f"{u.net_id}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        layer, k, _ = cfg.fixed_config(u.width)
        rows.append({**u.__dict__, "fixed_layer": layer, "fixed_k": k,
                     "net_passed": meta.get("passed", np.nan),
                     "net_task_acc": meta.get("task_acc", np.nan),
                     "net_neuron_iia_hyp": meta.get(f"neuron_iia_{u.hypothesis}", np.nan)})
    df = pd.DataFrame(rows)
    df["truth_class"] = np.select(
        [df.truth == "positive", df.neg_type == "random", df.neg_type == "wrong_algorithm"],
        ["positive", "negative_random", "negative_wrong_algorithm"], "unknown_standard")
    return df


def load_fits(cfg: Config, units: pd.DataFrame):
    """Fits of the configured grid for the given units.

    Returns (fits, shuffle, incomplete) where incomplete lists units with missing files.
    Raises StaleOutputError if any unit directory, pair file or fit has another fingerprint.
    """
    expected = expected_unit_fingerprints(cfg)
    fit_rows, shuf_rows, incomplete, stale = [], [], [], []
    for u in tqdm(list(units.itertuples()), desc="load fits"):
        fp = expected[u.unit_id]
        d = unit_dir(cfg, u.unit_id)
        if read_unit_fingerprint(cfg, u.unit_id) not in (None, fp):
            stale.append(u.unit_id)
            continue
        if (d / "pairs.npz").exists() and load_fingerprint(d / "pairs.npz") != fp:
            stale.append(u.unit_id)
            continue
        missing = 0
        for layer, k, s in cfg.das_configs(u.width):
            f = d / fit_name(layer, k, s)
            if not f.exists():
                missing += 1
                continue
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            if meta.get("fingerprint") != fp:
                stale.append(f"{u.unit_id}/{f.name}")
                continue
            pred = z["pred_test"]
            fit_rows.append({**meta, **{f"iia_{sp}": z[f"correct_{sp}"].mean()
                                        for sp in ("fit", "select", "test")},
                             "test_pred_constant": bool(pred.min() == pred.max())})
        B = cfg["null"]["shuffle_B"] if u.role == "eval" else 0
        for b in range(B):
            f = d / shuffle_name(b)
            if not f.exists():
                missing += 1
                continue
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            if meta.get("fingerprint") != fp:
                stale.append(f"{u.unit_id}/{f.name}")
                continue
            shuf_rows.append({**meta, "iia_test": z["correct_test"].mean()})
        if missing:
            incomplete.append({"unit_id": u.unit_id, "missing_files": missing})
    if stale:
        raise StaleOutputError(
            f"{len(stale)} fit outputs were made with different settings (e.g. {stale[0]}); "
            "rerun `cas fit --force` or use a new `name`.")
    fits = pd.DataFrame(fit_rows)
    shuf = pd.DataFrame(shuf_rows, columns=["unit_id", "b", "converged", "iia_test", "seconds",
                                            "steps", "stop_reason"])
    return fits, shuf, pd.DataFrame(incomplete, columns=["unit_id", "missing_files"])


def label_agreement(cfg: Config, units: pd.DataFrame) -> pd.Series:
    """Share of test pairs where the hypothesis and the trained algorithm give the same label."""
    out = {}
    for u in units[(units.kind == "iit") & (units.role == "eval")].itertuples():
        p = unit_dir(cfg, u.unit_id) / "pairs.npz"
        if p.exists():
            t = load_splits(p)["test"]
            out[u.unit_id] = float((t.labels[u.hypothesis] == t.labels[u.alg]).mean())
    return pd.Series(out, name="label_agreement", dtype=float)


# ---- tables -------------------------------------------------------------------------------
def rate_row(dec: pd.Series) -> dict:
    d = dec.dropna().astype(bool)
    n, x = len(d), int(d.sum())
    lo, hi = st.wilson(x, n)
    return {"n": n, "certified": x, "rate": x / n if n else np.nan, "wilson_lo": lo,
            "wilson_hi": hi, "upper95_one_sided": st.wilson_upper(x, n)}


def rates_table(dec: pd.DataFrame, by: list[str], alpha0: float) -> pd.DataFrame:
    """Binomial rates per group; each group must contain each network at most once.

    For negatives: exact one-sided binomial p for H0: alpha <= alpha0 and Holm's adjustment
    across the groups that share (scope, procedure, truth_class).
    """
    rows = []
    for key, g in dec.groupby(by, dropna=False):
        g = g.dropna(subset=["certified"])
        if g["net_id"].duplicated().any():
            raise ValueError(f"group {key} has several units per network; rates would treat "
                             "dependent units as independent")
        r = dict(zip(by, key)) | rate_row(g["certified"])
        r["n_networks"] = g["net_id"].nunique()
        r["binom_p_alpha_le_0.05"] = (st.binom_test_greater(r["certified"], r["n"], alpha0)
                                      if str(r.get("truth_class", "")).startswith("negative")
                                      else np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["holm_p"] = np.nan
    fam = [c for c in ("scope", "procedure", "truth_class") if c in out]
    for _, g in out.groupby(fam):
        out.loc[g.index, "holm_p"] = st.holm(g["binom_p_alpha_le_0.05"].to_numpy())
    return out


def pooled_rates(dec: pd.DataFrame, by: list[str], seed: int, B: int = 2000) -> pd.DataFrame:
    """Rates pooled over cells, with a 95% cluster-bootstrap interval over networks."""
    rows = []
    for key, g in dec.groupby(by, dropna=False):
        g = g.dropna(subset=["certified"])
        lo, hi = st.cluster_bootstrap_rate(
            g["net_id"], g["certified"], B, np.random.default_rng(derive_seed(seed, *key)))
        rows.append(dict(zip(by, key)) | {"n_units": len(g), "n_networks": g.net_id.nunique(),
                                          "certified": int(g.certified.sum()),
                                          "rate": g.certified.mean() if len(g) else np.nan,
                                          "cluster_boot_lo": lo, "cluster_boot_hi": hi})
    return pd.DataFrame(rows)


def recertify(dec: pd.DataFrame, c: float) -> pd.DataFrame:
    d = dec.copy()
    thr = d.procedure.isin(THRESHOLD_PROCS)
    d.loc[thr, "certified"] = np.where(d.loc[thr, "statistic"].isna(), np.nan,
                                       d.loc[thr, "statistic"] >= c)
    return d


def threshold_sweep(dec_all: pd.DataFrame, thresholds, seed: int):
    """Per-cell rates (one row per network) and per-task pooled rates (network bootstrap)."""
    cells, pooled = [], []
    for c in thresholds:
        d = recertify(dec_all, c)
        for key, g in d.groupby(HCELL + ["truth_class", "procedure"]):
            cells.append(dict(zip(HCELL + ["truth_class", "procedure"], key), c=c)
                         | rate_row(g["certified"]))
        p = pooled_rates(d, ["task", "truth_class", "procedure"], seed, B=500)
        pooled.append(p.assign(c=c))
    return pd.DataFrame(cells), pd.concat(pooled, ignore_index=True)


def regressions(dec_k: pd.DataFrame, figdir: Path) -> pd.DataFrame:
    """Logit(certified) ~ log2 width + k/width + log n_fit, by task x truth class x procedure.

    Rows are (unit, k); a network contributes several rows, so MLE uses cluster-robust SEs by
    network (Firth, used on separation, has model-based SEs; see the method column).
    """
    rows, panels = [], []
    for (task, cls, proc), g in dec_k.groupby(["task", "truth_class", "procedure"]):
        if cls not in EVAL_CLASSES:
            continue
        g = g.dropna(subset=["certified"])
        if len(g) < 10:
            continue
        g = g.assign(y=g["certified"].astype(int))
        r = st.fit_logistic(g, "y", X_COLS, group="net_id")
        row = {"task": task, "truth_class": cls, "procedure": proc, "n": r["n"],
               "n_networks": g.net_id.nunique(), "events": r["events"], "method": r["method"],
               "converged": r["converged"]}
        for name in ["const"] + X_COLS:
            row[f"b_{name}"] = r["params"].get(name, np.nan)
            row[f"se_{name}"] = r["bse"].get(name, np.nan)
        if 0 < r["events"] < r["n"] and r["covariates"]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = st.fit_mixed_logistic(g, "y", r["covariates"], "net_id")
                for name, v in m["params"].items():
                    row[f"mixed_b_{name}"] = v
                row["mixed_log_sd_net"] = m["log_sd_random_intercept"]
            except Exception as e:  # noqa: BLE001
                row["mixed_error"] = str(e)[:80]
        rows.append(row)
        if proc in THRESHOLD_PROCS:
            panels.append((task, cls, proc, st.binned_residuals(r["fitted"], g["y"])))
    _plot_binned_residuals(panels, figdir)
    return pd.DataFrame(rows)


def paired_tests(dec_all: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per cell (task, hypothesis, n_fit) x truth class, one row per network.

    Primary: pairwise exact McNemar over all procedures available for every unit, Holm-adjusted
    within the cell.  Secondary: Cochran's Q over the threshold procedures with a permutation
    p-value.
    """
    rows = []
    for key, g in dec_all.groupby(HCELL + ["truth_class"]):
        wide = g.pivot_table(index="net_id", columns="procedure", values="certified",
                             aggfunc="first")
        assert not g.groupby(["net_id", "procedure"]).size().gt(1).any()
        wide = wide[[p for p in ALL_PROCS if p in wide and wide[p].notna().all()]].dropna()
        if wide.shape[1] < 2 or len(wide) < 2:
            continue
        prim = wide[[p for p in PRIMARY if p in wide]]
        q, pq = (st.cochran_q_permutation(prim.to_numpy(),
                                          rng=np.random.default_rng(derive_seed(seed, *key)))
                 if prim.shape[1] >= 2 else (np.nan, np.nan))
        pairs = list(combinations(wide.columns, 2))
        res = [st.mcnemar_exact(wide[a].to_numpy(), wide[b].to_numpy()) for a, b in pairs]
        adj = st.holm([r[2] for r in res])
        for (a, b), (n01, n10, p), pa in zip(pairs, res, adj):
            rows.append(dict(zip(HCELL + ["truth_class"], key)) | {
                "n_networks": len(wide), "cochran_q_primary": q, "cochran_perm_p": pq,
                "proc_a": a, "proc_b": b, "rate_a": wide[a].mean(), "rate_b": wide[b].mean(),
                "only_a": n01, "only_b": n10, "mcnemar_p": p, "mcnemar_p_holm": pa})
    return pd.DataFrame(rows)


def bootstrap_table(cfg: Config, ev_all: pd.DataFrame) -> pd.DataFrame:
    """95% percentile intervals for the held-out (test split) fixed-config IIA, 3 schemes.

    Only the test split: on the fit split R was trained on the very pairs being resampled, so
    resampling them does not reflect refitting.
    """
    B = cfg["analysis"]["bootstrap_B"]
    rows = []
    for r in tqdm(list(ev_all.itertuples()), desc="bootstrap"):
        f = unit_dir(cfg, r.unit_id) / fit_name(r.fixed_layer, r.fixed_k, 0)
        if not f.exists():
            continue
        c = np.load(f)["correct_test"]
        sp = load_splits(unit_dir(cfg, r.unit_id) / "pairs.npz")["test"]
        for method in METHODS:
            rng = np.random.default_rng(derive_seed(cfg["seed"], "boot", r.unit_id, method))
            reps = bootstrap_iia(c, sp.pair_base, sp.pair_src, method, B, rng)
            lo, hi = percentile_interval(reps)
            rows.append({"unit_id": r.unit_id, "net_id": r.net_id, "task": r.task,
                         "truth_class": r.truth_class, "width": r.width, "n_fit": r.n_fit,
                         "n_pairs": len(c), "iia": c.mean(), "method": method,
                         "lo": lo, "hi": hi, "width_ci": hi - lo})
    return pd.DataFrame(rows)


# ---- figures ------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8986", "font.size": 9})
    return plt


def _plot_binned_residuals(panels, figdir: Path):
    plt = _plt()
    for task in sorted({p[0] for p in panels}):
        tp = [p for p in panels if p[0] == task]
        classes = [c for c in EVAL_CLASSES if any(p[1] == c for p in tp)]
        procs = [p for p in THRESHOLD_PROCS if any(q[2] == p for q in tp)]
        fig, axes = plt.subplots(len(classes), len(procs), squeeze=False,
                                 figsize=(2.6 * len(procs), 2.2 * len(classes)))
        for i, cls in enumerate(classes):
            for j, proc in enumerate(procs):
                ax = axes[i, j]
                hit = [p[3] for p in tp if p[1] == cls and p[2] == proc]
                if hit:
                    b = hit[0]
                    ax.fill_between(b.mean_fitted, -b.se2, b.se2, color="#e4e3df", lw=0)
                    ax.plot(b.mean_fitted, b.mean_residual, "o", ms=4, color=COLORS[cls])
                    ax.axhline(0, color="#8a8986", lw=0.8)
                ax.set_title(f"{proc}\n{cls}", fontsize=8)
                if i == len(classes) - 1:
                    ax.set_xlabel("mean fitted P(certify)")
                if j == 0:
                    ax.set_ylabel("mean residual")
        fig.suptitle(f"{task}: binned residuals (gray: ±2 SE)", fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / f"binned_residuals_{task}.png", dpi=130)
        plt.close(fig)


def plot_iia_distributions(ev_all: pd.DataFrame, figdir: Path):
    plt = _plt()
    for task, g in ev_all.groupby("task"):
        fig, axes = plt.subplots(1, len(THRESHOLD_PROCS), figsize=(2.7 * len(THRESHOLD_PROCS),
                                                                   2.8), sharey=True)
        classes = [c for c in CLASSES if c in set(g.truth_class)]
        for ax, s in zip(axes, THRESHOLD_PROCS):
            for i, cls in enumerate(classes):
                v = g.loc[g.truth_class == cls, s].dropna().to_numpy()
                jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
                ax.plot(np.full(len(v), i) + jitter, v, "o", ms=3.5, alpha=0.75,
                        color=COLORS[cls], label=cls)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([c.replace("_", "\n", 1) for c in classes], fontsize=7)
            ax.set_title(s, fontsize=9)
        axes[0].set_ylabel("IIA (scope: all configs)")
        fig.suptitle(f"{task}: IIA statistic by truth class (one dot per analysis unit)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / f"iia_by_class_{task}.png", dpi=130)
        plt.close(fig)


def plot_sweep(pooled: pd.DataFrame, figdir: Path):
    plt = _plt()
    for task, g in pooled.groupby("task"):
        procs = [p for p in THRESHOLD_PROCS if p in set(g.procedure)]
        fig, axes = plt.subplots(1, len(procs), figsize=(2.8 * len(procs), 2.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, proc in zip(axes, procs):
            for cls in EVAL_CLASSES:
                h = g[(g.procedure == proc) & (g.truth_class == cls)].sort_values("c")
                if h.empty or h["n_units"].max() == 0:
                    continue
                ax.fill_between(h.c, h.cluster_boot_lo, h.cluster_boot_hi, color=COLORS[cls],
                                alpha=0.15, lw=0)
                ax.plot(h.c, h.rate, "-o", ms=3, lw=2, color=COLORS[cls], label=cls)
            ax.axhline(0.05, color="#8a8986", lw=0.8, ls="--")
            ax.set_title(proc, fontsize=9)
            ax.set_xlabel("threshold c")
        axes[0].set_ylabel("certification rate (α̂ or π̂)")
        if axes[-1].get_legend_handles_labels()[0]:
            axes[-1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{task}: certification rate vs threshold, pooled over cells "
                     "(bands: 95% network-cluster bootstrap; dashed: 0.05)", fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / f"threshold_sweep_{task}.png", dpi=130)
        plt.close(fig)


def plot_bootstrap(boot: pd.DataFrame, figdir: Path):
    if boot.empty:
        return
    plt = _plt()
    piv = boot.pivot_table(index="unit_id", columns="method", values="width_ci")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, other, color in zip(axes, ("base", "twoway"), ("#2a78d6", "#eb6834")):
        ax.plot(piv["iid"], piv[other], "o", ms=4, alpha=0.7, color=color)
        m = np.nanmax(piv.to_numpy()) * 1.05
        ax.plot([0, m], [0, m], color="#8a8986", lw=0.8)
        ax.set_xlabel("iid-pair interval width")
        ax.set_ylabel(f"{other} interval width")
    fig.suptitle("Bootstrap 95% interval widths, held-out fixed-config IIA (one dot per unit)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "bootstrap_widths.png", dpi=130)
    plt.close(fig)


# ---- driver -------------------------------------------------------------------------------
def analyze(cfg: Config, allow_incomplete: bool = False) -> Path:
    t0 = time.perf_counter()
    out = cfg.out_dir
    tables, figdir = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    a = cfg["analysis"]
    c, alpha, seed = a["threshold"], a["alpha"], cfg["seed"]

    units = load_units(cfg)
    ok_units = units[units.net_passed == True]  # noqa: E712
    fits, shuf, incomplete = load_fits(cfg, ok_units)
    if len(incomplete):
        incomplete.to_csv(tables / "incomplete_units.csv", index=False)
        if not allow_incomplete:
            raise IncompleteError(
                f"{len(incomplete)} units have missing fits (see {tables}/incomplete_units.csv)"
                "; run `cas fit` to complete them, or `cas analyze --allow-incomplete` to drop "
                "those units (flagged in summary.md)")
        drop = set(incomplete.unit_id)
        ok_units = ok_units[~ok_units.unit_id.isin(drop)]
        fits = fits[~fits.unit_id.isin(drop)]
        shuf = shuf[~shuf.unit_id.isin(drop)]
    if fits.empty:
        raise SystemExit("no fits found; run `cas fit` first")
    fits = fits.merge(units[["unit_id", "task", "width", "kind", "role", "truth_class"]],
                      on="unit_id")
    fits.to_csv(tables / "fits.csv", index=False)
    shuf.to_csv(tables / "shuffle_fits.csv", index=False)

    nets = units.drop_duplicates("net_id")
    excl_nets = (nets.groupby(["task", "width", "kind", "alg", "role"], dropna=False)
                 .agg(networks=("net_id", "size"),
                      failed=("net_passed", lambda s: int((s == False).sum())))  # noqa: E712
                 .reset_index())
    excl_nets.to_csv(tables / "excluded_networks.csv", index=False)
    excl_fits = (fits.groupby(["task", "truth_class", "role", "layer", "k"])
                 .agg(fits=("converged", "size"),
                      not_converged=("converged", lambda s: int((~s.astype(bool)).sum())))
                 .reset_index())
    excl_fits.to_csv(tables / "excluded_fits.csv", index=False)

    ks_by_width = {w: cfg.ks(w) for w in units.width.unique()}
    ustats = unit_statistics(fits[["unit_id", "layer", "k", "seed_index", "converged",
                                   "iia_fit", "iia_select", "iia_test"]],
                             ok_units, ks_by_width)
    ev, null_sizes = add_null_pvalues(ustats, ok_units, shuf)
    ev = ev.merge(label_agreement(cfg, units), left_on="unit_id", right_index=True, how="left")
    ev["log2_width"] = np.log2(ev.width)
    ev["k_frac"] = ev.scope_k / ev.width
    ev["log_n_fit"] = np.log(ev.n_fit)
    ev.to_csv(tables / "unit_statistics.csv", index=False)
    ustats.merge(units, on="unit_id").query("role == 'cal'").to_csv(
        tables / "calibration_statistics.csv", index=False)
    null_sizes.to_csv(tables / "null_sizes.csv", index=False)

    dec = decisions(ev, c, alpha)
    dec.to_csv(tables / "decisions.csv", index=False)
    dec_main = dec[dec.scope.isin(["all", "site"])]
    dec_all = dec[dec.scope == "all"]
    dec_k = dec[dec.scope.str.startswith("k=")]

    headline = rates_table(dec_main, ["scope"] + HCELL + ["truth_class", "procedure"], alpha)
    headline.to_csv(tables / "rates_by_hypothesis_nfit.csv", index=False)
    cell_rates = rates_table(dec_main, ["scope"] + CELL + ["truth_class", "procedure"], alpha)
    cell_rates.to_csv(tables / "rates_by_cell.csv", index=False)
    pooled = pooled_rates(dec_main, ["scope", "task", "truth_class", "procedure"], seed)
    pooled.to_csv(tables / "rates_pooled.csv", index=False)
    sweep_cells, sweep_pooled = threshold_sweep(dec_all, a["thresholds"], seed)
    sweep_cells.to_csv(tables / "threshold_sweep.csv", index=False)
    sweep_pooled.to_csv(tables / "threshold_sweep_pooled.csv", index=False)
    reg = regressions(dec_k, figdir)
    reg.to_csv(tables / "regression.csv", index=False)
    paired = paired_tests(dec_all, seed)
    paired.to_csv(tables / "paired_tests.csv", index=False)

    ev_all = ev[ev.scope == "all"]
    boot = bootstrap_table(cfg, ev_all)
    boot.to_csv(tables / "bootstrap_intervals.csv", index=False)
    bw = (boot.groupby(["task", "truth_class", "width", "n_fit", "method"])["width_ci"]
          .median().unstack("method").reset_index()) if len(boot) else boot
    if len(boot):
        bw["base/iid"] = bw["base"] / bw["iid"]
        bw["twoway/iid"] = bw["twoway"] / bw["iid"]
    bw.to_csv(tables / "bootstrap_widths.csv", index=False)

    plot_iia_distributions(ev_all, figdir)
    plot_sweep(sweep_pooled, figdir)
    plot_bootstrap(boot, figdir)

    summary = write_summary(cfg, fits, shuf, incomplete, excl_nets, null_sizes, headline,
                            pooled, paired, reg, bw, ev_all, time.perf_counter() - t0)
    print(f"analysis written to {out} (summary: {summary})")
    return summary


def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _cell_text(r) -> str:
    if r is None or r["n"] == 0:
        return ""
    s = f"{r['certified']}/{r['n']}"
    if str(r["truth_class"]).startswith("negative"):
        s += f" (≤{r['upper95_one_sided']:.2f})"
    return s


def headline_table(headline: pd.DataFrame, scope: str, classes) -> str:
    out = []
    h = headline[headline.scope == scope]
    for key, g in h.groupby(HCELL):
        rows = []
        for proc in ALL_PROCS:
            gp = g[g.procedure == proc]
            if gp.empty or gp["n"].sum() == 0:
                continue
            row = {"procedure": proc + (" †" if proc in ORACLE else "")}
            for cls in classes:
                r = gp[gp.truth_class == cls]
                row[cls] = _cell_text(r.iloc[0]) if len(r) else ""
            rows.append(row)
        nets = {cls: int(g[g.truth_class == cls].n_networks.max())
                for cls in classes if (g.truth_class == cls).any()}
        out.append(f"\n**{key[0]}, hypothesis `{key[1]}`, n_fit = {key[2]}** — networks: "
                   + ", ".join(f"{k} {v}" for k, v in nets.items()) + "\n\n")
        out.append(_md(pd.DataFrame(rows)))
    return "".join(out)


def write_summary(cfg, fits, shuf, incomplete, excl_nets, null_sizes, headline, pooled,
                  paired, reg, bw, ev_all, seconds) -> Path:
    a = cfg["analysis"]
    c, alpha = a["threshold"], a["alpha"]
    L = [f"# Results: `{cfg['name']}`\n",
         f"Config: `{cfg.get('config_path', '')}`. Threshold c = {c}, alpha = {alpha}. "
         f"Analysis took {seconds:.0f} s. All numbers below are computed by `cas analyze` from "
         "the per-pair fit files; CSVs with more detail are in `tables/`.\n",
         "Truth classes: `positive` = IIT-trained toward the hypothesis; `negative_random` = "
         "frozen random network (at chance on the task); `negative_wrong_algorithm` = "
         "IIT-trained toward the other algorithm (solves the task); `unknown_standard` = "
         "trained on the task only (truth unknown, descriptive only).\n"]

    if len(incomplete):
        L.append(f"\n> **WARNING: {len(incomplete)} units had missing fits and were DROPPED** "
                 "(`--allow-incomplete`; list in `tables/incomplete_units.csv`). Rates below "
                 "exclude them.\n")
    L.append("\n## Exclusions\n")
    nc = int((~fits.converged.astype(bool)).sum())
    nsc = int((~shuf.converged.astype(bool)).sum()) if len(shuf) else 0
    L.append(f"- Networks failing their training targets (excluded): "
             f"{int(excl_nets.failed.sum())} of {int(excl_nets.networks.sum())}.\n"
             f"- DAS fits not converged (excluded): {nc} of {len(fits)}.\n"
             f"- Shuffled-null fits not converged (excluded): {nsc} of {len(shuf)}.\n")
    small = null_sizes[(null_sizes.B_fixed < 19) | (null_sizes.B_max < 19)]
    if len(small):
        L.append(f"- **Warning**: {len(small)} calibration nulls have B < 19, so "
                 "p = 1/(1+B) > 0.05 and those null-calibrated procedures can never "
                 "certify.\n")

    rnd = fits[(fits.truth_class == "negative_random") & (fits.role == "eval")]
    L.append("\n## Caveats (read before the tables)\n")
    L.append("- † `P3_null_wrong_algorithm` and `P4_maxT_wrong_algorithm` are **oracle nulls**: "
             "they calibrate on networks built exactly like the wrong-algorithm negatives they "
             "are tested on, so their α for those negatives is nominal by construction. Their "
             "rates are a check of the code, not a finding about the method; an analyst "
             "cannot build this null without knowing the true alternative.\n")
    if len(rnd):
        L.append(f"- Random networks: {rnd.test_pred_constant.mean():.0%} of their DAS fits "
                 "predict a single class on every test pair; with label-stratified pairs that "
                 "gives IIA exactly 0.5. α̂ ≈ 0 for random negatives under the threshold "
                 "procedures is therefore close to automatic and says little about DAS.\n")
    L.append("- Rates in each table below are over distinct networks (one unit per network per "
             "cell). Upper bounds in parentheses are one-sided 95% Wilson bounds: with 0 "
             "certified out of n the bound is about 3/n (n = 6: 0.31; n = 30: 0.08).\n")

    L.append("\n## Procedures\n")
    L.append(_md(pd.DataFrame([{"procedure": p, "evaluated on": v[0], "configuration": v[1],
                                "meaning": v[2]} for p, v in PROC_INFO.items()])))

    L.append(f"\n## Headline: certified / networks at c = {c}, scope `all` (all layers)\n")
    L.append("Each table is one cell (task, hypothesis, n_fit) pooled over widths; every "
             "network appears once. Positives: power; negatives: type I error with the "
             "one-sided 95% upper bound.\n")
    L.append(headline_table(headline, "all", CLASSES))
    L.append(f"\n## Type I error at the IIT site only (scope `site`: layer L* = "
             f"{cfg['network']['iit_layer']})\n")
    L.append("Truth is defined at L*. At other layers a wrong-algorithm network can genuinely "
             "implement the other algorithm, so this is the cleaner false-positive rate.\n")
    L.append(headline_table(headline, "site", EVAL_CLASSES))

    L.append("\n## Pooled over hypotheses and n_fit (network-cluster bootstrap 95% CI)\n")
    p = pooled[(pooled.scope == "all")].copy()
    p["procedure"] = p.procedure + np.where(p.procedure.isin(ORACLE), " †", "")
    L.append(_md(p[["task", "truth_class", "procedure", "n_networks", "n_units", "certified",
                    "rate", "cluster_boot_lo", "cluster_boot_hi"]]))

    L.append("\n## Mean IIA statistics by truth class (scope: all)\n")
    m = ev_all.groupby(["task", "truth_class"])[THRESHOLD_PROCS].mean().reset_index()
    L.append(_md(m))
    w = ev_all[ev_all.truth_class == "negative_wrong_algorithm"]
    if len(w):
        L.append(f"\nWrong-algorithm negatives: mean agreement between the hypothesis's and "
                 f"the trained algorithm's counterfactual labels on the test pairs = "
                 f"{w.label_agreement.mean():.3f} (range {w.label_agreement.min():.3f}–"
                 f"{w.label_agreement.max():.3f}).\n")

    sig = headline[(headline.holm_p <= 0.05)]
    L.append(f"\nCells (task, hypothesis, n_fit) where H0: α ≤ 0.05 is rejected after Holm: "
             f"{len(sig)}.\n")
    if len(sig):
        L.append(_md(sig[["scope"] + HCELL + ["truth_class", "procedure", "n", "certified",
                                              "holm_p"]]))
    L.append("\nPer (task, hypothesis, width, n_fit) cell: `tables/rates_by_cell.csv`.\n")

    L.append("\n## Paired comparisons of procedures (scope: all, one row per network)\n")
    L.append("Primary: pairwise exact McNemar tests, Holm within each cell "
             "(`tables/paired_tests.csv`). Secondary: Cochran's Q over the five threshold "
             "procedures, permutation p-value.\n")
    if len(paired):
        q = paired.drop_duplicates(HCELL + ["truth_class"])[
            HCELL + ["truth_class", "n_networks", "cochran_q_primary", "cochran_perm_p"]]
        L.append(_md(q))
        sigp = paired[paired.mcnemar_p_holm <= 0.05]
        L.append(f"\nMcNemar pairs significant after Holm: {len(sigp)} of {len(paired)}.\n")
    else:
        L.append("_(not enough units)_\n")

    L.append("\n## Logistic regression of the decision on design covariates\n")
    L.append("Rows are (unit, k) with scope `k=K`; covariates log2 width, k/width, log n_fit "
             "(constant ones dropped). MLE with cluster-robust SEs by network; Firth with "
             "model-based SEs when separated. Full table: `tables/regression.csv`.\n")
    if len(reg):
        L.append(_md(reg[["task", "truth_class", "procedure", "n", "n_networks", "events",
                          "method"] + [c_ for c_ in reg.columns if c_.startswith("b_")]]))

    L.append("\n## Bootstrap interval widths (held-out fixed-config IIA; median 95% width)\n")
    L.append("`base` (base-input cluster) is the recommended interval; `twoway` is about 2× too "
             "wide in this design (see docs/code/procedures_and_statistics.md).\n")
    L.append(_md(bw))
    L.append("\nFigures: `figures/iia_by_class_*.png`, `figures/threshold_sweep_*.png`, "
             "`figures/binned_residuals_*.png`, `figures/bootstrap_widths.png`.\n")
    path = cfg.out_dir / "summary.md"
    path.write_text("".join(x if x.endswith("\n") else x + "\n" for x in L))
    return path
