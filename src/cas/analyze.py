"""The ``analyze`` stage: per-pair fit files -> procedures -> statistics, tables and figures.

Everything here is computed offline from results/<name>/fits and results/<name>/networks; rerun
it as often as you like.  Outputs go to results/<name>/tables, results/<name>/figures and
results/<name>/summary.md.
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
from .pairs import load_splits
from .pipeline import unit_dir
from .procedures import (ALL_PROCS, CELL, PROC_INFO, THRESHOLD_PROCS, add_null_pvalues,
                         decisions, unit_statistics)

CLASSES = ["positive", "negative_random", "negative_wrong_algorithm", "unknown_standard"]
EVAL_CLASSES = CLASSES[:3]  # standard networks are descriptive only, never in alpha / power
COLORS = {"positive": "#2a78d6", "negative_random": "#eb6834",
          "negative_wrong_algorithm": "#1baf7a", "unknown_standard": "#8a8986"}
X_COLS = ["log2_width", "k_frac", "log_n_fit"]


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


def load_fits(cfg: Config, units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows, shuf_rows = [], []
    for uid in tqdm(units["unit_id"], desc="load fits"):
        d = unit_dir(cfg, uid)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npz")):
            if f.name == "pairs.npz":
                continue
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            if f.name.startswith("shuffle"):
                shuf_rows.append({**meta, "iia_test": z["correct_test"].mean()})
            else:
                fit_rows.append({**meta, **{f"iia_{s}": z[f"correct_{s}"].mean()
                                            for s in ("fit", "select", "test")}})
    fits = pd.DataFrame(fit_rows)
    shuf = pd.DataFrame(shuf_rows, columns=["unit_id", "b", "converged", "iia_test", "seconds",
                                            "steps", "stop_reason"])
    return fits, shuf


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
    rows = []
    for key, g in dec.groupby(by, dropna=False):
        r = dict(zip(by, key)) | rate_row(g["certified"])
        r["binom_p_alpha_le_0.05"] = (st.binom_test_greater(r["certified"], r["n"], alpha0)
                                      if str(r.get("truth_class", "")).startswith("negative")
                                      else np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    # Holm across cells, separately for each procedure x negative type
    out["holm_p"] = np.nan
    for _, g in out.groupby(["procedure", "truth_class"]):
        out.loc[g.index, "holm_p"] = st.holm(g["binom_p_alpha_le_0.05"].to_numpy())
    return out


def threshold_sweep(dec_all: pd.DataFrame, thresholds, alpha: float) -> pd.DataFrame:
    rows = []
    for c in thresholds:
        d = dec_all.copy()
        thr = d.procedure.isin(THRESHOLD_PROCS)
        d.loc[thr, "certified"] = np.where(d.loc[thr, "statistic"].isna(), np.nan,
                                           d.loc[thr, "statistic"] >= c)
        for key, g in d.groupby(["task", "truth_class", "procedure"]):
            rows.append(dict(zip(["task", "truth_class", "procedure"], key), c=c)
                        | rate_row(g["certified"]))
    return pd.DataFrame(rows)


def regressions(dec_k: pd.DataFrame, figdir: Path) -> pd.DataFrame:
    """Logit(certified) ~ log2 width + k/width + log n_fit, by task x truth class x procedure."""
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
               "events": r["events"], "method": r["method"], "converged": r["converged"]}
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


def paired_tests(dec_all: pd.DataFrame) -> pd.DataFrame:
    """Cochran's Q over all procedures available for every unit, then pairwise exact McNemar."""
    rows = []
    for (task, cls), g in dec_all.groupby(["task", "truth_class"]):
        wide = g.pivot_table(index="unit_id", columns="procedure", values="certified",
                             aggfunc="first")
        wide = wide[[p for p in ALL_PROCS if p in wide and wide[p].notna().all()]]
        wide = wide.dropna()
        if wide.shape[1] < 2 or len(wide) < 2:
            continue
        q, pq = st.cochran_q(wide.to_numpy())
        pairs = list(combinations(wide.columns, 2))
        res = [st.mcnemar_exact(wide[a].to_numpy(), wide[b].to_numpy()) for a, b in pairs]
        adj = st.holm([r[2] for r in res])
        for (a, b), (n01, n10, p), pa in zip(pairs, res, adj):
            rows.append({"task": task, "truth_class": cls, "n_units": len(wide),
                         "cochran_q": q, "cochran_p": pq, "proc_a": a, "proc_b": b,
                         "rate_a": wide[a].mean(), "rate_b": wide[b].mean(),
                         "only_a": n01, "only_b": n10, "mcnemar_p": p, "mcnemar_p_holm": pa})
    return pd.DataFrame(rows)


def bootstrap_table(cfg: Config, ev_all: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """95% percentile intervals for the fixed-config IIA (fit and test split), 3 schemes."""
    B = cfg["analysis"]["bootstrap_B"]
    rows = []
    for r in tqdm(list(ev_all.itertuples()), desc="bootstrap"):
        f = unit_dir(cfg, r.unit_id) / f"L{r.fixed_layer}_k{r.fixed_k}_s0.npz"
        if not f.exists():
            continue
        z = np.load(f)
        splits = load_splits(unit_dir(cfg, r.unit_id) / "pairs.npz")
        for split in ("fit", "test"):
            c = z[f"correct_{split}"]
            sp = splits[split]
            for method in METHODS:
                rng = np.random.default_rng(derive_seed(cfg["seed"], "boot", r.unit_id,
                                                        split, method))
                reps = bootstrap_iia(c, sp.pair_base, sp.pair_src, method, B, rng)
                lo, hi = percentile_interval(reps)
                rows.append({"unit_id": r.unit_id, "task": r.task, "truth_class": r.truth_class,
                             "width": r.width, "n_fit": r.n_fit, "split": split,
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
    stats_ = ["P1_naive", "P2_heldout", "F_insample_fixed", "F_heldout_fixed", "P4_split"]
    for task, g in ev_all.groupby("task"):
        fig, axes = plt.subplots(1, len(stats_), figsize=(2.7 * len(stats_), 2.8), sharey=True)
        classes = [c for c in CLASSES if c in set(g.truth_class)]
        for ax, s in zip(axes, stats_):
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


def plot_sweep(sweep: pd.DataFrame, figdir: Path):
    plt = _plt()
    for task, g in sweep.groupby("task"):
        procs = [p for p in THRESHOLD_PROCS if p in set(g.procedure)]
        fig, axes = plt.subplots(1, len(procs), figsize=(2.8 * len(procs), 2.8), sharey=True)
        for ax, proc in zip(np.atleast_1d(axes), procs):
            for cls in EVAL_CLASSES:
                h = g[(g.procedure == proc) & (g.truth_class == cls)].sort_values("c")
                if h.empty or h["n"].max() == 0:
                    continue
                ax.fill_between(h.c, h.wilson_lo, h.wilson_hi, color=COLORS[cls], alpha=0.15,
                                lw=0)
                ax.plot(h.c, h.rate, "-o", ms=3, lw=2, color=COLORS[cls], label=cls)
            ax.axhline(0.05, color="#8a8986", lw=0.8, ls="--")
            ax.set_title(proc, fontsize=9)
            ax.set_xlabel("threshold c")
        np.atleast_1d(axes)[0].set_ylabel("certification rate (α̂ or π̂)")
        if np.atleast_1d(axes)[-1].get_legend_handles_labels()[0]:
            np.atleast_1d(axes)[-1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{task}: certification rate vs threshold (bands: Wilson 95%; dashed: 0.05)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / f"threshold_sweep_{task}.png", dpi=130)
        plt.close(fig)


def plot_bootstrap(bw: pd.DataFrame, figdir: Path):
    if bw.empty:
        return
    plt = _plt()
    piv = bw.pivot_table(index=["unit_id", "split"], columns="method", values="width_ci")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, other, color in zip(axes, ("base", "twoway"), ("#2a78d6", "#eb6834")):
        for split, mk in (("fit", "o"), ("test", "s")):
            p = piv.xs(split, level="split")
            ax.plot(p["iid"], p[other], mk, ms=4, alpha=0.7, color=color, label=f"{split} split")
        m = np.nanmax(piv.to_numpy()) * 1.05
        ax.plot([0, m], [0, m], color="#8a8986", lw=0.8)
        ax.set_xlabel("iid-pair interval width")
        ax.set_ylabel(f"{other} interval width")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Bootstrap 95% interval widths for fixed-config IIA (one dot per unit)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "bootstrap_widths.png", dpi=130)
    plt.close(fig)


# ---- driver -------------------------------------------------------------------------------
def analyze(cfg: Config) -> Path:
    t0 = time.perf_counter()
    out = cfg.out_dir
    tables, figdir = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    a = cfg["analysis"]
    c, alpha = a["threshold"], a["alpha"]

    units = load_units(cfg)
    fits, shuf = load_fits(cfg, units)
    if fits.empty:
        raise SystemExit("no fits found; run `cas fit` first")
    fits = fits.merge(units[["unit_id", "task", "width", "kind", "role", "truth_class"]],
                      on="unit_id")
    fits.to_csv(tables / "fits.csv", index=False)
    shuf.to_csv(tables / "shuffle_fits.csv", index=False)

    # exclusions
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

    ok_units = units[units.net_passed == True]  # noqa: E712
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
    dec_all = dec[dec.scope == "all"]
    dec_k = dec[dec.scope != "all"]

    cell_rates = rates_table(dec_all, CELL + ["truth_class", "procedure"], alpha)
    cell_rates.to_csv(tables / "rates_by_cell.csv", index=False)
    pooled = rates_table(dec_all, ["task", "truth_class", "procedure"], alpha)
    pooled.to_csv(tables / "rates_pooled.csv", index=False)
    sweep = threshold_sweep(dec_all, a["thresholds"], alpha)
    sweep.to_csv(tables / "threshold_sweep.csv", index=False)
    reg = regressions(dec_k, figdir)
    reg.to_csv(tables / "regression.csv", index=False)
    paired = paired_tests(dec_all)
    paired.to_csv(tables / "paired_tests.csv", index=False)

    ev_all = ev[ev.scope == "all"]
    boot = bootstrap_table(cfg, ev_all, fits)
    boot.to_csv(tables / "bootstrap_intervals.csv", index=False)
    bw = (boot.groupby(["task", "truth_class", "width", "n_fit", "split", "method"])
          ["width_ci"].median().unstack("method").reset_index()) if len(boot) else boot
    if len(boot):
        bw["base/iid"] = bw["base"] / bw["iid"]
        bw["twoway/iid"] = bw["twoway"] / bw["iid"]
    bw.to_csv(tables / "bootstrap_widths.csv", index=False)

    plot_iia_distributions(ev_all, figdir)
    plot_sweep(sweep, figdir)
    plot_bootstrap(boot, figdir)

    summary = write_summary(cfg, units, fits, shuf, excl_nets, null_sizes, pooled, cell_rates,
                            paired, reg, bw, ev_all, time.perf_counter() - t0)
    print(f"analysis written to {out} (summary: {summary})")
    return summary


def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary(cfg, units, fits, shuf, excl_nets, null_sizes, pooled, cell_rates, paired,
                  reg, bw, ev_all, seconds) -> Path:
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

    L.append("## Exclusions\n")
    nf = len(fits)
    nc = int((~fits.converged.astype(bool)).sum())
    ns = len(shuf)
    nsc = int((~shuf.converged.astype(bool)).sum()) if ns else 0
    failed = int((excl_nets.failed).sum())
    L.append(f"- Networks failing their training targets (excluded): {failed} of "
             f"{int(excl_nets.networks.sum())}.\n- DAS fits not converged (excluded): {nc} of "
             f"{nf}.\n- Shuffled-null fits not converged (excluded): {nsc} of {ns}.\n")
    small = null_sizes[(null_sizes.B_fixed < 19) | (null_sizes.B_max < 19)]
    if len(small):
        L.append(f"- **Warning**: {len(small)} calibration nulls have B < 19, so "
                 f"p = 1/(1+B) > 0.05 and those null-calibrated procedures can never "
                 "certify.\n")
    L.append("\n## Procedures\n")
    L.append(_md(pd.DataFrame([{"procedure": p, "evaluated on": v[0], "configuration": v[1],
                                "meaning": v[2]} for p, v in PROC_INFO.items()])))

    L.append(f"\n## Headline: certification rates at c = {c} (scope: all configurations)\n")
    L.append("Pooled over the design cells of each task. For negatives the rate is the type I "
             "error α̂ (report the one-sided 95% upper bound, not just the point estimate); for "
             "positives it is the power π̂. `binom_p` tests H0: α ≤ 0.05 (exact, one-sided).\n")
    for task, g in pooled.groupby("task"):
        L.append(f"\n### {task}\n")
        for cls in CLASSES:
            h = g[g.truth_class == cls]
            if h.empty:
                continue
            h = h.set_index("procedure").reindex([p for p in ALL_PROCS if p in set(h.procedure)])
            h = h.reset_index()[["procedure", "n", "certified", "rate", "wilson_lo", "wilson_hi",
                                 "upper95_one_sided", "binom_p_alpha_le_0.05"]]
            h = h[h.n > 0]
            title = {"positive": "power π̂ (positives)",
                     "negative_random": "type I error α̂ (random negatives)",
                     "negative_wrong_algorithm": "type I error α̂ (wrong-algorithm negatives)",
                     "unknown_standard": "certification rate, standard networks "
                                         "(descriptive; truth unknown)"}[cls]
            L.append(f"\n**{title}**\n\n")
            L.append(_md(h.rename(columns={"binom_p_alpha_le_0.05": "binom_p"})))

    L.append("\n## Mean IIA statistics by truth class (scope: all)\n")
    cols = ["P1_naive", "P2_heldout", "F_insample_fixed", "F_heldout_fixed", "P4_split"]
    m = ev_all.groupby(["task", "truth_class"])[cols].mean().reset_index()
    L.append(_md(m))
    if "label_agreement" in ev_all:
        w = ev_all[ev_all.truth_class == "negative_wrong_algorithm"]
        if len(w):
            L.append(f"\nWrong-algorithm negatives: mean agreement between the hypothesis's and "
                     f"the trained algorithm's counterfactual labels on the test pairs = "
                     f"{w.label_agreement.mean():.3f} (range {w.label_agreement.min():.3f}–"
                     f"{w.label_agreement.max():.3f}).\n")

    L.append("\n## Per-cell rates\n\nSee `tables/rates_by_cell.csv` (Wilson intervals, one-sided "
             "upper bounds, exact binomial p and Holm-adjusted p across cells within each "
             "procedure and negative type).\n")
    sig = cell_rates[(cell_rates.holm_p <= 0.05)]
    L.append(f"Cells where H0: α ≤ 0.05 is rejected after Holm: {len(sig)}.\n")
    if len(sig):
        L.append(_md(sig[CELL + ["truth_class", "procedure", "n", "certified", "rate",
                                 "holm_p"]]))

    L.append("\n## Paired comparisons of procedures (scope: all)\n")
    if len(paired):
        q = paired.drop_duplicates(["task", "truth_class"])[
            ["task", "truth_class", "n_units", "cochran_q", "cochran_p"]]
        L.append(_md(q))
        L.append("\nPairwise exact McNemar tests with Holm adjustment are in "
                 "`tables/paired_tests.csv`.\n")
    else:
        L.append("_(not enough units)_\n")

    L.append("\n## Logistic regression of the decision on design covariates\n")
    L.append("Rows are (unit, k) with scope `k=K`; covariates log2 width, k/width, log n_fit "
             "(constant ones dropped). MLE with cluster-robust SEs by network, Firth when "
             "separated. Full table: `tables/regression.csv`.\n")
    if len(reg):
        L.append(_md(reg[["task", "truth_class", "procedure", "n", "events", "method"]
                         + [c_ for c_ in reg.columns if c_.startswith("b_")]]))

    L.append("\n## Bootstrap interval widths (fixed config; median 95% width)\n")
    L.append(_md(bw))
    L.append("\nFigures: `figures/iia_by_class_*.png`, `figures/threshold_sweep_*.png`, "
             "`figures/binned_residuals_*.png`, `figures/bootstrap_widths.png`.\n")
    path = cfg.out_dir / "summary.md"
    path.write_text("".join(x if x.endswith("\n") else x + "\n" for x in L))
    return path
