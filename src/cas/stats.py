"""Statistics for certification rates (proposal §4, Methods).

* Wilson score intervals for a proportion (two-sided 95%) and the one-sided 95% upper bound.
* Exact one-sided binomial test of H0: alpha <= alpha0 (alternative alpha > alpha0), Holm's
  step-down adjustment.
* Logistic regression of the decision on covariates, maximum likelihood with cluster-robust
  (by network) standard errors, and Firth's penalized likelihood when the data are separated.
* Cochran's Q for k paired binary decisions and pairwise exact McNemar tests.
* Binned residuals (Gelman & Hill 2007 §5.6) for the logistic fits.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

Z95_TWO = sps.norm.ppf(0.975)
Z95_ONE = sps.norm.ppf(0.95)


def wilson(x: int, n: int, z: float = Z95_TWO) -> tuple[float, float]:
    """Wilson score interval (lower, upper); (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def wilson_upper(x: int, n: int) -> float:
    """One-sided 95% upper confidence bound (Wilson, z = 1.645)."""
    return wilson(x, n, Z95_ONE)[1]


def binom_test_greater(x: int, n: int, p0: float) -> float:
    """Exact P(X >= x | n, p0): one-sided test of H0: p <= p0."""
    if n == 0:
        return float("nan")
    return float(sps.binomtest(x, n, p0, alternative="greater").pvalue)


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values; NaNs are left out of the family and stay NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


# ---- logistic regression ------------------------------------------------------------------
def _sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-eta))


def _loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def firth_penalized_loglik(X, y, beta) -> float:
    """l(beta) + 0.5 log det I(beta), I = X' W X (Firth 1993)."""
    p = _sigmoid(X @ beta)
    info = X.T @ ((p * (1 - p))[:, None] * X)
    sign, logdet = np.linalg.slogdet(info)
    return _loglik(X, y, beta) + 0.5 * logdet if sign > 0 else -np.inf


def firth_logit(X, y, max_iter: int = 200, tol: float = 1e-9) -> dict:
    """Firth's bias-reduced logistic regression by Newton steps with step halving.

    Modified score: U*(beta) = X'(y - p + h (1/2 - p)), h = diag of the hat matrix
    W^1/2 X (X'WX)^-1 X' W^1/2.  Standard errors from (X'WX)^-1 at the solution (Wald).
    For an intercept-only model the solution is logit((x + 1/2) / (n + 1)).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    pll = firth_penalized_loglik(X, y, beta)
    converged = False
    for it in range(max_iter):
        p = _sigmoid(X @ beta)
        w = p * (1 - p)
        info = X.T @ (w[:, None] * X)
        inv = np.linalg.inv(info)
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        score = X.T @ (y - p + h * (0.5 - p))
        step = inv @ score
        t = 1.0
        while True:
            cand = beta + t * step
            cand_pll = firth_penalized_loglik(X, y, cand)
            if cand_pll >= pll - 1e-12 or t < 1e-6:
                break
            t /= 2
        beta, pll = cand, cand_pll
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    p = _sigmoid(X @ beta)
    inv = np.linalg.inv(X.T @ ((p * (1 - p))[:, None] * X))
    return {"params": beta, "bse": np.sqrt(np.diag(inv)), "converged": converged,
            "iterations": it + 1, "penalized_loglik": pll, "fitted": p}


def fit_logistic(df: pd.DataFrame, ycol: str, xcols: list[str], group: str) -> dict:
    """Logit of ``ycol`` on ``xcols``; MLE with cluster-robust SEs, Firth on separation.

    Constant covariates are dropped.  "Separation" is declared when the outcome is constant,
    statsmodels warns about (quasi-)perfect separation or fails to converge, or any fitted
    linear predictor exceeds |15|.
    """
    import statsmodels.api as sm

    xcols = [c for c in xcols if df[c].nunique() > 1]
    X = sm.add_constant(df[xcols].to_numpy(dtype=float), has_constant="add")
    names = ["const"] + xcols
    y = df[ycol].to_numpy(dtype=float)
    res = {"n": len(y), "events": int(y.sum()), "covariates": xcols}
    separated = y.min() == y.max()
    if not separated:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                m = sm.Logit(y, X).fit(disp=0, maxiter=200, cov_type="cluster",
                                       cov_kwds={"groups": pd.factorize(df[group])[0]})
                eta = X @ m.params
                bad = (not m.mle_retvals.get("converged", True) or np.max(np.abs(eta)) > 15
                       or any("separation" in str(w.message).lower() for w in caught))
            except Exception:  # noqa: BLE001 - PerfectSeparationError and friends
                bad = True
        separated = bad
    if separated:
        f = firth_logit(X, y)
        res.update(method="firth", params=dict(zip(names, f["params"])),
                   bse=dict(zip(names, f["bse"])), converged=f["converged"],
                   fitted=f["fitted"])
    else:
        res.update(method="mle_cluster_robust", params=dict(zip(names, m.params)),
                   bse=dict(zip(names, m.bse)), converged=True, fitted=m.predict(X),
                   pvalues=dict(zip(names, m.pvalues)))
    return res


def fit_mixed_logistic(df: pd.DataFrame, ycol: str, xcols: list[str], group: str) -> dict:
    """Random-intercept logistic regression (variational Bayes, statsmodels).

    Returns posterior means and SDs of the fixed effects and the log-SD of the random intercept.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    xcols = [c for c in xcols if df[c].nunique() > 1]
    formula = f"{ycol} ~ " + (" + ".join(xcols) if xcols else "1")
    model = BinomialBayesMixedGLM.from_formula(formula, {"net": f"0 + C({group})"}, df)
    r = model.fit_vb()
    names = model.exog_names
    return {"params": dict(zip(names, r.fe_mean)), "sd": dict(zip(names, r.fe_sd)),
            "log_sd_random_intercept": float(r.vcp_mean[0]), "n": len(df)}


def binned_residuals(fitted, y, n_bins: int = 10) -> pd.DataFrame:
    """Bins by fitted probability (quantiles); mean fitted, mean residual, 2-SE band."""
    fitted, y = np.asarray(fitted, float), np.asarray(y, float)
    n_bins = max(1, min(n_bins, len(y) // 5))
    order = np.argsort(fitted, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        pf = fitted[chunk].mean()
        rows.append({"mean_fitted": pf, "mean_residual": (y[chunk] - fitted[chunk]).mean(),
                     "se2": 2 * np.sqrt(max(pf * (1 - pf), 1e-12) / len(chunk)),
                     "n": len(chunk)})
    return pd.DataFrame(rows)


# ---- paired comparisons --------------------------------------------------------------------
def cochran_q(decisions: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for an (n_units, k) 0/1 matrix; returns (Q, p).  NaN if all rows agree."""
    x = np.asarray(decisions, dtype=float)
    n, k = x.shape
    row, col = x.sum(1), x.sum(0)
    denom = k * row.sum() - (row**2).sum()
    if denom == 0:
        return float("nan"), float("nan")
    q = (k - 1) * (k * (col**2).sum() - row.sum() ** 2) / denom
    return float(q), float(sps.chi2.sf(q, k - 1))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired 0/1 vectors: (n a-only, n b-only, two-sided p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    p = min(1.0, 2 * sps.binom.cdf(min(n01, n10), m, 0.5))
    return n01, n10, float(p)
