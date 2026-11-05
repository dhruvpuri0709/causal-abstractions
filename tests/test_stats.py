"""Statistics: Wilson, binomial test, Holm, Firth, Cochran's Q, McNemar."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from statsmodels.stats.contingency_tables import cochrans_q as sm_cochrans_q
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from cas import stats as st


@pytest.mark.parametrize("x,n", [(0, 10), (3, 30), (30, 30), (7, 200)])
def test_wilson_matches_statsmodels(x, n):
    lo, hi = st.wilson(x, n)
    slo, shi = proportion_confint(x, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(slo, abs=1e-10) and hi == pytest.approx(shi, abs=1e-10)
    # one-sided 95% upper bound = upper end of the two-sided 90% interval
    assert st.wilson_upper(x, n) == pytest.approx(
        proportion_confint(x, n, alpha=0.10, method="wilson")[1], abs=1e-10)


def test_binom_test_greater():
    # P(X >= 1 | n=10, p=0.05) = 1 - 0.95^10
    assert st.binom_test_greater(1, 10, 0.05) == pytest.approx(1 - 0.95**10)
    assert st.binom_test_greater(0, 10, 0.05) == pytest.approx(1.0)


def test_holm_matches_statsmodels_and_skips_nan():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.2])
    ok = ~np.isnan(p)
    expect = multipletests(p[ok], method="holm")[1]
    got = st.holm(p)
    assert np.isnan(got[3])
    assert np.allclose(got[ok], expect)


def test_firth_intercept_only_closed_form():
    # intercept-only Firth estimate is logit((x + 1/2) / (n + 1))
    for x, n in [(0, 20), (3, 20), (20, 20)]:
        y = np.r_[np.ones(x), np.zeros(n - x)]
        f = st.firth_logit(np.ones((n, 1)), y)
        assert f["converged"]
        assert f["params"][0] == pytest.approx(logit((x + 0.5) / (n + 1)), abs=1e-6)


def test_firth_score_is_zero_and_finite_under_separation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = (x > 0).astype(float)  # perfectly separated
    X = np.c_[np.ones(40), x]
    f = st.firth_logit(X, y)
    assert f["converged"] and np.all(np.isfinite(f["params"]))
    # numerical gradient of the penalized log-likelihood vanishes at the solution
    eps = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        g = (st.firth_penalized_loglik(X, y, f["params"] + e)
             - st.firth_penalized_loglik(X, y, f["params"] - e)) / (2 * eps)
        assert abs(g) < 1e-4


def test_firth_close_to_mle_with_lots_of_data():
    import statsmodels.api as sm
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < expit(-0.5 + 1.2 * x)).astype(float)
    X = np.c_[np.ones(5000), x]
    mle = sm.Logit(y, X).fit(disp=0).params
    assert np.allclose(st.firth_logit(X, y)["params"], mle, atol=0.01)


def test_fit_logistic_switches_to_firth_on_separation():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=60), "g": np.repeat(np.arange(20), 3)})
    df["y"] = (df.x > 0).astype(int)
    assert st.fit_logistic(df, "y", ["x"], "g")["method"] == "firth"
    df["y"] = (rng.random(60) < expit(df.x)).astype(int)
    assert st.fit_logistic(df, "y", ["x"], "g")["method"] == "mle_cluster_robust"


def test_cochran_q_and_mcnemar_match_statsmodels():
    rng = np.random.default_rng(3)
    x = (rng.random((50, 3)) < [0.2, 0.4, 0.5]).astype(int)
    q, p = st.cochran_q(x)
    ref = sm_cochrans_q(x)
    assert q == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
    a, b = x[:, 0], x[:, 2]
    table = [[np.sum((a == 1) & (b == 1)), np.sum((a == 1) & (b == 0))],
             [np.sum((a == 0) & (b == 1)), np.sum((a == 0) & (b == 0))]]
    assert st.mcnemar_exact(a, b)[2] == pytest.approx(sm_mcnemar(table, exact=True).pvalue)


def test_binned_residuals_shape():
    b = st.binned_residuals(np.linspace(0.1, 0.9, 100), np.r_[np.zeros(50), np.ones(50)])
    assert len(b) == 10 and b["n"].sum() == 100
