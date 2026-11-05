# Procedures and statistics

Section references are to `docs/proposal.tex` (Methods). All of this is computed by
`cas analyze` from the per-pair fit files (`src/cas/procedures.py`, `stats.py`, `bootstrap.py`,
`analyze.py`).

## Splits and pairs (`pairs.py`)

Each analysis unit has three splits, **fit** (n_fit pairs; trains the rotation), **select** and
**test**. Each split has its own pool of base inputs and its own pool of source inputs, drawn
fresh, so no input is shared between splits. Each base input is used in at most `base_uses` (2)
pairs and each source in at most `source_uses` (2). Pairs intervene on V1, V2 or both with
probability 1/3 each, and are stratified so that each of the four cells (counterfactual label ∈
{0,1}) × (label changed relative to the base: yes/no) holds a quarter of the pairs. Chance IIA is
therefore 0.5 both for a constant guess and for "ignore the intervention".

## Configurations and scopes

A configuration is (layer l, subspace size k, rotation seed s) with l ∈ `das.layers`,
k ∈ round(`k_fracs` × width) ∪ {k*}, s < `das.seeds`. The pre-registered configuration is
(L*, k*, seed 0): the IIT site and size; random and standard networks of the same width use the
same nominal values. A *scope* is the set of configurations a procedure may use:
`all` (main analysis), or `k=K` (all layers and seeds at size K; fixed config (L*, K, 0)), used
for the regression on k/width.

## Procedures (proposal §4 (1)–(4) plus factorial cells)

With IIA_split(c) the IIA of configuration c on a split, C the scope's converged configurations,
c₀ the fixed configuration and threshold c (default 0.9):

| Name | Statistic T | Certify if | Proposal |
|---|---|---|---|
| `P1_naive` | max_{c∈C} IIA_fit(c) | T ≥ c | (1) naive: in-sample, best over seeds × layers × sizes |
| `P2_heldout` | max_{c∈C} IIA_test(c) | T ≥ c | (2) held-out: rotation fit on fit split, IIA on disjoint inputs |
| `F_insample_fixed` | IIA_fit(c₀) | T ≥ c | factorial cell: in-sample × pre-registered |
| `F_heldout_fixed` | IIA_test(c₀) | T ≥ c | factorial cell: held-out × pre-registered |
| `P4_split` | IIA_test(ĉ), ĉ = argmax_{c∈C} IIA_select(c) | T ≥ c | (4) selection on one split, one test on a third |
| `P3_null_<n>` | IIA_test(c₀) | p̂ ≤ α | (3) null-calibrated |
| `P4_maxT_<n>` | max_{c∈C} IIA_test(c) | p̂ ≤ α | (4) alternative: null of the max (Westfall–Young) |

The 2 × 2 factorial {in-sample, held-out} × {max over configs, fixed} is P1, P2,
F_insample_fixed, F_heldout_fixed, so the effect of held-out evaluation and the effect of
selection can be separated. P3 uses the F_heldout_fixed statistic with a calibrated critical
value instead of c; P4_maxT uses the P2 statistic with a calibrated critical value.

**Null-calibrated p-value.** p̂ = (1 + #{b : T₀_b ≥ T}) / (1 + B), where T₀_1..T₀_B are the
same statistic computed by the same pipeline on null replicates. Ties count against
certification. Because min p̂ = 1/(1+B), **B ≥ 19 is required to ever certify at α = 0.05**;
`cas analyze` warns in `summary.md` when any null has B < 19 (`tables/null_sizes.csv`). Nulls,
reported separately (`<n>`):

* `random`: calibration random networks of the same task, hypothesis, width and n_fit;
* `wrong_algorithm`: calibration networks IIT-trained to the *other* algorithm (distributive
  law only), same cell;
* `shuffled` (P3 only): the evaluation network itself, refit at c₀ on the fit split with the
  counterfactual labels randomly permuted, and evaluated on the test split against permuted
  test labels; B = `null.shuffle_B` refits per unit.

Calibration networks are different networks (own ids and training seeds) from the evaluation
networks, so no test is evaluated on data that calibrated it. Under exchangeability of a
negative with its null type, P3/P4_maxT control α at the nominal level for that negative type;
they are not guaranteed to for the other type.

**Exclusions.** Non-converged fits (see [das.md](das.md)) are dropped before anything else. A
max statistic uses the remaining configurations of the scope; a fixed-config statistic is
missing if that fit did not converge, and the unit then has no decision for that procedure.
Networks failing their training targets are excluded entirely. Counts:
`tables/excluded_networks.csv`, `tables/excluded_fits.csv`, and the top of `summary.md`.
Ties in argmax go to the first configuration in (layer, k, seed) order.

## Rates per design cell

A design cell is (task, hypothesis, width, n_fit). For each cell, truth class and procedure
(`tables/rates_by_cell.csv`; pooled over cells per task in `rates_pooled.csv`):

* α̂ (negatives) or π̂ (positives) = certified / n, with the 95% Wilson score interval
  p̃ ± z√(p̂(1−p̂)/n + z²/4n²)/(1 + z²/n), p̃ = (p̂ + z²/2n)/(1 + z²/n), z = 1.96;
* the one-sided 95% upper Wilson bound (z = 1.645): report this for α. Failing to reject
  H0: α ≤ 0.05 is not evidence that α is controlled;
* for negatives, the exact one-sided binomial test of H0: α ≤ 0.05, p = P(X ≥ x | n, 0.05), and
  Holm's step-down adjustment across cells, separately for each procedure × negative type;
* `threshold_sweep.csv` and `figures/threshold_sweep_*.png`: rates for c ∈
  `analysis.thresholds` (p-value procedures don't depend on c).

Standard networks (`unknown_standard`) appear in these tables only as descriptive certification
rates; they never enter α̂, π̂, tests or regressions.

## Across cells: logistic regression (`regression.csv`)

Rows are (unit, k) decisions in scope `k=K`, so k/width varies within a unit. For each task ×
truth class (positive, negative_random, negative_wrong_algorithm separately) × procedure:

logit P(certify) = β₀ + β₁ log₂ width + β₂ k/width + β₃ log n_fit

(covariates that are constant in the data are dropped, e.g. log n_fit in the pilot). Fitted by
maximum likelihood with cluster-robust (by network) standard errors; if the outcome is constant,
statsmodels reports (quasi-)separation or non-convergence, or a fitted |linear predictor| > 15,
Firth's penalized likelihood is used instead (`stats.firth_logit`: Newton steps on the modified
score X'(y − p + h(½ − p)) with step halving; Wald SEs from (X'WX)⁻¹; tested against the
closed form logit((x+½)/(n+1)) for an intercept-only model and against MLE on large data). The
`method` column says which. A random-intercept-per-network model (statsmodels
`BinomialBayesMixedGLM`, variational Bayes) is fitted alongside when there are both events and
non-events (`mixed_b_*`, `mixed_log_sd_net`). **Binned residuals** (mean residual vs mean fitted
probability in 10 quantile bins, ±2 SE band) are in `figures/binned_residuals_<task>.png`.

Not implemented: likelihood-ratio tests for interactions (proposal §4); profile-likelihood CIs
for Firth.

## Paired comparisons (`paired_tests.csv`)

All procedures are applied to the same units, so decisions are paired. Per task × truth class,
on units with a decision under every procedure available for that task: Cochran's Q omnibus
test, then pairwise exact McNemar tests (two-sided binomial on discordant pairs) with Holm
adjustment.

## Bootstrap intervals for IIA (`bootstrap_intervals.csv`, `bootstrap_widths.csv`)

For each evaluation unit, the fixed-config IIA on the fit split and on the test split, with
B = `analysis.bootstrap_B` replicates, 95% percentile intervals under three schemes
(`bootstrap.py`; IIA* = Σ w c / Σ w):

* `iid`: multinomial resampling of pairs (ignores shared inputs);
* `base`: base inputs resampled; pair weight = its base's count (one-way cluster bootstrap);
* `twoway`: base and source inputs resampled independently; pair weight = (base count) ×
  (product of its sources' counts), the pigeonhole product bootstrap for crossed factors
  (Owen 2007, extended to two sources by multiplying), which is mildly conservative.

`bootstrap_widths.csv` gives the median interval width per scheme and the ratios base/iid and
twoway/iid by task, class, width and n_fit (the planned preliminary result).

**Caveat (measured in the pilot).** With each input used in at most 2 pairs, the two-way
interval is about twice the iid width even for independent pairs: the product weights have
E[w²] = 4 (one source) or 8 (two sources) against 2 for iid weights, so the product bootstrap
over-counts the pair-level variance when clusters are this small. Treat it as a conservative
bound, not as evidence of crossed dependence (see [running.md](running.md)).
