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
k ∈ round(`k_fracs` × width) ∪ {k*}, s < `das.seeds`. The pre-registered configuration c₀ is
(L*, k*, seed 0): the IIT site and size; random and standard networks of the same width use the
same nominal values. A *scope* is the set of configurations a procedure may use:

* `all`: every layer, k and seed (main analysis);
* `site`: only layer L* (every k and seed). Truth is defined at L*, and at other layers a
  wrong-algorithm network can genuinely implement the other algorithm (example in
  [tasks_and_networks.md](tasks_and_networks.md)), so α̂ in scope `site` is the cleaner
  false-positive rate;
* `k=K`: every layer and seed at size K, fixed config (L*, K, 0); used for the regression on
  k/width.

Only fits in the configured grid are read; `analyze` refuses to run if any unit is missing a fit
(`--allow-incomplete` drops such units and flags them at the top of `summary.md`).

## Procedures (proposal §4 (1)–(4) plus factorial cells)

With IIA_split(c) the IIA of configuration c on a split, C the scope's converged configurations
and threshold c (default 0.9):

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
F_insample_fixed and F_heldout_fixed, so the effect of held-out evaluation and the effect of
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
networks, so no test is evaluated on data that calibrated it.

### Two caveats on reading the results

1. **The wrong-algorithm null is an oracle.** `P3_null_wrong_algorithm` and
   `P4_maxT_wrong_algorithm` calibrate on networks built exactly like the wrong-algorithm
   negatives they are then tested on. The two are exchangeable, so their α for those negatives is
   at the nominal level *by construction*. Their low false-positive rates are a correctness check
   of the code, not a finding about the method: in practice an analyst does not know the true
   alternative algorithm and cannot build this null. They are marked † in `summary.md`. The
   `random` and `shuffled` nulls are the ones an analyst could actually build.
2. **Random negatives are nearly automatic.** Most frozen random networks (PyTorch default init)
   predict one constant class whatever the intervention. With label-stratified pairs that gives
   IIA of exactly 0.5. `summary.md` reports the share of random-network fits whose test
   predictions are constant (61% in the pilot, as the review also found). So α̂ ≈ 0 for random negatives under the
   threshold procedures says little about DAS. Only wrong-algorithm negatives test the
   procedures seriously.

**Exclusions.** Non-converged fits (see [das.md](das.md)) are dropped before anything else. A
max statistic uses the remaining configurations of the scope; a fixed-config statistic is
missing if that fit did not converge, and the unit then has no decision for that procedure.
Networks failing their training targets are excluded entirely. Counts:
`tables/excluded_networks.csv`, `tables/excluded_fits.csv`, and the top of `summary.md`.
Ties in argmax go to the first configuration in (layer, k, seed) order.

## Independence: which units can be pooled

A network yields one analysis unit per (hypothesis, n_fit): each distributive-law random network
gives two units (one per hypothesis), and in the full design every network appears at three
n_fit values. These units are not independent. So:

* Rates, Wilson intervals, binomial tests and paired tests are computed only within groups where
  **each network appears once**: the headline cell (task, hypothesis, n_fit), pooled over widths
  (`tables/rates_by_hypothesis_nfit.csv`, the tables in `summary.md`), and the design cell
  (task, hypothesis, width, n_fit) (`tables/rates_by_cell.csv`). `analyze` raises an error if a
  rate group contains a network twice.
* Summaries pooled over hypotheses and n_fit (`tables/rates_pooled.csv`,
  `threshold_sweep_pooled.csv`, the sweep figures) report the rate with a 95% **cluster
  bootstrap over networks** (`stats.cluster_bootstrap_rate`: resample networks with
  replacement, keep all of a network's units together), never a binomial interval.
* The regression rows (unit × k) contain each network several times; the MLE uses
  cluster-robust standard errors by network, and the mixed model has a random intercept per
  network.

## Rates per cell

For each cell, truth class and procedure:

* α̂ (negatives) or π̂ (positives) = certified / n networks, with the 95% Wilson score interval
  p̃ ± z√(p̂(1−p̂)/n + z²/4n²)/(1 + z²/n), p̃ = (p̂ + z²/2n)/(1 + z²/n), z = 1.96;
* the one-sided 95% upper Wilson bound (z = 1.645). Report this for α: failing to reject
  H0: α ≤ 0.05 is not evidence that α is controlled. With 0 certified out of n the bound is
  roughly 3/n (n = 6: 0.31; n = 12: 0.18; n = 30: 0.08);
* for negatives, the exact one-sided binomial test of H0: α ≤ 0.05, p = P(X ≥ x | n, 0.05), and
  Holm's step-down adjustment across the cells of the same (scope, procedure, truth class);
* `threshold_sweep.csv` (per headline cell) and `threshold_sweep_pooled.csv`: rates for
  c ∈ `analysis.thresholds` (p-value procedures don't depend on c).

Standard networks (`unknown_standard`) appear only as descriptive certification rates; they
never enter α̂, π̂, tests or regressions.

## Across cells: logistic regression (`regression.csv`)

Rows are (unit, k) decisions in scope `k=K`, so k/width varies within a unit. For each task ×
truth class (positive, negative_random, negative_wrong_algorithm separately) × procedure:

logit P(certify) = β₀ + β₁ log₂ width + β₂ k/width + β₃ log n_fit

(covariates that are constant in the data are dropped, e.g. log n_fit in the pilot). Fitted by
maximum likelihood with cluster-robust (by network) standard errors. If the outcome is constant,
statsmodels reports (quasi-)separation or non-convergence, or a fitted |linear predictor| is
above 15, Firth's penalized likelihood is used instead (`stats.firth_logit`). This takes Newton
steps on the modified score X'(y − p + h(½ − p)) with step halving, and it is tested against the
closed form logit((x+½)/(n+1)) for an intercept-only model and against MLE on large data. Its
standard errors are **model-based** (Wald, from (X'WX)⁻¹), not cluster-robust; the `method`
column says `firth (model-based SE)` when this happens. A random-intercept-per-network model
(statsmodels `BinomialBayesMixedGLM`, variational Bayes) is fitted alongside when there are both
events and non-events (`mixed_b_*`, `mixed_log_sd_net`). **Binned residuals** (mean residual vs
mean fitted probability in 10 quantile bins, ±2 SE band) are in
`figures/binned_residuals_<task>.png`.

Not implemented: likelihood-ratio tests for interactions (proposal §4); profile-likelihood CIs
for Firth.

## Paired comparisons (`paired_tests.csv`)

All procedures are applied to the same units, so decisions are paired. Within each headline
cell (task, hypothesis, n_fit) × truth class, with one row per network and only the networks
that have a decision under every procedure available in that cell:

* **primary:** pairwise exact McNemar tests (two-sided binomial on the discordant pairs), Holm
  across all pairs of the cell;
* **secondary:** Cochran's Q over the five threshold procedures (P1, P2, F_insample_fixed,
  F_heldout_fixed, P4_split) with a **permutation p-value** (`stats.cochran_q_permutation`:
  each network's decisions are permuted across procedures, 10 000 permutations). This avoids
  the χ² approximation, which is poor with ~6–12 networks.

## Bootstrap intervals for IIA (`bootstrap_intervals.csv`, `bootstrap_widths.csv`)

For each evaluation unit, the held-out (test-split) IIA at c₀, with B = `analysis.bootstrap_B`
replicates, 95% percentile intervals under three schemes (`bootstrap.py`;
IIA* = Σ w c / Σ w):

* `iid`: multinomial resampling of pairs (ignores shared inputs);
* `base`: base inputs resampled; pair weight = its base's count (one-way cluster bootstrap).
  **Recommended primary interval**;
* `twoway`: base and source inputs resampled independently; pair weight = (base count) ×
  (product of the counts of its distinct sources). This is the pigeonhole product bootstrap for
  crossed factors (Owen 2007), extended to two sources.

Fit-split intervals are not computed. R was trained on those very pairs, so resampling them
does not reflect refitting.

**Evidence (Monte Carlo).** `scripts/bootstrap_monte_carlo.py` fixes a fitted R on four units
whose IIA is far from 0.5 and 1: two distributive-law wrong-algorithm units and a task-trained
HE network at two sites. It then draws 300 fresh test splits of 1024 pairs each and compares the
SD of IIA across splits with the bootstrap SDs from one split:

| unit | mean IIA | Monte Carlo SD | iid | base | two-way |
|---|---|---|---|---|---|
| DL IIT-AOA net 1 vs And-Or, L1 k6 | 0.785 | 0.0128 | 0.0127 | 0.0139 | 0.0269 |
| DL IIT-AOA net 2 vs And-Or, L1 k6 | 0.790 | 0.0125 | 0.0129 | 0.0138 | 0.0262 |
| HE standard net, L0 k4 | 0.721 | 0.0133 | 0.0138 | 0.0145 | 0.0295 |
| HE standard net, L1 k8 | 0.617 | 0.0160 | 0.0152 | 0.0158 | 0.0310 |

The independent review found the same with its own units: Monte Carlo SD 0.011–0.014;
iid/base 0.013–0.016; two-way 0.028–0.031. So with inputs used in at most 2 pairs:

* the base-cluster bootstrap is about right, slightly conservative;
* there is little dependence to correct for, so iid is close too;
* the two-way bootstrap is about **2× too wide**.

The weights explain this. For independent pairs, the variance of a weighted mean scales with
E[w²]. That is 2 for iid weights. For the product weights it is 4 with one source and 8 with two
(1/3 of pairs). The mix gives a Var(w) ratio of about 4.33 and a width ratio of about 2.1: the
product scheme counts each pair's own noise once per factor. Two-way intervals are reported but
should not be used for inference in this design.
