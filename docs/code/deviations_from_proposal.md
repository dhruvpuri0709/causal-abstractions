# Deviations from the proposal

Each item: what `docs/proposal.tex` says, what the code does, and why.

1. **Truth is defined for a linear abstraction at a specific site.** Proposal: "known positives
   are trained to compute the algorithm's variables". Code: positives are IIT-trained
   (Geiger et al. 2022) so V1, V2 live in neurons [0, k*) and [k*, 2k*) of layer L* (default
   L* = 1, k* = width/4); T is relative to a rotation abstraction at L*. Why: "implements the
   algorithm" needs an operational definition before an error rate means anything.

2. **Two kinds of negatives, kept separate everywhere.** Proposal: "known negatives are random
   or follow a different algorithm". Code: `negative_random` (frozen random init, at chance on
   the task, as in Geiger et al. 2024 §4.4) and `negative_wrong_algorithm` (distributive law:
   IIT-trained to And-Or-And and tested for And-Or, or vice versa; these solve the task). Every
   table and figure reports them separately. Why: random networks are much easier negatives
   (the proposal itself warns about this). No wrong-algorithm construction is implemented for
   hierarchical equality.

3. **Standard (task-trained) networks are descriptive only.** Their truth is unknown, so they
   never enter α̂, π̂, tests or regressions.

4. **Label agreement covariate.** For wrong-algorithm negatives the share of evaluated pairs on
   which the two algorithms' counterfactual labels agree is recorded (`label_agreement`), as
   the proposal suggests. It is recorded, not yet used as a regression covariate.

5. **Per-pair storage.** Every fit stores per-pair predictions and correctness for each split
   plus the base and source ids of every pair; all procedures, nulls and bootstraps are
   computed offline. (Not in the proposal; makes the analysis cheap to rerun and checkable.)

6. **Three disjoint splits with disjoint base and source pools.** Proposal: held-out base
   inputs. Code: fit / select / test, each with its own base pool and source pool, each input
   used in at most 2 pairs, pairs stratified on (label, changed). Why: held-out *source* inputs
   too, a third split for selection, and recorded clusters for the two-way bootstrap.

7. **Factorial of procedures.** Added `F_insample_fixed` and `F_heldout_fixed` (pre-registered
   configuration, in-sample or held-out) so the effect of held-out evaluation is not
   confounded with the effect of selection. The proposal's four procedures keep their names
   P1–P4.

8. **Null-calibrated procedure, per null type.** Proposal: one p̂ from "calibration negatives
   and pairs with shuffled counterfactual labels". Code: three separate nulls (`random`,
   `wrong_algorithm`, `shuffled`), each reported separately; the statistic is the held-out IIA
   at the pre-registered configuration. Calibration networks are disjoint from evaluation
   networks (own ids and seeds). B ≥ 19 is needed to certify at all at 0.05 (min p̂ =
   1/(1+B)); the pilot uses B = 19, the full design B = 39, and `analyze` warns otherwise.

9. **Selection correction, both versions.** Proposal: "selection split and tested once on a
   third split, or compared with the null distribution of the maximum IIA". Code: both
   (`P4_split`, `P4_maxT_<null>`).

10. **Bootstrap: three schemes, not one.** Proposal: base-input cluster bootstrap. Code: iid,
    base-cluster and two-way base × source (Owen 2007 pigeonhole), with a width comparison
    table, because pairs share source inputs as well as base inputs.

11. **Statistics added beyond the proposal.** One-sided upper Wilson bounds for α; Holm across
    cells within each procedure × negative type; Cochran's Q before pairwise McNemar (with
    Holm); cluster-robust SEs by network in the logistic regression. Mixed-effects logistic is
    fitted with statsmodels' variational-Bayes `BinomialBayesMixedGLM` (not a frequentist
    GLMM). Not implemented: likelihood-ratio tests for interactions.

12. **Regression unit.** The proposal's covariate "subspace size over width" is not defined for
    a procedure that maximises over subspace sizes. The regression therefore uses decisions
    within a fixed-k scope (rows = unit × k); the main α̂/π̂ tables use scope `all`.

13. **Convergence definition.** Proposal: "non-converged fits will be excluded and reported".
    Code: a fit has converged if its 50-step window training loss stops improving by more than
    1e-3 for 5 windows or drops below 1e-3, within 4000 steps. The step budget is independent
    of n_fit (an earlier epoch-based budget left small fitting sets under-trained; caught in a
    smoke run and fixed before the pilot). Networks failing their training targets are also
    excluded and reported.

14. **Sizes and data.** Objects are uniform in [-0.5, 0.5]^4 for both tasks (Sutter et al.'s
    range; the pyvene tutorial uses [-1, 1] rounded to 2 decimals). DAS uses far fewer pairs
    than Geiger et al. (pilot 512 fitting pairs, reproduce 8192, vs 640K), and our fitting loop
    and pair sampler differ from theirs. Widths: pilot 16/32 (HE) and 24 (DL); full 16/32/64 and
    24/48/96.

15. **Reproduction result differs for task-trained networks.** The proposal plans to reproduce
    near-perfect IIA on a trained network. IIT-trained networks reproduce Table 1 closely, but
    our task-trained networks reach only 0.67–0.94 best in-sample IIA at layer 0, k = 8
    (Table 1: 1.00), see [running.md](running.md). Not resolved.
