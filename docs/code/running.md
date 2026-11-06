# Running

## Setup and commands

```bash
uv sync
uv run pytest                    # 47 tests, ~40 s (includes a converging end-to-end run)
uv run pytest -m slow            # cross-check against pyvene's RotatedSpaceIntervention, ~7 s
uv run cas reproduce             # ~70 s -> results/reproduce/report.md
uv run cas plan --config configs/pilot.yaml      # design size, no computation
uv run cas pilot                 # = cas run --config configs/pilot.yaml, ~7.5 min
uv run cas run     --config configs/full.yaml    # train -> fit -> analyze
uv run cas train   --config configs/full.yaml    # or each stage separately; all resumable
uv run cas fit     --config configs/full.yaml
uv run cas analyze --config configs/full.yaml
uv run python scripts/bootstrap_monte_carlo.py   # bootstrap check, ~20 s
```

Options:

* `--workers N`: worker processes, spawn context, one torch thread each. The default is
  min(8, CPU count). For `reproduce`, it sets the number of torch threads instead, because
  reproduce runs its fits one after another.
* `--device cpu|mps`: the default is cpu; for these tiny networks the CPU is faster.
* `--force` (train, fit, run): rebuild outputs whose settings fingerprint no longer matches the
  config.
* `--allow-incomplete` (analyze, run): analyze even if some units have missing fits; those units
  are dropped and flagged at the top of `summary.md`.
* `cas reproduce --widths 16,64,256,1024`: override the random-network widths.

**Resuming and config changes.** `train` skips networks, and `fit` skips pair files and fits,
that already exist *with the same settings*. Every network, unit directory (`unit.json`), pair
file and fit stores a fingerprint of the config sections that produced it
(`Config.network_fingerprint`, `Config.unit_fingerprint`; a unit's fingerprint includes its
network's). If you change those settings (training budget, `das.fit`, pair counts, seed, ...)
and rerun, `train`/`fit`/`analyze` stop with an error naming the stale outputs. Then either give
the config a new `name` or rerun with `--force`, which rebuilds the mismatched outputs. Note what
"mismatched" means: a network's fingerprint covers its whole `train.<kind>` section, so changing
the IIT training budget makes **every** IIT network stale (and with it every unit of those
networks), not only the ones that failed.

* **Pass thresholds** (`train.<kind>.pass_*`) are *not* part of the fingerprint. `passed` is
  recomputed from the stored accuracies whenever networks are loaded, so changing a threshold
  needs no retraining. It just changes which networks are used.
* **A unit directory whose `unit.json` is missing or unreadable** counts as stale.
* **Within a unit, each pair file and fit is checked against its own fingerprint**, so
  `fit --force` rebuilds exactly what does not match.
* **Units of networks that fail their pass thresholds** are never fitted or analyzed. Their
  directories are left alone and do not block `fit`.
* **Changing only the grid** (`das.layers`, `k_fracs`, `seeds`, `null.shuffle_B`, network counts)
  keeps fingerprints: the missing fits are added and `analyze` reads exactly the configured grid.
* **Partially written files** (`.part`) are ignored; `.json` and `.npz` outputs are written via a
  `.part` file and an atomic rename.
* **Code changes are not covered**: after changing the code, use a new `name`.

Same config → same results. All seeds derive from `seed` and object ids, and a test checks that
1 and 2 workers give identical fits, shuffled fits and decisions. Each command appends its
stage timings to `results/<name>/timings.log`.

## Configs

| Config | Design | Time |
|---|---|---|
| `configs/reproduce.yaml` | Sections 1–4 of `cas reproduce` | 68 s measured |
| `configs/pilot.yaml` | HE widths 16, 32; DL width 24. Per (task, width): 6 IIT evaluation networks (per algorithm), 6 random, 2 standard; 19 random + 19 IIT (per algorithm, DL only) calibration networks. Layers 0–2 × k ∈ {w/4, w/2} × 2 seeds; n_fit = 512, select/test 1024 pairs; 19 shuffled refits per unit. 143 networks, 182 units, 2184 DAS fits + 1292 shuffled fits | 448 s measured (train 85 s, fit 333 s, analyze 27 s), 67 MB |
| `configs/full.yaml` | HE widths 16, 32, 64; DL widths 24, 48, 96. 30 IIT / 30 random / 10 standard evaluation networks; 39 + 39 calibration networks (B = 39). Layers 0–2 × k ∈ {w/8, w/4, w/2} × 3 seeds; n_fit ∈ {256, 1024, 4096}; 39 shuffled refits. 978 networks (564 trained), 4185 units, 112 995 DAS fits + 46 170 shuffled fits | **estimated ~6–7 h** with 8 workers, ~3 GB |

**How the full-design estimate was made.**

* **Fitting, ~5.4 h wall with 8 workers.**
  * Pilot mean wall time per fit with 8 concurrent workers: 0.44 s (width 16), 0.73 s (width 24),
    0.60 s (width 32). The step budget does not depend on n_fit.
  * Single-process fits at widths 48, 64 and 96 (IIT networks, 4096 pairs) took 0.4–1.1 s,
    0.3–1.1 s and 0.8–1.9 s.
  * Assuming 0.9 s (48, 64) and 1.6 s (96) per fit under 8-way load gives ~43 worker-hours.
* **Training 564 networks:** estimated 15–30 min. The pilot trained 68 in 85 s, and wider
  networks are slower.
* **Analysis:** estimated 15–30 min.
* **Storage:** ~3 GB (the pilot used 63 MB for 3476 fits).

These are extrapolations; the full design has not been run.

## `cas reproduce` results (measured 2026-11-05, M1, CPU)

From `results/reproduce/report.md` (8192 fitting pairs, 2048 test pairs, best of 3 seeds by
in-sample IIA as in the paper; our layer 0 = paper's Layer 1):

1. **Handcrafted network**, known neuron alignment: IIA = **1.0000** (expected 1.0).
2. **Table 1 analogue**, width 16:

   | network | layer 0, k=1 | k=2 | k=8 | layer 1, k=8 | layer 2, k=8 |
   |---|---|---|---|---|---|
   | Geiger et al. Table 1 (train IIA) | 0.88 | 0.97 | 1.00 | 0.57 | 0.50 |
   | IIT-trained (site layer 0, k* = 4), in-sample / held-out | 0.894 / 0.893 | 0.980 / 0.973 | 0.997 / 0.996 | 0.566 / 0.573 | 0.511 / 0.494 |
   | task-trained (standard), in-sample / held-out | 0.629 / 0.604 | 0.659 / 0.626 | 0.675 / 0.675 | 0.543 / 0.534 | 0.504 / 0.495 |

   Five more task-trained networks reach 0.70–0.87 best in-sample IIA at layer 0, k = 8
   (0.67–0.94 over the 11 we checked). **This is the expected spread across training seeds, not
   a failure to reproduce.** Sutter et al.'s published results for the same setting
   (`external/non-linear-representation-dilemma/Results.zip`,
   `Standard_DAS/Rotation/FullyTrained/Both_Equality_Relations/results.json`, checked by us)
   give 0.719–0.975 (mean 0.842) over 10 task-trained networks at the paper's Layer 1 with
   k = 8, and 0.560–0.919 (mean 0.730) with k = 1. The object range is not the cause: Geiger
   et al.'s `InterchangeInterventions/utils.py` (`randvec`) uses [-0.5, 0.5] as we do, and the
   independent review found similar spreads with the tutorial's [-1, 1] rounded inputs, one
   network reaching 0.993. Longer network training (20 000 steps, 0.9998 accuracy), longer
   rotation training and 65 536 fitting pairs did not change our values either. Table 1's 1.00
   is the best of 3 rotation seeds on a single network: a lucky draw from a wide distribution.
   IIT-trained networks reproduce Table 1 (layer 0: 0.89 / 0.98 / 1.00 vs 0.88 / 0.97 / 1.00).
3. **Random networks** (§4.4), k = N/2, layer 0, best in-sample / held-out of best:
   N = 16: 0.501 / 0.499 (paper 0.50); N = 64: 0.500 / 0.500 (0.50); N = 256: 0.517 / 0.491
   (0.51). The small in-sample excess at N = 256 disappears held-out. Width 1024 costs ~0.4 s per
   step (73 s for a 192-step fit in a development test; a 1024 × 1024 orthogonal
   parametrization per step) and was not run here; 4096 is impractical on this machine.
4. **Timing**: one DAS fit (width 16, layer 0, k = 8, 8192 pairs) took 0.75 s for 1300 steps
   (0.58 ms/step). Width 256: 4.0 s per fit.

## Pilot results (rerun after the review fixes, 2026-11-05)

`uv run cas pilot`: 448 s wall (train 85 s, fit 333 s, analyze 27 s). No network failed its
training targets (0 / 143), no DAS fit (0 / 2184) and no shuffled fit (0 / 1292) failed to
converge. The decisions are identical to the first pilot run; what changed is how they are
counted. Each network now appears once per cell. In the first version of `running.md`, "n = 12"
for DL random negatives was 6 networks counted twice (once per hypothesis), and the quoted
upper bound of 0.18 should have been 0.31.

**Read these with the caveats in [procedures_and_statistics.md](procedures_and_statistics.md):**

* the wrong-algorithm nulls (†) are oracles, so their α for wrong-algorithm negatives is nominal
  by construction;
* 61% of random-network fits predict one constant class (IIA exactly 0.5), so α̂ ≈ 0 for random
  negatives is close to automatic;
* with 6 networks per class and 0 certified, the one-sided 95% upper bound on α is 0.31. **The
  pilot cannot show that any procedure controls α.**

Certified / networks at c = 0.9, scope `all`, per cell (task, hypothesis, n_fit = 512), pooled
over widths (one row per network; upper one-sided 95% Wilson bound for negatives in
parentheses):

| procedure | DL `and_or`: pos (6) | rand (6) | wrong-alg (6) | DL `and_or_and`: pos (6) | rand (6) | wrong-alg (6) | HE: pos (12) | rand (12) |
|---|---|---|---|---|---|---|---|---|
| P1_naive | 6 | 0 (.31) | 0 (.31) | 6 | 0 (.31) | 0 (.31) | 12 | 0 (.18) |
| P2_heldout | 6 | 0 (.31) | 0 (.31) | 6 | 0 (.31) | 0 (.31) | 12 | 0 (.18) |
| F_insample_fixed | 6 | 0 (.31) | 0 (.31) | 6 | 0 (.31) | 0 (.31) | 12 | 0 (.18) |
| F_heldout_fixed | 6 | 0 (.31) | 0 (.31) | 6 | 0 (.31) | 0 (.31) | 12 | 0 (.18) |
| P4_split | 6 | 0 (.31) | 0 (.31) | 6 | 0 (.31) | 0 (.31) | 12 | 0 (.18) |
| P3_null_random | 6 | 1 (.50) | **6** (1.0) | 6 | 0 (.31) | **6** (1.0) | 12 | 0 (.18) |
| P3_null_shuffled | 6 | 1 (.50) | **6** (1.0) | 6 | 0 (.31) | **6** (1.0) | 12 | 0 (.18) |
| P4_maxT_random | 6 | 1 (.50) | **6** (1.0) | 6 | 0 (.31) | **6** (1.0) | 12 | 0 (.18) |
| P3_null_wrong_algorithm † | 6 | 0 (.31) | 0 (.31) | 6 | 0 (.31) | 0 (.31) | — | — |
| P4_maxT_wrong_algorithm † | 6 | 0 (.31) | 2 (.65) | 6 | 0 (.31) | 0 (.31) | — | — |

Standard (task-trained) networks, descriptive only: HE (4 networks) P1 3/4, P2 1/4,
F_heldout_fixed 0/4; DL (2 networks) P1 1/2 for `and_or` and 0/2 for `and_or_and`, the other
threshold procedures 0/2, the random and shuffled nulls 2/2.

What the pilot does show:

* **Nulls an analyst could build are too weak.** The random and shuffled nulls certify every
  wrong-algorithm network (6/6 in each DL cell; H0: α ≤ 0.05 rejected after Holm in all 6 cells
  per scope). A wrong algorithm with IIA ≈ 0.75 is far above a null centred at 0.5.
* **The threshold matters a lot for wrong-algorithm negatives.** Certified out of 6 per cell
  (`tables/threshold_sweep.csv`):

  | hypothesis | c | P1_naive | F_insample_fixed | P2_heldout | P4_split | F_heldout_fixed |
  |---|---|---|---|---|---|---|
  | `and_or` | 0.75 | 6 | 6 | 6 | 6 | 5 |
  | `and_or` | 0.80 | 6 | 2 | 1 | 0 | 0 |
  | `and_or` | 0.85 | 3 | 0 | 0 | 0 | 0 |
  | `and_or_and` | 0.75 | 4 | 2 | 2 | 2 | 1 |
  | `and_or_and` | 0.80 | 2 | 1 | 0 | 0 | 0 |

  At c ≤ 0.85 the naive procedure certifies wrong algorithms; held-out evaluation and removing
  selection lower that, in the expected order. All threshold procedures certify (nearly) every
  wrong-algorithm network at c = 0.7. Restricted to the IIT layer (scope `site`), P1 certifies
  8/12, 4/12 and 0/12 wrong-algorithm networks at c = 0.75, 0.8 and 0.85 (10/12, 8/12, 3/12 over
  all layers). Part of the naive procedure's excess comes from other layers, where the "wrong"
  algorithm may genuinely be implemented.
* Mean IIA (P1 naive / F_heldout_fixed): DL wrong-algorithm 0.818 / 0.748, positives
  1.000 / 0.992, random 0.507 / 0.501; HE positives 0.999 / 0.996, random 0.519 / 0.499,
  standard 0.915 / 0.504. Wrong-algorithm label agreement: 0.639 (0.612–0.677).

**Paired tests** (one row per network per cell): no McNemar pair is significant after Holm (0 of
444; with 6 networks per cell the smallest possible exact two-sided p is 0.03 before
adjustment). Cochran's Q (permutation) is undefined in most cells because the five threshold
procedures agree on every network. The Q = 101.7, p = 7e-18 reported in the first version came
from pooling dependent units and using the χ² approximation; it was wrong.

**Bootstrap** (held-out fixed-config IIA, 1024 test pairs): base/iid width ratio 0.98–1.07,
two-way/iid 1.6–2.1. The Monte Carlo check (`scripts/bootstrap_monte_carlo.py`, table in
[procedures_and_statistics.md](procedures_and_statistics.md)) shows the base-cluster interval is
right and the two-way one about 2× too wide. Use `base`.

**Regression**: with one n_fit, one DL width and mostly all-0 or all-1 outcomes, 42 of 46 pilot
regressions fell back to Firth (model-based SEs). The coefficients are uninformative at this
size.

## Known limitations

* No wrong-algorithm construction for hierarchical equality.
* Width 1024+ random networks only via `--widths`, slow; 4096 not attempted.
* The full design has not been run; its runtime is an extrapolation.
* The two-way bootstrap is about 2× too wide at the current input reuse (2).
* Likelihood-ratio tests for interactions are not implemented; Firth SEs are model-based Wald.
* `label_agreement` is recorded but not yet used as a regression covariate.
* Fingerprints cover config settings, not code: after changing the code, use a new `name`.
