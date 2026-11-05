# Running

## Setup and commands

```bash
uv sync
uv run pytest                    # 37 fast tests, ~8 s
uv run pytest -m slow            # cross-check against pyvene's RotatedSpaceIntervention, ~7 s
uv run cas reproduce             # ~70 s -> results/reproduce/report.md
uv run cas plan --config configs/pilot.yaml      # design size, no computation
uv run cas pilot                 # = train + fit + analyze on configs/pilot.yaml, ~7.5 min
uv run cas train   --config configs/full.yaml    # each stage separately; all resumable
uv run cas fit     --config configs/full.yaml
uv run cas analyze --config configs/full.yaml
```

Options: `--workers N` (default 8; worker processes, spawn context, one torch thread each) and
`--device cpu|mps` (default cpu; for these tiny networks the CPU is faster). `cas reproduce`
takes `--widths 16,64,256,1024` to override the random-network widths.

`train` skips networks whose files exist, `fit` skips fits whose files exist, so an interrupted
run can simply be restarted. `analyze` only reads files and can be rerun at any time (e.g. after
changing `analysis.threshold` in the config). Same config → same results: all seeds derive
from `seed` and object ids (`config.derive_seed`). Each command appends its stage timings to
`results/<name>/timings.log`.

## Configs

| Config | Design | Measured / estimated time |
|---|---|---|
| `configs/reproduce.yaml` | Section 1–4 of `cas reproduce` | 68 s measured |
| `configs/pilot.yaml` | HE widths 16, 32; DL width 24; per (task, width): 6 IIT (per algorithm), 6 random, 2 standard evaluation networks; 19 random + 19 IIT (per algorithm, DL only) calibration networks; layers 0–2 × k ∈ {w/4, w/2} × 2 seeds; n_fit = 512, select/test 1024 pairs; 19 shuffled refits per unit. 143 networks, 182 units, 2184 DAS fits + 1292 shuffled fits | 443 s measured (train 83 s, fit 321 s, analyze 37 s), 67 MB |
| `configs/full.yaml` | HE widths 16, 32, 64; DL widths 24, 48, 96; 30 IIT / 30 random / 10 standard evaluation networks; 39 + 39 calibration networks (B = 39); layers 0–2 × k ∈ {w/8, w/4, w/2} × 3 seeds; n_fit ∈ {256, 1024, 4096}; 39 shuffled refits. 978 networks (564 trained), 4185 units, 112 995 DAS fits + 46 170 shuffled fits | **estimated ~6–7 h** with 8 workers, ~3 GB (see below) |

**How the full-design estimate was made.** In the pilot, mean wall time per fit with 8
concurrent workers was 0.44 s (width 16), 0.73 s (width 24) and 0.60 s (width 32); the step
budget does not depend on n_fit. Single-process timings of fits at width 48, 64 and 96
(IIT networks, 4096 fitting pairs) were 0.4–1.1 s, 0.3–1.1 s and 0.8–1.9 s, about 1.5–2× width
24. Assuming 0.9 s (48, 64) and 1.6 s (96) per fit under 8-way load gives ~43 worker-hours of
fitting, i.e. **~5.4 h wall** with 8 workers, plus training 564 networks (pilot: 68 in 83 s;
wider networks are slower; estimate 15–30 min) and analysis (pilot 37 s for 182 units; scales
with units and bootstrap replicates; estimate 15–30 min). Storage scales with the number of fits
(pilot: 63 MB for 3476 fits → ~3 GB). These are extrapolations, not measurements; run `cas plan`
and time a single width first if it matters. The full design has not been run.

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

   **The near-perfect IIA on a task-trained network does not reproduce.** Five more
   task-trained networks reach 0.70–0.87 best in-sample IIA at layer 0, k = 8 (0.67–0.94 over
   the 11 task-trained networks we checked during development). We checked that this is not
   early stopping of the rotation (4000 steps without early stopping, lr 0.001/0.01/0.05: same
   values), not too little network training (20 000 steps to 0.9998 accuracy: same), and not too
   few DAS pairs (65 536 fitting pairs: same, 0.63 and 0.81 on two networks). Remaining
   differences from the paper: object range [-0.5, 0.5] vs the tutorial's [-1, 1] rounded to 2
   decimals, our stratified pair sampler, PyTorch default init and our training loop. The
   IIT-trained network does reproduce Table 1 (layer 0: 0.89 / 0.98 / 1.00 vs 0.88 / 0.97 /
   1.00; layers 1–2 near 0.5–0.57 as in the paper). The pyvene tutorial's own run reaches 0.97
   held-out on its trained network, so some of its networks do; ours vary by seed.
3. **Random networks** (§4.4), k = N/2, layer 0, best in-sample / held-out of best:
   N = 16: 0.501 / 0.499 (paper 0.50); N = 64: 0.500 / 0.500 (0.50); N = 256: 0.517 / 0.491
   (0.51). The small in-sample excess at N = 256 disappears held-out. Width 1024 costs ~0.4 s per
   step (73 s for a 192-step fit in a development test; a 1024 × 1024 orthogonal
   parametrization per step) and was not run here; 4096 is impractical on this machine.
4. **Timing**: one DAS fit (width 16, layer 0, k = 8, 8192 pairs) took 0.75 s for 1300 steps
   (0.58 ms/step). Width 256: 4.0 s per fit.

## Pilot results (measured 2026-11-05)

`uv run cas pilot`: 443 s wall (train 83 s, fit 321 s, analyze 37 s). No network failed its
training targets (0 / 143), no DAS fit (0 / 2184) and no shuffled fit (0 / 1292) failed to
converge. All rates below are at c = 0.9, scope `all`, pooled over cells; n = 12 units per class
and task (6 networks × 2 widths for HE; 6 networks × 2 hypotheses for DL). With n = 12 and 0
events the one-sided 95% upper bound on α is 0.18, so **the pilot cannot show α ≤ 0.05**; it
shows that the pipeline works and where the rates are clearly not small.

Mean IIA (P1 naive / F_heldout_fixed):

| task | positive | random negative | wrong-algorithm negative | standard |
|---|---|---|---|---|
| hierarchical equality | 0.999 / 0.996 | 0.519 / 0.499 | — | 0.915 / 0.504 |
| distributive law | 1.000 / 0.992 | 0.507 / 0.501 | 0.818 / 0.748 | 0.839 / 0.739 |

Wrong-algorithm label agreement (hypothesis vs trained algorithm, test pairs): mean 0.64
(0.61–0.68).

Certification rates (certified / n):

| procedure | HE π̂ | HE α̂ random | DL π̂ | DL α̂ random | DL α̂ wrong-alg |
|---|---|---|---|---|---|
| P1_naive | 12/12 | 0/12 | 12/12 | 0/12 | 0/12 |
| P2_heldout | 12/12 | 0/12 | 12/12 | 0/12 | 0/12 |
| F_insample_fixed | 12/12 | 0/12 | 12/12 | 0/12 | 0/12 |
| F_heldout_fixed | 12/12 | 0/12 | 12/12 | 0/12 | 0/12 |
| P4_split | 12/12 | 0/12 | 12/12 | 0/12 | 0/12 |
| P3_null_random | 12/12 | 0/12 | 12/12 | 1/12 | **12/12** |
| P3_null_wrong_algorithm | — | — | 12/12 | 0/12 | 0/12 |
| P3_null_shuffled | 12/12 | 0/12 | 12/12 | 1/12 | **12/12** |
| P4_maxT_random | 12/12 | 0/12 | 12/12 | 1/12 | **12/12** |
| P4_maxT_wrong_algorithm | — | — | 12/12 | 0/12 | 2/12 |

Reading: at c = 0.9 every threshold procedure separates the classes perfectly in this small
pilot (positives ≈ 1.0, wrong-algorithm negatives ≈ 0.72–0.87). The nulls calibrated on random
networks or on shuffled labels are much too weak: they certify every wrong-algorithm network
(H0: α ≤ 0.05 rejected after Holm in all 6 such cells), because a wrong algorithm with ~0.75 IIA
is far above a null centred at 0.5. The wrong-algorithm null behaves as intended (0/12 and
2/12). Cochran's Q across procedures for DL wrong-algorithm negatives: Q = 101.7, p = 7e-18.

The threshold matters a lot for the wrong-algorithm negatives (`tables/threshold_sweep.csv`,
certified out of 12):

| c | P1_naive | F_insample_fixed | P2_heldout | P4_split | F_heldout_fixed |
|---|---|---|---|---|---|
| 0.70 | 12 | 12 | 12 | 12 | 11 |
| 0.75 | 10 | 8 | 8 | 8 | 6 |
| 0.80 | 8 | 3 | 1 | 0 | 0 |
| 0.85 | 3 | 0 | 0 | 0 | 0 |
| 0.90 | 0 | 0 | 0 | 0 | 0 |

So at c ≤ 0.85 the naive procedure certifies wrong algorithms, and both held-out evaluation and
removing selection lower that rate, in the expected order. Random negatives are certified only
at c = 0.6 (P1, HE, 1/12). Positives stay at 12/12 up to c = 0.95; at c = 0.99 DL drops to 10/12
for the held-out and fixed procedures, and at c = 1.0 most procedures fail.

**Bootstrap widths** (median 95% width, fixed-config IIA, test split, 1024 pairs): iid ≈ 0.061,
base-cluster ≈ 0.061 (ratio 0.98–1.03), two-way ≈ 0.12 (ratio ≈ 2.0) for networks at chance.
With each base and source used in at most 2 pairs there is almost no within-base correlation to
detect, so the base-cluster interval matches the iid one. The two-way (product) interval is about
twice as wide, and this is mostly the known upward bias of the product bootstrap when clusters
are tiny, not real dependence. With weights w = W_b W_s (one source) or W_b W_s1 W_s2 (two),
E[w²] is 4 or 8 against 2 for iid, which predicts a variance ratio of ~4.3 and a width ratio of
~2.1 for independent pairs. That is what we see. So in this design the two-way interval is very
conservative and should not be read as evidence of crossed dependence. A design with more reuse
per input (`base_uses`, `source_uses`) would be needed to measure that.

**Regression**: with one n_fit, one DL width and many all-0 or all-1 outcomes, 42 of 46 pilot
regressions fell back to Firth, and the coefficients are uninformative (as expected for this
design). The mixed model fitted where there were both events and non-events (4 cases).

## Known limitations

* Task-trained networks do not reproduce Table 1's near-perfect IIA (above).
* No wrong-algorithm construction for hierarchical equality.
* Width 1024+ random networks only via `--widths`, slow; 4096 not attempted.
* The full design has not been run; its runtime is an extrapolation.
* The two-way bootstrap is strongly conservative at the current input reuse (2).
* Likelihood-ratio tests for interactions are not implemented; Firth SEs are Wald.
* `label_agreement` is recorded but not yet used as a regression covariate.
