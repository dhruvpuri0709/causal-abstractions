# Code overview

The package `cas` (in `src/cas/`) estimates how often "certify if IIA ≥ c" certifies an
algorithm that a network does not implement (type I error α) and one it does (power π), under
several reporting procedures. Everything is driven by a YAML config and a few commands
(see [running.md](running.md)).

## Data flow

```
configs/<name>.yaml
   │  Config.networks(), Config.units()            (config.py; ids and seeds are pure functions
   ▼                                                 of the config)
cas train ──► results/<name>/networks/<net_id>.pt|.json
   │            random (frozen init), standard (task loss), iit (IIT toward one algorithm);
   │            metadata records task accuracy, neuron-aligned IIA, pass/fail
   ▼
cas fit   ──► results/<name>/fits/<unit_id>/
   │            pairs.npz          fit/select/test pools of base and source inputs, pair ids,
   │                               counterfactual labels under every algorithm of the task
   │            L{l}_k{k}_s{s}.npz one DAS fit: per-pair predictions + correctness on each split,
   │                               loss curve, convergence flag, seconds
   │            shuffle_b{b}.npz   one shuffled-label refit at the pre-registered config
   ▼
cas analyze ► results/<name>/tables/*.csv, figures/*.png, summary.md
                fits.csv            one row per fit (IIA on each split)
                unit_statistics.csv one row per (evaluation unit, scope): each procedure's statistic
                                    and p-values, null sizes, covariates
                decisions.csv       one row per (unit, scope, procedure): statistic, p, decision
                rates_*.csv         α̂ / π̂ with Wilson intervals, upper bounds, binomial tests, Holm
                threshold_sweep.csv, regression.csv, paired_tests.csv, bootstrap_*.csv,
                excluded_*.csv, null_sizes.csv, calibration_statistics.csv
```

An **analysis unit** is (network, hypothesised algorithm, number of fitting pairs). Its truth
class is fixed by construction: `positive`, `negative_random`, `negative_wrong_algorithm` or
`unknown_standard` (see [tasks_and_networks.md](tasks_and_networks.md)). Evaluation units
(`role = eval`) get decisions; calibration units (`role = cal`, different networks, different
seeds) only supply null distributions.

The fit stage stores per-pair correctness with the base-input id and source-input ids of every
pair, so all procedures, nulls, bootstraps and statistics are recomputed offline by `cas
analyze` in under a minute. Nothing in `analyze` trains anything.

## File map

| File | Contents |
|---|---|
| `src/cas/tasks.py` | Tasks (hierarchical equality, distributive law), algorithms (`both_equality`, `and_or_and`, `and_or`), input sampling, counterfactual labels |
| `src/cas/models.py` | `MLP` (with `hidden(x, l)` / `from_hidden(h, l)`), the handcrafted pyvene-tutorial network, save/load |
| `src/cas/das.py` | `interchange` (rotated-basis interchange), `Rotation`, `fit_rotation` (DAS training loop and convergence rule), `predict` |
| `src/cas/train.py` | Standard training, IIT training, task / neuron-aligned interchange accuracy |
| `src/cas/pairs.py` | Disjoint input pools per split, pair generation with usage caps and (label, changed) stratification |
| `src/cas/config.py` | Defaults, YAML loading, deterministic seeds, enumeration of networks and analysis units |
| `src/cas/pipeline.py` | `train` and `fit` stages (resumable), file layout |
| `src/cas/parallel.py` | Process pool (spawn, one torch thread per worker) |
| `src/cas/procedures.py` | Statistics of each procedure, null p-values, decisions |
| `src/cas/stats.py` | Wilson, binomial test, Holm, logistic / Firth / mixed logistic, Cochran's Q, McNemar, binned residuals |
| `src/cas/bootstrap.py` | iid, base-cluster and two-way (pigeonhole) bootstrap of IIA |
| `src/cas/analyze.py` | Loading fit files, all tables and figures, `summary.md` |
| `src/cas/reproduce.py` | `cas reproduce`: handcrafted check, Table 1 and §4.4 analogues, timing |
| `src/cas/cli.py` | The `cas` command |
| `configs/` | `pilot.yaml`, `full.yaml`, `reproduce.yaml` |
| `tests/` | Fast tests (`uv run pytest`), including an end-to-end smoke run |

## Where to check things

* Interchange semantics: `das.interchange` and `tests/test_das.py`.
* Counterfactual labels: `tasks.Algorithm.counterfactual` and `tests/test_tasks.py`.
* Which network is a positive or negative for which hypothesis: `Config.units` and
  `tests/test_config.py`.
* Decision rules: `procedures.py` docstring and `tests/test_procedures.py`.
