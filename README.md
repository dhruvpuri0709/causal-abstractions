# Causal Abstraction Statistics

CS 7200 project: how often does certifying an algorithm when the interchange intervention
accuracy (IIA) from distributed alignment search (DAS) is at least c give a false positive,
and do held-out evaluation, null calibration and selection correction fix it? See
[`docs/proposal.tex`](docs/proposal.tex).

## Quickstart

```bash
uv sync
uv run pytest                     # fast tests (~10 s); `uv run pytest -m slow` for the pyvene cross-check
uv run cas reproduce              # ~1 min: Geiger et al. 2024 checks -> results/reproduce/report.md
uv run cas pilot                  # ~10 min: train -> fit -> analyze on configs/pilot.yaml
```

| Command | What it does | Output |
|---|---|---|
| `cas reproduce` | Handcrafted-network check (IIA = 1), Table 1 analogue (task-trained and IIT-trained hierarchical-equality MLP), §4.4 analogue (random networks vs width), timing of one DAS fit | `results/reproduce/report.md` |
| `cas plan --config C` | Prints the number of networks, analysis units and fits | stdout |
| `cas train --config C` | Builds random, task-trained and IIT-trained networks (skips existing) | `results/<name>/networks/` |
| `cas fit --config C` | DAS fits for every unit and configuration, plus shuffled-label null refits; resumable | `results/<name>/fits/<unit>/*.npz` (per-pair records) |
| `cas analyze --config C` | Procedures, nulls, α̂/π̂ with Wilson intervals, binomial tests, logistic/Firth regression, McNemar, bootstrap widths | `results/<name>/summary.md`, `tables/*.csv`, `figures/*.png` |
| `cas pilot` | `train`, `fit`, `analyze` on `configs/pilot.yaml` | as above, under `results/pilot/` |

All commands take `--workers N` (default 8, process pool) and `--device cpu|mps` (default cpu).
The same config gives the same results. Code documentation is in [`docs/code/`](docs/code/):
[overview](docs/code/overview.md), [tasks and networks](docs/code/tasks_and_networks.md),
[DAS](docs/code/das.md), [procedures and statistics](docs/code/procedures_and_statistics.md),
[running](docs/code/running.md), [deviations from the proposal](docs/code/deviations_from_proposal.md).

## Layout

```
configs/                         pilot.yaml, full.yaml, reproduce.yaml
docs/
  proposal.tex, references.bib   proposal and bibliography
  papers/                        PDFs of the cited works (see papers/README.md)
  code/                          documentation of the code
src/cas/                         the `cas` package (see docs/code/overview.md)
tests/                           pytest suite
external/                        git submodules, pinned, read-only reference code
  pyvene/                        stanfordnlp/pyvene: DAS tutorial on hierarchical equality
  non-linear-representation-dilemma/   Sutter et al. 2025: standalone DAS for MLPs, hidden-size sweeps
  InterchangeInterventions/      Geiger et al.: IIT trainer and equality causal model
```

`pyvene` is installed from PyPI, pinned in `pyproject.toml`. The submodules are for reading and
reproducing published results; our own code does not import from `external/`
(pyvene is used only in an optional slow cross-check test). Neither research repo has a license, so we reference them as submodules rather than
copying their code into ours.

## Setup

```bash
git clone --recurse-submodules git@github.com:nicholscrawford/causal-abstraction-statistics.git
cd causal-abstraction-statistics
uv sync
```

If you cloned without `--recurse-submodules`, run `git submodule update --init`.
