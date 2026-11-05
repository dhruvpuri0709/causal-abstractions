# Causal Abstraction Statistics

CS 7200 project: how often does certifying an algorithm when the interchange intervention
accuracy (IIA) from distributed alignment search (DAS) is at least c give a false positive,
and do held-out evaluation, null calibration and selection correction fix it? See
[`docs/proposal.tex`](docs/proposal.tex).

## Layout

```
docs/
  proposal.tex, references.bib   proposal and bibliography
  papers/                        PDFs of the cited works (see papers/README.md)
external/                        git submodules, pinned, read-only reference code
  pyvene/                        stanfordnlp/pyvene: DAS tutorial on hierarchical equality
  non-linear-representation-dilemma/   Sutter et al. 2025: standalone DAS for MLPs, hidden-size sweeps
  InterchangeInterventions/      Geiger et al.: IIT trainer and equality causal model
```

`pyvene` is installed from PyPI, pinned in `pyproject.toml`. The submodules are for reading and
reproducing published results; our own code imports `pyvene` and does not import from
`external/`. Neither research repo has a license, so we reference them as submodules rather than
copying their code into ours.

## Setup

```bash
git clone --recurse-submodules git@github.com:nicholscrawford/causal-abstraction-statistics.git
cd causal-abstraction-statistics
uv sync
```

If you cloned without `--recurse-submodules`, run `git submodule update --init`.
