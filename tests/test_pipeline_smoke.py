"""End to end on a minuscule design: train -> fit -> analyze runs and writes its outputs."""

import pandas as pd

from cas.analyze import analyze
from cas.config import Config
from cas.pipeline import fit_all, train_all


def test_pipeline_end_to_end(tmp_path):
    cfg = Config.from_dict({
        "name": "smoke", "seed": 3, "results_dir": str(tmp_path),
        "tasks": {"hierarchical_equality": {"widths": [8]},
                  "distributive_law": {"widths": [8]}},
        "networks": {"eval": {"iit": 1, "random": 1, "standard": 1},
                     "calibration": {"random": 2, "iit": 2}},
        "train": {"standard": {"max_steps": 200}, "iit": {"max_steps": 200}},
        "das": {"layers": [1], "k_fracs": [0.25], "seeds": 1, "n_fit": [64],
                "n_select": 64, "n_test": 64, "fit": {"max_steps": 100}},
        "null": {"shuffle_B": 2},
        "analysis": {"bootstrap_B": 50},
    })
    train_all(cfg, workers=1, device="cpu")
    fit_all(cfg, workers=1, device="cpu")
    n_files = len(list((tmp_path / "smoke" / "fits").rglob("*.npz")))
    fit_all(cfg, workers=1, device="cpu")  # resumable: nothing new
    assert len(list((tmp_path / "smoke" / "fits").rglob("*.npz"))) == n_files
    summary = analyze(cfg)
    assert summary.exists()
    dec = pd.read_csv(tmp_path / "smoke" / "tables" / "decisions.csv")
    assert set(dec.truth_class) <= {"positive", "negative_random", "negative_wrong_algorithm",
                                    "unknown_standard"}
    assert (tmp_path / "smoke" / "tables" / "bootstrap_widths.csv").exists()
