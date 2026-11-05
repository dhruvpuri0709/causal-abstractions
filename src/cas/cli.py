"""Command line interface.

    cas reproduce                  Geiger et al. 2024 checks and a DAS timing
    cas plan    --config C         design size (no computation)
    cas train   --config C         build / train networks          (resumable)
    cas fit     --config C         DAS fits and shuffled-null fits (resumable)
    cas analyze --config C         procedures, statistics, tables, figures
    cas run     --config C         train -> fit -> analyze
    cas pilot                      shortcut for `cas run --config configs/pilot.yaml`
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

from .config import Config

PILOT = "configs/pilot.yaml"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def _compute(p, config_default=None, stages=("train", "fit", "analyze")):
    if config_default:
        p.add_argument("--config", default=config_default, help="YAML config (default: %(default)s)")
    else:
        p.add_argument("--config", required=True, help="YAML config")
    if "train" in stages or "fit" in stages:
        p.add_argument("--device", default="cpu", help="cpu (default) or mps")
        p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                       help="worker processes (default: min(8, CPU count) = %(default)s)")
        p.add_argument("--force", action="store_true",
                       help="rebuild outputs whose config fingerprint no longer matches")
    if "analyze" in stages:
        p.add_argument("--allow-incomplete", action="store_true",
                       help="analyze even if some units have missing fits (they are dropped "
                            "and flagged in summary.md)")


def cmd_plan(cfg: Config):
    nets = cfg.networks()
    units = cfg.units()
    n_cfg = {w: len(cfg.das_configs(w)) for w in {u.width for u in units}}
    n_fits = sum(n_cfg[u.width] for u in units)
    n_shuf = sum(cfg["null"]["shuffle_B"] for u in units if u.role == "eval")
    print(f"config {cfg['name']}: {len(nets)} networks "
          f"({sum(n.kind != 'random' for n in nets)} to train), {len(units)} analysis units, "
          f"{n_fits} DAS fits + {n_shuf} shuffled-null fits")
    for w, n in sorted(n_cfg.items()):
        print(f"  width {w}: {n} configs per unit (layers {cfg['das']['layers']} x "
              f"k {cfg.ks(w)} x {cfg['das']['seeds']} seeds)")


def main(argv=None):
    warnings.filterwarnings("ignore", message=".*nnsight.*")
    ap = argparse.ArgumentParser(prog="cas", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce", help="reproduce Geiger et al. 2024 checks; time a DAS fit")
    r.add_argument("--config", default="configs/reproduce.yaml")
    r.add_argument("--device", default="cpu")
    r.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="torch CPU threads (reproduce runs its fits one after another); "
                        "default %(default)s")
    r.add_argument("--widths", default=None,
                   help="comma-separated random-network widths (overrides the config)")
    p = sub.add_parser("plan", help="print the design size")
    p.add_argument("--config", required=True)
    _compute(sub.add_parser("train", help="build/train all networks"), stages=("train",))
    _compute(sub.add_parser("fit", help="run DAS fits and null fits (resumable)"),
             stages=("fit",))
    _compute(sub.add_parser("analyze", help="procedures, statistics, tables and figures"),
             stages=("analyze",))
    _compute(sub.add_parser("run", help="train -> fit -> analyze"))
    _compute(sub.add_parser("pilot", help=f"run all stages on {PILOT}"), config_default=PILOT)
    args = ap.parse_args(argv)

    if args.cmd == "reproduce":
        import torch
        torch.set_num_threads(args.workers)
        from .reproduce import reproduce
        reproduce(args.config, args.device, args.widths)
        return
    cfg = Config.load(args.config)
    if args.cmd == "plan":
        cmd_plan(cfg)
        return
    t0 = time.perf_counter()
    from .pipeline import fit_all, train_all
    timings = {}
    all_stages = args.cmd in ("run", "pilot")
    if args.cmd == "train" or all_stages:
        cmd_plan(cfg)
        t = time.perf_counter()
        train_all(cfg, args.workers, args.device, force=args.force)
        timings["train"] = time.perf_counter() - t
    if args.cmd == "fit" or all_stages:
        t = time.perf_counter()
        fit_all(cfg, args.workers, args.device, force=args.force)
        timings["fit"] = time.perf_counter() - t
    if args.cmd == "analyze" or all_stages:
        from .analyze import analyze
        t = time.perf_counter()
        analyze(cfg, allow_incomplete=args.allow_incomplete)
        timings["analyze"] = time.perf_counter() - t
    msg = ", ".join(f"{k} {v:.0f} s" for k, v in timings.items())
    print(f"done in {time.perf_counter() - t0:.0f} s ({msg})")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    workers = getattr(args, "workers", "-")
    with open(cfg.out_dir / "timings.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {args.cmd} workers={workers} "
                f"device={getattr(args, 'device', '-')}: {msg}\n")


if __name__ == "__main__":
    main()
