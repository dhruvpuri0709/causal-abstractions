"""Command line interface: ``cas reproduce | plan | train | fit | analyze | pilot``."""

from __future__ import annotations

import argparse
import time
import warnings

from .config import Config

PILOT = "configs/pilot.yaml"


def _common(p):
    p.add_argument("--config", default=PILOT, help="YAML config (default: %(default)s)")
    p.add_argument("--device", default="cpu", help="cpu (default) or mps")
    p.add_argument("--workers", type=int, default=8, help="worker processes (default: 8)")


def cmd_plan(cfg: Config, args):
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
    ap = argparse.ArgumentParser(prog="cas", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce", help="reproduce Geiger et al. 2024 checks; time a DAS fit")
    r.add_argument("--config", default="configs/reproduce.yaml")
    r.add_argument("--device", default="cpu")
    r.add_argument("--widths", default=None,
                   help="comma-separated random-network widths (overrides the config)")
    for name, help_ in (("plan", "print the design size"),
                        ("train", "build/train all networks"),
                        ("fit", "run DAS fits and null fits (resumable)"),
                        ("analyze", "procedures, statistics, tables and figures"),
                        ("pilot", "train -> fit -> analyze")):
        _common(sub.add_parser(name, help=help_))
    args = ap.parse_args(argv)

    if args.cmd == "reproduce":
        from .reproduce import reproduce
        reproduce(args.config, args.device, args.widths)
        return
    cfg = Config.load(args.config)
    if args.device != "cpu" and args.workers > 1:
        print(f"note: device {args.device} with {args.workers} workers; each worker uses it")
    t0 = time.perf_counter()
    if args.cmd == "plan":
        cmd_plan(cfg, args)
        return
    from .pipeline import fit_all, train_all
    timings = {}
    if args.cmd in ("train", "pilot"):
        cmd_plan(cfg, args)
        t = time.perf_counter()
        train_all(cfg, args.workers, args.device)
        timings["train"] = time.perf_counter() - t
    if args.cmd in ("fit", "pilot"):
        t = time.perf_counter()
        fit_all(cfg, args.workers, args.device)
        timings["fit"] = time.perf_counter() - t
    if args.cmd in ("analyze", "pilot"):
        from .analyze import analyze
        t = time.perf_counter()
        analyze(cfg)
        timings["analyze"] = time.perf_counter() - t
    msg = ", ".join(f"{k} {v:.0f} s" for k, v in timings.items())
    print(f"done in {time.perf_counter() - t0:.0f} s ({msg})")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_dir / "timings.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {args.cmd} workers={args.workers} "
                f"device={args.device}: {msg}\n")


if __name__ == "__main__":
    main()
