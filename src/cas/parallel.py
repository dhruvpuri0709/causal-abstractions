"""Run independent jobs in worker processes (spawn context, one torch thread per worker)."""

from __future__ import annotations

import multiprocessing as mp
import traceback

from tqdm import tqdm


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _call(args):
    fn, job = args
    try:
        return True, fn(job)
    except Exception:  # noqa: BLE001 - report every failure, keep the others running
        return False, (job, traceback.format_exc())


def run_jobs(fn, jobs: list, workers: int, desc: str) -> list:
    """Apply ``fn`` (a module-level function) to every job; returns results in completion order.

    Failures are printed at the end and raised as one RuntimeError, after all other jobs ran.
    """
    results, failures = [], []
    if workers <= 1:
        _init_worker()
        it = map(_call, [(fn, j) for j in jobs])
        for ok, res in tqdm(it, total=len(jobs), desc=desc):
            (results if ok else failures).append(res)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            it = pool.imap_unordered(_call, [(fn, j) for j in jobs], chunksize=1)
            for ok, res in tqdm(it, total=len(jobs), desc=desc):
                (results if ok else failures).append(res)
    if failures:
        for job, tb in failures:
            print(f"FAILED job {job}:\n{tb}")
        raise RuntimeError(f"{len(failures)} of {len(jobs)} jobs failed (see above)")
    return results
