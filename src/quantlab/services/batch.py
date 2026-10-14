"""Batch experiments: parameter search on TRAIN+VALIDATION data (03 v2.0.0 §3.3 E).

Rules this module enforces:
  * every trial is registered in the registry BEFORE it runs, so no trial goes unrecorded;
  * workers receive only the dev slice of the dataset (never holdout dates);
  * results always carry the trial count and a deflated Sharpe for the best trial;
  * nothing is adopted automatically: the user confirms a candidate separately.

Cancellation leaves unfinished trials as 'cancelled'; resuming runs exactly the trials whose
latest event is 'registered' or 'cancelled', so trials are neither duplicated nor lost.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from quantlab.core.market_data import MarketData
from quantlab.strategies.spec import load_spec
from quantlab.validation import metrics as M

_W: dict = {}


def grid(space: dict[str, list]) -> list[dict]:
    if not space:
        return [{}]
    names = sorted(space)
    return [dict(zip(names, combo)) for combo in itertools.product(*(space[n] for n in names))]


def _init_worker(spec_path: str, strategy_hash: str, policy: dict, dev: MarketData, qc_fatal: list) -> None:
    loaded = load_spec(spec_path)
    if loaded.strategy_hash != strategy_hash:
        raise RuntimeError("spec changed on disk")
    _W.update(loaded=loaded, policy=policy, dev=dev, qc_fatal=qc_fatal)


def _run_trial(task: tuple[int, dict]) -> tuple[int, dict | None, list | None, str | None]:
    from quantlab.services.lab import run_dev

    idx, params = task
    try:
        p = {**_W["loaded"].default_params, **params}
        m, res, _ = run_dev(_W["loaded"].factory, _W["policy"], _W["dev"], p, _W["qc_fatal"])
        return idx, m, res.returns().tolist(), None
    except Exception as e:  # recorded as a failed trial, never silently dropped
        return idx, None, None, f"{type(e).__name__}: {e}"


def run_search(lab, strategy_hash: str, dataset_version: str, space: dict | None = None, workers: int = 0,
               resume: str | None = None, progress: Callable[[int, int], None] | None = None,
               should_stop: Callable[[], bool] | None = None) -> dict:
    from quantlab.validation.engine import dev_data

    loaded = lab.loaded_strategy(strategy_hash)
    policy_hash, policy = lab.policy_for(loaded)
    reg = lab.registry
    if resume:
        b = reg.batch(resume)
        if b["strategy_hash"] != strategy_hash or b["dataset_version"] != dataset_version:
            raise ValueError("resume: batch belongs to another strategy or dataset")
        batch_id, trials = resume, b["config"]["trials"]
    else:
        trials = grid(space if space is not None else loaded.search_space)
        batch_id = f"B-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
        reg.create_batch(batch_id, strategy_hash, dataset_version,
                         {"trials": trials, "policy_hash": policy_hash, "scope": "dev"})
        for i, p in enumerate(trials):
            reg.trial_event(batch_id, i, "registered", p)
    status = reg.trial_status(batch_id)
    todo = [(i, trials[i]) for i in range(len(trials)) if status.get(i) in ("registered", "cancelled")]

    data, manifest = lab._strategy_symbols_data(loaded, dataset_version)
    dev = dev_data(data, policy)  # holdout dates are cut here, before any worker exists
    spec_path = str(loaded.path)
    init = (spec_path, strategy_hash, policy, dev, manifest.get("qc_fatal", []))
    n_workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    done_before = sum(1 for v in status.values() if v in ("done", "failed"))
    total = len(trials)

    def record(idx, m, rets, err):
        if err is not None:
            reg.trial_event(batch_id, idx, "failed", trials[idx], detail=err)
            return
        eid = reg.record_experiment(
            kind="exploratory", strategy_id=loaded.strategy_id, strategy_version=loaded.version,
            dataset_version=dataset_version, config={"params": m["params"], "policy_hash": policy_hash, "scope": "dev"},
            metrics=m, validation_policy_version=policy_hash, strategy_hash=strategy_hash,
            spec_version=loaded.spec["spec_version"], strategy_origin=loaded.origin, batch_id=batch_id, trial_index=idx)
        reg.trial_event(batch_id, idx, "done", trials[idx], experiment_id=eid)
        returns[idx] = rets

    returns: dict[int, list] = {}
    stopped = False
    completed = done_before
    if n_workers == 1 or len(todo) <= 1:
        _init_worker(*init)
        for task in todo:
            if should_stop and should_stop():
                stopped = True
                break
            record(*_run_trial(task))
            completed += 1
            if progress:
                progress(completed, total)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=init) as pool:
            for out in pool.imap_unordered(_run_trial, todo):
                record(*out)
                completed += 1
                if progress:
                    progress(completed, total)
                if should_stop and should_stop():
                    stopped = True
                    pool.terminate()
                    break
    if stopped:
        st = reg.trial_status(batch_id)
        for i, ev in st.items():
            if ev == "registered":
                reg.trial_event(batch_id, i, "cancelled", trials[i])
    return summarize_batch(lab, batch_id)


def summarize_batch(lab, batch_id: str) -> dict:
    reg = lab.registry
    b = reg.batch(batch_id)
    status = reg.trial_status(batch_id)
    exps = {e["trial_index"]: e for e in reg.experiments() if e["batch_id"] == batch_id}
    rows = []
    for i, p in enumerate(b["config"]["trials"]):
        e = exps.get(i)
        m = e["metrics"] if e else {}
        rows.append({"trial": i, "params": p, "status": status.get(i), "experiment_id": e and e["experiment_id"],
                     "sharpe": m.get("strategy", {}).get("sharpe"), "sharpe_diff": m.get("sharpe_diff"),
                     "cagr": m.get("strategy", {}).get("cagr"), "max_drawdown": m.get("strategy", {}).get("max_drawdown"),
                     "sr_period": m.get("sr_period")})
    strategy = reg.strategy(b["strategy_hash"])
    n_family = reg.count_trials_for_family(strategy["strategy_id"])
    srs = np.array([r["sr_period"] for r in rows if r["sr_period"] is not None and np.isfinite(r["sr_period"])])
    best = max((r for r in rows if r["sharpe_diff"] is not None), key=lambda r: r["sharpe_diff"], default=None)
    dsr = None
    if best is not None and len(srs) > 1:
        # DSR of the best trial: its PSR against the expected max Sharpe of n_family unskilled trials
        sr_star = M.expected_max_sharpe(max(n_family, len(srs)), float(np.var(srs, ddof=1)))
        sr_b, n_obs = best["sr_period"], int(exps[best["trial"]]["metrics"]["strategy"]["sessions"])
        dsr = float(M.stats.norm.cdf((sr_b - sr_star) * np.sqrt(max(n_obs - 1, 1)) / np.sqrt(1 + 0.5 * sr_b**2)))
    return {"batch_id": batch_id, "strategy": f"{strategy['strategy_id']}@{strategy['version']}",
            "total": len(rows), "done": sum(1 for r in rows if r["status"] == "done"),
            "failed": sum(1 for r in rows if r["status"] == "failed"),
            "cancelled": sum(1 for r in rows if r["status"] == "cancelled"),
            "family_trials": n_family, "best": best, "best_dsr": dsr,
            "note": "Best trial is not adopted automatically; confirm a candidate to fix parameters.",
            "rows": rows}
