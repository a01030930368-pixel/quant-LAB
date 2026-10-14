"""End-to-end self-test on synthetic data, offline (03 v2.0.0 §8.1 independence test).

Runs the whole user flow in a throwaway work folder: CSV import → dataset →
spec import → exploratory backtest → parameter search → candidate confirmation
→ final evaluation → report → result packet. Used by the test suite, by
`quantlab selftest` and by the packaged app's `--selftest` flag in CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from quantlab.core.synthetic import regime_market
from quantlab.data.quality import exchange_sessions
from quantlab.services.lab import Lab, builtin_path

VERDICTS = {"PASS", "HOLD", "FAIL", "INVALID"}


def write_synthetic_csvs(folder: Path, seed: int = 0) -> dict[str, Path]:
    sessions = exchange_sessions("2007-06-01", "2024-12-31")
    data = regime_market(seed, index=sessions)
    out = {}
    for sym, df in data.frames.items():
        p = folder / f"{sym}_synthetic.csv"
        df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
        out[sym] = p
    return out


def write_synthetic_group(folder: Path, seed: int = 1) -> dict[str, Path]:
    """Random-walk prices for the AUTO-REAL group (the self-test's SPY and BIL come from above)."""
    import numpy as np

    from quantlab.core.synthetic import frame

    sessions = exchange_sessions("2007-06-01", "2024-12-31")
    rng = np.random.default_rng(seed)
    out = {}
    for sym in ("TLT", "IEF", "LQD", "GLD", "SLV", "DBC", "VNQ"):
        close = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(sessions))))
        opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, len(sessions)))
        df = frame(opn, close)
        df.index = sessions
        p = folder / f"{sym}_synthetic.csv"
        df.rename_axis("date").reset_index().to_csv(p, index=False, date_format="%Y-%m-%d")
        out[sym] = p
    return out


def run_selftest(root: str | Path | None = None, log=print) -> dict:
    tmp = None
    if root is None:
        tmp = tempfile.TemporaryDirectory(prefix="quantlab-selftest-")
        root = tmp.name
    root = Path(root)
    lab = Lab(root / "QuantLab")
    try:
        from quantlab.core.app_info import app_code_hash, app_version

        log(f"QUANT LAB {app_version()} code {app_code_hash()[:16]}")
        csvs = write_synthetic_csvs(root)
        for sym, p in csvs.items():
            rec = lab.import_csv(sym, p)
            log(f"imported {sym}: {len(rec.qc.fatal)} fatal QC findings")
            if rec.qc.fatal:
                raise AssertionError(f"synthetic data failed QC: {rec.qc.fatal}")
        version, _ = lab.build_dataset(list(csvs))
        log(f"dataset {version[:12]}")
        imp = lab.import_strategy(builtin_path("specs", "TT-SPY-SMA.qls.yaml"))
        log(f"strategy {imp.strategy_id}@{imp.version}: {imp.state} {imp.problems}")
        if imp.state != "RESEARCH":
            raise AssertionError(f"baseline import failed: {imp.problems}")
        bt = lab.backtest(imp.strategy_hash, version)
        log(f"exploratory backtest {bt.experiment_id}: sharpe diff {bt.metrics['sharpe_diff']:.3f} "
            f"on {bt.metrics['window'][0]}..{bt.metrics['window'][1]}")
        steps = {"dataset": version, "strategy_hash": imp.strategy_hash, "backtest": bt.experiment_id}
        if hasattr(lab, "search"):
            # two worker processes: also proves spawned workers start in a frozen build
            batch = lab.search(imp.strategy_hash, version, workers=2)
            log(f"best trial {batch['best']['params']} sharpe diff {batch['best']['sharpe_diff']:.3f}, "
                f"DSR {batch['best_dsr']:.3f} over {batch['family_trials']} trials (not adopted automatically)")
            log(f"search {batch['batch_id']}: {batch['done']} trials done")
            steps["batch"] = batch["batch_id"]
        if hasattr(lab, "auto_search"):
            # automatic search: a few combinations on synthetic bond/gold/REIT ETFs, two worker processes
            for sym, path in write_synthetic_group(root).items():
                lab.import_csv(sym, path)
            auto = lab.auto_search().run("AUTO-REAL", workers=2, limit=4)
            log(f"automatic search AUTO-REAL: {auto['done']} done, {auto['failed']} failed, "
                f"{auto['family_trials']} trials counted, t needed {auto['t_required']:.2f}")
            if auto["failed"] or auto["done"] < 4 or any(r["strategy_family"] == "AUTO-REAL"
                                                          for r in lab.registry.holdout_ledger()):
                raise AssertionError(f"automatic search check failed: {auto['done']} done, {auto['failed']} failed")
            steps["auto"] = auto["done"]
        lab.confirm_candidate(imp.strategy_hash, note="selftest")
        out = lab.final_evaluation(imp.strategy_hash, version)
        log(f"final evaluation {out.experiment_id}: {out.overall} "
            + "; ".join(f"{g.gate}={g.verdict}" for g in out.gates))
        if out.overall not in VERDICTS:
            raise AssertionError(out.overall)
        steps.update(final=out.experiment_id, verdict=out.overall)
        if hasattr(lab, "write_report"):
            rep = lab.write_report(out.experiment_id)
            log(f"report {rep}")
            steps["report"] = str(rep)
        if hasattr(lab, "export_packet"):
            pk = lab.export_packet(out.experiment_id)
            log(f"packet {pk}")
            steps["packet"] = str(pk)
        reproduced = lab.reproduce(bt.experiment_id)
        if not reproduced["identical"]:
            raise AssertionError("exploratory backtest did not reproduce")
        log("reproduction: identical")
        steps["ok"] = True
        return steps
    finally:
        lab.close()
        if tmp is not None:
            tmp.cleanup()
