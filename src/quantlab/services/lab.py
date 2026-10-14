"""Use-case API shared by the GUI and the CLI (03 v2.0.0 §3.2).

Screens and commands call only this layer. Verdict logic stays in
`quantlab.validation`; this layer wires workspace, registry, data store and
engine together and enforces who may see which data:

  * exploratory backtests and searches receive TRAIN+VALIDATION data only;
  * only `final_evaluation` reads the holdout, and only for a confirmed candidate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantlab.backtesting.engine import BacktestResult, run_backtest
from quantlab.core.hashing import stable_hash
from quantlab.core.market_data import MarketData
from quantlab.core.workspace import Workspace
from quantlab.data import providers
from quantlab.data.store import DataStore, RawRecord
from quantlab.registry.store import Registry
from quantlab.strategies.importer import ImportResult, import_spec
from quantlab.strategies.spec import LoadedSpec, load_spec
from quantlab.validation import metrics as M
from quantlab.validation.engine import (BuyAndHold, ValidationOutcome, _cfg, _rf, dev_data, validate)


def builtin_path(*parts: str) -> Path:
    return Path(str(resources.files("quantlab.builtin").joinpath(*parts)))


@dataclass
class BacktestSummary:
    experiment_id: str
    metrics: dict
    result: BacktestResult
    benchmark: BacktestResult


class Lab:
    def __init__(self, root: str | Path | None = None) -> None:
        self.ws = Workspace.open(root)
        self.registry = Registry(self.ws.registry_path)
        self.data = DataStore(self.ws.root, self.registry)
        self.register_builtin_policies()

    def close(self) -> None:
        self.registry.close()

    # --- policies ----------------------------------------------------------------
    def register_policy_file(self, path: str | Path) -> str:
        policy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        h = self.registry.register_policy(policy)
        dest = self.ws.path("policies", f"{policy['policy_id']}_v{policy['version']}_{h[:12]}.yaml")
        if not dest.exists():
            dest.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        return h

    def register_builtin_policies(self) -> None:
        for p in sorted(builtin_path("policies").glob("*.yaml")):
            self.register_policy_file(p)

    def policy_for(self, loaded: LoadedSpec) -> tuple[str, dict]:
        """`validation.policy` is 'name_vN' (exact version) or 'name' (latest version)."""
        name = str(loaded.spec["validation"]["policy"])
        m = re.fullmatch(r"(.+)_v(\d+)", name)
        if m:
            for row in self.registry.policies():
                if row["policy_id"] == m.group(1) and row["version"] == int(m.group(2)):
                    return row["policy_hash"], self.registry.get_policy(row["policy_hash"])
            raise KeyError(f"policy {name} is not registered")
        return self.registry.policy_by_name(name)

    # --- data --------------------------------------------------------------------
    def set_api_key(self, key: str, provider: str = "tiingo") -> None:
        providers.set_api_key(key, provider)

    def fetch(self, symbols: list[str], start: str, end: str | None = None) -> list[RawRecord]:
        out = []
        for s in symbols:
            frame, raw = providers.fetch_tiingo(s, start, end)
            out.append(self.data.ingest(s, frame, raw, {"provider": "tiingo", "start": start, "end": end}))
        return out

    def import_csv(self, symbol: str, path: str | Path, sessions: pd.DatetimeIndex | None = None) -> RawRecord:
        frame = providers.read_csv_prices(path)
        return self.data.ingest(symbol, frame, Path(path), {"provider": "csv", "file": Path(path).name}, sessions=sessions)

    def build_dataset(self, symbols: list[str]) -> tuple[str, dict]:
        return self.data.build_dataset(symbols)

    def load_dataset(self, version: str) -> tuple[MarketData, dict]:
        return self.data.load_dataset(version)

    # --- strategies --------------------------------------------------------------
    def import_strategy(self, path: str | Path) -> ImportResult:
        return import_spec(path, self.registry, self.ws.path("strategies"))

    def loaded_strategy(self, strategy_hash: str) -> LoadedSpec:
        row = self.registry.strategy(strategy_hash)
        files = list(self.ws.path("strategies", strategy_hash[:16]).glob("*.qls.yaml"))
        if not files:
            raise FileNotFoundError(f"spec file for {row['strategy_id']}@{row['version']} missing from work folder")
        loaded = load_spec(files[0])
        if loaded.strategy_hash != strategy_hash:
            raise ValueError("stored spec no longer matches its registered hash")
        return loaded

    def _strategy_symbols_data(self, loaded: LoadedSpec, version: str) -> tuple[MarketData, dict]:
        data, manifest = self.load_dataset(version)
        missing = set(loaded.spec["universe"]["symbols"]) - set(data.symbols)
        if missing:
            raise ValueError(f"dataset lacks {sorted(missing)} required by {loaded.strategy_id}")
        return data, manifest

    # --- exploratory backtest (dev data only) ---------------------------------------
    def backtest(self, strategy_hash: str, dataset_version: str, params: dict | None = None,
                 batch_id: str | None = None, trial_index: int | None = None) -> BacktestSummary:
        loaded = self.loaded_strategy(strategy_hash)
        policy_hash, policy = self.policy_for(loaded)
        data, manifest = self._strategy_symbols_data(loaded, dataset_version)
        dev = dev_data(data, policy)  # the holdout never reaches exploratory runs
        p = {**loaded.default_params, **(params or {})}
        res = run_backtest(loaded.factory(**p), dev, _cfg(policy))
        bench = run_backtest(BuyAndHold(policy["benchmark_symbol"]), dev, _cfg(policy))
        rf = _rf(dev, policy)
        m = {"strategy": M.summarize(res.equity, res.trades, rf), "benchmark": M.summarize(bench.equity, bench.trades, rf),
             "window": [str(dev.calendar[0].date()), str(dev.calendar[-1].date())], "params": p,
             "qc_fatal": manifest.get("qc_fatal", [])}
        m["sharpe_diff"] = m["strategy"]["sharpe"] - m["benchmark"]["sharpe"]
        eid = self.registry.record_experiment(
            kind="exploratory", strategy_id=loaded.strategy_id, strategy_version=loaded.version,
            dataset_version=dataset_version, config={"params": p, "policy_hash": policy_hash, "scope": "dev"},
            metrics=m, validation_policy_version=policy_hash, strategy_hash=strategy_hash,
            spec_version=loaded.spec["spec_version"], strategy_origin=loaded.origin, batch_id=batch_id,
            trial_index=trial_index)
        self._save_artifacts(eid, res, bench)
        return BacktestSummary(eid, m, res, bench)

    def _save_artifacts(self, eid: str, res: BacktestResult, bench: BacktestResult) -> None:
        d = self.ws.path("artifacts", eid)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"strategy": res.equity, "benchmark": bench.equity}).to_parquet(d / "equity.parquet")
        res.trades.to_parquet(d / "trades.parquet")

    def artifacts(self, eid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = self.ws.path("artifacts", eid)
        return pd.read_parquet(d / "equity.parquet"), pd.read_parquet(d / "trades.parquet")

    # --- candidate confirmation and final evaluation -----------------------------------
    def confirm_candidate(self, strategy_hash: str, params: dict | None = None, note: str = "") -> dict:
        loaded = self.loaded_strategy(strategy_hash)
        policy_hash, _ = self.policy_for(loaded)
        p = {**loaded.default_params, **(params or {})}
        self.registry.confirm_candidate(strategy_hash, p, policy_hash, note)
        self.registry.record_state_change(loaded.strategy_id, "VALIDATION",
                                          f"candidate confirmed: params {json.dumps(p, sort_keys=True)}, policy {policy_hash[:12]}")
        return p

    def final_evaluation(self, strategy_hash: str, dataset_version: str) -> ValidationOutcome:
        conf = self.registry.confirmation(strategy_hash)
        if conf is None:
            raise PermissionError("confirm the candidate (parameters and policy) before the final evaluation")
        loaded = self.loaded_strategy(strategy_hash)
        data, manifest = self._strategy_symbols_data(loaded, dataset_version)
        out = validate(loaded.factory, conf["params"], data, conf["policy_hash"], self.registry,
                       dataset_version=dataset_version, qc_fatal=manifest.get("qc_fatal", []),
                       strategy_hash=strategy_hash, spec_version=loaded.spec["spec_version"], origin=loaded.origin)
        return out

    # --- reproduction --------------------------------------------------------------
    def reproduce(self, experiment_id: str) -> dict:
        """Re-run an exploratory experiment from its record; compare result hashes."""
        e = self.registry.experiment(experiment_id)
        if e["kind"] != "exploratory":
            raise ValueError("formal evaluations are not re-run (the holdout is read once); "
                             "their inputs are fully recorded for audit instead")
        again = self.backtest(e["strategy_hash"], e["dataset_version"], e["config"]["params"])
        same = stable_hash(again.metrics) == e["result_hash"]
        return {"experiment_id": experiment_id, "rerun_id": again.experiment_id, "identical": same,
                "app_code_changed": again.experiment_id and self.registry.experiment(again.experiment_id)["app_code_hash"] != e["app_code_hash"]}
