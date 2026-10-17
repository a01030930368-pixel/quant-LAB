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

from quantlab.analysis.trades import episodes, excess_benchmark, not_trades, trade_stats, without_best_year
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


def run_dev(factory, policy: dict, dev: MarketData, params: dict, qc_fatal: list[str]):
    """One exploratory run on TRAIN+VALIDATION data: metrics, strategy result, benchmark result."""
    strategy = factory(**params)
    res = run_backtest(strategy, dev, _cfg(policy))
    bench = run_backtest(BuyAndHold(policy["benchmark_symbol"]), dev, _cfg(policy))
    rf = _rf(dev, policy)
    m = {"strategy": M.summarize(res.equity, res.trades, rf), "benchmark": M.summarize(bench.equity, bench.trades, rf),
         "window": [str(dev.calendar[0].date()), str(dev.calendar[-1].date())], "params": params,
         "qc_fatal": qc_fatal}
    m["sharpe_diff"] = m["strategy"]["sharpe"] - m["benchmark"]["sharpe"]
    ex = res.returns() - rf.reindex(res.equity.index).fillna(0.0)
    m["sr_period"] = M.sharpe(ex, annualize=False)
    m["trades"] = trade_stats(episodes(res, dev, exclude=not_trades(strategy),
                                         excess_vs=excess_benchmark(strategy)))
    m["without_best_year"] = without_best_year(res.equity)
    return m, res, bench


class Lab:
    def __init__(self, root: str | Path | None = None) -> None:
        self.ws = Workspace.open(root)
        self.registry = Registry(self.ws.registry_path)
        self.data = DataStore(self.ws.root, self.registry)
        self.register_builtin_policies()
        self.copy_example_specs()

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

    def copy_example_specs(self) -> Path:
        """Put the bundled example specs in the work folder, where file dialogs open."""
        dest = self.ws.path("examples")
        dest.mkdir(parents=True, exist_ok=True)
        for f in resources.files("quantlab.builtin").joinpath("specs").iterdir():
            # AUTO-* generator specs belong to the automatic search page, not to the strategy cards
            if f.name.endswith(".qls.yaml") and not f.name.startswith("AUTO-") and not (dest / f.name).exists():
                (dest / f.name).write_bytes(f.read_bytes())
        return dest

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
        m, res, bench = run_dev(loaded.factory, policy, dev, p, manifest.get("qc_fatal", []))
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

    # --- batch experiments (dev data only) -----------------------------------------------
    def auto_search(self):
        from quantlab.services.autosearch import AutoSearch

        return AutoSearch(self)

    def search(self, strategy_hash: str, dataset_version: str, space: dict | None = None, workers: int = 0,
               resume: str | None = None, progress=None, should_stop=None) -> dict:
        from quantlab.services.batch import run_search

        return run_search(self, strategy_hash, dataset_version, space, workers, resume, progress, should_stop)

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
        if out.results:
            self._save_artifacts(out.experiment_id, out.results["strategy"], out.results["benchmark"])
        return out

    # --- one-click flows (desktop home screen) ------------------------------------------
    def has_api_key(self) -> bool:
        return providers.get_api_key("tiingo") is not None

    def strategy_files(self) -> list[Path]:
        """Spec files the user can run: bundled examples plus every imported spec (latest copy per id@version)."""
        found: dict[str, Path] = {}
        for p in sorted(self.ws.path("examples").glob("*.qls.yaml")) + sorted(self.ws.path("strategies").glob("*/*.qls.yaml")):
            try:
                spec = yaml.safe_load(p.read_text(encoding="utf-8"))
                if str(spec["strategy"]["id"]).startswith("AUTO-"):
                    continue  # automatic search groups have their own page
                found.setdefault(f"{spec['strategy']['id']}@{spec['strategy']['version']}", p)
            except Exception:  # an unreadable file is shown nowhere rather than crashing the list
                continue
        return list(found.values())

    def ensure_dataset(self, symbols: list[str], start: str, say=None) -> str:
        """Dataset with today's data for `symbols`: fetch what is missing or not refreshed today
        (when a key is stored), then build. Without a key, existing data (e.g. CSV) is used as is."""
        from datetime import datetime, timezone

        say = say or (lambda s: None)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        key = self.has_api_key()
        for s in symbols:
            recs = self.data.raw_records(s)
            fresh = recs and recs[-1]["fetched_at"][:8] >= today
            if fresh or (recs and not key):
                continue
            if not key:
                raise RuntimeError(f"{s} 가격 자료가 없습니다. 먼저 Tiingo API 키를 저장하세요.")
            say(f"{s} 가격 받는 중")
            self.fetch([s], start)
        say("데이터셋 만드는 중")
        version, manifest = self.build_dataset(symbols)
        if manifest["qc_fatal"]:
            say("품질 검사 치명 항목이 있습니다. 결과는 INVALID로 처리됩니다.")
        return version

    # --- index datasets (S&P 500 stocks, membership, SEC insider filings) ---------------
    def _settings_path(self) -> Path:
        return self.ws.path("settings.json")

    def _settings(self) -> dict:
        p = self._settings_path()
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def set_setting(self, key: str, value) -> None:
        st = self._settings()
        st[key] = value
        self._settings_path().write_text(json.dumps(st, indent=1), encoding="utf-8")

    def sec_email(self) -> str | None:
        return self._settings().get("sec_email") or None

    def set_sec_email(self, email: str) -> None:
        """The SEC asks automated downloads to carry a contact e-mail; it is sent only to sec.gov."""
        from quantlab.data.sec_insider import user_agent

        user_agent(email.strip())  # raises on a malformed address
        st = self._settings()
        st["sec_email"] = email.strip()
        self._settings_path().write_text(json.dumps(st, indent=1), encoding="utf-8")

    def needs_sec_email(self, spec_path: str | Path) -> bool:
        spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
        return "index" in (spec.get("universe") or {}) and self.sec_email() is None

    def ensure_index_dataset(self, base_version: str, start: str, say=None, refresh: bool = False,
                             today: str | None = None) -> str:
        """S&P 500 dataset on top of the SPY/BIL dataset `base_version`: downloads the price bundle
        (once, or again with `refresh`) and the SEC quarters not yet stored, then builds or reuses.
        `today` (YYYYMMDD, UTC; paper trading) also fetches the bundle again when it lacks the latest
        SPY session (at most every few hours), and the current members' earnings filings unless
        fetched that day."""
        from quantlab.data import index_dataset, sec_earnings, sec_insider, sp500

        say = say or (lambda s: None)
        found = sp500.bundles(self.ws.root)
        if today and found:
            base = json.loads((self.data.proc_dir / base_version[:16] / "manifest.json").read_text(encoding="utf-8"))
            last = max(m["last"] for m in base["symbols"].values())
        if refresh or not found or (today and sp500.stale(found[-1], last)):
            found = found + [sp500.download_bundle(self.ws.root, say=say)]
        email = self.sec_email()
        if not email:
            raise RuntimeError("SEC 내부자 거래 자료를 받으려면 연락용 이메일이 필요합니다 (SEC 규칙). "
                               "백테스트 버튼을 다시 누르면 입력 창이 나옵니다.")
        sec_insider.download(self.ws.root, email, say=say)
        sec_records = sec_insider.records(self.ws.root)
        say("회사 목록 정리 중")
        links = index_dataset.issuer_links(found[-1], sec_records)
        ciks = sorted(set(links["issuer_cik"]))
        sec_earnings.download(self.ws.root, email, ciks, say=say)
        if today:
            now = set(sp500.current_members(found[-1]))
            sec_earnings.download(self.ws.root, email, sorted(set(links.loc[links["ticker"].isin(now), "issuer_cik"])),
                                  say=say, refresh_before=today)
        earnings = {c: m for c, m in sec_earnings.records(self.ws.root).items() if c in set(ciks)}
        version, manifest = index_dataset.build(self.data, base_version, found[-1], sec_records, start, say,
                                                earnings_records=earnings)
        if manifest["qc_fatal"]:
            say("품질 검사 치명 항목이 있습니다. 결과는 INVALID로 처리됩니다.")
        return version

    def ensure_spec_dataset(self, loaded: LoadedSpec, policy: dict, say=None, today: str | None = None) -> str:
        symbols, start = loaded.spec["universe"]["symbols"], policy["splits"]["train"][0]
        base = self.ensure_dataset(symbols, start, say)
        if "index" in loaded.spec["universe"]:
            return self.ensure_index_dataset(base, start, say, today=today)
        return base

    def quick_backtest(self, spec_path: str | Path, say=None) -> BacktestSummary:
        """Import (or reuse) a spec, prepare its data, run an exploratory backtest with defaults."""
        say = say or (lambda s: None)
        say("전략 확인 중")
        imp = self.import_strategy(spec_path)
        if imp.state == "IDEA" and imp.problems:
            raise RuntimeError("전략 검사를 통과하지 못했습니다: " + "; ".join(imp.problems))
        _, policy = self.policy_for(imp.loaded)
        version = self.ensure_spec_dataset(imp.loaded, policy, say)
        say("백테스트 실행 중")
        return self.backtest(imp.strategy_hash, version)

    def quick_final(self, spec_path: str | Path, say=None) -> ValidationOutcome:
        """Confirm the default parameters (unless already confirmed) and run the one final evaluation."""
        say = say or (lambda s: None)
        imp = self.import_strategy(spec_path)
        from quantlab.validation.engine import lineage_independence

        problems = lineage_independence(self.registry, imp.strategy_id, imp.version)
        if problems:  # the holdout could not count as evidence: do not spend it
            raise RuntimeError("이 전략은 보류 구간으로 판정할 수 없습니다 (" + "; ".join(problems) + "). "
                               "오늘 이후 새로 쌓이는 데이터(전진 보류 구간)로만 판정합니다.")
        _, policy = self.policy_for(imp.loaded)
        version = self.ensure_spec_dataset(imp.loaded, policy, say)
        if self.registry.confirmation(imp.strategy_hash) is None:
            self.confirm_candidate(imp.strategy_hash, note="desktop one-click")
        say("최종 평가 실행 중 (보류 구간 1회 열람)")
        return self.final_evaluation(imp.strategy_hash, version)

    def paper(self):
        from quantlab.services.paper import Paper

        return Paper(self)

    def factory(self):
        from quantlab.services.factory import Factory

        return Factory(self)

    def holdout_reads(self, strategy_id: str) -> int:
        return sum(1 for r in self.registry.holdout_ledger() if r["strategy_family"] == strategy_id)

    # --- results -------------------------------------------------------------------
    def write_report(self, experiment_id: str) -> Path:
        from quantlab.analysis.report import write_report

        return write_report(self, experiment_id)

    def export_packet(self, experiment_id: str) -> Path:
        from quantlab.analysis.report import export_packet

        return export_packet(self, experiment_id)

    def compare(self, experiment_ids: list[str]) -> pd.DataFrame:
        rows = []
        for eid in experiment_ids:
            e = self.registry.experiment(eid)
            m = e["metrics"]
            src = m.get("holdout") or m.get("dev") or m
            s = src.get("strategy", {})
            rows.append({"experiment": eid, "strategy": f"{e['strategy_id']}@{e['strategy_version']}", "kind": e["kind"],
                         "cagr": s.get("cagr"), "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
                         "sharpe_diff": src.get("sharpe_diff"), "n_trades": s.get("n_trades")})
        return pd.DataFrame(rows)

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
