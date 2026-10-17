"""모의 투자 (paper trading) on the forward holdout (D-030).

A frozen rule is run every day on data that did not exist when it was written: the sessions from its
policy's `forward_holdout_start` on. No broker is involved. Each refresh fetches today's data, runs
the rule with the policy's costs from that date (orders fill at the next open, as in every
backtest), and keeps an append-only record in the work folder:

    paper/<strategy>@<version>/ledger.json          every fill, as first computed, and when it was seen
    paper/<strategy>@<version>/days/<as-of>.json    each refresh: equity, holdings, next open's orders

The rule needs some history for its indicators, so the data starts `WARMUP_SESSIONS` before the
start date; the strategy is not asked for a decision before the start and nothing is computed or
shown about those sessions (they belong to the unread 2020+ holdout).

A later refresh recomputes everything from the data. When a revised price or a late filing changes
a fill already recorded, the record keeps the first version and the page lists the difference.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.analysis.trades import episodes, excess_benchmark, not_trades
from quantlab.backtesting.engine import run_backtest
from quantlab.strategies.base import MarketView, Strategy
from quantlab.validation import metrics as M
from quantlab.validation.engine import BuyAndHold, _cfg

WARMUP_SESSIONS = 300     # rolling windows of the rules (200-day average, 52-week high) fit in this
MIN_PRICED_SHARE = 0.90   # an index session counts once this share of its members has a close


class FromDate(Strategy):
    """Runs `inner` from `start` on; before it the account stays in cash and `inner` is never asked."""

    def __init__(self, inner: Strategy, start: pd.Timestamp) -> None:
        super().__init__(**inner.params)
        self.inner, self.start = inner, pd.Timestamp(start)
        self.strategy_id, self.version = inner.strategy_id, inner.version

    def __getattr__(self, name):  # park_symbol, excess_benchmark, ... of the rule
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def describe(self) -> dict:
        return {"forward_from": str(self.start.date()), **self.inner.describe()}

    def on_bar(self, view: MarketView):
        if view.now < self.start:
            return None
        return self.inner.on_bar(view)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def priced_until(data) -> pd.Timestamp:
    """Last session with prices for most index members (the stock bundle can lag the SPY data)."""
    if data.members is None:
        return data.calendar[-1]
    close = data.field("close")
    m = data.members.reindex(columns=close.columns, fill_value=False)
    share = (m & close.notna()).sum(axis=1) / m.sum(axis=1).replace(0, np.nan)
    ok = share[share >= MIN_PRICED_SHARE]
    if ok.empty:
        raise RuntimeError("S&P 500 종목 가격이 거의 없는 자료입니다")
    return ok.index[-1]


class Paper:
    def __init__(self, lab) -> None:
        self.lab = lab

    # --- which rules ---------------------------------------------------------------
    def candidates(self) -> list[dict]:
        """Spec files whose policy names a forward start: id, version, name, path, start."""
        import yaml

        out = []
        for p in self.lab.strategy_files():
            try:
                spec = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
                from quantlab.strategies.spec import load_spec

                _, policy = self.lab.policy_for(load_spec(p))
            except Exception:  # an unreadable spec or an unregistered policy is not offered
                continue
            start = policy.get("splits", {}).get("forward_holdout_start")
            if start:
                s = spec["strategy"]
                out.append({"id": s["id"], "version": str(s["version"]), "name": s.get("name", s["id"]),
                            "path": str(p), "start": str(start)})
        return sorted(out, key=lambda r: (r["id"] != "GAP-EARN-SP500", r["id"]))

    def folder(self, sid: str, version: str) -> Path:
        return self.lab.ws.path("paper", f"{sid}@{version}")

    def latest(self, sid: str, version: str) -> dict | None:
        days = sorted((self.folder(sid, version) / "days").glob("*.json"))
        return json.loads(days[-1].read_text(encoding="utf-8")) if days else None

    def days(self, sid: str, version: str) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.folder(sid, version) / "days").glob("*.json"))]

    # --- one refresh ---------------------------------------------------------------------
    def refresh(self, spec_path: str | Path, say=None, today: str | None = None) -> dict:
        say = say or (lambda s: None)
        lab = self.lab
        imp = lab.import_strategy(spec_path)
        if imp.state == "IDEA" and imp.problems:
            raise RuntimeError("전략 검사를 통과하지 못했습니다: " + "; ".join(imp.problems))
        loaded = imp.loaded
        policy_hash, policy = lab.policy_for(loaded)
        start = pd.Timestamp(policy["splits"]["forward_holdout_start"])
        today = today or datetime.now(timezone.utc).strftime("%Y%m%d")
        before = self._bundle_dirs()
        version = lab.ensure_spec_dataset(loaded, policy, say, today=today)
        say("모의 투자 계산 중")
        data, manifest = lab._strategy_symbols_data(loaded, version)
        asof = priced_until(data)
        cal = data.calendar[data.calendar <= asof]
        snap = {"strategy_id": loaded.strategy_id, "version": loaded.version, "strategy_hash": imp.strategy_hash,
                "policy_hash": policy_hash, "dataset_version": version, "start": _d(start), "asof": _d(asof),
                "refreshed_at": _now(), "params": loaded.default_params, "qc_fatal": manifest.get("qc_fatal", []),
                "qc_warnings": manifest.get("qc_warnings", [])}
        live = cal[cal >= start]
        folder = self.folder(loaded.strategy_id, loaded.version)
        if len(live) == 0:
            first = start
            snap.update(started=False, note=f"{_d(first)} 이후 첫 거래일 장 마감 뒤에 첫 신호가 나와요")
            self._prune(before, version)
            return snap
        lo = max(0, int(np.searchsorted(cal, live[0])) - WARMUP_SESSIONS)
        data = data.slice(cal[lo], asof)
        inner = loaded.factory(**loaded.default_params)
        strat = FromDate(inner, start)
        cfg = _cfg(policy)
        res = run_backtest(strat, data, cfg)
        fwd = data.slice(live[0], asof)
        bench = run_backtest(BuyAndHold(policy["benchmark_symbol"]), fwd, cfg)
        # the decision at the last close, filled at the next open
        nxt = inner.on_bar(MarketView(data, len(data.calendar) - 1)) or {}
        snap.update(started=True, **self._summary(res, bench, data, live[0], inner, nxt, policy))
        snap["ledger"] = self._ledger(folder, res, live[0], asof)
        d = folder / "days"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{_d(asof)}.json").write_text(json.dumps(snap, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        self._prune(before, version)
        return snap

    def _summary(self, res, bench, data, first, inner, nxt, policy) -> dict:
        eq = res.equity.loc[first:]
        beq = bench.equity
        park = getattr(inner, "park_symbol", None)
        sh = res.shares.iloc[-1]
        closes = data.field("close").ffill().iloc[-1]
        equity = float(eq.iloc[-1])
        held = getattr(inner, "held", {})
        cols = data.calendar
        holdings = []
        tr = res.trades
        for sym in sh.index[sh.to_numpy() > 1e-12]:
            buys = tr[(tr["symbol"] == sym) & (tr["qty"] > 0)]
            last_buy = buys.iloc[-1] if len(buys) else None
            row = {"symbol": sym, "shares": float(sh[sym]), "value": float(sh[sym] * closes[sym]),
                   "weight": float(sh[sym] * closes[sym] / equity), "park": sym == park}
            if last_buy is not None and sym != park:
                entry = pd.Timestamp(last_buy["date"])
                row.update(entry_date=_d(entry), entry_open=float(last_buy["open"]),
                           ret=float(closes[sym] / last_buy["open"] - 1),
                           sessions=int((cols > entry).sum()) + 1)
                spy = data.frames.get("SPY")
                if spy is not None and np.isfinite(spy["open"].get(entry, np.nan)):
                    row["spy_ret"] = float(spy["close"].ffill().iloc[-1] / spy["open"][entry] - 1)
            holdings.append(row)
        hold = int(inner.params.get("hold", 0)) or None
        orders = []
        for sym, w in nxt.items():
            if sym == park:
                orders.append({"side": "park", "symbol": sym, "weight": float(w)})
            elif w <= 1e-12:
                orders.append({"side": "sell", "symbol": sym, "weight": 0.0})
            else:
                orders.append({"side": "buy", "symbol": sym, "weight": float(w)})
        ep = episodes(res, data, start=first, exclude=not_trades(inner), excess_vs=excess_benchmark(inner))
        done = ep[~ep["open_at_end"].astype(bool)]
        closed = []
        for _, r in done.iterrows():
            closed.append({k: (None if pd.isna(v) else (_d(v) if isinstance(v, pd.Timestamp) else v))
                           for k, v in r.items() if k in ("symbol", "entry", "exit", "sessions", "gross", "net", "bench", "excess")})
        dd = float(M.max_drawdown(eq)) if len(eq) > 1 else 0.0
        bdd = float(M.max_drawdown(beq)) if len(beq) > 1 else 0.0
        return {"equity": equity, "initial": float(policy["initial_cash"]),
                "ret": equity / float(policy["initial_cash"]) - 1, "spy_ret": float(beq.iloc[-1] / beq.iloc[0] - 1),
                "mdd": dd, "spy_mdd": bdd, "sessions": int(len(eq)), "hold": hold,
                "curve": [[_d(i), float(v), float(beq.get(i, np.nan))] for i, v in eq.items()],
                "holdings": holdings, "orders": orders, "closed": closed}

    def _ledger(self, folder: Path, res, first, asof) -> dict:
        """Append-only record of fills; returns what changed against it."""
        path = folder / "ledger.json"
        folder.mkdir(parents=True, exist_ok=True)
        led = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"asof": None, "fills": []}
        tr = res.trades[res.trades["date"] >= first]
        now = [{"date": _d(r["date"]), "symbol": r["symbol"], "side": "buy" if r["qty"] > 0 else "sell",
                "qty": float(r["qty"]), "price": float(r["fill_price"])} for _, r in tr.iterrows()]
        key = lambda f: (f["date"], f["symbol"], f["side"])  # noqa: E731
        seen_until = led["asof"]
        old = {key(f): f for f in led["fills"]}
        new = {key(f): f for f in now}
        changed = []
        if seen_until:
            for k, f in old.items():
                g = new.get(k)
                if g is None:
                    changed.append({"kind": "missing", **f})
                elif abs(g["qty"] - f["qty"]) > 1e-9 or abs(g["price"] - f["price"]) > 1e-6 * max(1.0, f["price"]):
                    changed.append({"kind": "different", **f, "now_qty": g["qty"], "now_price": g["price"]})
            for k, g in new.items():
                if k not in old and g["date"] <= seen_until:
                    changed.append({"kind": "extra", **g})
        if not seen_until or _d(asof) > seen_until:
            stamp = _now()
            led["fills"] += [{**g, "seen_at": stamp} for k, g in new.items() if k not in old
                             and (not seen_until or g["date"] > seen_until)]
            led["asof"] = _d(asof)
            path.write_text(json.dumps(led, indent=1), encoding="utf-8")
        return {"fills": len(led["fills"]), "changed": changed, "asof": led["asof"]}

    # --- disk ---------------------------------------------------------------------------
    def _bundle_dirs(self) -> set[str]:
        from quantlab.data import sp500

        return {b["dir"] for b in sp500.bundles(self.lab.ws.root)}

    def _prune(self, before: set[str], version: str) -> None:
        """A refresh downloads the S&P 500 bundle (about 200 MB) again each day. Bundles and index
        datasets that a refresh created are removed once a newer one exists, unless an experiment
        used them."""
        from quantlab.data import sp500

        path = self.lab.ws.path("paper", "created.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        made = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"bundles": [], "datasets": []}
        made["bundles"] += sorted(self._bundle_dirs() - before - set(made["bundles"]))
        if version not in made["datasets"]:
            made["datasets"].append(version)
        used = {e["dataset_version"] for e in self.lab.registry.experiments()}
        store = self.lab.data
        keep_sha = set()
        for v in used:
            mf = store.proc_dir / v[:16] / "manifest.json"
            if mf.exists():
                keep_sha |= set(json.loads(mf.read_text(encoding="utf-8")).get("inputs", {}).get("bundle", {}).values())
        latest = sp500.bundles(self.lab.ws.root)[-1]["dir"]
        for d in list(made["datasets"]):
            if d != version and d not in used:
                shutil.rmtree(store.proc_dir / d[:16], ignore_errors=True)
                made["datasets"].remove(d)
        for b in list(made["bundles"]):
            if b == latest:
                continue
            meta = Path(b) / "meta.json"
            shas = set(f["sha256"] for f in json.loads(meta.read_text(encoding="utf-8"))["files"].values()) if meta.exists() else set()
            if not shas & keep_sha:
                shutil.rmtree(b, ignore_errors=True)
                made["bundles"].remove(b)
        path.write_text(json.dumps(made, indent=1), encoding="utf-8")
