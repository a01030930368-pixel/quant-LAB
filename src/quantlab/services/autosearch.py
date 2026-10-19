"""Automatic strategy search (design doc 2026-10-14, user-approved; D-020).

The app combines building blocks (`quantlab.strategies.combo`) on one ETF group at a time and keeps
only combinations that clear fixed filters. It never needs Claude and never reads the holdout:

  * one generator spec per group (builtin specs AUTO-*), one validation policy per group whose
    `auto_search` block holds the filters, fixed before any data of the group was seen;
  * the full grid is shuffled once with a fixed seed; each run takes the next untried
    combinations (3,000 by default) and registers ALL of them before the first one runs;
  * workers receive only TRAIN+VALIDATION data (`dev_data` cuts the holdout before they start);
  * every trial counts: the t-value a candidate needs grows with the family's trial count
    (one-sided Bonferroni), and the final evaluation's DSR uses the same count;
  * candidates are listed for the user; confirming one and running its final evaluation is the
    user's decision, and a second final evaluation in the same group cannot PASS (holdout reused).

A trial's numbers are stored on its 'done' trial event (no experiment row per trial, so the
history screen is not flooded); opening a candidate runs a normal recorded backtest.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.analysis.trades import episodes, without_best_year
from quantlab.backtesting.engine import run_backtest
from quantlab.core.hashing import canonical_json, restore_numbers
from quantlab.core.market_data import MarketData
from quantlab.data.providers import display_name
from quantlab.strategies import combo, stock_combo
from quantlab.strategies.spec import load_spec
from quantlab.validation import metrics as M
from quantlab.validation.engine import BuyAndHold, _cfg, _rf, dev_data

GROUPS = ("AUTO-DEVA", "AUTO-DEVB", "AUTO-EM", "AUTO-REAL", "AUTO-LEV", "AUTO-LEVP", "AUTO-SP500", "AUTO-SP500X",
          "AUTO-SP500Y", "AUTO-KR2X")
# groups that trade the same ETFs share one holdout: only one final evaluation across them
SHARED_HOLDOUT = {"AUTO-LEV": ("AUTO-LEV", "AUTO-LEVP"), "AUTO-LEVP": ("AUTO-LEV", "AUTO-LEVP"),
                  # the pre-fixed rule taken from this search's near misses (D-029) reads the same holdout
                  "AUTO-SP500": ("AUTO-SP500", "GAP-EARN-SP500", "AUTO-SP500X", "AUTO-SP500Y"),
                  # the research factory searches the same stocks after that holdout was read (D-033, D-034)
                  "AUTO-SP500X": ("AUTO-SP500", "GAP-EARN-SP500", "AUTO-SP500X", "AUTO-SP500Y"),
                  "AUTO-SP500Y": ("AUTO-SP500", "GAP-EARN-SP500", "AUTO-SP500X", "AUTO-SP500Y")}

# filter code -> Korean label, in the order they are checked
FILTERS = [
    ("trades", "거래 수 부족"),
    ("gross", "거래당 비용 전 수익 부족"),
    ("splits", "TRAIN·VALIDATION 중 손실 구간"),
    ("cost2x", "비용 2배에서 손실"),
    ("spy", "SPY 보유보다 낮은 수익률"),
    ("mdd", "최대 낙폭 과다"),
    ("top10", "상위 10건 빼면 손실"),
    ("bestyear", "최고의 해 빼면 손실"),
    ("tvalue", "시험 개수 대비 통계 강도 부족"),
    ("neighbours", "이웃 조합이 버티지 못함"),
]
FILTER_LABEL = dict(FILTERS)
SPY_SHARPE_LABEL = "SPY 보유보다 낮은 샤프 (위험 대비 수익)"
# one near miss per holdout fixed in advance by the user's decision (D-024): it may have the final
# evaluation although it is not a candidate. Fixed before the holdout was read; never change it.
PREREGISTERED = {
    "AUTO-LEVP": {"entry": "ret5_15", "exit": "rsi2_70", "filter": "down200", "max_hold": 5, "stop": 0.2, "take": 0.1},
}
PREREGISTERED_LABEL = "사전 고정 (최종 평가 가능)"

# the cheap filters; a trial past them counts as a near miss when it fails a later one
EARLY_FILTERS = ("trades", "gross", "splits", "cost2x")


def key(params: dict) -> str:
    names = stock_combo.PARAMS if "slots" in params and "hold" in params else combo.PARAMS
    return canonical_json({k: params[k] for k in names})


_PARAM_SETS = (frozenset(combo.PARAMS), frozenset(stock_combo.PARAMS))


def blocks(loaded):
    """The building-block module of a generator spec and its rules block (ETF `combo` or index `stock_combo`)."""
    rules = loaded.spec["rules"]
    if "stock_combo" in rules:
        return stock_combo, rules["stock_combo"]
    return combo, rules["combo"]


class LazyGrid:
    """The full grid of a generator spec in a fixed shuffled order, without building it.

    For grids too large to hold as a list (D-033, `auto_search.order: affine`): combination i of the
    shuffled order is combination (a*i + b) mod N of the plain product order, with a coprime to N and
    a, b drawn from the policy's seed. Combinations the block module says were already run elsewhere
    (`already_tried`) are skipped."""

    def __init__(self, space: dict[str, list], seed: int, skip=None) -> None:
        self.names = sorted(space)
        self.lists = [list(space[n]) for n in self.names]
        self.n = int(np.prod([len(v) for v in self.lists], dtype=np.int64)) if self.lists else 1
        rng = np.random.default_rng(seed)
        a = int(rng.integers(1, max(self.n, 2)))
        while math.gcd(a, self.n) != 1:
            a += 1
        self.a, self.b = a, int(rng.integers(0, max(self.n, 1)))
        self.skip = skip

    def at(self, i: int) -> dict:
        j = (self.a * i + self.b) % self.n
        out = {}
        for name, values in zip(reversed(self.names), reversed(self.lists)):
            j, r = divmod(j, len(values))
            out[name] = values[r]
        return {k: out[k] for k in self.names}

    def __iter__(self):
        for i in range(self.n):
            p = self.at(i)
            if self.skip is None or not self.skip(p):
                yield key(p), p

    def size(self) -> int:
        if self.skip is None:
            return self.n
        return self.n - sum(1 for i in range(self.n) if self.skip(self.at(i)))


def required_t(n_trials: int, alpha: float) -> float:
    """One-sided Bonferroni threshold for the family's trial count."""
    return float(stats.norm.ppf(1 - alpha / max(int(n_trials), 1)))


# ----------------------------------------------------------------------------- one trial
def _split_return(equity: pd.Series, start, end) -> float | None:
    e = equity[(equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))]
    before = equity[equity.index < pd.Timestamp(start)]
    base = before.iloc[-1] if len(before) else (e.iloc[0] if len(e) else None)
    if base is None or not len(e) or base <= 0:
        return None
    return float(e.iloc[-1] / base - 1)


def trial_metrics(factory, params: dict, dev: MarketData, policy: dict) -> dict:
    """Numbers the filters need, from one run at normal costs (and one at doubled costs when the
    cheap filters pass). TRAIN+VALIDATION data only."""
    auto = policy["auto_search"]
    sp = policy["splits"]
    res = run_backtest(factory(**params), dev, _cfg(policy))
    eq = res.equity
    versus = auto.get("versus")  # stock rules: judge each trade against SPY held for the same days
    ep = episodes(res, dev, exclude=(auto["park_symbol"],) if auto.get("park_symbol") else (),
                  excess_vs=versus).dropna(subset=["gross"])
    train_end = pd.Timestamp(sp["train"][1])
    m: dict = {"n": int(len(ep)), "n_train": int((ep["entry"] <= train_end).sum()),
               "n_validation": int((ep["entry"] > train_end).sum())}
    if len(ep):
        if versus:
            ep = ep.assign(net=ep["net"] - ep["bench"])  # net return over SPY
        net = ep["net"].to_numpy()
        by_day = ep.groupby("entry")["net"].mean().to_numpy()
        k = len(by_day)
        sd = float(np.std(by_day, ddof=1)) if k > 1 else float("nan")
        m.update(gross=float(ep["gross"].mean()), net=float(net.mean()), entry_days=int(k),
                 net_ex_top10=float(np.sort(net)[::-1][10:].mean()) if len(net) > 10 else None,
                 t=float(by_day.mean() / (sd / math.sqrt(k))) if k > 1 and sd > 0 else None,
                 win_rate=float((net > 0).mean()), mean_sessions=float(ep["sessions"].mean()))
        if versus:
            m["excess"] = float(ep["excess"].mean())
    rf = _rf(dev, policy)
    m.update(cagr=M.cagr(eq), mdd=M.max_drawdown(eq), sharpe=M.sharpe(res.returns(), rf),
             ret_train=_split_return(eq, sp["train"][0], sp["train"][1]),
             ret_validation=_split_return(eq, sp["validation"][0], sp["validation"][1]),
             cagr_ex_best_year=without_best_year(eq).get("cagr_ex_best_year"))
    if _support(m, auto):
        stressed = run_backtest(factory(**params), dev, _cfg(policy, float(auto["stress_cost_multiplier"])))
        m["cagr_2x"] = M.cagr(stressed.equity)
    return m


def _support(m: dict, auto: dict) -> bool:
    """trades, gross and both splits positive (the cheap filters; neighbours must clear these too)."""
    return (m["n_train"] >= auto["min_trades_train"] and m["n_validation"] >= auto["min_trades_validation"]
            and (m.get(_gross_field(auto)) or -1) >= auto["min_gross_per_trade"]
            and (m.get("ret_train") or -1) > 0 and (m.get("ret_validation") or -1) > 0)


def _gross_field(auto: dict) -> str:
    """Per-trade return before costs that the gross filter reads: over SPY for stock rules."""
    return "excess" if auto.get("versus") else "gross"


def first_failure(m: dict, auto: dict, beat: float, t_req: float) -> str | None:
    """The first filter a trial fails (before the neighbour check), or None. `beat` is the benchmark's
    value of the policy's `beat_metric` (CAGR unless the policy says sharpe)."""
    if m["n_train"] < auto["min_trades_train"] or m["n_validation"] < auto["min_trades_validation"]:
        return "trades"
    if (m.get(_gross_field(auto)) or -1) < auto["min_gross_per_trade"]:
        return "gross"
    if not ((m.get("ret_train") or -1) > 0 and (m.get("ret_validation") or -1) > 0):
        return "splits"
    if not (m.get("cagr_2x") is not None and m["cagr_2x"] > 0):
        return "cost2x"
    metric = auto.get("beat_metric", "cagr")
    if not (m.get(metric) is not None and m[metric] > beat):
        return "spy"
    if not m["mdd"] > auto["max_drawdown_floor"]:
        return "mdd"
    if not (m.get("net_ex_top10") is not None and m["net_ex_top10"] > 0):
        return "top10"
    if not (m.get("cagr_ex_best_year") is not None and m["cagr_ex_best_year"] > 0):
        return "bestyear"
    if not (m.get("t") is not None and m["t"] >= t_req):
        return "tvalue"
    return None


def neighbour_ok(m: dict, auto: dict) -> bool:
    return _support(m, auto) and m.get("cagr_2x") is not None and m["cagr_2x"] > 0


# trial states and grid orders kept between calls (see AutoSearch.trials and AutoSearch.order)
_LOCK = threading.RLock()
_STATE: dict[tuple[str, str], dict] = {}
_ORDER: dict[tuple, list[tuple[str, dict]]] = {}


# ----------------------------------------------------------------------------- workers
_W: dict = {}


def _init_worker(spec_path: str, strategy_hash: str, policy: dict, dev: MarketData) -> None:
    loaded = load_spec(spec_path)
    if loaded.strategy_hash != strategy_hash:
        raise RuntimeError("spec changed on disk")
    _W.update(loaded=loaded, policy=policy, dev=dev)


def _run_trial(task: tuple[int, dict]) -> tuple[int, dict | None, str | None]:
    idx, params = task
    try:
        p = {**_W["loaded"].default_params, **params}
        return idx, trial_metrics(_W["loaded"].factory, p, _W["dev"], _W["policy"]), None
    except Exception as e:  # recorded as a failed trial, never silently dropped
        return idx, None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------- service
class AutoSearch:
    def __init__(self, lab) -> None:
        self.lab = lab
        self._loaded: dict[str, tuple] = {}

    # --- set-up --------------------------------------------------------------------
    def spec_path(self, group: str):
        from quantlab.services.lab import builtin_path

        if group not in GROUPS:
            raise KeyError(group)
        return builtin_path("specs", f"{group}.qls.yaml")

    def loaded(self, group: str):
        """Register the group's generator spec (once) and return (strategy hash, loaded spec)."""
        if group not in self._loaded:
            imp = self.lab.import_strategy(self.spec_path(group))
            if imp.problems and imp.state == "IDEA":
                raise RuntimeError("; ".join(imp.problems))
            park = blocks(imp.loaded)[1].get("park")
            _, policy = self.lab.policy_for(imp.loaded)
            if park != policy["auto_search"].get("park_symbol"):
                raise RuntimeError(f"{group}: spec park {park!r} differs from the policy's park_symbol")
            self._loaded[group] = (imp.strategy_hash, imp.loaded)
        return self._loaded[group]

    def policy(self, group: str) -> tuple[str, dict]:
        _, loaded = self.loaded(group)
        return self.lab.policy_for(loaded)

    def catalog(self, group: str) -> int:
        mod, block = blocks(self.loaded(group)[1])
        return mod.catalog_of(block)

    def blocks(self, group: str):
        return blocks(self.loaded(group)[1])[0]

    def order(self, loaded, policy: dict) -> list[dict]:
        """The fixed order of the whole grid (pre-registered by the spec, the catalog and the seed)."""
        return [p for _, p in self._keyed_order(loaded, policy)]

    def _keyed_order(self, loaded, policy: dict):
        from quantlab.services.batch import grid

        seed = int(policy["auto_search"]["order_seed"])
        if policy["auto_search"].get("order") == "affine":
            with _LOCK:
                if (loaded.strategy_hash, seed) not in _ORDER:
                    mod, block = blocks(loaded)
                    cat = mod.catalog_of(block)
                    skip = getattr(mod, "already_tried", None)
                    _ORDER[(loaded.strategy_hash, seed)] = LazyGrid(
                        loaded.search_space, seed, (lambda p: skip(p, cat)) if skip else None)
                return _ORDER[(loaded.strategy_hash, seed)]
        with _LOCK:
            if (loaded.strategy_hash, seed) not in _ORDER:
                g = grid(loaded.search_space)
                perm = np.random.default_rng(seed).permutation(len(g))
                _ORDER[(loaded.strategy_hash, seed)] = [(key(g[i]), g[i]) for i in perm]
            return _ORDER[(loaded.strategy_hash, seed)]

    # --- records -------------------------------------------------------------------
    def trials(self, group: str) -> dict[str, dict]:
        """Latest state of every trial of the family, by combination.

        Kept per registry file and group for the life of the process and brought up to date with only
        the events written since the last call: a group can hold hundreds of thousands of trials and
        the page reads this every few seconds while a search runs."""
        reg = self.lab.registry
        with _LOCK:
            cache_key = (str(reg.path.resolve()), group)
            st = _STATE.get(cache_key)
            if st is None or reg.last_trial_event_id() < st["last"]:  # new or replaced registry file
                st = _STATE[cache_key] = {"last": 0, "by_id": {}, "by_key": {}}
            by_id, by_key = st["by_id"], st["by_key"]
            for ev in reg.trial_events_for(group, after=st["last"]):
                tid = (ev["batch_id"], ev["trial_index"])
                v = by_id.get(tid)
                if v is None:
                    v = by_id[tid] = {"id": tid, "params": None, "status": None, "metrics": None, "detail": None}
                if v["params"] is None and ev["params"]:
                    v["params"] = json.loads(ev["params"])
                    # stored params are canonical JSON; with exactly the combo keys that is already the key
                    by_key[ev["params"] if frozenset(v["params"]) in _PARAM_SETS else key(v["params"])] = v
                v["status"] = ev["event"]
                if ev["event"] == "done" and ev["detail"]:
                    d = ev["detail"]
                    v["metrics"] = restore_numbers(json.loads(d)) if ("NaN" in d or "Inf" in d) else json.loads(d)
                    v.pop("pre", None)
                elif ev["event"] == "failed":
                    v["detail"] = ev["detail"]
                st["last"] = ev["event_id"]
            return dict(by_key)

    def n_trials(self, group: str, policy: dict) -> int:
        return self.lab.registry.count_trials_for_family(group) + int(policy["auto_search"].get("prior_trials", 0))

    def _beat_file(self, group: str):
        return self.lab.ws.root / "auto" / f"{group}.json"

    def beat_value(self, group: str, metric: str = "cagr") -> float | None:
        """The benchmark's CAGR or Sharpe over TRAIN+VALIDATION, from the latest run of the group."""
        try:
            v = json.loads(self._beat_file(group).read_text(encoding="utf-8")).get(f"beat_{metric}")
            if v is not None:
                return float(v)
        except (OSError, ValueError):
            pass
        for b in reversed(self.lab.registry.batches_for(group)):
            if b["config"].get(f"beat_{metric}") is not None:
                return float(b["config"][f"beat_{metric}"])
        return None

    def beat_cagr(self, group: str) -> float | None:
        return self.beat_value(group, "cagr")

    # --- running -------------------------------------------------------------------
    def run(self, group: str, workers: int = 0, progress: Callable[[int, int], None] | None = None,
            should_stop: Callable[[], bool] | None = None, say: Callable[[str], None] | None = None,
            limit: int | None = None) -> dict:
        """One run: the next `trials_per_run` untried combinations (or `limit`, for tests), or the
        unfinished trials of an interrupted run. Neighbours of combinations that clear every other
        filter are registered and run afterwards."""
        say = say or (lambda s: None)
        reg = self.lab.registry
        strategy_hash, loaded = self.loaded(group)
        policy_hash, policy = self.lab.policy_for(loaded)
        auto = policy["auto_search"]
        version = self.lab.ensure_spec_dataset(loaded, policy, say)
        data, _ = self.lab._strategy_symbols_data(loaded, version)
        dev = dev_data(data, policy)  # the holdout is cut here, before any worker exists
        bench = run_backtest(BuyAndHold(auto["beat_symbol"]), dev, _cfg(policy))
        beats = {"beat_cagr": M.cagr(bench.equity), "beat_sharpe": M.sharpe(bench.returns(), _rf(dev, policy))}
        beat = beats[f"beat_{auto.get('beat_metric', 'cagr')}"]
        f = self._beat_file(group)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(beats | {"dataset_version": version}), encoding="utf-8")

        known = self.trials(group)
        pending = [v for v in known.values() if v["status"] in ("registered", "cancelled")]
        if pending:  # an interrupted run: finish exactly those trials first
            say(f"이전 탐색에서 남은 {len(pending)}개를 이어서 시험합니다")
            batch_id = pending[0]["id"][0]
            todo = [(v["id"][1], v["params"], v["id"][0]) for v in pending]
        else:
            want = int(limit or auto["trials_per_run"])
            fresh = []
            for k, p in self._keyed_order(loaded, policy):
                if k not in known:
                    fresh.append(p)
                    if len(fresh) >= want:
                        break
            if not fresh:
                return self.summary(group) | {"note": "이 묶음의 조합을 모두 시험했습니다."}
            batch_id = f"A-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
            reg.create_batch(batch_id, strategy_hash, version,
                             {"auto": True, "catalog": self.catalog(group), "trials": fresh, "policy_hash": policy_hash,
                              "scope": "dev", "beat_symbol": auto["beat_symbol"], **beats})
            reg.register_trials(batch_id, list(enumerate(fresh)))
            todo = [(i, p, batch_id) for i, p in enumerate(fresh)]

        n_workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        init = (str(loaded.path), strategy_hash, policy, dev)
        done = [0]
        total = [len(todo)]
        stopped = [False]

        def record(task_batch: str, idx: int, params: dict, m: dict | None, err: str | None) -> None:
            if err is not None:
                reg.trial_event(task_batch, idx, "failed", params, detail=err)
            else:
                reg.trial_event(task_batch, idx, "done", params, detail=canonical_json(m))
            done[0] += 1
            if progress:
                progress(done[0], total[0])

        def run_tasks(tasks: list[tuple[int, dict, str]]) -> None:
            by_idx = {(b, i): p for i, p, b in tasks}
            batch_of = {i: b for i, p, b in tasks}
            if len({b for _, _, b in tasks}) > 1:  # resume across batches: run each batch's trials in turn
                for b in sorted({b for _, _, b in tasks}):
                    run_tasks([t for t in tasks if t[2] == b])
                    if stopped[0]:
                        return
                return
            jobs = [(i, p) for i, p, _ in tasks]
            if n_workers == 1 or len(jobs) <= 1:
                _init_worker(*init)
                for job in jobs:
                    if should_stop and should_stop():
                        stopped[0] = True
                        return
                    i, m, err = _run_trial(job)
                    record(batch_of[i], i, by_idx[(batch_of[i], i)], m, err)
                return
            ctx = mp.get_context("spawn")
            with ctx.Pool(min(n_workers, len(jobs)), initializer=_init_worker, initargs=init) as pool:
                for i, m, err in pool.imap_unordered(_run_trial, jobs):
                    record(batch_of[i], i, by_idx[(batch_of[i], i)], m, err)
                    if should_stop and should_stop():
                        stopped[0] = True
                        pool.terminate()
                        return

        say("조합 시험 중")
        run_tasks(todo)
        # neighbours of every combination that cleared the other filters (a few rounds at most)
        for _ in range(3):
            if stopped[0]:
                break
            need = self._missing_neighbours(group, policy, beat)
            if not need:
                break
            say(f"후보의 이웃 조합 {len(need)}개 확인 중")
            start = 1 + max((v["id"][1] for v in self.trials(group).values() if v["id"][0] == batch_id), default=-1)
            extra = [(start + j, p) for j, p in enumerate(need)]
            reg.register_trials(batch_id, extra)
            total[0] += len(extra)
            run_tasks([(i, p, batch_id) for i, p in extra])
        if stopped[0]:
            for v in self.trials(group).values():
                if v["status"] == "registered":
                    reg.trial_event(v["id"][0], v["id"][1], "cancelled", v["params"])
        return self.summary(group) | {"stopped": stopped[0]}

    def _missing_neighbours(self, group: str, policy: dict, beat: float) -> list[dict]:
        auto = policy["auto_search"]
        known = self.trials(group)
        t_req = required_t(self.n_trials(group, policy), auto["alpha"])
        need: dict[str, dict] = {}
        for v in known.values():
            if v["metrics"] is None or first_failure(v["metrics"], auto, beat, t_req) is not None:
                continue
            for nb in self.blocks(group).neighbours(v["params"], self.catalog(group)):
                k = key(nb)
                if k not in known:
                    need[k] = nb
        return list(need.values())

    # --- results -------------------------------------------------------------------
    def summary(self, group: str) -> dict:
        strategy_hash, loaded = self.loaded(group)
        _, policy = self.lab.policy_for(loaded)
        auto = policy["auto_search"]
        known = self.trials(group)
        mod = self.blocks(group)
        n = self.n_trials(group, policy)
        t_req = required_t(n, auto["alpha"])
        metric = auto.get("beat_metric", "cagr")
        beat = self.beat_value(group, metric)
        spy_label = SPY_SHARPE_LABEL if metric == "sharpe" else FILTER_LABEL["spy"]
        if auto["beat_symbol"] != "SPY":  # e.g. KODEX 200 for the KRX group
            spy_label = spy_label.replace("SPY", display_name(auto["beat_symbol"]))
        funnel = {code: 0 for code, _ in FILTERS}
        rows = []
        near = []  # reached the SPY check (cleared trades, gross, splits and 2x cost) but failed there or later
        for v in known.values():
            m = v["metrics"]
            if m is None or beat is None:
                continue
            pre = v.get("pre")  # first failure apart from the t-value threshold, which moves with n
            if pre is None or pre[0] != (metric, beat):
                pre = v["pre"] = ((metric, beat), first_failure(m, auto, beat, -math.inf))
            fail = pre[1] if pre[1] is not None else first_failure(m, auto, beat, t_req)
            if fail is None:
                nbs = [known.get(key(p)) for p in mod.neighbours(v["params"], self.catalog(group))]
                tried = [x for x in nbs if x is not None and x["metrics"] is not None]
                ok = sum(1 for x in tried if neighbour_ok(x["metrics"], auto))
                if len(tried) < len(nbs):
                    fail = "neighbours" if tried and ok < auto["neighbour_min_share"] * len(nbs) else None
                    status = "이웃 확인 대기" if fail is None else None
                else:
                    fail = None if nbs and ok >= auto["neighbour_min_share"] * len(nbs) else "neighbours"
                    status = None
                if fail is None:
                    rows.append({"params": v["params"], "rule": mod.describe(v["params"]),
                                 "neighbours": f"{ok}/{len(nbs)}", "status": status or "후보", **m})
            if fail is not None:
                funnel[fail] += 1
                if fail not in EARLY_FILTERS:
                    if "rule" not in v:
                        v["rule"] = mod.describe(v["params"])
                    pre_reg = PREREGISTERED.get(group) == v["params"]
                    near.append({**m, "params": v["params"], "rule": v["rule"], "fail": fail, "preregistered": pre_reg,
                                 "status": PREREGISTERED_LABEL if pre_reg else
                                 spy_label if fail == "spy" else FILTER_LABEL[fail]})
        rows.sort(key=lambda r: -(r.get("t") or 0))
        near.sort(key=lambda r: (not r["preregistered"], -math.inf if r.get(metric) is None else -r[metric]))
        order = self._keyed_order(loaded, policy)
        grid_size = self._grid_size(loaded, order)
        done = sum(1 for v in known.values() if v["status"] == "done")
        reads = self.holdout_reads(group)
        return {"group": group, "name": loaded.spec["strategy"]["name"], "strategy_hash": strategy_hash,
                "grid": grid_size, "tried": len(known), "done": done,
                "failed": sum(1 for v in known.values() if v["status"] == "failed"),
                "pending": sum(1 for v in known.values() if v["status"] in ("registered", "cancelled")),
                "family_trials": n, "t_required": t_req, "beat_symbol": auto["beat_symbol"],
                "beat_metric": metric, "beat_value": beat, "beat_cagr": self.beat_value(group, "cagr"),
                "funnel": [{"code": c, "label": spy_label if c == "spy" else lbl,
                            "count": funnel[c]} for c, lbl in FILTERS],
                "candidates": rows, "near": near, "holdout_start": str(policy["splits"]["holdout"][0]), "holdout_reads": reads}

    # --- a candidate -----------------------------------------------------------------
    def open_candidate(self, group: str, params: dict):
        """A normal recorded exploratory backtest of one combination (TRAIN+VALIDATION)."""
        strategy_hash, loaded = self.loaded(group)
        _, policy = self.lab.policy_for(loaded)
        version = self.lab.ensure_spec_dataset(loaded, policy)
        return self.lab.backtest(strategy_hash, version, params)

    def _grid_size(self, loaded, order) -> int:
        if not isinstance(order, LazyGrid):
            return len(order)
        with _LOCK:
            cache = _ORDER.setdefault(("size", loaded.strategy_hash), [])
            if not cache:
                cache.append(order.size())
            return cache[0]

    def holdout_reads(self, group: str) -> int:
        """Final evaluations already run on this group's holdout (and on groups sharing it)."""
        families = SHARED_HOLDOUT.get(group, (group,))
        return sum(1 for r in self.lab.registry.holdout_ledger() if r["strategy_family"] in families)

    def final_evaluation(self, group: str, params: dict):
        """Confirm the combination and read the group's holdout once. Only the user starts this."""
        others = [g for g in SHARED_HOLDOUT.get(group, ()) if g != group]
        if others and any(r["strategy_family"] in others for r in self.lab.registry.holdout_ledger()):
            raise RuntimeError("같은 ETF의 보류 구간을 다른 묶음에서 이미 열었습니다. 이 묶음은 최종 평가를 할 수 없습니다.")
        strategy_hash, loaded = self.loaded(group)
        _, policy = self.lab.policy_for(loaded)
        version = self.lab.ensure_spec_dataset(loaded, policy)
        self.lab.confirm_candidate(strategy_hash, params, note="automatic search candidate")
        return self.lab.final_evaluation(strategy_hash, version)
