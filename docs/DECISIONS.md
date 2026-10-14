# Decision log

Technical decisions made under 04_DEVELOPMENT_HANDOFF §2 (Claude decides; records rationale).
Scope-level defaults are in `/mnt/project-files/plans/MVP_EXECUTION_PLAN.md` §1–§2; this file
records implementation-level choices. Newest last. Never edit past entries; supersede them.

## D-001 · 2026-10-14 · In-house daily backtester
Event-driven, daily bars, long-only, no leverage, target-weight interface. Off-the-shelf engines
(backtrader, vectorbt, zipline forks) cost more to audit for fill assumptions than ~200 lines
cost to write and test. Acceptance per 03 §2.1: independent reference implementation
(`backtesting/reference.py`, value-based instead of share-based) must match to 1e-9.

## D-002 · 2026-10-14 · Unadjusted prices + explicit corporate actions
Engine consumes raw OHLC plus `dividend` and `split` columns, credits dividends as cash on the
ex-date and restates shares (and open orders) on splits. Back-adjusted prices embed future
dividends and would hide accounting errors. Strategies that need a return series use
`total_return_index`, built forward so later data never revises earlier values.
Known simplification: dividends are credited on the ex-date, not the pay date.

## D-003 · 2026-10-14 · Order timing and sizing
Signal at close t → target share count computed from close-t equity and close-t price → fills at
open t+1, sells before buys, buys capped by available cash (integer shares by default). If the
asset has no open price that day the order carries forward. Slippage worsens the fill price;
commission = bps of notional + fixed fee.

## D-004 · 2026-10-14 · Calendar knowledge is not look-ahead
`MarketView.is_last_session_of_month` reads the next session date from the shared calendar.
Exchange schedules are published in advance, so this is information available at t.

## D-005 · 2026-10-14 · Code location (provisional)
Until the user picks a code host, the repository lives at `/mnt/project-files/quant-lab` with a
local git history. The virtualenv lives outside the shared folder (`UV_PROJECT_ENVIRONMENT`).
On the network mount, run commands from `/mnt/attach/project-files/quant-lab` (the symlinked
path breaks `os.getcwd()`).

## D-006 · 2026-10-14 · Test adequacy check by mutation
After the M3 suite passed on first run, six deliberate bugs were injected one at a time (skip
dividend, no split restatement of open orders, favourable slippage sign, mark-to-market at open,
strategy sees bar t+1, commission not charged). Each was caught by the suite.

## D-007 · 2026-10-14 · Leak detection has two layers (gap found by a control test)
The perturbation test (alter prices after a cut, decisions before it must not change) missed a
control strategy that read prices from a closure over the full dataset: perturbing the view's data
cannot affect data the strategy holds elsewhere. Added `isolation_check` to Gate 1: strategy
methods may not capture or reference DataFrame/Series/ndarray/MarketData objects via closures or
module globals, nor use `_data`, `read_*` or `urlopen`. Both layers have control tests. Neither is a
security boundary; a determined strategy can still read files. Phase 4 needs process-level isolation.

## D-008 · 2026-10-14 · Synthetic PASS control and what it says about test power
Control market: monthly Markov regimes (bull +1.5%/mo σ3%, bear −8%/mo σ6%, stay 0.97/0.93).
Even here, the frozen gates gave PASS on only about half of seeds: when the 8-year holdout happened
to contain few bear regimes, trend timing correctly showed no Sharpe advantage, or made fewer than
6 position changes (HOLD). The engine is behaving as designed; the lesson is that a single-asset
monthly timing rule has low statistical power over one holdout window. Expect HOLD/FAIL on real SPY
and prefer higher-breadth hypotheses (multi-asset) next. The control test uses seed 0, whose holdout
contains bear regimes; this choice was made on synthetic data only.

## D-009 · 2026-10-14 · Desktop-app re-architecture (03 v2.0.0): registry fields and naming
`code_commit` replaced by `app_version` + `app_code_hash` (sha256 over the installed package's .py
sources, CRLF-normalized, so source checkouts and frozen builds both produce a stable identity).
Experiments also carry strategy_hash, spec_version, strategy_origin, batch_id, trial_index. New
append-only tables: strategies (with parent / derived_after_holdout_view lineage), batches,
trial_events (registered → done/failed/cancelled, status = latest event), exports. Gate 3 adds a
lineage check: a strategy derived after holdout results were seen, or whose ancestor's holdout
results were exported, cannot PASS on that holdout. Baseline family renamed TREND_TIMING_SMA →
TT-SPY-SMA (spec id in 03 §4.2) in the class and policy v1 before the policy was ever registered or
any data downloaded; thresholds unchanged.

## D-010 · 2026-10-14 · Strategy spec v1 import
`*.qls.yaml` is parsed with `yaml.safe_load`, schema-checked against every protocol 제3조 item
(missing → rejected, nothing registered), hashed (canonical JSON of the spec + code bytes, so
formatting-only YAML edits keep the hash, any content change makes a new one), and registered.
Same id@version with different content is refused: bump the version. Then a synthetic smoke test
and leak checks decide RESEARCH vs IDEA. Declarative rules (conditional signal; rank top-N with
absolute filter) are the default; `close` sources are forward-built split-adjusted prices.
Python-kind code is statically checked: import allow-list, no I/O/eval/getattr, no private
attributes (except `__init__`), and **no shared mutable state** (module/class-level values must be
constants, no global/nonlocal, no mutable defaults). The last rule was added after a control
strategy stored prices in a module dict during one run and read them in the next: per-run leak
tests cannot see that. Code runs in-process for now; a worker process with a time limit arrives
with batch execution (M11). Not a security boundary: users should import code only from sources
they trust, and the app shows code before import.
Baseline param renamed sma_months → lookback_months (class, policy v1 before registration).

## D-011 · 2026-10-14 · Service layer, CLI, self-test; builtin assets in the package
`quantlab.services.Lab` is the only API the GUI and CLI call. Exploratory backtests receive
TRAIN+VALIDATION data only; `final_evaluation` requires a recorded candidate confirmation
(append-only `confirmations` table; state → VALIDATION) and uses the confirmed parameters, never
caller-supplied ones. Policies and the baseline spec ship inside the package
(`quantlab/builtin`) so a frozen build carries them; the repo no longer has data/, experiments/
or reports/ folders (research data lives in the user's work folder; the repo is public).
`quantlab selftest` runs the whole flow on synthetic data drawn on real NYSE sessions; an
integration test runs it with sockets disabled (R1–R3).

## D-012 · 2026-10-14 · Costs from the user's standing rule; baseline is engine verification only
The prior-project handoff fixes one-way commission at 0.25% (Kiwoom US equities) and distrusts
profits that vanish at 2x cost. Policy trend_timing_v1 now uses 25bp commission + 5bp slippage and
a 2x cost stress in Gate 4 (was 5bp total and a 25bp slippage stress). Changed before any data
download or registration; stricter only. The user excludes index-tracking/momentum strategies,
so TT-SPY-SMA stays solely to verify the app end to end and is labelled so in its spec.

## D-013 · 2026-10-14 · Results: charts, report, packet
Charts are built with `matplotlib.figure.Figure` only (no pyplot) so the same functions serve
headless reports and the Qt canvas. `Lab.write_report` writes report.md/.html with fixed Korean
sections, including the standing limits (no tax, FX or regulatory fees; synthetic or seen data is
not evidence). `Lab.export_packet` builds a ZIP for outside review (Claude or a person) with
metrics, verdicts, policy, spec, trial summary, daily equity, trades and charts — never raw price
files — and refuses to write it if the stored API key string appears in any member. Every export
is recorded with whether it includes holdout results, because a viewed holdout result makes later
derived strategies non-independent (lineage check).

## D-014 · 2026-10-14 · Swing entry/exit rules and trade-level checks
The prior project's research (handoff 2026-10-14) is swing-style: buy on a condition, sell on a
condition, after N sessions, or at a stop/target. Spec v1 gains `rules.entry_exit` (legs, one
position per leg, `equal_active` or fixed-slot sizing). Every decision, including stops and
targets, is taken at the close and filled at the next open; the prior 15:55-decide/15:59-fill
variant needs intraday bars and is not approximated. Indicators read split-adjusted prices in a
bounded window (`max_hold_days` is mandatory, ≤ 252) through `MarketView.tail_values`, which
keeps a 4-symbol, 6-year run under a second. Reports add descriptive trade-level checks (gross
mean per trade, median, mean without the top 10, day-clustered t, CAGR without the best year);
they are not gate criteria in policy v1. An N4 daily approximation lives in test fixtures only
to prove the rules can express it; N4 is not registered as a candidate (seen data, 100+ trials).
