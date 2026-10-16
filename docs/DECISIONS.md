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

## D-015 · 2026-10-14 · Desktop screens (PySide6-Essentials)
Six screens (데이터·전략·실험·검증·결과·기록) call `Lab` only; no verdict logic in the UI. Qt
comes from PySide6-Essentials (no WebEngine), keeping the packaged app small; reports open in the
user's browser instead of an embedded view. Long actions run one at a time on a worker thread
(the SQLite connection allows cross-thread use for that single job; buttons are disabled
meanwhile). The final evaluation asks for explicit confirmation showing the confirmed parameters
and the family's previous holdout reads. The first run asks for a work folder (stored in
QSettings). Engine modules never import Qt (tested in a subprocess).

## D-016 · 2026-10-14 · Windows packaging and CI
PyInstaller folder build (`packaging/QuantLab.spec`, windowed `QuantLab.exe`). The quantlab .py
sources ship inside the bundle as data, so `app_code_hash` of a frozen build equals the hash of
the same checkout (verified: both 54daeb343d06f535 on the Linux build). `QuantLab.exe
--selftest` runs the synthetic end-to-end flow with two spawned search workers and writes its
log to QUANTLAB_SELFTEST_LOG (a windowed exe has no console). GitHub Actions
(`windows-app.yml`, public repo so minutes are free) runs the tests, builds, removes Python from
PATH, runs the self-test and uploads the zipped app as an artifact for 30 days. The bundle is
large (~0.5 GB unpacked, mostly pyarrow and Qt); trimming is deferred until it matters.

## D-017 · 2026-10-14 · Second study: N4 rules, frozen, unlevered, on unseen markets
The baseline TT-SPY-SMA failed its one holdout read on the user's PC (Gate 3: holdout Sharpe
difference -0.273, p 0.87); that family's holdout is spent. Per plan v0.3 the next study tests
whether the prior project's only surviving idea (N4, five oversold-rebound legs found by mining
US data) has an edge without leverage on developed-market country ETFs the prior project never
tested (EWJ EWG EWU EWC EWA EWQ EWL EWH EWS), benchmark EFA, risk-free BIL. Rules are copied as
recorded and frozen (no search space); signals on each ETF itself; decide at close, fill next
open. Policy oversold_intl_v1 keeps every threshold and cost of trend_timing_v1; splits TRAIN
2007-06..2012, VALIDATION 2013..2016, HOLDOUT 2017-01 onward, fixed before any data for these
symbols was downloaded. Evidence is only partly independent (EFA/EEM were seen in b35).

## D-018 · 2026-10-14 · Slots sizing trades only the legs that change
The N4X-INTL-1X run on the user's PC (gross +0.34% per trade, net -0.26%, CAGR -26%) spent 26%
of its costs on re-sizing positions it already held: every entry or exit re-targeted all
symbols to the new equal weights. That is inherent to `equal_active` and stays as is (results
recorded under it remain reproducible). `{slots: K}` now sizes a position once, at entry, to 1/K
of equity and leaves the other holdings untouched when a leg opens or closes, as a swing
trader would. Mechanism: strategies may return `PartialWeights` (only listed symbols are
re-targeted); a plain dict still means "every symbol". Experiments run with slots sizing
before this change will not reproduce bit-for-bit under the new code; none were recorded.

## D-019 · 2026-10-14 · Third study: deep-oversold only, rarer and longer
N4X-INTL-1X lost to costs (D-018 context): its gross edge was real but +0.34% per trade. The user
chose (decision card) to test only deep declines on the same country ETFs: 20-day Bollinger z
below -2.5 or -3 with the 200-day trend filter, exit back at the 20-day average or after 10/20
sessions, slots sizing 1/5. The 4-setting search space and the Gate 4 neighbours are fixed in the
spec and policy `oversold_intl_deep_v1` (splits, gates and costs copied from oversold_intl_v1)
before any run. Chosen after seeing N4X's TRAIN+VALIDATION results only; the 2017+ holdout of
these ETFs remains unopened. Pass bar per trade: +1.2% gross (survives Gate 4's 2x cost).

## D-020 · 2026-10-14 · Automatic strategy search inside the app
N5X-INTL-DEEP also lost to costs on the user's PC (gross +0.02% per trade, net -0.57%, CAGR -1.3%
vs EFA -0.9%; not taken to the holdout). The user asked for an app that keeps testing strategies
without Claude, approved the design doc (building blocks, filters, trial-count correction) and
added: MACD and 5/20/60/120-day moving averages as entry signals, "must beat SPY buy-and-hold",
3,000 combinations per run, and more ETF groups.

- Building blocks: `quantlab.strategies.combo`, catalog 1 (34 entries, 6 trend filters, 7 exits,
  max hold 5/10/20/30, stop 0/10%, take 0/10%; 22,848 combinations per group). A recorded trial
  always means the same rule: changing a key's meaning needs catalog 2. New indicators (MACD,
  MA ratio/crosses) are whole-series and causal (`quantlab.strategies.indicators`); combo rules read
  every indicator that way, in the search and in the final evaluation alike. Hand-written specs
  keep the windowed calculation, so their recorded results do not change.
- Groups, each a generator spec AUTO-* plus a policy fixed before its data was downloaded: 선진국 A
  (the N4X/N5X ETFs; their two runs count as prior trials), 선진국 B, 신흥국, 채권·금·원자재·리츠.
  US sector ETFs are left out: the prior project saw US stocks 2010-2026. Splits TRAIN 2007-06..2012,
  VALIDATION 2013..2016, HOLDOUT 2017+; benchmark SPY; costs 0.25% + 5bp.
- Order: the full grid shuffled with the policy's seed; each run registers its next 3,000 untried
  combinations before the first runs; an interrupted run is resumed before new ones start.
- Filters (TRAIN+VALIDATION only, in this order): trades >= 40 TRAIN and >= 20 VALIDATION; gross
  >= +1.2% per trade; both splits positive; CAGR > 0 at 2x costs; CAGR above SPY buy-and-hold;
  max drawdown shallower than -35%; net mean positive without the 10 best trades; CAGR positive
  without the best year; day-clustered net t >= one-sided Bonferroni threshold for the family's
  trial count (alpha 5%); at least half of the one-step neighbours (adjacent threshold, adjacent
  max hold) clear trades/gross/splits/2x cost. Neighbours not yet tried are registered and run.
- Nothing is adopted automatically. The user confirms a candidate and starts its final evaluation;
  a group's holdout read is logged per family, so a second final evaluation in a group cannot PASS.
- Trials store their numbers on the trial event (no experiment row per trial); opening a candidate
  records a normal exploratory backtest. The registry uses WAL so the search can write from its own
  connection while the window is in use.
- Charter/protocol wording for large-scale machine search is pending in the charter thread.

## D-021 · 2026-10-15 · Fifth search group: 3x leveraged non-US ETFs
All four 1x groups ran to the end on the user's PC (4 x 22,848 combinations) with no candidate.
The funnels add up exactly per group and spread over several filters, so this is a result, not a
fault: about 60-64% failed gross >= +1.2% per trade, 30-41% had too few trades, and most of the
few hundred left fell short of SPY; two (신흥국) reached the t-value filter and failed it. With a
0.6% round trip, 1x country/bond/commodity ETFs rarely move enough in 2 days to 6 weeks.

The user chose (decision card) a group of instruments that move about three times as much:
AUTO-LEV, Direxion 3x EDC (emerging markets), YINN (China large caps), KORU (Korea) and EURL
(developed Europe), policy `auto_lev_v1` fixed before any of their data was downloaded. The prior
project tested US 3x ETFs (SPXL, TQQQ) only. Same costs and filters (the +1.2% bar is not lowered); 4 slots,
one per ETF. The funds list 2008-12 .. 2014-01, so the splits are TRAIN
2010..2015, VALIDATION 2016..2019, HOLDOUT 2020+; a fund's rows before it lists are empty and
its rules simply do not trade there. The 1x results do not count as prior trials of this family:
nothing in the catalog or filters was changed after seeing them, only the universe.

The user then asked to add very many more combinations. AUTO-LEV uses building-block catalog 2:
catalog 1 unchanged plus deeper thresholds (z -3, RSI(2) < 2, 6 down days, 5/10-day drops of 12-20%,
deeper distances below the 5/20/60/120-day averages, IBS < 0.05), RSI(14) < 30/25/20, 20-day drops
of 10/15/20%, 60/120-day lows; filters 60-day average, MACD below 0, SPY above/below its 200-day
average; exits RSI(2) > 50/90 and a 5-day high; holding 3 days added; stops 0/10/20%, takes
0/10/20/30%. 57 x 10 x 10 x 5 x 3 x 4 = 342,000 combinations, so the required t rises from about
4.6 to about 5.1 (Bonferroni grows only with the log of the count). Catalog 1 lists, order,
neighbours and rules are byte-for-byte what they were, so the 1x groups' recorded trials keep their
meaning. With this many trials the search keeps each group's trial states in memory and reads only
new events, and the page reads its summary off the window thread.

## D-022 · 2026-10-15 · Beating SPY: risk-adjusted, and with idle equity in SPY
After 85,600 of AUTO-LEV's combinations: 38,750 too few trades, 40,397 below +1.2% gross per trade,
3,149 a losing split, 852 lost at 2x costs, and all 2,452 that were left fell short of SPY's CAGR
(about 13%/yr over 2010-2019). A swing rule is out of the market most of the time, so on CAGR it
rarely beats an index that is fully invested through a bull market. The user chose to try both:

- AUTO-LEV, policy `auto_lev_v2`: the SPY filter compares Sharpe (rule vs SPY buy-and-hold, with
  BIL as the risk-free rate, TRAIN+VALIDATION), as Gates 2 and 3 of the final evaluation already do.
  Trials recorded under v1 keep their numbers (each stored its Sharpe) and are judged again.
- AUTO-LEVP, policy `auto_levp_v1`: the same 342,000 combinations with equity not in an open leg
  held in SPY (`rules.combo.park`, entry_exit `sizing: {slots: K, park: SPY}`: SPY is bought on the
  first session and resized, with costs, whenever a leg opens or closes). Its total CAGR must beat
  SPY buy-and-hold. SPY's round trips are not trades of the rule: per-trade numbers and t exclude them.
- Each family counts the other's 342,000 combinations as prior trials (required t about 5.3).
  They trade the same ETFs and so share one holdout: once either has had its final evaluation, the
  other cannot have one (the app refuses).
- Drawdown, cost, consistency and t filters are unchanged.

## D-023 · 2026-10-15 · Near misses can be reviewed, not promoted

The user wanted to look again at the ~8,000 AUTO-LEV combinations that fell short only at the SPY
Sharpe check.
- The 자동 탐색 page has a second list, "아깝게 떨어진 전략": trials that cleared trades, gross,
  both splits and 2x cost but failed the SPY check or a later filter, ordered by the group's
  `beat_metric`. The page shows the top 500; "CSV로 저장" writes all of them.
- A near miss can be backtested on TRAIN+VALIDATION like a candidate, but it cannot be sent to the
  final evaluation: picking the best of thousands that failed would be selection on noise.

## D-024 · 2026-10-16 · One pre-fixed near miss may read the 3x holdout

AUTO-LEV and AUTO-LEVP tried all 684,000 combinations with no candidate. In AUTO-LEVP, 1,840
combinations beat SPY's CAGR and passed every filter but the t threshold (best t 3.4, needed about
5.3). Their entries cluster: "5-day return below -15%" (560, 17x its share) and "3 down days"
(407, 12.6x); filters, exits, stops and takes are spread evenly. The same oversold-rebound theme
was the one survivor (N4) of the earlier project.
- The user chose to spend the shared LEV/LEVP holdout (2020+) on one rule fixed before it is read.
- The rule was fixed by a stated principle, not by the best t: the most frequent entry (ret5_15),
  then the most frequent value of every other setting among that entry's 560 t-only near misses:
  `ret5_15 · down200 · rsi2_70 · max_hold 5 · stop 20% · take 10%` (dev: CAGR 14.7%, MDD -27%,
  99 trades, t 2.4). It is `PREREGISTERED["AUTO-LEVP"]` and must not change.
- It is not a candidate. The final evaluation runs as usual (DSR over the family's 684,000 trials),
  and is read as a check of the theme, not as an adoption. After it, neither group can read the holdout.
