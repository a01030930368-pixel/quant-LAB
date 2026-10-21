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

## D-025 · 2026-10-16 · Oversold rebound: replication on new 3x ETFs, verdict on forward data

The D-024 rule's final evaluation (EXP-20261016-aa7f8c248d) failed Gate 3: holdout CAGR 19.1% vs
SPY 13.4%, but Sharpe 0.61 vs 0.65 and max drawdown -55.7% vs -30.2%. Its per-trade edge repeated out
of sample (+2.26% net per trade vs +2.42% on TRAIN+VALIDATION, win rate 67%, 135 trades). The user
chose to check the theme on new ETFs and then paper-trade it.
- OSR-SECT3X 1.0.0: the same rule, unchanged, on Direxion 3x bull sector/size ETFs listed before 2012
  that are not index trackers (SOXL, TNA, FAS, ERX, TECL, DRN, CURE), idle equity in SPY. The only
  change is the size: 10% per leg (slots 10) instead of 25%, so at most 70% sits in 3x funds.
- The size was chosen after seeing AUTO-LEVP's 2020+ result, and that result was exported, so the
  2020+ holdout of these ETFs cannot count as independent evidence (03 §4.4). Their TRAIN+VALIDATION
  backtest (2010-2019, never run on these ETFs) is the fast replication check; the verdict comes from
  the forward holdout, data from 2026-10-17 on.
- The home page's final evaluation now refuses before reading the holdout when the lineage check
  would make it invalid, instead of spending the holdout on an INVALID result.

## D-026 · 2026-10-16 · Insider purchase clusters in S&P 500 stocks (study A)

The oversold-rebound line is closed. The user chose event-driven ideas: A insider buying, then B
post-earnings drift, then C calendar effects. A needs every stock that was in the S&P 500, delisted
names included, and the SEC's insider filings.
- Prices and membership: the public GitHub project Johnbrick123/sp500-data (release `data`), free and
  without a key. Checked 2026-10-16 by counting only (no returns read): every one of 717 stocks that
  were members in 2010-2019 has prices on 99.99% of its member sessions. It has no licence and part of
  it is labelled as CRSP-derived, so it is used for the user's own research only, stays in the work
  folder and never goes into the repository or result packets. The app scales prices by
  adj_close / close (total-return prices, dividends 0, splits 1); rows without an open (the
  CRSP-derived ones) trade at that day's close.
- Insider filings: SEC Insider Transactions Data Sets (quarterly zips, 2009q3 on), downloaded by the
  app with the user's contact e-mail in the User-Agent as the SEC requires. Original Form 4 only;
  open-market purchases (code P, acquired). An issuer is linked to a price ticker through the symbols
  it filed under, so FB-era filings reach META and a recycled symbol does not reach the old company.
- INS-CLUSTER 1.0.0, fixed before any insider data was read: two or more different officers or
  directors, each buying at least $10,000 in one filing, filings public within 30 calendar days; the
  signal is the session the second insider's filing appears, for a stock that is an S&P 500 member that
  day and is not held. Buy at the next open with 1/10 of equity, sell 20 sessions later; idle equity in
  SPY; more signals than free slots: larger total purchase value first. One rule, no search.
- Policy `ins_cluster_v1`: TRAIN 2010-2015, VALIDATION 2016-2019, holdout 2020 on, the standing costs
  (0.25% + 0.05% one way) and gates. The stock prices were seen by the user's earlier project; insider
  filings were never used there.
- SPY holdings of a strategy that declares a `park_symbol` are not counted as trades of the rule.

## D-027 · 2026-10-16 · Post-earnings drift in S&P 500 stocks (study B)

INS-CLUSTER's TRAIN+VALIDATION backtest (EXP-20261016-901f30e428) found no edge: 707 trades, +1.12%
gross per trade, but SPY gained +0.88% over the same 20 sessions (excess +0.23%, t 0.8), below the
0.6% round-trip cost; CAGR 6.1% vs SPY 12.3%. Its holdout was not read. The user chose study B next.
- Earnings dates: original Form 8-K filings listing item 2.02, from the SEC submissions API (one JSON per
  company, older pages included), for the companies linked to index tickers through their Form 4s.
  Only the filing date is used; the reaction window covers the filing-date session and the next one,
  so the release time (before, during or after the session) does not matter.
- PEAD-SP500 1.0.0, fixed before any earnings-date data was read: the stock's return minus SPY's
  from the last close before the filing date to the close of the first session after it is +5% or
  more; the stock is an S&P 500 member that day and not held. Buy at the next open with 1/20 of
  equity, sell 30 sessions later; larger abnormal return first when slots are short. One rule, no search.
- Idle equity stays in SPY, but cash from exits goes back into SPY only once two slots' worth is idle
  (`park_band`): INS-CLUSTER's per-event SPY resizing cost about as much as its stock trades.
- Event strategies report each trade's return over SPY held for the same days (`mean_excess`,
  `t_excess_day_clustered`), the first number to read: below the round-trip cost there is no edge.
- Policy `pead_v1`: same splits, costs and gates as `ins_cluster_v1`.

## D-028 · 2026-10-16 · Automatic search on S&P 500 stocks (AUTO-SP500)
- The user asked for the app to combine many variables and search on its own. After PEAD-SP500
  (+0.62% per trade over SPY before costs, about the 0.6% round trip) the next step is stacking
  conditions: fewer trades, possibly a larger edge each.
- New rules block `stock_combo` (catalog 1, `quantlab.strategies.stock_combo`): 15 conditions in
  8 groups (earnings reaction over SPY within 5 sessions 3/5/8%, 1/2 officers or directors buying
  ≥$10k in 30 days, 5-day return -5/-10%, RSI(14)<30 / RSI(2)<10, opening gap +3/+5%, above/below
  the 200-day average, SPY above its 200-day average, 52-week high). An entry is 1 to 3 conditions
  from different groups (465), with hold 5/10/20/30/60, stop 0/10/20%, take 0/10/20%, 10 or 20
  slots: 41,850 combinations. Volume was planned but the price tables carry no volume; the opening
  gap took its place.
- A stock is bought at the next open on the session all its conditions become true (not all true the
  session before), if it is an index member. More signals than free slots: a fixed seeded random
  order picks. Idle equity in SPY with `park_band` as PEAD-SP500.
- Condition tables are built once per dataset from data up to each row (`MarketView.panel`, read
  one row at a time); the leak test covers them.
- Policy `auto_sp500_v1`: AUTO-LEVP's filters, with the per-trade filters (gross ≥ 1.2%, top-10,
  t-value) read on the return over SPY held for the same days (`versus: SPY`), so market drift does
  not pass as an edge. prior_trials 2 (INS-CLUSTER, PEAD-SP500). Holdout 2020+, final evaluation by
  the user only.

## D-029 · 2026-10-17 · One pre-fixed rule from AUTO-SP500 (GAP-EARN-SP500)
- AUTO-SP500 ran all 41,850 combinations: 0 candidates. 97.5% of the 691 near misses that failed
  only the t-value contain an opening gap-up; earn8+gap5 cleared every other filter in 50 of its 54
  exit settings (Sharpe about 0.9 vs SPY 0.88: most equity sat in SPY between rare signals).
- The user chose to test a concentrated version before any holdout read. Fixed before running:
  earn8+gap5, hold 30, no stop or take, 5 slots (20% each), park SPY with park_band 1. Catalog 2 of
  `stock_combo` adds 5 slots; catalog 1 is unchanged.
- Policy `gap_earn_v1`: auto_sp500_v1's splits, costs and gates; `prior_trials: 41852` is added to
  the final evaluation's trial count (validate now reads a policy's `prior_trials`). AUTO-SP500 and
  GAP-EARN-SP500 share one holdout (SHARED_HOLDOUT).
- Evidence order: the 2010-2019 backtest first; the user decides whether to read the 2020+ holdout.

## D-030 · 2026-10-17 · Paper trading (모의 투자) on the forward holdout
- GAP-EARN-SP500's 2010-2019 backtest (EXP-20261017-97f9c7d27c): +2.26% per trade over SPY before
  costs (t 3.0, 233 trades), CAGR 18.3% vs 12.3%, but Sharpe 0.92 vs 0.88 and max drawdown -32% vs
  -18%, so a 2020+ final evaluation would almost surely fail Gate 3 on the drawdown ratio. The user
  chose to keep the holdout unread and paper-trade the rule on new data instead, inside the app
  (no broker account; their Kiwoom paper account runs another app).
- New page 모의 투자 (design option B, chosen by the user): the orders for the next open first, then
  the slots, the score against SPY since the start, closed trades and a day-by-day record.
- `quantlab.services.paper`: each refresh fetches today's data (SPY/BIL as before, the S&P 500
  bundle again unless fetched that UTC day, and the current members' SEC submissions main files,
  keeping the stored older pages) and runs the frozen rule with its policy's costs from
  `forward_holdout_start`. The data starts 300 sessions earlier for the indicators; the strategy is
  never asked for a decision before the start and nothing about those sessions is computed or shown.
  The last session used is the last one with prices for 90% of the members.
- Record in the work folder (`paper/<id>@<version>/`): ledger.json keeps every fill as first computed;
  a later refresh that computes a different past (revised prices, late filings) is listed on the page,
  and the first version stays. days/<as-of>.json keeps each refresh.
- Bundles (about 200 MB) and index datasets created by refreshes are deleted once a newer one exists,
  unless an experiment used them.
- The 2020+ holdout of these stocks includes the paper period. Once paper results have been seen, a
  final evaluation over 2020+ is not blind for those months; the verdict on this rule comes from the
  forward data.

## D-031 · 2026-10-17 · Automatic paper-trading refresh
- The user asked not to have to press the refresh button every day. A checkbox on 모의 투자 registers a
  Windows Task Scheduler entry (`quantlab.services.scheduler`) that runs `QuantLab.exe --paper-refresh`
  at 08:30 and 21:00 on the PC's clock; a run missed while the PC was off starts when it is next on.
  It runs as the signed-in user, so the stored Tiingo key and the chosen work folder are the same.
- The run refreshes every rule switched on and appends one line per rule to `paper/auto.log`.
  A lock file keeps the app and the scheduled run from refreshing at the same time.
- The S&P 500 bundle is downloaded again only when it lacks the latest SPY session (and was fetched
  over six hours earlier), so two runs a day do not mean two 200 MB downloads.
- Missing a day loses nothing: each refresh recomputes from the start date, and fills first seen
  late are recorded with the time they were seen.
- Fixed on the way: re-importing a spec from its stored copy in the work folder (a spec added with
  명세서 추가, then run from the strategy list) copied the file onto itself and failed.

## D-032 · 2026-10-17 · Turn-of-the-month rule on sector 3x ETFs (CAL-TOM-SECT3X)
- GAP-EARN-SP500 failed its final evaluation the same day (EXP-20261017-d6ae28622a); the user asked to
  go on with idea C of the 2026-10-16 plan (calendar effects).
- One rule, fixed before any run: hold SOXL, TNA, FAS, ERX, TECL, DRN, CURE at 7.14% each (50% in all)
  from the open of the last session of each month to the open of the 4th session of the next; the
  other half sits in BIL from the first session and is never traded. Sessions come from the NYSE
  schedule. New rules block `calendar` (`quantlab.strategies.calendar_tom`).
- Why these choices: the effect is about +0.1% a day on the market, below the 0.6% round trip on 1x
  funds; 3x funds lift it above the 1.2%-per-trade bar if it still exists. Half the equity keeps one
  bad window (2011-08-style) near -10% instead of -20% or worse. Pre-holiday and FOMC days were left
  out (one session each, too short for the cost).
- Policy cal_tom_v1 = osr_sect3x_v1's splits, costs and gates; holdout 2020+ (never read for these
  funds), forward holdout from 2026-10-18. Expected outcome stated before the run: likely fails on
  return against SPY, since the money is out of the market about 80% of the time.

## D-033 · 2026-10-17 · Automatic research factory (AUTO-SP500X)
- After A, B and C failed costs the user asked for strategies to be made and backtested "끊임 없이
  자동으로", and chose both an always-on factory in the app and new ingredients designed by Claude.
- Stock catalog 3 (`stock_combo`): 13 more conditions — earnings disappointment over SPY -5/-8%, one
  officer or director buying $250k+, opening gap down -3/-5%, SPY below its 200-day average, SPY down
  5% in five days, 52-week low, one-day drop -4/-7%, 20-day drop -15%, 3 and 5 down closes in a row,
  20-day volatility under 60% of 120-day — in 4 new groups, plus a 3-session hold and 5 slots.
  Catalogs 1 and 2 keep the same items in the same order. 3,337 entries, 540,594 combinations; the
  41,850 that AUTO-SP500 already ran are skipped and counted as prior trials.
- New group AUTO-SP500X, policy auto_sp500x_v1: AUTO-SP500's splits, costs, gates and filters;
  `order: affine` walks the grid as (a*i + b) mod N instead of storing a 540k-item shuffle;
  prior_trials 41,853 (AUTO-SP500, INS-CLUSTER, PEAD-SP500, GAP-EARN-SP500), so the t-value needed
  starts near 5.3. The holdout is shared with AUTO-SP500 and was spent by GAP-EARN-SP500.
- Verdicts come from forward paper trading only. The factory sends every candidate, and the five
  strongest rules that failed only the t-value (labelled as such), to 모의 투자 from the day after they
  were picked; these are refreshed with the scheduled paper refresh.
- `QuantLab.exe --factory` runs at idle priority from sign-in (and hourly, if closed), with no time
  limit, until switched off or done. The user asked for continuous running instead of nightly, and
  for no weekly report: the app shows a report when the whole grid is done, and a notice at once if
  a full candidate appears. Files in `<work folder>/factory/`.
- 2026-10-17, after release: the user's PC is a laptop and they asked for the factory to run only on
  the charger. The task is never started on battery and is started again on the power-source event
  (Kernel-Power 105); a running factory stops itself within 30 seconds of being unplugged (the
  switch stays on, unfinished trials run again next time).

## D-034 · 2026-10-19 · Factory round 2 with new information (AUTO-SP500Y)
- Round 1 (AUTO-SP500X) tried 498,744 combinations in 40.6 hours: none passed (t needed 5.21, best
  4.53). About 1.08 million rules have now been tried on the same prices and events, so the user
  chose a second round built on information the factory had not seen yet.
- Data: the SEC insider parser (PARSE_VERSION 2) also keeps open-market sales by officers and
  directors (code S, disposition D) as `insider_sell` events; dataset processing version 3. Every
  older rule filters events by kind, so their results do not change.
- Stock catalog 4: 5 more conditions — joined the S&P 500 in the last 20 sessions, left it in the
  last 20 sessions (bought even though it is no longer a member), 3+ officers or directors each
  selling $100k+ in 30 days, no officer or director sale in 90 days, 80–88 days after the last
  earnings filing (one to two weeks before the next). Catalogs 1–3 are unchanged. 5,621 entries,
  910,602 combinations; the 540,594 that rounds 0 and 1 ran are skipped, leaving 370,008.
- New group AUTO-SP500Y, policy auto_sp500y_v1: auto_sp500x_v1's rules, prior_trials 540,597,
  forward holdout from 2026-10-21. Holdout shared with AUTO-SP500, GAP-EARN-SP500 and AUTO-SP500X.
- The factory runs its groups in order and moves to the next when one is done; a report is shown
  for each round. A round 1 finished by build 54 moves on to round 2 by itself.
- Watch list fix: the factory keeps the five strongest t-only near misses of each round (the old
  version kept the first ones it found). Weaker ones leave 모의 투자's list but stay in the file; a
  round finished by an older build refreshes its list once from the final summary.

## D-035 · 2026-10-19 · Korean 2x leveraged and inverse ETFs (AUTO-KR2X)
- The user asked for a swing strategy on leveraged ETFs, then for KRX-listed funds, and to use the 2x
  inverse fund too so that results do not rest on one index going up. The US 3x line was closed on
  2026-10-16: its per-trade rebound edge replicated but lost to SPY on Sharpe and drawdown; the same
  edge at 1x (N4X-INTL-1X, +0.34% per trade, t 5.2) lost only to the 0.6% US round trip.
- KRX prices: new provider Naver Finance daily charts (`fchart.stock.naver.com`, no key), no
  distributions (dividend 0). A close step beyond ±35% is recorded as a split factor at the nearest KRX
  ratio (2, 2.5, 4, 5, 10, ... or reverse) when previous close / open is within 10% of it. Otherwise it is
  kept as a real move when the fund's leverage allows it (2x funds up to ±70%) and refused beyond that.
  The first downloads (2026-10-21) stopped on KODEX 레버리지's +47% on 2026-07-31; that was no split but
  a real day (KOSPI +17.9%, its largest daily rise). KRX codes (six characters starting with a digit) go to Naver,
  US tickers to Tiingo; KRX rows are checked against the XKRX schedule. The format was written from the
  public chart layout; the first download on the user's PC is the real format check.
- New group AUTO-KR2X (combo catalog 2 without the SPY market filters, 273,600 combinations): KODEX
  레버리지 122630, KODEX 코스닥150레버리지 233740, KODEX 200선물인버스2X 252670, slots 3, idle cash.
  The inverse fund is its own leg: an oversold inverse fund is an overbought market.
- Policy auto_kr2x_v1: KODEX 200 (069500) buy-and-hold as benchmark, Sharpe compared; no risk-free fund
  (idle cash earns 0; `risk_free_symbol: null` now means a zero series). Costs 1.5 bps commission +
  5 bps slippage one way (Kiwoom domestic online, no transaction tax on ETF sales; not yet checked
  against the user's account), stress at 2x; gross bar 0.5% per trade. Splits TRAIN 2010-03..2017,
  VALIDATION 2018..2021, HOLDOUT 2022+, forward from 2026-10-20. prior_trials 0, with the KORU overlap
  (AUTO-LEV/LEVP, D-024) disclosed. Pre-tax: leveraged/inverse funds may be "기타 ETF" for tax.
- Market filters that read another symbol are now validated against the filters a spec lists, not the
  whole catalog, so a spec without SPY can use catalog 2.

## D-036 · 2026-10-22 · Rules held for months: LOWVOL-SP500 and INS-HOLD6M
- After factory rounds 1–2 and AUTO-KR2X failed (every line lost its small per-trade edge to the
  0.6% round trip), the user kept the 0.25% commission and chose rules held for months. Instead of
  searching combinations, a few theory-based rules (protocol 제4조 A) are fixed before they are run.
- LOWVOL-SP500 1.0.0 (`lowvol` rules): at the last close of March, June, September and December the
  50 S&P 500 members with the lowest standard deviation of daily returns over 252 sessions are held at
  equal weight from the next open; everything else is sold. SPY is held until the first pick (data
  starts in 2010, so the first pick is early 2011). Policy lowvol_v1 = ins_cluster_v1's splits, costs
  and gates, with three neighbours fixed with the rule (top 30, top 100, 126-session volatility).
- INS-HOLD6M 1.0.0: INS-CLUSTER's signal (2+ officers or directors buying $10k+ within 30 days) held
  126 sessions in 20 slots, idle equity in SPY refilled two slots at a time (`park_band`, new optional
  key of the insider block; INS-CLUSTER is unchanged). Policy ins_hold6m_v1.
- Both families are new, so they have their own 2020+ holdout. The S&P 500 prices after 2020 were
  already used by GAP-EARN-SP500's final evaluation (a different rule); whether to open the holdout for
  these is the user's decision. Forward paper data counts from 2026-10-23.
