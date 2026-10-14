# QUANT LAB — research engine (MVP)

Research engine for discovering and validating quantitative strategies under the
QUANT LAB charter (`docs/charter/`). The plan this code follows is
`/mnt/project-files/plans/MVP_EXECUTION_PLAN.md`; technical decisions and their
rationale are logged in `docs/DECISIONS.md`.

## Setup

```bash
uv sync
uv run pytest
```

## Layout

- `src/quantlab/core` — shared types and hashing
- `src/quantlab/data` — provider adapters, quality checks, dataset versions
- `src/quantlab/strategies` — `Strategy` interface, point-in-time `MarketView`, baseline strategy
- `src/quantlab/backtesting` — fills, costs, cash/share accounting, independent reference implementation
- `src/quantlab/validation` — splits, metrics, statistical tests, gates, verdicts
- `src/quantlab/registry` — append-only experiment store and reproduction
- `src/quantlab/builtin` — bundled validation policies (frozen) and the baseline strategy spec

This repository holds application source only. Market data, experiment records, reports and
imported strategies live in the user's work folder (default `Documents\QuantLab`), never here.

Credentials are read from environment variables only and are never written to
logs, the registry or files.
