"""Charts shared by the app screens and the reports (03 v2.0.0 §3.3 F).

Every function returns a `matplotlib.figure.Figure` built without pyplot, so it
works headless (reports, CI) and inside a Qt canvas (GUI) alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

STRATEGY = "#2a6fdb"
BENCHMARK = "#8a8f98"
HOLDOUT = "#f2c94c"


def _fig(w=8.0, h=3.6) -> Figure:
    fig = Figure(figsize=(w, h), dpi=110, layout="constrained")
    return fig


def equity(eq: pd.DataFrame, holdout_start=None, title="Cumulative growth (start = 1)") -> Figure:
    fig = _fig()
    ax = fig.add_subplot()
    norm = eq / eq.iloc[0]
    ax.plot(norm.index, norm["strategy"], color=STRATEGY, lw=1.6, label="strategy")
    ax.plot(norm.index, norm["benchmark"], color=BENCHMARK, lw=1.2, label="benchmark")
    if holdout_start is not None:
        ax.axvspan(pd.Timestamp(holdout_start), norm.index[-1], color=HOLDOUT, alpha=0.18, lw=0, label="final holdout")
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    return fig


def drawdown(eq: pd.DataFrame, title="Drawdown") -> Figure:
    fig = _fig(8.0, 2.6)
    ax = fig.add_subplot()
    for col, color in (("strategy", STRATEGY), ("benchmark", BENCHMARK)):
        dd = eq[col] / eq[col].cummax() - 1
        ax.plot(dd.index, dd * 100, color=color, lw=1.2, label=col)
    ax.set_ylabel("%")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    return fig


def monthly_table(eq: pd.Series) -> pd.DataFrame:
    m = eq.resample("ME").last().pct_change().dropna()
    t = pd.DataFrame({"year": m.index.year, "month": m.index.month, "r": m.values})
    return t.pivot(index="year", columns="month", values="r")


def monthly_returns(eq: pd.Series, title="Monthly returns (strategy, %)") -> Figure:
    tab = monthly_table(eq) * 100
    fig = _fig(8.0, max(2.4, 0.28 * len(tab) + 1))
    ax = fig.add_subplot()
    lim = np.nanmax(np.abs(tab.values)) if tab.size else 1
    im = ax.imshow(tab.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(tab.columns)), [str(c) for c in tab.columns])
    ax.set_yticks(range(len(tab.index)), [str(i) for i in tab.index])
    for (i, j), v in np.ndenumerate(tab.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, loc="left", fontsize=11)
    return fig


def parameter_sensitivity(rows: list[dict], metric="sharpe_diff", title=None) -> Figure:
    """1 parameter → line; 2 parameters → heatmap. Rows from a batch summary."""
    done = [r for r in rows if r.get(metric) is not None]
    names = sorted({k for r in done for k in r["params"]})
    fig = _fig(6.5, 3.4)
    ax = fig.add_subplot()
    ax.set_title(title or f"Parameter sensitivity: {metric}", loc="left", fontsize=11)
    if len(names) == 1:
        xs = [r["params"][names[0]] for r in done]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array([r[metric] for r in done])[order], marker="o", color=STRATEGY)
        ax.axhline(0, color=BENCHMARK, lw=0.8)
        ax.set_xlabel(names[0])
        ax.grid(alpha=0.25)
    elif len(names) == 2:
        df = pd.DataFrame([{**r["params"], metric: r[metric]} for r in done]).pivot(index=names[0], columns=names[1],
                                                                                   values=metric)
        im = ax.imshow(df.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(df.columns)), [str(c) for c in df.columns])
        ax.set_yticks(range(len(df.index)), [str(i) for i in df.index])
        ax.set_xlabel(names[1])
        ax.set_ylabel(names[0])
        fig.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "sensitivity chart needs 1 or 2 searched parameters", ha="center", transform=ax.transAxes)
    return fig


def save(fig: Figure, path) -> None:
    fig.savefig(path)
