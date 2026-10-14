"""Charts shared by the app screens and the reports (03 v2.0.0 §3.3 F).

Every function returns a `matplotlib.figure.Figure` built without pyplot, so it
works headless (reports, CI) and inside a Qt canvas (GUI) alike.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

STRATEGY = "#3182F6"
BENCHMARK = "#B0B8C1"
HOLDOUT = "#FFD158"
INK = "#4E5968"
# Reports use the light palette; the app wraps chart creation in `dark()`.
LIGHT = {"benchmark": BENCHMARK, "ink": INK, "spine": "#E5E8EB", "grid": "#F2F4F6", "rc": {}}
DARK = {"benchmark": "#5E5E6A", "ink": "#9E9EA8", "spine": "#2C2C35", "grid": "#26262E",
        "rc": {"figure.facecolor": "#1C1C22", "axes.facecolor": "#1C1C22", "savefig.facecolor": "#1C1C22",
               "text.color": "#B0B0BA", "axes.labelcolor": "#B0B0BA", "xtick.color": "#9E9EA8",
               "ytick.color": "#9E9EA8", "axes.edgecolor": "#2C2C35", "legend.labelcolor": "#B0B0BA"}}
_P = LIGHT


@contextmanager
def dark():
    """Build figures in the app's dark colors (colors are fixed when a figure is created)."""
    global _P
    old, _P = _P, DARK
    try:
        with matplotlib.rc_context(DARK["rc"]):
            yield
    finally:
        _P = old


EN = {"strategy": "strategy", "benchmark": "benchmark", "holdout": "final holdout"}
KO = {"strategy": "전략", "benchmark": "벤치마크", "holdout": "최종 평가 구간"}


def _fig(w=8.0, h=3.6) -> Figure:
    fig = Figure(figsize=(w, h), dpi=110, layout="constrained")
    return fig


def _clean(ax) -> None:
    """Quiet axes: no box, light grid, gray tick labels."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_P["spine"])
    ax.tick_params(colors=_P["ink"], labelsize=9, length=0)
    ax.grid(axis="y", color=_P["grid"], lw=1)
    ax.set_axisbelow(True)


def equity(eq: pd.DataFrame, holdout_start=None, title="Cumulative return (log scale)", names=EN) -> Figure:
    fig = _fig()
    ax = fig.add_subplot()
    norm = eq / eq.iloc[0]
    ax.plot(norm.index, norm["benchmark"], color=_P["benchmark"], lw=1.2, label=names["benchmark"])
    ax.plot(norm.index, norm["strategy"], color=STRATEGY, lw=1.8, label=names["strategy"])
    if holdout_start is not None:
        ax.axvspan(pd.Timestamp(holdout_start), norm.index[-1], color=HOLDOUT, alpha=0.18, lw=0, label=names["holdout"])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{(y - 1) * 100:+.0f}%"))
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    _clean(ax)
    ax.legend(frameon=False, fontsize=9)
    return fig


def drawdown(eq: pd.DataFrame, title="Drawdown", names=EN) -> Figure:
    fig = _fig(8.0, 2.6)
    ax = fig.add_subplot()
    for col, color in (("benchmark", _P["benchmark"]), ("strategy", STRATEGY)):
        dd = eq[col] / eq[col].cummax() - 1
        ax.plot(dd.index, dd * 100, color=color, lw=1.2, label=names[col])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    _clean(ax)
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
    im = ax.imshow(tab.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(tab.columns)), [str(c) for c in tab.columns])
    ax.set_yticks(range(len(tab.index)), [str(i) for i in tab.index])
    for (i, j), v in np.ndenumerate(tab.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.tick_params(colors=_P["ink"], labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
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
        _clean(ax)
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
