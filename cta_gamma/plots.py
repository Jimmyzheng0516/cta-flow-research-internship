"""Figure helpers. Plain matplotlib, one colour family, no decoration: the figures are
meant to be pasted into a report as they are."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
BLUE, GREY, RED = "#1f4e79", "#7f7f7f", "#a02c2c"


def save(fig, name: str):
    p = C.FIGS / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def curve_example(grid: pd.DataFrame, shelves: pd.DataFrame, ticker: str, day, sigma20: float, name: str):
    """One day's curve with the ±1% slope and the ±2-sigma readings marked."""
    g = grid[(grid.identifier == ticker) & (grid.timestamp == day)].sort_values("pct")
    s = shelves[(shelves.identifier == ticker) & (shelves.timestamp == day)]
    xs = np.concatenate([g.pct.to_numpy(), s.pct.to_numpy(), [0.0]])
    ys = np.concatenate([g.net.to_numpy(), s.net_at.to_numpy(), [0.0]])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(xs * 100, ys, color=BLUE, lw=1.5, label="cumulative net flow N(x)")
    ax.scatter(g.pct * 100, g.net, color=BLUE, s=14, zorder=3, label="reference grid rows")
    if len(s):
        ax.scatter(s.pct * 100, s.net_at, color=RED, s=14, marker="s", zorder=3, label="trigger shelves")
    for x, lab in [(1, "+1%"), (-1, "-1%")]:
        ax.axvline(x, color=GREY, ls=":", lw=0.8)
    for x in [2 * sigma20 * 100, -2 * sigma20 * 100]:
        ax.axvline(x, color=RED, ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("price move from the anchor (per cent)")
    ax.set_ylabel("contracts (net buying > 0)")
    ax.set_title(f"{ticker}, curve of {pd.Timestamp(day).date()}: dotted = ±1%, dashed = ±2 sigma")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return save(fig, name)


def bar_by_group(df: pd.DataFrame, value: str, err: str | None, title: str, ylabel: str, name: str,
                 index: str | None = None, figsize=(6.4, 3.4)):
    d = df.set_index(index) if index else df
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(d)), d[value], color=BLUE, yerr=(1.96 * d[err]) if err else None, capsize=2, ecolor=GREY)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.index, rotation=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(title); ax.set_ylabel(ylabel)
    return save(fig, name)


def quantile_plot(tabs: dict[str, pd.DataFrame], ycol: str, title: str, ylabel: str, name: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (lab, t) in enumerate(tabs.items()):
        ax.plot(t.index + 1, t[ycol], marker="o", lw=1.2, label=lab,
                color=[BLUE, RED, GREY, "#2e7d32", "#6a1b9a"][i % 5])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("within-market quintile of the measure (1 = lowest)")
    ax.set_xticks(range(1, len(next(iter(tabs.values()))) + 1))
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    return save(fig, name)


def time_series(df: pd.DataFrame, cols: list[str], labels: list[str], title: str, ylabel: str, name: str,
                figsize=(7.2, 3.4), log=False):
    fig, ax = plt.subplots(figsize=figsize)
    for c, lab, col in zip(cols, labels, [BLUE, RED, GREY, "#2e7d32"]):
        ax.plot(df.index, df[c], lw=1.0, label=lab, color=col)
    if log:
        ax.set_yscale("log")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=8, frameon=False)
    return save(fig, name)


def scatter_markets(x: pd.Series, y: pd.Series, labels: pd.Series, xlabel: str, ylabel: str, title: str, name: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(x, y, s=16, color=BLUE)
    for xi, yi, li in zip(x, y, labels):
        ax.annotate(li, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points", color=GREY)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    return save(fig, name)
