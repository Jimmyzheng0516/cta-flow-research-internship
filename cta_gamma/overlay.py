"""The value test: does a volatility forecast that uses the positioning data make a
volatility-targeted futures book better?

A volatility-targeted book holds each market in inverse proportion to a forecast of
its volatility, so a forecast matters through sizing, not through direction. We build
the two simplest such books a client would recognise on the commodity universe:

  * a long-only equal-risk basket (every market held long, sized to the same risk);
  * a long-short trend basket (sign of the 120-session price change, sized the same way);

and size each with (a) the trailing 20-session volatility, the convention every earlier
strategy in this repository uses, (b) the HAR benchmark forecast, (c) HAR plus the
public state variables, and (d) HAR plus the public state variables plus the CTA gamma.
The forecasts are the year-by-year out-of-sample ones from stats.oos_forecast (21-day
horizon), so no forecast uses information after its date.

Two things are scored. First, how well each rule keeps every market position at its
risk target, measured as the dispersion of realised volatility across market-years
(the forecast's own job). Second, the Sharpe ratio of the book. The warning of
Cederburg, O'Doherty, Wang and Yan (2020) applies: volatility-managed gains are usually
much smaller out of sample than in sample, and that is why only out-of-sample forecasts
are used here.

Clock: positions are set at the close of t+1 (day t's curve is delivered the morning of
t+1) and earn the return of session t+2 onwards, the Strategy 5 / 6 execution clock.
Costs: 1 basis point of notional one way on turnover, the same simple rule for every
variant, because the object under test is the sizing rule and not the cost model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

COST_BP_ONE_WAY = 1.0
TARGET_DAILY_VOL = 0.10 / np.sqrt(252)   # 10% annualised per market position
MAX_LEVERAGE = 5.0                       # cap on target / forecast, per market


def book_inputs(pan: pd.DataFrame, oos_rows: pd.DataFrame, pred_cols: dict[str, str],
                sectors=C.COMMODITY_SECTORS) -> pd.DataFrame:
    """Rows: (market, curve day t) with the out-of-sample forecasts made on t, sigma20 on t,
    the trend sign known on t, and the return of session t+2 (the first session a position
    set at the close of t+1 can earn)."""
    d = pan[pan["sector"].isin(sectors)][["identifier", "timestamp", "sigma20", "r_1"]].copy()
    d = d.merge(oos_rows[["identifier", "timestamp"] + list(pred_cols.values())],
                on=["identifier", "timestamp"], how="inner").sort_values(["identifier", "timestamp"])
    d["r_exec"] = d.groupby("identifier")["r_1"].shift(-1)          # r_{t+2}
    for lab, col in pred_cols.items():
        d["vol_" + lab] = np.exp(d[col])
    d["vol_sigma20"] = d["sigma20"]
    trend = []
    for t, sub in d.groupby("identifier"):
        px = D.load_prices(t)
        s = np.sign(px["close_adj"] - px["close_adj"].shift(120))
        trend.append(pd.Series(s.reindex(sub["timestamp"]).to_numpy(), index=sub.index))
    d["trend_sign"] = pd.concat(trend)
    return d


def run_book(d: pd.DataFrame, vol_col: str, long_short: bool = False) -> dict:
    x = d.dropna(subset=[vol_col, "r_exec"]).copy()
    x["lev"] = (TARGET_DAILY_VOL / x[vol_col].clip(lower=1e-5)).clip(upper=MAX_LEVERAGE)
    x["w"] = x["lev"] * (x["trend_sign"].fillna(0.0) if long_short else 1.0)
    W = x.pivot_table(index="timestamp", columns="identifier", values="w").fillna(0.0)
    R = x.pivot_table(index="timestamp", columns="identifier", values="r_exec").reindex(index=W.index, columns=W.columns).fillna(0.0)
    n_live = (W != 0).sum(axis=1).clip(lower=1)
    W = W.div(n_live, axis=0)                                   # equal-risk, unit gross when all live
    gross = (W * R).sum(axis=1)
    cost = (W - W.shift().fillna(0.0)).abs().sum(axis=1) * COST_BP_ONE_WAY / 1e4
    net = gross - cost
    # the forecast's own job: realised vol of each scaled position vs its target, by market-year
    x["scaled"] = x["lev"] * x["r_exec"]
    my = x.assign(year=x["timestamp"].dt.year).groupby(["identifier", "year"])["scaled"].std() * np.sqrt(252)
    my = my[x.assign(year=x["timestamp"].dt.year).groupby(["identifier", "year"]).size() >= 100]
    ratio = my / 0.10
    return {"gross": gross, "net": net, "cost": cost, "W": W, "ratio": ratio,
            "sharpe_gross": float(gross.mean() / gross.std() * np.sqrt(252)),
            "sharpe_net": float(net.mean() / net.std() * np.sqrt(252)),
            "ann_ret_net": float(net.mean() * 252), "ann_vol": float(net.std() * np.sqrt(252)),
            "max_dd": float(((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()),
            "turnover": float((W - W.shift().fillna(0.0)).abs().sum(axis=1).mean()),
            "tracking_rmse": float(np.sqrt(((ratio - 1.0) ** 2).mean())),
            "tracking_mad": float((ratio - 1.0).abs().median())}


def summarise(results: dict[str, dict]) -> pd.DataFrame:
    keys = ["sharpe_gross", "sharpe_net", "ann_ret_net", "ann_vol", "max_dd", "turnover", "tracking_rmse", "tracking_mad"]
    return pd.DataFrame({k: {m: v[m] for m in keys} for k, v in results.items()}).T
