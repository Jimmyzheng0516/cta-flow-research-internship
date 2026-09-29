"""The estimation panel: one row per (market, curve day t) with

* the curve measures of day t (measures.py), all known by 07:00 UTC on t+1;
* controls known at the close of t (the volatility benchmark, public liquidity
  variables, trend strength, days to delivery, the Table 2 crowding level);
* targets measured strictly after the close of t (next-session absolute return and
  range, realised volatility over the next 5 and 21 sessions, tail flags);
* the mechanical flow that actually fired in session t+1 when day t's curve is walked
  along session t+1's realised path (the Strategy 5 quantity), used to check that the
  measures forecast the flow they claim to describe.

Volatility benchmark.  There is no intraday data, so realised variance is proxied by
the Parkinson range estimator, range^2 / (4 ln 2), which is the standard daily-bar
estimator and is roughly five times as efficient as the squared close-to-close return.
The benchmark forecast is the heterogeneous autoregressive (HAR) model of Corsi
(2009): the log of tomorrow's variance regressed on the logs of the trailing
one-day, five-day and 22-day average Parkinson variances. That is the model a
volatility desk would actually run on daily bars, and it is the null every measure
here has to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import measures as M

LN4 = 4.0 * np.log(2.0)
HORIZONS = (5, 21)
Z_COLS = ["gamma_pct_adv", "gamma_sig_adv", "gamma_pct_oi", "gamma_sig_oi",
          "up_2s_adv", "dn_2s_adv", "adverse_2s_adv", "favour_2s_adv", "adverse_2s_oi",
          "adverse_2s_pos", "asym_2s_adv", "cliff_2s_adv", "expected_flow_adv", "expected_flow_oi",
          "abs_pos_oi", "reach_5_oi"]


def fired_flow_next(ticker: str, anchors, grid, shelves, px: pd.DataFrame) -> pd.Series:
    """Day-t curve evaluated at session t+1's realised path, indexed by curve day t.

    Same rule as backtest/strategy5_utils.mechanical_flow_market: interpolate the merged
    curve at the close-to-close move, then add the jumps of any shelves the session's
    high/low touched beyond the closed segment (stop orders fire on touch).
    """
    g = grid[grid["identifier"] == ticker]
    s = shelves[shelves["identifier"] == ticker]
    g_by = {ts: (sub["pct"].to_numpy(), sub["net"].to_numpy()) for ts, sub in g.groupby("timestamp")}
    s_by = {ts: (sub["pct"].to_numpy(), sub["net_at"].to_numpy(), sub["jump"].to_numpy())
            for ts, sub in s.groupby("timestamp")}
    a = anchors[anchors["identifier"] == ticker].set_index("timestamp")["current_price"]
    idx = px.index
    c, h, l = px["close_adj"].to_numpy(), px["high_adj"].to_numpy(), px["low_adj"].to_numpy()
    out = {}
    for i in range(len(idx) - 1):
        d = idx[i]
        if d not in g_by:
            continue
        anchor = a.get(d, np.nan)
        if not np.isfinite(anchor) or anchor <= 0:
            continue
        r_c, r_h, r_l = (c[i + 1] - c[i]) / anchor, (h[i + 1] - c[i]) / anchor, (l[i + 1] - c[i]) / anchor
        if not np.isfinite([r_c, r_h, r_l]).all() or abs(r_c) > 0.5 or r_h < r_l:
            continue
        gp, gn = g_by[d]
        sp, sn, sj = s_by.get(d, (np.array([]), np.array([]), np.array([])))
        xs, ys = M._merged_curve(gp, gn, sp, sn)
        mech = float(np.interp(r_c, xs, ys))
        lo, hi = min(0.0, r_c), max(0.0, r_c)
        touched = (sp >= r_l) & (sp <= r_h)
        outside = touched & ((sp < lo) | (sp > hi))
        out[d] = mech + float(sj[outside].sum())
    return pd.Series(out, name="fired_next", dtype=float)


def build_panel(sens: int = C.SENS, verbose: bool = True) -> pd.DataFrame:
    f = C.CACHE / f"panel_s{sens}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    meas = M.build_measures(sens, verbose=verbose)
    anchors, grid, shelves = D.curve_tables(sens)
    frames = []
    names = sorted(meas["identifier"].unique())
    for i, t in enumerate(names, 1):
        px = D.load_prices(t)
        rr = D.session_returns(px)
        r, rng = rr["r"], rr["range"]
        pv = (rng ** 2) / LN4                                   # Parkinson variance, session t
        ctl = pd.DataFrame(index=px.index)
        ctl["l_pv_d"] = np.log(pv.clip(lower=1e-10))
        ctl["l_pv_w"] = np.log(pv.rolling(5).mean().clip(lower=1e-10))
        ctl["l_pv_m"] = np.log(pv.rolling(22).mean().clip(lower=1e-10))
        ctl["l_turnover"] = np.log((px["adv"] / px["oi"]).clip(lower=1e-6))
        ctl["l_vol_rel"] = np.log(px["volume"].clip(lower=1.0)) - np.log(px["adv"].clip(lower=1.0))
        ma = px["close_adj"].rolling(120).mean()
        ctl["trend_abs"] = ((px["close_adj"] - ma) / px["close_raw"]).abs() / rr["sigma20"]
        ctl["l_dte"] = np.log(D.days_to_delivery(px).astype(float))
        # targets, strictly after the close of t
        tg = pd.DataFrame(index=px.index)
        tg["absr_1"] = r.abs().shift(-1)
        tg["range_1"] = rng.shift(-1)
        tg["r_1"] = r.shift(-1)
        for hzn in HORIZONS:
            tg[f"rv_{hzn}"] = np.sqrt((r ** 2).rolling(hzn).mean().shift(-hzn))
            tg[f"prv_{hzn}"] = np.sqrt(pv.rolling(hzn).mean().shift(-hzn))
            tg[f"cum_r_{hzn}"] = r.rolling(hzn).sum().shift(-hzn)
        tg["tail_1"] = (tg["absr_1"] > 2.0 * rr["sigma20"]).astype(float).where(tg["absr_1"].notna())
        tg["down_1"] = (tg["r_1"] < -2.0 * rr["sigma20"]).astype(float).where(tg["r_1"].notna())
        tg["up_1"] = (tg["r_1"] > 2.0 * rr["sigma20"]).astype(float).where(tg["r_1"].notna())
        # fired flow in t+1 from day-t curve
        ff = fired_flow_next(t, anchors, grid, shelves, px)
        d = meas[meas["identifier"] == t].merge(ctl, left_on="timestamp", right_index=True, how="left")
        d = d.merge(tg, left_on="timestamp", right_index=True, how="left")
        d["fired_next"] = ff.reindex(d["timestamp"]).to_numpy()
        frames.append(d)
        if verbose and (i % 20 == 0 or i == len(names)):
            print(f"  panel {i}/{len(names)} markets")
    pan = pd.concat(frames, ignore_index=True).sort_values(["identifier", "timestamp"]).reset_index(drop=True)
    pan["fired_next_adv"] = pan["fired_next"].abs() / pan["adv"]
    g = pan.groupby("identifier", group_keys=False)
    for c in Z_COLS:
        pan["z_" + c] = g[c].transform(M.trailing_z)
    pan = pan.replace([np.inf, -np.inf], np.nan)
    pan.to_parquet(f, index=False)
    return pan


def commodity_rows(pan: pd.DataFrame) -> pd.DataFrame:
    return pan[pan["sector"].isin(C.COMMODITY_SECTORS)]
