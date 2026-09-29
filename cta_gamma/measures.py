"""The CTA gamma measures, read off the Table 3 managed-money curve.

Definitions (per market i and curve day t; the curve is anchored at the close of t
and describes session t+1). N(x) is the cumulative signed managed-money flow, in
contracts, if price moves by the fraction x from the anchor: positive = net buying.
N(0) = 0 by construction. Between delivered rows (the ±1..5% reference grid and the
trigger shelves) N is linearly interpolated; that is the same reading of the curve
that Strategy 5 uses to evaluate "fired flow".

  up_pct   = N(+1%)                contracts bought on a one per cent rally
  dn_pct   = -N(-1%)               contracts sold on a one per cent decline
  gamma_pct = (up_pct + dn_pct)/2  contracts traded per one per cent move: the literal
                                   "CTA gamma" of the proposal
  up_sig, dn_sig, gamma_sig        the same read at ±sigma20 instead of ±1%, i.e. per
                                   one-standard-deviation move (volatility-adjusted)
  up_2s, dn_2s                     N(+2 sigma), -N(-2 sigma): "forced flow at two sigma"
  adverse_2s / favour_2s           up_2s or dn_2s picked by the sign of the net position:
                                   the flow that would *unwind* the book on a 2-sigma
                                   adverse move, and the flow that would *add* to it
  asym_2s  = up_2s - dn_2s         flow-implied asymmetry (positive = more forced buying
                                   above than forced selling below)
  cliff_2s                         absolute shelf jumps within ±2 sigma (concentrated flow)
  expected_flow                    E|N(r)| for r ~ Normal(0, sigma20): the mechanical
                                   volume the curve implies for an ordinary session
  reach_5                          N(+5%) - N(-5%): all the flow the delivered curve describes

Each contract quantity is then expressed as a share of trailing average daily volume
(`_adv`), of open interest (`_oi`) and, where it makes sense, of the absolute net
position (`_pos`). The ADV share is the headline, as proposed; OI is the house
convention in the white paper and is carried for comparison.

A note on sign. Managed money buys into rallies and sells into declines (white paper
Section 5.3), so gamma_pct is positive almost everywhere. In the option-dealer analogy
this is the *short-gamma* (destabilising) side; we keep the natural sign, "contracts
traded per unit move", and state the analogy in words rather than flipping signs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(9)     # probabilists' Hermite, 9 nodes
_GH_W = _GH_W / _GH_W.sum()


def _merged_curve(gp, gn, sp, sn):
    xs = np.concatenate([gp, sp, [0.0]])
    ys = np.concatenate([gn, sn, [0.0]])
    o = np.argsort(xs)
    return xs[o], ys[o]


def curve_measures_market(ticker: str, anchors: pd.DataFrame, grid: pd.DataFrame,
                          shelves: pd.DataFrame, sigma20: pd.Series) -> pd.DataFrame:
    """All curve measures for one market. `sigma20` is indexed by curve day (info through t)."""
    g = grid[grid["identifier"] == ticker]
    s = shelves[shelves["identifier"] == ticker]
    a = anchors[anchors["identifier"] == ticker].set_index("timestamp")["current_price"]
    g_by = {ts: (sub["pct"].to_numpy(), sub["net"].to_numpy()) for ts, sub in g.groupby("timestamp")}
    s_by = {ts: (sub["pct"].to_numpy(), sub["net_at"].to_numpy(), sub["jump"].to_numpy())
            for ts, sub in s.groupby("timestamp")}
    rows = []
    for ts in sorted(g_by):
        gp, gn = g_by[ts]
        sp, sn, sj = s_by.get(ts, (np.array([]), np.array([]), np.array([])))
        xs, ys = _merged_curve(gp, gn, sp, sn)
        f = lambda x: float(np.interp(x, xs, ys))
        sg = float(sigma20.get(ts, np.nan))
        row = dict(identifier=ticker, timestamp=ts, anchor=float(a.get(ts, np.nan)),
                   n_shelves=int(len(sp)),
                   up_pct=f(0.01), dn_pct=-f(-0.01), reach_5=f(0.05) - f(-0.05))
        if np.isfinite(sg) and sg > 0:
            k = C.SHOCK_SIGMAS
            row.update(up_sig=f(sg), dn_sig=-f(-sg), up_2s=f(k * sg), dn_2s=-f(-k * sg),
                       cliff_2s=float(np.abs(sj[np.abs(sp) <= k * sg]).sum()),
                       n_shelves_2s=int((np.abs(sp) <= k * sg).sum()),
                       expected_flow=float(np.sum(_GH_W * np.abs(np.interp(_GH_X * sg, xs, ys)))))
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ["up_sig", "dn_sig", "up_2s", "dn_2s", "cliff_2s", "n_shelves_2s", "expected_flow"]:
        if c not in out.columns:
            out[c] = np.nan
    out["gamma_pct"] = (out["up_pct"] + out["dn_pct"]) / 2.0
    out["gamma_sig"] = (out["up_sig"] + out["dn_sig"]) / 2.0
    out["asym_2s"] = out["up_2s"] - out["dn_2s"]
    return out


def build_measures(sens: int = C.SENS, verbose: bool = True) -> pd.DataFrame:
    """Curve measures for every market in the universe, joined to volume, OI and position.

    Cached as jimmy_intern/cache/measures_s{sens}.parquet. One row per (market, curve day).
    """
    f = C.CACHE / f"measures_s{sens}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    anchors, grid, shelves = D.curve_tables(sens)
    pos = D.load_positions()
    frames = []
    names = D.universe(sens)
    for i, t in enumerate(names, 1):
        px = D.load_prices(t)
        rr = D.session_returns(px)
        m = curve_measures_market(t, anchors, grid, shelves, rr["sigma20"])
        if m.empty:
            continue
        m = m.merge(px[["volume", "oi", "adv", "close_raw"]], left_on="timestamp", right_index=True, how="left")
        m = m.merge(rr[["sigma20"]], left_on="timestamp", right_index=True, how="left")
        p = pos[pos["identifier"] == t].set_index("timestamp")["net_position"]
        m["net_position"] = p.reindex(m["timestamp"]).to_numpy()
        m["sector"] = D.sector_of(t)
        m["multiplier"] = D.multiplier_of(t)
        m["usd"] = D.is_usd(t)
        frames.append(m)
        if verbose and (i % 20 == 0 or i == len(names)):
            print(f"  curve measures {i}/{len(names)} markets")
    out = pd.concat(frames, ignore_index=True).sort_values(["identifier", "timestamp"]).reset_index(drop=True)
    out = add_normalisations(out)
    out.to_parquet(f, index=False)
    return out


def add_normalisations(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    abs_pos = m["net_position"].abs().clip(lower=1.0)
    m["pos_oi"] = m["net_position"] / m["oi"]
    m["abs_pos_oi"] = m["pos_oi"].abs()
    long_book = m["net_position"] >= 0
    m["adverse_2s"] = np.where(long_book, m["dn_2s"], m["up_2s"])
    m["favour_2s"] = np.where(long_book, m["up_2s"], m["dn_2s"])
    for c in ["gamma_pct", "gamma_sig", "up_pct", "dn_pct", "up_sig", "dn_sig", "up_2s", "dn_2s",
              "adverse_2s", "favour_2s", "asym_2s", "cliff_2s", "expected_flow", "reach_5"]:
        m[c + "_adv"] = m[c] / m["adv"]
        m[c + "_oi"] = m[c] / m["oi"]
    m["adverse_2s_pos"] = m["adverse_2s"] / abs_pos
    m["expected_flow_pos"] = m["expected_flow"] / abs_pos
    # notional in USD for USD-denominated contracts (multiplier x raw price)
    usd_mult = np.where(m["usd"], m["multiplier"] * m["close_raw"], np.nan)
    for c in ["up_2s", "dn_2s", "gamma_pct", "expected_flow"]:
        m[c + "_usd"] = m[c] * usd_mult
    return m


def trailing_z(s: pd.Series, window: int = C.Z_WINDOW, min_periods: int = C.Z_MIN) -> pd.Series:
    """Within-market trailing z-score: uses only the past `window` sessions, so it is point-in-time."""
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)
