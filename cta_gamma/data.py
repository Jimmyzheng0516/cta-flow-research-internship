"""Loaders: prices, Table 3 curves, Table 2 positions, and the product mapping.

Conventions (all inherited from the earlier work in this repository):

* Prices.  Two Norgate files per market. `&T.csv` is the raw continuous front
  contract (its close on day t is the anchor price of that day's Table 3 curve,
  exactly, in every market checked). `&T_CCB.csv` is the back-adjusted series;
  the adjustment is additive, so price *differences* are exact and returns are
  computed as adjusted differences divided by the raw close of the previous
  session (the Strategy 6 convention, see backtest/strategy6_utils.py). Volume and
  open interest are total across delivery months.
* Timing.  A Table 3 row stamped day t is anchored at the close of t, delivered
  at 07:00 UTC on t+1, and describes what managed money would do during session
  t+1. Every target in this project is measured strictly after the close of t.
* Sensitivity.  The managed-money product has four sensitivities; 3 is the white
  paper's default and the headline here, 2 (used by Strategy 5) is the robustness run.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------
# product mapping
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def mapping() -> pd.DataFrame:
    m = pd.read_csv(C.MAPPING_FILE)
    return m.set_index("identifier")


def sector_of(ticker: str) -> str:
    m = mapping()
    return m.loc[ticker, "sector"] if ticker in m.index else "?"


def multiplier_of(ticker: str) -> float:
    m = mapping()
    return float(m.loc[ticker, "multiplier"]) if ticker in m.index else np.nan


def is_usd(ticker: str) -> bool:
    m = mapping()
    if ticker not in m.index:
        return False
    fx = m.loc[ticker, "fx"]
    return not isinstance(fx, str) or fx.strip() == ""


# ----------------------------------------------------------------------------
# prices
# ----------------------------------------------------------------------------
def has_prices(ticker: str) -> bool:
    return (C.PRICE_DIR / f"&{ticker}.csv").exists() and (C.PRICE_DIR / f"&{ticker}_CCB.csv").exists()


@lru_cache(maxsize=None)
def load_prices(ticker: str) -> pd.DataFrame:
    """One row per session with raw close, adjusted OHLC, volume, OI, delivery month.

    Columns: close_raw, open_adj, high_adj, low_adj, close_adj, volume, oi,
    delivery_month (YYYYMM integer), adv (trailing ADV_WINDOW mean of volume).
    """
    raw = pd.read_csv(C.PRICE_DIR / f"&{ticker}.csv", parse_dates=["Date"]).set_index("Date")
    adj = pd.read_csv(C.PRICE_DIR / f"&{ticker}_CCB.csv", parse_dates=["Date"]).set_index("Date")
    px = pd.DataFrame({
        "close_raw": raw["Close"],
        "open_adj": adj["Open"], "high_adj": adj["High"], "low_adj": adj["Low"], "close_adj": adj["Close"],
        "volume": raw["Volume"].astype(float),
        "oi": raw["Open Interest"].astype(float).replace(0.0, np.nan).ffill(),
        "delivery_month": raw["Delivery Month"],
    }).sort_index()
    px.index.name = "timestamp"
    px["adv"] = px["volume"].replace(0.0, np.nan).rolling(C.ADV_WINDOW, min_periods=C.ADV_MIN).mean()
    return px


def session_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close return and intraday range, both as fractions of the prior raw close.

    r_t     = (close_adj_t - close_adj_{t-1}) / close_raw_{t-1}
    range_t = (high_adj_t - low_adj_t)        / close_raw_{t-1}
    The 0.5 guard removes the handful of distorted rows the house hygiene mask also removes.
    """
    prev_raw = px["close_raw"].shift(1)
    r = (px["close_adj"] - px["close_adj"].shift(1)) / prev_raw
    rng = (px["high_adj"] - px["low_adj"]) / prev_raw
    ok = (prev_raw > 0) & (r.abs() < 0.5) & (rng >= 0) & (rng < 1.0)
    out = pd.DataFrame({"r": r.where(ok), "range": rng.where(ok)})
    out["sigma20"] = out["r"].rolling(C.VOL_WINDOW, min_periods=10).std()
    out.loc[~out["sigma20"].between(1e-4, 0.5), "sigma20"] = np.nan
    return out


def days_to_delivery(px: pd.DataFrame) -> pd.Series:
    """Calendar days from the session date to the first day of the front contract's delivery month."""
    dm = px["delivery_month"].astype("Int64")
    first = pd.to_datetime(dm.astype(str).str.slice(0, 4) + "-" + dm.astype(str).str.slice(4, 6) + "-01", errors="coerce")
    return (first - px.index.to_series()).dt.days.clip(lower=1)


# ----------------------------------------------------------------------------
# Table 3 curves
# ----------------------------------------------------------------------------
def load_scenarios(sens: int = C.SENS) -> pd.DataFrame:
    cols = ["identifier", "timestamp", "scenario_type", "scenario_label", "shock_price",
            f"buy_contracts_sensitivity_{sens}", f"sell_contracts_sensitivity_{sens}"]
    raw = pd.read_parquet(C.SCENARIO_FILE, columns=cols).rename(columns={
        f"buy_contracts_sensitivity_{sens}": "buy", f"sell_contracts_sensitivity_{sens}": "sell"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["net"] = raw["buy"] - raw["sell"]
    return raw


def curve_tables(sens: int = C.SENS):
    """Collapse the raw ladder into three flat tables (cached under jimmy_intern/cache).

    anchors: identifier, timestamp, current_price
    grid:    identifier, timestamp, pct, net         (the ±1..5% reference rows)
    shelves: identifier, timestamp, pct, net_at, jump (trigger rows; jump = net_at - net_pre)
    `pct` is the fractional distance of the row's price from the anchor. `net` is the
    cumulative signed managed-money flow (buy minus sell contracts) at that price.

    This is the same reduction Strategy 5 used (backtest/strategy5_utils.build_curve_tables);
    it is restated here so this project has no import dependency on the backtest folder.
    """
    fa, fg, fs = (C.CACHE / f"anchors_s{sens}.parquet", C.CACHE / f"grid_s{sens}.parquet",
                  C.CACHE / f"shelves_s{sens}.parquet")
    if fa.exists() and fg.exists() and fs.exists():
        return pd.read_parquet(fa), pd.read_parquet(fg), pd.read_parquet(fs)
    raw = load_scenarios(sens)
    cur = (raw[raw["scenario_type"] == "current"].drop_duplicates(["identifier", "timestamp"])
           [["identifier", "timestamp", "shock_price"]].rename(columns={"shock_price": "current_price"}))
    grid = raw[raw["scenario_type"] == "reference"].merge(cur, on=["identifier", "timestamp"])
    grid["pct"] = grid["shock_price"] / grid["current_price"] - 1.0
    grid = grid[["identifier", "timestamp", "pct", "net"]]
    trig = raw[raw["scenario_type"] == "trigger"].copy()
    is_pre = trig["scenario_label"].str.endswith("_pre")
    fire, pre = trig[~is_pre].copy(), trig[is_pre].copy()
    pre["base"] = pre["scenario_label"].str.replace("_pre", "", regex=False)
    fire["base"] = fire["scenario_label"]
    shelves = fire.merge(pre[["identifier", "timestamp", "base", "net"]],
                         on=["identifier", "timestamp", "base"], suffixes=("", "_pre"), how="left")
    shelves["jump"] = shelves["net"] - shelves["net_pre"]
    shelves = shelves.merge(cur, on=["identifier", "timestamp"])
    shelves["pct"] = shelves["shock_price"] / shelves["current_price"] - 1.0
    shelves = shelves[["identifier", "timestamp", "pct", "net", "jump"]].rename(columns={"net": "net_at"})
    shelves = shelves.dropna(subset=["jump"])
    cur.to_parquet(fa, index=False); grid.to_parquet(fg, index=False); shelves.to_parquet(fs, index=False)
    return cur, grid, shelves


# ----------------------------------------------------------------------------
# Table 2 positions
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_positions() -> pd.DataFrame:
    """Managed-money net position (long minus short contracts), sensitivity 3, all markets."""
    d = pd.read_parquet(C.POSITIONS_FILE)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.sort_values(["identifier", "timestamp"]).reset_index(drop=True)


def universe(sens: int = C.SENS) -> list[str]:
    """Markets with a curve and a usable price file (the five LME metals have no volume data)."""
    anchors, _, _ = curve_tables(sens)
    return sorted(t for t in anchors["identifier"].unique() if has_prices(t) and t not in C.LME)
