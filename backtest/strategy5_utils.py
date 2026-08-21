"""Shared helpers for the Strategy 5 (mechanical-share) notebooks.

Everything here runs offline from the repository caches - no unifier calls.
The notebooks in this folder import this module; `refresh_data.py` documents
how the caches were produced in the first place.

Conventions inherited from the root notebooks (02/05/07/08):
- prices come from the back-adjusted `&TICKER_CCB.csv` files; the adjustment is
  additive, so price *differences* are exact while old *levels* can be distorted
  (CL crosses zero in April 2020). Percent returns therefore get the usual
  hygiene mask, and everything that must line up with raw price levels (the
  scenario ladder) is computed from adjusted differences scaled by the raw
  anchor price, which the additive adjustment preserves.
- a model row stamped trade day D is delivered D+1 07:00 UTC and is tradeable
  from the close of D+1 at the earliest.
- portfolio construction: cross-sectionally demeaned, inverse-vol weighted,
  unit gross, overlapping tranches; costs charged on realised turnover with the
  tick-based table from notebook 05/08.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths (the notebooks live in Exp.Tech/backtest/, data in the parent folder)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
PRICE_DIR = ROOT / "price_data_full" / "continuous_futures"
BT_CACHE = Path(__file__).resolve().parent / ".cache_s5"

SCEN_CACHE = CACHE / "strategy4_scenarios_full_managed_money.parquet"
DROP_LME = {"AHD", "NID", "PBD", "SND", "ZSD"}  # no CCB price series

SENS = 2            # scenario sensitivity (strategy4 convention; 1/3 for robustness)
ZWIN = 252          # trailing z-score window
VOL_WIN = 20        # sigma for weights, house convention
EPS = 1e-9

REQUIRED = {
    "scenario curves (Table 3, managed money, full history)": SCEN_CACHE,
    "managed-money positions s3 (bulk cache)": CACHE / "strategy2_positions_managed_money_s3.parquet",
    "price files": PRICE_DIR,
}


def preflight(extra: dict | None = None) -> None:
    """Fail fast, with the offending path, if a required input is missing."""
    req = dict(REQUIRED)
    if extra:
        req.update(extra)
    missing = [f"  {k}: {p}" for k, p in req.items() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input(s) - run refresh_data.py or copy the .cache "
            "folder from the research repo:\n" + "\n".join(missing)
        )
    n_px = len(list(Path(PRICE_DIR).glob("&*_CCB.csv")))
    print(f"preflight OK - scenario cache present, {n_px} CCB price files")


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def price_path(ticker: str) -> Path:
    return PRICE_DIR / f"&{ticker}_CCB.csv"


@lru_cache(maxsize=None)
def load_prices(ticker: str) -> pd.DataFrame:
    """Daily OHLC + open interest from the back-adjusted continuous file."""
    px = pd.read_csv(
        price_path(ticker),
        usecols=["Date", "Open", "High", "Low", "Close", "Open Interest"],
        parse_dates=["Date"],
    ).rename(columns={
        "Date": "timestamp", "Open": "open_adj", "High": "high_adj",
        "Low": "low_adj", "Close": "close_adj", "Open Interest": "oi",
    })
    px = px.sort_values("timestamp").reset_index(drop=True)
    px["oi"] = px["oi"].replace(0, np.nan).ffill()
    return px


def clean_returns(close: pd.Series) -> pd.Series:
    """Percent returns with the house hygiene mask for distorted stretches."""
    r = close.pct_change()
    r = r.where(close > 0)
    return r.where(r.abs() < 0.5)


# --------------------------------------------------------------------------
# Positions (model books, from the repository caches)
# --------------------------------------------------------------------------
def load_mm_positions() -> pd.DataFrame:
    """Managed-money net position, sensitivity 3, full universe (bulk cache)."""
    d = pd.read_parquet(CACHE / "strategy2_positions_managed_money_s3.parquet")
    long_c, short_c = "long_contracts_sensitivity_3", "short_contracts_sensitivity_3"
    if "net_position" not in d.columns:
        d["net_position"] = d[long_c] - d[short_c]
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d[["identifier", "timestamp", "net_position"]].sort_values(
        ["identifier", "timestamp"]).reset_index(drop=True)


def load_group_ticker(ticker: str, group: str, sens: int) -> pd.DataFrame:
    """Per-ticker cached positions (same parquets common_utils writes)."""
    f = CACHE / f"pos_{ticker}_{group}_s{sens}.parquet"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_parquet(f)
    if d.empty or "net_position" not in d.columns:
        return pd.DataFrame()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d[["timestamp", "net_position"]].sort_values("timestamp")


def load_producer_net(ticker: str) -> pd.DataFrame:
    return load_group_ticker(ticker, "producer_merchant", 1)


def load_swap_net(ticker: str) -> pd.DataFrame:
    return load_group_ticker(ticker, "swap_dealer", 1)


# --------------------------------------------------------------------------
# Scenario curves -> compact per-day objects
# --------------------------------------------------------------------------
def load_scenarios(columns_sens: int = SENS) -> pd.DataFrame:
    """The raw Table 3 cache, one sensitivity's buy/sell columns only."""
    cols = ["identifier", "timestamp", "scenario_type", "scenario_label", "shock_price",
            f"buy_contracts_sensitivity_{columns_sens}",
            f"sell_contracts_sensitivity_{columns_sens}"]
    raw = pd.read_parquet(SCEN_CACHE, columns=cols)
    raw = raw.rename(columns={
        f"buy_contracts_sensitivity_{columns_sens}": "buy",
        f"sell_contracts_sensitivity_{columns_sens}": "sell",
    })
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["net"] = raw["buy"] - raw["sell"]
    return raw


def build_curve_tables(raw: pd.DataFrame, cache_tag: str = f"s{SENS}"):
    """Collapse the raw ladder into three flat tables, cached to parquet.

    anchors:  (identifier, timestamp, current_price)
    grid:     (identifier, timestamp, pct, net)      - reference rows
    shelves:  (identifier, timestamp, pct, net_at, jump) - trigger rows, one per
              fired level, jump = net_at - net_pre (the discrete step)
    All `pct` are fractional distances from the day's raw anchor price.
    """
    BT_CACHE.mkdir(exist_ok=True)
    fa, fg, fs = (BT_CACHE / f"anchors_{cache_tag}.parquet",
                  BT_CACHE / f"grid_{cache_tag}.parquet",
                  BT_CACHE / f"shelves_{cache_tag}.parquet")
    if fa.exists() and fg.exists() and fs.exists():
        return pd.read_parquet(fa), pd.read_parquet(fg), pd.read_parquet(fs)

    cur = (raw[raw["scenario_type"] == "current"]
           .drop_duplicates(["identifier", "timestamp"])
           [["identifier", "timestamp", "shock_price"]]
           .rename(columns={"shock_price": "current_price"}))

    grid = raw[raw["scenario_type"] == "reference"].merge(cur, on=["identifier", "timestamp"])
    grid["pct"] = grid["shock_price"] / grid["current_price"] - 1.0
    grid = grid[["identifier", "timestamp", "pct", "net"]]

    trig = raw[raw["scenario_type"] == "trigger"].copy()
    is_pre = trig["scenario_label"].str.endswith("_pre")
    fire = trig[~is_pre].copy()
    pre = trig[is_pre].copy()
    pre["base"] = pre["scenario_label"].str.replace("_pre", "", regex=False)
    fire["base"] = fire["scenario_label"]
    shelves = fire.merge(
        pre[["identifier", "timestamp", "base", "net"]],
        on=["identifier", "timestamp", "base"], suffixes=("", "_pre"), how="left",
    )
    shelves["jump"] = shelves["net"] - shelves["net_pre"]
    shelves = shelves.merge(cur, on=["identifier", "timestamp"])
    shelves["pct"] = shelves["shock_price"] / shelves["current_price"] - 1.0
    shelves = shelves[["identifier", "timestamp", "pct", "net", "jump"]].rename(
        columns={"net": "net_at"})
    shelves = shelves.dropna(subset=["jump"])

    cur.to_parquet(fa, index=False)
    grid.to_parquet(fg, index=False)
    shelves.to_parquet(fs, index=False)
    return cur, grid, shelves


# --------------------------------------------------------------------------
# The mechanical-flow evaluation
# --------------------------------------------------------------------------
def mechanical_flow_market(ticker, anchors, grid, shelves, px) -> pd.DataFrame:
    """Evaluate the day-(e-1) curve at session e's realised path, one market.

    Returns one row per event session e with:
      r_close   - close-to-close move of session e, as a fraction of the anchor
                  (adjusted price differences over the raw anchor: roll-safe)
      mech      - curve-implied net managed-money contracts for the session:
                  full-curve interpolation at r_close, plus the jumps of
                  shelves touched intraday beyond the close (stop runs fire on
                  touch and do not un-fire)
      mech_jump - the shelf-only part of `mech`
      n_fired   - how many shelves the session touched
    """
    a = anchors[anchors["identifier"] == ticker].set_index("timestamp")["current_price"].sort_index()
    if a.empty:
        return pd.DataFrame()
    g = grid[grid["identifier"] == ticker]
    s = shelves[shelves["identifier"] == ticker]
    g_by_day = {ts: (sub["pct"].to_numpy(), sub["net"].to_numpy())
                for ts, sub in g.groupby("timestamp")}
    s_by_day = {ts: (sub["pct"].to_numpy(), sub["net_at"].to_numpy(), sub["jump"].to_numpy())
                for ts, sub in s.groupby("timestamp")}

    px = px.set_index("timestamp").sort_index()
    days = px.index
    out = []
    # position of each curve day within the price calendar
    prev_close = px["close_adj"].shift(1)
    for i in range(1, len(days)):
        e = days[i]           # event session
        d = days[i - 1]       # curve day (delivered morning of e)
        if d not in g_by_day:
            continue
        anchor = a.get(d, np.nan)
        c0 = prev_close.iloc[i]
        hi, lo, cl = px["high_adj"].iloc[i], px["low_adj"].iloc[i], px["close_adj"].iloc[i]
        if not np.isfinite([anchor, c0, hi, lo, cl]).all() or anchor <= 0:
            continue
        # adjusted differences over the raw anchor: exact under additive CCB
        r_close = (cl - c0) / anchor
        r_high = (hi - c0) / anchor
        r_low = (lo - c0) / anchor
        if abs(r_close) > 0.5 or r_high < r_low:   # distorted stretch guard
            continue

        gp, gn = g_by_day[d]
        sp, sn, sj = s_by_day.get(d, (np.array([]), np.array([]), np.array([])))
        # merged cumulative curve: grid + shelf 'at' points + the origin
        xs = np.concatenate([gp, sp, [0.0]])
        ys = np.concatenate([gn, sn, [0.0]])
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        mech_close = float(np.interp(r_close, xs, ys))
        # shelves touched intraday but outside the closed segment
        lo_seg, hi_seg = min(0.0, r_close), max(0.0, r_close)
        touched = (sp >= r_low) & (sp <= r_high)
        outside = touched & ((sp < lo_seg) | (sp > hi_seg))
        extra = float(sj[outside].sum())
        out.append({
            "timestamp": e, "curve_day": d, "r_close": r_close,
            "r_high": r_high, "r_low": r_low,
            "mech": mech_close + extra, "mech_close": mech_close,
            "mech_jump": float(sj[touched].sum()), "extra_jump": extra,
            "n_fired": int(touched.sum()),
        })
    res = pd.DataFrame(out)
    if not res.empty:
        res.insert(0, "identifier", ticker)
    return res


def ztrail(s: pd.Series, w: int = ZWIN) -> pd.Series:
    mu = s.rolling(w, min_periods=w // 2).mean()
    sd = s.rolling(w, min_periods=w // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def sector_map() -> dict:
    m = pd.read_csv(ROOT / "product_cot_mapping.csv")
    return dict(zip(m["identifier"], m["sector"]))


def load_curve_tables(sens: int = SENS):
    """Anchors/grid/shelves for one sensitivity, built on first use and cached."""
    tag = f"s{sens}"
    fa = BT_CACHE / f"anchors_{tag}.parquet"
    if fa.exists():
        return (pd.read_parquet(fa),
                pd.read_parquet(BT_CACHE / f"grid_{tag}.parquet"),
                pd.read_parquet(BT_CACHE / f"shelves_{tag}.parquet"))
    return build_curve_tables(load_scenarios(sens), tag)


def build_mech_panel(sens: int = SENS, tickers: list | None = None,
                     tag: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Session-level mechanical flow for every market, cached to parquet."""
    BT_CACHE.mkdir(exist_ok=True)
    tag = tag or (f"s{sens}" if tickers is None else f"s{sens}_{len(tickers)}n")
    f = BT_CACHE / f"mech_panel_{tag}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    anchors, grid, shelves = load_curve_tables(sens)
    names = tickers or sorted(anchors["identifier"].unique())
    frames = []
    for i, t in enumerate(names, 1):
        if not price_path(t).exists():
            continue
        m = mechanical_flow_market(t, anchors, grid, shelves, load_prices(t))
        if not m.empty:
            frames.append(m)
        if verbose and (i % 20 == 0 or i == len(names)):
            print(f"  mechanical flow {i}/{len(names)} markets...")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(f, index=False)
    return out


HORIZONS = [1, 3, 5, 10, 20]


def build_analysis_panel(sens: int = SENS, verbose: bool = True) -> pd.DataFrame:
    """The estimation panel: mechanical flow + prices, flows, and controls.

    One row per (market, event session). Cached; delete the parquet under
    backtest/.cache_s5/ to force a rebuild.
    """
    BT_CACHE.mkdir(exist_ok=True)
    f = BT_CACHE / f"panel_s{sens}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    mech = build_mech_panel(sens, verbose=verbose)
    mm = load_mm_positions()
    sector_of = sector_map()
    frames = []
    for t, m in mech.groupby("identifier"):
        px = load_prices(t).sort_values("timestamp")
        px["ret_cc"] = clean_returns(px["close_adj"])
        px["sigma20"] = px["ret_cc"].rolling(VOL_WIN, min_periods=10).std()
        px.loc[~px["sigma20"].between(1e-4, 0.5), "sigma20"] = np.nan   # 08 hygiene
        adj = px["close_adj"]
        for h in HORIZONS:
            fw = (adj.shift(-h) / adj - 1.0).where(adj > 0)
            px[f"fwd{h}"] = fw.where(fw.abs() < 0.5 * np.sqrt(h))
        px["ret_fwd"] = px["fwd1"]
        px["trend_z"] = ztrail(adj / adj.rolling(120).mean() - 1.0)
        d = m.merge(px[["timestamp", "sigma20", "oi", "ret_fwd", "trend_z"]
                       + [f"fwd{h}" for h in HORIZONS]], on="timestamp", how="left")
        g = mm[mm["identifier"] == t].set_index("timestamp")["net_position"].sort_index()
        d["mm_flow"] = (g.reindex(d["timestamp"]).values
                        - g.reindex(d["curve_day"]).values)
        d["mm_flow_next"] = (g.shift(-1) - g).reindex(d["timestamp"]).values
        pr = load_producer_net(t)
        if not pr.empty:
            p = pr.set_index("timestamp")["net_position"].sort_index()
            d["prod_flow5_lag"] = (p - p.shift(5)).shift(1).reindex(d["timestamp"]).values
        else:
            d["prod_flow5_lag"] = np.nan
        d["sector"] = sector_of.get(t, "?")
        frames.append(d)
    pan = pd.concat(frames, ignore_index=True).sort_values(["identifier", "timestamp"])
    g = pan.groupby("identifier", group_keys=False)
    pan["mech_oi"] = pan["mech"] / pan["oi"]
    pan["flow_oi"] = pan["mm_flow"] / pan["oi"]
    pan["mech_z"] = g["mech_oi"].apply(ztrail)
    pan["flow_z"] = g["flow_oi"].apply(ztrail)
    pan["move_z"] = pan["r_close"] / pan["sigma20"]
    pan.to_parquet(f, index=False)
    return pan


# --------------------------------------------------------------------------
# Performance metrics (identical to common_utils, restated to stay offline)
# --------------------------------------------------------------------------
def sharpe_daily(rets: pd.Series, ann: float = 252.0) -> float:
    r = rets.dropna()
    if r.empty or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ann))


def sortino_daily(rets: pd.Series, ann: float = 252.0) -> float:
    r = rets.dropna()
    if r.empty:
        return np.nan
    dn = r[r < 0]
    if dn.empty or dn.std() == 0:
        return np.nan
    return float(r.mean() / dn.std() * np.sqrt(ann))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


# --------------------------------------------------------------------------
# Cost model - the tick-based table from notebooks 05/08, extended with a
# flat per-side rate for the financial futures (Timmy's replication notebook
# uses the same convention for schedule-less names)
# --------------------------------------------------------------------------
TICK = {
    "CL": 0.01, "NG": 0.001, "RB": 0.0001, "HO": 0.0001, "BRN": 0.01, "GAS": 0.25,
    "WBS": 0.01, "GC": 0.1, "SI": 0.005, "HG": 0.0005, "PL": 0.1, "PA": 0.5,
    "ZC": 0.25, "ZS": 0.25, "ZW": 0.25, "KE": 0.25, "MWE": 0.25, "ZO": 0.25,
    "ZM": 0.1, "ZL": 0.01, "ZR": 0.005, "RS": 0.1, "KC": 0.05, "SB": 0.01,
    "CT": 0.01, "CC": 1.0, "OJ": 0.05, "LE": 0.025, "HE": 0.025, "GF": 0.025,
    "DC": 0.01, "LBR": 0.5, "LCC": 1.0, "LRC": 1.0, "LSU": 0.1,
}
THIN = {"PA", "OJ", "GF", "HE", "ZO", "ZR", "DC", "LBR", "MWE", "LSU"}   # x4 slippage
MEDIUM = {"PL", "CT", "RS", "LE", "CC", "KE", "ZM", "LCC", "LRC"}        # x2 slippage
FLAT_FIN_BPS = 0.5   # per-side, for financials with no tick schedule here

# Cost scenarios follow the unified table in Timmy's replication notebook:
# ticks per side, liquidity-tier multipliers, and a per-contract fee.
# `institutional` is the headline basis there (settlement/TAS-style desk);
# `base` is the replication standard (the 05/08 half-tick rule).
COST_SCENARIOS = {
    "optimistic":    dict(ticks=0.25, tiers=(1.0, 1.0, 1.0), fee_usd=0.0),
    "institutional": dict(ticks=0.30, tiers=(1.0, 1.5, 2.0), fee_usd=2.5),
    "base":          dict(ticks=0.50, tiers=(1.0, 2.0, 4.0), fee_usd=0.0),
    "conservative":  dict(ticks=1.00, tiers=(1.0, 2.0, 4.0), fee_usd=0.0),
}


@lru_cache(maxsize=None)
def _multiplier(ticker: str) -> float:
    m = pd.read_csv(ROOT / "product_cot_mapping.csv")
    row = m.loc[m["identifier"] == ticker]
    if row.empty or "multiplier" not in m.columns:
        return np.nan
    return float(row["multiplier"].iloc[0])


@lru_cache(maxsize=None)
def cost_bps(ticker: str, scenario: str = "base") -> float:
    """One-way execution cost in bps of notional (close/settlement execution)."""
    p = COST_SCENARIOS[scenario]
    fee_flat = 0.5 if p["fee_usd"] == 0 else 0.0     # legacy flat fee for tickless rule
    if ticker not in TICK:
        return FLAT_FIN_BPS + fee_flat
    px = load_prices(ticker)["close_adj"].dropna()
    if px.empty or px.iloc[-1] <= 0:
        return FLAT_FIN_BPS + fee_flat
    price = float(px.iloc[-1])
    tier = p["tiers"][2] if ticker in THIN else p["tiers"][1] if ticker in MEDIUM else p["tiers"][0]
    slip = p["ticks"] * tier * TICK[ticker] / price * 1e4
    mult = _multiplier(ticker)
    fee = (p["fee_usd"] / (mult * price) * 1e4) if (p["fee_usd"] and np.isfinite(mult) and mult > 0) else fee_flat
    return slip + fee


# --------------------------------------------------------------------------
# The book engine (identical mechanics to notebook 08's run_book)
# --------------------------------------------------------------------------
def run_book(sp: pd.DataFrame, score_col: str, hold: int = 10, min_names: int = 10,
             scenario: str = "base", extra_lag: int = 0) -> dict:
    """Daily cross-sectional book from a long panel.

    sp needs: timestamp, identifier, <score_col>, sigma20, ret_fwd
    (ret_fwd = close(e+1)/close(e) - 1 for a position taken at the close of
    event day e). `extra_lag` shifts execution by n further sessions.
    """
    d = sp.dropna(subset=[score_col, "sigma20"]).copy()
    sig = d.pivot_table(index="timestamp", columns="identifier", values=score_col)
    vol = d.pivot_table(index="timestamp", columns="identifier", values="sigma20")
    ret = sp.pivot_table(index="timestamp", columns="identifier", values="ret_fwd")
    sig = sig.rolling(hold, min_periods=1).mean()          # overlapping tranches
    if extra_lag:
        sig = sig.shift(extra_lag)
    ret = ret.reindex(index=sig.index, columns=sig.columns)
    vol = vol.reindex(index=sig.index, columns=sig.columns)

    keep = sig.notna().sum(axis=1) >= min_names
    sig, vol, ret = sig[keep], vol[keep], ret[keep]
    s = sig.sub(sig.mean(axis=1), axis=0)                  # market-neutral
    w = s / vol.replace(0, np.nan)                         # inverse-vol
    w = w.div(w.abs().sum(axis=1), axis=0).fillna(0.0)     # unit gross
    contrib = w * ret.fillna(0.0)
    gross = contrib.sum(axis=1)
    costs = pd.Series({t: cost_bps(t, scenario) for t in w.columns})
    cost = ((w - w.shift().fillna(0.0)).abs() * costs).sum(axis=1) / 1e4
    return {"gross": gross, "net": gross - cost, "cost": cost,
            "contrib": contrib, "w": w}


def run_event_book(sp: pd.DataFrame, score_col: str, thr: float = 1.0,
                   hold: int = 3, cap: float = 3.0, scenario: str = "base",
                   extra_lag: int = 0) -> dict:
    """Event-style accounting: enter on the event, hold `hold` sessions, exit.

    Position size per event is the score's excess over the threshold (capped),
    divided by the market's sigma20 - a fixed ticket per event, never
    renormalised afterwards, so turnover is exactly two executions per event.
    The book return is scaled so that average gross exposure is one.
    """
    d = sp.dropna(subset=[score_col, "sigma20"]).copy()
    excess = (d[score_col].abs() - thr).clip(lower=0).clip(upper=cap)
    d["ticket"] = np.sign(d[score_col]) * excess / d["sigma20"]
    tick = d.pivot_table(index="timestamp", columns="identifier", values="ticket")
    ret = sp.pivot_table(index="timestamp", columns="identifier", values="ret_fwd")
    ret = ret.reindex(index=tick.index, columns=tick.columns)
    # a ticket opened on event day e is live for the next `hold` sessions
    W = tick.fillna(0.0).rolling(hold, min_periods=1).sum()
    if extra_lag:
        W = W.shift(extra_lag)
    scale = W.abs().sum(axis=1).mean()
    if not np.isfinite(scale) or scale == 0:
        return {}
    W = W / scale
    contrib = W * ret.fillna(0.0)
    gross = contrib.sum(axis=1)
    costs = pd.Series({t: cost_bps(t, scenario) for t in W.columns})
    cost = ((W - W.shift().fillna(0.0)).abs() * costs).sum(axis=1) / 1e4
    return {"gross": gross, "net": gross - cost, "cost": cost,
            "contrib": contrib, "w": W}


def summarize_book(res: dict, label: str) -> dict:
    net, gross = res["net"], res["gross"]
    return {
        "book": label, "days": len(net),
        "ann_ret_net": net.mean() * 252, "ann_vol": net.std() * np.sqrt(252),
        "sharpe_gross": sharpe_daily(gross), "sharpe_net": sharpe_daily(net),
        "sortino_net": sortino_daily(net),
        "max_dd": max_drawdown((1 + net).cumprod()),
        "cost_ann": res["cost"].mean() * 252,
        "turnover": float((res["w"] - res["w"].shift().fillna(0.0)).abs().sum(axis=1).mean()),
    }
