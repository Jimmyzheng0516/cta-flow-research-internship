"""Shared helpers for the Strategy 6 ("herd contagion") notebooks.

Strategy 6 reads the CTA herd's *fired* mechanical flow (the Table 3 curve of
trade day D-1 evaluated at session D's realized path, exactly as strategy 5
builds it) and asks a question strategy 5 never asked: what does the flow that
fired in a market's *correlated neighbours* today imply for that market
tomorrow?  Everything here runs offline from the repository caches; the
curve math, price loaders, cost table and book engines are imported from
strategy5_utils so the accounting is identical across the two strategies.

Conventions (inherited):
- a curve stamped trade day D is delivered D+1 07:00 UTC; evaluated at session
  D+1's tape it is known at the settlement of D+1 ("the S5 clock"); a position
  taken at that settlement earns close(D+2)/close(D+1) - 1 (`ret_fwd`).
- returns come from back-adjusted `&TICKER_CCB.csv` files with the house
  hygiene mask; sessions where the adjusted level is far from the raw anchor
  price are dropped (the notebook-08 zero-cross problem).
- books: event accounting (fixed ticket per event, no renormalisation), the
  tick-based cost ladder from Timmy's replication notebook.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import strategy5_utils as s5

BT_CACHE = Path(__file__).resolve().parent / ".cache_s6"
SENS = s5.SENS                      # curve sensitivity (2), as in strategy 4/5
ZWIN = 252
CORR_WIN = 252                      # trailing window for the return-correlation weights
CORR_MIN = 120                      # minimum history before weights exist
START = pd.Timestamp("2011-06-01")  # first session with a full trailing z window available

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# Signal universe: US-listed commodity futures with a Table 3 curve and a price
# file.  ICE Europe (BRN, GAS, LCC, LRC, LSU) and WBS settle on a different
# clock, LME metals have no CCB price series and LBR has almost no history:
# all are excluded so that "today's session" means the same thing across peers.
SIGNAL_UNIVERSE = [
    "CL", "HO", "RB", "NG",                                   # NYMEX energy
    "GC", "SI", "HG", "PL", "PA",                             # COMEX / NYMEX metals
    "ZC", "ZS", "ZW", "KE", "ZM", "ZL", "ZO", "ZR", "MWE", "RS",  # CBOT/KC/MGEX grains, oilseeds (RS = ICE canola, same session)
    "LE", "GF", "HE", "DC",                                   # CME livestock, dairy
    "KC", "SB", "CT", "CC", "OJ",                             # ICE US softs
]
# Same-settlement complexes (the CTA "sleeve" structure), used for the
# asynchronous-close check and the equal-weight variant of the signal.
COMPLEX = {
    "CL": "energy", "HO": "energy", "RB": "energy", "NG": "energy",
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals", "PA": "metals",
    "ZC": "grains", "ZS": "grains", "ZW": "grains", "KE": "grains", "ZM": "grains", "ZL": "grains",
    "ZO": "grains", "ZR": "grains", "MWE": "grains", "RS": "grains",
    "LE": "livestock", "GF": "livestock", "HE": "livestock", "DC": "livestock",
    "KC": "softs", "SB": "softs", "CT": "softs", "CC": "softs", "OJ": "softs",
}
# Executed universe: the paper's drop-thin convention.  Thin names stay in the
# signal (their fires are informative for their neighbours) but are never traded.
THIN = sorted(set(s5.THIN) & set(SIGNAL_UNIVERSE))          # PA, OJ, GF, HE, ZO, ZR, DC, MWE
EXECUTED = [t for t in SIGNAL_UNIVERSE if t not in THIN]     # 20 markets
NO_LIMIT = ["CL", "HO", "RB", "NG", "GC", "SI", "HG", "PL", "PA", "KC", "SB", "CT", "CC"]  # no daily price limit

REQUIRED = dict(s5.REQUIRED)


def preflight() -> None:
    s5.preflight()
    missing = [t for t in SIGNAL_UNIVERSE if not s5.price_path(t).exists()]
    if missing:
        raise FileNotFoundError(f"price files missing for {missing}")
    print(f"strategy 6: {len(SIGNAL_UNIVERSE)} signal markets, {len(EXECUTED)} executed, "
          f"{len(THIN)} thin names ranked but never traded ({', '.join(THIN)})")


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
def _true_returns(ticker: str, anchors: pd.DataFrame) -> pd.DataFrame:
    """Session returns as adjusted price DIFFERENCES over the RAW price level.

    The CCB files are additively back-adjusted: differences are exact, levels are
    not (cocoa's adjusted level is negative for years; soybean oil's sits at a
    quarter of its raw price 2014-2021).  Percent changes of the adjusted level
    - the convention of the earlier notebooks - over-state returns wherever the
    level has drifted.  The raw price is recovered from the Table 3 anchor on
    curve days (offset = adjusted - raw, constant between rolls, forward-filled).
    """
    px = s5.load_prices(ticker).sort_values("timestamp").reset_index(drop=True)
    a = anchors[anchors["identifier"] == ticker].set_index("timestamp")["current_price"]
    px["raw"] = px["close_adj"] - (px["close_adj"] - px["timestamp"].map(a)).ffill().bfill()
    px.loc[px["raw"] <= 0, "raw"] = np.nan
    prev_raw = px["raw"].shift(1)
    px["ret_cc"] = ((px["close_adj"] - px["close_adj"].shift(1)) / prev_raw).where(lambda r: r.abs() < 0.5)
    px["sigma20"] = px["ret_cc"].rolling(s5.VOL_WIN, min_periods=10).std()
    px.loc[~px["sigma20"].between(1e-4, 0.5), "sigma20"] = np.nan
    px["ret_fwd"] = px["ret_cc"].shift(-1)                      # close(e) -> close(e+1), S5 clock
    px["ret_fwd_lag1"] = px["ret_cc"].shift(-2)                 # close(e+1) -> close(e+2), conservative clock
    for h in s5.HORIZONS:
        fw = (px["close_adj"].shift(-h) - px["close_adj"]) / px["raw"]
        px[f"fwd{h}"] = fw.where(fw.abs() < 0.5 * np.sqrt(h))
    px["r_open_close"] = ((px["close_adj"] - px["open_adj"]) / px["raw"].shift(1)).where(lambda r: r.abs() < 0.5)
    px["identifier"] = ticker
    return px[["identifier", "timestamp", "close_adj", "open_adj", "raw", "ret_cc", "sigma20", "ret_fwd",
               "ret_fwd_lag1", "r_open_close"] + [f"fwd{h}" for h in s5.HORIZONS]]


def _base_panel(sens: int = SENS) -> pd.DataFrame:
    """Strategy 5's session panel (fired flow, moves, controls) restricted to
    the signal universe, with returns recomputed on the raw price level."""
    pan = s5.build_analysis_panel(sens, verbose=False)
    pan = pan[pan["identifier"].isin(SIGNAL_UNIVERSE) & (pan["timestamp"] >= START)].copy()
    pan = pan.drop(columns=["sigma20", "ret_fwd", "move_z"] + [f"fwd{h}" for h in s5.HORIZONS])
    anchors, _, _ = s5.load_curve_tables(sens)
    px = pd.concat([_true_returns(t, anchors) for t in SIGNAL_UNIVERSE], ignore_index=True)
    pan = pan.merge(px, on=["identifier", "timestamp"], how="left")
    pan["move_z"] = (pan["r_close"] / pan["sigma20"]).clip(-10, 10)
    pan["distorted"] = pan["raw"].isna() | pan["sigma20"].isna()
    pan["complex"] = pan["identifier"].map(COMPLEX)
    pan["executed"] = pan["identifier"].isin(EXECUTED)
    return pan.sort_values(["identifier", "timestamp"]).reset_index(drop=True)


def _corr_weights(R: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Trailing-window return correlations, refreshed at each month start and
    lagged one session (the window ends the day before it is first used)."""
    out, cur, last_month = {}, None, None
    for d in dates:
        key = (d.year, d.month)
        if cur is None or key != last_month:
            hist = R.loc[:d].iloc[:-1].tail(CORR_WIN)
            if len(hist) >= CORR_MIN:
                c = hist.corr(min_periods=CORR_MIN // 2)
                arr = c.to_numpy(copy=True)
                np.fill_diagonal(arr, 0.0)
                cur = pd.DataFrame(np.clip(arr, 0.0, None), index=c.index, columns=c.columns)
            last_month = key
        out[d] = cur
    return out


def build_signal_panel(sens: int = SENS, verbose: bool = True) -> pd.DataFrame:
    """One row per (market, session): own fired flow, the peers' fired flow
    projected on the market's own risk factor, the placebo built from peers'
    price moves, and the hedger books.  Cached under backtest/.cache_s6/."""
    BT_CACHE.mkdir(exist_ok=True)
    f = BT_CACHE / f"signal_panel_s{sens}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    pan = _base_panel(sens)
    if verbose:
        print(f"  base panel: {len(pan):,} rows, {pan['identifier'].nunique()} markets")

    M = pan.pivot_table(index="timestamp", columns="identifier", values="mech_z")
    V = pan.pivot_table(index="timestamp", columns="identifier", values="move_z")
    R = pan.pivot_table(index="timestamp", columns="identifier", values="r_close")
    W = _corr_weights(R, M.index)

    def project(X: pd.DataFrame) -> pd.DataFrame:
        """Correlation-weighted mean of the OTHER markets' standardised values."""
        out = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
        for d in X.index:
            w = W.get(d)
            if w is None:
                continue
            x = X.loc[d]
            ok = x.notna()
            if ok.sum() < 5:
                continue
            cols = ok.index[ok]
            ww = w.reindex(index=cols, columns=cols).fillna(0.0)
            num = ww.to_numpy() @ x[cols].to_numpy()
            den = ww.sum(axis=1).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                out.loc[d, cols] = np.where(den > 0, num / den, np.nan)
        return out

    def complex_mean(X: pd.DataFrame) -> pd.DataFrame:
        """Leave-one-out equal-weight mean within the same-settlement complex."""
        out = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
        for cx in set(COMPLEX.values()):
            cols = [c for c in X.columns if COMPLEX.get(c) == cx]
            sub = X[cols]
            n = sub.notna().sum(axis=1)
            s = sub.sum(axis=1)
            loo = (s.to_numpy()[:, None] - sub.fillna(0.0).to_numpy()) / (n.to_numpy()[:, None] - 1)
            loo = np.where((n.to_numpy()[:, None] > 1) & sub.notna().to_numpy(), loo, np.nan)
            out[cols] = loo
        return out

    peer = project(M)
    peer_move = project(V)
    cx_peer = complex_mean(M)
    cx_move = complex_mean(V)
    if verbose:
        print("  peer aggregates built")

    def melt(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return df.stack(future_stack=True).rename(name).reset_index().rename(columns={"level_1": "identifier"})

    for df, name in [(peer, "peer_raw"), (peer_move, "peer_move_raw"), (cx_peer, "cx_raw"), (cx_move, "cx_move_raw")]:
        pan = pan.merge(melt(df, name), on=["timestamp", "identifier"], how="left")
    g = pan.groupby("identifier", group_keys=False)
    for raw, z in [("peer_raw", "peer_z"), ("peer_move_raw", "peer_move_z"), ("cx_raw", "cx_z"), ("cx_move_raw", "cx_move_z")]:
        pan[z] = g[raw].apply(lambda s: s5.ztrail(s, ZWIN)).clip(-5, 5)

    # hedger books (producers everywhere; swap dealers where the model has a book)
    frames = []
    for t in SIGNAL_UNIVERSE:
        pr, sw = s5.load_producer_net(t), s5.load_swap_net(t)
        d = pan.loc[pan["identifier"] == t, ["identifier", "timestamp", "oi"]].copy()
        for src, col in [(pr, "prod_net"), (sw, "swap_net")]:
            if src.empty:
                d[col] = np.nan
            else:
                d[col] = src.set_index("timestamp")["net_position"].reindex(d["timestamp"]).to_numpy()
        frames.append(d)
    hb = pd.concat(frames, ignore_index=True)
    hg = hb.groupby("identifier", group_keys=False)
    for col in ["prod_net", "swap_net"]:
        flow = hg[col].diff() / hb["oi"]
        hb[col.replace("net", "flow5")] = flow.groupby(hb["identifier"]).transform(lambda s: s.rolling(5, min_periods=5).sum())
    pan = pan.merge(hb[["identifier", "timestamp", "prod_flow5", "swap_flow5"]], on=["identifier", "timestamp"], how="left")
    g = pan.groupby("identifier", group_keys=False)
    for col in ["prod_flow5", "swap_flow5"]:
        pan[col + "_z"] = g[col].apply(lambda s: s5.ztrail(s, ZWIN)).clip(-5, 5)
    # the hedgers' flow projected the same way as the herd's fired flow
    P5 = pan.pivot_table(index="timestamp", columns="identifier", values="prod_flow5_z")
    pan = pan.merge(melt(project(P5), "peer_prod_raw"), on=["timestamp", "identifier"], how="left")
    S5w = pan.pivot_table(index="timestamp", columns="identifier", values="swap_flow5_z")
    pan = pan.merge(melt(project(S5w), "peer_swap_raw"), on=["timestamp", "identifier"], how="left")
    pan = pan.sort_values(["identifier", "timestamp"]).reset_index(drop=True)
    pan.to_parquet(f, index=False)
    return pan


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def nw_table(df: pd.DataFrame, y: str, xs: list, lags: int = 5, scale: float = 1e4) -> pd.DataFrame:
    """OLS of `y` (in bp) on `xs` with Newey-West standard errors."""
    import statsmodels.api as sm
    d = df.dropna(subset=[y] + xs)
    m = sm.OLS(d[y] * scale, sm.add_constant(d[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = pd.DataFrame({"coef_bp": m.params, "t": m.tvalues})
    out.attrs["n"] = len(d)
    return out


def tstat(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 1 and s.std() > 0 else np.nan


def event_book(pan: pd.DataFrame, score: str, hold: int = 2, thr: float = 1.0, cap: float = 3.0,
               scenario: str = "institutional", executed_only: bool = True, extra_lag: int = 0,
               ret_col: str = "ret_fwd") -> dict:
    """Strategy-5-style event book on the S5 clock, executed names only by default."""
    d = pan[pan["executed"]] if executed_only else pan
    d = d[["timestamp", "identifier", score, "sigma20", ret_col]].rename(columns={ret_col: "ret_fwd"})
    return s5.run_event_book(d, score, thr=thr, hold=hold, cap=cap, scenario=scenario, extra_lag=extra_lag)


def summarize(res: dict, label: str) -> dict:
    return s5.summarize_book(res, label)


def yearly_sharpe(net: pd.Series) -> pd.Series:
    return net.groupby(net.index.year).apply(s5.sharpe_daily)
