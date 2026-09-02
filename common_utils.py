"""Shared helpers for the positioning research notebooks."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from unifier import unifier

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
PRICE_DIR = ROOT / "price_data_full" / "continuous_futures"
PRICE_DIR_LEGACY = ROOT / "Continuous Futures"
MAPPING_PATH = ROOT / "product_cot_mapping.csv"

GROUP_MM = "managed_money"
GROUP_PROD = "producer_merchant"
GROUP_SWAP = "swap_dealer"
SENS_MM = 3
SENS_PROD = 1  # producer product delivers a single sensitivity
SENS_SWAP = 1  # swap dealer book, like producers, only fills sensitivity 1


def setup_unifier(user: str | None = None, token: str | None = None) -> None:
    user = user or os.environ.get("UNIFIER_USER", "")
    token = token or os.environ.get(
        "UNIFIER_TOKEN", ""  # set in your environment; never commit
    )
    unifier.user = user
    unifier.token = token
    os.environ["UNIFIER_USER"] = user
    os.environ["UNIFIER_TOKEN"] = token


def load_mapping() -> pd.DataFrame:
    m = pd.read_csv(MAPPING_PATH)
    m["has_cot_mapping"] = m["has_cot_mapping"].astype(bool)
    return m


def price_path(ticker: str) -> Path | None:
    for base in (PRICE_DIR, PRICE_DIR_LEGACY):
        p = base / f"&{ticker}_CCB.csv"
        if p.exists():
            return p
    return None


def load_prices(ticker: str, with_open: bool = False) -> pd.DataFrame:
    path = price_path(ticker)
    if path is None:
        raise FileNotFoundError(f"No price file for {ticker}")
    cols = ["Date", "Close"]
    if with_open:
        cols = ["Date", "Open", "Close"]
    px = pd.read_csv(path, usecols=cols, parse_dates=["Date"])
    px = px.sort_values("Date").reset_index(drop=True)
    rename = {"Date": "timestamp", "Close": "close_adj"}
    if with_open:
        rename["Open"] = "open_adj"
    return px.rename(columns=rename)


def load_positions_ticker(
    ticker: str,
    group: str = GROUP_MM,
    sensitivity: int = SENS_MM,
    cache: bool = True,
) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"pos_{ticker}_{group}_s{sensitivity}.parquet"
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    frame = unifier.get_dataframe(
        name="xtech_futures_flow_forecasts_positions", key=ticker
    )
    if frame.empty:
        return frame

    long_col = f"long_contracts_sensitivity_{sensitivity}"
    short_col = f"short_contracts_sensitivity_{sensitivity}"
    frame = frame[frame["xt_group"] == group].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["asof_datetime"] = pd.to_datetime(frame["asof_datetime"])
    frame["net_position"] = frame[long_col] - frame[short_col]
    frame["gross_position"] = frame[long_col] + frame[short_col]
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if cache:
        frame.to_parquet(cache_file, index=False)
    return frame


def load_positions_panel(
    tickers: list[str],
    group: str = GROUP_MM,
    sensitivity: int = SENS_MM,
) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            d = load_positions_ticker(t, group=group, sensitivity=sensitivity)
        except Exception:
            continue
        if d.empty:
            continue
        d["identifier"] = t
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _salvage_row_groups(text: str) -> list[list[dict]]:
    """Pull every complete row group out of a possibly-truncated JSON array.

    The payload is `[[{col: val}, ...], [...], ...]`. Scanning for balanced
    second-level brackets recovers every finished row and simply drops the partial
    one at the cut. Safe here because no field value contains a bracket.
    """
    cut = text.find('{"error"')
    if cut != -1:
        text = text[:cut]

    groups: list[list[dict]] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
            if depth == 2:
                start = i
        elif ch == "]":
            if depth == 2 and start is not None:
                try:
                    groups.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
            depth -= 1
    return groups


def _positions_rows_raw(ticker: str, group: str, timeout: int = 900) -> tuple[list[dict], bool]:
    """POST the positions table for one group, tolerating a truncated response.

    This table fails in two ways: the server splices a `{"error": ...500...}` object
    into the middle of the array, and the chunked transfer sometimes ends early. Both
    make `unifier.get_dataframe` return an empty frame, which reads as "no data" rather
    than "broken payload". Streaming the body and salvaging complete row groups keeps
    whatever arrived. Returns (rows, complete).
    """
    payload = {
        "name": "xtech_futures_flow_forecasts_positions",
        "user": unifier.user,
        "token": unifier.token,
        "disable_view": False,
        "key": ticker,
        "column_filters": f"xt_group = '{group}'",
    }
    resp = requests.post(
        unifier.url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
        stream=True,
    )
    if resp.status_code != 200:
        return [], False

    chunks: list[bytes] = []
    truncated = False
    try:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                chunks.append(chunk)
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError):
        truncated = True
    text = b"".join(chunks).decode("utf-8", errors="ignore")

    if not truncated:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [{k: v for d in item for k, v in d.items()} for item in data], True
        except json.JSONDecodeError:
            pass
    return [{k: v for d in item for k, v in d.items()} for item in _salvage_row_groups(text)], False


def load_positions_group_robust(
    ticker: str,
    group: str = GROUP_SWAP,
    sensitivity: int = SENS_SWAP,
    cache: bool = True,
    attempts: int = 3,
) -> pd.DataFrame:
    """Positions for one group, tolerant of the truncating server error.

    Each attempt returns a different (random-ordered) slice of the history, so unioning
    a few attempts on `timestamp` recovers effectively the whole series.
    """
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"pos_{ticker}_{group}_s{sensitivity}.parquet"
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    rows: dict[str, dict] = {}
    for _ in range(max(1, attempts)):
        before = len(rows)
        try:
            batch, complete = _positions_rows_raw(ticker, group)
        except requests.exceptions.RequestException:
            continue
        for r in batch:
            ts = r.get("timestamp")
            if ts is not None:
                rows.setdefault(ts, r)
        if complete and batch:
            break
        if rows and len(rows) == before:
            break   # an attempt that adds nothing new means we have all the server will give
    if not rows:
        # Cache the empty result too: a handful of small markets genuinely carry no
        # swap-dealer book, and without this every run re-attempts them for minutes.
        # Delete the parquet to force a refetch if the absence looks like an outage.
        if cache:
            pd.DataFrame(columns=["timestamp", "asof_datetime", "net_position", "gross_position"]).to_parquet(
                cache_file, index=False
            )
        return pd.DataFrame()

    frame = pd.DataFrame(list(rows.values()))
    long_col = f"long_contracts_sensitivity_{sensitivity}"
    short_col = f"short_contracts_sensitivity_{sensitivity}"
    if long_col not in frame or short_col not in frame:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["asof_datetime"] = pd.to_datetime(frame["asof_datetime"])
    frame["net_position"] = frame[long_col] - frame[short_col]
    frame["gross_position"] = frame[long_col] + frame[short_col]
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if cache:
        frame.to_parquet(cache_file, index=False)
    return frame


def load_positions_swap(ticker: str, cache: bool = True, attempts: int = 3) -> pd.DataFrame:
    return load_positions_group_robust(
        ticker, GROUP_SWAP, SENS_SWAP, cache=cache, attempts=attempts
    )


def prefetch_swap(tickers: list[str], max_workers: int = 6) -> dict[str, int]:
    """Warm the swap-dealer cache in parallel; returns rows fetched per ticker."""
    todo = [
        t
        for t in tickers
        if not (CACHE / f"pos_{t}_{GROUP_SWAP}_s{SENS_SWAP}.parquet").exists()
    ]

    def one(t: str) -> tuple[str, int]:
        try:
            return t, len(load_positions_swap(t))
        except Exception:
            return t, 0

    out: dict[str, int] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for t, n in ex.map(one, todo):
                out[t] = n
    return out


def trailing_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mu = series.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = series.rolling(window, min_periods=max(20, window // 4)).std()
    return (series - mu) / sd.replace(0, np.nan)


def add_daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    out = px.copy()
    out = out[out["close_adj"] > 0]
    out["ret_oc"] = out["close_adj"] / out["open_adj"] - 1.0 if "open_adj" in out else np.nan
    out["ret_cc"] = out["close_adj"].pct_change()
    out["ret_co"] = out["close_adj"].shift(-1) / out["close_adj"] - 1.0
    out["sigma_oc_20"] = (
        (out["close_adj"] / out["open_adj"] - 1.0).rolling(20, min_periods=10).std()
        if "open_adj" in out
        else out["ret_cc"].rolling(20, min_periods=10).std()
    )
    return out


def load_cot_disaggregated(back_to: str | None = None) -> pd.DataFrame:
    """Recent cross-section (fast). For history use load_cot_mapped_panel()."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / "cot_disagg_recent.parquet"
    if cache_file.exists() and back_to is None:
        return pd.read_parquet(cache_file)
    kwargs = {"name": "cftc_disaggregated_futures_only", "key": None, "back_to": back_to or "2026-07-01"}
    df = unifier.get_dataframe(**kwargs)
    if not df.empty and back_to is None:
        df.to_parquet(cache_file, index=False)
    return df


def load_cot_tff(back_to: str | None = None) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / "cot_tff_recent.parquet"
    if cache_file.exists() and back_to is None:
        return pd.read_parquet(cache_file)
    kwargs = {"name": "cftc_tff_futures_only", "key": None, "back_to": back_to or "2026-07-01"}
    df = unifier.get_dataframe(**kwargs)
    if not df.empty and back_to is None:
        df.to_parquet(cache_file, index=False)
    return df


def _cot_ident_map(mapping: pd.DataFrame | None = None) -> dict[str, str]:
    """Map XTech identifier -> COT ident string (from recent cross-section)."""
    mapping = mapping if mapping is not None else load_mapping()
    recent = load_cot_disaggregated()
    tff = load_cot_tff()
    idmap: dict[str, str] = {}
    for _, r in mapping[mapping.has_cot_mapping].iterrows():
        if r.cot_source == "DISAG":
            sub = recent[recent["cftc_contract_market_code"].astype(str) == str(r.lookup_value)]
            if not sub.empty:
                idmap[r.identifier] = sub.iloc[0]["ident"]
        elif r.cot_source == "TFF":
            sub = tff[tff["contract_market_name"] == r.lookup_value]
            if not sub.empty:
                idmap[r.identifier] = sub.iloc[0]["ident"]
    return idmap


def load_cot_product_history(
    ticker: str,
    mapping: pd.DataFrame | None = None,
    idmap: dict[str, str] | None = None,
    back_to: str = "2014-01-01",
) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"cot_hist_{ticker}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    mapping = mapping if mapping is not None else load_mapping()
    idmap = idmap if idmap is not None else _cot_ident_map(mapping)
    ident = idmap.get(ticker)
    if not ident:
        return pd.DataFrame()

    row = mapping.loc[mapping.identifier == ticker].iloc[0]
    source = "cftc_disaggregated_futures_only" if row.cot_source == "DISAG" else "cftc_tff_futures_only"
    df = unifier.get_dataframe(name=source, key=ident, back_to=back_to)
    if df.empty:
        return df
    df["identifier"] = ticker
    df["sector"] = row.sector
    df["multiplier"] = row.multiplier
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df.to_parquet(cache_file, index=False)
    return df


def load_cot_mapped_panel(
    tickers: list[str] | None = None,
    mapping: pd.DataFrame | None = None,
    back_to: str = "2014-01-01",
) -> pd.DataFrame:
    mapping = mapping if mapping is not None else load_mapping()
    if tickers is None:
        tickers = sorted(mapping.loc[mapping.has_cot_mapping, "identifier"].unique())
    idmap = _cot_ident_map(mapping)
    frames = []
    for t in tickers:
        d = load_cot_product_history(t, mapping, idmap, back_to=back_to)
        if not d.empty:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def join_cot_to_mapping(cot: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach XTech identifier to COT rows using product_cot_mapping lookup rules."""
    rows = []
    m = mapping[mapping["has_cot_mapping"]].copy()
    for _, r in m.iterrows():
        if r["cot_source"] == "DISAG":
            sub = cot[cot["cftc_contract_market_code"].astype(str) == str(r["lookup_value"])]
        elif r["cot_source"] == "TFF":
            sub = cot[cot["contract_market_name"] == r["lookup_value"]]
        elif r["cot_source"] == "ICE":
            sub = cot[cot["contract_market_name"] == r["lookup_value"]]
        else:
            continue
        if sub.empty:
            continue
        sub = sub.copy()
        sub["identifier"] = r["identifier"]
        sub["sector"] = r["sector"]
        sub["multiplier"] = r["multiplier"]
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["report_date"] = pd.to_datetime(out["report_date_as_yyyy_mm_dd"])
    return out.sort_values(["identifier", "timestamp"])


def anchored_producer_accuracy(
    ticker: str,
    cot: pd.DataFrame | None = None,
    mapping: pd.DataFrame | None = None,
) -> dict | None:
    """Weekly directional accuracy of anchored producer forecast vs COT change."""
    mapping = mapping if mapping is not None else load_mapping()
    if cot is None:
        cot = load_cot_product_history(ticker, mapping)
    cot_t = cot[cot["identifier"] == ticker].sort_values("report_date").drop_duplicates("report_date")
    if len(cot_t) < 30:
        return None
    cot_t = cot_t.copy()
    cot_t["cot_net"] = cot_t.apply(producer_net_cot, axis=1)

    model = load_positions_ticker(ticker, GROUP_PROD, SENS_PROD).dropna(subset=["net_position"])
    if model.empty:
        return None
    model = model.sort_values("timestamp")

    hits = n = 0
    for i in range(1, len(cot_t)):
        prev, cur = cot_t.iloc[i - 1], cot_t.iloc[i]
        asof = pd.Timestamp(cur["timestamp"])
        m = model[model["timestamp"] <= asof]
        if m.empty:
            continue
        pred = m.iloc[-1]["net_position"] - prev["cot_net"]
        actual = cur["cot_net"] - prev["cot_net"]
        if pred == 0 or actual == 0:
            continue
        n += 1
        hits += int(np.sign(pred) == np.sign(actual))
    if n < 20:
        return None
    return {
        "ticker": ticker,
        "sector": mapping.loc[mapping.identifier == ticker, "sector"].iloc[0],
        "n_weeks": n,
        "accuracy": hits / n,
    }


def producer_net_cot(row: pd.Series) -> float:
    return row["prod_merc_positions_long"] - row["prod_merc_positions_short"]


def mm_net_cot(row: pd.Series) -> float:
    return row["m_money_positions_long_all"] - row["m_money_positions_short_all"]


def swap_net_cot(row: pd.Series) -> float:
    # note the double underscore in the CFTC short column name
    return row["swap_positions_long_all"] - row["swap__positions_short_all"]


def weekly_cot_changes(cot_mapped: pd.DataFrame, net_fn) -> pd.DataFrame:
    """One row per (identifier, report week) with net level and weekly change."""
    df = cot_mapped.copy()
    df["net"] = df.apply(net_fn, axis=1)
    df["oi"] = df["open_interest_all"].replace(0, np.nan)
    df = df.sort_values(["identifier", "timestamp"])
    g = df.groupby("identifier", group_keys=False)
    df["net_chg"] = g["net"].diff()
    df["net_pct_oi"] = df["net"] / df["oi"]
    df["net_chg_pct_oi"] = df["net_chg"] / df["oi"]
    return df


def sharpe_daily(rets: pd.Series, ann: float = 252.0) -> float:
    r = rets.dropna()
    if r.empty or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ann))


def sortino_daily(rets: pd.Series, ann: float = 252.0) -> float:
    r = rets.dropna()
    if r.empty:
        return np.nan
    downside = r[r < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    return float(r.mean() / downside.std() * np.sqrt(ann))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def producer_tickers(mapping: pd.DataFrame | None = None) -> list[str]:
    mapping = mapping if mapping is not None else load_mapping()
    # commodities with producer group in XTech (exclude pure financial PROXY rows)
    return sorted(
        mapping.loc[
            mapping["has_cot_mapping"] & mapping["sector"].isin(["Agriculture", "Energy", "Metals"]),
            "identifier",
        ].unique()
    )
