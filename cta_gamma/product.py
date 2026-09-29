"""The daily index file: one row per market per curve day, and the cross-market snapshot.

This is the "one number per market per day" the proposal asked for, in the form a
client can read without the rest of the project. Contract quantities are given as
shares of trailing average daily volume (ADV, 63 sessions) and of open interest, and in
US dollars of notional for the USD-denominated contracts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

INDEX_COLS = {
    "identifier": "market",
    "sector": "sector",
    "timestamp": "curve_day",
    "anchor": "anchor_price",
    "sigma20": "sigma20",
    "adv": "adv_63",
    "oi": "open_interest",
    "net_position": "mm_net_position",
    "pos_oi": "mm_net_position_share_oi",
    "gamma_pct": "gamma_per_1pct_contracts",
    "gamma_pct_adv": "gamma_per_1pct_share_adv",
    "gamma_sig": "gamma_per_1sigma_contracts",
    "gamma_sig_adv": "gamma_per_1sigma_share_adv",
    "up_2s": "forced_buying_2sigma_contracts",
    "dn_2s": "forced_selling_2sigma_contracts",
    "up_2s_adv": "forced_buying_2sigma_share_adv",
    "dn_2s_adv": "forced_selling_2sigma_share_adv",
    "adverse_2s_adv": "unwind_flow_2sigma_share_adv",
    "adverse_2s_pos": "unwind_flow_2sigma_share_position",
    "asym_2s_adv": "asymmetry_2sigma_share_adv",
    "expected_flow_adv": "expected_flow_share_adv",
    "up_2s_usd": "forced_buying_2sigma_usd",
    "dn_2s_usd": "forced_selling_2sigma_usd",
}


def daily_index(meas: pd.DataFrame) -> pd.DataFrame:
    out = meas[list(INDEX_COLS)].rename(columns=INDEX_COLS).copy()
    out = out.sort_values(["curve_day", "market"]).reset_index(drop=True)
    return out


def write_daily_index(meas: pd.DataFrame) -> pd.DataFrame:
    idx = daily_index(meas)
    idx.to_csv(C.OUTPUT / "cta_gamma_daily.csv", index=False, float_format="%.6g")
    idx.to_parquet(C.OUTPUT / "cta_gamma_daily.parquet", index=False)
    return idx


def snapshot(meas: pd.DataFrame, day: str | None = None) -> pd.DataFrame:
    """Cross-market table for one curve day (default: the last one), sorted by unwind share of ADV."""
    day = pd.Timestamp(day or meas["timestamp"].max())
    d = meas[meas["timestamp"] == day].copy()
    cols = ["identifier", "sector", "anchor", "pos_oi", "gamma_pct", "gamma_pct_adv", "gamma_sig_adv",
            "up_2s_adv", "dn_2s_adv", "adverse_2s_adv", "adverse_2s_pos", "asym_2s_adv",
            "up_2s_usd", "dn_2s_usd"]
    return d[cols].sort_values("adverse_2s_adv", ascending=False).reset_index(drop=True)


def aggregate_usd(meas: pd.DataFrame, sectors=C.COMMODITY_SECTORS) -> pd.DataFrame:
    """Daily total forced buying / selling at 2 sigma across USD commodity contracts, in USD billions.

    This is the commodity counterpart of the "forced selling versus forced buying" statistic
    that sell-side systematic-flow monitors publish for equity index futures.
    """
    d = meas[meas["sector"].isin(sectors) & meas["usd"]]
    g = d.groupby("timestamp").agg(buy_usd=("up_2s_usd", "sum"), sell_usd=("dn_2s_usd", "sum"),
                                   n=("identifier", "size"))
    g["buy_bn"] = g["buy_usd"] / 1e9; g["sell_bn"] = g["sell_usd"] / 1e9
    g["sell_to_buy"] = g["sell_usd"] / g["buy_usd"].replace(0, np.nan)
    return g
