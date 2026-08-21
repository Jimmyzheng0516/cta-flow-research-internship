"""Rebuild the upstream caches this folder depends on. Requires unifier API access.

The notebooks themselves never call the API - they read the parquet caches in the
repository's `.cache/`. This script exists so the caches can be refreshed or
extended past 2026-06-02. Run it from the repository root (the parent of
backtest/), with the `unifier` package importable:

    python backtest/refresh_data.py            # refresh everything missing
    python backtest/refresh_data.py --force    # re-pull even if a cache exists

What it (re)builds:
  .cache/strategy4_scenarios_full_managed_money.parquet   Table 3, full trigger
      detail, managed money, one API call per market (slow: ~1-2 min each).
  .cache/strategy2_positions_managed_money_s3.parquet     bulk MM net positions.
  .cache/pos_{T}_producer_merchant_s1.parquet             per-market producer books
      (via common_utils loaders, same files the root notebooks use).

Extension stub: Table 3 also carries `producer_merchant` and `swap_dealer`
curves. Pulling them (set GROUPS below) is the prerequisite for the
forward-looking absorption idea in strategy5_backtest.ipynb §7 - the producer
book delivers sensitivity 1 only, so read those curves at sensitivity 1.
Prices come from the Norgate `&{T}_CCB.csv` bundle and are refreshed separately.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GROUPS = ["managed_money"]          # add "producer_merchant", "swap_dealer" to extend
SCEN_TYPES = ["current", "reference", "trigger"]
SENS_COLS = ([f"buy_contracts_sensitivity_{s}" for s in (1, 2, 3, 4)]
             + [f"sell_contracts_sensitivity_{s}" for s in (1, 2, 3, 4)])
KEEP = ["identifier", "timestamp", "scenario_type", "scenario_label", "shock_price"] + SENS_COLS


def main(force: bool = False) -> None:
    import common_utils as cu           # noqa: root module; needs unifier
    from unifier import unifier
    cu.setup_unifier()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)

    ident = unifier.get_dataframe(name="xtech_futures_flow_forecasts_ident", key=None)
    tickers = [t for t in ident["identifier"] if cu.price_path(t)]

    # ---- Table 3, full curve ------------------------------------------------
    for group in GROUPS:
        out = cache / f"strategy4_scenarios_full_{group}.parquet"
        if out.exists() and not force:
            print(f"skip {out.name} (exists)")
            continue
        frames = []
        for i, t in enumerate(tickers, 1):
            try:
                scen = unifier.get_dataframe(
                    name="xtech_futures_flow_forecasts_scenarios", key=t)
            except Exception as e:      # noqa: BLE001 - report and continue
                print(f"  skip {t}: {e}")
                continue
            if scen.empty:
                continue
            scen = scen[(scen["xt_group"] == group)
                        & (scen["scenario_type"].isin(SCEN_TYPES))].copy()
            if scen.empty:
                continue
            scen["timestamp"] = pd.to_datetime(scen["timestamp"])
            scen["identifier"] = t
            frames.append(scen[[c for c in KEEP if c in scen.columns]])
            print(f"  {group} {i}/{len(tickers)}: {t} ({len(scen):,} rows)")
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        print(f"wrote {out}")

    # ---- bulk managed-money positions, sensitivity 3 ------------------------
    out = cache / "strategy2_positions_managed_money_s3.parquet"
    if force or not out.exists():
        frame = unifier.get_dataframe(name="xtech_futures_flow_forecasts_positions", key=None)
        frame = frame[frame["xt_group"] == "managed_money"].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["net_position"] = (frame["long_contracts_sensitivity_3"]
                                 - frame["short_contracts_sensitivity_3"])
        frame[["identifier", "timestamp", "net_position"]].to_parquet(out, index=False)
        print(f"wrote {out}")

    # ---- per-market producer books ------------------------------------------
    for t in tickers:
        try:
            cu.load_positions_ticker(t, cu.GROUP_PROD, cu.SENS_PROD)   # caches itself
        except Exception:
            pass
    print("producer caches warmed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)
