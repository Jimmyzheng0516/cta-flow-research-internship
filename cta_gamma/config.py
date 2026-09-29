"""Paths and fixed settings for the CTA gamma project.

Everything runs offline from files already in the Exp.Tech folder. Nothing here
calls the unifier API.

Folder layout assumed (relative to the Exp.Tech root, i.e. the parent of
`jimmy_intern/`):

    original_job_prices_2026-09-27/norgate/&<TICKER>.csv       raw continuous prices
    original_job_prices_2026-09-27/norgate/&<TICKER>_CCB.csv   back-adjusted prices
    .cache/strategy4_scenarios_full_managed_money.parquet      Table 3, all sensitivities
    .cache/strategy2_positions_managed_money_s3.parquet         Table 2 net position, s3
    product_cot_mapping.csv                                     sector and contract multiplier

All intermediate files this project writes go under `jimmy_intern/cache/` and all
results under `jimmy_intern/output/`, so nothing outside `jimmy_intern/` is touched.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../jimmy_intern/cta_gamma
PROJECT = HERE.parent                            # .../jimmy_intern
ROOT = PROJECT.parent                            # .../Exp.Tech

PRICE_DIR = ROOT / "original_job_prices_2026-09-27" / "norgate"
SCENARIO_FILE = ROOT / ".cache" / "strategy4_scenarios_full_managed_money.parquet"
POSITIONS_FILE = ROOT / ".cache" / "strategy2_positions_managed_money_s3.parquet"
MAPPING_FILE = ROOT / "product_cot_mapping.csv"

CACHE = PROJECT / "cache"
OUTPUT = PROJECT / "output"
TABLES = OUTPUT / "tables"
FIGS = OUTPUT / "figs"
for _p in (CACHE, OUTPUT, TABLES, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

# --- fixed settings ---------------------------------------------------------
SENS = 3            # managed-money sensitivity used for the headline (white paper default)
ROBUST_SENS = 2     # the sensitivity Strategy 5 used; re-run for robustness
ADV_WINDOW = 63     # trailing sessions for average daily volume ("the trailing quarter")
ADV_MIN = 40        # minimum sessions before an ADV value is used
VOL_WINDOW = 20     # trailing sessions for the daily volatility sigma20 (house convention)
Z_WINDOW = 252      # trailing window for within-market z-scores
Z_MIN = 126         # minimum observations before a z-score is used
SHOCK_SIGMAS = 2.0  # "forced flow at 2 sigma": the adverse move used for the flow-at-risk measures

# The scenario cache stops on this date; prices run to 2026-09-25, so the last
# curve days still have complete forward windows for every target used here.
LAST_CURVE_DAY = "2026-06-02"

# Markets with a managed-money curve but no CCB price file (LME, priced from LSEG).
# They are kept where only the raw close is needed and dropped from return-based tests.
LME = {"AHD", "NID", "PBD", "SND", "ZSD"}

COMMODITY_SECTORS = ("Agriculture", "Energy", "Metals")
