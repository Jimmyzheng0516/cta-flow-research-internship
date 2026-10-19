# cta-flow-research-internship

Code skeleton from my quantitative research internship at Exponential Technology (XTech), June to September 2026, working with their daily futures-flow forecast tables (estimated positioning and price-reaction curves by investor group for about 80 futures markets).

**Only my own library code is published here.** The data, the executed notebooks, the result tables, the reports and the strategy write-ups belong to the firm and are excluded, and so are the firm's documents. What remains shows how the research was engineered, not what it found.

## Contents

| Path | What it is |
|---|---|
| `cta_gamma/` | Python package for the intern project: a daily, per-market measure of the *mechanical* flow that trend-following funds (CTAs) must trade per unit of price move, read off the slope of the forecast reaction curve at the current price and normalized by trailing volume. `data.py` loaders and return/range conventions, `measures.py` the curve measures, `panel.py` the estimation panel (measures, controls, forward targets, realized flow), `stats.py` date-clustered fixed-effects regressions, sorts and out-of-sample comparison against a HAR volatility benchmark, `overlay.py` volatility-targeted overlay books for the economic-value test, `product.py` the daily index file, `plots.py` figures. |
| `backtest/strategy5_utils.py` | Shared engine for a session-level event strategy on a market's own realized mechanical flow: loaders, curve math, event book, cost model with several cost scenarios. |
| `backtest/strategy6_utils.py` | Same engine applied to the flow of a market's correlated neighbours (contagion signal). |
| `backtest/refresh_data.py` | Rebuilds the local caches from the firm's data API (requires credentials). |
| `common_utils.py` | Shared data access, product mapping and price utilities. |

The code reads credentials from environment variables (`UNIFIER_USER`, `UNIFIER_TOKEN`); a token that had been hard-coded in the original working copy was removed before publishing.

## Commit dates

Files are committed on their last-modified date from the original working folder, so the dates reflect when the work was actually done.
