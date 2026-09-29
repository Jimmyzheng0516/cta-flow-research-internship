"""Estimation helpers.

All regressions are panel regressions with market fixed effects (every variable is
demeaned within market, so only variation *over time within a market* identifies the
coefficients) and standard errors clustered by date (the same day's residuals are
correlated across markets, and the date cluster corrects for that; it is the
conservative choice for a panel of 74 markets over roughly 3,800 sessions).

Targets are in logs (log absolute return, log range, log realised volatility), so a
coefficient of 0.01 on a z-scored regressor means "one standard deviation more of the
regressor goes with about one per cent more volatility", holding the controls fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C

HAR = ["l_pv_d", "l_pv_w", "l_pv_m"]
PUBLIC = ["l_turnover", "l_vol_rel", "trend_abs", "l_dte"]
CONTROLS = HAR + PUBLIC


def _log(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1e-7))


def _demean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols] - df.groupby("identifier")[cols].transform("mean")


def fe_regression(pan: pd.DataFrame, target: str, regressors: list[str],
                  controls: list[str] = CONTROLS, log_target: bool = True) -> dict:
    """Market-fixed-effects OLS of log(target) on regressors + controls, date-clustered SE.

    Returns dict(table=DataFrame[coef, se, t], n, n_dates, r2_within, r2_controls_only).
    """
    cols = ["identifier", "timestamp", target] + list(dict.fromkeys(regressors + controls))
    d = pan[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d["y"] = _log(d[target]) if log_target else d[target]
    X = _demean(d, list(dict.fromkeys(regressors + controls)))
    y = _demean(d, ["y"])["y"]
    groups = pd.factorize(d["timestamp"])[0]
    full = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    base = sm.OLS(y, X[controls]).fit() if controls else None
    tab = pd.DataFrame({"coef": full.params, "se": full.bse, "t": full.tvalues})
    return {"table": tab, "n": int(full.nobs), "n_dates": int(len(np.unique(groups))),
            "r2_within": float(full.rsquared),
            "r2_controls_only": float(base.rsquared) if base is not None else np.nan}


def coef_summary(pan: pd.DataFrame, targets: list[str], variables: list[str],
                 controls: list[str] = CONTROLS, joint: bool = False) -> pd.DataFrame:
    """One row per (target, variable): coefficient and t, each variable added to the controls alone
    (or all together when joint=True)."""
    rows = []
    for tg in targets:
        if joint:
            r = fe_regression(pan, tg, variables, controls)
            for v in variables:
                rows.append(dict(target=tg, variable=v, coef=r["table"].loc[v, "coef"], t=r["table"].loc[v, "t"],
                                 n=r["n"], r2_within=r["r2_within"], r2_controls=r["r2_controls_only"]))
        else:
            for v in variables:
                r = fe_regression(pan, tg, [v], controls)
                rows.append(dict(target=tg, variable=v, coef=r["table"].loc[v, "coef"], t=r["table"].loc[v, "t"],
                                 n=r["n"], r2_within=r["r2_within"], r2_controls=r["r2_controls_only"]))
    return pd.DataFrame(rows)


def by_group(pan: pd.DataFrame, group_col: str, target: str, variables: list[str],
             controls: list[str] = CONTROLS, min_rows: int = 3000) -> pd.DataFrame:
    rows = []
    for g, d in pan.groupby(group_col):
        if len(d.dropna(subset=[target] + variables)) < min_rows:
            continue
        r = fe_regression(d, target, variables, controls)
        row = {group_col: g, "n": r["n"]}
        for v in variables:
            row[v + "_coef"] = r["table"].loc[v, "coef"]; row[v + "_t"] = r["table"].loc[v, "t"]
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_sort(pan: pd.DataFrame, var: str, target: str, q: int = 5,
                  controls: list[str] = HAR, within_market: bool = True) -> pd.DataFrame:
    """Mean benchmark residual of log(target) by quantile of `var` (quantiles formed within market).

    The residual is from the controls-only fixed-effects regression, so the table reads
    "how much higher or lower than the benchmark forecast volatility turned out to be",
    in log points (x100 = per cent), in each bucket of the sorting variable.
    """
    cols = ["identifier", "timestamp", target, var] + controls
    d = pan[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d["y"] = _log(d[target])
    X = _demean(d, controls); y = _demean(d, ["y"])["y"]
    d["resid"] = sm.OLS(y, X).fit().resid
    if within_market:
        d["bucket"] = d.groupby("identifier")[var].transform(
            lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    else:
        d["bucket"] = pd.qcut(d[var].rank(method="first"), q, labels=False)
    out = d.groupby("bucket").agg(n=("resid", "size"), mean_resid=("resid", "mean"),
                                  mean_var=(var, "mean"), mean_target=(target, "mean"))
    out["mean_resid_pct"] = out["mean_resid"] * 100
    return out


def tail_sort(pan: pd.DataFrame, var: str, flag: str, q: int = 5) -> pd.DataFrame:
    """Frequency (per cent) of a tail flag by within-market quantile of `var`, with a date-clustered t
    for the top-minus-bottom difference."""
    d = pan[["identifier", "timestamp", var, flag]].dropna().copy()
    d["bucket"] = d.groupby("identifier")[var].transform(lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    tab = d.groupby("bucket")[flag].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "freq"})
    tab["freq_pct"] = tab["freq"] * 100
    top, bot = d[d["bucket"] == q - 1], d[d["bucket"] == 0]
    dd = pd.concat([top.assign(top=1.0), bot.assign(top=0.0)])
    m = sm.OLS(dd[flag], sm.add_constant(dd["top"])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["timestamp"])[0]})
    tab.attrs["top_minus_bottom_pct"] = float(m.params["top"] * 100)
    tab.attrs["t"] = float(m.tvalues["top"])
    return tab


# ----------------------------------------------------------------------------
# out-of-sample forecast comparison
# ----------------------------------------------------------------------------
def qlike(var_real: np.ndarray, var_hat: np.ndarray) -> np.ndarray:
    z = var_real / var_hat
    return z - np.log(z) - 1.0


def oos_forecast(pan: pd.DataFrame, target: str, specs: dict[str, list[str]],
                 first_year: int = 2015, last_year: int = 2026, horizon_days: int = 21) -> dict:
    """Expanding-window, year-by-year out-of-sample comparison of forecasting specs.

    For each calendar year Y, fit every spec (market fixed effects, OLS) on all rows whose
    target window is complete before 1 January Y (a `horizon_days` guard keeps the last
    training targets from overlapping the first test dates), forecast every row of Y, and
    score with the squared error of log volatility and with QLIKE on the variance scale
    (the loss a volatility forecaster is judged on, robust to noise in the realised proxy).

    Returns dict(by_year=DataFrame, overall=DataFrame, dm=DataFrame of Diebold-Mariano
    t-statistics of each spec against the first spec, computed on daily cross-market
    average loss differences with Newey-West standard errors).
    """
    allcols = sorted(set(sum(specs.values(), [])))
    d = pan[["identifier", "timestamp", target] + allcols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d["y"] = _log(d[target])
    preds = {k: pd.Series(np.nan, index=d.index) for k in specs}
    for Y in range(first_year, last_year + 1):
        cutoff = pd.Timestamp(f"{Y}-01-01")
        train = d[d["timestamp"] < cutoff - pd.Timedelta(days=int(horizon_days * 1.5))]
        test_idx = d.index[(d["timestamp"] >= cutoff) & (d["timestamp"] < pd.Timestamp(f"{Y + 1}-01-01"))]
        if len(train) < 5000 or len(test_idx) == 0:
            continue
        mkt_mean = train.groupby("identifier")[["y"] + allcols].mean()
        pooled = train[["y"] + allcols].mean()
        test = d.loc[test_idx]
        for k, cols in specs.items():
            Xtr = train[cols] - train.groupby("identifier")[cols].transform("mean")
            ytr = train["y"] - train.groupby("identifier")["y"].transform("mean")
            beta = np.linalg.lstsq(Xtr.to_numpy(), ytr.to_numpy(), rcond=None)[0]
            mm = mkt_mean.reindex(test["identifier"]).fillna(pooled)
            fe = mm["y"].to_numpy() - mm[cols].to_numpy() @ beta
            preds[k].loc[test_idx] = test[cols].to_numpy() @ beta + fe
    out = d[["identifier", "timestamp", "y"]].copy()
    for k in specs:
        out["p_" + k] = preds[k]
    out = out.dropna()
    out["year"] = out["timestamp"].dt.year
    rows, daily = [], {}
    for k in specs:
        e2 = (out["y"] - out["p_" + k]) ** 2
        ql = qlike(np.exp(2 * out["y"]), np.exp(2 * out["p_" + k]))
        out["se_" + k], out["ql_" + k] = e2, ql
    by_year = out.groupby("year").agg(**{f"mse_{k}": (f"se_{k}", "mean") for k in specs},
                                      **{f"qlike_{k}": (f"ql_{k}", "mean") for k in specs}, n=("y", "size"))
    overall = pd.DataFrame({k: dict(mse=out["se_" + k].mean(), qlike=out["ql_" + k].mean()) for k in specs}).T
    base = list(specs)[0]
    dm_rows = []
    for k in list(specs)[1:]:
        for loss in ["se", "ql"]:
            diff = (out[f"{loss}_{base}"] - out[f"{loss}_{k}"]).groupby(out["timestamp"]).mean()
            m = sm.OLS(diff.to_numpy(), np.ones(len(diff))).fit(cov_type="HAC", cov_kwds={"maxlags": horizon_days})
            dm_rows.append(dict(spec=k, loss=loss, mean_gain=float(m.params[0]), t=float(m.tvalues[0]),
                                gain_pct=float(m.params[0] / out[f"{loss}_{base}"].mean() * 100)))
    return {"by_year": by_year, "overall": overall, "dm": pd.DataFrame(dm_rows), "rows": out}
