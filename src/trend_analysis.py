"""
trend_analysis.py
=================
Formal, non-eyeballed trend detection for climate time series.

Methods
-------
* Mann-Kendall  : non-parametric monotonic-trend test (S, tau, p-value).
* Theil-Sen     : robust slope estimator with confidence interval.
* OLS           : parametric comparison (slope, R^2, CI) with explicit
                  caveats about autocorrelated residuals.
* Diagnostics   : Durbin-Watson, lag-1 autocorrelation, and an
                  effective-sample-size correction to the OLS slope CI.
* Uncertainty-aware verdict: is the slope distinguishable from zero once
  the data's own stated measurement uncertainty is folded in?

Why robust methods here
-----------------------
Annual climate series have strongly autocorrelated residuals (DW ~ 0.5).
OLS standard errors assume independence, so naive OLS *overstates*
significance. Mann-Kendall (rank-based) and Theil-Sen (median of pairwise
slopes) do not assume a parametric error model and are robust to outliers,
making them the primary estimators; OLS is reported only for comparison
with an explicit autocorrelation correction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

logger = logging.getLogger(__name__)


@dataclass
class TrendResult:
    series: str
    n_years: int
    # Mann-Kendall
    mk_tau: float
    mk_s: float
    mk_p: float
    mk_trend: str
    # Theil-Sen (per decade)
    ts_slope_decade: float
    ts_ci_low: float
    ts_ci_high: float
    # OLS (per decade)
    ols_slope_decade: float
    ols_r2: float
    ols_ci_low: float
    ols_ci_high: float
    ols_p: float
    # Diagnostics
    durbin_watson: float
    lag1_acf: float
    n_eff: float
    ols_ci_low_corr: float
    ols_ci_high_corr: float
    ols_significant_corrected: bool
    # Uncertainty-aware verdict
    slope_exceeds_measurement_unc: bool
    verdict: str

    def as_row(self) -> dict:
        return asdict(self)


def _lag1_autocorr(resid: np.ndarray) -> float:
    r = resid - resid.mean()
    denom = np.sum(r ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(r[1:] * r[:-1]) / denom)


def analyze_trend(annual: pd.Series, annual_unc: pd.Series,
                  name: str, alpha: float = 0.05) -> TrendResult:
    annual = annual.dropna()
    years = annual.index.year.values.astype(float)
    y = annual.values.astype(float)
    n = len(y)

    # ---- Mann-Kendall ---------------------------------------------------
    mkr = mk.original_test(y, alpha=alpha)

    # ---- Theil-Sen ------------------------------------------------------
    ts_slope, ts_intercept, ts_lo, ts_hi = stats.theilslopes(y, years, alpha=1 - alpha)

    # ---- OLS ------------------------------------------------------------
    X = sm.add_constant(years)
    ols = sm.OLS(y, X).fit()
    slope = ols.params[1]
    ci = ols.conf_int(alpha=alpha)[1]
    dw = durbin_watson(ols.resid)
    r1 = _lag1_autocorr(ols.resid)

    # Effective sample size correction for AR(1) residuals:
    #   n_eff = n * (1 - r1) / (1 + r1)
    r1c = min(max(r1, 0.0), 0.99)
    n_eff = n * (1 - r1c) / (1 + r1c)
    # Inflate the OLS slope SE by sqrt(n / n_eff), widen the CI accordingly.
    se = ols.bse[1]
    se_corr = se * np.sqrt(n / max(n_eff, 1.0))
    tcrit = stats.t.ppf(1 - alpha / 2, df=max(n_eff - 2, 1))
    ci_lo_corr = slope - tcrit * se_corr
    ci_hi_corr = slope + tcrit * se_corr
    sig_corr = not (ci_lo_corr <= 0 <= ci_hi_corr)

    # ---- Uncertainty-aware verdict -------------------------------------
    # Total warming implied by the Theil-Sen slope across the record, vs the
    # typical annual measurement uncertainty. The slope is "real" only if the
    # cumulative signal dwarfs the per-year measurement noise.
    span_decades = (years.max() - years.min()) / 10.0
    total_signal = abs(ts_slope * 10) * span_decades
    typ_unc = float(np.nanmedian(annual_unc.dropna().values)) if annual_unc.notna().any() else 0.0
    exceeds = total_signal > 2 * typ_unc

    if mkr.p < alpha and sig_corr and exceeds:
        verdict = ("Significant monotonic warming, robust to autocorrelation "
                   "and to the dataset's measurement uncertainty.")
    elif mkr.p < alpha and exceeds:
        verdict = ("Significant warming by rank tests and exceeds measurement "
                   "uncertainty, but OLS significance weakens after the "
                   "autocorrelation correction (use Theil-Sen / MK).")
    else:
        verdict = "No trend distinguishable from noise at this confidence level."

    res = TrendResult(
        series=name, n_years=n,
        mk_tau=float(mkr.Tau), mk_s=float(mkr.s), mk_p=float(mkr.p), mk_trend=mkr.trend,
        ts_slope_decade=float(ts_slope * 10),
        ts_ci_low=float(ts_lo * 10), ts_ci_high=float(ts_hi * 10),
        ols_slope_decade=float(slope * 10), ols_r2=float(ols.rsquared),
        ols_ci_low=float(ci[0] * 10), ols_ci_high=float(ci[1] * 10),
        ols_p=float(ols.pvalues[1]),
        durbin_watson=float(dw), lag1_acf=float(r1), n_eff=float(n_eff),
        ols_ci_low_corr=float(ci_lo_corr * 10), ols_ci_high_corr=float(ci_hi_corr * 10),
        ols_significant_corrected=bool(sig_corr),
        slope_exceeds_measurement_unc=bool(exceeds),
        verdict=verdict,
    )

    logger.info("Mann-Kendall (%s): tau=%.3f, p=%.2e -> %s",
                name, res.mk_tau, res.mk_p, res.mk_trend)
    logger.info("Theil-Sen (%s): %.4f C/decade (95%% CI %.4f-%.4f)",
                name, res.ts_slope_decade, res.ts_ci_low, res.ts_ci_high)
    logger.info("OLS (%s): %.4f C/decade, R2=%.3f, DW=%.3f, lag-1 acf=%.3f",
                name, res.ols_slope_decade, res.ols_r2, res.durbin_watson, res.lag1_acf)
    logger.info("Autocorr correction (%s): n_eff=%.1f (from n=%d); "
                "OLS slope %s after widening CI.",
                name, res.n_eff, res.n_years,
                "remains significant" if sig_corr else "no longer significant")
    return res


def results_table(results: list[TrendResult]) -> pd.DataFrame:
    rows = [r.as_row() for r in results]
    return pd.DataFrame(rows)
