"""
regional.py
===========
Regional warming breakdown from the Berkeley Earth companion files.

* Per-country Theil-Sen warming slope (C/decade), restricted to regions
  with sufficient annual coverage (min_years).
* Ranked table of fastest / slowest warming countries.
* Turkey case study: Istanbul, Ankara, Izmir individually (trend + MK p).

These companion CSVs (GlobalLandTemperaturesByCountry.csv,
GlobalLandTemperaturesByMajorCity.csv) are large and are NOT bundled in the
repo. If they are absent, every function degrades gracefully: it logs a
warning and returns empty results so the pipeline still completes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CityTrend:
    city: str
    n_years: int
    slope_decade: float
    ci_low: float
    ci_high: float
    mk_p: float
    significant: bool


def _annual_from_long(df: pd.DataFrame, value_col: str = "AverageTemperature"):
    """Long-format -> per-year mean for a single region's frame."""
    d = df.dropna(subset=[value_col]).copy()
    d["dt"] = pd.to_datetime(d["dt"])
    d["year"] = d["dt"].dt.year
    annual = d.groupby("year")[value_col].mean()
    # Require >= 6 months per year for that year to count.
    counts = d.groupby("year")[value_col].count()
    annual = annual[counts >= 6]
    return annual


def _theilsen_slope(annual: pd.Series, alpha: float = 0.05):
    years = annual.index.values.astype(float)
    vals = annual.values.astype(float)
    slope, _, lo, hi = stats.theilslopes(vals, years, alpha=1 - alpha)
    p = mk.original_test(vals, alpha=alpha).p
    return slope * 10, lo * 10, hi * 10, p


def country_warming_rates(config: dict) -> pd.DataFrame:
    path = config["paths"]["raw_country"]
    min_years = config["regional"]["min_years"]
    if not os.path.exists(path):
        logger.warning("Country file not found (%s); skipping country ranking. "
                       "Download GlobalLandTemperaturesByCountry.csv into data/raw/ "
                       "to enable this step.", path)
        return pd.DataFrame(columns=["country", "n_years", "slope_decade",
                                     "ci_low", "ci_high", "mk_p"])

    df = pd.read_csv(path, parse_dates=["dt"])
    rows = []
    excluded = 0
    for country, grp in df.groupby("Country"):
        annual = _annual_from_long(grp)
        if len(annual) < min_years:
            excluded += 1
            continue
        slope, lo, hi, p = _theilsen_slope(annual)
        rows.append({"country": country, "n_years": len(annual),
                     "slope_decade": slope, "ci_low": lo, "ci_high": hi, "mk_p": p})
    out = pd.DataFrame(rows).sort_values("slope_decade", ascending=False)
    logger.info("Country warming: %d countries qualified (>= %d yrs), "
                "%d excluded for sparse coverage.", len(out), min_years, excluded)
    if len(out):
        top = out.iloc[0]
        logger.info("Fastest-warming country: %s at %.3f C/decade.",
                    top["country"], top["slope_decade"])

    out_dir = config["paths"]["metrics_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "country_warming_rates.csv"), index=False)
    logger.info("Saved %s/country_warming_rates.csv", out_dir)
    return out


def turkey_case_study(config: dict) -> list[CityTrend]:
    path = config["paths"]["raw_city"]
    cities = config["regional"]["turkey_cities"]
    alpha = config["analysis"]["alpha"]
    if not os.path.exists(path):
        logger.warning("Major-city file not found (%s); skipping Turkey case "
                       "study. Download GlobalLandTemperaturesByMajorCity.csv "
                       "into data/raw/ to enable it.", path)
        return []

    df = pd.read_csv(path, parse_dates=["dt"])
    results = []
    for city in cities:
        sub = df[df["City"] == city]
        if sub.empty:
            logger.warning("City %s not present in city file.", city)
            continue
        annual = _annual_from_long(sub)
        if len(annual) < config["regional"]["min_years"]:
            logger.warning("City %s has only %d years; below threshold.",
                           city, len(annual))
            continue
        slope, lo, hi, p = _theilsen_slope(annual, alpha)
        ct = CityTrend(city, len(annual), slope, lo, hi, p, p < alpha)
        results.append(ct)
        logger.info("Turkey case study - %s: %.3f C/decade "
                    "(95%% CI %.3f-%.3f), MK p=%.2e %s.",
                    city, slope, lo, hi, p,
                    "(significant)" if ct.significant else "(n.s.)")
    return results
