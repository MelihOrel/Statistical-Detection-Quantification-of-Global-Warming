"""
data_processor.py
=================
Data engineering for the Berkeley Earth global surface-temperature series.

Responsibilities
----------------
* Load ``GlobalTemperatures.csv`` with a clean monthly DatetimeIndex.
* Expose the land-only series (from 1750) and the land+ocean series
  (restricted to 1850+), each retaining its paired uncertainty column.
* Detect and report missing months; interpolate only short gaps.
* Aggregate to annual means while *propagating* uncertainty rather than
  naively averaging it.
* Compute anomalies relative to a configurable baseline period.

Design note on uncertainty propagation
---------------------------------------
Each monthly uncertainty ``u_i`` is the dataset's stated 95% half-range.
Treating the monthly values as independent measurements of the annual
mean, the standard error of the mean of ``n`` months combines in
quadrature:  ``u_annual = sqrt(sum(u_i^2)) / n``.  This is conservative
(months are not perfectly independent) but is the correct first-order
propagation and is far more honest than averaging the half-ranges.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SeriesBundle:
    """A temperature series paired with its uncertainty, monthly + annual."""
    name: str
    monthly: pd.Series          # monthly mean temperature (DatetimeIndex)
    monthly_unc: pd.Series      # monthly 95% uncertainty half-range
    annual: pd.Series           # annual mean temperature
    annual_unc: pd.Series       # propagated annual uncertainty
    anomaly: pd.Series          # annual anomaly vs baseline
    anomaly_unc: pd.Series      # uncertainty carried onto the anomaly
    baseline_mean: float        # baseline reference value


class DataProcessor:
    def __init__(self, config: dict):
        self.cfg = config
        self.s = config["series"]
        self.bl = config["baseline"]
        self.an = config["analysis"]
        self.raw: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        path = self.cfg["paths"]["raw_global"]
        df = pd.read_csv(path, parse_dates=["dt"])
        df = df.sort_values("dt").set_index("dt")
        # Enforce a regular monthly index (month-start frequency).
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
        df = df.reindex(full_idx)
        df.index.name = "dt"
        self.raw = df
        logger.info(
            "Loaded global file: %d months, %s -> %s",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        return df

    # ------------------------------------------------------------------ #
    # Missing-value handling
    # ------------------------------------------------------------------ #
    def _handle_missing(self, temp: pd.Series, unc: pd.Series, label: str):
        """Interpolate short gaps; log longer ones. Returns cleaned pair."""
        n_missing = int(temp.isna().sum())
        if n_missing == 0:
            logger.info("%s: no missing months.", label)
            return temp, unc

        # Identify run lengths of consecutive NaNs.
        is_na = temp.isna()
        group = (is_na != is_na.shift()).cumsum()
        run_len = is_na.groupby(group).transform("sum").where(is_na, 0)
        long_gap = (run_len > self.an["interpolation_limit"]) & is_na

        # Time-based interpolation for short gaps only.
        temp_i = temp.interpolate(method="time", limit=self.an["interpolation_limit"],
                                  limit_area="inside")
        unc_i = unc.interpolate(method="time", limit=self.an["interpolation_limit"],
                                limit_area="inside")
        n_long = int(long_gap.sum())
        n_interp = n_missing - int(temp_i.isna().sum())
        logger.info(
            "%s: %d missing months, %d interpolated (<= %d-month gaps), "
            "%d left as NaN (longer gaps).",
            label, n_missing, n_interp, self.an["interpolation_limit"],
            int(temp_i.isna().sum()),
        )
        if n_long:
            logger.warning("%s: %d months lie in gaps longer than the "
                           "interpolation limit and were not filled.",
                           label, n_long)
        return temp_i, unc_i

    # ------------------------------------------------------------------ #
    # Annual aggregation with uncertainty propagation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _annual_with_uncertainty(temp: pd.Series, unc: pd.Series):
        """Annual mean + propagated annual uncertainty (quadrature SEM)."""
        df = pd.DataFrame({"t": temp, "u": unc})
        df["year"] = df.index.year

        def _agg(g):
            t = g["t"].dropna()
            u = g["u"].reindex(t.index).dropna()
            n = len(t)
            if n == 0:
                return pd.Series({"mean": np.nan, "unc": np.nan, "n": 0})
            mean = t.mean()
            # Propagate in quadrature; fall back to mean unc if missing.
            if len(u) == n and n > 0:
                prop = np.sqrt(np.sum(u.values ** 2)) / n
            else:
                prop = u.mean() / np.sqrt(max(n, 1))
            return pd.Series({"mean": mean, "unc": prop, "n": n})

        out = df.groupby("year", group_keys=False).apply(_agg)
        # Keep only reasonably complete years (>= 6 months observed).
        out = out[out["n"] >= 6]
        idx = pd.to_datetime(out.index.astype(int).astype(str) + "-01-01")
        annual = pd.Series(out["mean"].values, index=idx, name="annual")
        annual_unc = pd.Series(out["unc"].values, index=idx, name="annual_unc")
        return annual, annual_unc

    # ------------------------------------------------------------------ #
    # Anomalies
    # ------------------------------------------------------------------ #
    def _anomaly(self, annual: pd.Series, annual_unc: pd.Series):
        mask = (annual.index.year >= self.bl["start"]) & \
               (annual.index.year <= self.bl["end"])
        if mask.sum() == 0:
            raise ValueError("Baseline window has no data for this series.")
        baseline_mean = annual[mask].mean()
        anomaly = annual - baseline_mean
        # Uncertainty on the anomaly carries the annual uncertainty plus the
        # uncertainty of the baseline mean itself (quadrature).
        bl_unc = np.sqrt(np.sum(annual_unc[mask].values ** 2)) / mask.sum()
        anomaly_unc = np.sqrt(annual_unc ** 2 + bl_unc ** 2)
        logger.info("Anomaly baseline %d-%d: mean=%.3f C (n=%d years).",
                    self.bl["start"], self.bl["end"], baseline_mean, int(mask.sum()))
        return anomaly, anomaly_unc, baseline_mean

    # ------------------------------------------------------------------ #
    # Public build
    # ------------------------------------------------------------------ #
    def build_series(self, kind: str) -> SeriesBundle:
        """kind in {'land', 'land_ocean'}."""
        if self.raw is None:
            self.load()
        df = self.raw

        if kind == "land":
            tcol, ucol, start, name = (
                self.s["land_col"], self.s["land_unc_col"], 1750, "Land-only",
            )
        elif kind == "land_ocean":
            tcol, ucol, start, name = (
                self.s["land_ocean_col"], self.s["land_ocean_unc_col"],
                self.s["land_ocean_start"], "Land+Ocean",
            )
        else:
            raise ValueError(f"Unknown series kind: {kind}")

        temp = df[tcol].copy()
        unc = df[ucol].copy()
        # Restrict to the valid start year and the analysis end year.
        temp = temp[(temp.index.year >= start) & (temp.index.year <= self.an["end_year"])]
        unc = unc.reindex(temp.index)

        temp, unc = self._handle_missing(temp, unc, name)
        annual, annual_unc = self._annual_with_uncertainty(temp, unc)
        anomaly, anomaly_unc, bl_mean = self._anomaly(annual, annual_unc)

        logger.info("%s series ready: %d-%d, %d annual points.",
                    name, annual.index.year.min(), annual.index.year.max(), len(annual))
        return SeriesBundle(name, temp, unc, annual, annual_unc,
                            anomaly, anomaly_unc, bl_mean)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, bundles: dict[str, SeriesBundle]):
        out_dir = self.cfg["paths"]["processed_dir"]
        os.makedirs(out_dir, exist_ok=True)
        for key, b in bundles.items():
            monthly = pd.DataFrame({"temperature": b.monthly,
                                    "uncertainty": b.monthly_unc})
            annual = pd.DataFrame({"annual_mean": b.annual,
                                   "annual_unc": b.annual_unc,
                                   "anomaly": b.anomaly,
                                   "anomaly_unc": b.anomaly_unc})
            monthly.to_parquet(os.path.join(out_dir, f"{key}_monthly.parquet"))
            annual.to_parquet(os.path.join(out_dir, f"{key}_annual.parquet"))
        logger.info("Saved processed frames to %s/", out_dir)

    # ------------------------------------------------------------------ #
    # Sanity anchor
    # ------------------------------------------------------------------ #
    def sanity_anchor(self, bundle: SeriesBundle) -> float:
        e0, e1 = self.an["anchor_early"]
        l0, l1 = self.an["anchor_late"]
        a = bundle.annual
        early = a[(a.index.year >= e0) & (a.index.year <= e1)].mean()
        late = a[(a.index.year >= l0) & (a.index.year <= l1)].mean()
        rise = late - early
        logger.info("Sanity anchor (%s): %d-%d=%.3f, %d-%d=%.3f, rise=%.3f C "
                    "(expected ~0.57).", bundle.name, e0, e1, early, l0, l1, late, rise)
        return rise
