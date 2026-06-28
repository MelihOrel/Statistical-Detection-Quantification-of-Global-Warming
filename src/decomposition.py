"""
decomposition.py
================
STL (Seasonal-Trend decomposition using Loess) of the monthly series.

Separates the monthly temperature signal into trend, an annual (period=12)
seasonal component, and a residual. Then quantifies:

* how much the STL trend component rose across the record, and
* whether the seasonal amplitude (peak-to-trough of the seasonal component)
  has changed between the first and last thirds of the record.

The decomposition figure is written to reports/figures/stl_decomposition.png.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    series: str
    trend_rise_total: float       # C, end-minus-start of STL trend
    seasonal_amp_early: float     # C, mean peak-to-trough, first third
    seasonal_amp_late: float      # C, mean peak-to-trough, last third
    seasonal_amp_change: float    # C, late minus early
    resid_std: float


def run_stl(monthly: pd.Series, name: str, figures_dir: str,
            dpi: int = 300) -> DecompositionResult:
    s = monthly.dropna().asfreq("MS")
    # STL needs a strictly regular series with no internal gaps.
    s = s.interpolate(method="time", limit_area="inside")
    s = s.dropna()

    stl = STL(s, period=12, robust=True)
    res = stl.fit()

    trend = res.trend.dropna()
    trend_rise = float(trend.iloc[-1] - trend.iloc[0])

    # Seasonal amplitude over time: peak-to-trough within each calendar year.
    seas = res.seasonal
    seas_df = pd.DataFrame({"s": seas})
    seas_df["year"] = seas_df.index.year
    amp = seas_df.groupby("year")["s"].agg(lambda g: g.max() - g.min())
    third = max(len(amp) // 3, 1)
    amp_early = float(amp.iloc[:third].mean())
    amp_late = float(amp.iloc[-third:].mean())

    result = DecompositionResult(
        series=name,
        trend_rise_total=trend_rise,
        seasonal_amp_early=amp_early,
        seasonal_amp_late=amp_late,
        seasonal_amp_change=amp_late - amp_early,
        resid_std=float(res.resid.std()),
    )

    # ---- Figure ---------------------------------------------------------
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(s.index, s.values, color="#222", lw=0.6)
    axes[0].set_ylabel("Observed (C)")
    axes[0].set_title(f"STL decomposition - {name} monthly series")
    axes[1].plot(res.trend.index, res.trend.values, color="#c0392b", lw=1.4)
    axes[1].set_ylabel("Trend (C)")
    axes[2].plot(res.seasonal.index, res.seasonal.values, color="#2980b9", lw=0.4)
    axes[2].set_ylabel("Seasonal (C)")
    axes[3].plot(res.resid.index, res.resid.values, color="#7f8c8d", lw=0.3)
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("Residual (C)")
    axes[3].set_xlabel("Year")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figures_dir, "stl_decomposition.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)

    logger.info("STL (%s): trend component rose %.3f C across the record.",
                name, trend_rise)
    logger.info("STL (%s): seasonal amplitude %.3f C (early) -> %.3f C (late), "
                "change %.3f C.", name, amp_early, amp_late,
                result.seasonal_amp_change)
    logger.info("Saved %s", out)
    return result
