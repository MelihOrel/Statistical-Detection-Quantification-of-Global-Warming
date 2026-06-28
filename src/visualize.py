"""
visualize.py
============
All publication-quality (300 dpi) figures, with honest uncertainty shading
wherever the data carries uncertainty.

Figures
-------
* global_trend.png          : anomaly series + uncertainty band + Theil-Sen line
* decadal_anomalies.png     : warming-stripes-style bar chart of decadal anomalies
* country_warming_ranking.png : ranked horizontal bar of fastest-warming countries
* turkey_case_study.png     : three-city panel (Istanbul / Ankara / Izmir)
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

logger = logging.getLogger(__name__)


def _ensure(figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)


def plot_global_trend(bundle, trend_result, figures_dir: str, dpi: int = 300):
    _ensure(figures_dir)
    years = bundle.anomaly.index.year.values
    anom = bundle.anomaly.values
    unc = bundle.anomaly_unc.values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(years, anom - unc, anom + unc, color="#b0c4de", alpha=0.5,
                    label="95% measurement uncertainty")
    ax.plot(years, anom, color="#34495e", lw=1.1, label=f"{bundle.name} anomaly")

    # Theil-Sen fitted line through the anomaly series.
    slope_dec = trend_result.ts_slope_decade
    slope_yr = slope_dec / 10.0
    yr0 = years.min()
    fit = slope_yr * (years - yr0)
    fit = fit - np.mean(fit) + np.mean(anom)
    ax.plot(years, fit, color="#c0392b", lw=2.2,
            label=f"Theil-Sen {slope_dec:.3f} C/decade")

    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature anomaly (C)")
    ax.set_title(f"{bundle.name} anomaly vs baseline, with uncertainty band "
                 f"and robust trend")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figures_dir, "global_trend.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_decadal_anomalies(bundle, figures_dir: str, dpi: int = 300):
    _ensure(figures_dir)
    anom = bundle.anomaly.copy()
    df = pd.DataFrame({"anom": anom.values}, index=anom.index.year)
    df["decade"] = (df.index // 10) * 10
    dec = df.groupby("decade")["anom"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    norm = TwoSlopeNorm(vmin=dec.min(), vcenter=0, vmax=dec.max())
    colors = plt.cm.RdBu_r(norm(dec.values))
    ax.bar(dec.index.astype(str), dec.values, color=colors, width=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Mean anomaly (C)")
    ax.set_title(f"Decadal mean temperature anomalies - {bundle.name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(figures_dir, "decadal_anomalies.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_country_ranking(country_df: pd.DataFrame, figures_dir: str,
                         top_n: int = 15, dpi: int = 300):
    _ensure(figures_dir)
    if country_df is None or country_df.empty:
        logger.warning("No country data; skipping country ranking figure.")
        return
    top = country_df.sort_values("slope_decade", ascending=False).head(top_n)
    top = top.iloc[::-1]  # so largest is at top of horizontal bar

    fig, ax = plt.subplots(figsize=(10, 8))
    err_low = top["slope_decade"] - top["ci_low"]
    err_high = top["ci_high"] - top["slope_decade"]
    ax.barh(top["country"], top["slope_decade"],
            xerr=[err_low, err_high], color="#e74c3c", alpha=0.85,
            error_kw=dict(ecolor="#555", lw=1, capsize=3))
    ax.set_xlabel("Theil-Sen warming rate (C/decade)")
    ax.set_title(f"Top {top_n} fastest-warming countries (95% CI)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = os.path.join(figures_dir, "country_warming_ranking.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_turkey_case_study(city_trends, figures_dir: str, dpi: int = 300):
    _ensure(figures_dir)
    if not city_trends:
        logger.warning("No Turkey city trends; skipping case-study figure.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [c.city for c in city_trends]
    slopes = [c.slope_decade for c in city_trends]
    err_low = [c.slope_decade - c.ci_low for c in city_trends]
    err_high = [c.ci_high - c.slope_decade for c in city_trends]
    colors = ["#27ae60" if c.significant else "#95a5a6" for c in city_trends]
    bars = ax.bar(names, slopes, yerr=[err_low, err_high], color=colors,
                  capsize=5, alpha=0.9)
    for c, b in zip(city_trends, bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"p={c.mk_p:.1e}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Theil-Sen warming rate (C/decade)")
    ax.set_title("Turkey case study: city-level warming (green = significant)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(figures_dir, "turkey_case_study.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", out)
