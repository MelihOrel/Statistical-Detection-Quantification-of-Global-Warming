# 🌡️ Statistical Detection & Quantification of Global Warming

**A rigorous, uncertainty-aware trend-analysis pipeline on the Berkeley Earth surface-temperature record — formal hypothesis tests and effect sizes with confidence intervals, not a line drawn by eye.**

This project treats global warming as a *statistical detection problem*. Rather than plotting a series that visibly rises and declaring victory, it quantifies the trend with non-parametric hypothesis tests, propagates the dataset's own stated measurement uncertainty into every conclusion, corrects for the autocorrelation that inflates naive significance, and reports slopes as effect sizes with 95% confidence intervals. The result is meant to read as defensible statistics on a politically charged signal — not advocacy.

---

## 🔎 Data-Honesty Note

The pipeline is built around the real structure and limitations of the Berkeley Earth data, and surfaces them rather than hiding them:

- **Monthly, 1750–2015.** The **land-only** average runs from **1750**; the combined **land-and-ocean** average only begins in **1850**. Every ocean-inclusive analysis is hard-restricted to 1850 onward.
- **Every estimate carries uncertainty.** Each temperature column has a paired `*Uncertainty` column (a 95% confidence half-range). These are propagated into annual means (in quadrature, not naively averaged), onto anomalies, and into the final warming verdict.
- **Pre-1850 data is sparse and far more uncertain.** This is shown explicitly via the uncertainty band, which is visibly wider for early decades.
- **Missing months are handled explicitly.** The land-only series has 12 missing months; short gaps (≤ 3 months) are time-interpolated and longer gaps are logged and left as `NaN`, never silently filled.
- **Sanity anchor.** The combined land+ocean mean rises **0.567 °C** from 1850–1899 to 1965–2014 — squarely in the expected ~0.57 °C ballpark, confirming the preprocessing is faithful.

---

## 🏗️ Architecture

**Anomaly baseline.** Temperatures are converted to anomalies relative to the standard **1951–1980** climatological baseline — the conventional way to present warming, isolating change from the absolute mean.

**Trend testing — why robust methods over naive OLS.** Annual climate series have strongly autocorrelated residuals (Durbin-Watson ≈ 0.54 for land+ocean, lag-1 ACF ≈ 0.71), which violates the independence assumption behind OLS standard errors and *overstates* significance. The pipeline therefore leads with:
- **Mann-Kendall** — a non-parametric, rank-based test for monotonic trend (reports S, Kendall's τ, p-value).
- **Theil-Sen** — a robust slope estimator (median of pairwise slopes) with a confidence interval, insensitive to outliers.
- **OLS** — reported only for comparison, alongside an **effective-sample-size (AR(1)) correction** that widens the slope CI to reflect the real, reduced amount of independent information.

**Uncertainty-aware verdict.** The cumulative warming implied by the Theil-Sen slope is compared against the typical annual measurement uncertainty. The trend is only declared real when the signal dwarfs the dataset's own stated noise — not merely the sampling error.

**STL decomposition.** The monthly series is decomposed (robust STL, period = 12) into trend, annual seasonality, and residual, then the pipeline quantifies how far the trend component rose and whether seasonal amplitude shifted over time.

**Regional breakdown.** Using the by-country and by-major-city companion files, per-region **Theil-Sen warming rates (°C/decade)** are computed with a minimum-coverage threshold (exclusions logged), producing a ranked table and a focused Turkey case study (Istanbul / Ankara / Izmir).

---

## 📁 Project Structure

```
global-warming-detection/
├── data/
│   ├── raw/                  # GlobalTemperatures.csv (+ optional companions)
│   └── processed/            # annual/monthly parquet frames
├── src/
│   ├── data_processor.py     # load, clean, aggregate, anomalies, uncertainty
│   ├── trend_analysis.py     # Mann-Kendall, Theil-Sen, OLS + autocorr correction
│   ├── decomposition.py      # STL seasonal-trend decomposition
│   ├── regional.py           # per-country rates + Turkey case study
│   └── visualize.py          # 300 dpi figures with uncertainty shading
├── reports/
│   ├── figures/              # global_trend, decadal_anomalies, stl_decomposition, ...
│   └── metrics/              # trend_summary.csv, country_warming_rates.csv
├── config.yaml               # paths, baseline, thresholds, analysis params
├── main.py                   # end-to-end orchestration
├── requirements.txt
└── README.md
```

---

## 📊 Results

Trend summary (reproduced from the real Berkeley Earth CSV):

| Series      | Theil-Sen slope (°C/decade) | 95% CI          | Mann-Kendall p | Durbin-Watson |
|-------------|-----------------------------|-----------------|----------------|---------------|
| Land-only   | **0.052**                   | 0.044 – 0.060   | < 1e-10        | 0.85          |
| Land+Ocean  | **0.053**                   | 0.047 – 0.059   | < 1e-10        | 0.54          |

Both series show **significant monotonic warming, robust to autocorrelation and to the dataset's measurement uncertainty.** The OLS slope remains significant even after widening its confidence interval for the AR(1)-reduced effective sample size (n_eff ≈ 28 for land+ocean, down from 166 annual points).

**STL decomposition (land+ocean):** the trend component rose **≈ 1.24 °C** across 1850–2015, while seasonal amplitude was essentially flat (−0.18 °C).

**Fastest-warming countries** (242 of 243 qualified with ≥ 50 years of coverage; full table in `reports/metrics/country_warming_rates.csv`, chart in `reports/figures/country_warming_ranking.png`):

| Rank | Country | Theil-Sen °C/decade | 95% CI | MK p | Years |
|------|---------|---------------------|--------|------|-------|
| 1 | French Southern And Antarctic Lands | 0.151 | 0.119–0.191 | < 1e-10 | 66 |
| 2 | Heard Island And McDonald Islands | 0.138 | 0.098–0.180 | < 1e-9  | 66 |
| 3 | Uzbekistan | 0.107 | 0.088–0.126 | < 1e-12 | 184 |
| 4 | South Georgia And The South Sandwich Islands | 0.106 | 0.086–0.127 | < 1e-12 | 123 |
| 5 | Kiribati | 0.106 | 0.093–0.120 | < 1e-12 | 131 |

> **Honest caveat on the country ranking.** The very top of the list is dominated by small sub-Antarctic territories and Pacific atolls (French Southern and Antarctic Lands, Heard Island, South Georgia, Kiribati, Palmyra Atoll…). These are not noise — every entry has tens to hundreds of years of coverage and p-values below 1e-9 — but in Berkeley Earth these "countries" are tiny grid-cell averages that inherit strongly-warming regional ocean/polar series, so they are not directly comparable to large continental countries. The *large-country* signal is more interpretable: Central-Asian nations (Uzbekistan, Turkmenistan, Kyrgyzstan ~0.10 °C/decade) rank highest among them, consistent with known accelerated continental-interior warming. **Turkey** sits at **0.045 °C/decade** (95% CI 0.033–0.058, MK p < 1e-11, 237 years).

**Turkey case study** — per-city Theil-Sen slope, 95% CI, and Mann-Kendall significance (`reports/figures/turkey_case_study.png`). All three cities show statistically significant warming:

| City | Theil-Sen °C/decade | 95% CI | Mann-Kendall p | Significant |
|------|---------------------|--------|----------------|-------------|
| Istanbul | 0.029 | 0.019–0.040 | 9.1e-08 | ✅ |
| Ankara   | 0.040 | 0.028–0.053 | 4.2e-10 | ✅ |
| İzmir    | 0.030 | 0.020–0.039 | 7.0e-10 | ✅ |

Ankara (continental interior) warms fastest; the two coastal cities (Istanbul, İzmir) are slower and nearly identical — a small but plausible maritime-moderation contrast. Note the city-level rates run below the country-level Turkey rate, reflecting the shorter, single-station city series versus the spatially-averaged national series.

> The repository ships `GlobalTemperatures.csv` plus the `GlobalLandTemperaturesByCountry.csv` and `GlobalLandTemperaturesByMajorCity.csv` companions in `data/raw/`. If the companions are removed, the regional steps log a clear notice and the rest of the pipeline still runs fully.

---

## ⚙️ Installation

```bash
git clone https://github.com/MelihOrel/Statistical-Detection-Quantification-of-Global-Warming.git
cd global-warming-detection
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Place `GlobalTemperatures.csv` in `data/raw/` (and, optionally, the by-country / by-major-city companion files).

---

## ▶️ Usage

```bash
python main.py
```

Expected terminal output (abridged):

```
[1/6] Loading and preprocessing Berkeley Earth data ...
Loaded global file: 3192 months, 1750-01-01 -> 2015-12-01
Land-only: 12 missing months, 10 interpolated (<= 3-month gaps), 2 left as NaN (longer gaps).
Land+Ocean series ready: 1850-2015, 166 annual points.
Sanity anchor (Land+Ocean): 1850-1899=14.970, 1965-2014=15.537, rise=0.567 C (expected ~0.57).
[2/6] Running formal trend tests ...
Mann-Kendall (Land+Ocean): tau=0.681, p=0.00e+00 -> increasing
Theil-Sen (Land+Ocean): 0.0530 C/decade (95% CI 0.0474-0.0587)
OLS (Land+Ocean): 0.0535 C/decade, R2=0.741, DW=0.537, lag-1 acf=0.709
Autocorr correction (Land+Ocean): n_eff=28.3 (from n=166); OLS slope remains significant after widening CI.
Verdict (Land+Ocean): Significant monotonic warming, robust to autocorrelation and to the dataset's measurement uncertainty.
[3/6] STL seasonal-trend decomposition ...
[4/6] Regional breakdown ...
Country warming: 242 countries qualified (>= 50 yrs), 1 excluded for sparse coverage.
Fastest-warming country: French Southern And Antarctic Lands at 0.151 C/decade.
Turkey case study - Ankara: 0.040 C/decade (95% CI 0.028-0.053), MK p=4.22e-10 (significant).
[5/6] Generating figures ...
[6/6] Final statistical summary
======================================================================

    Series  Years  TS C/dec  CI low  CI high   MK p     DW  OLS sig (corr)
 Land-only    266    0.0520  0.0440   0.0601 0.0000 0.8476            True
Land+Ocean    166    0.0530  0.0474   0.0587 0.0000 0.5366            True

Turkey case study (C/decade):
  Istanbul  : 0.029  (CI 0.019-0.040, MK p=9.06e-08, significant)
  Ankara    : 0.040  (CI 0.028-0.053, MK p=4.22e-10, significant)
  Izmir     : 0.030  (CI 0.020-0.039, MK p=6.95e-10, significant)
```

Outputs land in `reports/figures/` (global trend with uncertainty band, decadal anomalies, STL panel, country ranking, Turkey case study) and `reports/metrics/` (`trend_summary.csv`, `country_warming_rates.csv`).

---

## 📚 Data Source

[Berkeley Earth surface-temperature data](http://berkeleyearth.org/data/), distributed via the Kaggle *Climate Change: Earth Surface Temperature Data* dataset. Land-only series 1750–2015; land-and-ocean 1850–2015.
