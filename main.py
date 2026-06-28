"""
main.py
=======
Orchestrates the Global Warming Detection pipeline end to end:

    Load & Preprocess
      -> Compute Anomalies
      -> Trend Tests (Mann-Kendall, Theil-Sen, OLS + autocorr correction)
      -> STL Decomposition
      -> Regional Breakdown (country ranking + Turkey case study)
      -> Visualizations
      -> Final statistical summary table

Run:
    python main.py
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import yaml

# Make src importable whether run from root or elsewhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processor import DataProcessor          # noqa: E402
from trend_analysis import analyze_trend, results_table  # noqa: E402
from decomposition import run_stl                 # noqa: E402
import regional                                    # noqa: E402
import visualize                                    # noqa: E402


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    setup_logging()
    log = logging.getLogger("main")
    cfg = load_config()

    log.info("=" * 70)
    log.info("GLOBAL WARMING DETECTION PIPELINE")
    log.info("=" * 70)

    # -- Step 1: Load & preprocess --------------------------------------
    log.info("[1/6] Loading and preprocessing Berkeley Earth data ...")
    dp = DataProcessor(cfg)
    dp.load()
    bundles = {
        "land": dp.build_series("land"),
        "land_ocean": dp.build_series("land_ocean"),
    }
    dp.save(bundles)
    dp.sanity_anchor(bundles["land_ocean"])

    # -- Step 2: Trend tests --------------------------------------------
    log.info("[2/6] Running formal trend tests ...")
    trend_results = []
    for key, b in bundles.items():
        tr = analyze_trend(b.annual, b.annual_unc, b.name,
                           alpha=cfg["analysis"]["alpha"])
        log.info("Verdict (%s): %s", b.name, tr.verdict)
        trend_results.append(tr)

    tdf = results_table(trend_results)
    os.makedirs(cfg["paths"]["metrics_dir"], exist_ok=True)
    tdf.to_csv(os.path.join(cfg["paths"]["metrics_dir"], "trend_summary.csv"),
               index=False)

    # -- Step 3: STL decomposition --------------------------------------
    log.info("[3/6] STL seasonal-trend decomposition ...")
    decomp = run_stl(bundles["land_ocean"].monthly, bundles["land_ocean"].name,
                     cfg["paths"]["figures_dir"], dpi=cfg["viz"]["dpi"])

    # -- Step 4: Regional breakdown -------------------------------------
    log.info("[4/6] Regional breakdown ...")
    country_df = regional.country_warming_rates(cfg)
    turkey = regional.turkey_case_study(cfg)

    # -- Step 5: Visualizations -----------------------------------------
    log.info("[5/6] Generating figures ...")
    lo_bundle = bundles["land_ocean"]
    lo_trend = next(t for t in trend_results if t.series == lo_bundle.name)
    visualize.plot_global_trend(lo_bundle, lo_trend, cfg["paths"]["figures_dir"],
                                dpi=cfg["viz"]["dpi"])
    visualize.plot_decadal_anomalies(lo_bundle, cfg["paths"]["figures_dir"],
                                     dpi=cfg["viz"]["dpi"])
    visualize.plot_country_ranking(country_df, cfg["paths"]["figures_dir"],
                                   top_n=cfg["regional"]["top_n_ranking"],
                                   dpi=cfg["viz"]["dpi"])
    visualize.plot_turkey_case_study(turkey, cfg["paths"]["figures_dir"],
                                     dpi=cfg["viz"]["dpi"])

    # -- Step 6: Final summary table ------------------------------------
    log.info("[6/6] Final statistical summary")
    log.info("=" * 70)
    summary = tdf[["series", "n_years", "ts_slope_decade", "ts_ci_low",
                   "ts_ci_high", "mk_p", "durbin_watson",
                   "ols_significant_corrected"]].copy()
    summary.columns = ["Series", "Years", "TS C/dec", "CI low", "CI high",
                       "MK p", "DW", "OLS sig (corr)"]
    print("\n" + summary.to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))
    print()
    if not country_df.empty:
        print("Top 5 fastest-warming countries (C/decade):")
        print(country_df.head(5)[["country", "slope_decade", "mk_p"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if turkey:
        print("\nTurkey case study (C/decade):")
        for c in turkey:
            flag = "significant" if c.significant else "n.s."
            print(f"  {c.city:10s}: {c.slope_decade:.3f}  "
                  f"(CI {c.ci_low:.3f}-{c.ci_high:.3f}, MK p={c.mk_p:.2e}, {flag})")
    log.info("=" * 70)
    log.info("Pipeline complete. Figures in %s/, metrics in %s/.",
             cfg["paths"]["figures_dir"], cfg["paths"]["metrics_dir"])


if __name__ == "__main__":
    main()
