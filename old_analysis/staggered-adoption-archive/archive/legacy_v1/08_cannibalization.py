#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Cannibalization Test
Tests whether promoted sales cannibalize organic sales.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyfixest as pf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "08_cannibalization.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("STAGGERED ADOPTION: CANNIBALIZATION TEST", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Test whether promoted (attributed) sales cannibalize organic sales.", f)
        log("  A key concern is that ads simply steal sales that would have", f)
        log("  happened anyway, providing no incremental value.", f)
        log("", f)
        log("MODEL:", f)
        log("  Organic_GMV_it = alpha + delta * Promoted_GMV_it + mu_i + lambda_t + epsilon_it", f)
        log("", f)
        log("INTERPRETATION OF delta:", f)
        log("  delta = 0:  Pure incrementality (1 promoted sale = 1 new sale)", f)
        log("  delta = -1: Full cannibalization (1 promoted sale = 1 lost organic)", f)
        log("  -1 < delta < 0: Partial cannibalization", f)
        log("  delta > 0:  Halo effect (promoted sales boost organic visibility)", f)
        log("", f)
        log("INCREMENTAL ROAS CALCULATION:", f)
        log("  Incremental Sales = Promoted_Sales * (1 + delta)", f)
        log("  iROAS = (Incremental Sales * ASP) / Ad Spend", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_cohorts.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            log("  Run 03_cohort_assignment.py first", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])

        log(f"  Panel shape: {panel.shape}", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log("", f)

        # Note: In the current panel construction, we only have promoted_gmv
        # We need to compute organic_gmv as total - promoted
        # For now, we'll work with what we have and note the limitation

        log("  DATA LIMITATION:", f)
        log("  Current panel has promoted_gmv but no organic_gmv.", f)
        log("  For a proper cannibalization test, we need:", f)
        log("    1. Total GMV (all purchases)", f)
        log("    2. Promoted GMV (attributed to ads)", f)
        log("    3. Organic GMV = Total - Promoted", f)
        log("", f)
        log("  We will estimate the model using available proxies.", f)
        log("", f)

        # Create string identifiers
        panel['vendor_str'] = panel['VENDOR_ID'].astype(str)
        panel['week_str'] = panel['week'].astype(str)

        # -----------------------------------------------------------------
        # Summary Statistics
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SUMMARY STATISTICS", f)
        log("-" * 40, f)

        log(f"  Total observations: {len(panel):,}", f)
        log(f"  Mean promoted_gmv: {panel['promoted_gmv'].mean():.2f}", f)
        log(f"  Std promoted_gmv: {panel['promoted_gmv'].std():.2f}", f)
        log(f"  Mean total_spend: {panel['total_spend'].mean():.4f}", f)
        log(f"  Mean clicks: {panel['clicks'].mean():.2f}", f)
        log(f"  Mean impressions: {panel['impressions'].mean():.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Cannibalization Proxy Test
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("CANNIBALIZATION PROXY: CLICKS vs PROMOTED GMV", f)
        log("-" * 40, f)

        log("  Alternative test: Do clicks (paid traffic) crowd out organic behavior?", f)
        log("  Model: impressions ~ clicks | vendor + week", f)
        log("  If delta < 0, more clicks might reduce organic browsing.", f)
        log("", f)

        try:
            # Test if clicks reduce impressions (proxy for organic engagement)
            model1 = pf.feols("impressions ~ clicks | vendor_str + week_str", data=panel)

            log("  Impressions ~ Clicks:", f)
            log(f"    Coefficient (delta): {model1.coef()['clicks']:.6f}", f)
            log(f"    Std Error: {model1.se()['clicks']:.6f}", f)
            log(f"    t-statistic: {model1.tstat()['clicks']:.4f}", f)
            log(f"    p-value: {model1.pvalue()['clicks']:.6f}", f)
            log(f"    N: {model1.nobs():,}", f)
            log("", f)

            coef = model1.coef()['clicks']
            if coef > 0:
                log("  INTERPRETATION: Positive relationship.", f)
                log("  More clicks associated with more impressions (complementary).", f)
            else:
                log("  INTERPRETATION: Negative relationship.", f)
                log("  More clicks associated with fewer impressions (substitution).", f)

        except Exception as e:
            log(f"    ERROR: {str(e)}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Treatment Effect on Total Engagement
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TREATMENT EFFECT ON ENGAGEMENT METRICS", f)
        log("-" * 40, f)

        log("  Test: Does starting to advertise affect total engagement?", f)
        log("", f)

        engagement_vars = ['impressions', 'clicks', 'auction_participations']

        for var in engagement_vars:
            if var not in panel.columns:
                continue

            try:
                model = pf.feols(f"{var} ~ treated | vendor_str + week_str", data=panel)

                log(f"  {var} ~ treated:", f)
                log(f"    Coefficient: {model.coef()['treated']:.4f}", f)
                log(f"    Std Error: {model.se()['treated']:.4f}", f)
                log(f"    t-statistic: {model.tstat()['treated']:.4f}", f)
                log(f"    p-value: {model.pvalue()['treated']:.6f}", f)
                log("", f)

            except Exception as e:
                log(f"    ERROR for {var}: {str(e)}", f)

        # -----------------------------------------------------------------
        # Spend Efficiency Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SPEND EFFICIENCY ANALYSIS", f)
        log("-" * 40, f)

        log("  Compute basic ROAS metrics.", f)
        log("", f)

        # Filter to periods with positive spend
        spend_panel = panel[panel['total_spend'] > 0].copy()

        if len(spend_panel) > 0:
            # Compute ROAS
            spend_panel['roas'] = spend_panel['promoted_gmv'] / spend_panel['total_spend']
            spend_panel['roas'] = spend_panel['roas'].replace([np.inf, -np.inf], np.nan)

            valid_roas = spend_panel['roas'].dropna()

            log(f"  Observations with positive spend: {len(spend_panel):,}", f)
            log(f"  ROAS (Promoted GMV / Spend):", f)
            log(f"    Mean: {valid_roas.mean():.4f}", f)
            log(f"    Median: {valid_roas.median():.4f}", f)
            log(f"    Std: {valid_roas.std():.4f}", f)
            log(f"    25th percentile: {valid_roas.quantile(0.25):.4f}", f)
            log(f"    75th percentile: {valid_roas.quantile(0.75):.4f}", f)
            log("", f)

            # ROAS by spend quartile
            spend_panel['spend_quartile'] = pd.qcut(
                spend_panel['total_spend'],
                q=4,
                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
            )

            log("  ROAS by Spend Quartile:", f)
            roas_by_quartile = spend_panel.groupby('spend_quartile')['roas'].agg(['mean', 'median', 'count'])
            for idx, row in roas_by_quartile.iterrows():
                log(f"    {idx}: Mean={row['mean']:.4f}, Median={row['median']:.4f}, N={row['count']:,}", f)

            log("", f)

            # Diminishing returns test
            log("  DIMINISHING RETURNS TEST:", f)
            log("  Model: log_promoted_gmv ~ log_spend + log_spend^2 | vendor + week", f)
            log("", f)

            try:
                spend_panel['log_spend_sq'] = spend_panel['log_spend'] ** 2

                model_dr = pf.feols(
                    "log_promoted_gmv ~ log_spend + log_spend_sq | vendor_str + week_str",
                    data=spend_panel
                )

                log(f"    log_spend coefficient: {model_dr.coef()['log_spend']:.6f}", f)
                log(f"    log_spend^2 coefficient: {model_dr.coef()['log_spend_sq']:.6f}", f)
                log("", f)

                if model_dr.coef()['log_spend_sq'] < 0:
                    log("    INTERPRETATION: Negative squared term indicates diminishing returns.", f)
                    log("    Additional spend has decreasing marginal impact on GMV.", f)
                else:
                    log("    INTERPRETATION: No evidence of diminishing returns.", f)

            except Exception as e:
                log(f"    ERROR: {str(e)}", f)

        else:
            log("  No observations with positive spend", f)

        log("", f)

        # -----------------------------------------------------------------
        # iROAS Calculation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INCREMENTAL ROAS (iROAS) ESTIMATE", f)
        log("-" * 40, f)

        log("  Without organic sales data, we cannot compute true iROAS.", f)
        log("  We provide a framework for when data becomes available:", f)
        log("", f)
        log("  If delta (cannibalization rate) were estimated:", f)
        log("    iROAS = ROAS * (1 + delta)", f)
        log("", f)
        log("  Example interpretations:", f)
        log("    delta = 0 (no cannibalization): iROAS = ROAS", f)
        log("    delta = -0.2 (20% cannibalization): iROAS = 0.8 * ROAS", f)
        log("    delta = -0.5 (50% cannibalization): iROAS = 0.5 * ROAS", f)
        log("", f)

        if len(spend_panel) > 0:
            mean_roas = spend_panel['roas'].mean()

            log(f"  Current mean ROAS: {mean_roas:.4f}", f)
            log("", f)
            log("  Implied iROAS under different cannibalization scenarios:", f)
            for delta in [0, -0.1, -0.2, -0.3, -0.5]:
                iroas = mean_roas * (1 + delta)
                log(f"    delta = {delta:.1f}: iROAS = {iroas:.4f}", f)

        log("", f)
        log("=" * 80, f)
        log("CANNIBALIZATION TEST COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
