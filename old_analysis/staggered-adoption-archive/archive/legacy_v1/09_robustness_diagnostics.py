#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Robustness Diagnostics
Performs placebo tests, heterogeneity analysis, and sensitivity checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyfixest as pf
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "09_robustness_diagnostics.txt"

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
        log("STAGGERED ADOPTION: ROBUSTNESS DIAGNOSTICS", f)
        log("=" * 80, f)
        log("", f)

        log("ROBUSTNESS CHECKS:", f)
        log("  1. Placebo test: Fake treatment timing for control group", f)
        log("  2. Heterogeneity by vendor size", f)
        log("  3. Heterogeneity by spend intensity", f)
        log("  4. Balance test: Pre-treatment characteristics", f)
        log("  5. Attrition analysis", f)
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
        panel['cohort_week'] = pd.to_datetime(panel['cohort_week'])

        log(f"  Panel shape: {panel.shape}", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log("", f)

        panel['vendor_str'] = panel['VENDOR_ID'].astype(str)
        panel['week_str'] = panel['week'].astype(str)

        # -----------------------------------------------------------------
        # 1. Placebo Test
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("1. PLACEBO TEST", f)
        log("-" * 40, f)

        log("  Randomly assign fake treatment timing to never-treated units.", f)
        log("  If we find significant effects, our model may be capturing", f)
        log("  spurious correlations rather than causal effects.", f)
        log("", f)

        np.random.seed(42)

        # Get never-treated vendors
        never_treated_vendors = panel[panel['cohort_id'] == 0]['VENDOR_ID'].unique()
        log(f"  Never-treated vendors: {len(never_treated_vendors):,}", f)

        if len(never_treated_vendors) > 10:
            # Assign random fake cohorts from the real cohort distribution
            real_cohorts = panel[panel['cohort_week'].notna()]['cohort_week'].unique()

            if len(real_cohorts) > 0:
                # Create placebo panel
                placebo_panel = panel.copy()

                # For never-treated, assign random cohort
                for vendor in tqdm(never_treated_vendors, desc="Assigning placebo"):
                    fake_cohort = np.random.choice(real_cohorts)
                    mask = placebo_panel['VENDOR_ID'] == vendor
                    placebo_panel.loc[mask, 'cohort_week'] = fake_cohort
                    placebo_panel.loc[mask, 'treated'] = (
                        placebo_panel.loc[mask, 'week'] >= fake_cohort
                    ).astype(int)

                # Run TWFE on placebo
                log("", f)
                log("  PLACEBO TWFE RESULTS:", f)

                try:
                    # Only use placebo-assigned vendors
                    placebo_subset = placebo_panel[placebo_panel['VENDOR_ID'].isin(never_treated_vendors)]
                    placebo_subset['vendor_str'] = placebo_subset['VENDOR_ID'].astype(str)
                    placebo_subset['week_str'] = placebo_subset['week'].astype(str)

                    placebo_model = pf.feols(
                        "log_promoted_gmv ~ treated | vendor_str + week_str",
                        data=placebo_subset
                    )

                    log(f"    Placebo coefficient: {placebo_model.coef()['treated']:.6f}", f)
                    log(f"    Std Error: {placebo_model.se()['treated']:.6f}", f)
                    log(f"    t-statistic: {placebo_model.tstat()['treated']:.4f}", f)
                    log(f"    p-value: {placebo_model.pvalue()['treated']:.6f}", f)
                    log("", f)

                    if placebo_model.pvalue()['treated'] > 0.05:
                        log("    PASS: Placebo effect is not significant (p > 0.05)", f)
                    else:
                        log("    WARNING: Significant placebo effect detected", f)
                        log("    This may indicate model misspecification", f)

                except Exception as e:
                    log(f"    ERROR: {str(e)}", f)

            else:
                log("  No real cohorts available for placebo assignment", f)
        else:
            log("  Insufficient never-treated vendors for placebo test", f)

        log("", f)

        # -----------------------------------------------------------------
        # 2. Heterogeneity by Vendor Size
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("2. HETEROGENEITY BY VENDOR SIZE", f)
        log("-" * 40, f)

        log("  Split vendors by baseline activity level.", f)
        log("  Test if treatment effects differ by vendor size.", f)
        log("", f)

        # Compute baseline characteristics (pre-treatment average)
        pre_treatment = panel[panel['treated'] == 0].copy()

        if len(pre_treatment) > 0:
            vendor_baseline = pre_treatment.groupby('VENDOR_ID').agg(
                avg_impressions=('impressions', 'mean'),
                avg_clicks=('clicks', 'mean'),
                avg_participations=('auction_participations', 'mean'),
            ).reset_index()

            # Classify vendors by size (median split)
            median_imp = vendor_baseline['avg_impressions'].median()
            vendor_baseline['size_group'] = np.where(
                vendor_baseline['avg_impressions'] >= median_imp,
                'Large',
                'Small'
            )

            # Merge back
            panel_with_size = panel.merge(
                vendor_baseline[['VENDOR_ID', 'size_group']],
                on='VENDOR_ID',
                how='left'
            )
            panel_with_size['size_group'] = panel_with_size['size_group'].fillna('Unknown')

            log(f"  Large vendors (impressions >= {median_imp:.1f}): {(vendor_baseline['size_group'] == 'Large').sum():,}", f)
            log(f"  Small vendors (impressions < {median_imp:.1f}): {(vendor_baseline['size_group'] == 'Small').sum():,}", f)
            log("", f)

            # Run separate regressions
            for size in ['Large', 'Small']:
                subset = panel_with_size[panel_with_size['size_group'] == size]
                subset['vendor_str'] = subset['VENDOR_ID'].astype(str)
                subset['week_str'] = subset['week'].astype(str)

                if len(subset) > 100:
                    try:
                        model = pf.feols(
                            "log_promoted_gmv ~ treated | vendor_str + week_str",
                            data=subset
                        )

                        log(f"  {size} Vendors:", f)
                        log(f"    Coefficient: {model.coef()['treated']:.6f}", f)
                        log(f"    Std Error: {model.se()['treated']:.6f}", f)
                        log(f"    t-statistic: {model.tstat()['treated']:.4f}", f)
                        log(f"    p-value: {model.pvalue()['treated']:.6f}", f)
                        log(f"    N: {model.nobs():,}", f)
                        log("", f)

                    except Exception as e:
                        log(f"  {size} Vendors: ERROR - {str(e)}", f)
                else:
                    log(f"  {size} Vendors: Insufficient observations ({len(subset)})", f)

        log("", f)

        # -----------------------------------------------------------------
        # 3. Heterogeneity by Spend Intensity
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("3. HETEROGENEITY BY SPEND INTENSITY", f)
        log("-" * 40, f)

        log("  Test if effects differ for high vs low spenders.", f)
        log("", f)

        treated_panel = panel[panel['treated'] == 1].copy()

        if len(treated_panel) > 0:
            vendor_spend = treated_panel.groupby('VENDOR_ID')['total_spend'].mean().reset_index()
            vendor_spend.columns = ['VENDOR_ID', 'avg_spend']

            median_spend = vendor_spend['avg_spend'].median()
            vendor_spend['spend_group'] = np.where(
                vendor_spend['avg_spend'] >= median_spend,
                'High Spender',
                'Low Spender'
            )

            panel_with_spend = panel.merge(
                vendor_spend[['VENDOR_ID', 'spend_group']],
                on='VENDOR_ID',
                how='left'
            )
            panel_with_spend['spend_group'] = panel_with_spend['spend_group'].fillna('Non-Spender')

            log(f"  High spenders (avg spend >= {median_spend:.4f}): {(vendor_spend['spend_group'] == 'High Spender').sum():,}", f)
            log(f"  Low spenders (avg spend < {median_spend:.4f}): {(vendor_spend['spend_group'] == 'Low Spender').sum():,}", f)
            log("", f)

            for group in ['High Spender', 'Low Spender']:
                subset = panel_with_spend[panel_with_spend['spend_group'] == group]
                subset['vendor_str'] = subset['VENDOR_ID'].astype(str)
                subset['week_str'] = subset['week'].astype(str)

                if len(subset) > 100:
                    try:
                        model = pf.feols(
                            "log_promoted_gmv ~ treated | vendor_str + week_str",
                            data=subset
                        )

                        log(f"  {group}:", f)
                        log(f"    Coefficient: {model.coef()['treated']:.6f}", f)
                        log(f"    Std Error: {model.se()['treated']:.6f}", f)
                        log(f"    p-value: {model.pvalue()['treated']:.6f}", f)
                        log(f"    N: {model.nobs():,}", f)
                        log("", f)

                    except Exception as e:
                        log(f"  {group}: ERROR - {str(e)}", f)

        log("", f)

        # -----------------------------------------------------------------
        # 4. Balance Test
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("4. BALANCE TEST (PRE-TREATMENT CHARACTERISTICS)", f)
        log("-" * 40, f)

        log("  Compare pre-treatment characteristics between treated and control.", f)
        log("  Imbalance may indicate selection bias.", f)
        log("", f)

        # Get pre-treatment data
        pre_treatment = panel[panel['treated'] == 0].copy()

        if len(pre_treatment) > 0:
            # Identify which vendors eventually get treated
            ever_treated = panel[panel['cohort_id'] > 0]['VENDOR_ID'].unique()
            never_treated = panel[panel['cohort_id'] == 0]['VENDOR_ID'].unique()

            pre_ever_treated = pre_treatment[pre_treatment['VENDOR_ID'].isin(ever_treated)]
            pre_never_treated = pre_treatment[pre_treatment['VENDOR_ID'].isin(never_treated)]

            log(f"  Pre-treatment observations:", f)
            log(f"    Eventually treated: {len(pre_ever_treated):,}", f)
            log(f"    Never treated: {len(pre_never_treated):,}", f)
            log("", f)

            vars_to_compare = ['impressions', 'clicks', 'auction_participations']

            log(f"  {'Variable':<25} {'Treated Mean':>12} {'Control Mean':>12} {'Diff':>10} {'p-value':>10}", f)
            log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", f)

            for var in vars_to_compare:
                if var not in panel.columns:
                    continue

                treated_mean = pre_ever_treated[var].mean()
                control_mean = pre_never_treated[var].mean()
                diff = treated_mean - control_mean

                # t-test
                try:
                    t_stat, p_val = stats.ttest_ind(
                        pre_ever_treated[var].dropna(),
                        pre_never_treated[var].dropna()
                    )
                except:
                    p_val = np.nan

                log(f"  {var:<25} {treated_mean:>12.2f} {control_mean:>12.2f} {diff:>10.2f} {p_val:>10.4f}", f)

            log("", f)

            # Check for Ashenfelter's dip
            log("  ASHENFELTER'S DIP CHECK:", f)
            log("  Do vendors start advertising because sales are declining?", f)
            log("", f)

            if len(pre_ever_treated) > 0:
                # Look at trend in outcomes before treatment
                pre_ever_treated['weeks_to_treatment'] = (
                    pre_ever_treated['week'] - pre_ever_treated['cohort_week']
                ).dt.days / 7

                trend = pre_ever_treated.groupby('weeks_to_treatment')['impressions'].mean()

                if len(trend) > 1:
                    log("  Mean impressions by weeks-to-treatment:", f)
                    for t, val in trend.items():
                        log(f"    e = {int(t):+d}: {val:.2f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # 5. Attrition Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("5. ATTRITION ANALYSIS", f)
        log("-" * 40, f)

        log("  Check if vendors drop out of the panel after treatment.", f)
        log("  Differential attrition can bias estimates.", f)
        log("", f)

        # Check observations per vendor over time
        obs_per_vendor = panel.groupby('VENDOR_ID').size()

        log(f"  Observations per vendor:", f)
        log(f"    Mean: {obs_per_vendor.mean():.2f}", f)
        log(f"    Std: {obs_per_vendor.std():.2f}", f)
        log(f"    Min: {obs_per_vendor.min()}", f)
        log(f"    Max: {obs_per_vendor.max()}", f)
        log("", f)

        # Check if treated vendors have different coverage
        if 'cohort_id' in panel.columns:
            treated_vendors = panel[panel['cohort_id'] > 0]['VENDOR_ID'].unique()
            control_vendors = panel[panel['cohort_id'] == 0]['VENDOR_ID'].unique()

            treated_obs = obs_per_vendor[obs_per_vendor.index.isin(treated_vendors)]
            control_obs = obs_per_vendor[obs_per_vendor.index.isin(control_vendors)]

            log(f"  Treated vendors avg observations: {treated_obs.mean():.2f}", f)
            log(f"  Control vendors avg observations: {control_obs.mean():.2f}", f)

            # t-test for difference
            if len(treated_obs) > 0 and len(control_obs) > 0:
                t_stat, p_val = stats.ttest_ind(treated_obs, control_obs)
                log(f"  Difference p-value: {p_val:.4f}", f)

                if p_val < 0.05:
                    log("  WARNING: Significant difference in panel coverage", f)
                else:
                    log("  No significant difference in coverage", f)

        log("", f)
        log("=" * 80, f)
        log("ROBUSTNESS DIAGNOSTICS COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
