#!/usr/bin/env python3
"""
Refined Incrementality Analysis (V2)
Master model: TWFE with fixed cohort (vendors with pre-treatment activity).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import pyfixest as pf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "18_refined_analysis.txt"

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
        log("REFINED INCREMENTALITY ANALYSIS (V2)", f)
        log("=" * 80, f)
        log("", f)

        log("RESEARCH QUESTION:", f)
        log("  Does vendor advertising cause incremental promoted sales (GMV)?", f)
        log("", f)
        log("MASTER MODEL: TWFE", f)
        log("  Y_it = α_i + λ_t + β D_it + ε_it", f)
        log("", f)
        log("KEY FIX: Cohort restricted to vendors with pre-treatment activity.", f)
        log("", f)
        log("DATA LIMITATION:", f)
        log("  Cannot compute organic_gmv or total_gmv (PURCHASES lacks VENDOR_ID).", f)
        log("  Primary outcome: promoted_gmv (click-attributed only).", f)
        log("", f)

        # =================================================================
        # PHASE 1: SAMPLE & COHORT DEFINITION
        # =================================================================
        log("=" * 80, f)
        log("PHASE 1: SAMPLE DEFINITION", f)
        log("=" * 80, f)
        log("", f)

        panel_path = DATA_DIR / "panel_with_cohorts.parquet"
        if not panel_path.exists():
            log("ERROR: panel_with_cohorts.parquet not found", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])
        panel['cohort_week'] = pd.to_datetime(panel['cohort_week'])

        cov_path = DATA_DIR / "vendor_covariates.parquet"
        if cov_path.exists():
            covariates = pd.read_parquet(cov_path)
        else:
            log("ERROR: vendor_covariates.parquet not found", f)
            return

        log("ORIGINAL SAMPLE:", f)
        log(f"  Panel observations: {len(panel):,}", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log(f"  Weeks: {panel['week'].nunique()}", f)
        log("", f)

        # Analysis cohort: vendors with pre-treatment activity
        vendors_with_preactivity = covariates[covariates['pre_auction_count'] > 0]['VENDOR_ID'].unique()

        log("ANALYSIS COHORT:", f)
        log("  Criteria: pre_auction_count > 0", f)
        log(f"  Vendors with pre-treatment activity: {len(vendors_with_preactivity):,}", f)
        log("", f)

        panel_filtered = panel[panel['VENDOR_ID'].isin(vendors_with_preactivity)].copy()

        log("FILTERED SAMPLE:", f)
        log(f"  Panel observations: {len(panel_filtered):,}", f)
        log(f"  Unique vendors: {panel_filtered['VENDOR_ID'].nunique():,}", f)
        log("", f)

        treated_vendors = panel_filtered[panel_filtered['cohort_id'] > 0]['VENDOR_ID'].unique()
        control_vendors = panel_filtered[panel_filtered['cohort_id'] == 0]['VENDOR_ID'].unique()

        log("TREATMENT/CONTROL:", f)
        log(f"  Treated (ever spent): {len(treated_vendors):,}", f)
        log(f"  Control (never spent): {len(control_vendors):,}", f)
        log(f"  Treatment rate: {len(treated_vendors) / (len(treated_vendors) + len(control_vendors)) * 100:.1f}%", f)
        log("", f)

        original_vendors = panel['VENDOR_ID'].nunique()
        dropped = original_vendors - len(vendors_with_preactivity)
        log("EXCLUSION SUMMARY:", f)
        log(f"  Original vendors: {original_vendors:,}", f)
        log(f"  Dropped (no pre-treatment activity): {dropped:,} ({dropped/original_vendors*100:.1f}%)", f)
        log(f"  Final cohort: {len(vendors_with_preactivity):,}", f)
        log("", f)

        # =================================================================
        # PHASE 2: COVARIATE BALANCE
        # =================================================================
        log("=" * 80, f)
        log("PHASE 2: COVARIATE BALANCE", f)
        log("=" * 80, f)
        log("", f)

        cov_analysis = covariates[covariates['VENDOR_ID'].isin(vendors_with_preactivity)].copy()
        cov_analysis['is_control'] = cov_analysis['VENDOR_ID'].isin(control_vendors).astype(int)
        cov_analysis['is_treated'] = (~cov_analysis['VENDOR_ID'].isin(control_vendors)).astype(int)

        balance_vars = ['pre_auction_count', 'pre_bid_count', 'pre_win_rate',
                       'pre_weeks_active', 'pre_avg_price_point', 'pre_avg_ranking']

        log("STANDARDIZED DIFFERENCE: d = (μ_T - μ_C) / pooled_σ", f)
        log("  |d| < 0.10: Well-balanced | |d| < 0.25: Acceptable | |d| > 0.25: Imbalanced", f)
        log("", f)

        log(f"  {'Variable':<25} {'Treated':<12} {'Control':<12} {'Std Diff':<10} {'Status':<12}", f)
        log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*12}", f)

        max_imbalance = 0
        for var in balance_vars:
            if var not in cov_analysis.columns:
                continue

            treated_data = cov_analysis[cov_analysis['is_treated'] == 1][var].dropna()
            control_data = cov_analysis[cov_analysis['is_control'] == 1][var].dropna()

            if len(treated_data) == 0 or len(control_data) == 0:
                continue

            mu_t, mu_c = treated_data.mean(), control_data.mean()
            var_t, var_c = treated_data.var(), control_data.var()
            pooled_sd = np.sqrt((var_t + var_c) / 2)

            std_diff = (mu_t - mu_c) / pooled_sd if pooled_sd > 0 else 0
            max_imbalance = max(max_imbalance, abs(std_diff))

            if abs(std_diff) < 0.10:
                status = "Balanced"
            elif abs(std_diff) < 0.25:
                status = "Acceptable"
            else:
                status = "IMBALANCED"

            log(f"  {var:<25} {mu_t:<12.2f} {mu_c:<12.2f} {std_diff:<10.4f} {status:<12}", f)

        log("", f)
        log(f"  Max |d| = {max_imbalance:.4f}", f)
        log("", f)

        # =================================================================
        # PHASE 3: MASTER MODEL (TWFE)
        # =================================================================
        log("=" * 80, f)
        log("PHASE 3: MASTER MODEL (TWFE)", f)
        log("=" * 80, f)
        log("", f)

        log("MODEL SPECIFICATION:", f)
        log("  Y_it = α_i + λ_t + β D_it + ε_it", f)
        log("", f)
        log("  α_i = vendor fixed effect", f)
        log("  λ_t = week fixed effect", f)
        log("  D_it = 1{vendor i treated at time t}", f)
        log("  β = ATT (average treatment effect on treated)", f)
        log("", f)

        panel_filtered['vendor_str'] = panel_filtered['VENDOR_ID'].astype(str)
        panel_filtered['week_str'] = panel_filtered['week'].dt.strftime('%Y-%m-%d')

        results = {}
        for outcome in ['log_promoted_gmv', 'promoted_gmv', 'clicks', 'impressions']:
            if outcome not in panel_filtered.columns:
                continue

            try:
                model = pf.feols(
                    f"{outcome} ~ treated | vendor_str + week_str",
                    data=panel_filtered,
                    vcov={'CRV1': 'vendor_str'}
                )

                coef = model.coef()['treated']
                se = model.se()['treated']
                t = model.tstat()['treated']
                p = model.pvalue()['treated']

                results[outcome] = {'coef': coef, 'se': se, 't': t, 'p': p}

                log(f"  {outcome}:", f)
                log(f"    β = {coef:.6f} (SE: {se:.6f})", f)
                log(f"    t = {t:.4f}, p = {p:.4f}", f)
                log("", f)

            except Exception as e:
                log(f"  {outcome}: Error - {str(e)[:50]}", f)
                log("", f)

        # =================================================================
        # PHASE 4: iROAS CALCULATION
        # =================================================================
        log("=" * 80, f)
        log("PHASE 4: iROAS CALCULATION", f)
        log("=" * 80, f)
        log("", f)

        if 'promoted_gmv' in results:
            att_gmv = results['promoted_gmv']['coef']
            att_gmv_se = results['promoted_gmv']['se']
        else:
            att_gmv = 0
            att_gmv_se = 0

        if 'log_promoted_gmv' in results:
            att_log = results['log_promoted_gmv']['coef']
            pct_effect = (np.exp(att_log) - 1) * 100
        else:
            att_log = 0
            pct_effect = 0

        log("ATT ESTIMATES:", f)
        log(f"  ATT(promoted_gmv): ${att_gmv:,.2f} (SE: ${att_gmv_se:,.2f})", f)
        log(f"  ATT(log_promoted_gmv): {att_log:.6f} ({pct_effect:.4f}%)", f)
        log("", f)

        treated_panel = panel_filtered[panel_filtered['VENDOR_ID'].isin(treated_vendors)]
        treated_post = treated_panel[treated_panel['treated'] == 1]

        total_spend = treated_post['total_spend'].sum()
        avg_weekly_spend = total_spend / len(treated_post) if len(treated_post) > 0 else 0

        log("SPEND METRICS:", f)
        log(f"  Total spend (treated, post): ${total_spend:,.2f}", f)
        log(f"  Treated vendor-weeks: {len(treated_post):,}", f)
        log(f"  Avg weekly spend: ${avg_weekly_spend:,.2f}", f)
        log("", f)

        iroas = att_gmv / avg_weekly_spend if avg_weekly_spend > 0 else np.nan

        log("iROAS:", f)
        log(f"  iROAS = ATT / avg_spend = ${att_gmv:,.2f} / ${avg_weekly_spend:,.2f} = {iroas:.4f}", f)
        log("", f)

        if not np.isnan(iroas):
            if iroas > 1:
                log(f"  Result: Profitable (${iroas:.2f} return per $1 spent)", f)
            else:
                log(f"  Result: Unprofitable (${iroas:.2f} return per $1 spent)", f)
        log("", f)

        # =================================================================
        # PHASE 5: VALIDATION (ATTRITION)
        # =================================================================
        log("=" * 80, f)
        log("PHASE 5: VALIDATION", f)
        log("=" * 80, f)
        log("", f)

        treated_weeks = panel_filtered[panel_filtered['VENDOR_ID'].isin(treated_vendors)].groupby('VENDOR_ID').size()
        control_weeks = panel_filtered[panel_filtered['VENDOR_ID'].isin(control_vendors)].groupby('VENDOR_ID').size()

        log("PANEL PRESENCE:", f)
        log(f"  Treated: {treated_weeks.mean():.2f} weeks avg (n={len(treated_weeks):,})", f)
        log(f"  Control: {control_weeks.mean():.2f} weeks avg (n={len(control_weeks):,})", f)
        log("", f)

        t_stat, p_val = stats.ttest_ind(treated_weeks, control_weeks)
        log("DIFFERENTIAL ATTRITION TEST:", f)
        log(f"  t = {t_stat:.4f}, p = {p_val:.4f}", f)
        if p_val < 0.05:
            log("  WARNING: Significant differential attrition.", f)
        else:
            log("  No significant differential attrition.", f)
        log("", f)

        log("NOTE: Control group has promoted_gmv = 0 by definition (never won auctions).", f)
        log("", f)

        # =================================================================
        # SUMMARY
        # =================================================================
        log("=" * 80, f)
        log("SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("SAMPLE:", f)
        log(f"  {len(vendors_with_preactivity):,} vendors with pre-treatment activity", f)
        log(f"  Treated: {len(treated_vendors):,} | Control: {len(control_vendors):,}", f)
        log("", f)

        log("MASTER MODEL (TWFE):", f)
        log(f"  ATT(promoted_gmv): ${att_gmv:,.2f}", f)
        log(f"  ATT(log): {att_log:.6f} ({pct_effect:.4f}%)", f)
        log("", f)

        log("ECONOMICS:", f)
        log(f"  iROAS: {iroas:.4f}", f)
        log("", f)

        log("VALIDATION:", f)
        log(f"  Max covariate imbalance: {max_imbalance:.4f}", f)
        log(f"  Attrition p-value: {p_val:.4f}", f)
        log("", f)

        log("=" * 80, f)

if __name__ == "__main__":
    main()
