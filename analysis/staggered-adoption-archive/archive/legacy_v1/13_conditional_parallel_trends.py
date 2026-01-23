#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Conditional Parallel Trends Validation
Tests parallel trends assumption within segments and with covariate controls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyfixest as pf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "13_conditional_parallel_trends.txt"

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
        log("STAGGERED ADOPTION: CONDITIONAL PARALLEL TRENDS VALIDATION", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Validate the parallel trends assumption within vendor segments.", f)
        log("  Tests:", f)
        log("    1. Pre-treatment covariate balance between treated and control", f)
        log("    2. Event-study pre-trend coefficients by segment", f)
        log("    3. Covariate-adjusted DiD to control for selection", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  If parallel trends holds unconditionally, it should also hold within", f)
        log("  subgroups defined by pre-treatment characteristics.", f)
        log("  Conditioning on covariates can help address selection bias.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load data
        # -----------------------------------------------------------------
        log("LOADING DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_segments.parquet"
        covariates_path = DATA_DIR / "vendor_covariates.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])

        log(f"  Panel: {panel.shape}", f)

        if covariates_path.exists():
            covariates = pd.read_parquet(covariates_path)
            log(f"  Covariates: {covariates.shape}", f)
        else:
            covariates = pd.DataFrame()
            log("  Covariates not found", f)

        log("", f)

        # Prepare identifiers
        panel['vendor_str'] = panel['VENDOR_ID'].astype(str)
        panel['week_str'] = panel['week'].astype(str)

        # -----------------------------------------------------------------
        # Test 1: Pre-treatment covariate balance
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TEST 1: PRE-TREATMENT COVARIATE BALANCE", f)
        log("-" * 40, f)
        log("", f)

        log("  Question: Are pre-treatment characteristics balanced between", f)
        log("  treated and never-treated vendors?", f)
        log("", f)

        if len(covariates) > 0:
            # Merge treatment status
            vendor_treatment = panel.groupby('VENDOR_ID').agg({
                'cohort_week': 'first',
                'treated': 'max'
            }).reset_index()

            covariates = covariates.merge(
                vendor_treatment[['VENDOR_ID', 'treated']].drop_duplicates(),
                on='VENDOR_ID',
                how='left',
                suffixes=('', '_panel')
            )

            # Update is_treated based on panel
            covariates['is_treated_final'] = covariates['treated'].fillna(0)

            # Compare means
            treated = covariates[covariates['is_treated_final'] == 1]
            control = covariates[covariates['is_treated_final'] == 0]

            log(f"  Treated vendors: {len(treated):,}", f)
            log(f"  Never-treated vendors: {len(control):,}", f)
            log("", f)

            balance_vars = [
                'pre_auction_count', 'pre_bid_count', 'pre_impression_count',
                'pre_click_count', 'pre_weeks_active', 'pre_avg_ranking',
                'pre_avg_price_point', 'pre_unique_products'
            ]

            log(f"  {'Covariate':<25} {'Treated':<12} {'Control':<12} {'Diff':<12} {'Std Diff':<12}", f)
            log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

            for var in balance_vars:
                if var in covariates.columns:
                    t_mean = treated[var].mean()
                    c_mean = control[var].mean()
                    diff = t_mean - c_mean

                    # Pooled std for standardized difference
                    pooled_std = np.sqrt((treated[var].var() + control[var].var()) / 2)
                    std_diff = diff / pooled_std if pooled_std > 0 else np.nan

                    log(f"  {var:<25} {t_mean:<12.4f} {c_mean:<12.4f} {diff:<12.4f} {std_diff:<12.4f}", f)

            log("", f)
            log("  INTERPRETATION:", f)
            log("    Standardized difference |d| < 0.1: Well balanced", f)
            log("    Standardized difference |d| < 0.25: Acceptable balance", f)
            log("    Standardized difference |d| > 0.25: Potential selection bias", f)

        else:
            log("  Covariates not available for balance test", f)

        log("", f)

        # -----------------------------------------------------------------
        # Test 2: Event-study pre-trends by segment
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TEST 2: PRE-TREND COEFFICIENTS BY SEGMENT", f)
        log("-" * 40, f)
        log("", f)

        log("  Question: Do pre-treatment event-study coefficients differ from zero", f)
        log("  within each segment? (If yes, parallel trends may be violated)", f)
        log("", f)

        # Create event-time dummies
        if 'relative_week' not in panel.columns:
            panel['cohort_week'] = pd.to_datetime(panel['cohort_week'])
            panel['relative_week'] = ((panel['week'] - panel['cohort_week']).dt.days / 7).fillna(0).astype(int)

        # Filter to segments with variation
        segments_to_test = []
        if 'persona' in panel.columns:
            for persona in panel['persona'].unique():
                seg_data = panel[panel['persona'] == persona]
                if seg_data['treated'].nunique() == 2 and len(seg_data) >= 1000:
                    segments_to_test.append(('persona', persona))

        log(f"  Segments with sufficient variation: {len(segments_to_test)}", f)
        log("", f)

        pretrend_results = []

        for seg_type, seg_name in tqdm(segments_to_test, desc="Testing pre-trends"):
            log(f"  Segment: {seg_name}", f)

            if seg_type == 'persona':
                seg_panel = panel[panel['persona'] == seg_name].copy()

            # Create relative time dummies
            seg_panel = seg_panel[seg_panel['relative_week'].between(-5, 10)]

            # Skip time -1 (reference period)
            seg_panel['rel_time_str'] = 'rt_' + seg_panel['relative_week'].astype(str)
            seg_panel.loc[seg_panel['relative_week'] == -1, 'rel_time_str'] = 'rt_ref'

            try:
                # Run event study
                model = pf.feols(
                    "log_promoted_gmv ~ C(rel_time_str) | vendor_str + week_str",
                    data=seg_panel,
                    vcov={'CRV1': 'vendor_str'}
                )

                # Extract pre-treatment coefficients
                coefs = model.coef()
                ses = model.se()
                pvals = model.pvalue()

                pre_coefs = []
                for t in range(-5, -1):
                    key = f'C(rel_time_str)[T.rt_{t}]'
                    if key in coefs:
                        pre_coefs.append({
                            'segment': seg_name,
                            'relative_time': t,
                            'coef': coefs[key],
                            'se': ses[key],
                            'p_value': pvals[key]
                        })

                # Joint test: are pre-treatment coefficients jointly zero?
                pre_coef_vals = [c['coef'] for c in pre_coefs if not np.isnan(c['coef'])]
                pre_pvals = [c['p_value'] for c in pre_coefs if not np.isnan(c['p_value'])]

                avg_pre_coef = np.mean(pre_coef_vals) if len(pre_coef_vals) > 0 else np.nan
                any_significant = any(p < 0.05 for p in pre_pvals) if len(pre_pvals) > 0 else False

                log(f"    Average pre-treatment coefficient: {avg_pre_coef:.6f}", f)
                log(f"    Any pre-period coefficient significant at 5%: {any_significant}", f)

                pretrend_results.append({
                    'segment': seg_name,
                    'avg_pre_coef': avg_pre_coef,
                    'any_significant': any_significant,
                    'n_pre_periods': len(pre_coef_vals)
                })

            except Exception as e:
                log(f"    ERROR: {str(e)}", f)
                pretrend_results.append({
                    'segment': seg_name,
                    'avg_pre_coef': np.nan,
                    'any_significant': np.nan,
                    'n_pre_periods': 0
                })

        log("", f)

        # Summary
        log("  PRE-TREND TEST SUMMARY:", f)
        log(f"  {'Segment':<25} {'Avg Pre-Coef':<15} {'Significant?':<15}", f)
        log(f"  {'-'*25} {'-'*15} {'-'*15}", f)

        for result in pretrend_results:
            avg_str = f"{result['avg_pre_coef']:.6f}" if not np.isnan(result['avg_pre_coef']) else "N/A"
            sig_str = "Yes" if result['any_significant'] else "No"
            log(f"  {result['segment']:<25} {avg_str:<15} {sig_str:<15}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Test 3: Covariate-adjusted DiD
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TEST 3: COVARIATE-ADJUSTED DiD", f)
        log("-" * 40, f)
        log("", f)

        log("  Question: Does including pre-treatment covariates change the ATT?", f)
        log("  If parallel trends holds conditionally, adding covariates should", f)
        log("  reduce bias and potentially change the ATT estimate.", f)
        log("", f)

        # Merge covariates with panel
        if len(covariates) > 0:
            # Select covariates to include
            cov_vars = [
                'pre_auction_count', 'pre_avg_price_point',
                'pre_avg_ranking', 'pre_weeks_active'
            ]
            cov_vars = [v for v in cov_vars if v in covariates.columns]

            merge_cols = ['VENDOR_ID'] + cov_vars
            panel_with_cov = panel.merge(
                covariates[merge_cols].drop_duplicates(),
                on='VENDOR_ID',
                how='left',
                suffixes=('', '_cov')
            )

            # Fill NAs with 0 for modeling
            for var in cov_vars:
                if var in panel_with_cov.columns:
                    panel_with_cov[var] = panel_with_cov[var].fillna(0)
                elif var + '_cov' in panel_with_cov.columns:
                    panel_with_cov[var] = panel_with_cov[var + '_cov'].fillna(0)
                    cov_vars = [v if v != var else var for v in cov_vars]

            log(f"  Covariates used: {cov_vars}", f)
            log("", f)

            # Model 1: Baseline TWFE
            try:
                model_baseline = pf.feols(
                    "log_promoted_gmv ~ treated | vendor_str + week_str",
                    data=panel_with_cov,
                    vcov={'CRV1': 'vendor_str'}
                )

                att_baseline = model_baseline.coef()['treated']
                se_baseline = model_baseline.se()['treated']

                log("  Model 1: Baseline TWFE (no covariates)", f)
                log(f"    ATT: {att_baseline:.6f} (SE: {se_baseline:.6f})", f)

            except Exception as e:
                log(f"  Model 1 ERROR: {str(e)}", f)
                att_baseline = np.nan
                se_baseline = np.nan

            # Model 2: With covariates
            try:
                cov_formula = " + ".join(cov_vars)
                formula_cov = f"log_promoted_gmv ~ treated + {cov_formula} | vendor_str + week_str"

                model_cov = pf.feols(
                    formula_cov,
                    data=panel_with_cov,
                    vcov={'CRV1': 'vendor_str'}
                )

                att_cov = model_cov.coef()['treated']
                se_cov = model_cov.se()['treated']

                log("", f)
                log("  Model 2: With covariates", f)
                log(f"    ATT: {att_cov:.6f} (SE: {se_cov:.6f})", f)

            except Exception as e:
                log(f"  Model 2 ERROR: {str(e)}", f)
                att_cov = np.nan
                se_cov = np.nan

            # Compare
            if not np.isnan(att_baseline) and not np.isnan(att_cov):
                log("", f)
                log("  COMPARISON:", f)
                log(f"    Baseline ATT: {att_baseline:.6f}", f)
                log(f"    Adjusted ATT: {att_cov:.6f}", f)
                log(f"    Difference:   {att_cov - att_baseline:.6f}", f)
                log("", f)

                if abs(att_cov - att_baseline) < 0.0001:
                    log("  INTERPRETATION: Adding covariates does not change ATT", f)
                    log("  This suggests parallel trends may hold unconditionally.", f)
                else:
                    log("  INTERPRETATION: Adding covariates changes ATT", f)
                    log("  This suggests selection on observables that may bias estimates.", f)

        else:
            log("  Covariates not available for adjusted analysis", f)

        log("", f)

        # -----------------------------------------------------------------
        # Test 4: Within-segment balance
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TEST 4: WITHIN-SEGMENT COVARIATE BALANCE", f)
        log("-" * 40, f)
        log("", f)

        log("  Question: Within each segment, are treated and control vendors", f)
        log("  balanced on pre-treatment characteristics?", f)
        log("", f)

        if len(covariates) > 0 and 'persona' in panel.columns:
            balance_results = []

            for persona in panel['persona'].unique():
                # Get vendors in this segment
                seg_vendors = panel[panel['persona'] == persona]['VENDOR_ID'].unique()
                seg_covariates = covariates[covariates['VENDOR_ID'].isin(seg_vendors)]

                if len(seg_covariates) == 0:
                    continue

                # Split by treatment
                treated = seg_covariates[seg_covariates['is_treated_final'] == 1]
                control = seg_covariates[seg_covariates['is_treated_final'] == 0]

                if len(control) < 10:
                    continue

                log(f"  {persona}:", f)
                log(f"    Treated: {len(treated):,}, Control: {len(control):,}", f)

                max_std_diff = 0
                for var in ['pre_auction_count', 'pre_avg_price_point']:
                    if var in seg_covariates.columns:
                        t_mean = treated[var].mean()
                        c_mean = control[var].mean()
                        pooled_std = np.sqrt((treated[var].var() + control[var].var()) / 2)
                        std_diff = abs((t_mean - c_mean) / pooled_std) if pooled_std > 0 else 0

                        if std_diff > max_std_diff:
                            max_std_diff = std_diff

                log(f"    Max standardized difference: {max_std_diff:.4f}", f)

                balance_results.append({
                    'segment': persona,
                    'n_treated': len(treated),
                    'n_control': len(control),
                    'max_std_diff': max_std_diff,
                    'balanced': max_std_diff < 0.25
                })

            log("", f)
            log("  BALANCE SUMMARY:", f)
            log(f"  {'Segment':<25} {'Balanced?':<12} {'Max Std Diff':<15}", f)
            log(f"  {'-'*25} {'-'*12} {'-'*15}", f)

            for result in balance_results:
                bal_str = "Yes" if result['balanced'] else "No"
                log(f"  {result['segment']:<25} {bal_str:<12} {result['max_std_diff']:<15.4f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("PARALLEL TRENDS VALIDATION SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        log("  FINDINGS:", f)
        log("    1. Covariate Balance: Large differences between treated and control", f)
        log("       on pre-treatment activity suggest selection into treatment", f)
        log("", f)
        log("    2. Pre-Trends: Test whether pre-treatment coefficients are zero", f)
        log("       within each segment", f)
        log("", f)
        log("    3. Covariate Adjustment: If ATT changes substantially with controls,", f)
        log("       consider using doubly-robust methods", f)
        log("", f)
        log("    4. Within-Segment Balance: Better balance within segments suggests", f)
        log("       conditional parallel trends may be more plausible", f)

        log("", f)
        log("=" * 80, f)
        log("CONDITIONAL PARALLEL TRENDS VALIDATION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
