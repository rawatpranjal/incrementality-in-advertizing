#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: TWFE Comparison
Compares naive Two-Way Fixed Effects to Callaway-Sant'Anna estimates.
Demonstrates potential bias from negative weights in staggered settings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyfixest as pf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "06_twfe_comparison.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# GOODMAN-BACON DECOMPOSITION (Simplified)
# =============================================================================
def bacon_decomposition(panel, outcome_col, treatment_col, unit_col, time_col):
    """
    Simplified Goodman-Bacon (2021) decomposition.

    Shows which 2x2 DiD comparisons contribute to TWFE estimate
    and their weights.

    Returns:
    --------
    DataFrame with decomposition components
    """
    results = []

    # Get unique cohorts
    cohorts = panel[panel['cohort_week'].notna()]['cohort_week'].unique()
    cohorts = sorted(cohorts)

    for i, g1 in enumerate(cohorts):
        for g2 in cohorts[i+1:]:
            # Early vs Late comparison
            # Treated (early) vs Control (late, before they're treated)

            early_treated = panel[panel['cohort_week'] == g1]
            late_treated = panel[panel['cohort_week'] == g2]

            # Time window: between g1 and g2
            time_window = panel[(panel['week'] >= g1) & (panel['week'] < g2)]['week'].unique()

            if len(time_window) == 0:
                continue

            # Calculate the 2x2 DiD
            # Early group is treated, late group is control
            early_post = early_treated[early_treated['week'].isin(time_window)][outcome_col].mean()
            early_pre = early_treated[early_treated['week'] < g1][outcome_col].mean()
            late_post = late_treated[late_treated['week'].isin(time_window)][outcome_col].mean()
            late_pre = late_treated[late_treated['week'] < g1][outcome_col].mean()

            if np.isnan(early_pre) or np.isnan(late_pre):
                continue

            did_estimate = (early_post - early_pre) - (late_post - late_pre)

            # Weight based on group sizes and variance
            n_early = len(early_treated['VENDOR_ID'].unique())
            n_late = len(late_treated['VENDOR_ID'].unique())

            results.append({
                'comparison': f'Early({g1.date()}) vs Late({g2.date()})',
                'early_cohort': g1,
                'late_cohort': g2,
                'did_estimate': did_estimate,
                'n_early': n_early,
                'n_late': n_late,
                'type': 'early_vs_late'
            })

    return pd.DataFrame(results)

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("STAGGERED ADOPTION: TWFE COMPARISON", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Compare naive Two-Way Fixed Effects (TWFE) estimator to", f)
        log("  Callaway-Sant'Anna heterogeneity-robust estimator.", f)
        log("", f)
        log("  TWFE Model: Y_it = alpha_i + lambda_t + beta * D_it + epsilon_it", f)
        log("", f)
        log("  Known Issues with TWFE in Staggered Settings:", f)
        log("    1. Negative weights on some ATT(g,t) components", f)
        log("    2. Already-treated units used as controls for newly-treated", f)
        log("    3. Bias when treatment effects are heterogeneous over time", f)
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
        log(f"  Unique weeks: {panel['week'].nunique()}", f)
        log("", f)

        # Create string identifiers for fixed effects
        panel['vendor_str'] = panel['VENDOR_ID'].astype(str)
        panel['week_str'] = panel['week'].astype(str)

        # -----------------------------------------------------------------
        # TWFE Estimation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TWFE ESTIMATION", f)
        log("-" * 40, f)

        outcomes = ['log_promoted_gmv', 'promoted_gmv', 'clicks', 'impressions']

        for outcome in outcomes:
            if outcome not in panel.columns:
                continue

            log(f"\n  OUTCOME: {outcome}", f)
            log(f"  Model: {outcome} ~ treated | vendor + week", f)
            log("", f)

            try:
                # TWFE with vendor and week fixed effects
                model = pf.feols(f"{outcome} ~ treated | vendor_str + week_str", data=panel)

                log("  TWFE Results:", f)
                log(f"    Coefficient (beta): {model.coef()['treated']:.6f}", f)
                log(f"    Std Error: {model.se()['treated']:.6f}", f)
                log(f"    t-statistic: {model.tstat()['treated']:.4f}", f)
                log(f"    p-value: {model.pvalue()['treated']:.6f}", f)
                log(f"    N observations: {model.nobs():,}", f)
                log(f"    R-squared: {model.r2():.4f}", f)

            except Exception as e:
                log(f"    ERROR: {str(e)}", f)

            log("", f)

        # -----------------------------------------------------------------
        # Compare to CS estimates
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("COMPARISON: TWFE vs CALLAWAY-SANTANNA", f)
        log("-" * 40, f)

        # Load event study results
        es_path = DATA_DIR / "event_study_coefficients.parquet"

        if es_path.exists():
            event_study = pd.read_parquet(es_path)

            for outcome in outcomes:
                if outcome not in panel.columns:
                    continue

                log(f"\n  OUTCOME: {outcome}", f)

                # CS average effect (post-treatment)
                cs_subset = event_study[
                    (event_study['outcome'] == outcome) &
                    (event_study['relative_time'] >= 0)
                ]

                if len(cs_subset) > 0:
                    cs_avg = cs_subset['theta'].mean()
                    log(f"    CS Average Effect (e >= 0): {cs_avg:.6f}", f)
                else:
                    log("    CS estimates not available", f)
                    continue

                # TWFE estimate
                try:
                    model = pf.feols(f"{outcome} ~ treated | vendor_str + week_str", data=panel)
                    twfe_beta = model.coef()['treated']
                    log(f"    TWFE Coefficient: {twfe_beta:.6f}", f)

                    # Difference
                    diff = twfe_beta - cs_avg
                    log(f"    Difference (TWFE - CS): {diff:.6f}", f)

                    if abs(diff) > abs(cs_avg) * 0.1:  # More than 10% difference
                        log("    WARNING: Substantial difference between estimators", f)
                        log("    This may indicate heterogeneous treatment effects", f)

                except Exception as e:
                    log(f"    TWFE failed: {str(e)}", f)

        else:
            log("  Event study results not found. Run 05_event_study_aggregation.py first.", f)

        log("", f)

        # -----------------------------------------------------------------
        # Goodman-Bacon Decomposition
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("GOODMAN-BACON DECOMPOSITION (SIMPLIFIED)", f)
        log("-" * 40, f)

        log("  This shows which 2x2 comparisons contribute to TWFE.", f)
        log("  In staggered settings, early-treated units can be used", f)
        log("  as controls for late-treated, which can create bias.", f)
        log("", f)

        decomp = bacon_decomposition(
            panel=panel,
            outcome_col='log_promoted_gmv',
            treatment_col='treated',
            unit_col='VENDOR_ID',
            time_col='week'
        )

        if len(decomp) > 0:
            log(f"  {'Comparison':<50} {'DiD Est':>10} {'N_early':>8} {'N_late':>8}", f)
            log(f"  {'-'*50} {'-'*10} {'-'*8} {'-'*8}", f)

            for _, row in decomp.iterrows():
                log(f"  {row['comparison']:<50} {row['did_estimate']:>10.4f} {row['n_early']:>8} {row['n_late']:>8}", f)

            log("", f)
            log("  Note: In staggered DiD, 'Early vs Late' comparisons can", f)
            log("  produce biased estimates if treatment effects are dynamic.", f)
        else:
            log("  Unable to compute decomposition (insufficient cohort variation)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Dynamic TWFE (Event Study Specification)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("DYNAMIC TWFE (EVENT STUDY SPECIFICATION)", f)
        log("-" * 40, f)

        log("  Model: Y_it = alpha_i + lambda_t + sum_e beta_e * 1{t-g = e} + epsilon_it", f)
        log("", f)

        # Create event-time dummies
        if 'relative_week' in panel.columns:
            # Filter to valid relative times
            valid_panel = panel[panel['relative_week'].notna()].copy()

            if len(valid_panel) > 0:
                # Create dummies for each relative time
                unique_rel_times = sorted(valid_panel['relative_week'].unique())

                for outcome in ['log_promoted_gmv', 'promoted_gmv']:
                    if outcome not in valid_panel.columns:
                        continue

                    log(f"\n  OUTCOME: {outcome}", f)

                    try:
                        # Use relative_week as categorical
                        valid_panel['rel_week_cat'] = valid_panel['relative_week'].astype(int).astype(str)

                        # Run with omitted category (e.g., -1)
                        model = pf.feols(
                            f"{outcome} ~ C(rel_week_cat) | vendor_str + week_str",
                            data=valid_panel
                        )

                        log("  Dynamic TWFE Coefficients:", f)
                        log(f"  {'e':<6} {'beta':>12} {'SE':>12} {'t':>10} {'p':>10}", f)
                        log(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", f)

                        coefs = model.coef()
                        ses = model.se()
                        tstats = model.tstat()
                        pvals = model.pvalue()

                        for key in sorted(coefs.keys()):
                            if 'rel_week_cat' in key:
                                # Extract relative time from key
                                rel_t = key.replace('C(rel_week_cat)[T.', '').replace(']', '')
                                sig = "***" if pvals[key] < 0.01 else ("**" if pvals[key] < 0.05 else ("*" if pvals[key] < 0.1 else ""))
                                log(f"  {rel_t:<6} {coefs[key]:>12.4f} {ses[key]:>12.4f} {tstats[key]:>10.3f} {pvals[key]:>10.4f} {sig}", f)

                    except Exception as e:
                        log(f"    ERROR: {str(e)}", f)

                log("", f)

        log("", f)
        log("=" * 80, f)
        log("TWFE COMPARISON COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
