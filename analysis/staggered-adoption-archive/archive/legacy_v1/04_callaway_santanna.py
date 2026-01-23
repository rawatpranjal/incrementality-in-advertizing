#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Callaway & Sant'Anna Estimation
Estimates Group-Time ATT(g,t) using the CS (2021) methodology.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "04_callaway_santanna.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# CALLAWAY-SANTANNA ESTIMATOR (Manual Implementation)
# =============================================================================
def estimate_att_gt(panel, cohort_week, time_week, outcome_col, never_treated_cohort=0):
    """
    Estimate ATT(g, t) for a specific cohort g at time t.

    ATT(g, t) = E[Y_it - Y_{i,g-1} | G_i = g] - E[Y_it - Y_{i,g-1} | G_i = infinity]

    Parameters:
    -----------
    panel : DataFrame
        Panel data with vendor-week observations
    cohort_week : datetime
        The cohort week (G_i = g)
    time_week : datetime
        The time period t
    outcome_col : str
        Name of outcome variable
    never_treated_cohort : int
        Value indicating never-treated in cohort_id column

    Returns:
    --------
    dict with ATT estimate and diagnostics
    """
    # Get the baseline period (g-1)
    unique_weeks = sorted(panel['week'].unique())
    cohort_idx = unique_weeks.index(cohort_week) if cohort_week in unique_weeks else None

    if cohort_idx is None or cohort_idx == 0:
        return {'att': np.nan, 'se': np.nan, 'n_treated': 0, 'n_control': 0}

    baseline_week = unique_weeks[cohort_idx - 1]

    # Get treated cohort (G_i = g)
    treated_vendors = panel[panel['cohort_week'] == cohort_week]['VENDOR_ID'].unique()

    # Get never-treated control group
    control_vendors = panel[panel['cohort_id'] == never_treated_cohort]['VENDOR_ID'].unique()

    if len(treated_vendors) == 0 or len(control_vendors) == 0:
        return {'att': np.nan, 'se': np.nan, 'n_treated': 0, 'n_control': 0}

    # Calculate outcome changes for treated group
    treated_current = panel[(panel['VENDOR_ID'].isin(treated_vendors)) & (panel['week'] == time_week)]
    treated_baseline = panel[(panel['VENDOR_ID'].isin(treated_vendors)) & (panel['week'] == baseline_week)]

    if len(treated_current) == 0 or len(treated_baseline) == 0:
        return {'att': np.nan, 'se': np.nan, 'n_treated': 0, 'n_control': 0}

    # Merge to get paired differences
    treated_merged = treated_current[['VENDOR_ID', outcome_col]].merge(
        treated_baseline[['VENDOR_ID', outcome_col]],
        on='VENDOR_ID',
        suffixes=('_t', '_base')
    )
    treated_merged['delta'] = treated_merged[f'{outcome_col}_t'] - treated_merged[f'{outcome_col}_base']
    treated_delta_mean = treated_merged['delta'].mean()
    treated_delta_var = treated_merged['delta'].var()
    n_treated = len(treated_merged)

    # Calculate outcome changes for control group
    control_current = panel[(panel['VENDOR_ID'].isin(control_vendors)) & (panel['week'] == time_week)]
    control_baseline = panel[(panel['VENDOR_ID'].isin(control_vendors)) & (panel['week'] == baseline_week)]

    if len(control_current) == 0 or len(control_baseline) == 0:
        return {'att': np.nan, 'se': np.nan, 'n_treated': n_treated, 'n_control': 0}

    control_merged = control_current[['VENDOR_ID', outcome_col]].merge(
        control_baseline[['VENDOR_ID', outcome_col]],
        on='VENDOR_ID',
        suffixes=('_t', '_base')
    )
    control_merged['delta'] = control_merged[f'{outcome_col}_t'] - control_merged[f'{outcome_col}_base']
    control_delta_mean = control_merged['delta'].mean()
    control_delta_var = control_merged['delta'].var()
    n_control = len(control_merged)

    # ATT(g, t) = treated delta - control delta
    att = treated_delta_mean - control_delta_mean

    # Standard error (assuming independence)
    if n_treated > 1 and n_control > 1:
        se = np.sqrt(treated_delta_var / n_treated + control_delta_var / n_control)
    else:
        se = np.nan

    return {
        'att': att,
        'se': se,
        'n_treated': n_treated,
        'n_control': n_control,
        'treated_delta_mean': treated_delta_mean,
        'control_delta_mean': control_delta_mean,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("STAGGERED ADOPTION: CALLAWAY & SANT'ANNA ESTIMATION", f)
        log("=" * 80, f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  Callaway & Sant'Anna (2021) Group-Time ATT Estimator", f)
        log("", f)
        log("  ATT(g, t) = E[Y_it - Y_{i,g-1} | G_i = g] - E[Y_it - Y_{i,g-1} | G_i = inf]", f)
        log("", f)
        log("  Where:", f)
        log("    g = cohort (first treatment period)", f)
        log("    t = time period", f)
        log("    Y = outcome (log_promoted_gmv or other)", f)
        log("    G_i = inf represents never-treated control group", f)
        log("", f)
        log("  Identification Assumptions:", f)
        log("    1. Parallel Trends (conditional)", f)
        log("    2. No Anticipation", f)
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

        # Check for never-treated
        n_never_treated = panel[panel['cohort_id'] == 0]['VENDOR_ID'].nunique()
        log(f"  Never-treated vendors (control): {n_never_treated:,}", f)

        if n_never_treated == 0:
            log("  WARNING: No never-treated vendors available for control group", f)
            log("  Using not-yet-treated as control instead", f)

        log("", f)

        # -----------------------------------------------------------------
        # Identify cohorts and time periods
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("COHORT AND TIME STRUCTURE", f)
        log("-" * 40, f)

        # Get unique cohorts (excluding never-treated)
        cohorts = panel[panel['cohort_week'].notna()]['cohort_week'].unique()
        cohorts = sorted(cohorts)

        # Get unique time periods
        time_periods = sorted(panel['week'].unique())

        log(f"  Number of cohorts: {len(cohorts)}", f)
        log(f"  Number of time periods: {len(time_periods)}", f)
        log("", f)

        log("  Cohorts (first treatment week):", f)
        for g in cohorts:
            n_vendors = panel[panel['cohort_week'] == g]['VENDOR_ID'].nunique()
            log(f"    {g.date()}: {n_vendors:,} vendors", f)

        log("", f)

        # -----------------------------------------------------------------
        # Estimate ATT(g, t) for each cohort-time combination
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ESTIMATING ATT(g, t)", f)
        log("-" * 40, f)

        outcome_cols = ['log_promoted_gmv', 'promoted_gmv', 'clicks', 'impressions']
        results = []

        for outcome_col in outcome_cols:
            if outcome_col not in panel.columns:
                continue

            log(f"\n  Outcome: {outcome_col}", f)
            log(f"  {'Cohort':<12} {'Time':<12} {'ATT':>10} {'SE':>10} {'N_treat':>8} {'N_ctrl':>8}", f)
            log(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}", f)

            for cohort_week in tqdm(cohorts, desc=f"  Estimating {outcome_col}"):
                for time_week in time_periods:
                    result = estimate_att_gt(
                        panel=panel,
                        cohort_week=cohort_week,
                        time_week=time_week,
                        outcome_col=outcome_col,
                        never_treated_cohort=0
                    )

                    if not np.isnan(result['att']):
                        # Calculate relative time
                        rel_time = (time_week - cohort_week).days // 7

                        results.append({
                            'outcome': outcome_col,
                            'cohort_week': cohort_week,
                            'time_week': time_week,
                            'relative_time': rel_time,
                            'att': result['att'],
                            'se': result['se'],
                            'n_treated': result['n_treated'],
                            'n_control': result['n_control'],
                        })

                        log(f"  {cohort_week.date()!s:<12} {time_week.date()!s:<12} {result['att']:>10.4f} {result['se']:>10.4f} {result['n_treated']:>8} {result['n_control']:>8}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Convert results to DataFrame
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("RESULTS SUMMARY", f)
        log("-" * 40, f)

        if len(results) > 0:
            results_df = pd.DataFrame(results)

            # Save results
            output_path = DATA_DIR / "att_gt_estimates.parquet"
            results_df.to_parquet(output_path, index=False)
            log(f"  Saved {len(results_df):,} ATT(g,t) estimates to {output_path}", f)
            log("", f)

            # Summary by outcome
            for outcome_col in results_df['outcome'].unique():
                subset = results_df[results_df['outcome'] == outcome_col]

                log(f"  {outcome_col}:", f)
                log(f"    Number of ATT(g,t) estimates: {len(subset)}", f)
                log(f"    Mean ATT: {subset['att'].mean():.4f}", f)
                log(f"    Std ATT: {subset['att'].std():.4f}", f)
                log(f"    Min ATT: {subset['att'].min():.4f}", f)
                log(f"    Max ATT: {subset['att'].max():.4f}", f)
                log("", f)

            # Pre-trends check (relative_time < 0)
            log("  PRE-TRENDS CHECK (relative_time < 0):", f)
            for outcome_col in results_df['outcome'].unique():
                pre_trend = results_df[
                    (results_df['outcome'] == outcome_col) &
                    (results_df['relative_time'] < 0)
                ]
                if len(pre_trend) > 0:
                    log(f"    {outcome_col}:", f)
                    log(f"      Mean pre-trend ATT: {pre_trend['att'].mean():.4f}", f)
                    log(f"      # estimates: {len(pre_trend)}", f)
                    # Joint test: are pre-trends different from 0?
                    # Simple t-test on mean
                    if len(pre_trend) > 1:
                        t_stat = pre_trend['att'].mean() / (pre_trend['att'].std() / np.sqrt(len(pre_trend)))
                        log(f"      t-statistic (H0: mean=0): {t_stat:.4f}", f)

            log("", f)

            # Post-treatment effects (relative_time >= 0)
            log("  POST-TREATMENT EFFECTS (relative_time >= 0):", f)
            for outcome_col in results_df['outcome'].unique():
                post_treat = results_df[
                    (results_df['outcome'] == outcome_col) &
                    (results_df['relative_time'] >= 0)
                ]
                if len(post_treat) > 0:
                    log(f"    {outcome_col}:", f)
                    log(f"      Mean post-treatment ATT: {post_treat['att'].mean():.4f}", f)
                    log(f"      # estimates: {len(post_treat)}", f)
                    if len(post_treat) > 1:
                        t_stat = post_treat['att'].mean() / (post_treat['att'].std() / np.sqrt(len(post_treat)))
                        log(f"      t-statistic (H0: mean=0): {t_stat:.4f}", f)

        else:
            log("  No valid ATT estimates produced", f)
            log("  This may indicate insufficient data or panel structure issues", f)

        log("", f)
        log("=" * 80, f)
        log("CALLAWAY-SANTANNA ESTIMATION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
