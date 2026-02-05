#!/usr/bin/env python3
"""
03_estimate_models.py - Causal Estimation (Callaway-Sant'Anna & TWFE)

Estimates Average Treatment Effect on the Treated (ATT) using:
1. Callaway-Sant'Anna (2021) group-time estimator
2. Two-Way Fixed Effects (TWFE) for comparison
3. Event study aggregation for pre-trend tests

Input: data/vendor_weekly_panel.parquet
Output: results/03_estimate_models.txt
"""

import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

LOG_FILE = RESULTS_DIR / '03_estimate_models.txt'

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)

def assign_cohorts(panel):
    """Assign treatment cohorts based on first click week."""
    # First week with clicks > 0
    first_click = panel[panel['HAS_CLICKS'] == 1].groupby('VENDOR_ID')['WEEK'].min().reset_index()
    first_click.columns = ['VENDOR_ID', 'COHORT_WEEK']

    # Merge back
    panel = panel.merge(first_click, on='VENDOR_ID', how='left')

    # Never-treated get cohort = infinity (we'll use NaT)
    panel['IS_TREATED'] = panel['COHORT_WEEK'].notna().astype(int)

    # Post-treatment indicator
    panel['POST'] = (panel['WEEK'] >= panel['COHORT_WEEK']).astype(int)
    panel.loc[panel['COHORT_WEEK'].isna(), 'POST'] = 0

    # Relative time (event time)
    panel['RELATIVE_WEEK'] = np.where(
        panel['COHORT_WEEK'].notna(),
        (pd.to_datetime(panel['WEEK']) - pd.to_datetime(panel['COHORT_WEEK'])).dt.days // 7,
        np.nan
    )

    return panel

def twfe_estimate(panel, outcome='TOTAL_GMV'):
    """
    Two-Way Fixed Effects estimator.
    Y_it = alpha_i + lambda_t + beta * D_it + epsilon_it
    """
    import statsmodels.api as sm
    from scipy import sparse

    # Filter to valid observations
    df = panel[['VENDOR_ID', 'WEEK', outcome, 'POST']].dropna()
    df = df[df[outcome] >= 0].copy()

    # Log transform
    df['Y'] = np.log(df[outcome] + 1)

    # Create numeric IDs
    vendor_ids = df['VENDOR_ID'].astype('category').cat.codes
    week_ids = df['WEEK'].astype('category').cat.codes

    # Demeaning approach (within transformation)
    df['vendor_code'] = vendor_ids
    df['week_code'] = week_ids

    # Vendor means
    vendor_means = df.groupby('vendor_code')['Y'].transform('mean')
    week_means = df.groupby('week_code')['Y'].transform('mean')
    grand_mean = df['Y'].mean()

    df['Y_within'] = df['Y'] - vendor_means - week_means + grand_mean

    # Same for treatment
    vendor_post_means = df.groupby('vendor_code')['POST'].transform('mean')
    week_post_means = df.groupby('week_code')['POST'].transform('mean')
    grand_post_mean = df['POST'].mean()
    df['POST_within'] = df['POST'] - vendor_post_means - week_post_means + grand_post_mean

    # OLS on within-transformed data
    X = sm.add_constant(df['POST_within'])
    model = sm.OLS(df['Y_within'], X).fit(cov_type='cluster', cov_kwds={'groups': df['vendor_code']})

    return {
        'beta': model.params['POST_within'],
        'se': model.bse['POST_within'],
        't': model.tvalues['POST_within'],
        'p': model.pvalues['POST_within'],
        'n_obs': len(df),
        'n_vendors': df['vendor_code'].nunique(),
        'n_weeks': df['week_code'].nunique()
    }

def callaway_santanna_manual(panel, outcome='TOTAL_GMV', control='never_treated'):
    """
    Manual implementation of Callaway-Sant'Anna (2021) group-time ATT.

    ATT(g,t) = E[Y_it - Y_{i,g-1} | G=g] - E[Y_it - Y_{i,g-1} | G=inf]

    Uses never-treated units as comparison group.
    """
    df = panel.copy()
    df['Y'] = np.log(df[outcome] + 1)
    df['WEEK'] = pd.to_datetime(df['WEEK'])
    df['COHORT_WEEK'] = pd.to_datetime(df['COHORT_WEEK'])

    # Get unique cohorts (excluding never-treated)
    cohorts = df[df['COHORT_WEEK'].notna()]['COHORT_WEEK'].unique()
    cohorts = sorted(cohorts)

    # Get unique time periods
    periods = sorted(df['WEEK'].unique())

    # Never-treated group
    never_treated = df[df['COHORT_WEEK'].isna()]['VENDOR_ID'].unique()

    results = []

    for g in tqdm(cohorts, desc="Computing ATT(g,t)"):
        # Baseline period (g-1)
        baseline = g - pd.Timedelta(weeks=1)
        if baseline not in periods:
            continue

        # Vendors in cohort g
        cohort_vendors = df[df['COHORT_WEEK'] == g]['VENDOR_ID'].unique()
        n_treated = len(cohort_vendors)

        if n_treated < 10:
            continue

        # Baseline outcome for treated
        treated_baseline = df[(df['VENDOR_ID'].isin(cohort_vendors)) &
                               (df['WEEK'] == baseline)]['Y'].mean()

        # Baseline for never-treated
        control_baseline = df[(df['VENDOR_ID'].isin(never_treated)) &
                               (df['WEEK'] == baseline)]['Y'].mean()

        for t in periods:
            if t < g:  # Pre-treatment
                rel_time = int((t - g).days / 7)
            else:  # Post-treatment
                rel_time = int((t - g).days / 7)

            # Outcome at time t for treated
            treated_t = df[(df['VENDOR_ID'].isin(cohort_vendors)) &
                           (df['WEEK'] == t)]['Y'].mean()

            # Outcome at time t for never-treated
            control_t = df[(df['VENDOR_ID'].isin(never_treated)) &
                           (df['WEEK'] == t)]['Y'].mean()

            # DiD
            att_gt = (treated_t - treated_baseline) - (control_t - control_baseline)

            # Standard error (simplified)
            n_control = len(never_treated)
            treated_var = df[(df['VENDOR_ID'].isin(cohort_vendors)) & (df['WEEK'] == t)]['Y'].var()
            control_var = df[(df['VENDOR_ID'].isin(never_treated)) & (df['WEEK'] == t)]['Y'].var()

            se = np.sqrt(treated_var / n_treated + control_var / n_control) if n_treated > 0 and n_control > 0 else np.nan

            results.append({
                'cohort': g,
                'period': t,
                'relative_time': rel_time,
                'att_gt': att_gt,
                'se': se,
                'n_treated': n_treated,
                'n_control': n_control
            })

    return pd.DataFrame(results)

def aggregate_event_study(att_gt_df):
    """Aggregate ATT(g,t) to event-study coefficients theta(e)."""
    # Weight by cohort size
    agg = att_gt_df.groupby('relative_time').apply(
        lambda x: pd.Series({
            'theta': np.average(x['att_gt'], weights=x['n_treated']),
            'se': np.sqrt(np.average(x['se']**2, weights=x['n_treated'])),
            'n': x['n_treated'].sum()
        })
    ).reset_index()

    agg['t_stat'] = agg['theta'] / agg['se']
    agg['p_value'] = 2 * (1 - pd.Series([abs(t) for t in agg['t_stat']]).apply(
        lambda t: 0.5 * (1 + np.sign(t) * (1 - np.exp(-0.717 * t - 0.416 * t**2)))
    ))

    return agg

def main():
    print("=" * 70)
    print("03_ESTIMATE_MODELS.PY - Causal Estimation")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # =========================================================================
    # Load Panel
    # =========================================================================
    print("[LOADING] Vendor-Week Panel...")
    panel_path = DATA_DIR / 'vendor_weekly_panel.parquet'
    if not panel_path.exists():
        print(f"  ERROR: Panel not found at {panel_path}")
        print("  Run 01_build_panel.py first.")
        return

    panel = pd.read_parquet(panel_path)
    print(f"  Loaded: {len(panel):,} rows")
    print()

    # =========================================================================
    # STEP 1: Assign Cohorts
    # =========================================================================
    print("[STEP 1] Assigning Treatment Cohorts...")
    panel = assign_cohorts(panel)

    n_treated = panel[panel['IS_TREATED'] == 1]['VENDOR_ID'].nunique()
    n_never = panel[panel['IS_TREATED'] == 0]['VENDOR_ID'].nunique()
    n_cohorts = panel[panel['COHORT_WEEK'].notna()]['COHORT_WEEK'].nunique()

    print(f"  Treated vendors: {n_treated:,}")
    print(f"  Never-treated vendors: {n_never:,}")
    print(f"  Unique cohorts: {n_cohorts}")
    print()

    if n_never < 10:
        print("  WARNING: Very few never-treated units. CS estimates may be unstable.")
        print("  Consider using not-yet-treated as comparison group.")
    print()

    # =========================================================================
    # STEP 2: TWFE Estimation
    # =========================================================================
    print("=" * 70)
    print("[STEP 2] TWO-WAY FIXED EFFECTS (TWFE)")
    print("=" * 70)
    print()

    outcomes = ['TOTAL_GMV', 'PROMOTED_GMV', 'CLICKS', 'IMPRESSIONS']

    print("  Model: log(Y + 1) = alpha_i + lambda_t + beta * POST_it + epsilon_it")
    print("  Standard errors: Clustered by vendor")
    print()

    twfe_results = {}
    print("  " + "-" * 70)
    print(f"  {'Outcome':<15} {'Beta':>10} {'SE':>10} {'t-stat':>10} {'p-value':>10} {'N':>10}")
    print("  " + "-" * 70)

    for outcome in outcomes:
        try:
            res = twfe_estimate(panel, outcome)
            twfe_results[outcome] = res
            sig = "***" if res['p'] < 0.01 else "**" if res['p'] < 0.05 else "*" if res['p'] < 0.1 else ""
            print(f"  {outcome:<15} {res['beta']:>10.6f} {res['se']:>10.6f} {res['t']:>10.3f} {res['p']:>10.4f} {res['n_obs']:>10,} {sig}")
        except Exception as e:
            print(f"  {outcome:<15} ERROR: {str(e)[:40]}")

    print("  " + "-" * 70)
    print("  Significance: *** p<0.01, ** p<0.05, * p<0.1")
    print()

    print("  Interpretation:")
    if 'TOTAL_GMV' in twfe_results:
        beta = twfe_results['TOTAL_GMV']['beta']
        pct_effect = (np.exp(beta) - 1) * 100
        print(f"    TOTAL_GMV: {pct_effect:.3f}% change from treatment")
    print()

    # =========================================================================
    # STEP 3: Callaway-Sant'Anna Group-Time ATT
    # =========================================================================
    print("=" * 70)
    print("[STEP 3] CALLAWAY-SANT'ANNA (2021) GROUP-TIME ATT")
    print("=" * 70)
    print()

    print("  Estimating ATT(g,t) for all cohort-time pairs...")
    print("  Comparison group: Never-treated vendors")
    print()

    att_gt = callaway_santanna_manual(panel, outcome='TOTAL_GMV')
    print(f"  Computed {len(att_gt):,} group-time ATT estimates")
    print()

    # Show sample of ATT(g,t)
    print("  Sample ATT(g,t) estimates:")
    print("  " + "-" * 70)
    sample = att_gt.sort_values(['cohort', 'period']).head(20)
    print(f"  {'Cohort':<12} {'Period':<12} {'Rel.Time':>8} {'ATT(g,t)':>12} {'SE':>10} {'N_treat':>8}")
    print("  " + "-" * 70)
    for _, row in sample.iterrows():
        print(f"  {str(row['cohort'].date()):<12} {str(row['period'].date()):<12} {row['relative_time']:>8} {row['att_gt']:>12.6f} {row['se']:>10.6f} {row['n_treated']:>8}")
    print("  " + "-" * 70)
    print()

    # =========================================================================
    # STEP 4: Event Study Aggregation
    # =========================================================================
    print("=" * 70)
    print("[STEP 4] EVENT STUDY AGGREGATION")
    print("=" * 70)
    print()

    event_study = aggregate_event_study(att_gt)

    print("  theta(e) = weighted average of ATT(g, g+e) across cohorts")
    print()
    print("  " + "-" * 70)
    print(f"  {'Rel.Week':>8} {'theta(e)':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10} {'N':>10}")
    print("  " + "-" * 70)

    for _, row in event_study.sort_values('relative_time').iterrows():
        marker = "<<<" if row['relative_time'] == 0 else ""
        sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
        print(f"  {row['relative_time']:>8} {row['theta']:>12.6f} {row['se']:>10.6f} {row['t_stat']:>10.3f} {row['p_value']:>10.4f} {int(row['n']):>10,} {sig} {marker}")

    print("  " + "-" * 70)
    print()

    # Pre-trend test
    pre_periods = event_study[event_study['relative_time'] < 0]
    if len(pre_periods) > 0:
        max_pre_t = pre_periods['t_stat'].abs().max()
        any_sig = (pre_periods['p_value'] < 0.05).any()

        print("  Pre-Trend Test (H0: theta(e) = 0 for e < 0):")
        print(f"    Pre-periods tested: {len(pre_periods)}")
        print(f"    Max |t-stat|: {max_pre_t:.3f}")
        print(f"    Any significant at 5%: {any_sig}")
        print(f"    Status: {'FAIL - Pre-trends detected' if any_sig else 'PASS - No pre-trends'}")
    print()

    # Average post-treatment effect
    post_periods = event_study[event_study['relative_time'] >= 0]
    if len(post_periods) > 0:
        avg_post = np.average(post_periods['theta'], weights=post_periods['n'])
        print("  Average Post-Treatment Effect:")
        print(f"    theta_post = {avg_post:.6f}")
        print(f"    Percent change: {(np.exp(avg_post) - 1) * 100:.3f}%")
    print()

    # =========================================================================
    # STEP 5: iROAS Calculation
    # =========================================================================
    print("=" * 70)
    print("[STEP 5] INCREMENTAL ROAS (iROAS)")
    print("=" * 70)
    print()

    # Treated observations
    treated_obs = panel[(panel['POST'] == 1) & (panel['IS_TREATED'] == 1)]

    total_gmv_treated = treated_obs['TOTAL_GMV'].sum()
    total_promoted_gmv = treated_obs['PROMOTED_GMV'].sum()
    total_clicks = treated_obs['CLICKS'].sum()

    # Assume average CPC (we don't have spend data without AUCTIONS_RESULTS)
    # Use promoted_gmv / clicks as a proxy for revenue per click
    if total_clicks > 0:
        revenue_per_click = total_promoted_gmv / total_clicks
        print(f"  Total Clicks (treated, post): {total_clicks:,}")
        print(f"  Total Promoted GMV: ${total_promoted_gmv:,.0f}")
        print(f"  Revenue per Click: ${revenue_per_click:.2f}")
        print()

        # ATT in levels (approximate)
        if 'TOTAL_GMV' in twfe_results:
            att_log = twfe_results['TOTAL_GMV']['beta']
            baseline_gmv = panel[panel['POST'] == 0]['TOTAL_GMV'].mean()
            att_levels = baseline_gmv * (np.exp(att_log) - 1)

            print(f"  ATT (log scale): {att_log:.6f}")
            print(f"  Baseline GMV: ${baseline_gmv:.2f}")
            print(f"  ATT (level approx): ${att_levels:.4f}")
            print()

            # iROAS = incremental GMV / spend
            # We don't have spend, so report incremental GMV per click
            print(f"  Incremental GMV per Click: ${att_levels:.4f}")
            print()

            if att_levels > 0:
                print("  Interpretation:")
                print(f"    Each click generates ~${att_levels:.4f} in incremental GMV")
                print("    (Caution: This is a rough approximation without spend data)")
    print()

    # =========================================================================
    # STEP 6: Robustness Summary
    # =========================================================================
    print("=" * 70)
    print("[STEP 6] COMPARISON: TWFE vs CS")
    print("=" * 70)
    print()

    if 'TOTAL_GMV' in twfe_results and len(event_study) > 0:
        twfe_beta = twfe_results['TOTAL_GMV']['beta']
        cs_avg = np.average(post_periods['theta'], weights=post_periods['n']) if len(post_periods) > 0 else np.nan

        print(f"  TWFE estimate (beta): {twfe_beta:.6f}")
        print(f"  CS estimate (avg post): {cs_avg:.6f}")
        print(f"  Difference: {abs(twfe_beta - cs_avg):.6f}")
        print()

        if abs(twfe_beta - cs_avg) > 0.01:
            print("  NOTE: Large difference between TWFE and CS may indicate:")
            print("    - Treatment effect heterogeneity across cohorts")
            print("    - Dynamic treatment effects (effects change over time)")
            print("    - Violation of parallel trends")
        else:
            print("  NOTE: TWFE and CS estimates are similar, suggesting:")
            print("    - Homogeneous treatment effects across cohorts")
            print("    - Stable treatment effects over time")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Panel: {len(panel):,} observations")
    print(f"  Treated vendors: {n_treated:,}")
    print(f"  Never-treated vendors: {n_never:,}")
    print(f"  Cohorts: {n_cohorts}")
    print()
    print("  Primary Estimates (TOTAL_GMV):")
    if 'TOTAL_GMV' in twfe_results:
        print(f"    TWFE beta: {twfe_results['TOTAL_GMV']['beta']:.6f} (p={twfe_results['TOTAL_GMV']['p']:.4f})")
    if len(post_periods) > 0:
        print(f"    CS avg post: {np.average(post_periods['theta'], weights=post_periods['n']):.6f}")
    print()
    print(f"Completed: {datetime.now()}")

if __name__ == '__main__':
    main()
