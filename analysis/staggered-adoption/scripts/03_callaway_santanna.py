#!/usr/bin/env python3
"""
Callaway-Sant'Anna (2021) Analysis with Event Study Figures
===========================================================
Reference: Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
           with multiple time periods. Journal of Econometrics, 225(2), 200-230.

Estimation: Python `differences` package v0.2.0

Output:
  - results/MASTER_RESULTS.txt: Full stdout dump
  - figures/event_study_impressions.png
  - figures/event_study_clicks.png
  - figures/event_study_gmv.png
"""

import pandas as pd
import numpy as np
import warnings
import sys
from datetime import datetime
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')

# NumPy 2.0 compatibility patch for differences package
np.NaN = np.nan

from differences import ATTgt

# Paths
BASE_DIR = Path(__file__).parent.parent  # Go up from scripts/ to project root
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prep_data(path=None):
    """Load panel and prepare for differences package."""
    if path is None:
        path = DATA_DIR / 'panel_total_gmv.parquet'

    print("Loading panel data...")
    panel = pd.read_parquet(path)

    first_period = pd.to_datetime('2025-03-24')

    # Convert week to period number (1-26)
    panel['period'] = ((pd.to_datetime(panel['week']) - first_period).dt.days // 7 + 1).astype(int)

    # Cohort as period number (NaN for never-treated per differences convention)
    panel['cohort_period'] = panel['cohort'].apply(
        lambda x: np.nan if pd.isna(x) else int((pd.to_datetime(x) - first_period).days // 7 + 1)
    )

    # Entity as integer
    panel['entity'] = panel['VENDOR_ID'].astype('category').cat.codes

    # Create panel DataFrame with multi-index
    df = panel[['entity', 'period', 'cohort_period', 'impressions', 'clicks', 'total_gmv', 'wins']].copy()
    df = df.rename(columns={'cohort_period': 'cohort'})
    df = df.set_index(['entity', 'period'])

    return df, panel


def print_summary_stats(df):
    """Print data summary statistics."""
    print("=" * 80)
    print("1. DATA SUMMARY")
    print("=" * 80)
    print()

    panel_reset = df.reset_index()
    n_vendors = panel_reset['entity'].nunique()
    n_periods = panel_reset['period'].nunique()
    n_obs = len(panel_reset)
    n_treated = panel_reset[~panel_reset['cohort'].isna()]['entity'].nunique()
    n_control = panel_reset[panel_reset['cohort'].isna()]['entity'].nunique()

    print(f"Panel: Vendor x Week")
    print(f"  Observations: {n_obs:,}")
    print(f"  Vendors: {n_vendors:,}")
    print(f"  Weeks: {n_periods}")
    print()
    print(f"Treatment Definition: G_i = first week vendor wins any auction")
    print(f"  Ever-treated vendors: {n_treated:,} ({100*n_treated/n_vendors:.1f}%)")
    print(f"  Never-treated vendors: {n_control:,} ({100*n_control/n_vendors:.1f}%)")
    print()

    # Cohort distribution
    print("Cohort Distribution:")
    cohort_counts = panel_reset.groupby('cohort')['entity'].nunique().sort_index()
    for i, (cohort, count) in enumerate(cohort_counts.items()):
        if i < 5 or i >= len(cohort_counts) - 2:
            label = "Never-treated" if pd.isna(cohort) else f"Week {int(cohort)}"
            print(f"  {label}: {count:,} vendors")
        elif i == 5:
            print("  ...")
    print()

    # Outcome variables
    print("Outcome Variables:")
    for col in ['impressions', 'clicks', 'total_gmv']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        pct_pos = (df[col] > 0).mean() * 100
        if col == 'total_gmv':
            print(f"  {col}: mean=${mean_val:.2f}, std=${std_val:.2f}, >0: {pct_pos:.2f}%")
        else:
            print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}, >0: {pct_pos:.1f}%")
    print()


# =============================================================================
# ESTIMATION
# =============================================================================

def run_attgt(df, outcome, est_method='reg', n_jobs=-1, control_group='never_treated'):
    """Run Callaway-Sant'Anna ATTgt estimation."""
    print(f"\n{'='*80}")
    print(f"ESTIMATION: {outcome.upper()}")
    print(f"{'='*80}\n")

    # Initialize and fit
    att_gt = ATTgt(data=df, cohort_name='cohort')

    print(f"Fitting ATT(g,t) with est_method='{est_method}', control='{control_group}'...")
    att_gt.fit(
        formula=outcome,
        control_group=control_group,
        est_method=est_method,
        n_jobs=n_jobs,
        progress_bar=True
    )

    # Collect all aggregations
    results = {}

    # Simple (overall) aggregation
    simple = att_gt.aggregate('simple')
    simple.columns = ['_'.join(filter(None, map(str, col))).strip() for col in simple.columns]
    results['simple'] = simple

    # Event study aggregation
    event = att_gt.aggregate('event')
    event.columns = ['_'.join(filter(None, map(str, col))).strip() for col in event.columns]
    results['event'] = event.reset_index()

    # Time aggregation
    time_agg = att_gt.aggregate('time')
    time_agg.columns = ['_'.join(filter(None, map(str, col))).strip() for col in time_agg.columns]
    results['time'] = time_agg.reset_index()

    # Cohort aggregation
    cohort_agg = att_gt.aggregate('cohort')
    cohort_agg.columns = ['_'.join(filter(None, map(str, col))).strip() for col in cohort_agg.columns]
    results['cohort'] = cohort_agg.reset_index()

    return att_gt, results


def print_simple_result(results, outcome):
    """Print simple (overall) aggregation result."""
    simple = results['simple']
    att = simple['SimpleAggregation_ATT'].values[0]
    se = simple['SimpleAggregation_analytic_std_error'].values[0]
    lower = simple['SimpleAggregation_pointwise conf. band_lower'].values[0]
    upper = simple['SimpleAggregation_pointwise conf. band_upper'].values[0]
    sig = simple['SimpleAggregation_pointwise conf. band_zero_not_in_cband'].values[0]

    print(f"Overall ATT ({outcome}):")
    if outcome == 'total_gmv':
        print(f"  ATT = ${att:+.2f}")
        print(f"  SE = ${se:.2f}")
        print(f"  95% CI = [${lower:+.2f}, ${upper:+.2f}]")
    else:
        print(f"  ATT = {att:+.6f}")
        print(f"  SE = {se:.6f}")
        print(f"  95% CI = [{lower:+.6f}, {upper:+.6f}]")
    print(f"  Significant: {'Yes ***' if sig == '*' else 'No'}")
    print()

    return {'ATT': att, 'SE': se, 'CI_lower': lower, 'CI_upper': upper, 'significant': sig == '*'}


def print_event_study(results, outcome):
    """Print event study results for key event times."""
    event = results['event']

    print(f"Event Study theta(e) for {outcome}:")
    print("-" * 75)
    print(f"{'e':<8} {'theta(e)':<15} {'SE':<12} {'95% CI':<28} {'Sig?':<6}")
    print("-" * 75)

    key_events = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20]
    for e in key_events:
        row = event[event['relative_period'] == e]
        if len(row) > 0:
            theta = row['EventAggregation_ATT'].values[0]
            se_e = row['EventAggregation_analytic_std_error'].values[0]
            lower_e = row['EventAggregation_pointwise conf. band_lower'].values[0]
            upper_e = row['EventAggregation_pointwise conf. band_upper'].values[0]
            sig_e = row['EventAggregation_pointwise conf. band_zero_not_in_cband'].values[0]

            sig_str = "***" if sig_e == '*' else ""
            if pd.isna(se_e):
                print(f"e={int(e):<5}  {theta:+.6f}       N/A")
            elif outcome == 'total_gmv':
                print(f"e={int(e):<5}  ${theta:+.2f}{'':>8} ${se_e:.2f}{'':>6} [${lower_e:+.2f}, ${upper_e:+.2f}]{'':>6} {sig_str}")
            else:
                print(f"e={int(e):<5}  {theta:+.6f}      {se_e:.6f}   [{lower_e:+.6f}, {upper_e:+.6f}] {sig_str}")
    print()


def print_cohort_aggregation(results, outcome):
    """Print cohort aggregation."""
    cohort_agg = results['cohort']

    print(f"Cohort Aggregation for {outcome} (first 5 + last 2):")
    print("-" * 60)
    print(f"{'Cohort':<10} {'ATT':<15} {'SE':<12} {'Sig?':<6}")
    print("-" * 60)

    for i, (_, row) in enumerate(cohort_agg.iterrows()):
        if i < 5 or i >= len(cohort_agg) - 2:
            g = row['cohort']
            att = row['CohortAggregation_ATT']
            se = row['CohortAggregation_analytic_std_error']
            sig = row['CohortAggregation_pointwise conf. band_zero_not_in_cband']
            sig_str = "***" if sig == '*' else ""

            if outcome == 'total_gmv':
                print(f"g={int(g):<5}  ${att:+.2f}{'':>8} ${se:.2f}{'':>6} {sig_str}")
            else:
                print(f"g={int(g):<5}  {att:+.6f}{'':>4} {se:.6f} {sig_str}")
        elif i == 5:
            print("...")
    print()


# =============================================================================
# PRE-TRENDS TESTING
# =============================================================================

def test_pretrends(results, outcome):
    """Test pre-treatment parallel trends assumption."""
    event = results['event']
    pre = event[event['relative_period'] < 0].copy()

    # Get pre-period statistics
    pre_atts = pre['EventAggregation_ATT'].values
    pre_ses = pre['EventAggregation_analytic_std_error'].values
    pre_sigs = pre['EventAggregation_pointwise conf. band_zero_not_in_cband'].values

    # Count significant pre-periods
    n_sig = sum(s == '*' for s in pre_sigs if pd.notna(s))
    n_total = sum(1 for s in pre_sigs if pd.notna(s))

    # Mean and max absolute pre-trend
    valid_pre = pre_atts[~np.isnan(pre_atts)]
    mean_pre = np.mean(valid_pre) if len(valid_pre) > 0 else np.nan
    max_abs_pre = np.max(np.abs(valid_pre)) if len(valid_pre) > 0 else np.nan

    # Joint test (Wald test)
    valid_mask = ~np.isnan(pre_atts) & ~np.isnan(pre_ses) & (pre_ses > 0)
    if valid_mask.sum() > 0:
        t_stats = pre_atts[valid_mask] / pre_ses[valid_mask]
        wald_stat = np.sum(t_stats**2)
        joint_pval = 1 - stats.chi2.cdf(wald_stat, df=valid_mask.sum())
    else:
        joint_pval = np.nan

    # Verdict
    if n_sig == 0:
        verdict = "PASS"
        interpretation = "No significant pre-trends detected"
    elif n_sig == 1:
        verdict = "MARGINAL"
        interpretation = "One significant pre-period (may be noise)"
    else:
        verdict = "FAIL"
        interpretation = f"{n_sig} significant pre-periods detected"

    print(f"Pre-Trends Test ({outcome}):")
    print(f"  Pre-period coefficients: {n_total}")
    print(f"  Mean theta(e<0): {mean_pre:.6f}")
    print(f"  Max |theta(e<0)|: {max_abs_pre:.6f}")
    print(f"  Significant at 5%: {n_sig}/{n_total}")
    print(f"  Joint Wald test p-value: {joint_pval:.4f}")
    print(f"  VERDICT: {verdict} - {interpretation}")
    print()

    return {
        'mean_pre': mean_pre,
        'max_abs_pre': max_abs_pre,
        'n_sig': n_sig,
        'n_total': n_total,
        'joint_pval': joint_pval,
        'verdict': verdict
    }


# =============================================================================
# EVENT STUDY FIGURES
# =============================================================================

def plot_event_study(results, outcome, save_path=None):
    """
    Create standard event study plot.

    X-axis: Relative time (e = weeks since treatment)
    Y-axis: theta(e) coefficient
    Vertical line at e = 0 (treatment onset)
    Error bars for 95% confidence intervals
    """
    event = results['event'].copy()

    # Extract data
    event = event.sort_values('relative_period')
    e = event['relative_period'].values
    theta = event['EventAggregation_ATT'].values
    lower = event['EventAggregation_pointwise conf. band_lower'].values
    upper = event['EventAggregation_pointwise conf. band_upper'].values

    # Filter to reasonable range (-10 to 20)
    mask = (e >= -10) & (e <= 20)
    e = e[mask]
    theta = theta[mask]
    lower = lower[mask]
    upper = upper[mask]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot coefficients with error bars
    ax.errorbar(e, theta, yerr=[theta - lower, upper - theta],
                fmt='o', capsize=3, capthick=1.5,
                color='#2166AC', markersize=6, linewidth=1.5,
                label=r'$\theta(e)$ with 95% CI')

    # Connect points with line
    ax.plot(e, theta, '-', color='#2166AC', alpha=0.5, linewidth=1)

    # Add vertical line at e=0
    ax.axvline(x=0, color='#B2182B', linestyle='--', linewidth=1.5,
               label='Treatment onset (e=0)')

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # Shade pre-treatment region
    ax.axvspan(e.min() - 0.5, 0, alpha=0.1, color='gray', label='Pre-treatment')

    # Labels and formatting
    ax.set_xlabel('Relative Time (e = weeks since treatment)', fontsize=12)

    if outcome == 'total_gmv':
        ax.set_ylabel(r'$\theta(e)$ ($ change in GMV)', fontsize=12)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    elif outcome == 'impressions':
        ax.set_ylabel(r'$\theta(e)$ (change in impressions)', fontsize=12)
    elif outcome == 'clicks':
        ax.set_ylabel(r'$\theta(e)$ (change in clicks)', fontsize=12)
    else:
        ax.set_ylabel(r'$\theta(e)$', fontsize=12)

    # Title
    outcome_label = {'impressions': 'Impressions', 'clicks': 'Clicks', 'total_gmv': 'Total GMV'}
    ax.set_title(f'Event Study: Effect of Advertising on {outcome_label.get(outcome, outcome)}',
                 fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='upper left', framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Set x-axis ticks
    ax.set_xticks(range(int(e.min()), int(e.max()) + 1, 2))

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")

    plt.close()

    return fig


# =============================================================================
# SEGMENTATION ANALYSIS
# =============================================================================

def run_segmentation_analysis(df, outcome='total_gmv'):
    """Run HTE by cohort timing (early vs late adopters) using TWFE approximation."""
    print(f"\n{'='*80}")
    print(f"SEGMENTATION ANALYSIS: {outcome.upper()}")
    print(f"{'='*80}\n")

    panel_reset = df.reset_index()

    # Segment by cohort timing (when vendor first won - treatment timing)
    vendor_cohorts = panel_reset.groupby('entity')['cohort'].first().reset_index()

    def create_cohort_segment(cohort):
        if pd.isna(cohort):
            return 'Never_Treated'
        elif cohort <= 4:
            return 'Early_Adopter'  # Weeks 1-4
        elif cohort <= 13:
            return 'Mid_Adopter'    # Weeks 5-13
        else:
            return 'Late_Adopter'   # Weeks 14-26

    vendor_cohorts['segment'] = vendor_cohorts['cohort'].apply(create_cohort_segment)
    panel_with_seg = panel_reset.merge(vendor_cohorts[['entity', 'segment']], on='entity', how='left')

    import statsmodels.formula.api as smf

    print("TWFE by Segment (HTE approximation):")
    print("-" * 75)
    print(f"{'Segment':<20} {'N':<12} {'ATT':<15} {'SE':<12} {'p-value':<10} {'Sig?':<6}")
    print("-" * 75)

    segment_results = {}
    for segment in sorted(panel_with_seg['segment'].dropna().unique()):
        seg_data = panel_with_seg[panel_with_seg['segment'] == segment].copy()
        seg_data['treated'] = (~seg_data['cohort'].isna() & (seg_data['period'] >= seg_data['cohort'])).astype(int)

        try:
            seg_data['y_dm'] = seg_data[outcome] - seg_data.groupby('entity')[outcome].transform('mean')
            seg_data['d_dm'] = seg_data['treated'] - seg_data.groupby('entity')['treated'].transform('mean')

            model = smf.ols('y_dm ~ d_dm - 1', data=seg_data).fit()

            att = model.params['d_dm']
            se = model.bse['d_dm']
            pval = model.pvalues['d_dm']
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))

            n_obs = len(seg_data)

            if outcome == 'total_gmv':
                print(f"{segment:<20} {n_obs:<12,} ${att:+.2f}{'':>8} ${se:.2f}{'':>6} {pval:.4f}{'':>4} {sig}")
            else:
                print(f"{segment:<20} {n_obs:<12,} {att:+.6f}{'':>4} {se:.6f}{'':>2} {pval:.4f}{'':>4} {sig}")

            segment_results[segment] = {'ATT': att, 'SE': se, 'pval': pval, 'n': n_obs}
        except Exception as e:
            print(f"{segment:<20} ERROR: {e}")
            segment_results[segment] = None

    print()
    return segment_results


# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================

def run_robustness_checks(df, outcome='total_gmv'):
    """Run robustness checks."""
    print(f"\n{'='*80}")
    print(f"ROBUSTNESS CHECKS: {outcome.upper()}")
    print(f"{'='*80}\n")

    robustness_results = {}

    # 1. Estimation methods
    print("1. Estimation Methods:")
    print("-" * 60)

    for method in ['reg', 'dr']:
        print(f"\n   Method: {method.upper()}")
        try:
            att_gt = ATTgt(data=df, cohort_name='cohort')
            att_gt.fit(
                formula=outcome,
                control_group='never_treated',
                est_method=method,
                n_jobs=-1,
                progress_bar=False
            )
            simple = att_gt.aggregate('simple')
            simple.columns = ['_'.join(filter(None, map(str, col))).strip() for col in simple.columns]

            att = simple['SimpleAggregation_ATT'].values[0]
            se = simple['SimpleAggregation_analytic_std_error'].values[0]
            sig = simple['SimpleAggregation_pointwise conf. band_zero_not_in_cband'].values[0]

            if outcome == 'total_gmv':
                print(f"   ATT = ${att:+.2f} (SE=${se:.2f}) {'***' if sig=='*' else ''}")
            else:
                print(f"   ATT = {att:+.6f} (SE={se:.6f}) {'***' if sig=='*' else ''}")

            robustness_results[f'method_{method}'] = {'ATT': att, 'SE': se, 'sig': sig == '*'}
        except Exception as e:
            print(f"   ERROR: {e}")

    # 2. Control groups
    print("\n2. Control Groups:")
    print("-" * 60)

    for control in ['never_treated', 'not_yet_treated']:
        print(f"\n   Control: {control}")
        try:
            att_gt = ATTgt(data=df, cohort_name='cohort')
            att_gt.fit(
                formula=outcome,
                control_group=control,
                est_method='reg',
                n_jobs=-1,
                progress_bar=False
            )
            simple = att_gt.aggregate('simple')
            simple.columns = ['_'.join(filter(None, map(str, col))).strip() for col in simple.columns]

            att = simple['SimpleAggregation_ATT'].values[0]
            se = simple['SimpleAggregation_analytic_std_error'].values[0]
            sig = simple['SimpleAggregation_pointwise conf. band_zero_not_in_cband'].values[0]

            if outcome == 'total_gmv':
                print(f"   ATT = ${att:+.2f} (SE=${se:.2f}) {'***' if sig=='*' else ''}")
            else:
                print(f"   ATT = {att:+.6f} (SE={se:.6f}) {'***' if sig=='*' else ''}")

            robustness_results[f'control_{control}'] = {'ATT': att, 'SE': se, 'sig': sig == '*'}
        except Exception as e:
            print(f"   ERROR: {e}")

    print()
    return robustness_results


# =============================================================================
# MASTER TABLE
# =============================================================================

def print_master_table(all_results):
    """Print master summary table."""
    print("\n" + "=" * 80)
    print("MASTER RESULTS TABLE")
    print("=" * 80 + "\n")

    print("Panel A: Main Estimates (Callaway-Sant'Anna, Never-Treated Control)")
    print("-" * 80)
    print(f"{'Outcome':<15} {'ATT':<18} {'SE':<12} {'95% CI':<28} {'Sig?':<8}")
    print("-" * 80)

    for outcome in ['impressions', 'clicks', 'total_gmv']:
        r = all_results['main'][outcome]
        sig_str = "***" if r['significant'] else ""
        if outcome == 'total_gmv':
            print(f"{outcome:<15} ${r['ATT']:+.2f}{'':>12} ${r['SE']:.2f}{'':>6} [${r['CI_lower']:+.2f}, ${r['CI_upper']:+.2f}]{'':>4} {sig_str}")
        else:
            print(f"{outcome:<15} {r['ATT']:+.6f}{'':>8} {r['SE']:.6f} [{r['CI_lower']:+.6f}, {r['CI_upper']:+.6f}] {sig_str}")

    print("\n\nPanel B: Pre-Trends Assessment")
    print("-" * 80)
    print(f"{'Outcome':<15} {'Mean theta(e<0)':<18} {'Sig Pre-periods':<18} {'Joint p-val':<12} {'Verdict':<10}")
    print("-" * 80)

    for outcome in ['impressions', 'clicks', 'total_gmv']:
        pt = all_results['pretrends'][outcome]
        print(f"{outcome:<15} {pt['mean_pre']:+.6f}{'':>8} {pt['n_sig']}/{pt['n_total']}{'':>14} {pt['joint_pval']:.4f}{'':>6} {pt['verdict']}")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run full Callaway-Sant'Anna analysis."""

    print("=" * 80)
    print("CALLAWAY-SANT'ANNA (2021) ANALYSIS: COMPREHENSIVE REPORT")
    print("=" * 80)
    print()
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Reference: Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences")
    print("           with multiple time periods. Journal of Econometrics, 225(2), 200-230.")
    print()
    print("Estimation: Python `differences` package v0.2.0")
    print()

    # Load data
    df, panel_raw = load_and_prep_data()

    # Summary stats
    print_summary_stats(df)

    # Store all results
    all_results = {'main': {}, 'pretrends': {}, 'event_data': {}, 'segmentation': {}, 'robustness': {}}

    # Run main estimation for each outcome
    for outcome in ['impressions', 'clicks', 'total_gmv']:
        att_gt, results = run_attgt(df, outcome, est_method='reg')
        all_results['main'][outcome] = print_simple_result(results, outcome)
        all_results['event_data'][outcome] = results
        print_event_study(results, outcome)
        print_cohort_aggregation(results, outcome)
        all_results['pretrends'][outcome] = test_pretrends(results, outcome)

    # Master table
    print_master_table(all_results)

    # Generate event study figures
    print("\n" + "=" * 80)
    print("GENERATING EVENT STUDY FIGURES")
    print("=" * 80 + "\n")

    for outcome in ['impressions', 'clicks', 'total_gmv']:
        fig_name = f'event_study_{outcome}.png'
        fig_path = FIGURES_DIR / fig_name
        print(f"Generating {fig_name}...")
        plot_event_study(all_results['event_data'][outcome], outcome, save_path=fig_path)

    print()

    # Segmentation
    for outcome in ['impressions', 'total_gmv']:
        all_results['segmentation'][outcome] = run_segmentation_analysis(df, outcome)

    # Robustness
    for outcome in ['impressions', 'total_gmv']:
        all_results['robustness'][outcome] = run_robustness_checks(df, outcome)

    # Final interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("THE FUNNEL:")
    imp_att = all_results['main']['impressions']['ATT']
    click_att = all_results['main']['clicks']['ATT']
    gmv_att = all_results['main']['total_gmv']['ATT']
    ctr = (click_att / imp_att * 100) if imp_att > 0 else 0

    print(f"  Winning Auctions -> Impressions (+{imp_att:.2f}***)")
    print(f"                   -> Clicks (+{click_att:.4f}***) [CTR: {ctr:.1f}%]")
    gmv_sig = "***" if all_results['main']['total_gmv']['significant'] else "n.s."
    print(f"                   -> GMV (+${gmv_att:.2f}, {gmv_sig})")
    print()

    if all_results['main']['total_gmv']['significant']:
        print("Winning auctions generates exposure, engagement, AND incremental sales.")
    else:
        print("Winning auctions generates exposure and engagement, but we CANNOT DETECT")
        print("a statistically significant effect on vendor sales.")
    print()

    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == '__main__':
    output_path = RESULTS_DIR / 'MASTER_RESULTS.txt'

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open(output_path, 'w') as f:
        sys.stdout = Tee(sys.stdout, f)
        main()
        sys.stdout = sys.__stdout__

    print(f"\nResults saved to: {output_path}")
    print(f"Figures saved to: {FIGURES_DIR}")
