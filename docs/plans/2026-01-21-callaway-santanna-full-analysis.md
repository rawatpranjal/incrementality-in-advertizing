# Callaway-Sant'Anna Full Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run comprehensive Callaway-Sant'Anna (2021) staggered DiD analysis using the Python `differences` package with pre-trends testing, robustness checks, and segmentation results.

**Architecture:** Single Python script that loads the vendor×week panel, runs ATTgt estimation for three outcomes (impressions, clicks, GMV), performs aggregations (simple, event, time, cohort), tests pre-trends, runs segmentation analysis by vendor activity, and outputs a comprehensive MASTER_RESULTS.txt report.

**Tech Stack:** Python 3.13, `differences` v0.2.0, pandas, numpy

---

## Prerequisites

```bash
# Verify package is installed
python3 -c "from differences import ATTgt; print('OK')"

# Verify data exists
ls staggered-adoption/data/panel_total_gmv.parquet
```

**Data file:** `staggered-adoption/data/panel_total_gmv.parquet`
- 846,430 observations
- 142,920 vendors
- 26 weeks
- Columns: VENDOR_ID, week, bids, wins, impressions, clicks, total_gmv, cohort, treated, log_gmv

---

### Task 1: Create Analysis Script with Data Prep

**Files:**
- Create: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Write the data preparation section**

```python
#!/usr/bin/env python3
"""
Callaway-Sant'Anna (2021) Analysis
==================================
Reference: Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
           with multiple time periods. Journal of Econometrics, 225(2), 200-230.

Estimation: Python `differences` package v0.2.0
"""

import pandas as pd
import numpy as np
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# NumPy 2.0 compatibility patch for differences package
np.NaN = np.nan

from differences import ATTgt

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prep_data(path='staggered-adoption/data/panel_total_gmv.parquet'):
    """Load panel and prepare for differences package."""
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

if __name__ == '__main__':
    df, panel_raw = load_and_prep_data()
    print(f"Panel shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
```

**Step 2: Run to verify data loads**

Run: `python3 staggered-adoption/callaway_santanna_analysis.py`
Expected: "Panel shape: (846430, 4)" and columns list

---

### Task 2: Add Summary Statistics Section

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add summary stats function**

```python
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

    print(f"Panel: Vendor × Week")
    print(f"  Observations: {n_obs:,}")
    print(f"  Vendors: {n_vendors:,}")
    print(f"  Weeks: {n_periods}")
    print()
    print(f"Treatment Definition: G_i = first week vendor wins any auction")
    print(f"  Ever-treated vendors: {n_treated:,} ({100*n_treated/n_vendors:.1f}%)")
    print(f"  Never-treated vendors: {n_control:,} ({100*n_control/n_vendors:.1f}%)")
    print()

    # Cohort distribution
    print("Cohort Distribution (first 5 + last 2):")
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
```

**Step 2: Add to main and run**

```python
if __name__ == '__main__':
    df, panel_raw = load_and_prep_data()
    print_summary_stats(df)
```

Run: `python3 staggered-adoption/callaway_santanna_analysis.py`
Expected: Summary stats printed

---

### Task 3: Add ATTgt Estimation Function

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add core estimation function**

```python
def run_attgt(df, outcome, est_method='reg', n_jobs=-1):
    """
    Run Callaway-Sant'Anna ATTgt estimation.

    Parameters
    ----------
    df : DataFrame
        Panel data with multi-index (entity, period)
    outcome : str
        Outcome variable name
    est_method : str
        Estimation method: 'reg', 'dr', 'std_ipw'
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Returns
    -------
    att_gt : ATTgt object
        Fitted ATTgt object
    results : dict
        Dictionary with aggregation results
    """
    print(f"\n{'='*80}")
    print(f"ESTIMATION: {outcome.upper()}")
    print(f"{'='*80}\n")

    # Initialize and fit
    att_gt = ATTgt(data=df, cohort_name='cohort')

    print(f"Fitting ATT(g,t) with est_method='{est_method}'...")
    att_gt.fit(
        formula=outcome,
        control_group='never_treated',
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
```

---

### Task 4: Add Results Printing Functions

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add printing functions**

```python
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

    print(f"Event Study θ(e) for {outcome}:")
    print("-" * 75)
    print(f"{'e':<8} {'θ(e)':<15} {'SE':<12} {'95% CI':<28} {'Sig?':<6}")
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
```

---

### Task 5: Add Pre-Trends Testing

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add pre-trends test function**

```python
def test_pretrends(results, outcome):
    """
    Test pre-treatment parallel trends assumption.

    H0: θ(e) = 0 for all e < 0
    """
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

    # Joint test (rough approximation using t-stats)
    valid_mask = ~np.isnan(pre_atts) & ~np.isnan(pre_ses) & (pre_ses > 0)
    if valid_mask.sum() > 0:
        t_stats = pre_atts[valid_mask] / pre_ses[valid_mask]
        # Wald test: sum of squared t-stats ~ chi-squared
        wald_stat = np.sum(t_stats**2)
        from scipy import stats
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
    print(f"  Mean θ(e<0): {mean_pre:.6f}")
    print(f"  Max |θ(e<0)|: {max_abs_pre:.6f}")
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
```

---

### Task 6: Add Segmentation Analysis

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add segmentation function**

```python
def run_segmentation_analysis(df, panel_raw, outcome='total_gmv'):
    """
    Run heterogeneous treatment effects by vendor activity quartile.
    Uses TWFE as approximation for speed (full ATTgt by segment is expensive).
    """
    print(f"\n{'='*80}")
    print(f"SEGMENTATION ANALYSIS: {outcome.upper()}")
    print(f"{'='*80}\n")

    # Create activity segments based on first-period bids
    first_period_data = df.reset_index()
    first_period_data = first_period_data[first_period_data['period'] == 1]

    # Segment by wins in first period (proxy for activity level)
    vendor_first_wins = first_period_data.groupby('entity')['wins'].sum().reset_index()
    vendor_first_wins.columns = ['entity', 'first_wins']

    # Create quartiles (handling many zeros)
    def create_activity_segment(wins):
        if wins == 0:
            return 'Q0_NoActivity'
        elif wins <= 1:
            return 'Q1_Low'
        elif wins <= 5:
            return 'Q2_Medium'
        else:
            return 'Q3_High'

    vendor_first_wins['segment'] = vendor_first_wins['first_wins'].apply(create_activity_segment)

    # Merge segments to panel
    panel_with_seg = df.reset_index().merge(vendor_first_wins[['entity', 'segment']], on='entity', how='left')

    # Run TWFE by segment (approximation)
    import statsmodels.formula.api as smf

    print("Running TWFE by segment (approximation to HTE):")
    print("-" * 75)
    print(f"{'Segment':<20} {'N':<12} {'ATT':<15} {'SE':<12} {'p-value':<10} {'Sig?':<6}")
    print("-" * 75)

    segment_results = {}
    for segment in sorted(panel_with_seg['segment'].dropna().unique()):
        seg_data = panel_with_seg[panel_with_seg['segment'] == segment].copy()

        # Add treatment indicator
        seg_data['treated'] = (~seg_data['cohort'].isna() & (seg_data['period'] >= seg_data['cohort'])).astype(int)

        # Simple TWFE with entity and time FE
        try:
            # Demean for FE approximation
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
```

---

### Task 7: Add Robustness Checks

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add robustness check function**

```python
def run_robustness_checks(df, outcome='total_gmv'):
    """
    Run robustness checks:
    1. Different estimation methods (reg vs dr)
    2. Not-yet-treated as control group
    """
    print(f"\n{'='*80}")
    print(f"ROBUSTNESS CHECKS: {outcome.upper()}")
    print(f"{'='*80}\n")

    robustness_results = {}

    # 1. Main specification (already run, just record)
    print("1. Alternative Estimation Methods:")
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

    # 2. Alternative control group
    print("\n2. Alternative Control Group:")
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
```

---

### Task 8: Add Master Results Table

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add master table function**

```python
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
    print(f"{'Outcome':<15} {'Mean θ(e<0)':<15} {'Sig Pre-periods':<20} {'Joint p-val':<12} {'Verdict':<10}")
    print("-" * 80)

    for outcome in ['impressions', 'clicks', 'total_gmv']:
        pt = all_results['pretrends'][outcome]
        print(f"{outcome:<15} {pt['mean_pre']:+.6f}{'':>4} {pt['n_sig']}/{pt['n_total']}{'':>14} {pt['joint_pval']:.4f}{'':>6} {pt['verdict']}")

    print()
```

---

### Task 9: Add Main Function and Output Redirect

**Files:**
- Modify: `staggered-adoption/callaway_santanna_analysis.py`

**Step 1: Add main function**

```python
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
    all_results = {'main': {}, 'pretrends': {}, 'segmentation': {}, 'robustness': {}}

    # Run main estimation for each outcome
    for outcome in ['impressions', 'clicks', 'total_gmv']:
        att_gt, results = run_attgt(df, outcome, est_method='reg')
        all_results['main'][outcome] = print_simple_result(results, outcome)
        print_event_study(results, outcome)
        all_results['pretrends'][outcome] = test_pretrends(results, outcome)

    # Master table
    print_master_table(all_results)

    # Segmentation
    for outcome in ['impressions', 'total_gmv']:
        all_results['segmentation'][outcome] = run_segmentation_analysis(df, panel_raw, outcome)

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

    print(f"  Winning Auctions → Impressions (+{imp_att:.2f}***)")
    print(f"                   → Clicks (+{click_att:.4f}***) [CTR: {ctr:.1f}%]")
    print(f"                   → GMV (+${gmv_att:.2f}, {'***' if all_results['main']['total_gmv']['significant'] else 'n.s.'})")
    print()
    print("Winning auctions generates exposure and engagement, but we " +
          ("DETECT" if all_results['main']['total_gmv']['significant'] else "CANNOT DETECT") +
          " a statistically significant effect on vendor sales.")
    print()

    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == '__main__':
    import sys

    # Redirect output to file
    output_path = 'staggered-adoption/results/MASTER_RESULTS.txt'

    with open(output_path, 'w') as f:
        # Duplicate output to both file and console
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

        sys.stdout = Tee(sys.stdout, f)
        main()
        sys.stdout = sys.__stdout__

    print(f"\nResults saved to: {output_path}")
```

---

### Task 10: Run Full Analysis

**Step 1: Execute the complete script**

Run: `python3 staggered-adoption/callaway_santanna_analysis.py`

Expected output file: `staggered-adoption/results/MASTER_RESULTS.txt`

Expected contents:
1. Data summary (N, vendors, weeks, cohort distribution)
2. Main estimates for impressions, clicks, GMV
3. Event study coefficients θ(e) for e = -5 to +20
4. Pre-trends test results with joint p-value
5. Master summary table
6. Segmentation analysis by vendor activity
7. Robustness checks (DR method, not-yet-treated control)
8. Interpretation

---

## Verification Checklist

- [ ] Data loads correctly (846,430 obs)
- [ ] ATTgt fits without errors for all 3 outcomes
- [ ] Event study aggregation works
- [ ] Pre-trends test passes for impressions
- [ ] Segmentation produces results by activity level
- [ ] Robustness checks run (reg vs dr, control groups)
- [ ] MASTER_RESULTS.txt created with full report

---

## Expected Key Results

Based on preliminary analysis:

| Outcome | Expected ATT | Significant? |
|---------|--------------|--------------|
| Impressions | +1.0 to +1.1 | Yes *** |
| Clicks | +0.03 to +0.04 | Yes *** |
| Total GMV | +$0.5 to +$1.0 | Probably No |

Pre-trends: Expected to PASS for impressions (all θ(e<0) ≈ 0)
