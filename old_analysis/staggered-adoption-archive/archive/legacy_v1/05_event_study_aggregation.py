#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Event Study Aggregation
Aggregates ATT(g,t) estimates into event study coefficients theta(e) by relative time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "05_event_study_aggregation.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================
def aggregate_to_event_time(att_df, weighting='cohort_size'):
    """
    Aggregate ATT(g, t) to event time coefficients theta(e).

    theta(e) = sum_g w_g * ATT(g, g+e)

    Parameters:
    -----------
    att_df : DataFrame
        ATT(g, t) estimates from Callaway-Sant'Anna
    weighting : str
        'cohort_size' - weight by number of treated units
        'equal' - equal weights across cohorts

    Returns:
    --------
    DataFrame with event study coefficients
    """
    results = []

    for outcome in att_df['outcome'].unique():
        subset = att_df[att_df['outcome'] == outcome].copy()

        # Group by relative time
        for rel_time in subset['relative_time'].unique():
            rel_subset = subset[subset['relative_time'] == rel_time]

            if len(rel_subset) == 0:
                continue

            if weighting == 'cohort_size':
                # Weight by number of treated units
                weights = rel_subset['n_treated'] / rel_subset['n_treated'].sum()
                theta = (rel_subset['att'] * weights).sum()

                # Weighted variance (simplified)
                var_components = (weights ** 2) * (rel_subset['se'] ** 2)
                se = np.sqrt(var_components.sum())
            else:
                # Equal weights
                theta = rel_subset['att'].mean()
                se = rel_subset['se'].mean() / np.sqrt(len(rel_subset))

            # Calculate t-stat and p-value
            if se > 0:
                t_stat = theta / se
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            else:
                t_stat = np.nan
                p_value = np.nan

            results.append({
                'outcome': outcome,
                'relative_time': rel_time,
                'theta': theta,
                'se': se,
                't_stat': t_stat,
                'p_value': p_value,
                'n_cohorts': len(rel_subset),
                'total_treated': rel_subset['n_treated'].sum(),
            })

    return pd.DataFrame(results)

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("STAGGERED ADOPTION: EVENT STUDY AGGREGATION", f)
        log("=" * 80, f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  Aggregate ATT(g, t) to event time coefficients:", f)
        log("", f)
        log("  theta(e) = sum_g w_g * ATT(g, g+e)", f)
        log("", f)
        log("  Where:", f)
        log("    e = relative time (t - g)", f)
        log("    e < 0: Pre-treatment periods (test parallel trends)", f)
        log("    e = 0: Treatment period", f)
        log("    e > 0: Post-treatment periods", f)
        log("", f)
        log("  Hypothesis Tests:", f)
        log("    H0 (pre-trends): theta(e) = 0 for all e < 0", f)
        log("    H0 (no effect): theta(e) = 0 for all e >= 0", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load ATT estimates
        # -----------------------------------------------------------------
        log("LOADING ATT(g,t) ESTIMATES", f)
        log("-" * 40, f)

        att_path = DATA_DIR / "att_gt_estimates.parquet"

        if not att_path.exists():
            log(f"  ERROR: ATT estimates not found at {att_path}", f)
            log("  Run 04_callaway_santanna.py first", f)
            return

        att_df = pd.read_parquet(att_path)

        log(f"  Loaded {len(att_df):,} ATT(g,t) estimates", f)
        log(f"  Outcomes: {att_df['outcome'].unique().tolist()}", f)
        log(f"  Relative time range: {att_df['relative_time'].min()} to {att_df['relative_time'].max()}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Aggregate to event time
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("AGGREGATING TO EVENT TIME COEFFICIENTS", f)
        log("-" * 40, f)

        event_study = aggregate_to_event_time(att_df, weighting='cohort_size')

        log(f"  Generated {len(event_study):,} event time coefficients", f)
        log("", f)

        # -----------------------------------------------------------------
        # Display results by outcome
        # -----------------------------------------------------------------
        for outcome in event_study['outcome'].unique():
            log("=" * 80, f)
            log(f"EVENT STUDY: {outcome.upper()}", f)
            log("-" * 40, f)

            subset = event_study[event_study['outcome'] == outcome].sort_values('relative_time')

            log(f"  {'e':<6} {'theta':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10} {'N_cohorts':>10}", f)
            log(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)

            for _, row in subset.iterrows():
                sig = ""
                if row['p_value'] < 0.01:
                    sig = "***"
                elif row['p_value'] < 0.05:
                    sig = "**"
                elif row['p_value'] < 0.10:
                    sig = "*"

                log(f"  {row['relative_time']:<6} {row['theta']:>12.4f} {row['se']:>12.4f} {row['t_stat']:>10.3f} {row['p_value']:>10.4f} {row['n_cohorts']:>10} {sig}", f)

            log("", f)

            # Pre-trends summary
            pre_trends = subset[subset['relative_time'] < 0]
            post_treat = subset[subset['relative_time'] >= 0]

            log("  PRE-TRENDS SUMMARY (e < 0):", f)
            if len(pre_trends) > 0:
                log(f"    Number of periods: {len(pre_trends)}", f)
                log(f"    Mean theta: {pre_trends['theta'].mean():.4f}", f)

                # Joint F-test approximation
                if len(pre_trends) > 1:
                    # Chi-squared test: sum of (theta/se)^2
                    chi2_stat = ((pre_trends['theta'] / pre_trends['se']) ** 2).sum()
                    df = len(pre_trends)
                    p_joint = 1 - stats.chi2.cdf(chi2_stat, df)
                    log(f"    Joint chi2 statistic: {chi2_stat:.4f} (df={df})", f)
                    log(f"    Joint p-value: {p_joint:.4f}", f)
                    if p_joint > 0.05:
                        log("    CONCLUSION: Cannot reject parallel trends (p > 0.05)", f)
                    else:
                        log("    WARNING: Parallel trends may be violated (p <= 0.05)", f)
            else:
                log("    No pre-treatment periods available", f)

            log("", f)

            log("  POST-TREATMENT SUMMARY (e >= 0):", f)
            if len(post_treat) > 0:
                log(f"    Number of periods: {len(post_treat)}", f)
                log(f"    Mean theta: {post_treat['theta'].mean():.4f}", f)

                # Joint significance test
                if len(post_treat) > 1:
                    chi2_stat = ((post_treat['theta'] / post_treat['se']) ** 2).sum()
                    df = len(post_treat)
                    p_joint = 1 - stats.chi2.cdf(chi2_stat, df)
                    log(f"    Joint chi2 statistic: {chi2_stat:.4f} (df={df})", f)
                    log(f"    Joint p-value: {p_joint:.4f}", f)
                    if p_joint < 0.05:
                        log("    CONCLUSION: Joint significant effect (p < 0.05)", f)
                    else:
                        log("    CONCLUSION: No significant joint effect (p >= 0.05)", f)

                # Average treatment effect on the treated (simple aggregation)
                avg_att = post_treat['theta'].mean()
                avg_se = np.sqrt((post_treat['se'] ** 2).sum()) / len(post_treat)
                t_stat = avg_att / avg_se if avg_se > 0 else np.nan
                p_val = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan

                log("", f)
                log(f"    AVERAGE POST-TREATMENT EFFECT:", f)
                log(f"      ATT: {avg_att:.4f}", f)
                log(f"      SE: {avg_se:.4f}", f)
                log(f"      t-stat: {t_stat:.4f}", f)
                log(f"      p-value: {p_val:.4f}", f)

            else:
                log("    No post-treatment periods available", f)

            log("", f)

        # -----------------------------------------------------------------
        # Save event study results
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SAVING RESULTS", f)
        log("-" * 40, f)

        output_path = DATA_DIR / "event_study_coefficients.parquet"
        event_study.to_parquet(output_path, index=False)
        log(f"  Saved event study coefficients to {output_path}", f)

        log("", f)
        log("=" * 80, f)
        log("EVENT STUDY AGGREGATION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
