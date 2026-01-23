#!/usr/bin/env python3
"""
EDA Q5: Is there an "Ashenfelter's Dip"?

Plot the average Total_GMV for treated vendors in the 4 weeks leading up to their
adoption date. Do vendors turn on ads because their sales are crashing (dip)?
If so, Post - Pre comparisons will bias the treatment effect upwards (mean reversion).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # staggered-adoption/
EDA_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "05_ashenfelter_dip.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# SNOWFLAKE CONNECTION
# =============================================================================
def get_snowflake_connection():
    """Establish Snowflake connection using environment variables."""
    try:
        import snowflake.connector
        load_dotenv()

        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            database='INCREMENTALITY',
            schema='INCREMENTALITY_RESEARCH'
        )
        return conn
    except Exception as e:
        return None

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("EDA Q5: ASHENFELTER'S DIP ANALYSIS", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  Do vendors turn on ads because their sales are declining?", f)
        log("  If so, post-treatment gains may be mean reversion, not causal effect.", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. For treated vendors, compute average GMV at relative time e=-4,-3,-2,-1,0,+1,+2", f)
        log("  2. Test for declining pre-trend (Ashenfelter's dip)", f)
        log("  3. Compare pre-treatment trend in treated vs never-treated", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel data
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_cohorts.parquet"
        if not panel_path.exists():
            log(f"  [ERROR] File not found: {panel_path}", f)
            return

        panel = pd.read_parquet(panel_path)
        log(f"  Loaded {len(panel):,} vendor-week observations", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log("", f)

        # Note: Using promoted_gmv as proxy since total_gmv not available
        log("  [NOTE] Using promoted_gmv as outcome (total_gmv not in panel)", f)
        log("  This may underestimate pre-treatment activity for never-promoted vendors.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Compute event-time averages
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("EVENT-TIME ANALYSIS FOR TREATED VENDORS", f)
        log("-" * 40, f)
        log("", f)

        # Filter to treated vendors
        if 'relative_week' in panel.columns:
            treated_panel = panel[panel['treated'] == 1].copy()
        else:
            # Compute relative week
            if 'cohort_week' in panel.columns:
                panel['relative_week'] = (pd.to_datetime(panel['week']) -
                                          pd.to_datetime(panel['cohort_week'])).dt.days // 7
            treated_panel = panel[panel['cohort_week'].notna()].copy()

        log(f"  Treated vendor-weeks: {len(treated_panel):,}", f)
        log(f"  Unique treated vendors: {treated_panel['VENDOR_ID'].nunique():,}", f)
        log("", f)

        # Average GMV by relative week
        event_time_gmv = treated_panel.groupby('relative_week').agg({
            'promoted_gmv': ['mean', 'std', 'count'],
            'impressions': 'mean',
            'clicks': 'mean',
            'wins': 'mean'
        }).reset_index()

        event_time_gmv.columns = ['relative_week', 'mean_gmv', 'std_gmv', 'n_obs',
                                   'mean_impressions', 'mean_clicks', 'mean_wins']

        # Focus on event window [-6, +6]
        event_window = event_time_gmv[
            (event_time_gmv['relative_week'] >= -6) &
            (event_time_gmv['relative_week'] <= 6)
        ].sort_values('relative_week')

        log("  EVENT-TIME AVERAGE PROMOTED GMV:", f)
        log(f"  {'e (rel week)':>12} {'Mean GMV':>12} {'Std GMV':>12} {'N':>10} {'Impressions':>12}", f)
        log(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*12}", f)

        for _, row in event_window.iterrows():
            e = int(row['relative_week'])
            marker = "<<<" if e == 0 else ""
            log(f"  {e:>12} ${row['mean_gmv']:>10.2f} ${row['std_gmv']:>10.2f} {row['n_obs']:>10,} {row['mean_impressions']:>11.1f} {marker}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Test for pre-treatment trend
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("PRE-TREATMENT TREND TEST", f)
        log("-" * 40, f)
        log("", f)

        pre_treatment = event_window[event_window['relative_week'] < 0].copy()

        if len(pre_treatment) >= 2:
            # Simple linear trend
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                pre_treatment['relative_week'],
                pre_treatment['mean_gmv']
            )

            log(f"  Linear trend in pre-treatment GMV (e < 0):", f)
            log(f"    Slope: {slope:.4f} (change per week)", f)
            log(f"    R-squared: {r_value**2:.4f}", f)
            log(f"    P-value: {p_value:.4f}", f)
            log("", f)

            if slope < 0 and p_value < 0.05:
                log("  [WARNING] Significant NEGATIVE pre-trend detected.", f)
                log("  This suggests Ashenfelter's dip - vendors may adopt ads", f)
                log("  in response to declining sales. Post-treatment gains", f)
                log("  could be mean reversion rather than causal effect.", f)
            elif slope > 0 and p_value < 0.05:
                log("  [NOTE] Significant POSITIVE pre-trend detected.", f)
                log("  Vendors may be adopting ads while growing.", f)
                log("  This could bias treatment effects downward.", f)
            else:
                log("  [OK] No significant pre-trend detected.", f)
                log("  Pre-treatment GMV appears stable.", f)
        else:
            log("  [WARNING] Insufficient pre-treatment periods for trend test.", f)

        log("", f)

        # -----------------------------------------------------------------
        # Compare treated vs never-treated pre-trends
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TREATED VS NEVER-TREATED COMPARISON", f)
        log("-" * 40, f)
        log("", f)

        never_treated = panel[panel['treated'] == 0].copy() if 'treated' in panel.columns else \
                        panel[panel['cohort_week'].isna()].copy()

        if len(never_treated) > 0:
            log(f"  Never-treated vendor-weeks: {len(never_treated):,}", f)
            log(f"  Never-treated vendors: {never_treated['VENDOR_ID'].nunique():,}", f)
            log("", f)

            # Average by calendar week for never-treated
            nt_by_week = never_treated.groupby('week').agg({
                'promoted_gmv': 'mean',
                'impressions': 'mean'
            }).reset_index()
            nt_by_week.columns = ['week', 'nt_mean_gmv', 'nt_mean_impressions']

            log("  NEVER-TREATED AVERAGE BY CALENDAR WEEK (first 10 weeks):", f)
            log(f"  {'Week':<12} {'Mean GMV':>12} {'Impressions':>12}", f)
            log(f"  {'-'*12} {'-'*12} {'-'*12}", f)
            for _, row in nt_by_week.head(10).iterrows():
                log(f"  {str(row['week'])[:10]:<12} ${row['nt_mean_gmv']:>10.2f} {row['nt_mean_impressions']:>11.1f}", f)

            log("", f)

            # Overall comparison
            treated_pre_mean = pre_treatment['mean_gmv'].mean() if len(pre_treatment) > 0 else 0
            never_treated_mean = never_treated['promoted_gmv'].mean()

            log(f"  AVERAGE GMV COMPARISON:", f)
            log(f"    Treated vendors (pre-treatment): ${treated_pre_mean:.2f}", f)
            log(f"    Never-treated vendors (all time): ${never_treated_mean:.2f}", f)
            log("", f)

        else:
            log("  [WARNING] No never-treated vendors found for comparison.", f)
            log("", f)

        # -----------------------------------------------------------------
        # Activity ramp-up around adoption
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ACTIVITY RAMP-UP AROUND ADOPTION", f)
        log("-" * 40, f)
        log("", f)

        log("  Checking if vendors gradually increase activity before formal adoption...", f)
        log("", f)

        # Wins/impressions by event time
        log("  EVENT-TIME AVERAGE WINS AND IMPRESSIONS:", f)
        log(f"  {'e (rel week)':>12} {'Mean Wins':>12} {'Mean Imps':>12} {'Mean Clicks':>12}", f)
        log(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

        for _, row in event_window.iterrows():
            e = int(row['relative_week'])
            log(f"  {e:>12} {row['mean_wins']:>11.1f} {row['mean_impressions']:>11.1f} {row['mean_clicks']:>11.1f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        log("  KEY FINDINGS:", f)
        log("", f)

        # Check for dip pattern
        if len(pre_treatment) >= 4:
            e_minus_4 = pre_treatment[pre_treatment['relative_week'] == -4]['mean_gmv'].values
            e_minus_1 = pre_treatment[pre_treatment['relative_week'] == -1]['mean_gmv'].values

            if len(e_minus_4) > 0 and len(e_minus_1) > 0:
                change_pct = (e_minus_1[0] - e_minus_4[0]) / e_minus_4[0] * 100 if e_minus_4[0] != 0 else 0
                log(f"    GMV change from e=-4 to e=-1: {change_pct:+.1f}%", f)

                if change_pct < -10:
                    log("    --> Possible Ashenfelter's dip detected", f)
                elif change_pct > 10:
                    log("    --> Pre-treatment growth observed", f)
                else:
                    log("    --> Relatively stable pre-treatment", f)

        log("", f)
        log("  RECOMMENDATIONS:", f)
        log("    1. If dip detected: Use matching on pre-treatment trends", f)
        log("    2. Include pre-treatment outcome as covariate in doubly-robust", f)
        log("    3. Test sensitivity with different event windows", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
