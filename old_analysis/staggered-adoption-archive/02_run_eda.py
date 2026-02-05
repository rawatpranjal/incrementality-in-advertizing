#!/usr/bin/env python3
"""
02_run_eda.py - 10-Point EDA Diagnostics

Validates the vendor-week panel for Callaway-Sant'Anna DiD assumptions.
Checks parallel trends, treatment dynamics, balance, and concentration.

Input: data/vendor_weekly_panel.parquet
Output: results/02_run_eda.txt
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

LOG_FILE = RESULTS_DIR / '02_run_eda.txt'

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

def gini_coefficient(x):
    """Calculate Gini coefficient."""
    x = np.array(x, dtype=float)
    x = x[x > 0]  # Exclude zeros for meaningful Gini
    if len(x) == 0:
        return 0
    x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1) * x)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

def main():
    print("=" * 70)
    print("02_RUN_EDA.PY - 10-Point EDA Diagnostics")
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
    print(f"  Columns: {list(panel.columns)}")
    print(f"  Vendors: {panel['VENDOR_ID'].nunique():,}")
    print(f"  Weeks: {panel['WEEK'].nunique()}")
    print()

    # Ensure WEEK is datetime
    panel['WEEK'] = pd.to_datetime(panel['WEEK'])

    # =========================================================================
    # Q1: Bridge Integrity (Orphan Rate)
    # =========================================================================
    print("=" * 70)
    print("Q1: BRIDGE INTEGRITY (Orphan Rate)")
    print("=" * 70)
    print()
    print("  NOTE: Orphan rate was computed during data pull.")
    print("  Known result: ~82% of GMV is orphan (organic purchases not in catalog).")
    print("  This is expected - CATALOG only contains promoted products.")
    print("  For DiD analysis, we focus on vendors with catalog presence.")
    print()
    print("  Status: ACKNOWLEDGED (by design)")
    print()

    # =========================================================================
    # Q2: Panel Balance
    # =========================================================================
    print("=" * 70)
    print("Q2: PANEL BALANCE")
    print("=" * 70)
    print()

    n_vendors = panel['VENDOR_ID'].nunique()
    n_weeks = panel['WEEK'].nunique()
    expected_rows = n_vendors * n_weeks
    actual_rows = len(panel)
    balance_ratio = actual_rows / expected_rows

    print(f"  Vendors: {n_vendors:,}")
    print(f"  Weeks: {n_weeks}")
    print(f"  Expected rows (balanced): {expected_rows:,}")
    print(f"  Actual rows: {actual_rows:,}")
    print(f"  Balance ratio: {balance_ratio:.4f}")
    print(f"  Status: {'BALANCED' if balance_ratio > 0.99 else 'UNBALANCED'}")
    print()

    # Rows per vendor
    rows_per_vendor = panel.groupby('VENDOR_ID').size()
    print(f"  Rows per vendor: min={rows_per_vendor.min()}, max={rows_per_vendor.max()}, median={rows_per_vendor.median():.0f}")
    print()

    # =========================================================================
    # Q3: Treatment Absorbing (Flicker Rate)
    # =========================================================================
    print("=" * 70)
    print("Q3: TREATMENT ABSORBING (Flicker Rate)")
    print("=" * 70)
    print()

    panel_sorted = panel.sort_values(['VENDOR_ID', 'WEEK'])
    panel_sorted['prev_treated'] = panel_sorted.groupby('VENDOR_ID')['HAS_CLICKS'].shift(1)

    transitions = panel_sorted.dropna(subset=['prev_treated'])
    on_to_off = ((transitions['prev_treated'] == 1) & (transitions['HAS_CLICKS'] == 0)).sum()
    on_to_on = ((transitions['prev_treated'] == 1) & (transitions['HAS_CLICKS'] == 1)).sum()
    off_to_on = ((transitions['prev_treated'] == 0) & (transitions['HAS_CLICKS'] == 1)).sum()
    off_to_off = ((transitions['prev_treated'] == 0) & (transitions['HAS_CLICKS'] == 0)).sum()

    flicker_rate = on_to_off / (on_to_off + on_to_on) if (on_to_off + on_to_on) > 0 else 0

    print("  Transition Matrix:")
    print(f"    OFF -> OFF: {off_to_off:,}")
    print(f"    OFF -> ON:  {off_to_on:,}")
    print(f"    ON  -> OFF: {on_to_off:,}")
    print(f"    ON  -> ON:  {on_to_on:,}")
    print()
    print(f"  Flicker Rate (ON->OFF / (ON->ON + ON->OFF)): {flicker_rate:.2%}")
    print(f"  Status: {'OK (<20%)' if flicker_rate < 0.2 else 'WARNING (>20%)'}")
    print()
    print("  Interpretation:")
    print("    Flicker <10%: Treatment is absorbing (good for DiD)")
    print("    Flicker 10-30%: Moderate switching, use with caution")
    print("    Flicker >30%: Treatment is not absorbing, consider alternative definition")
    print()

    # =========================================================================
    # Q4: Adoption Velocity
    # =========================================================================
    print("=" * 70)
    print("Q4: ADOPTION VELOCITY")
    print("=" * 70)
    print()

    # First week with clicks > 0 per vendor
    first_click = panel[panel['HAS_CLICKS'] == 1].groupby('VENDOR_ID')['WEEK'].min().reset_index()
    first_click.columns = ['VENDOR_ID', 'COHORT_WEEK']

    n_treated = len(first_click)
    n_never_treated = n_vendors - n_treated

    print(f"  Treated vendors: {n_treated:,} ({n_treated/n_vendors*100:.1f}%)")
    print(f"  Never-treated vendors: {n_never_treated:,} ({n_never_treated/n_vendors*100:.1f}%)")
    print()

    if n_treated > 0:
        adoption_by_week = first_click.groupby('COHORT_WEEK').size().reset_index(name='N_NEW')
        adoption_by_week['CUMULATIVE'] = adoption_by_week['N_NEW'].cumsum()
        adoption_by_week['PCT_NEW'] = adoption_by_week['N_NEW'] / adoption_by_week['N_NEW'].sum() * 100
        adoption_by_week['PCT_CUMULATIVE'] = adoption_by_week['CUMULATIVE'] / n_treated * 100

        print("  Adoption by Cohort Week:")
        print("  " + "-" * 60)
        print(f"  {'WEEK':<12} {'NEW':>8} {'CUMUL':>8} {'%NEW':>8} {'%CUM':>8}")
        print("  " + "-" * 60)
        for _, row in adoption_by_week.iterrows():
            print(f"  {str(row['COHORT_WEEK'].date()):<12} {row['N_NEW']:>8,} {row['CUMULATIVE']:>8,} {row['PCT_NEW']:>7.1f}% {row['PCT_CUMULATIVE']:>7.1f}%")
        print("  " + "-" * 60)
        print()

        # Concentration metrics
        max_cohort_pct = adoption_by_week['PCT_NEW'].max()
        week1_pct = adoption_by_week['PCT_NEW'].iloc[0] if len(adoption_by_week) > 0 else 0

        print(f"  Week 1 adoption: {week1_pct:.1f}%")
        print(f"  Largest cohort: {max_cohort_pct:.1f}%")
        print(f"  Number of cohorts: {len(adoption_by_week)}")
        print()

        print("  Status: ", end="")
        if max_cohort_pct > 50:
            print("WARNING - Single cohort dominates (>50%)")
        elif max_cohort_pct > 30:
            print("CAUTION - Large cohort concentration")
        else:
            print("OK - Good cohort spread")
    print()

    # =========================================================================
    # Q5: Ashenfelter's Dip
    # =========================================================================
    print("=" * 70)
    print("Q5: ASHENFELTER'S DIP (Pre-Treatment GMV Trajectory)")
    print("=" * 70)
    print()

    if n_treated > 0:
        # Merge cohort week to panel
        panel_cohort = panel.merge(first_click, on='VENDOR_ID', how='left')
        panel_cohort['RELATIVE_WEEK'] = (
            (pd.to_datetime(panel_cohort['WEEK']) -
             pd.to_datetime(panel_cohort['COHORT_WEEK'])).dt.days // 7
        )

        # Event study for treated vendors
        treated = panel_cohort[panel_cohort['COHORT_WEEK'].notna()]
        event_study = treated.groupby('RELATIVE_WEEK').agg({
            'TOTAL_GMV': ['mean', 'std', 'count'],
            'PROMOTED_GMV': 'mean',
            'CLICKS': 'mean'
        }).reset_index()
        event_study.columns = ['RELATIVE_WEEK', 'GMV_MEAN', 'GMV_STD', 'N', 'PROMOTED_MEAN', 'CLICKS_MEAN']

        print("  Event Study (Mean TOTAL_GMV by Relative Week):")
        print("  " + "-" * 70)
        print(f"  {'WEEK':>6} {'GMV_MEAN':>12} {'GMV_STD':>12} {'N':>10} {'PROMOTED':>12}")
        print("  " + "-" * 70)
        for _, row in event_study[(event_study['RELATIVE_WEEK'] >= -8) & (event_study['RELATIVE_WEEK'] <= 8)].iterrows():
            marker = "<<<" if row['RELATIVE_WEEK'] == 0 else ""
            print(f"  {row['RELATIVE_WEEK']:>6} {row['GMV_MEAN']:>12.2f} {row['GMV_STD']:>12.2f} {row['N']:>10,} {row['PROMOTED_MEAN']:>12.2f} {marker}")
        print("  " + "-" * 70)
        print()

        # Check for dip (compare t-1, t-2 to t-3, t-4)
        pre_close = event_study[(event_study['RELATIVE_WEEK'] >= -2) & (event_study['RELATIVE_WEEK'] < 0)]['GMV_MEAN'].mean()
        pre_far = event_study[(event_study['RELATIVE_WEEK'] >= -5) & (event_study['RELATIVE_WEEK'] < -2)]['GMV_MEAN'].mean()
        post = event_study[(event_study['RELATIVE_WEEK'] > 0) & (event_study['RELATIVE_WEEK'] <= 3)]['GMV_MEAN'].mean()

        print(f"  Pre-treatment (t-5 to t-2): ${pre_far:.2f}")
        print(f"  Pre-treatment (t-1, t-2): ${pre_close:.2f}")
        print(f"  Post-treatment (t+1 to t+3): ${post:.2f}")
        print()

        if pre_far > 0:
            dip_pct = (pre_far - pre_close) / pre_far * 100
            print(f"  Dip magnitude: {dip_pct:.1f}%")
            print(f"  Status: {'WARNING - Ashenfelter Dip detected' if dip_pct > 10 else 'OK - No significant dip'}")
        else:
            print("  Status: INSUFFICIENT DATA")
    print()

    # =========================================================================
    # Q6 & Q7: Mechanism Checks (Auction-level)
    # =========================================================================
    print("=" * 70)
    print("Q6: CPC VERIFICATION")
    print("=" * 70)
    print()
    print("  NOTE: Requires AUCTIONS_RESULTS table (62B rows, too slow to query).")
    print("  Status: SKIPPED")
    print()

    print("=" * 70)
    print("Q7: RANK DETERMINISM")
    print("=" * 70)
    print()
    print("  NOTE: Requires AUCTIONS_RESULTS table (62B rows, too slow to query).")
    print("  Status: SKIPPED")
    print()

    # =========================================================================
    # Q8: Zero-Inflation
    # =========================================================================
    print("=" * 70)
    print("Q8: ZERO-INFLATION (True Population)")
    print("=" * 70)
    print()

    print("  Percentage of zeros by column:")
    print("  " + "-" * 40)
    for col in ['TOTAL_GMV', 'PROMOTED_GMV', 'ORGANIC_GMV', 'CLICKS', 'IMPRESSIONS']:
        zero_pct = (panel[col] == 0).mean() * 100
        print(f"    {col:<15}: {zero_pct:>6.2f}%")
    print("  " + "-" * 40)
    print()

    # Zero breakdown
    all_zero = (panel['TOTAL_GMV'] == 0) & (panel['CLICKS'] == 0)
    gmv_only = (panel['TOTAL_GMV'] > 0) & (panel['CLICKS'] == 0)
    clicks_only = (panel['TOTAL_GMV'] == 0) & (panel['CLICKS'] > 0)
    both = (panel['TOTAL_GMV'] > 0) & (panel['CLICKS'] > 0)

    print("  Joint Distribution:")
    print(f"    No GMV, No Clicks: {all_zero.mean()*100:.2f}%")
    print(f"    GMV only (organic): {gmv_only.mean()*100:.2f}%")
    print(f"    Clicks only (no purchase): {clicks_only.mean()*100:.2f}%")
    print(f"    Both GMV and Clicks: {both.mean()*100:.2f}%")
    print()

    print("  Implication:")
    if (panel['TOTAL_GMV'] == 0).mean() > 0.9:
        print("    CRITICAL: >90% zeros in outcome. Consider Hurdle/ZI models.")
    elif (panel['TOTAL_GMV'] == 0).mean() > 0.7:
        print("    WARNING: High zero-inflation. DiD may have power issues.")
    else:
        print("    OK: Manageable zero-inflation for DiD.")
    print()

    # =========================================================================
    # Q9: Whale Concentration
    # =========================================================================
    print("=" * 70)
    print("Q9: WHALE CONCENTRATION")
    print("=" * 70)
    print()

    # Total GMV per vendor
    vendor_totals = panel.groupby('VENDOR_ID')['TOTAL_GMV'].sum().sort_values(ascending=False)
    total_gmv = vendor_totals.sum()
    n_vendors_gmv = len(vendor_totals)

    if total_gmv > 0:
        top_1_pct = vendor_totals.head(max(1, int(n_vendors_gmv * 0.01))).sum() / total_gmv * 100
        top_5_pct = vendor_totals.head(max(1, int(n_vendors_gmv * 0.05))).sum() / total_gmv * 100
        top_10_pct = vendor_totals.head(max(1, int(n_vendors_gmv * 0.10))).sum() / total_gmv * 100
        gini = gini_coefficient(vendor_totals.values)

        print(f"  Total GMV: ${total_gmv:,.0f}")
        print(f"  Vendors with GMV > 0: {(vendor_totals > 0).sum():,}")
        print()
        print("  Concentration Metrics:")
        print(f"    Top 1% vendors: {top_1_pct:.1f}% of GMV")
        print(f"    Top 5% vendors: {top_5_pct:.1f}% of GMV")
        print(f"    Top 10% vendors: {top_10_pct:.1f}% of GMV")
        print(f"    Gini coefficient: {gini:.3f}")
        print()

        # Top 10 vendors
        print("  Top 10 Vendors by GMV:")
        print("  " + "-" * 50)
        for i, (vendor, gmv) in enumerate(vendor_totals.head(10).items()):
            pct = gmv / total_gmv * 100
            print(f"    {i+1}. {vendor[:30]:<30} ${gmv:>12,.0f} ({pct:.2f}%)")
        print("  " + "-" * 50)
        print()

        print("  Implication:")
        if gini > 0.8:
            print("    HIGH concentration - Whales dominate. Consider robust SEs / stratification.")
        elif gini > 0.6:
            print("    MODERATE concentration - Standard for marketplaces.")
        else:
            print("    LOW concentration - Good spread of activity.")
    else:
        print("  No GMV in panel.")
    print()

    # =========================================================================
    # Q10: Cannibalization
    # =========================================================================
    print("=" * 70)
    print("Q10: CANNIBALIZATION (Promoted vs Organic GMV)")
    print("=" * 70)
    print()

    # Filter to observations with any activity
    active = panel[(panel['PROMOTED_GMV'] > 0) | (panel['ORGANIC_GMV'] > 0)]

    if len(active) > 0:
        corr = active['PROMOTED_GMV'].corr(active['ORGANIC_GMV'])
        print(f"  Observations with activity: {len(active):,}")
        print(f"  Correlation(Promoted, Organic): {corr:.4f}")
        print()

        # Regression: Organic ~ Promoted
        if len(active) > 100:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(
                active['PROMOTED_GMV'], active['ORGANIC_GMV']
            )
            print("  Simple Regression: ORGANIC_GMV = a + b * PROMOTED_GMV")
            print(f"    Intercept: {intercept:.4f}")
            print(f"    Slope: {slope:.4f}")
            print(f"    R-squared: {r_value**2:.4f}")
            print(f"    P-value: {p_value:.4f}")
            print()

            print("  Interpretation:")
            if slope < -0.1 and p_value < 0.05:
                print("    SUBSTITUTION: Promoted sales cannibalize organic sales.")
            elif slope > 0.1 and p_value < 0.05:
                print("    COMPLEMENT: Promoted and organic sales move together.")
            else:
                print("    INDEPENDENT: No clear relationship between promoted and organic.")
    else:
        print("  No observations with GMV activity.")
    print()

    # =========================================================================
    # SUMMARY SCORECARD
    # =========================================================================
    print("=" * 70)
    print("EDA SCORECARD")
    print("=" * 70)
    print()

    scorecard = {
        'Q1_Orphan_Rate': 'ACKNOWLEDGED (82% by design)',
        'Q2_Panel_Balance': 'BALANCED' if balance_ratio > 0.99 else 'UNBALANCED',
        'Q3_Flicker_Rate': f'{flicker_rate:.1%}' + (' (OK)' if flicker_rate < 0.2 else ' (WARNING)'),
        'Q4_Treated_Vendors': f'{n_treated:,} / {n_vendors:,}',
        'Q5_Ashenfelter_Dip': 'CHECK ABOVE',
        'Q6_CPC_Verification': 'SKIPPED (data too large)',
        'Q7_Rank_Determinism': 'SKIPPED (data too large)',
        'Q8_Zero_Inflation': f"{(panel['TOTAL_GMV'] == 0).mean()*100:.1f}% zeros",
        'Q9_Gini_Coefficient': f'{gini:.3f}' if total_gmv > 0 else 'N/A',
        'Q10_Cannibalization': f'r={corr:.3f}' if len(active) > 0 else 'N/A'
    }

    print("  " + "-" * 50)
    for check, result in scorecard.items():
        print(f"  {check:<25}: {result}")
    print("  " + "-" * 50)
    print()

    print(f"Completed: {datetime.now()}")

if __name__ == '__main__':
    main()
