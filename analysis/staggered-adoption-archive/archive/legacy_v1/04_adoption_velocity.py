#!/usr/bin/env python3
"""
EDA Q4: What is the Adoption Velocity?

Plot the cumulative number of treated vendors over time (Weeks 1-26).
Do we have sufficient variation in start dates? If 90% of vendors start in Week 1,
we don't have a staggered design; we have a cross-sectional comparison.
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
BASE_DIR = Path(__file__).parent.parent  # staggered-adoption/
EDA_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "04_adoption_velocity.txt"

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
        log("EDA Q4: ADOPTION VELOCITY", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  Do we have sufficient variation in treatment adoption timing?", f)
        log("  Problem: If 90%+ adopt in week 1, we have no staggered design.", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. Extract cohort_week (G_i = first week of treatment)", f)
        log("  2. Tabulate cumulative adoption by week", f)
        log("  3. Assess variation in treatment timing", f)
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
        log(f"  Unique weeks: {panel['week'].nunique()}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Extract cohort assignment
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("COHORT ASSIGNMENT ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # Get unique vendor-cohort mapping
        if 'cohort_week' in panel.columns:
            vendor_cohorts = panel.groupby('VENDOR_ID')['cohort_week'].first().reset_index()
        else:
            # Compute cohort_week as first week with positive spend
            first_treatment = panel[panel['has_spend'] == 1].groupby('VENDOR_ID')['week'].min().reset_index()
            first_treatment.columns = ['VENDOR_ID', 'cohort_week']

            all_vendors = panel[['VENDOR_ID']].drop_duplicates()
            vendor_cohorts = all_vendors.merge(first_treatment, on='VENDOR_ID', how='left')

        total_vendors = len(vendor_cohorts)
        treated_vendors = vendor_cohorts['cohort_week'].notna().sum()
        never_treated = vendor_cohorts['cohort_week'].isna().sum()

        log(f"  Total vendors: {total_vendors:,}", f)
        log(f"  Treated vendors (ever): {treated_vendors:,} ({treated_vendors/total_vendors*100:.1f}%)", f)
        log(f"  Never-treated vendors: {never_treated:,} ({never_treated/total_vendors*100:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Cohort distribution by week
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ADOPTION BY WEEK (COHORT DISTRIBUTION)", f)
        log("-" * 40, f)
        log("", f)

        # Filter to treated vendors
        treated = vendor_cohorts[vendor_cohorts['cohort_week'].notna()].copy()
        treated['cohort_week'] = pd.to_datetime(treated['cohort_week'])

        # Count by cohort week
        cohort_counts = treated.groupby('cohort_week').size().reset_index(name='n_adopted')
        cohort_counts = cohort_counts.sort_values('cohort_week')

        # Cumulative adoption
        cohort_counts['cumulative'] = cohort_counts['n_adopted'].cumsum()
        cohort_counts['pct_adopted'] = cohort_counts['n_adopted'] / treated_vendors * 100
        cohort_counts['cumulative_pct'] = cohort_counts['cumulative'] / treated_vendors * 100

        log(f"  {'Week':<12} {'New':>8} {'Cumulative':>12} {'% New':>10} {'% Cum':>10}", f)
        log(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*10} {'-'*10}", f)

        for _, row in cohort_counts.iterrows():
            week_str = str(row['cohort_week'])[:10]
            log(f"  {week_str:<12} {row['n_adopted']:>8,} {row['cumulative']:>12,} {row['pct_adopted']:>9.1f}% {row['cumulative_pct']:>9.1f}%", f)

        log("", f)

        # -----------------------------------------------------------------
        # Staggered design assessment
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STAGGERED DESIGN ASSESSMENT", f)
        log("-" * 40, f)
        log("", f)

        # Check concentration in early weeks
        first_week_pct = cohort_counts.iloc[0]['cumulative_pct'] if len(cohort_counts) > 0 else 0
        first_3_weeks = cohort_counts.head(3)['cumulative_pct'].max() if len(cohort_counts) >= 3 else 0

        log(f"  % adopted in Week 1: {first_week_pct:.1f}%", f)
        log(f"  % adopted by Week 3: {first_3_weeks:.1f}%", f)
        log("", f)

        # Number of effective cohorts
        n_cohorts = len(cohort_counts)
        log(f"  Number of cohort weeks: {n_cohorts}", f)
        log("", f)

        # Cohort size distribution
        log("  COHORT SIZE DISTRIBUTION:", f)
        size_stats = cohort_counts['n_adopted'].describe()
        log(f"    Mean cohort size: {size_stats['mean']:,.0f}", f)
        log(f"    Median cohort size: {size_stats['50%']:,.0f}", f)
        log(f"    Min cohort size: {size_stats['min']:,.0f}", f)
        log(f"    Max cohort size: {size_stats['max']:,.0f}", f)
        log("", f)

        # Concentration metrics
        largest_cohort_pct = cohort_counts['pct_adopted'].max()
        top_3_cohorts_pct = cohort_counts.nlargest(3, 'n_adopted')['pct_adopted'].sum()

        log(f"  CONCENTRATION METRICS:", f)
        log(f"    Largest cohort: {largest_cohort_pct:.1f}% of treated", f)
        log(f"    Top 3 cohorts: {top_3_cohorts_pct:.1f}% of treated", f)
        log("", f)

        # Herfindahl index (measure of concentration)
        hhi = (cohort_counts['pct_adopted'] ** 2).sum() / 100  # Normalize to 0-100
        log(f"    Herfindahl Index (HHI): {hhi:.2f}", f)
        log(f"      (1/N = {100/n_cohorts:.2f} for uniform distribution)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Early vs Late adopters
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("EARLY VS LATE ADOPTER ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        median_cohort = treated['cohort_week'].median()
        early_adopters = (treated['cohort_week'] <= median_cohort).sum()
        late_adopters = (treated['cohort_week'] > median_cohort).sum()

        log(f"  Median adoption week: {str(median_cohort)[:10]}", f)
        log(f"  Early adopters (<=median): {early_adopters:,} ({early_adopters/treated_vendors*100:.1f}%)", f)
        log(f"  Late adopters (>median): {late_adopters:,} ({late_adopters/treated_vendors*100:.1f}%)", f)
        log("", f)

        # Quartile breakdown
        quartiles = treated['cohort_week'].quantile([0.25, 0.5, 0.75])
        log("  ADOPTION QUARTILES:", f)
        for q, val in quartiles.items():
            log(f"    Q{int(q*100)}: {str(val)[:10]}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        if first_week_pct > 90:
            log("  [CRITICAL] >90% adopted in Week 1.", f)
            log("  This is NOT a staggered design - it's essentially cross-sectional.", f)
            log("  Callaway-Sant'Anna gains minimal power over simple 2x2 DiD.", f)
        elif first_week_pct > 50:
            log("  [WARNING] >50% adopted in Week 1.", f)
            log("  Adoption is front-loaded. Limited variation in timing.", f)
            log("  CS estimator will work but precision depends on later cohorts.", f)
        elif n_cohorts < 5:
            log("  [WARNING] Fewer than 5 cohort weeks.", f)
            log("  Limited variation in treatment timing.", f)
        else:
            log("  [OK] Good variation in adoption timing.", f)
            log("  Staggered DiD design is appropriate.", f)

        log("", f)

        if never_treated < 100:
            log(f"  [NOTE] Only {never_treated} never-treated vendors.", f)
            log("  Small control group may limit precision of ATT estimates.", f)
            log("", f)

        log("  RECOMMENDATIONS:", f)
        log(f"    1. {n_cohorts} cohorts provide variation for group-time ATTs", f)
        log(f"    2. Never-treated group ({never_treated:,}) available for comparison", f)
        log(f"    3. Consider robustness with not-yet-treated as controls", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
