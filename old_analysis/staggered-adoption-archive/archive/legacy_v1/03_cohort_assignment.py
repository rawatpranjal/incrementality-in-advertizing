#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Cohort Assignment
Assigns treatment cohorts (G_i) based on first week of positive ad spend.
Creates treatment indicators D_it = 1{t >= G_i}.
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

OUTPUT_FILE = RESULTS_DIR / "03_cohort_assignment.txt"

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
        log("STAGGERED ADOPTION: COHORT ASSIGNMENT", f)
        log("=" * 80, f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  Staggered DiD requires identifying:", f)
        log("    - G_i: The first time period when unit i becomes treated", f)
        log("    - D_it: Treatment indicator = 1{t >= G_i}", f)
        log("", f)
        log("  Treatment definition: First week with positive ad spend", f)
        log("  Control group: Vendors who never spend (G_i = infinity)", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_vendor_week.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            log("  Run 02_panel_construction.py first", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])

        log(f"  Panel shape: {panel.shape}", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log(f"  Unique weeks: {panel['week'].nunique()}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Step 1: Identify first treatment week per vendor
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 1: IDENTIFY FIRST TREATMENT WEEK (G_i)", f)
        log("-" * 40, f)

        # Vendors with positive spend
        spending_vendors = panel[panel['total_spend'] > 0][['VENDOR_ID', 'week']].copy()

        if len(spending_vendors) > 0:
            # First week with spend per vendor
            first_spend = spending_vendors.groupby('VENDOR_ID')['week'].min().reset_index()
            first_spend.columns = ['VENDOR_ID', 'cohort_week']

            log(f"  Vendors with any spend: {len(first_spend):,}", f)
            log("", f)

            # Merge back to panel
            panel = panel.merge(first_spend, on='VENDOR_ID', how='left')

            # Vendors without spend get cohort_week = NaT (never treated)
            n_never_treated = panel['cohort_week'].isna().sum()
            log(f"  Observations from never-treated vendors: {n_never_treated:,}", f)

        else:
            log("  No vendors with positive spend found", f)
            panel['cohort_week'] = pd.NaT

        log("", f)

        # -----------------------------------------------------------------
        # Step 2: Create treatment indicator D_it
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 2: CREATE TREATMENT INDICATOR D_it", f)
        log("-" * 40, f)

        # D_it = 1 if t >= G_i (i.e., week >= cohort_week)
        panel['treated'] = (panel['week'] >= panel['cohort_week']).astype(int)

        # Never-treated should have treated = 0
        panel.loc[panel['cohort_week'].isna(), 'treated'] = 0

        n_treated_obs = panel['treated'].sum()
        n_control_obs = len(panel) - n_treated_obs

        log(f"  Treated observations (D_it = 1): {n_treated_obs:,}", f)
        log(f"  Control observations (D_it = 0): {n_control_obs:,}", f)
        log(f"  Treatment rate: {n_treated_obs / len(panel) * 100:.2f}%", f)

        log("", f)

        # -----------------------------------------------------------------
        # Step 3: Create relative time to treatment (event time)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 3: CREATE RELATIVE TIME (e = t - G_i)", f)
        log("-" * 40, f)

        # Calculate relative time in weeks
        panel['relative_week'] = np.nan
        mask = panel['cohort_week'].notna()
        panel.loc[mask, 'relative_week'] = (
            (panel.loc[mask, 'week'] - panel.loc[mask, 'cohort_week']).dt.days / 7
        ).round().astype(int)

        if mask.any():
            min_rel = panel.loc[mask, 'relative_week'].min()
            max_rel = panel.loc[mask, 'relative_week'].max()

            log(f"  Relative time range: {min_rel} to {max_rel} weeks", f)
            log("", f)

            # Distribution of relative time
            rel_time_dist = panel[mask].groupby('relative_week').size()
            log("  Observations by relative time:", f)
            for rel_t, count in rel_time_dist.items():
                label = f"e = {int(rel_t):+d}"
                log(f"    {label:<10}: {count:,} observations", f)

        log("", f)

        # -----------------------------------------------------------------
        # Step 4: Cohort summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 4: COHORT SUMMARY", f)
        log("-" * 40, f)

        # Convert cohort_week to string for display
        panel['cohort_week_str'] = panel['cohort_week'].astype(str)
        panel.loc[panel['cohort_week'].isna(), 'cohort_week_str'] = 'Never Treated'

        cohort_summary = panel.groupby('cohort_week_str').agg(
            n_vendors=('VENDOR_ID', 'nunique'),
            n_observations=('VENDOR_ID', 'count'),
            mean_spend=('total_spend', 'mean'),
            mean_promoted_gmv=('promoted_gmv', 'mean'),
        ).reset_index()

        log(f"  {'Cohort':<25} {'Vendors':>10} {'Obs':>10} {'Avg Spend':>12} {'Avg GMV':>12}", f)
        log(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*12}", f)

        for _, row in cohort_summary.iterrows():
            log(f"  {row['cohort_week_str']:<25} {row['n_vendors']:>10,} {row['n_observations']:>10,} {row['mean_spend']:>12.2f} {row['mean_promoted_gmv']:>12.2f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Step 5: Create numeric cohort identifier
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 5: CREATE NUMERIC COHORT IDENTIFIER", f)
        log("-" * 40, f)

        # Map cohort weeks to integers (for differences library)
        unique_cohorts = panel[panel['cohort_week'].notna()]['cohort_week'].unique()
        unique_cohorts = sorted(unique_cohorts)

        cohort_map = {cohort: i + 1 for i, cohort in enumerate(unique_cohorts)}
        cohort_map[pd.NaT] = 0  # Never treated = 0

        panel['cohort_id'] = panel['cohort_week'].map(cohort_map)
        panel['cohort_id'] = panel['cohort_id'].fillna(0).astype(int)

        # Create numeric week identifier
        unique_weeks = sorted(panel['week'].unique())
        week_map = {week: i + 1 for i, week in enumerate(unique_weeks)}
        panel['week_id'] = panel['week'].map(week_map)

        log(f"  Number of unique cohorts: {len(cohort_map)}", f)
        log(f"  Number of unique weeks: {len(week_map)}", f)
        log("", f)

        log("  Cohort ID mapping:", f)
        for cohort, cohort_id in sorted(cohort_map.items(), key=lambda x: x[1]):
            if pd.isna(cohort):
                log(f"    Never Treated -> {cohort_id}", f)
            else:
                log(f"    {cohort.date()} -> {cohort_id}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Step 6: Identify never-treated control group
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 6: CONTROL GROUP SUMMARY", f)
        log("-" * 40, f)

        never_treated = panel[panel['cohort_id'] == 0]
        ever_treated = panel[panel['cohort_id'] > 0]

        log(f"  Never-treated vendors: {never_treated['VENDOR_ID'].nunique():,}", f)
        log(f"  Ever-treated vendors: {ever_treated['VENDOR_ID'].nunique():,}", f)
        log("", f)

        if len(never_treated) > 0:
            log("  Never-treated group characteristics:", f)
            log(f"    Mean auction participations: {never_treated['auction_participations'].mean():.2f}", f)
            log(f"    Mean impressions: {never_treated['impressions'].mean():.2f}", f)
            log(f"    Mean clicks: {never_treated['clicks'].mean():.2f}", f)

        if len(ever_treated) > 0:
            log("", f)
            log("  Ever-treated group characteristics:", f)
            log(f"    Mean auction participations: {ever_treated['auction_participations'].mean():.2f}", f)
            log(f"    Mean impressions: {ever_treated['impressions'].mean():.2f}", f)
            log(f"    Mean clicks: {ever_treated['clicks'].mean():.2f}", f)
            log(f"    Mean spend (when treated): {ever_treated[ever_treated['treated'] == 1]['total_spend'].mean():.2f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Save panel with cohorts
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SAVING PANEL WITH COHORTS", f)
        log("-" * 40, f)

        # Drop helper columns
        panel = panel.drop(columns=['cohort_week_str'], errors='ignore')

        output_path = DATA_DIR / "panel_with_cohorts.parquet"
        panel.to_parquet(output_path, index=False)

        log(f"  Saved to {output_path}", f)
        log(f"  Final panel shape: {panel.shape}", f)
        log("", f)

        log("  Final panel columns:", f)
        for col in panel.columns:
            log(f"    - {col}: {panel[col].dtype}", f)

        log("", f)
        log("=" * 80, f)
        log("COHORT ASSIGNMENT COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
