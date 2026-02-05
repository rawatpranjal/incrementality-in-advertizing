#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Vendor Segmentation
Creates quartile and persona-based segments for HTE analysis.
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

OUTPUT_FILE = RESULTS_DIR / "11_vendor_segmentation.txt"

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
        log("STAGGERED ADOPTION: VENDOR SEGMENTATION", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Create vendor segments for heterogeneous treatment effect analysis.", f)
        log("  Segmentation allows us to answer:", f)
        log("    - For whom does advertising work?", f)
        log("    - Under what conditions are treatment effects largest?", f)
        log("    - Do parallel trends hold within segments?", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load covariates
        # -----------------------------------------------------------------
        log("LOADING VENDOR COVARIATES", f)
        log("-" * 40, f)

        covariates_path = DATA_DIR / "vendor_covariates.parquet"

        if not covariates_path.exists():
            log(f"  ERROR: Covariates not found at {covariates_path}", f)
            log("  Run 10_pretreatment_covariates.py first", f)
            return

        covariates = pd.read_parquet(covariates_path)
        log(f"  Loaded {len(covariates):,} vendors", f)
        log(f"  Columns: {list(covariates.columns)}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Data availability summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("DATA AVAILABILITY FOR SEGMENTATION", f)
        log("-" * 40, f)

        # Check how many vendors have pre-treatment activity
        has_pre_activity = covariates['pre_auction_count'] > 0
        n_with_activity = has_pre_activity.sum()
        n_without_activity = (~has_pre_activity).sum()

        log(f"  Vendors with pre-treatment auction activity: {n_with_activity:,} ({n_with_activity/len(covariates)*100:.2f}%)", f)
        log(f"  Vendors without pre-treatment activity: {n_without_activity:,} ({n_without_activity/len(covariates)*100:.2f}%)", f)
        log("", f)

        # For vendors with activity, check covariate availability
        active_vendors = covariates[has_pre_activity]
        log(f"  Among active vendors ({n_with_activity:,}):", f)
        for col in ['pre_avg_price_point', 'pre_avg_ranking', 'pre_win_rate']:
            if col in covariates.columns:
                n_avail = active_vendors[col].notna().sum()
                log(f"    {col}: {n_avail:,} available ({n_avail/len(active_vendors)*100:.2f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Strategy 1: Quartile Segmentation by Activity
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SEGMENTATION STRATEGY 1: QUARTILE BY ACTIVITY", f)
        log("-" * 40, f)

        log("  Definition:", f)
        log("    Q1 (Low): pre_auction_count = 0 (no pre-treatment activity)", f)
        log("    Q2-Q4: Quartiles among vendors with pre_auction_count > 0", f)
        log("", f)

        # Create activity quartile
        # Use rank-based approach to handle ties
        try:
            # Try standard qcut first
            active_quartiles = pd.qcut(
                active_vendors['pre_auction_count'],
                q=4,
                labels=['Q2_Low', 'Q3_Medium', 'Q4_High', 'Q5_VeryHigh'],
                duplicates='drop'
            )
        except ValueError:
            # Fall back to rank-based quantiles for data with many ties
            ranks = active_vendors['pre_auction_count'].rank(method='first')
            active_quartiles = pd.qcut(
                ranks,
                q=4,
                labels=['Q2_Low', 'Q3_Medium', 'Q4_High', 'Q5_VeryHigh']
            )

        # Map back to full dataset
        covariates['activity_quartile_among_active'] = pd.Series(
            active_quartiles.astype(str).values,
            index=active_vendors.index
        )
        covariates['activity_quartile'] = covariates.apply(
            lambda x: 'Q1_Inactive' if x['pre_auction_count'] == 0
            else x['activity_quartile_among_active'],
            axis=1
        )

        # Segment counts
        activity_counts = covariates['activity_quartile'].value_counts().sort_index()
        log("  Activity Quartile Distribution:", f)
        for seg, count in activity_counts.items():
            pct = count / len(covariates) * 100
            log(f"    {seg}: {count:,} ({pct:.2f}%)", f)
        log("", f)

        # Summary stats by segment
        log("  Activity Quartile Summary Statistics:", f)
        log(f"  {'Segment':<20} {'N':<10} {'Mean Auctions':<15} {'Mean Price':<15}", f)
        log(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*15}", f)

        for seg in activity_counts.index:
            seg_data = covariates[covariates['activity_quartile'] == seg]
            n = len(seg_data)
            mean_auctions = seg_data['pre_auction_count'].mean()
            mean_price = seg_data['pre_avg_price_point'].mean()
            log(f"  {seg:<20} {n:<10,} {mean_auctions:<15.2f} {mean_price:<15.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Strategy 2: Price Point Quartiles (among active vendors)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SEGMENTATION STRATEGY 2: PRICE POINT QUARTILES", f)
        log("-" * 40, f)

        log("  Definition:", f)
        log("    Quartiles based on pre_avg_price_point (mean price of advertised products)", f)
        log("    Only computed for vendors with valid price data", f)
        log("", f)

        # Filter to vendors with price data
        price_vendors = covariates[covariates['pre_avg_price_point'].notna()]

        if len(price_vendors) > 0:
            try:
                price_quartiles = pd.qcut(
                    price_vendors['pre_avg_price_point'],
                    q=4,
                    labels=['P1_Budget', 'P2_MidLow', 'P3_MidHigh', 'P4_Premium'],
                    duplicates='drop'
                )
            except ValueError:
                ranks = price_vendors['pre_avg_price_point'].rank(method='first')
                price_quartiles = pd.qcut(
                    ranks,
                    q=4,
                    labels=['P1_Budget', 'P2_MidLow', 'P3_MidHigh', 'P4_Premium']
                )

            covariates['price_quartile'] = pd.Series(
                price_quartiles.astype(str).values,
                index=price_vendors.index
            )
            covariates['price_quartile'] = covariates['price_quartile'].fillna('P0_Unknown')

            price_counts = covariates['price_quartile'].value_counts().sort_index()
            log("  Price Quartile Distribution:", f)
            for seg, count in price_counts.items():
                pct = count / len(covariates) * 100
                log(f"    {seg}: {count:,} ({pct:.2f}%)", f)
            log("", f)

            # Price quartile thresholds
            log("  Price Quartile Thresholds:", f)
            price_quantiles = price_vendors['pre_avg_price_point'].quantile([0.25, 0.5, 0.75])
            log(f"    P1 (Budget): <= ${price_quantiles[0.25]:.2f}", f)
            log(f"    P2 (Mid-Low): ${price_quantiles[0.25]:.2f} - ${price_quantiles[0.5]:.2f}", f)
            log(f"    P3 (Mid-High): ${price_quantiles[0.5]:.2f} - ${price_quantiles[0.75]:.2f}", f)
            log(f"    P4 (Premium): > ${price_quantiles[0.75]:.2f}", f)
        else:
            covariates['price_quartile'] = 'P0_Unknown'
            log("  No price data available for quartile segmentation", f)

        log("", f)

        # -----------------------------------------------------------------
        # Strategy 3: Ranking Quartiles (auction performance)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SEGMENTATION STRATEGY 3: RANKING QUARTILES (PERFORMANCE)", f)
        log("-" * 40, f)

        log("  Definition:", f)
        log("    Quartiles based on pre_avg_ranking (lower = better performance)", f)
        log("    R1_Top: Lowest ranking (best performers)", f)
        log("    R4_Bottom: Highest ranking (worst performers)", f)
        log("", f)

        # Filter to vendors with ranking data
        ranking_vendors = covariates[covariates['pre_avg_ranking'].notna()]

        if len(ranking_vendors) > 0:
            try:
                ranking_quartiles = pd.qcut(
                    ranking_vendors['pre_avg_ranking'],
                    q=4,
                    labels=['R1_Top', 'R2_Good', 'R3_Average', 'R4_Bottom'],
                    duplicates='drop'
                )
            except ValueError:
                ranks = ranking_vendors['pre_avg_ranking'].rank(method='first')
                ranking_quartiles = pd.qcut(
                    ranks,
                    q=4,
                    labels=['R1_Top', 'R2_Good', 'R3_Average', 'R4_Bottom']
                )

            covariates['ranking_quartile'] = pd.Series(
                ranking_quartiles.astype(str).values,
                index=ranking_vendors.index
            )
            covariates['ranking_quartile'] = covariates['ranking_quartile'].fillna('R0_Unknown')

            ranking_counts = covariates['ranking_quartile'].value_counts().sort_index()
            log("  Ranking Quartile Distribution:", f)
            for seg, count in ranking_counts.items():
                pct = count / len(covariates) * 100
                log(f"    {seg}: {count:,} ({pct:.2f}%)", f)
            log("", f)

            # Ranking quartile thresholds
            log("  Ranking Quartile Thresholds (avg ranking):", f)
            ranking_quantiles = ranking_vendors['pre_avg_ranking'].quantile([0.25, 0.5, 0.75])
            log(f"    R1 (Top): <= {ranking_quantiles[0.25]:.1f}", f)
            log(f"    R2 (Good): {ranking_quantiles[0.25]:.1f} - {ranking_quantiles[0.5]:.1f}", f)
            log(f"    R3 (Average): {ranking_quantiles[0.5]:.1f} - {ranking_quantiles[0.75]:.1f}", f)
            log(f"    R4 (Bottom): > {ranking_quantiles[0.75]:.1f}", f)
        else:
            covariates['ranking_quartile'] = 'R0_Unknown'
            log("  No ranking data available for quartile segmentation", f)

        log("", f)

        # -----------------------------------------------------------------
        # Strategy 4: Persona-Based Segmentation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SEGMENTATION STRATEGY 4: PERSONA-BASED ARCHETYPES", f)
        log("-" * 40, f)

        log("  Definition:", f)
        log("    Combines multiple dimensions to create interpretable vendor personas", f)
        log("", f)
        log("  Personas:", f)
        log("    1. Power Seller: High activity (Q5) AND good ranking (R1-R2)", f)
        log("    2. Premium Boutique: High price (P4) AND any activity", f)
        log("    3. Active Generalist: Medium-high activity (Q3-Q4) AND not premium", f)
        log("    4. Casual Seller: Low activity (Q2) OR poor ranking (R4)", f)
        log("    5. New Adopter: No pre-treatment activity (Q1_Inactive)", f)
        log("", f)

        def assign_persona(row):
            activity = row['activity_quartile']
            price = row['price_quartile']
            ranking = row['ranking_quartile']

            # New Adopter: no pre-treatment activity
            if activity == 'Q1_Inactive':
                return 'New_Adopter'

            # Power Seller: high activity + good ranking
            if activity in ['Q4_High', 'Q5_VeryHigh'] and ranking in ['R1_Top', 'R2_Good']:
                return 'Power_Seller'

            # Premium Boutique: high price point
            if price == 'P4_Premium':
                return 'Premium_Boutique'

            # Active Generalist: medium-high activity
            if activity in ['Q3_Medium', 'Q4_High', 'Q5_VeryHigh']:
                return 'Active_Generalist'

            # Casual Seller: everyone else
            return 'Casual_Seller'

        covariates['persona'] = covariates.apply(assign_persona, axis=1)

        persona_counts = covariates['persona'].value_counts()
        log("  Persona Distribution:", f)
        for persona, count in persona_counts.items():
            pct = count / len(covariates) * 100
            log(f"    {persona}: {count:,} ({pct:.2f}%)", f)
        log("", f)

        # Persona characteristics
        log("  Persona Characteristics:", f)
        log(f"  {'Persona':<20} {'N':<10} {'Auctions':<12} {'Price':<12} {'Ranking':<12} {'Treated %':<12}", f)
        log(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

        for persona in persona_counts.index:
            seg_data = covariates[covariates['persona'] == persona]
            n = len(seg_data)
            mean_auctions = seg_data['pre_auction_count'].mean()
            mean_price = seg_data['pre_avg_price_point'].mean()
            mean_ranking = seg_data['pre_avg_ranking'].mean()
            treated_pct = seg_data['is_treated'].mean() * 100

            log(f"  {persona:<20} {n:<10,} {mean_auctions:<12.2f} {mean_price:<12.2f} {mean_ranking:<12.2f} {treated_pct:<12.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Strategy 5: Simple Binary Segments (for robust estimation)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SEGMENTATION STRATEGY 5: BINARY SEGMENTS (ROBUST)", f)
        log("-" * 40, f)

        log("  Definition:", f)
        log("    Simple binary splits for robust HTE estimation with larger sample sizes", f)
        log("", f)

        # Active vs Inactive
        covariates['is_active'] = (covariates['pre_auction_count'] > 0).astype(int)
        active_counts = covariates['is_active'].value_counts()
        log(f"  Active (pre_auction_count > 0): {active_counts.get(1, 0):,}", f)
        log(f"  Inactive (pre_auction_count = 0): {active_counts.get(0, 0):,}", f)
        log("", f)

        # High vs Low Activity (among active)
        median_activity = covariates[covariates['pre_auction_count'] > 0]['pre_auction_count'].median()
        covariates['high_activity'] = (covariates['pre_auction_count'] > median_activity).astype(int)
        log(f"  Median activity threshold: {median_activity:.0f} auctions", f)
        high_activity_counts = covariates['high_activity'].value_counts()
        log(f"  High activity: {high_activity_counts.get(1, 0):,}", f)
        log(f"  Low activity: {high_activity_counts.get(0, 0):,}", f)
        log("", f)

        # Premium vs Non-Premium (among those with price data)
        median_price = covariates[covariates['pre_avg_price_point'].notna()]['pre_avg_price_point'].median()
        covariates['is_premium'] = (covariates['pre_avg_price_point'] > median_price).astype(int)
        covariates.loc[covariates['pre_avg_price_point'].isna(), 'is_premium'] = np.nan
        log(f"  Median price threshold: ${median_price:.2f}", f)
        premium_counts = covariates['is_premium'].value_counts()
        log(f"  Premium: {premium_counts.get(1.0, 0):,}", f)
        log(f"  Non-Premium: {premium_counts.get(0.0, 0):,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Cross-tabulation of segments
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SEGMENT CROSS-TABULATION", f)
        log("-" * 40, f)
        log("", f)

        # Activity x Price
        log("  Activity Quartile x Price Quartile:", f)
        crosstab = pd.crosstab(covariates['activity_quartile'], covariates['price_quartile'])
        log(crosstab.to_string(), f)
        log("", f)

        # Activity x Persona
        log("  Activity Quartile x Persona:", f)
        crosstab2 = pd.crosstab(covariates['activity_quartile'], covariates['persona'])
        log(crosstab2.to_string(), f)
        log("", f)

        # -----------------------------------------------------------------
        # Save segmented covariates
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SAVING VENDOR SEGMENTS", f)
        log("-" * 40, f)

        # Select columns to save
        segment_cols = [
            'VENDOR_ID', 'cohort_week', 'is_treated',
            # Covariates
            'pre_auction_count', 'pre_bid_count', 'pre_win_rate',
            'pre_impression_count', 'pre_click_count',
            'pre_weeks_active', 'pre_ctr', 'pre_avg_ranking',
            'pre_avg_price_point', 'pre_unique_products',
            # Segments
            'activity_quartile', 'price_quartile', 'ranking_quartile', 'persona',
            'is_active', 'high_activity', 'is_premium'
        ]

        segments_df = covariates[[c for c in segment_cols if c in covariates.columns]]

        output_path = DATA_DIR / "vendor_segments.parquet"
        segments_df.to_parquet(output_path, index=False)

        log(f"  Output shape: {segments_df.shape}", f)
        log(f"  Columns: {list(segments_df.columns)}", f)
        log(f"  Saved to: {output_path}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Merge segments with panel
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("MERGING SEGMENTS WITH PANEL", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_cohorts.parquet"
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
            panel['week'] = pd.to_datetime(panel['week'])

            log(f"  Loaded panel: {panel.shape}", f)

            # Merge segments
            merge_cols = ['VENDOR_ID', 'activity_quartile', 'price_quartile',
                         'ranking_quartile', 'persona', 'is_active', 'high_activity', 'is_premium',
                         'pre_auction_count', 'pre_avg_price_point', 'pre_avg_ranking']
            merge_cols = [c for c in merge_cols if c in segments_df.columns]

            panel_with_segments = panel.merge(
                segments_df[merge_cols],
                on='VENDOR_ID',
                how='left'
            )

            log(f"  Panel with segments: {panel_with_segments.shape}", f)
            log("", f)

            # Save
            output_panel_path = DATA_DIR / "panel_with_segments.parquet"
            panel_with_segments.to_parquet(output_panel_path, index=False)
            log(f"  Saved to: {output_panel_path}", f)

            # Segment distribution in panel
            log("", f)
            log("  Segment Distribution in Panel (observations):", f)
            for seg_col in ['activity_quartile', 'persona']:
                if seg_col in panel_with_segments.columns:
                    log(f"", f)
                    log(f"  {seg_col}:", f)
                    seg_counts = panel_with_segments[seg_col].value_counts()
                    for seg, count in seg_counts.items():
                        pct = count / len(panel_with_segments) * 100
                        log(f"    {seg}: {count:,} obs ({pct:.2f}%)", f)
        else:
            log("  Panel not found, skipping merge", f)

        log("", f)
        log("=" * 80, f)
        log("VENDOR SEGMENTATION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
