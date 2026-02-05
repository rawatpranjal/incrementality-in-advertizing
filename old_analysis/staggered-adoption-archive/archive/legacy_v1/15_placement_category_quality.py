#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Placement, Category, and Quality Cuts
Diagnostic analysis to identify profitable sub-segments by context and content.
Uses Oct 11 data pull which includes PLACEMENT and QUALITY columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Oct 11 data pull (has PLACEMENT, QUALITY)
DATA_PULL_DIR = BASE_DIR.parent / "data_pull" / "data"

# Shopping sessions data (for catalog)
SS_DATA_DIR = BASE_DIR.parent / "shopping-sessions" / "data"

OUTPUT_FILE = RESULTS_DIR / "15_placement_category_quality.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def parse_category_field(cats, field_type='category'):
    """Extract specific field from CATEGORIES JSON array."""
    if cats is None:
        return 'unknown'

    if isinstance(cats, str):
        try:
            cats = json.loads(cats)
        except:
            return 'unknown'

    if not isinstance(cats, list):
        return 'unknown'

    for item in cats:
        if isinstance(item, str) and item.startswith(f'{field_type}#'):
            return item.split('#')[1]

    return 'unknown'

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("STAGGERED ADOPTION: PLACEMENT, CATEGORY, AND QUALITY ANALYSIS", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Identify profitable sub-segments by analyzing:", f)
        log("    1. PLACEMENT: Search vs. Feed ad placements", f)
        log("    2. CATEGORY: Product vertical performance", f)
        log("    3. QUALITY: Ad quality score correlation with conversion", f)
        log("", f)

        log("DATA SOURCE: Oct 11 Data Pull (extended schema with PLACEMENT, QUALITY)", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load Data
        # -----------------------------------------------------------------
        log("LOADING DATA", f)
        log("-" * 40, f)

        # Check for Oct 11 data files
        ar_path = DATA_PULL_DIR / "raw_auctions_results_20251011.parquet"
        au_path = DATA_PULL_DIR / "raw_auctions_users_20251011.parquet"
        imp_path = DATA_PULL_DIR / "raw_impressions_20251011.parquet"
        clicks_path = DATA_PULL_DIR / "raw_clicks_20251011.parquet"
        purchases_path = DATA_PULL_DIR / "raw_purchases_20251011.parquet"
        catalog_path = DATA_PULL_DIR / "catalog_20251011.parquet"

        # Alternative catalog path
        if not catalog_path.exists():
            catalog_path = SS_DATA_DIR / "processed_sample_catalog.parquet"

        files_exist = {
            'auctions_results': ar_path.exists(),
            'auctions_users': au_path.exists(),
            'impressions': imp_path.exists(),
            'clicks': clicks_path.exists(),
            'purchases': purchases_path.exists(),
            'catalog': catalog_path.exists()
        }

        log(f"  File availability:", f)
        for name, exists in files_exist.items():
            status = "FOUND" if exists else "MISSING"
            log(f"    {name}: {status}", f)
        log("", f)

        if not files_exist['auctions_results'] or not files_exist['auctions_users']:
            log("  ERROR: Required Oct 11 data files not found", f)
            log("  Cannot proceed with placement/quality analysis", f)
            return

        # Load auction data
        log("  Loading auctions results...", f)
        ar = pd.read_parquet(ar_path)
        log(f"    Shape: {ar.shape}", f)
        log(f"    Columns: {list(ar.columns)}", f)

        log("  Loading auctions users...", f)
        au = pd.read_parquet(au_path)
        log(f"    Shape: {au.shape}", f)
        log(f"    Columns: {list(au.columns)}", f)

        # Merge to get PLACEMENT on auction results
        log("  Merging PLACEMENT to auction results...", f)
        ar = ar.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
        log(f"    Merged shape: {ar.shape}", f)
        log(f"    PLACEMENT coverage: {ar['PLACEMENT'].notna().mean()*100:.1f}%", f)

        # Load funnel data
        impressions = None
        clicks = None
        purchases = None

        if files_exist['impressions']:
            log("  Loading impressions...", f)
            impressions = pd.read_parquet(imp_path)
            log(f"    Shape: {impressions.shape}", f)

        if files_exist['clicks']:
            log("  Loading clicks...", f)
            clicks = pd.read_parquet(clicks_path)
            log(f"    Shape: {clicks.shape}", f)

        if files_exist['purchases']:
            log("  Loading purchases...", f)
            purchases = pd.read_parquet(purchases_path)
            log(f"    Shape: {purchases.shape}", f)

        # Load catalog
        catalog = None
        if files_exist['catalog']:
            log("  Loading catalog...", f)
            catalog = pd.read_parquet(catalog_path)
            log(f"    Shape: {catalog.shape}", f)
            if 'CATEGORIES' in catalog.columns:
                log("    CATEGORIES column found", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 1: PLACEMENT ANALYSIS
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: PLACEMENT ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Ads in 'Search' (high intent) convert better than 'Feed' (low intent)", f)
        log("", f)

        if 'PLACEMENT' in ar.columns:
            # Basic placement distribution
            log("PLACEMENT DISTRIBUTION (Auction Level):", f)
            placement_dist = ar['PLACEMENT'].value_counts().sort_index()
            total_auctions = len(ar['AUCTION_ID'].unique())

            log(f"  {'Placement':<12} {'Bids':<15} {'% of Bids':<12} {'Unique Auctions':<15}", f)
            log(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*15}", f)

            for placement in sorted(ar['PLACEMENT'].dropna().unique()):
                placement_bids = ar[ar['PLACEMENT'] == placement]
                n_bids = len(placement_bids)
                pct_bids = n_bids / len(ar) * 100
                n_auctions = placement_bids['AUCTION_ID'].nunique()
                log(f"  {placement:<12} {n_bids:<15,} {pct_bids:<12.1f} {n_auctions:<15,}", f)

            log("", f)

            # Win rate by placement
            log("WIN RATE BY PLACEMENT:", f)
            placement_wins = ar.groupby('PLACEMENT').agg({
                'IS_WINNER': ['sum', 'count', 'mean'],
                'FINAL_BID': 'sum',
                'QUALITY': 'mean'
            }).reset_index()
            placement_wins.columns = ['PLACEMENT', 'wins', 'bids', 'win_rate', 'total_spend', 'avg_quality']

            log(f"  {'Placement':<12} {'Win Rate':<12} {'Avg Quality':<15} {'Total Spend':<15}", f)
            log(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*15}", f)

            for _, row in placement_wins.iterrows():
                log(f"  {row['PLACEMENT']:<12} {row['win_rate']*100:<12.2f}% {row['avg_quality']:<15.4f} {row['total_spend']:<15,.0f}", f)

            log("", f)

            # CTR by placement (if impressions and clicks available)
            if impressions is not None and clicks is not None:
                log("FUNNEL METRICS BY PLACEMENT:", f)

                # Merge PLACEMENT to impressions via AUCTION_ID
                imp_with_placement = impressions.merge(
                    ar[['AUCTION_ID', 'PLACEMENT', 'VENDOR_ID']].drop_duplicates(),
                    on='AUCTION_ID',
                    how='left',
                    suffixes=('', '_ar')
                )

                # Use VENDOR_ID from impressions if available
                if 'VENDOR_ID_ar' in imp_with_placement.columns:
                    imp_with_placement['VENDOR_ID'] = imp_with_placement['VENDOR_ID'].fillna(imp_with_placement['VENDOR_ID_ar'])

                # Merge PLACEMENT to clicks
                clicks_with_placement = clicks.merge(
                    ar[['AUCTION_ID', 'PLACEMENT']].drop_duplicates(),
                    on='AUCTION_ID',
                    how='left'
                )

                # Aggregate by placement
                imp_by_placement = imp_with_placement.groupby('PLACEMENT').size().reset_index(name='impressions')
                clicks_by_placement = clicks_with_placement.groupby('PLACEMENT').size().reset_index(name='clicks')

                funnel_by_placement = imp_by_placement.merge(clicks_by_placement, on='PLACEMENT', how='outer')
                funnel_by_placement = funnel_by_placement.merge(placement_wins[['PLACEMENT', 'total_spend']], on='PLACEMENT', how='left')

                funnel_by_placement['CTR'] = funnel_by_placement['clicks'] / funnel_by_placement['impressions'] * 100
                funnel_by_placement['CPC'] = funnel_by_placement['total_spend'] / funnel_by_placement['clicks']

                log(f"  {'Placement':<12} {'Impressions':<15} {'Clicks':<12} {'CTR %':<10} {'CPC':<12}", f)
                log(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*10} {'-'*12}", f)

                for _, row in funnel_by_placement.sort_values('PLACEMENT').iterrows():
                    imp_str = f"{row['impressions']:,.0f}" if pd.notna(row['impressions']) else "N/A"
                    clicks_str = f"{row['clicks']:,.0f}" if pd.notna(row['clicks']) else "N/A"
                    ctr_str = f"{row['CTR']:.2f}" if pd.notna(row['CTR']) else "N/A"
                    cpc_str = f"{row['CPC']:,.2f}" if pd.notna(row['CPC']) else "N/A"
                    log(f"  {row['PLACEMENT']:<12} {imp_str:<15} {clicks_str:<12} {ctr_str:<10} {cpc_str:<12}", f)

                log("", f)

            # Placement interpretation
            log("PLACEMENT INTERPRETATION:", f)
            log("  Placement 5: Likely sponsored search (high intent, ~39% of auctions)", f)
            log("  Placement 2: Likely browse/feed (lower intent, ~35% of auctions)", f)
            log("  Placement 1: Premium position (rare, ~9%)", f)
            log("  Placement 3: Mid-tier position (~15%)", f)
            log("  Placement 4: Specialty position (rare, ~2%)", f)

        else:
            log("  PLACEMENT column not found in data", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 2: QUALITY ANALYSIS
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: QUALITY SCORE ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Low-quality ads fill slots but don't convert", f)
        log("", f)

        if 'QUALITY' in ar.columns:
            # Quality distribution
            log("QUALITY SCORE DISTRIBUTION:", f)
            quality_stats = ar['QUALITY'].describe()
            for stat, val in quality_stats.items():
                log(f"  {stat}: {val:.6f}", f)
            log("", f)

            # Quality quartiles
            ar['quality_quartile'] = pd.qcut(ar['QUALITY'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High'], duplicates='drop')

            quality_by_quartile = ar.groupby('quality_quartile', observed=True).agg({
                'QUALITY': 'mean',
                'IS_WINNER': ['sum', 'mean'],
                'FINAL_BID': 'sum',
                'AUCTION_ID': 'count'
            }).reset_index()
            quality_by_quartile.columns = ['quartile', 'avg_quality', 'wins', 'win_rate', 'total_spend', 'n_bids']

            log("METRICS BY QUALITY QUARTILE:", f)
            log(f"  {'Quartile':<12} {'Avg Quality':<15} {'Win Rate':<12} {'Total Spend':<15} {'N Bids':<12}", f)
            log(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*15} {'-'*12}", f)

            for _, row in quality_by_quartile.iterrows():
                log(f"  {row['quartile']:<12} {row['avg_quality']:<15.4f} {row['win_rate']*100:<12.2f}% {row['total_spend']:<15,.0f} {row['n_bids']:<12,}", f)

            log("", f)

            # Quality vs. Conversion Rate correlation
            if 'CONVERSION_RATE' in ar.columns:
                valid_mask = ar['QUALITY'].notna() & ar['CONVERSION_RATE'].notna()
                if valid_mask.sum() > 100:
                    corr = ar.loc[valid_mask, 'QUALITY'].corr(ar.loc[valid_mask, 'CONVERSION_RATE'])
                    log(f"CORRELATION (Quality vs. Conversion Rate): {corr:.4f}", f)

                    if abs(corr) < 0.1:
                        log("  Interpretation: Weak/no correlation - quality score may not predict conversion", f)
                    elif corr > 0.1:
                        log("  Interpretation: Positive correlation - higher quality ads convert better", f)
                    else:
                        log("  Interpretation: Negative correlation - unexpected, investigate", f)
                else:
                    log("  Insufficient data for Quality-Conversion correlation", f)

            log("", f)

            # Quality by placement interaction
            if 'PLACEMENT' in ar.columns:
                log("QUALITY BY PLACEMENT:", f)
                quality_by_placement = ar.groupby('PLACEMENT')['QUALITY'].agg(['mean', 'std', 'count']).reset_index()
                quality_by_placement.columns = ['PLACEMENT', 'avg_quality', 'std_quality', 'n_obs']

                log(f"  {'Placement':<12} {'Avg Quality':<15} {'Std Quality':<15} {'N Obs':<12}", f)
                log(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*12}", f)

                for _, row in quality_by_placement.sort_values('PLACEMENT').iterrows():
                    log(f"  {row['PLACEMENT']:<12} {row['avg_quality']:<15.4f} {row['std_quality']:<15.4f} {row['n_obs']:<12,}", f)

                log("", f)

        else:
            log("  QUALITY column not found in data", f)
            log("  Using RANKING as proxy for quality", f)

            if 'RANKING' in ar.columns:
                log("", f)
                log("RANKING DISTRIBUTION (lower = better):", f)
                ranking_stats = ar['RANKING'].describe()
                for stat, val in ranking_stats.items():
                    log(f"  {stat}: {val:.2f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 3: CATEGORY ANALYSIS
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: CATEGORY/VERTICAL ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Visual categories (Handbags, Shoes) convert better than commodities", f)
        log("", f)

        if catalog is not None and 'CATEGORIES' in catalog.columns:
            log("Parsing CATEGORIES from catalog...", f)

            # Extract category fields
            tqdm.pandas(desc="Extracting categories")
            catalog['category_id'] = catalog['CATEGORIES'].progress_apply(lambda x: parse_category_field(x, 'category'))
            catalog['brand'] = catalog['CATEGORIES'].apply(lambda x: parse_category_field(x, 'brand'))
            catalog['department'] = catalog['CATEGORIES'].apply(lambda x: parse_category_field(x, 'department'))

            log(f"  Unique categories: {catalog['category_id'].nunique()}", f)
            log(f"  Unique brands: {catalog['brand'].nunique()}", f)
            log(f"  Unique departments: {catalog['department'].nunique()}", f)
            log("", f)

            # Join catalog to auction results
            if 'PRODUCT_ID' in ar.columns and 'PRODUCT_ID' in catalog.columns:
                log("Joining catalog to auction results...", f)
                ar_with_cat = ar.merge(
                    catalog[['PRODUCT_ID', 'category_id', 'brand', 'department', 'PRICE']].drop_duplicates(),
                    on='PRODUCT_ID',
                    how='left'
                )

                cat_coverage = ar_with_cat['category_id'].notna().mean() * 100
                log(f"  Category coverage: {cat_coverage:.1f}%", f)
                log("", f)

                # Category performance
                log("CATEGORY PERFORMANCE (Top 15 by bid volume):", f)
                cat_perf = ar_with_cat.groupby('category_id').agg({
                    'AUCTION_ID': 'count',
                    'IS_WINNER': ['sum', 'mean'],
                    'FINAL_BID': 'sum',
                    'QUALITY': 'mean'
                }).reset_index()
                cat_perf.columns = ['category', 'n_bids', 'wins', 'win_rate', 'total_spend', 'avg_quality']
                cat_perf = cat_perf.sort_values('n_bids', ascending=False).head(15)

                log(f"  {'Category':<30} {'Bids':<12} {'Win Rate':<12} {'Spend':<15} {'Avg Quality':<12}", f)
                log(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*15} {'-'*12}", f)

                for _, row in cat_perf.iterrows():
                    cat_name = str(row['category'])[:28] if len(str(row['category'])) > 28 else str(row['category'])
                    log(f"  {cat_name:<30} {row['n_bids']:<12,} {row['win_rate']*100:<12.2f}% {row['total_spend']:<15,.0f} {row['avg_quality']:<12.4f}", f)

                log("", f)

                # Brand performance
                log("BRAND PERFORMANCE (Top 15 by bid volume):", f)
                brand_perf = ar_with_cat.groupby('brand').agg({
                    'AUCTION_ID': 'count',
                    'IS_WINNER': ['sum', 'mean'],
                    'FINAL_BID': 'sum'
                }).reset_index()
                brand_perf.columns = ['brand', 'n_bids', 'wins', 'win_rate', 'total_spend']
                brand_perf = brand_perf[brand_perf['brand'] != 'unknown'].sort_values('n_bids', ascending=False).head(15)

                log(f"  {'Brand':<30} {'Bids':<12} {'Win Rate':<12} {'Spend':<15}", f)
                log(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*15}", f)

                for _, row in brand_perf.iterrows():
                    brand_name = str(row['brand'])[:28] if len(str(row['brand'])) > 28 else str(row['brand'])
                    log(f"  {brand_name:<30} {row['n_bids']:<12,} {row['win_rate']*100:<12.2f}% {row['total_spend']:<15,.0f}", f)

                log("", f)

        else:
            log("  Catalog with CATEGORIES not available", f)
            log("  Skipping category analysis", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 4: VENDOR-LEVEL AGGREGATION
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: VENDOR-LEVEL SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        vendor_summary = ar.groupby('VENDOR_ID').agg({
            'AUCTION_ID': 'count',
            'IS_WINNER': 'sum',
            'FINAL_BID': 'sum',
            'QUALITY': 'mean'
        }).reset_index()
        vendor_summary.columns = ['VENDOR_ID', 'n_bids', 'wins', 'total_spend', 'avg_quality']
        vendor_summary['win_rate'] = vendor_summary['wins'] / vendor_summary['n_bids']

        log(f"Total vendors: {len(vendor_summary):,}", f)
        log("", f)

        log("VENDOR ACTIVITY DISTRIBUTION:", f)
        log(f"  {'Metric':<25} {'Mean':<15} {'Median':<15} {'P90':<15} {'Max':<15}", f)
        log(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15}", f)

        for col in ['n_bids', 'wins', 'total_spend', 'avg_quality', 'win_rate']:
            mean_val = vendor_summary[col].mean()
            median_val = vendor_summary[col].median()
            p90_val = vendor_summary[col].quantile(0.90)
            max_val = vendor_summary[col].max()

            if col in ['avg_quality', 'win_rate']:
                log(f"  {col:<25} {mean_val:<15.4f} {median_val:<15.4f} {p90_val:<15.4f} {max_val:<15.4f}", f)
            else:
                log(f"  {col:<25} {mean_val:<15,.1f} {median_val:<15,.1f} {p90_val:<15,.1f} {max_val:<15,.1f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Summary and Key Findings
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("KEY FINDINGS SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        log("1. PLACEMENT:", f)
        if 'PLACEMENT' in ar.columns:
            top_placement = ar.groupby('PLACEMENT')['IS_WINNER'].mean().idxmax()
            log(f"   - Highest win rate: Placement {top_placement}", f)
            log(f"   - Placement 5 (Search) and 2 (Feed) dominate volume", f)
        log("", f)

        log("2. QUALITY:", f)
        if 'QUALITY' in ar.columns:
            high_q_win_rate = ar[ar['quality_quartile'] == 'Q4_High']['IS_WINNER'].mean() * 100
            low_q_win_rate = ar[ar['quality_quartile'] == 'Q1_Low']['IS_WINNER'].mean() * 100
            log(f"   - Q4 (High Quality) win rate: {high_q_win_rate:.2f}%", f)
            log(f"   - Q1 (Low Quality) win rate: {low_q_win_rate:.2f}%", f)
        log("", f)

        log("3. CATEGORIES:", f)
        log("   - See category performance table above for top-performing verticals", f)
        log("", f)

        log("4. RECOMMENDATIONS:", f)
        log("   - Analyze conversion rates by placement to identify high-ROI positions", f)
        log("   - Consider quality score thresholds for bid filtering", f)
        log("   - Deep-dive into top-performing categories for optimization", f)

        log("", f)
        log("=" * 80, f)
        log("PLACEMENT/CATEGORY/QUALITY ANALYSIS COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
