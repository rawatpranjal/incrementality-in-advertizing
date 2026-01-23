#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Pre-Treatment Covariate Construction
Constructs time-invariant vendor characteristics from pre-treatment period for HTE analysis.
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
RAW_DATA_DIR = BASE_DIR / ".." / "shopping-sessions" / "data"

OUTPUT_FILE = RESULTS_DIR / "10_pretreatment_covariates.txt"

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
    DATA_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("STAGGERED ADOPTION: PRE-TREATMENT COVARIATE CONSTRUCTION", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Construct time-invariant vendor characteristics from pre-treatment period.", f)
        log("  These covariates will be used for:", f)
        log("    1. Conditional parallel trends validation (within-segment balance)", f)
        log("    2. Vendor segmentation (quartiles, personas)", f)
        log("    3. Heterogeneous treatment effect estimation by subgroup", f)
        log("", f)

        log("PRE-TREATMENT WINDOW DEFINITION:", f)
        log("  For each vendor i: use all data BEFORE their cohort_week (treatment date)", f)
        log("  For never-treated vendors: use entire observation period", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel with cohorts
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_cohorts.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            log("  Run 03_cohort_assignment.py first", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])
        panel['cohort_week'] = pd.to_datetime(panel['cohort_week'])

        log(f"  Panel shape: {panel.shape}", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log(f"  Unique weeks: {panel['week'].nunique()}", f)
        log("", f)

        # Get cohort week per vendor
        vendor_cohort = panel.groupby('VENDOR_ID')['cohort_week'].first().reset_index()
        vendor_cohort.columns = ['VENDOR_ID', 'cohort_week']

        n_treated = vendor_cohort['cohort_week'].notna().sum()
        n_never_treated = vendor_cohort['cohort_week'].isna().sum()

        log(f"  Treated vendors: {n_treated:,}", f)
        log(f"  Never-treated vendors: {n_never_treated:,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Load raw data sources
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("LOADING RAW DATA SOURCES", f)
        log("-" * 40, f)

        # Auctions results
        auctions_path = RAW_DATA_DIR / "raw_sample_auctions_results.parquet"
        if auctions_path.exists():
            auctions = pd.read_parquet(auctions_path)
            auctions['CREATED_AT'] = pd.to_datetime(auctions['CREATED_AT'])
            auctions['week'] = auctions['CREATED_AT'].dt.to_period('W').apply(lambda x: x.start_time)
            log(f"  auctions_results: {len(auctions):,} rows", f)
        else:
            log("  ERROR: auctions_results not found", f)
            auctions = pd.DataFrame()

        # Impressions
        impressions_path = RAW_DATA_DIR / "raw_sample_impressions.parquet"
        if impressions_path.exists():
            impressions = pd.read_parquet(impressions_path)
            impressions['OCCURRED_AT'] = pd.to_datetime(impressions['OCCURRED_AT'])
            impressions['week'] = impressions['OCCURRED_AT'].dt.to_period('W').apply(lambda x: x.start_time)
            log(f"  impressions: {len(impressions):,} rows", f)
        else:
            log("  impressions not found", f)
            impressions = pd.DataFrame()

        # Clicks
        clicks_path = RAW_DATA_DIR / "raw_sample_clicks.parquet"
        if clicks_path.exists():
            clicks = pd.read_parquet(clicks_path)
            clicks['OCCURRED_AT'] = pd.to_datetime(clicks['OCCURRED_AT'])
            clicks['week'] = clicks['OCCURRED_AT'].dt.to_period('W').apply(lambda x: x.start_time)
            log(f"  clicks: {len(clicks):,} rows", f)
        else:
            log("  clicks not found", f)
            clicks = pd.DataFrame()

        # Catalog (for product prices)
        catalog_path = RAW_DATA_DIR / "processed_sample_catalog.parquet"
        if catalog_path.exists():
            catalog = pd.read_parquet(catalog_path)
            log(f"  catalog: {len(catalog):,} rows", f)
        else:
            log("  catalog not found", f)
            catalog = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Compute pre-treatment covariates
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("COMPUTING PRE-TREATMENT COVARIATES", f)
        log("-" * 40, f)

        log("", f)
        log("COVARIATE DEFINITIONS:", f)
        log("  SIZE/ACTIVITY:", f)
        log("    pre_auction_count: Unique auctions participated in", f)
        log("    pre_bid_count: Total bids submitted", f)
        log("    pre_win_count: Auction wins", f)
        log("    pre_win_rate: pre_win_count / pre_bid_count", f)
        log("    pre_impression_count: Total impressions received", f)
        log("    pre_click_count: Total clicks received", f)
        log("    pre_weeks_active: Weeks with any auction activity", f)
        log("  ENGAGEMENT:", f)
        log("    pre_ctr: pre_click_count / pre_impression_count", f)
        log("    pre_avg_ranking: Mean auction ranking (lower = better)", f)
        log("  PRODUCT MIX:", f)
        log("    pre_avg_price_point: Mean price of advertised products", f)
        log("    pre_unique_products: Count of unique products advertised", f)
        log("    pre_unique_categories: Count of unique categories", f)
        log("    pre_category_hhi: Herfindahl-Hirschman concentration index", f)
        log("", f)

        # Prepare vendor list
        all_vendors = vendor_cohort['VENDOR_ID'].unique()
        log(f"  Processing {len(all_vendors):,} vendors...", f)
        log("", f)

        # Create lookup for cohort week
        cohort_lookup = vendor_cohort.set_index('VENDOR_ID')['cohort_week'].to_dict()

        # Compute covariates from auctions
        covariate_records = []

        if len(auctions) > 0:
            log("  Computing auction-based covariates...", f)

            # Group auctions by vendor for efficiency
            vendor_groups = auctions.groupby('VENDOR_ID')

            for vendor_id in tqdm(all_vendors, desc="Processing vendors"):
                cohort_week = cohort_lookup.get(vendor_id)

                # Get vendor's auction data
                if vendor_id in vendor_groups.groups:
                    vendor_auctions = vendor_groups.get_group(vendor_id)

                    # Filter to pre-treatment period
                    if pd.notna(cohort_week):
                        vendor_auctions = vendor_auctions[vendor_auctions['CREATED_AT'] < cohort_week]

                    if len(vendor_auctions) == 0:
                        # No pre-treatment data
                        covariate_records.append({
                            'VENDOR_ID': vendor_id,
                            'pre_auction_count': 0,
                            'pre_bid_count': 0,
                            'pre_win_count': 0,
                            'pre_win_rate': np.nan,
                            'pre_weeks_active': 0,
                            'pre_avg_ranking': np.nan,
                            'pre_unique_products': 0,
                        })
                        continue

                    # Compute auction covariates
                    pre_auction_count = vendor_auctions['AUCTION_ID'].nunique()
                    pre_bid_count = len(vendor_auctions)
                    pre_win_count = vendor_auctions['IS_WINNER'].sum() if 'IS_WINNER' in vendor_auctions.columns else 0
                    pre_win_rate = pre_win_count / pre_bid_count if pre_bid_count > 0 else np.nan
                    pre_weeks_active = vendor_auctions['week'].nunique()
                    pre_avg_ranking = vendor_auctions['RANKING'].mean() if 'RANKING' in vendor_auctions.columns else np.nan
                    pre_unique_products = vendor_auctions['PRODUCT_ID'].nunique()

                    covariate_records.append({
                        'VENDOR_ID': vendor_id,
                        'pre_auction_count': pre_auction_count,
                        'pre_bid_count': pre_bid_count,
                        'pre_win_count': pre_win_count,
                        'pre_win_rate': pre_win_rate,
                        'pre_weeks_active': pre_weeks_active,
                        'pre_avg_ranking': pre_avg_ranking,
                        'pre_unique_products': pre_unique_products,
                    })
                else:
                    # Vendor not in auctions
                    covariate_records.append({
                        'VENDOR_ID': vendor_id,
                        'pre_auction_count': 0,
                        'pre_bid_count': 0,
                        'pre_win_count': 0,
                        'pre_win_rate': np.nan,
                        'pre_weeks_active': 0,
                        'pre_avg_ranking': np.nan,
                        'pre_unique_products': 0,
                    })

        covariates_df = pd.DataFrame(covariate_records)
        log(f"  Auction covariates computed for {len(covariates_df):,} vendors", f)
        log("", f)

        # Add impression/click covariates
        if len(impressions) > 0:
            log("  Computing impression-based covariates...", f)

            imp_records = []
            imp_groups = impressions.groupby('VENDOR_ID')

            for vendor_id in tqdm(all_vendors, desc="Processing impressions"):
                cohort_week = cohort_lookup.get(vendor_id)

                if vendor_id in imp_groups.groups:
                    vendor_imps = imp_groups.get_group(vendor_id)

                    if pd.notna(cohort_week):
                        vendor_imps = vendor_imps[vendor_imps['OCCURRED_AT'] < cohort_week]

                    imp_records.append({
                        'VENDOR_ID': vendor_id,
                        'pre_impression_count': len(vendor_imps),
                    })
                else:
                    imp_records.append({
                        'VENDOR_ID': vendor_id,
                        'pre_impression_count': 0,
                    })

            imp_df = pd.DataFrame(imp_records)
            covariates_df = covariates_df.merge(imp_df, on='VENDOR_ID', how='left')
            log(f"  Impression covariates merged", f)

        else:
            covariates_df['pre_impression_count'] = 0

        if len(clicks) > 0:
            log("  Computing click-based covariates...", f)

            click_records = []
            click_groups = clicks.groupby('VENDOR_ID')

            for vendor_id in tqdm(all_vendors, desc="Processing clicks"):
                cohort_week = cohort_lookup.get(vendor_id)

                if vendor_id in click_groups.groups:
                    vendor_clicks = click_groups.get_group(vendor_id)

                    if pd.notna(cohort_week):
                        vendor_clicks = vendor_clicks[vendor_clicks['OCCURRED_AT'] < cohort_week]

                    click_records.append({
                        'VENDOR_ID': vendor_id,
                        'pre_click_count': len(vendor_clicks),
                    })
                else:
                    click_records.append({
                        'VENDOR_ID': vendor_id,
                        'pre_click_count': 0,
                    })

            click_df = pd.DataFrame(click_records)
            covariates_df = covariates_df.merge(click_df, on='VENDOR_ID', how='left')
            log(f"  Click covariates merged", f)

        else:
            covariates_df['pre_click_count'] = 0

        log("", f)

        # Compute derived engagement metrics
        log("  Computing derived metrics...", f)

        covariates_df['pre_ctr'] = np.where(
            covariates_df['pre_impression_count'] > 0,
            covariates_df['pre_click_count'] / covariates_df['pre_impression_count'],
            np.nan
        )

        log("", f)

        # Add product price/category covariates if catalog available
        if len(catalog) > 0 and len(auctions) > 0:
            log("  Computing product mix covariates from catalog...", f)

            # Get price lookup
            if 'PRICE' in catalog.columns:
                price_lookup = catalog.set_index('PRODUCT_ID')['PRICE'].to_dict()
            else:
                price_lookup = {}

            # Get category lookup (first category if array)
            if 'CATEGORIES' in catalog.columns:
                def get_first_category(cats):
                    if isinstance(cats, list) and len(cats) > 0:
                        return cats[0]
                    return None
                category_lookup = catalog.set_index('PRODUCT_ID')['CATEGORIES'].apply(get_first_category).to_dict()
            else:
                category_lookup = {}

            price_records = []
            category_records = []

            vendor_groups = auctions.groupby('VENDOR_ID')

            for vendor_id in tqdm(all_vendors, desc="Processing product mix"):
                cohort_week = cohort_lookup.get(vendor_id)

                if vendor_id in vendor_groups.groups:
                    vendor_auctions = vendor_groups.get_group(vendor_id)

                    if pd.notna(cohort_week):
                        vendor_auctions = vendor_auctions[vendor_auctions['CREATED_AT'] < cohort_week]

                    if len(vendor_auctions) > 0:
                        products = vendor_auctions['PRODUCT_ID'].unique()

                        # Get prices
                        prices = [price_lookup.get(p) for p in products if p in price_lookup]
                        prices = [p for p in prices if p is not None and not np.isnan(p)]
                        avg_price = np.mean(prices) if len(prices) > 0 else np.nan

                        # Get categories
                        categories = [category_lookup.get(p) for p in products if p in category_lookup]
                        categories = [c for c in categories if c is not None]
                        unique_categories = len(set(categories))

                        # Compute HHI (category concentration)
                        if len(categories) > 0:
                            from collections import Counter
                            cat_counts = Counter(categories)
                            total = sum(cat_counts.values())
                            hhi = sum((count / total) ** 2 for count in cat_counts.values())
                        else:
                            hhi = np.nan

                        price_records.append({
                            'VENDOR_ID': vendor_id,
                            'pre_avg_price_point': avg_price,
                            'pre_unique_categories': unique_categories,
                            'pre_category_hhi': hhi,
                        })
                    else:
                        price_records.append({
                            'VENDOR_ID': vendor_id,
                            'pre_avg_price_point': np.nan,
                            'pre_unique_categories': 0,
                            'pre_category_hhi': np.nan,
                        })
                else:
                    price_records.append({
                        'VENDOR_ID': vendor_id,
                        'pre_avg_price_point': np.nan,
                        'pre_unique_categories': 0,
                        'pre_category_hhi': np.nan,
                    })

            price_df = pd.DataFrame(price_records)
            covariates_df = covariates_df.merge(price_df, on='VENDOR_ID', how='left')
            log(f"  Product mix covariates merged", f)

        else:
            covariates_df['pre_avg_price_point'] = np.nan
            covariates_df['pre_unique_categories'] = 0
            covariates_df['pre_category_hhi'] = np.nan

        log("", f)

        # Add cohort information
        covariates_df = covariates_df.merge(vendor_cohort, on='VENDOR_ID', how='left')
        covariates_df['is_treated'] = covariates_df['cohort_week'].notna().astype(int)

        log("", f)

        # -----------------------------------------------------------------
        # Summary statistics
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("COVARIATE SUMMARY STATISTICS", f)
        log("-" * 40, f)
        log("", f)

        numeric_cols = [
            'pre_auction_count', 'pre_bid_count', 'pre_win_count', 'pre_win_rate',
            'pre_impression_count', 'pre_click_count', 'pre_weeks_active',
            'pre_ctr', 'pre_avg_ranking', 'pre_avg_price_point',
            'pre_unique_products', 'pre_unique_categories', 'pre_category_hhi'
        ]

        log(f"  {'Covariate':<25} {'N':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Median':<12} {'Max':<12}", f)
        log(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

        for col in numeric_cols:
            if col in covariates_df.columns:
                n_obs = covariates_df[col].notna().sum()
                mean_val = covariates_df[col].mean()
                std_val = covariates_df[col].std()
                min_val = covariates_df[col].min()
                med_val = covariates_df[col].median()
                max_val = covariates_df[col].max()

                log(f"  {col:<25} {n_obs:<10,} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {med_val:<12.4f} {max_val:<12.4f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Compare treated vs never-treated
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TREATED VS NEVER-TREATED COMPARISON", f)
        log("-" * 40, f)
        log("", f)

        treated = covariates_df[covariates_df['is_treated'] == 1]
        never_treated = covariates_df[covariates_df['is_treated'] == 0]

        log(f"  Treated vendors: {len(treated):,}", f)
        log(f"  Never-treated vendors: {len(never_treated):,}", f)
        log("", f)

        log(f"  {'Covariate':<25} {'Treated Mean':<15} {'Never-Treated Mean':<20} {'Diff':<12}", f)
        log(f"  {'-'*25} {'-'*15} {'-'*20} {'-'*12}", f)

        for col in numeric_cols:
            if col in covariates_df.columns:
                treated_mean = treated[col].mean()
                never_treated_mean = never_treated[col].mean()
                diff = treated_mean - never_treated_mean

                log(f"  {col:<25} {treated_mean:<15.4f} {never_treated_mean:<20.4f} {diff:<12.4f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Quantile distributions for key covariates
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("QUANTILE DISTRIBUTIONS", f)
        log("-" * 40, f)
        log("", f)

        key_covariates = ['pre_auction_count', 'pre_avg_price_point', 'pre_win_rate', 'pre_weeks_active']

        for col in key_covariates:
            if col in covariates_df.columns and covariates_df[col].notna().sum() > 0:
                log(f"  {col}:", f)
                quantiles = covariates_df[col].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
                for q, val in quantiles.items():
                    log(f"    P{int(q*100):02d}: {val:,.4f}", f)
                log("", f)

        # -----------------------------------------------------------------
        # Save covariates
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SAVING COVARIATES", f)
        log("-" * 40, f)

        output_path = DATA_DIR / "vendor_covariates.parquet"
        covariates_df.to_parquet(output_path, index=False)

        log(f"  Output shape: {covariates_df.shape}", f)
        log(f"  Columns: {list(covariates_df.columns)}", f)
        log(f"  Saved to: {output_path}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Missing data summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("MISSING DATA SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        for col in covariates_df.columns:
            if col != 'VENDOR_ID':
                n_missing = covariates_df[col].isna().sum()
                pct_missing = n_missing / len(covariates_df) * 100
                log(f"  {col:<25}: {n_missing:,} missing ({pct_missing:.2f}%)", f)

        log("", f)
        log("=" * 80, f)
        log("PRE-TREATMENT COVARIATE CONSTRUCTION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
