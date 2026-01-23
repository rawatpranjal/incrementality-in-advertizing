#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Data Exploration & Cohort Discovery
Profiles raw data, identifies treatment cohorts, and summarizes cohort sizes.
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

OUTPUT_FILE = RESULTS_DIR / "00_explore_cohorts.txt"

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
        log("STAGGERED ADOPTION: DATA EXPLORATION & COHORT DISCOVERY", f)
        log("=" * 80, f)
        log("", f)

        log("RESEARCH HYPOTHESES:", f)
        log("  1. Does ad adoption cause incremental GMV lift? (Staggered DiD)", f)
        log("  2. What is the elasticity of GMV w.r.t. Ad Spend? (2SLS)", f)
        log("  3. Do promoted sales cannibalize organic sales?", f)
        log("  4. Are there dynamic effects (ramp-up, decay) after adoption?", f)
        log("  5. Do pre-trends suggest selection bias?", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load raw data
        # -----------------------------------------------------------------
        log("LOADING RAW DATA", f)
        log("-" * 40, f)

        files_to_load = {
            'auctions_users': 'raw_sample_auctions_users.parquet',
            'auctions_results': 'raw_sample_auctions_results.parquet',
            'impressions': 'raw_sample_impressions.parquet',
            'clicks': 'raw_sample_clicks.parquet',
            'purchases': 'raw_sample_purchases.parquet',
        }

        tables = {}
        for name, filename in tqdm(files_to_load.items(), desc="Loading tables"):
            path = RAW_DATA_DIR / filename
            if path.exists():
                tables[name] = pd.read_parquet(path)
                log(f"  {name}: {len(tables[name]):,} rows, {len(tables[name].columns)} cols", f)
            else:
                log(f"  {name}: FILE NOT FOUND at {path}", f)
                tables[name] = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Profile each table
        # -----------------------------------------------------------------
        log("TABLE SCHEMAS", f)
        log("-" * 40, f)

        for name, df in tables.items():
            if len(df) > 0:
                log(f"\n{name.upper()}", f)
                log(f"  Shape: {df.shape}", f)
                log(f"  Columns: {list(df.columns)}", f)
                log(f"  Dtypes:", f)
                for col, dtype in df.dtypes.items():
                    log(f"    {col}: {dtype}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Date ranges
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("DATE RANGES", f)
        log("-" * 40, f)

        date_cols = {
            'auctions_users': 'CREATED_AT',
            'auctions_results': 'CREATED_AT',
            'impressions': 'OCCURRED_AT',
            'clicks': 'OCCURRED_AT',
            'purchases': 'PURCHASED_AT',
        }

        for name, col in date_cols.items():
            df = tables[name]
            if len(df) > 0 and col in df.columns:
                df[col] = pd.to_datetime(df[col])
                min_dt = df[col].min()
                max_dt = df[col].max()
                days = (max_dt - min_dt).days + 1
                log(f"  {name}: {min_dt.date()} to {max_dt.date()} ({days} days)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Vendor analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR ANALYSIS", f)
        log("-" * 40, f)

        auctions = tables['auctions_results']

        if len(auctions) > 0:
            n_vendors = auctions['VENDOR_ID'].nunique()
            n_campaigns = auctions['CAMPAIGN_ID'].nunique()
            n_products = auctions['PRODUCT_ID'].nunique()
            n_auctions = auctions['AUCTION_ID'].nunique()

            log(f"  Unique vendors: {n_vendors:,}", f)
            log(f"  Unique campaigns: {n_campaigns:,}", f)
            log(f"  Unique products: {n_products:,}", f)
            log(f"  Unique auctions: {n_auctions:,}", f)
            log("", f)

            # Auction wins
            if 'IS_WINNER' in auctions.columns:
                n_winners = auctions[auctions['IS_WINNER'] == True]['AUCTION_ID'].nunique()
                win_rate = auctions['IS_WINNER'].mean() * 100
                log(f"  Auctions with winners: {n_winners:,}", f)
                log(f"  Overall win rate: {win_rate:.2f}%", f)

            log("", f)

        # -----------------------------------------------------------------
        # Cohort discovery: First auction participation per vendor
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("COHORT DISCOVERY (Treatment Timing)", f)
        log("-" * 40, f)

        if len(auctions) > 0 and 'CREATED_AT' in auctions.columns:
            auctions['CREATED_AT'] = pd.to_datetime(auctions['CREATED_AT'])
            auctions['date'] = auctions['CREATED_AT'].dt.date
            auctions['week'] = auctions['CREATED_AT'].dt.to_period('W').apply(lambda x: x.start_time.date())

            # First participation per vendor
            first_auction = auctions.groupby('VENDOR_ID')['CREATED_AT'].min().reset_index()
            first_auction.columns = ['VENDOR_ID', 'first_auction_time']
            first_auction['first_auction_date'] = first_auction['first_auction_time'].dt.date
            first_auction['first_auction_week'] = first_auction['first_auction_time'].dt.to_period('W').apply(lambda x: x.start_time.date())

            log(f"  Vendors with auction history: {len(first_auction):,}", f)
            log("", f)

            # Cohort sizes by week
            cohort_sizes = first_auction.groupby('first_auction_week').size().reset_index(name='n_vendors')
            cohort_sizes = cohort_sizes.sort_values('first_auction_week')

            log("  COHORT SIZES BY WEEK (first auction participation):", f)
            log(f"  {'Week':<15} {'N Vendors':<12} {'Cumulative':<12}", f)
            log(f"  {'-'*15} {'-'*12} {'-'*12}", f)

            cumulative = 0
            for _, row in cohort_sizes.iterrows():
                cumulative += row['n_vendors']
                log(f"  {str(row['first_auction_week']):<15} {row['n_vendors']:<12,} {cumulative:<12,}", f)

            log("", f)

            # Cohort sizes by day
            cohort_sizes_daily = first_auction.groupby('first_auction_date').size().reset_index(name='n_vendors')
            cohort_sizes_daily = cohort_sizes_daily.sort_values('first_auction_date')

            log("  COHORT SIZES BY DAY (first auction participation):", f)
            log(f"  {'Date':<15} {'N Vendors':<12}", f)
            log(f"  {'-'*15} {'-'*12}", f)

            for _, row in cohort_sizes_daily.iterrows():
                log(f"  {str(row['first_auction_date']):<15} {row['n_vendors']:<12,}", f)

            log("", f)

            # Save cohort data for downstream scripts
            first_auction.to_parquet(DATA_DIR / "vendor_first_auction.parquet", index=False)
            log(f"  Saved vendor first auction data to {DATA_DIR / 'vendor_first_auction.parquet'}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Spending analysis (if FINAL_BID available)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SPENDING ANALYSIS", f)
        log("-" * 40, f)

        if len(auctions) > 0 and 'FINAL_BID' in auctions.columns and 'IS_WINNER' in auctions.columns:
            # Only count winning bids as spend
            winning_bids = auctions[auctions['IS_WINNER'] == True]

            if len(winning_bids) > 0:
                total_spend = winning_bids['FINAL_BID'].sum()
                avg_bid = winning_bids['FINAL_BID'].mean()
                median_bid = winning_bids['FINAL_BID'].median()

                log(f"  Total spend (winning bids): {total_spend:,.2f}", f)
                log(f"  Average winning bid: {avg_bid:,.4f}", f)
                log(f"  Median winning bid: {median_bid:,.4f}", f)
                log("", f)

                # Spend by vendor
                vendor_spend = winning_bids.groupby('VENDOR_ID')['FINAL_BID'].sum().reset_index()
                vendor_spend.columns = ['VENDOR_ID', 'total_spend']

                log(f"  Vendors with spend: {len(vendor_spend):,}", f)
                log(f"  Spend distribution:", f)
                log(f"    Min:    {vendor_spend['total_spend'].min():,.4f}", f)
                log(f"    25%:    {vendor_spend['total_spend'].quantile(0.25):,.4f}", f)
                log(f"    50%:    {vendor_spend['total_spend'].quantile(0.50):,.4f}", f)
                log(f"    75%:    {vendor_spend['total_spend'].quantile(0.75):,.4f}", f)
                log(f"    90%:    {vendor_spend['total_spend'].quantile(0.90):,.4f}", f)
                log(f"    99%:    {vendor_spend['total_spend'].quantile(0.99):,.4f}", f)
                log(f"    Max:    {vendor_spend['total_spend'].max():,.4f}", f)
        else:
            log("  FINAL_BID or IS_WINNER column not available", f)

        log("", f)

        # -----------------------------------------------------------------
        # Click and purchase summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("FUNNEL SUMMARY", f)
        log("-" * 40, f)

        clicks = tables['clicks']
        purchases = tables['purchases']
        impressions = tables['impressions']

        if len(impressions) > 0:
            log(f"  Impressions: {len(impressions):,}", f)
            log(f"    Unique users: {impressions['USER_ID'].nunique():,}", f)
            log(f"    Unique vendors: {impressions['VENDOR_ID'].nunique():,}", f)

        if len(clicks) > 0:
            log(f"  Clicks: {len(clicks):,}", f)
            log(f"    Unique users: {clicks['USER_ID'].nunique():,}", f)
            log(f"    Unique vendors: {clicks['VENDOR_ID'].nunique():,}", f)

        if len(purchases) > 0:
            log(f"  Purchases: {len(purchases):,}", f)
            log(f"    Unique users: {purchases['USER_ID'].nunique():,}", f)
            if 'UNIT_PRICE' in purchases.columns and 'QUANTITY' in purchases.columns:
                total_gmv = (purchases['UNIT_PRICE'] * purchases['QUANTITY']).sum()
                log(f"    Total GMV: {total_gmv:,.2f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Data quality checks
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("DATA QUALITY CHECKS", f)
        log("-" * 40, f)

        # Check for nulls in key columns
        key_cols = {
            'auctions_results': ['AUCTION_ID', 'VENDOR_ID', 'PRODUCT_ID', 'CREATED_AT'],
            'clicks': ['AUCTION_ID', 'USER_ID', 'VENDOR_ID', 'PRODUCT_ID', 'OCCURRED_AT'],
            'purchases': ['USER_ID', 'PRODUCT_ID', 'PURCHASED_AT'],
        }

        for table_name, cols in key_cols.items():
            df = tables[table_name]
            if len(df) > 0:
                log(f"\n  {table_name.upper()} null check:", f)
                for col in cols:
                    if col in df.columns:
                        null_pct = df[col].isnull().mean() * 100
                        log(f"    {col}: {null_pct:.2f}% null", f)

        log("", f)
        log("=" * 80, f)
        log("EXPLORATION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
