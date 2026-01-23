#!/usr/bin/env python3
"""
01_data_audit.py
Validates data integrity, join rates, and purchase mappability before panel construction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = Path("/Users/pranjal/Code/marketplace-incrementality/shopping-sessions/data")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "01_data_audit.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("01_DATA_AUDIT", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script validates the raw data before analysis begins. We check whether", f)
        log("the user identifiers are consistent across tables (OPAQUE_USER_ID vs USER_ID),", f)
        log("whether the composite key joins (AUCTION_ID + PRODUCT_ID + VENDOR_ID + CAMPAIGN_ID)", f)
        log("achieve 100% match rates, and what fraction of total spend can be attributed to", f)
        log("promoted clicks. Low mappability would limit the scope of causal inference.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # Load source tables
        log("Loading source tables...", f)
        tables = {}
        files = [
            ('auctions_users', 'raw_sample_auctions_users.parquet'),
            ('auctions_results', 'raw_sample_auctions_results.parquet'),
            ('impressions', 'raw_sample_impressions.parquet'),
            ('clicks', 'raw_sample_clicks.parquet'),
            ('purchases', 'raw_sample_purchases.parquet'),
            ('catalog', 'processed_sample_catalog.parquet'),
        ]

        for name, filename in tqdm(files, desc="Loading"):
            path = SOURCE_DIR / filename
            if path.exists():
                tables[name] = pd.read_parquet(path)
                log(f"  {name}: {len(tables[name]):,} rows, {tables[name].shape[1]} cols", f)
            else:
                log(f"  {name}: FILE NOT FOUND at {path}", f)

        log(f"\nLoaded {len(tables)} tables", f)
        log("", f)

        # Table Summary
        log("=" * 80, f)
        log("TABLE SUMMARY", f)
        log("=" * 80, f)

        for name, df in tables.items():
            log(f"\n--- {name.upper()} ---", f)
            log(f"Rows: {len(df):,}", f)
            log(f"Columns: {list(df.columns)}", f)
            log(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB", f)

            ts_cols = [c for c in df.columns if 'AT' in c.upper() or 'TIME' in c.upper()]
            for col in ts_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                    log(f"{col}: {df[col].min()} to {df[col].max()}", f)
                except:
                    pass

        # Unique Entity Counts
        log("", f)
        log("=" * 80, f)
        log("UNIQUE ENTITY COUNTS", f)
        log("=" * 80, f)

        user_cols = {'auctions_users': 'OPAQUE_USER_ID', 'impressions': 'USER_ID',
                     'clicks': 'USER_ID', 'purchases': 'USER_ID'}
        log("\nUnique Users by Table:", f)
        for name, col in user_cols.items():
            if name in tables and col in tables[name].columns:
                n = tables[name][col].nunique()
                log(f"  {name}: {n:,} unique users", f)

        log("\nUnique Vendors by Table:", f)
        for name in ['auctions_results', 'impressions', 'clicks']:
            if name in tables and 'VENDOR_ID' in tables[name].columns:
                n = tables[name]['VENDOR_ID'].nunique()
                log(f"  {name}: {n:,} unique vendors", f)

        log("\nUnique Products by Table:", f)
        for name in ['auctions_results', 'impressions', 'clicks', 'purchases', 'catalog']:
            if name in tables and 'PRODUCT_ID' in tables[name].columns:
                n = tables[name]['PRODUCT_ID'].nunique()
                log(f"  {name}: {n:,} unique products", f)

        log("\nUnique Auctions by Table:", f)
        for name in ['auctions_users', 'auctions_results', 'impressions', 'clicks']:
            if name in tables and 'AUCTION_ID' in tables[name].columns:
                n = tables[name]['AUCTION_ID'].nunique()
                log(f"  {name}: {n:,} unique auctions", f)

        # User ID Coherence Check
        log("", f)
        log("=" * 80, f)
        log("USER ID COHERENCE CHECK", f)
        log("=" * 80, f)

        if 'auctions_users' in tables and 'clicks' in tables:
            auctions = tables['auctions_users']
            clicks = tables['clicks']

            opaque_users = set(auctions['OPAQUE_USER_ID'].unique())
            click_users = set(clicks['USER_ID'].unique())

            overlap = opaque_users & click_users
            only_opaque = opaque_users - click_users
            only_click = click_users - opaque_users

            log(f"OPAQUE_USER_ID unique values: {len(opaque_users):,}", f)
            log(f"USER_ID unique values: {len(click_users):,}", f)
            log(f"Overlap (same IDs in both): {len(overlap):,}", f)
            log(f"Only in OPAQUE_USER_ID: {len(only_opaque):,}", f)
            log(f"Only in USER_ID: {len(only_click):,}", f)

            overlap_rate = len(overlap) / min(len(opaque_users), len(click_users)) * 100 if min(len(opaque_users), len(click_users)) > 0 else 0
            log(f"\nOverlap rate: {overlap_rate:.1f}%", f)

        # Join Rate Diagnostics
        log("", f)
        log("=" * 80, f)
        log("JOIN RATE DIAGNOSTICS", f)
        log("=" * 80, f)

        COMPOSITE_KEYS = ['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID']

        log("\nKey availability by table:", f)
        for name, df in tables.items():
            available = [k for k in COMPOSITE_KEYS if k in df.columns]
            log(f"  {name}: {available}", f)

        # Clicks -> Auctions Results
        log("\n--- CLICKS -> AUCTIONS_RESULTS ---", f)
        if 'clicks' in tables and 'auctions_results' in tables:
            clicks = tables['clicks']
            bids = tables['auctions_results']

            common_keys = [k for k in COMPOSITE_KEYS if k in clicks.columns and k in bids.columns]
            log(f"Common keys: {common_keys}", f)

            clicks_keys = clicks[common_keys].drop_duplicates()
            bids_keys = bids[common_keys].drop_duplicates()

            merged = clicks_keys.merge(bids_keys, on=common_keys, how='inner')
            forward_rate = len(merged) / len(clicks_keys) * 100 if len(clicks_keys) > 0 else 0
            log(f"Clicks with matching bids: {len(merged):,} / {len(clicks_keys):,} ({forward_rate:.1f}%)", f)

        # Purchase Mappability
        log("", f)
        log("=" * 80, f)
        log("PURCHASE MAPPABILITY", f)
        log("=" * 80, f)

        if 'purchases' in tables and 'clicks' in tables:
            purchases = tables['purchases'].copy()
            clicks = tables['clicks'].copy()

            log(f"\nTotal purchases: {len(purchases):,}", f)
            log(f"Total clicks: {len(clicks):,}", f)

            purchases['PURCHASED_AT'] = pd.to_datetime(purchases['PURCHASED_AT'])
            clicks['OCCURRED_AT'] = pd.to_datetime(clicks['OCCURRED_AT'])

            purchases['spend'] = purchases['QUANTITY'] * purchases['UNIT_PRICE'] / 100
            total_spend = purchases['spend'].sum()
            log(f"Total spend: ${total_spend:,.2f}", f)

            click_user_product = clicks[['USER_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates()
            log(f"\nUnique (user, product) pairs in clicks: {len(click_user_product):,}", f)

            purchases_with_click = purchases.merge(
                click_user_product,
                on=['USER_ID', 'PRODUCT_ID'],
                how='inner'
            )

            n_mapped = len(purchases_with_click)
            spend_mapped = purchases_with_click['spend'].sum()

            log(f"\n--- Mappability (user + product match) ---", f)
            log(f"Purchases mappable: {n_mapped:,} / {len(purchases):,} ({n_mapped/len(purchases)*100:.1f}%)", f)
            log(f"Spend mappable: ${spend_mapped:,.2f} / ${total_spend:,.2f} ({spend_mapped/total_spend*100:.1f}%)", f)

            # Time-constrained mappability
            log("\n--- Mappability with time constraint ---", f)
            first_clicks = clicks.groupby(['USER_ID', 'PRODUCT_ID'])['OCCURRED_AT'].min().reset_index()
            first_clicks.columns = ['USER_ID', 'PRODUCT_ID', 'first_click_time']

            purchases_timed = purchases.merge(first_clicks, on=['USER_ID', 'PRODUCT_ID'], how='left')

            for window_days in [0, 1, 7, 14, 30]:
                if window_days == 0:
                    mask = purchases_timed['first_click_time'].notna() & \
                           (purchases_timed['PURCHASED_AT'].dt.date == purchases_timed['first_click_time'].dt.date)
                    label = "Same day"
                else:
                    mask = purchases_timed['first_click_time'].notna() & \
                           (purchases_timed['PURCHASED_AT'] >= purchases_timed['first_click_time']) & \
                           (purchases_timed['PURCHASED_AT'] <= purchases_timed['first_click_time'] + pd.Timedelta(days=window_days))
                    label = f"Within {window_days}d"

                n = mask.sum()
                spend = purchases_timed.loc[mask, 'spend'].sum()
                log(f"  {label}: {n:,} purchases ({n/len(purchases)*100:.1f}%), ${spend:,.2f} ({spend/total_spend*100:.1f}%)", f)

        # Summary
        log("", f)
        log("=" * 80, f)
        log("SUMMARY", f)
        log("=" * 80, f)

        if 'clicks' in tables and 'purchases' in tables:
            log(f"\nUsers in clicks: {clicks['USER_ID'].nunique():,}", f)
            log(f"Users in purchases: {purchases['USER_ID'].nunique():,}", f)
            log(f"Vendors: {tables['auctions_results']['VENDOR_ID'].nunique():,}", f)
            log(f"Total clicks: {len(clicks):,}", f)
            log(f"Total purchases: {len(purchases):,}", f)
            log(f"Total spend: ${total_spend:,.2f}", f)
            log(f"Mappable spend: ${spend_mapped:,.2f} ({spend_mapped/total_spend*100:.1f}%)", f)

        # Date range
        log("\n--- Date Range ---", f)
        if 'clicks' in tables:
            log(f"Clicks: {clicks['OCCURRED_AT'].min()} to {clicks['OCCURRED_AT'].max()}", f)
        if 'purchases' in tables:
            log(f"Purchases: {purchases['PURCHASED_AT'].min()} to {purchases['PURCHASED_AT'].max()}", f)

        log("", f)
        log("=" * 80, f)
        log("AUDIT COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
