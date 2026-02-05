#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Data Audit
Validates composite key joins, checks orphan rates, and reports data quality.
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

OUTPUT_FILE = RESULTS_DIR / "01_data_audit.txt"

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
        log("STAGGERED ADOPTION: DATA AUDIT", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Validate composite key joins between auction funnel tables.", f)
        log("  Check orphan rates (clicks without auctions, etc.).", f)
        log("  Identify data quality issues that may affect DiD estimation.", f)
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
                log(f"  {name}: {len(tables[name]):,} rows", f)
            else:
                log(f"  {name}: FILE NOT FOUND", f)
                tables[name] = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Audit 1: Auction Users <-> Auction Results join
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("AUDIT 1: AUCTIONS_USERS <-> AUCTIONS_RESULTS JOIN", f)
        log("-" * 40, f)
        log("  Join key: AUCTION_ID", f)
        log("", f)

        au = tables['auctions_users']
        ar = tables['auctions_results']

        if len(au) > 0 and len(ar) > 0:
            au_auction_ids = set(au['AUCTION_ID'].unique())
            ar_auction_ids = set(ar['AUCTION_ID'].unique())

            in_both = au_auction_ids & ar_auction_ids
            only_in_users = au_auction_ids - ar_auction_ids
            only_in_results = ar_auction_ids - au_auction_ids

            log(f"  Unique auctions in USERS:   {len(au_auction_ids):,}", f)
            log(f"  Unique auctions in RESULTS: {len(ar_auction_ids):,}", f)
            log(f"  In both:                    {len(in_both):,}", f)
            log(f"  Only in USERS (orphans):    {len(only_in_users):,} ({len(only_in_users)/len(au_auction_ids)*100:.2f}%)", f)
            log(f"  Only in RESULTS (orphans):  {len(only_in_results):,} ({len(only_in_results)/len(ar_auction_ids)*100:.2f}%)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Audit 2: Clicks <-> Auctions join
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("AUDIT 2: CLICKS <-> AUCTIONS_RESULTS JOIN", f)
        log("-" * 40, f)
        log("  Join key: (AUCTION_ID, PRODUCT_ID, VENDOR_ID)", f)
        log("", f)

        clicks = tables['clicks']

        if len(clicks) > 0 and len(ar) > 0:
            # Create composite keys
            clicks['composite_key'] = clicks['AUCTION_ID'].astype(str) + '_' + \
                                       clicks['PRODUCT_ID'].astype(str) + '_' + \
                                       clicks['VENDOR_ID'].astype(str)
            ar['composite_key'] = ar['AUCTION_ID'].astype(str) + '_' + \
                                  ar['PRODUCT_ID'].astype(str) + '_' + \
                                  ar['VENDOR_ID'].astype(str)

            click_keys = set(clicks['composite_key'].unique())
            ar_keys = set(ar['composite_key'].unique())

            in_both = click_keys & ar_keys
            orphan_clicks = click_keys - ar_keys

            log(f"  Unique click composite keys:   {len(click_keys):,}", f)
            log(f"  Unique auction composite keys: {len(ar_keys):,}", f)
            log(f"  Clicks matched to auctions:    {len(in_both):,} ({len(in_both)/len(click_keys)*100:.2f}%)", f)
            log(f"  Orphan clicks (no auction):    {len(orphan_clicks):,} ({len(orphan_clicks)/len(click_keys)*100:.2f}%)", f)

            # Simpler join on just AUCTION_ID
            log("", f)
            log("  Alternative join on AUCTION_ID only:", f)
            click_auction_ids = set(clicks['AUCTION_ID'].unique())
            ar_auction_ids = set(ar['AUCTION_ID'].unique())
            matched = click_auction_ids & ar_auction_ids
            log(f"    Clicks with matching auction: {len(matched):,} / {len(click_auction_ids):,} ({len(matched)/len(click_auction_ids)*100:.2f}%)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Audit 3: Impressions <-> Auctions join
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("AUDIT 3: IMPRESSIONS <-> AUCTIONS_RESULTS JOIN", f)
        log("-" * 40, f)

        impressions = tables['impressions']

        if len(impressions) > 0 and len(ar) > 0:
            imp_auction_ids = set(impressions['AUCTION_ID'].unique())
            matched = imp_auction_ids & ar_auction_ids
            orphan_imps = imp_auction_ids - ar_auction_ids

            log(f"  Unique impression auctions:    {len(imp_auction_ids):,}", f)
            log(f"  Matched to auction results:    {len(matched):,} ({len(matched)/len(imp_auction_ids)*100:.2f}%)", f)
            log(f"  Orphan impressions (no auction): {len(orphan_imps):,} ({len(orphan_imps)/len(imp_auction_ids)*100:.2f}%)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Audit 4: Purchases attribution check
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("AUDIT 4: PURCHASE ATTRIBUTION FEASIBILITY", f)
        log("-" * 40, f)
        log("  Attribution: PURCHASES -> CLICKS on (USER_ID, PRODUCT_ID)", f)
        log("  Condition: PURCHASED_AT > CLICK_TIME, within 7-day window", f)
        log("", f)

        purchases = tables['purchases']

        if len(purchases) > 0 and len(clicks) > 0:
            # Convert timestamps
            purchases['PURCHASED_AT'] = pd.to_datetime(purchases['PURCHASED_AT'])
            clicks['OCCURRED_AT'] = pd.to_datetime(clicks['OCCURRED_AT'])

            # Create user-product keys
            purchase_user_products = set(
                purchases['USER_ID'].astype(str) + '_' + purchases['PRODUCT_ID'].astype(str)
            )
            click_user_products = set(
                clicks['USER_ID'].astype(str) + '_' + clicks['PRODUCT_ID'].astype(str)
            )

            matched_pairs = purchase_user_products & click_user_products
            organic_pairs = purchase_user_products - click_user_products

            log(f"  Unique (USER, PRODUCT) in purchases: {len(purchase_user_products):,}", f)
            log(f"  Unique (USER, PRODUCT) in clicks:    {len(click_user_products):,}", f)
            log(f"  Matched pairs (potential attributed): {len(matched_pairs):,} ({len(matched_pairs)/len(purchase_user_products)*100:.2f}%)", f)
            log(f"  Unmatched pairs (organic):            {len(organic_pairs):,} ({len(organic_pairs)/len(purchase_user_products)*100:.2f}%)", f)
            log("", f)

            # Detailed attribution with time window
            log("  Detailed attribution (7-day window):", f)

            # Merge purchases with clicks
            purch_df = purchases[['PURCHASE_ID', 'USER_ID', 'PRODUCT_ID', 'PURCHASED_AT', 'UNIT_PRICE', 'QUANTITY']].copy()
            click_df = clicks[['USER_ID', 'PRODUCT_ID', 'VENDOR_ID', 'OCCURRED_AT', 'AUCTION_ID']].copy()

            # Merge on user + product
            merged = purch_df.merge(
                click_df,
                on=['USER_ID', 'PRODUCT_ID'],
                how='left',
                suffixes=('_purch', '_click')
            )

            # Calculate time difference
            merged['time_diff_hours'] = (merged['PURCHASED_AT'] - merged['OCCURRED_AT']).dt.total_seconds() / 3600

            # Valid attribution: click before purchase, within 7 days
            merged['valid_attribution'] = (
                (merged['time_diff_hours'] > 0) &
                (merged['time_diff_hours'] <= 7 * 24)
            )

            # Get best attribution per purchase (shortest valid time)
            attributed = merged[merged['valid_attribution']].copy()
            if len(attributed) > 0:
                best_attr = attributed.loc[attributed.groupby('PURCHASE_ID')['time_diff_hours'].idxmin()]

                n_attributed = best_attr['PURCHASE_ID'].nunique()
                n_total = purch_df['PURCHASE_ID'].nunique()

                log(f"    Total purchases: {n_total:,}", f)
                log(f"    Attributed purchases (7-day window): {n_attributed:,} ({n_attributed/n_total*100:.2f}%)", f)
                log(f"    Organic purchases: {n_total - n_attributed:,} ({(n_total - n_attributed)/n_total*100:.2f}%)", f)

                # GMV breakdown
                if 'UNIT_PRICE' in purch_df.columns and 'QUANTITY' in purch_df.columns:
                    total_gmv = (purch_df['UNIT_PRICE'] * purch_df['QUANTITY']).sum()
                    attributed_gmv = (best_attr['UNIT_PRICE'] * best_attr['QUANTITY']).sum()

                    log("", f)
                    log(f"    Total GMV: {total_gmv:,.2f}", f)
                    log(f"    Attributed GMV: {attributed_gmv:,.2f} ({attributed_gmv/total_gmv*100:.2f}%)", f)
                    log(f"    Organic GMV: {total_gmv - attributed_gmv:,.2f} ({(total_gmv - attributed_gmv)/total_gmv*100:.2f}%)", f)
            else:
                log("    No valid attributions found within 7-day window", f)

        log("", f)

        # -----------------------------------------------------------------
        # Audit 5: Vendor coverage
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("AUDIT 5: VENDOR COVERAGE ACROSS TABLES", f)
        log("-" * 40, f)

        vendors_by_table = {}

        if len(ar) > 0:
            vendors_by_table['auctions_results'] = set(ar['VENDOR_ID'].unique())
        if len(clicks) > 0:
            vendors_by_table['clicks'] = set(clicks['VENDOR_ID'].unique())
        if len(impressions) > 0:
            vendors_by_table['impressions'] = set(impressions['VENDOR_ID'].unique())

        if len(vendors_by_table) >= 2:
            all_vendors = set().union(*vendors_by_table.values())
            log(f"  Total unique vendors across all tables: {len(all_vendors):,}", f)
            log("", f)

            for name, vendors in vendors_by_table.items():
                log(f"  {name}: {len(vendors):,} vendors", f)

            # Vendors in all tables
            if len(vendors_by_table) >= 2:
                in_all = set.intersection(*vendors_by_table.values())
                log(f"\n  Vendors present in ALL ad tables: {len(in_all):,}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("AUDIT SUMMARY", f)
        log("-" * 40, f)

        log("", f)
        log("  KEY FINDINGS:", f)
        log("  - Composite key (AUCTION_ID, PRODUCT_ID, VENDOR_ID) is required for", f)
        log("    reliable joins between funnel events", f)
        log("  - Some orphan records exist (clicks without matching auctions)", f)
        log("  - Purchase attribution requires USER_ID + PRODUCT_ID match with", f)
        log("    time window constraint", f)
        log("", f)

        log("=" * 80, f)
        log("AUDIT COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
