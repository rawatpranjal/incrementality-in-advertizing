#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Panel Construction
Builds a balanced Vendor x Week panel with outcomes and treatment variables.
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

OUTPUT_FILE = RESULTS_DIR / "02_panel_construction.txt"

# =============================================================================
# CONFIG
# =============================================================================
ATTRIBUTION_WINDOW_DAYS = 7  # Days for click-to-purchase attribution

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
        log("STAGGERED ADOPTION: PANEL CONSTRUCTION", f)
        log("=" * 80, f)
        log("", f)

        log("PANEL SPECIFICATION:", f)
        log("  Unit of observation: VENDOR_ID x WEEK", f)
        log("  Variables:", f)
        log("    - TOTAL_GMV: Sum of all purchases (organic + attributed)", f)
        log("    - ORGANIC_GMV: Purchases NOT linked to ad clicks", f)
        log("    - PROMOTED_GMV: Purchases attributed to ad clicks", f)
        log("    - TOTAL_SPEND: Sum of winning bids (FINAL_BID)", f)
        log("    - IMPRESSIONS: Count of impressions", f)
        log("    - CLICKS: Count of clicks", f)
        log("    - AUCTION_PARTICIPATIONS: Count of auctions entered", f)
        log(f"  Attribution window: {ATTRIBUTION_WINDOW_DAYS} days", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load raw data
        # -----------------------------------------------------------------
        log("LOADING RAW DATA", f)
        log("-" * 40, f)

        files = {
            'auctions_results': 'raw_sample_auctions_results.parquet',
            'impressions': 'raw_sample_impressions.parquet',
            'clicks': 'raw_sample_clicks.parquet',
            'purchases': 'raw_sample_purchases.parquet',
        }

        tables = {}
        for name, filename in tqdm(files.items(), desc="Loading"):
            path = RAW_DATA_DIR / filename
            if path.exists():
                tables[name] = pd.read_parquet(path)
                log(f"  {name}: {len(tables[name]):,} rows", f)
            else:
                log(f"  {name}: NOT FOUND", f)
                tables[name] = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Parse timestamps and create week column
        # -----------------------------------------------------------------
        log("PARSING TIMESTAMPS", f)
        log("-" * 40, f)

        ar = tables['auctions_results'].copy()
        impressions = tables['impressions'].copy()
        clicks = tables['clicks'].copy()
        purchases = tables['purchases'].copy()

        if len(ar) > 0:
            ar['CREATED_AT'] = pd.to_datetime(ar['CREATED_AT'])
            ar['week'] = ar['CREATED_AT'].dt.to_period('W').apply(lambda x: x.start_time)
            log(f"  auctions_results: {ar['week'].nunique()} unique weeks", f)

        if len(impressions) > 0:
            impressions['OCCURRED_AT'] = pd.to_datetime(impressions['OCCURRED_AT'])
            impressions['week'] = impressions['OCCURRED_AT'].dt.to_period('W').apply(lambda x: x.start_time)
            log(f"  impressions: {impressions['week'].nunique()} unique weeks", f)

        if len(clicks) > 0:
            clicks['OCCURRED_AT'] = pd.to_datetime(clicks['OCCURRED_AT'])
            clicks['week'] = clicks['OCCURRED_AT'].dt.to_period('W').apply(lambda x: x.start_time)
            log(f"  clicks: {clicks['week'].nunique()} unique weeks", f)

        if len(purchases) > 0:
            purchases['PURCHASED_AT'] = pd.to_datetime(purchases['PURCHASED_AT'])
            purchases['week'] = purchases['PURCHASED_AT'].dt.to_period('W').apply(lambda x: x.start_time)
            log(f"  purchases: {purchases['week'].nunique()} unique weeks", f)

        log("", f)

        # -----------------------------------------------------------------
        # Step 1: Attribute purchases to clicks
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 1: PURCHASE ATTRIBUTION", f)
        log("-" * 40, f)

        if len(purchases) > 0 and len(clicks) > 0:
            # Merge purchases with clicks on (USER_ID, PRODUCT_ID)
            purch = purchases[['PURCHASE_ID', 'USER_ID', 'PRODUCT_ID', 'PURCHASED_AT',
                               'UNIT_PRICE', 'QUANTITY', 'week']].copy()
            purch['gmv'] = purch['UNIT_PRICE'] * purch['QUANTITY']

            click_df = clicks[['USER_ID', 'PRODUCT_ID', 'VENDOR_ID', 'OCCURRED_AT']].copy()

            # Merge
            merged = purch.merge(
                click_df,
                on=['USER_ID', 'PRODUCT_ID'],
                how='left'
            )

            # Calculate time difference
            merged['time_diff_hours'] = (merged['PURCHASED_AT'] - merged['OCCURRED_AT']).dt.total_seconds() / 3600

            # Valid attribution: click before purchase, within window
            merged['is_attributed'] = (
                (merged['time_diff_hours'] > 0) &
                (merged['time_diff_hours'] <= ATTRIBUTION_WINDOW_DAYS * 24)
            )

            # Get unique attributed purchases (take first valid click)
            attributed = merged[merged['is_attributed']].copy()
            if len(attributed) > 0:
                # Take the click closest in time to the purchase
                attributed = attributed.sort_values('time_diff_hours')
                attributed = attributed.drop_duplicates(subset=['PURCHASE_ID'], keep='first')

                attributed_purchase_ids = set(attributed['PURCHASE_ID'])
            else:
                attributed_purchase_ids = set()

            # Tag purchases
            purch['is_attributed'] = purch['PURCHASE_ID'].isin(attributed_purchase_ids)

            # Get vendor for attributed purchases
            purch_with_vendor = purch.merge(
                attributed[['PURCHASE_ID', 'VENDOR_ID']].drop_duplicates(),
                on='PURCHASE_ID',
                how='left'
            )

            n_attributed = purch['is_attributed'].sum()
            n_organic = (~purch['is_attributed']).sum()

            log(f"  Total purchases: {len(purch):,}", f)
            log(f"  Attributed purchases: {n_attributed:,} ({n_attributed/len(purch)*100:.2f}%)", f)
            log(f"  Organic purchases: {n_organic:,} ({n_organic/len(purch)*100:.2f}%)", f)
            log("", f)

            # GMV breakdown
            total_gmv = purch['gmv'].sum()
            attributed_gmv = purch[purch['is_attributed']]['gmv'].sum()
            organic_gmv = purch[~purch['is_attributed']]['gmv'].sum()

            log(f"  Total GMV: {total_gmv:,.2f}", f)
            log(f"  Attributed GMV: {attributed_gmv:,.2f} ({attributed_gmv/total_gmv*100:.2f}%)", f)
            log(f"  Organic GMV: {organic_gmv:,.2f} ({organic_gmv/total_gmv*100:.2f}%)", f)
        else:
            purch_with_vendor = pd.DataFrame()
            log("  Insufficient data for attribution", f)

        log("", f)

        # -----------------------------------------------------------------
        # Step 2: Aggregate auction participation by vendor x week
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 2: AGGREGATE AUCTION METRICS BY VENDOR x WEEK", f)
        log("-" * 40, f)

        if len(ar) > 0:
            # Auction participations (all bids)
            auction_agg = ar.groupby(['VENDOR_ID', 'week']).agg(
                auction_participations=('AUCTION_ID', 'nunique'),
                total_bids=('AUCTION_ID', 'count'),
                wins=('IS_WINNER', 'sum'),
            ).reset_index()

            # Check if FINAL_BID column exists for spend calculation
            if 'FINAL_BID' in ar.columns:
                winning = ar[ar['IS_WINNER'] == True]
                spend_agg = winning.groupby(['VENDOR_ID', 'week']).agg(
                    total_spend=('FINAL_BID', 'sum'),
                ).reset_index()
                auction_agg = auction_agg.merge(spend_agg, on=['VENDOR_ID', 'week'], how='left')
                auction_agg['total_spend'] = auction_agg['total_spend'].fillna(0)
                log(f"  Total spend: {auction_agg['total_spend'].sum():,.2f}", f)
            else:
                # Use wins as proxy for treatment intensity when FINAL_BID not available
                log("  NOTE: FINAL_BID column not available, using wins as treatment proxy", f)
                auction_agg['total_spend'] = auction_agg['wins']  # Use wins as proxy

            log(f"  Rows in auction aggregation: {len(auction_agg):,}", f)
            log(f"  Unique vendors: {auction_agg['VENDOR_ID'].nunique():,}", f)
            log(f"  Total wins (treatment proxy): {auction_agg['wins'].sum():,.0f}", f)
        else:
            auction_agg = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Step 3: Aggregate impressions by vendor x week
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 3: AGGREGATE IMPRESSIONS BY VENDOR x WEEK", f)
        log("-" * 40, f)

        if len(impressions) > 0:
            imp_agg = impressions.groupby(['VENDOR_ID', 'week']).agg(
                impressions=('INTERACTION_ID', 'count'),
            ).reset_index()

            log(f"  Rows in impression aggregation: {len(imp_agg):,}", f)
            log(f"  Total impressions: {imp_agg['impressions'].sum():,}", f)
        else:
            imp_agg = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Step 4: Aggregate clicks by vendor x week
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 4: AGGREGATE CLICKS BY VENDOR x WEEK", f)
        log("-" * 40, f)

        if len(clicks) > 0:
            click_agg = clicks.groupby(['VENDOR_ID', 'week']).agg(
                clicks=('INTERACTION_ID', 'count'),
            ).reset_index()

            log(f"  Rows in click aggregation: {len(click_agg):,}", f)
            log(f"  Total clicks: {click_agg['clicks'].sum():,}", f)
        else:
            click_agg = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Step 5: Aggregate GMV by vendor x week
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 5: AGGREGATE GMV BY VENDOR x WEEK", f)
        log("-" * 40, f)

        if len(purch_with_vendor) > 0:
            # Attributed GMV by vendor
            attr_gmv = purch_with_vendor[purch_with_vendor['is_attributed']].groupby(
                ['VENDOR_ID', 'week']
            ).agg(
                promoted_gmv=('gmv', 'sum'),
                promoted_units=('PURCHASE_ID', 'count'),
            ).reset_index()

            log(f"  Rows in attributed GMV aggregation: {len(attr_gmv):,}", f)
            log(f"  Total promoted GMV: {attr_gmv['promoted_gmv'].sum():,.2f}", f)
        else:
            attr_gmv = pd.DataFrame()

        log("", f)

        # -----------------------------------------------------------------
        # Step 6: Build the panel
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STEP 6: BUILD VENDOR x WEEK PANEL", f)
        log("-" * 40, f)

        # Get all unique vendor-week combinations from auctions
        if len(auction_agg) > 0:
            panel = auction_agg[['VENDOR_ID', 'week']].copy()

            # Merge in metrics
            if len(imp_agg) > 0:
                panel = panel.merge(imp_agg, on=['VENDOR_ID', 'week'], how='left')
            else:
                panel['impressions'] = 0

            if len(click_agg) > 0:
                panel = panel.merge(click_agg, on=['VENDOR_ID', 'week'], how='left')
            else:
                panel['clicks'] = 0

            if len(attr_gmv) > 0:
                panel = panel.merge(attr_gmv, on=['VENDOR_ID', 'week'], how='left')
            else:
                panel['promoted_gmv'] = 0
                panel['promoted_units'] = 0

            # Merge auction metrics
            panel = panel.merge(
                auction_agg[['VENDOR_ID', 'week', 'auction_participations', 'total_bids', 'wins', 'total_spend']],
                on=['VENDOR_ID', 'week'],
                how='left'
            )

            # Fill NAs with 0
            numeric_cols = ['impressions', 'clicks', 'promoted_gmv', 'promoted_units',
                           'auction_participations', 'total_bids', 'wins', 'total_spend']
            for col in numeric_cols:
                if col in panel.columns:
                    panel[col] = panel[col].fillna(0)

            # Create treatment indicator (has any spend)
            panel['has_spend'] = (panel['total_spend'] > 0).astype(int)

            # Create log outcomes
            panel['log_promoted_gmv'] = np.log1p(panel['promoted_gmv'])
            panel['log_spend'] = np.log1p(panel['total_spend'])

            log(f"  Panel shape: {panel.shape}", f)
            log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
            log(f"  Unique weeks: {panel['week'].nunique()}", f)
            log(f"  Total observations: {len(panel):,}", f)
            log("", f)

            log("  Panel summary statistics:", f)
            log(f"  {'Variable':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}", f)
            log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

            for col in numeric_cols:
                if col in panel.columns:
                    log(f"  {col:<25} {panel[col].mean():>12.2f} {panel[col].std():>12.2f} {panel[col].min():>12.2f} {panel[col].max():>12.2f}", f)

            log("", f)

            # Week distribution
            log("  Observations per week:", f)
            week_counts = panel.groupby('week').size().reset_index(name='n_vendors')
            for _, row in week_counts.iterrows():
                log(f"    {row['week'].date()}: {row['n_vendors']:,} vendors", f)

            log("", f)

            # Save panel
            output_path = DATA_DIR / "panel_vendor_week.parquet"
            panel.to_parquet(output_path, index=False)
            log(f"  Saved panel to {output_path}", f)

        else:
            log("  No auction data available to build panel", f)

        log("", f)
        log("=" * 80, f)
        log("PANEL CONSTRUCTION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
