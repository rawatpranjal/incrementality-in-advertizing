#!/usr/bin/env python3
"""
02_canonical_tables.py
Builds auditable canonical tables for panel construction:
1. PROMOTED_EVENTS - One row per promoted click with full auction metadata
2. PURCHASES_MAPPED - Purchases with vendor attribution (promoted-linked only)
3. EVENTS_WITH_SESSIONS - Session IDs with multiple gap thresholds (1/2/3/5/7 days)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/Users/pranjal/Code/marketplace-incrementality/eda/data")
OUTPUT_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "02_canonical_tables.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("02_CANONICAL_TABLES", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script builds the foundational tables for causal analysis by joining the", f)
        log("ad funnel data (clicks, impressions, bids, auctions) into a single promoted", f)
        log("events table, and by mapping purchases to vendors via the promoted journey.", f)
        log("The key question is what fraction of total marketplace spend can be reliably", f)
        log("attributed to a vendor through a promoted click. If mappability is low, the", f)
        log("causal estimates will only apply to a subset of transactions. We also construct", f)
        log("session IDs using multiple inactivity gap thresholds (1/2/3/5/7 days) to test", f)
        log("how sensitive the analysis is to the definition of a shopping session.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # Load source tables
        log("Loading source tables...", f)

        clicks = pd.read_parquet(DATA_DIR / 'clicks_365d.parquet')
        log(f"Clicks: {len(clicks):,}", f)

        impressions = pd.read_parquet(DATA_DIR / 'impressions_365d.parquet')
        log(f"Impressions: {len(impressions):,}", f)

        bids = pd.read_parquet(DATA_DIR / 'auctions_results_365d.parquet')
        log(f"Bids: {len(bids):,}", f)

        auctions = pd.read_parquet(DATA_DIR / 'auctions_users_365d.parquet')
        log(f"Auctions: {len(auctions):,}", f)

        purchases = pd.read_parquet(DATA_DIR / 'purchases_365d.parquet')
        log(f"Purchases: {len(purchases):,}", f)

        # Parse timestamps
        log("\nParsing timestamps...", f)
        clicks['click_time'] = pd.to_datetime(clicks['OCCURRED_AT'])
        impressions['impression_time'] = pd.to_datetime(impressions['OCCURRED_AT'])
        bids['bid_time'] = pd.to_datetime(bids['CREATED_AT'])
        auctions['auction_time'] = pd.to_datetime(auctions['CREATED_AT'])
        purchases['purchase_time'] = pd.to_datetime(purchases['PURCHASED_AT'])
        log("Done.", f)

        # ============================================================
        # 1. BUILD PROMOTED_EVENTS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING PROMOTED_EVENTS", f)
        log("=" * 80, f)

        # Start with clicks
        promoted_events = clicks[['INTERACTION_ID', 'AUCTION_ID', 'PRODUCT_ID', 'USER_ID',
                                  'VENDOR_ID', 'CAMPAIGN_ID', 'click_time']].copy()
        promoted_events.columns = ['click_id', 'auction_id', 'product_id', 'user_id',
                                   'vendor_id', 'campaign_id', 'click_time']
        log(f"Starting clicks: {len(promoted_events):,}", f)

        # Join to impressions
        log("\nJoining to impressions...", f)
        impressions_slim = impressions[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID', 'impression_time']].copy()
        impressions_slim.columns = ['auction_id', 'product_id', 'vendor_id', 'campaign_id', 'impression_time']
        impressions_slim = impressions_slim.sort_values('impression_time').drop_duplicates(
            subset=['auction_id', 'product_id', 'vendor_id', 'campaign_id'], keep='first'
        )
        promoted_events = promoted_events.merge(
            impressions_slim,
            on=['auction_id', 'product_id', 'vendor_id', 'campaign_id'],
            how='left'
        )
        imp_match_rate = promoted_events['impression_time'].notna().mean() * 100
        log(f"Clicks with impression match: {imp_match_rate:.1f}%", f)

        # Join to bids
        log("\nJoining to bids...", f)
        bid_cols = ['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID',
                    'RANKING', 'IS_WINNER', 'FINAL_BID', 'QUALITY', 'PACING',
                    'CONVERSION_RATE', 'PRICE']
        bid_cols_available = [c for c in bid_cols if c in bids.columns]
        bids_slim = bids[bid_cols_available].copy()
        bids_slim.columns = [c.lower() for c in bids_slim.columns]
        if 'is_winner' in bids_slim.columns:
            bids_slim = bids_slim.sort_values(['is_winner', 'ranking'], ascending=[False, True])
        bids_slim = bids_slim.drop_duplicates(
            subset=['auction_id', 'product_id', 'vendor_id', 'campaign_id'], keep='first'
        )
        promoted_events = promoted_events.merge(
            bids_slim,
            on=['auction_id', 'product_id', 'vendor_id', 'campaign_id'],
            how='left'
        )
        bid_match_rate = promoted_events['ranking'].notna().mean() * 100
        log(f"Clicks with bid match: {bid_match_rate:.1f}%", f)

        # Join to auctions
        log("\nJoining to auctions...", f)
        if 'PLACEMENT' in auctions.columns:
            auctions_slim = auctions[['AUCTION_ID', 'OPAQUE_USER_ID', 'PLACEMENT']].copy()
            auctions_slim.columns = ['auction_id', 'opaque_user_id', 'placement']
        else:
            auctions_slim = auctions[['AUCTION_ID', 'OPAQUE_USER_ID']].copy()
            auctions_slim.columns = ['auction_id', 'opaque_user_id']
        auctions_slim = auctions_slim.drop_duplicates(subset=['auction_id'], keep='first')
        promoted_events = promoted_events.merge(
            auctions_slim,
            on=['auction_id'],
            how='left'
        )
        auction_match_rate = promoted_events['opaque_user_id'].notna().mean() * 100
        log(f"Clicks with auction match: {auction_match_rate:.1f}%", f)

        # Summary
        log("\n" + "=" * 80, f)
        log("PROMOTED_EVENTS SUMMARY", f)
        log("=" * 80, f)
        log(f"Rows: {len(promoted_events):,}", f)
        log(f"Columns: {list(promoted_events.columns)}", f)
        log(f"\nUnique users: {promoted_events['user_id'].nunique():,}", f)
        log(f"Unique vendors: {promoted_events['vendor_id'].nunique():,}", f)
        log(f"Unique products: {promoted_events['product_id'].nunique():,}", f)
        log(f"Date range: {promoted_events['click_time'].min()} to {promoted_events['click_time'].max()}", f)

        promoted_events.to_parquet(OUTPUT_DIR / 'promoted_events.parquet', index=False)
        log(f"\nSaved to {OUTPUT_DIR / 'promoted_events.parquet'}", f)

        # ============================================================
        # 2. BUILD PURCHASES_MAPPED
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING PURCHASES_MAPPED", f)
        log("=" * 80, f)

        purchases_mapped = purchases[['PURCHASE_ID', 'USER_ID', 'PRODUCT_ID', 'purchase_time',
                                       'QUANTITY', 'UNIT_PRICE', 'PURCHASE_LINE']].copy()
        purchases_mapped.columns = ['purchase_id', 'user_id', 'product_id', 'purchase_time',
                                    'quantity', 'unit_price', 'purchase_line']
        purchases_mapped['spend'] = purchases_mapped['quantity'] * purchases_mapped['unit_price'] / 100

        log(f"Total purchases: {len(purchases_mapped):,}", f)
        log(f"Total spend: ${purchases_mapped['spend'].sum():,.2f}", f)

        # Get promoted click info
        click_info = promoted_events.groupby(['user_id', 'product_id']).agg({
            'click_time': 'min',
            'vendor_id': 'first',
            'campaign_id': 'first',
            'ranking': 'first',
            'is_winner': 'first'
        }).reset_index()
        click_info.columns = ['user_id', 'product_id', 'first_click_time',
                              'click_vendor_id', 'click_campaign_id',
                              'click_ranking', 'click_is_winner']
        log(f"\nUnique (user, product) pairs with clicks: {len(click_info):,}", f)

        purchases_mapped = purchases_mapped.merge(
            click_info,
            on=['user_id', 'product_id'],
            how='left'
        )
        purchases_mapped['is_promoted_linked'] = purchases_mapped['first_click_time'].notna()
        purchases_mapped['click_to_purchase_hours'] = np.where(
            purchases_mapped['is_promoted_linked'],
            (purchases_mapped['purchase_time'] - purchases_mapped['first_click_time']).dt.total_seconds() / 3600,
            np.nan
        )
        purchases_mapped['is_post_click'] = (
            purchases_mapped['is_promoted_linked'] &
            (purchases_mapped['click_to_purchase_hours'] >= 0)
        )

        log("\n--- Mapping Summary ---", f)
        log(f"Total purchases: {len(purchases_mapped):,}", f)
        log(f"Promoted-linked: {purchases_mapped['is_promoted_linked'].sum():,} ({purchases_mapped['is_promoted_linked'].mean()*100:.1f}%)", f)
        log(f"Post-click (valid): {purchases_mapped['is_post_click'].sum():,} ({purchases_mapped['is_post_click'].mean()*100:.1f}%)", f)

        total_spend = purchases_mapped['spend'].sum()
        linked_spend = purchases_mapped.loc[purchases_mapped['is_promoted_linked'], 'spend'].sum()
        valid_spend = purchases_mapped.loc[purchases_mapped['is_post_click'], 'spend'].sum()

        log("\n--- Spend Coverage ---", f)
        log(f"Total spend: ${total_spend:,.2f}", f)
        log(f"Promoted-linked spend: ${linked_spend:,.2f} ({linked_spend/total_spend*100:.1f}%)", f)
        log(f"Valid post-click spend: ${valid_spend:,.2f} ({valid_spend/total_spend*100:.1f}%)", f)

        valid_purchases = purchases_mapped[purchases_mapped['is_post_click']].copy()
        log("\n--- Click-to-Purchase Lag (valid only) ---", f)
        log(str(valid_purchases['click_to_purchase_hours'].describe()), f)

        log("\nPercentiles (hours):", f)
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = valid_purchases['click_to_purchase_hours'].quantile(p/100)
            log(f"  P{p}: {val:.1f}h ({val/24:.1f} days)", f)

        purchases_mapped.to_parquet(OUTPUT_DIR / 'purchases_mapped.parquet', index=False)
        log(f"\nSaved to {OUTPUT_DIR / 'purchases_mapped.parquet'}", f)

        # ============================================================
        # 3. BUILD EVENTS WITH SESSION IDS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING SESSION IDS", f)
        log("=" * 80, f)

        events_list = []

        # Clicks
        click_events = promoted_events[['user_id', 'click_time', 'vendor_id', 'product_id']].copy()
        click_events['event_type'] = 'click'
        click_events.columns = ['user_id', 'timestamp', 'vendor_id', 'product_id', 'event_type']
        events_list.append(click_events)

        # Purchases (only post-click valid ones)
        purchase_events = purchases_mapped[purchases_mapped['is_post_click']][['user_id', 'purchase_time', 'click_vendor_id', 'product_id', 'spend']].copy()
        purchase_events['event_type'] = 'purchase'
        purchase_events.columns = ['user_id', 'timestamp', 'vendor_id', 'product_id', 'spend', 'event_type']
        events_list.append(purchase_events)

        all_events = pd.concat(events_list, ignore_index=True)
        all_events = all_events.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        log(f"Total events: {len(all_events):,}", f)
        log(f"Unique users: {all_events['user_id'].nunique():,}", f)

        SESSION_GAPS = [1, 2, 3, 5, 7]
        log("\nCreating session IDs for multiple gap thresholds...", f)

        all_events['time_since_last'] = all_events.groupby('user_id')['timestamp'].diff()

        for gap_days in tqdm(SESSION_GAPS, desc="Session gaps"):
            gap_threshold = pd.Timedelta(days=gap_days)
            col_name = f'session_id_{gap_days}d'
            all_events['new_session'] = (
                (all_events['time_since_last'] > gap_threshold) |
                (all_events['time_since_last'].isnull())
            )
            all_events['session_num'] = all_events.groupby('user_id')['new_session'].cumsum()
            all_events[col_name] = all_events['user_id'].astype(str) + '_S' + all_events['session_num'].astype(str)

        all_events = all_events.drop(columns=['time_since_last', 'new_session', 'session_num'])

        log("\n--- Session Counts by Gap Threshold ---", f)
        for gap_days in SESSION_GAPS:
            col = f'session_id_{gap_days}d'
            n_sessions = all_events[col].nunique()
            avg_per_user = n_sessions / all_events['user_id'].nunique()
            log(f"  {gap_days}-day gap: {n_sessions:,} sessions ({avg_per_user:.1f} per user)", f)

        all_events['week'] = all_events['timestamp'].dt.isocalendar().week
        all_events['year'] = all_events['timestamp'].dt.year
        all_events['year_week'] = all_events['year'].astype(str) + '_W' + all_events['week'].astype(str).str.zfill(2)

        log(f"\nWeeks in data: {all_events['year_week'].nunique()}", f)
        log(f"Range: {all_events['year_week'].min()} to {all_events['year_week'].max()}", f)

        all_events.to_parquet(OUTPUT_DIR / 'events_with_sessions.parquet', index=False)
        log(f"\nSaved to {OUTPUT_DIR / 'events_with_sessions.parquet'}", f)
        log(f"Columns: {list(all_events.columns)}", f)

        # ============================================================
        # SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("CANONICAL TABLES SUMMARY", f)
        log("=" * 80, f)

        log("\n1. PROMOTED_EVENTS", f)
        log(f"   Rows: {len(promoted_events):,}", f)
        log(f"   Users: {promoted_events['user_id'].nunique():,}", f)
        log(f"   Vendors: {promoted_events['vendor_id'].nunique():,}", f)

        log("\n2. PURCHASES_MAPPED", f)
        log(f"   Total purchases: {len(purchases_mapped):,}", f)
        log(f"   Valid (post-click): {purchases_mapped['is_post_click'].sum():,}", f)
        log(f"   Valid spend: ${purchases_mapped.loc[purchases_mapped['is_post_click'], 'spend'].sum():,.2f}", f)

        log("\n3. EVENTS_WITH_SESSIONS", f)
        log(f"   Events: {len(all_events):,}", f)
        log(f"   Users: {all_events['user_id'].nunique():,}", f)
        log(f"   Weeks: {all_events['year_week'].nunique()}", f)
        for gap in SESSION_GAPS:
            log(f"   Sessions ({gap}d gap): {all_events[f'session_id_{gap}d'].nunique():,}", f)

        log("", f)
        log("=" * 80, f)
        log("CANONICAL TABLES COMPLETE", f)
        log("=" * 80, f)

        log("\nOutput files in data/:", f)
        for fp in OUTPUT_DIR.glob('*.parquet'):
            size_mb = fp.stat().st_size / 1e6
            log(f"  {fp.name}: {size_mb:.1f} MB", f)

if __name__ == "__main__":
    main()
