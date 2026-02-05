"""
Rebuild the 28,619 row user-vendor-week panel from shopping-sessions raw data.
Source: shopping-sessions/data/raw_sample_*.parquet (Mar-Sep 2025, 26 weeks)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('/Users/pranjal/Code/marketplace-incrementality/shopping-sessions/data')
OUTPUT_DIR = Path('/Users/pranjal/Code/marketplace-incrementality/shopping-episode/archive/data')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("REBUILDING USER-VENDOR-WEEK PANEL")
print("=" * 80)

# ============================================================================
# 1. LOAD SOURCE TABLES
# ============================================================================
print("\n1. Loading source tables...")

clicks = pd.read_parquet(DATA_DIR / 'raw_sample_clicks.parquet')
impressions = pd.read_parquet(DATA_DIR / 'raw_sample_impressions.parquet')
bids = pd.read_parquet(DATA_DIR / 'raw_sample_auctions_results.parquet')
auctions = pd.read_parquet(DATA_DIR / 'raw_sample_auctions_users.parquet')
purchases = pd.read_parquet(DATA_DIR / 'raw_sample_purchases.parquet')

print(f"Clicks: {len(clicks):,}")
print(f"Impressions: {len(impressions):,}")
print(f"Bids: {len(bids):,}")
print(f"Auctions: {len(auctions):,}")
print(f"Purchases: {len(purchases):,}")

# Parse timestamps
print("\nParsing timestamps...")
clicks['click_time'] = pd.to_datetime(clicks['OCCURRED_AT'])
impressions['impression_time'] = pd.to_datetime(impressions['OCCURRED_AT'])
bids['bid_time'] = pd.to_datetime(bids['CREATED_AT'])
auctions['auction_time'] = pd.to_datetime(auctions['CREATED_AT'])
purchases['purchase_time'] = pd.to_datetime(purchases['PURCHASED_AT'])

print(f"Click date range: {clicks['click_time'].min()} to {clicks['click_time'].max()}")

# ============================================================================
# 2. BUILD PROMOTED_EVENTS
# ============================================================================
print("\n2. Building PROMOTED_EVENTS...")

promoted_events = clicks[['INTERACTION_ID', 'AUCTION_ID', 'PRODUCT_ID', 'USER_ID',
                          'VENDOR_ID', 'CAMPAIGN_ID', 'click_time']].copy()
promoted_events.columns = ['click_id', 'auction_id', 'product_id', 'user_id',
                           'vendor_id', 'campaign_id', 'click_time']

# Join to bids for auction metadata
bid_cols = ['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID',
            'RANKING', 'IS_WINNER', 'FINAL_BID', 'QUALITY', 'PACING',
            'CONVERSION_RATE', 'PRICE']
bid_cols_available = [c for c in bid_cols if c in bids.columns]

bids_slim = bids[bid_cols_available].copy()
bids_slim.columns = [c.lower() for c in bids_slim.columns]

# Dedupe bids
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

print(f"Promoted events: {len(promoted_events):,}")
print(f"Unique users: {promoted_events['user_id'].nunique():,}")
print(f"Unique vendors: {promoted_events['vendor_id'].nunique():,}")

promoted_events.to_parquet(OUTPUT_DIR / 'promoted_events.parquet', index=False)

# ============================================================================
# 3. BUILD PURCHASES_MAPPED
# ============================================================================
print("\n3. Building PURCHASES_MAPPED...")

purchases_mapped = purchases[['PURCHASE_ID', 'USER_ID', 'PRODUCT_ID', 'purchase_time',
                               'QUANTITY', 'UNIT_PRICE']].copy()
purchases_mapped.columns = ['purchase_id', 'user_id', 'product_id', 'purchase_time',
                            'quantity', 'unit_price']

# Calculate spend (cents to dollars)
purchases_mapped['spend'] = purchases_mapped['quantity'] * purchases_mapped['unit_price'] / 100

# Get promoted click info for each (user, product)
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

print(f"Total purchases: {len(purchases_mapped):,}")
print(f"Post-click valid: {purchases_mapped['is_post_click'].sum():,} ({purchases_mapped['is_post_click'].mean()*100:.1f}%)")
print(f"Valid spend: ${purchases_mapped.loc[purchases_mapped['is_post_click'], 'spend'].sum():,.2f}")

purchases_mapped.to_parquet(OUTPUT_DIR / 'purchases_mapped.parquet', index=False)

# ============================================================================
# 4. BUILD EVENTS_WITH_SESSIONS
# ============================================================================
print("\n4. Building EVENTS_WITH_SESSIONS...")

events_list = []

# Clicks
click_events = promoted_events[['user_id', 'click_time', 'vendor_id', 'product_id']].copy()
click_events['event_type'] = 'click'
click_events['spend'] = 0
click_events.columns = ['user_id', 'timestamp', 'vendor_id', 'product_id', 'event_type', 'spend']
events_list.append(click_events)

# Purchases (only post-click valid ones)
purchase_events = purchases_mapped[purchases_mapped['is_post_click']][
    ['user_id', 'purchase_time', 'click_vendor_id', 'product_id', 'spend']
].copy()
purchase_events['event_type'] = 'purchase'
purchase_events.columns = ['user_id', 'timestamp', 'vendor_id', 'product_id', 'spend', 'event_type']
events_list.append(purchase_events)

all_events = pd.concat(events_list, ignore_index=True)
all_events = all_events.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Create session IDs
SESSION_GAPS = [1, 2, 3, 5, 7]
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

# Add week info
all_events['week'] = all_events['timestamp'].dt.isocalendar().week
all_events['year'] = all_events['timestamp'].dt.year
all_events['year_week'] = all_events['year'].astype(str) + '_W' + all_events['week'].astype(str).str.zfill(2)

print(f"Total events: {len(all_events):,}")
print(f"Weeks: {all_events['year_week'].nunique()}")
print(f"Range: {all_events['year_week'].min()} to {all_events['year_week'].max()}")

all_events.to_parquet(OUTPUT_DIR / 'events_with_sessions.parquet', index=False)

# ============================================================================
# 5. BUILD PANEL_UTV (User × Week × Vendor)
# ============================================================================
print("\n5. Building PANEL_UTV...")

# Add week info to promoted_events
promoted_events['week'] = pd.to_datetime(promoted_events['click_time']).dt.isocalendar().week
promoted_events['year'] = pd.to_datetime(promoted_events['click_time']).dt.year
promoted_events['year_week'] = promoted_events['year'].astype(str) + '_W' + promoted_events['week'].astype(str).str.zfill(2)

# Aggregate clicks to (user, week, vendor) - only use available columns
agg_dict = {'click_id': 'count'}
if 'ranking' in promoted_events.columns:
    agg_dict['ranking'] = ['mean', 'min']
if 'is_winner' in promoted_events.columns:
    agg_dict['is_winner'] = 'mean'

clicks_utv = promoted_events.groupby(['user_id', 'year_week', 'vendor_id']).agg(agg_dict).reset_index()

# Flatten column names
clicks_utv.columns = ['user_id', 'year_week', 'vendor_id', 'C'] + (
    ['avg_rank', 'min_rank'] if 'ranking' in promoted_events.columns else []
) + (
    ['share_winner'] if 'is_winner' in promoted_events.columns else []
)

# Share rank=1
rank1_counts = promoted_events[promoted_events['ranking'] == 1].groupby(
    ['user_id', 'year_week', 'vendor_id']
).size().reset_index(name='rank1_clicks')

clicks_utv = clicks_utv.merge(rank1_counts, on=['user_id', 'year_week', 'vendor_id'], how='left')
clicks_utv['rank1_clicks'] = clicks_utv['rank1_clicks'].fillna(0)
clicks_utv['share_rank1'] = clicks_utv['rank1_clicks'] / clicks_utv['C']

print(f"Click aggregates: {len(clicks_utv):,} (u,t,v) observations")

# Aggregate spend
purchases_valid = purchases_mapped[purchases_mapped['is_post_click']].copy()
purchases_valid['week'] = pd.to_datetime(purchases_valid['purchase_time']).dt.isocalendar().week
purchases_valid['year'] = pd.to_datetime(purchases_valid['purchase_time']).dt.year
purchases_valid['year_week'] = purchases_valid['year'].astype(str) + '_W' + purchases_valid['week'].astype(str).str.zfill(2)

spend_utv = purchases_valid.groupby(['user_id', 'year_week', 'click_vendor_id']).agg({
    'spend': 'sum',
    'purchase_id': 'count'
}).reset_index()
spend_utv.columns = ['user_id', 'year_week', 'vendor_id', 'Y', 'n_purchases']

print(f"Spend aggregates: {len(spend_utv):,} (u,t,v) observations")

# Merge
panel_utv = clicks_utv.merge(
    spend_utv,
    on=['user_id', 'year_week', 'vendor_id'],
    how='outer'
)

panel_utv['C'] = panel_utv['C'].fillna(0).astype(int)
panel_utv['Y'] = panel_utv['Y'].fillna(0)
panel_utv['n_purchases'] = panel_utv['n_purchases'].fillna(0).astype(int)
panel_utv['D'] = (panel_utv['Y'] > 0).astype(int)
panel_utv['log_Y'] = np.log1p(panel_utv['Y'])

# Create FE indices
panel_utv['user_fe'] = pd.Categorical(panel_utv['user_id']).codes
panel_utv['week_fe'] = pd.Categorical(panel_utv['year_week']).codes
panel_utv['vendor_fe'] = pd.Categorical(panel_utv['vendor_id']).codes

print(f"\nPanel dimensions: {len(panel_utv):,} observations")
print(f"Unique users: {panel_utv['user_id'].nunique():,}")
print(f"Unique weeks: {panel_utv['year_week'].nunique()}")
print(f"Unique vendors: {panel_utv['vendor_id'].nunique():,}")

panel_utv.to_parquet(OUTPUT_DIR / 'panel_utv.parquet', index=False)

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("REBUILD COMPLETE")
print("=" * 80)

print(f"\nPanel A (u,t,v): {len(panel_utv):,} rows")
print(f"  Users: {panel_utv['user_id'].nunique():,}")
print(f"  Weeks: {panel_utv['year_week'].nunique()}")
print(f"  Vendors: {panel_utv['vendor_id'].nunique():,}")
print(f"  C range: [{panel_utv['C'].min()}, {panel_utv['C'].max()}], mean={panel_utv['C'].mean():.2f}")
print(f"  Y range: [${panel_utv['Y'].min():.2f}, ${panel_utv['Y'].max():.2f}], mean=${panel_utv['Y'].mean():.2f}")
print(f"  Conversion: {panel_utv['D'].mean()*100:.2f}%")

print("\nOutput files:")
for f in OUTPUT_DIR.glob('*.parquet'):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name}: {size_mb:.1f} MB")
