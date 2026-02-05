#!/usr/bin/env python3
"""
00_explore_data.py - Explore real data and prepare for pipeline

Usage:
    python 00_explore_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

DATA_SOURCE = Path('../eda/data')  # Symlinks to real data
DATA_OUTPUT = Path('data')
DATA_OUTPUT.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING REAL DATA")
print("=" * 80)

files = {
    'auctions_results': 'auctions_results_365d.parquet',
    'auctions_users': 'auctions_users_365d.parquet',
    'clicks': 'clicks_365d.parquet',
    'impressions': 'impressions_365d.parquet',
    'purchases': 'purchases_365d.parquet',
    'catalog': 'catalog_365d.parquet'
}

data = {}
for name, filename in tqdm(files.items(), desc="Loading"):
    path = DATA_SOURCE / filename
    data[name] = pd.read_parquet(path)
    print(f"  {name}: {len(data[name]):,} rows, {len(data[name].columns)} cols")

# ============================================================================
# SCHEMA INSPECTION
# ============================================================================

print("\n" + "=" * 80)
print("SCHEMA INSPECTION")
print("=" * 80)

for name, df in data.items():
    print(f"\n--- {name.upper()} ---")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes.to_string()}")

# ============================================================================
# DATE RANGE
# ============================================================================

print("\n" + "=" * 80)
print("DATE RANGE ANALYSIS")
print("=" * 80)

# Find timestamp columns
ts_cols = {
    'auctions_results': 'CREATED_AT',
    'auctions_users': 'CREATED_AT',
    'clicks': 'OCCURRED_AT',
    'impressions': 'OCCURRED_AT',
    'purchases': 'PURCHASED_AT'
}

for name, col in ts_cols.items():
    df = data[name]
    if col in df.columns:
        ts = pd.to_datetime(df[col])
        print(f"{name}.{col}: {ts.min()} to {ts.max()} ({(ts.max() - ts.min()).days} days)")

# ============================================================================
# KEY FIELD ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("KEY FIELD ANALYSIS")
print("=" * 80)

# Check key fields exist
key_checks = {
    'auctions_results': ['AUCTION_ID', 'VENDOR_ID', 'CAMPAIGN_ID', 'PRODUCT_ID'],
    'auctions_users': ['AUCTION_ID', 'OPAQUE_USER_ID'],
    'clicks': ['AUCTION_ID', 'USER_ID', 'VENDOR_ID', 'PRODUCT_ID'],
    'impressions': ['AUCTION_ID', 'USER_ID', 'VENDOR_ID', 'PRODUCT_ID'],
    'purchases': ['USER_ID', 'PRODUCT_ID', 'PURCHASE_ID']
}

for name, keys in key_checks.items():
    df = data[name]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        print(f"WARNING: {name} missing keys: {missing}")
    else:
        print(f"{name}: all keys present")

# ============================================================================
# UNIQUE COUNTS
# ============================================================================

print("\n" + "=" * 80)
print("UNIQUE ENTITY COUNTS")
print("=" * 80)

# Users
users_auctions = data['auctions_users']['OPAQUE_USER_ID'].nunique()
users_clicks = data['clicks']['USER_ID'].nunique()
users_impressions = data['impressions']['USER_ID'].nunique()
users_purchases = data['purchases']['USER_ID'].nunique()

print(f"Users in auctions: {users_auctions:,}")
print(f"Users in clicks: {users_clicks:,}")
print(f"Users in impressions: {users_impressions:,}")
print(f"Users in purchases: {users_purchases:,}")

# Vendors
vendors_bids = data['auctions_results']['VENDOR_ID'].nunique()
vendors_clicks = data['clicks']['VENDOR_ID'].nunique()
print(f"\nVendors in bids: {vendors_bids:,}")
print(f"Vendors in clicks: {vendors_clicks:,}")

# Products
products_bids = data['auctions_results']['PRODUCT_ID'].nunique()
products_catalog = data['catalog']['PRODUCT_ID'].nunique()
print(f"\nProducts in bids: {products_bids:,}")
print(f"Products in catalog: {products_catalog:,}")

# ============================================================================
# FUNNEL METRICS
# ============================================================================

print("\n" + "=" * 80)
print("FUNNEL METRICS")
print("=" * 80)

n_auctions = data['auctions_users']['AUCTION_ID'].nunique()
n_bids = len(data['auctions_results'])
n_winners = data['auctions_results']['IS_WINNER'].sum() if 'IS_WINNER' in data['auctions_results'].columns else "N/A"
n_impressions = len(data['impressions'])
n_clicks = len(data['clicks'])
n_purchases = len(data['purchases'])

print(f"Auctions: {n_auctions:,}")
print(f"Bids: {n_bids:,}")
print(f"Winners: {n_winners:,}" if isinstance(n_winners, int) else f"Winners: {n_winners}")
print(f"Impressions: {n_impressions:,}")
print(f"Clicks: {n_clicks:,}")
print(f"Purchases: {n_purchases:,}")

if n_impressions > 0:
    ctr = n_clicks / n_impressions * 100
    print(f"\nCTR (clicks/impressions): {ctr:.2f}%")

# ============================================================================
# JOIN FEASIBILITY TEST
# ============================================================================

print("\n" + "=" * 80)
print("JOIN FEASIBILITY TEST")
print("=" * 80)

# Test: clicks → impressions via AUCTION_ID
clicks = data['clicks']
impressions = data['impressions']

click_auctions = set(clicks['AUCTION_ID'].unique())
impression_auctions = set(impressions['AUCTION_ID'].unique())
overlap = click_auctions & impression_auctions

print(f"Click auction IDs: {len(click_auctions):,}")
print(f"Impression auction IDs: {len(impression_auctions):,}")
print(f"Overlap: {len(overlap):,} ({len(overlap)/len(click_auctions)*100:.1f}% of clicks)")

# Test: auctions_users → auctions_results via AUCTION_ID
auction_ids_users = set(data['auctions_users']['AUCTION_ID'].unique())
auction_ids_results = set(data['auctions_results']['AUCTION_ID'].unique())
overlap_auctions = auction_ids_users & auction_ids_results

print(f"\nAuction IDs in users: {len(auction_ids_users):,}")
print(f"Auction IDs in results: {len(auction_ids_results):,}")
print(f"Overlap: {len(overlap_auctions):,} ({len(overlap_auctions)/len(auction_ids_users)*100:.1f}%)")

# ============================================================================
# USER ID FORMAT CHECK
# ============================================================================

print("\n" + "=" * 80)
print("USER ID FORMAT CHECK")
print("=" * 80)

# Check if user IDs are consistent across tables
sample_users_auctions = data['auctions_users']['OPAQUE_USER_ID'].head(3).tolist()
sample_users_clicks = data['clicks']['USER_ID'].head(3).tolist()
sample_users_purchases = data['purchases']['USER_ID'].head(3).tolist()

print(f"Sample OPAQUE_USER_ID (auctions): {sample_users_auctions}")
print(f"Sample USER_ID (clicks): {sample_users_clicks}")
print(f"Sample USER_ID (purchases): {sample_users_purchases}")

# Check overlap
users_in_auctions = set(data['auctions_users']['OPAQUE_USER_ID'].unique())
users_in_clicks = set(data['clicks']['USER_ID'].unique())
user_overlap = users_in_auctions & users_in_clicks
print(f"\nUsers in both auctions and clicks: {len(user_overlap):,}")

# ============================================================================
# PURCHASE ATTRIBUTION CHECK
# ============================================================================

print("\n" + "=" * 80)
print("PURCHASE ATTRIBUTION FEASIBILITY")
print("=" * 80)

purchases = data['purchases']
clicks = data['clicks']

# Users who purchased
purchase_users = set(purchases['USER_ID'].unique())
click_users = set(clicks['USER_ID'].unique())
users_with_both = purchase_users & click_users

print(f"Users who purchased: {len(purchase_users):,}")
print(f"Users who clicked: {len(click_users):,}")
print(f"Users with both click and purchase: {len(users_with_both):,}")

if len(purchase_users) > 0:
    pct = len(users_with_both) / len(purchase_users) * 100
    print(f"  → {pct:.1f}% of purchasers have at least one click")

# ============================================================================
# SPEND ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SPEND ANALYSIS")
print("=" * 80)

if 'UNIT_PRICE' in purchases.columns and 'QUANTITY' in purchases.columns:
    purchases['spend'] = purchases['UNIT_PRICE'] * purchases['QUANTITY']
    print(f"Total spend: ${purchases['spend'].sum():,.2f}")
    print(f"Avg purchase: ${purchases['spend'].mean():,.2f}")
    print(f"Median purchase: ${purchases['spend'].median():,.2f}")
    print(f"Max purchase: ${purchases['spend'].max():,.2f}")

# ============================================================================
# WEEKLY DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("WEEKLY DISTRIBUTION")
print("=" * 80)

clicks['week'] = pd.to_datetime(clicks['OCCURRED_AT']).dt.isocalendar().week
weekly_clicks = clicks.groupby('week').size()
print(f"Clicks per week: min={weekly_clicks.min()}, max={weekly_clicks.max()}, mean={weekly_clicks.mean():.0f}")
print(f"Weeks with data: {len(weekly_clicks)}")

# ============================================================================
# DATA QUALITY SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DATA QUALITY SUMMARY")
print("=" * 80)

issues = []

# Check for nulls in key fields
for name, keys in key_checks.items():
    df = data[name]
    for k in keys:
        if k in df.columns:
            null_pct = df[k].isna().mean() * 100
            if null_pct > 0:
                issues.append(f"{name}.{k}: {null_pct:.2f}% null")

if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("No major data quality issues found.")

# ============================================================================
# READY FOR PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE READINESS")
print("=" * 80)

ready = True
checks = [
    ("Clicks exist", n_clicks > 0),
    ("Purchases exist", n_purchases > 0),
    ("User overlap (auctions↔clicks)", len(user_overlap) > 0),
    ("Purchasers with clicks", len(users_with_both) > 0),
    ("Auction join feasible", len(overlap_auctions) > 0),
]

for check_name, passed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check_name}")
    if not passed:
        ready = False

print("\n" + ("=" * 80))
if ready:
    print("DATA READY FOR PIPELINE")
    print("Run: python -m papermill 01_data_audit.ipynb /tmp/01_out.ipynb")
else:
    print("DATA NOT READY - fix issues above")
print("=" * 80)
