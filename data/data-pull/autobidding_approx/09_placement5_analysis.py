"""
Re-extract all parameters for Placement 5 (sponsored search) only.
Use correct formula: value = CONVERSION_RATE × AOV (no pCTR multiplication)
"""

import pandas as pd
import numpy as np

print("="*80)
print("PLACEMENT 5 (SPONSORED SEARCH) PARAMETER EXTRACTION")
print("="*80)
print()

# Load data
print("Loading data...")
df_results = pd.read_parquet('../data/raw_auctions_results_20251011.parquet')
df_users = pd.read_parquet('../data/raw_auctions_users_20251011.parquet')
df_catalog = pd.read_parquet('../data/catalog_20251011.parquet')

print(f"Total bids: {len(df_results):,}")

# Merge to get placement
df = pd.merge(
    df_results,
    df_users[['AUCTION_ID', 'PLACEMENT', 'CREATED_AT', 'OPAQUE_USER_ID']],
    on='AUCTION_ID',
    how='left',
    suffixes=('_bid', '_auction')
)

# Filter to Placement 5
df_p5 = df[df['PLACEMENT'] == '5'].copy()

print(f"Placement 5 bids: {len(df_p5):,} ({len(df_p5)/len(df)*100:.1f}%)")
print()

# Prepare data (use auction creation time)
df_p5['datetime'] = pd.to_datetime(df_p5['CREATED_AT_auction'])
df_p5['date'] = df_p5['datetime'].dt.date
df_p5['FINAL_BID_DOLLARS'] = df_p5['FINAL_BID'] / 100.0
df_p5['PRICE_DOLLARS'] = df_p5['PRICE'] / 100.0

# Map AOV
product_prices = df_catalog.set_index('PRODUCT_ID')['PRICE'].to_dict()
df_p5['AOV'] = df_p5['PRODUCT_ID'].map(product_prices)
median_aov = df_catalog['PRICE'].median()
df_p5['AOV'] = df_p5['AOV'].fillna(median_aov)

# Calculate value using CORRECT formula
df_p5['value'] = df_p5['CONVERSION_RATE'] * df_p5['AOV']
df_p5['roas'] = df_p5['value'] / (df_p5['FINAL_BID_DOLLARS'] + 1e-9)

print("="*80)
print("A) DAILY BUDGETS (Placement 5)")
print("="*80)
print()

campaign_day = df_p5.groupby(['CAMPAIGN_ID', 'date']).agg({
    'PACING': 'mean',
    'PRICE_DOLLARS': lambda x: (x * df_p5.loc[x.index, 'IS_WINNER']).sum(),
    'AUCTION_ID': 'count'
}).reset_index()
campaign_day.columns = ['CAMPAIGN_ID', 'date', 'pacing_mean', 'total_spend_dollars', 'num_bids']

budget_constrained = campaign_day[campaign_day['pacing_mean'] >= 0.95].copy()
budget_constrained['approx_daily_budget'] = budget_constrained['total_spend_dollars']

print(f"Campaign-days: {len(campaign_day):,}")
print(f"Budget-constrained (pacing >= 0.95): {len(budget_constrained):,} ({len(budget_constrained)/len(campaign_day)*100:.1f}%)")
print()
print(f"Daily Budget Statistics:")
print(f"  Mean: ${budget_constrained['approx_daily_budget'].mean():.2f}")
print(f"  Median: ${budget_constrained['approx_daily_budget'].median():.2f}")
print(f"  p10: ${budget_constrained['approx_daily_budget'].quantile(0.10):.2f}")
print(f"  p25: ${budget_constrained['approx_daily_budget'].quantile(0.25):.2f}")
print(f"  p50: ${budget_constrained['approx_daily_budget'].quantile(0.50):.2f}")
print(f"  p75: ${budget_constrained['approx_daily_budget'].quantile(0.75):.2f}")
print(f"  p90: ${budget_constrained['approx_daily_budget'].quantile(0.90):.2f}")
print()

print("="*80)
print("B) AUCTION VOLUME (Placement 5)")
print("="*80)
print()

df_users_p5 = df_users[df_users['PLACEMENT'] == '5']
total_auctions_p5 = df_users_p5['AUCTION_ID'].nunique()
df_users_p5['date'] = pd.to_datetime(df_users_p5['CREATED_AT']).dt.date
daily_auctions_p5 = df_users_p5.groupby('date')['AUCTION_ID'].nunique()

print(f"Total auctions (Placement 5): {total_auctions_p5:,}")
print(f"Auctions per day: {daily_auctions_p5.mean():.0f} (mean), {daily_auctions_p5.median():.0f} (median)")

bids_per_auction_p5 = df_p5.groupby('AUCTION_ID').size()
print(f"Bids per auction: {bids_per_auction_p5.mean():.1f} (mean), {bids_per_auction_p5.median():.0f} (median)")
print()

print("="*80)
print("C) BID RANGES (Placement 5)")
print("="*80)
print()

print(f"Bid Statistics (dollars):")
print(f"  Mean: ${df_p5['FINAL_BID_DOLLARS'].mean():.4f}")
print(f"  Median: ${df_p5['FINAL_BID_DOLLARS'].median():.4f}")
print(f"  p10: ${df_p5['FINAL_BID_DOLLARS'].quantile(0.10):.4f}")
print(f"  p90: ${df_p5['FINAL_BID_DOLLARS'].quantile(0.90):.4f}")
print()

print("="*80)
print("D) CONVERSION RATE (Placement 5)")
print("="*80)
print()

print(f"pCVR Statistics:")
print(f"  Mean: {df_p5['CONVERSION_RATE'].mean():.6f}")
print(f"  Median: {df_p5['CONVERSION_RATE'].median():.6f}")
print(f"  p10: {df_p5['CONVERSION_RATE'].quantile(0.10):.6f}")
print(f"  p90: {df_p5['CONVERSION_RATE'].quantile(0.90):.6f}")
print()

print("="*80)
print("E) VALUE = CONVERSION_RATE × AOV (Placement 5)")
print("="*80)
print()

print(f"Value Statistics:")
print(f"  Mean: ${df_p5['value'].mean():.4f}")
print(f"  Median: ${df_p5['value'].median():.4f}")
print(f"  p10: ${df_p5['value'].quantile(0.10):.4f}")
print(f"  p25: ${df_p5['value'].quantile(0.25):.4f}")
print(f"  p50: ${df_p5['value'].median():.4f}")
print(f"  p75: ${df_p5['value'].quantile(0.75):.4f}")
print(f"  p90: ${df_p5['value'].quantile(0.90):.4f}")
print()

# ROAS validation
roas_filtered = df_p5[(df_p5['roas'] > 0) & (df_p5['roas'] < 100)]['roas']
print(f"ROAS Statistics (validation):")
print(f"  Mean: {roas_filtered.mean():.2f}x")
print(f"  Median: {roas_filtered.median():.2f}x")
print(f"  p10: {roas_filtered.quantile(0.10):.2f}x")
print(f"  p90: {roas_filtered.quantile(0.90):.2f}x")
print()

if roas_filtered.median() >= 0.5 and roas_filtered.median() <= 10.0:
    print("✓ ROAS is economically rational!")
else:
    print("⚠️  ROAS seems questionable")
print()

print("="*80)
print("F) QUALITY (Placement 5)")
print("="*80)
print()

print(f"Quality Statistics:")
print(f"  Mean: {df_p5['QUALITY'].mean():.6f}")
print(f"  Median: {df_p5['QUALITY'].median():.6f}")
print(f"  p10: {df_p5['QUALITY'].quantile(0.10):.6f}")
print(f"  p90: {df_p5['QUALITY'].quantile(0.90):.6f}")
print()

print("="*80)
print("SUMMARY FOR PLACEMENT 5")
print("="*80)
print()

print("Key Parameters for FPPE Simulation:")
print(f"  num_bidders: ~{int(bids_per_auction_p5.median())}")
print(f"  num_goods: ~{int(daily_auctions_p5.median() / 1000)} (thousands of auctions aggregated)")
print(f"  Budgets (B): median ${budget_constrained['approx_daily_budget'].median():.2f}, range ${budget_constrained['approx_daily_budget'].quantile(0.10):.2f}-${budget_constrained['approx_daily_budget'].quantile(0.90):.2f}")
print(f"  Valuations (V): median ${df_p5['value'].median():.4f}, range ${df_p5['value'].quantile(0.10):.4f}-${df_p5['value'].quantile(0.90):.4f}")
print(f"  ROAS: median {roas_filtered.median():.2f}x")
print()

# Save placement 5 data for copula
print("Saving Placement 5 data for copula modeling...")
df_p5.to_parquet('placement5_data.parquet')
print(f"✓ Saved {len(df_p5):,} rows to placement5_data.parquet")

