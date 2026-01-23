"""
Vendor → Campaign → Product Hierarchical Analysis Over Weeks

Analysis of 3-level hierarchy (Vendor > Campaign > Product) with weekly temporal dynamics.

Data Period: 14 days (2025-09-02 to 2025-09-08)
Sample: 0.1% of users
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from scipy.stats import linregress
warnings.filterwarnings('ignore')

print("="*80)
print("VENDOR → CAMPAIGN → PRODUCT WEEKLY ANALYSIS")
print("="*80)
print()

# Load data
print("Loading data...")
print("-" * 80)
df_auctions_results = pd.read_parquet('data/raw_auctions_results_20251011.parquet')
print(f"✓ Loaded AUCTIONS_RESULTS: {len(df_auctions_results):,} rows")

df_auctions_users = pd.read_parquet('data/raw_auctions_users_20251011.parquet')
print(f"✓ Loaded AUCTIONS_USERS: {len(df_auctions_users):,} rows")

df_catalog = pd.read_parquet('data/catalog_20251011.parquet')
print(f"✓ Loaded CATALOG: {len(df_catalog):,} rows")

# Merge and prepare
print("\nMerging and preparing data...")
df_auctions_results_clean = df_auctions_results.drop(columns=['CREATED_AT'])
df = pd.merge(df_auctions_results_clean,
              df_auctions_users[['AUCTION_ID', 'CREATED_AT', 'PLACEMENT', 'OPAQUE_USER_ID']],
              on='AUCTION_ID', how='left')

df['datetime'] = pd.to_datetime(df['CREATED_AT'])
df['date'] = df['datetime'].dt.date
df['week'] = df['datetime'].dt.isocalendar().week
df['week_start'] = df['datetime'].dt.to_period('W').dt.start_time.dt.date
df['day_of_week'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour
df['FINAL_BID_DOLLARS'] = df['FINAL_BID'] / 100
df['PRICE_DOLLARS'] = df['PRICE'] / 100

print(f"✓ Merged dataset: {len(df):,} rows")
print(f"  Total vendors: {df['VENDOR_ID'].nunique():,}")
print(f"  Total campaigns: {df['CAMPAIGN_ID'].nunique():,}")
print(f"  Total products: {df['PRODUCT_ID'].nunique():,}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Weeks covered: {sorted(df['week'].unique())}")
print()

# ============================================================================
# SECTION 1: VENDOR-LEVEL ANALYSIS
# ============================================================================

print("="*80)
print("SECTION 1: VENDOR-LEVEL ANALYSIS (TOP OF HIERARCHY)")
print("="*80)
print()

print("1.1 VENDOR COUNTS AND ACTIVITY")
print("-" * 80)
total_vendors = df['VENDOR_ID'].nunique()
vendors_with_wins = df[df['IS_WINNER'] == True]['VENDOR_ID'].nunique()
print(f"Total unique vendors: {total_vendors:,}")
print(f"Vendors with at least 1 win: {vendors_with_wins:,} ({vendors_with_wins/total_vendors*100:.2f}%)")

# Weekly vendor activity
weekly_vendors = df.groupby('week_start')['VENDOR_ID'].nunique().reset_index()
weekly_vendors.columns = ['week_start', 'n_vendors']
print(f"\nVendors active by week:")
for _, row in weekly_vendors.iterrows():
    print(f"  Week {row['week_start']}: {row['n_vendors']:,} vendors")

print("\n1.2 VENDOR CONCENTRATION")
print("-" * 80)
vendor_stats = df.groupby('VENDOR_ID').agg({
    'AUCTION_ID': 'count',  # total bids
    'IS_WINNER': ['sum', 'mean'],
    'PRICE': 'sum',
    'CAMPAIGN_ID': 'nunique',
    'PRODUCT_ID': 'nunique'
}).reset_index()
vendor_stats.columns = ['VENDOR_ID', 'n_bids', 'n_wins', 'win_rate', 'total_revenue', 'n_campaigns', 'n_products']
vendor_stats = vendor_stats.sort_values('n_bids', ascending=False)

total_bids = vendor_stats['n_bids'].sum()
total_wins = vendor_stats['n_wins'].sum()
total_revenue = vendor_stats['total_revenue'].sum()

print(f"Total bids across all vendors: {total_bids:,}")
print(f"Total wins across all vendors: {total_wins:,}")
print(f"Total revenue: ${total_revenue/100:,.2f}")

print("\nVendor concentration metrics:")
top_10 = vendor_stats.head(10)
top_100 = vendor_stats.head(100)
print(f"  Top 10 vendors:")
print(f"    Bids: {top_10['n_bids'].sum():,} ({top_10['n_bids'].sum()/total_bids*100:.2f}%)")
print(f"    Wins: {top_10['n_wins'].sum():,} ({top_10['n_wins'].sum()/total_wins*100:.2f}%)")
print(f"    Revenue: ${top_10['total_revenue'].sum()/100:,.2f} ({top_10['total_revenue'].sum()/total_revenue*100:.2f}%)")
print(f"  Top 100 vendors:")
print(f"    Bids: {top_100['n_bids'].sum():,} ({top_100['n_bids'].sum()/total_bids*100:.2f}%)")
print(f"    Wins: {top_100['n_wins'].sum():,} ({top_100['n_wins'].sum()/total_wins*100:.2f}%)")
print(f"    Revenue: ${top_100['total_revenue'].sum()/100:,.2f} ({top_100['total_revenue'].sum()/total_revenue*100:.2f}%)")

# Gini coefficient for bid distribution
sorted_bids = np.sort(vendor_stats['n_bids'].values)
n = len(sorted_bids)
gini_bids = (2 * np.sum(np.arange(1, n+1) * sorted_bids)) / (n * np.sum(sorted_bids)) - (n + 1) / n
print(f"\nGini coefficient (bid distribution): {gini_bids:.4f}")

# HHI for market concentration
bid_shares = vendor_stats['n_bids'] / total_bids
hhi = (bid_shares ** 2).sum()
print(f"Herfindahl-Hirschman Index (HHI): {hhi:.4f}")

print("\n1.3 VENDOR PERFORMANCE DISTRIBUTION")
print("-" * 80)
print("Vendor statistics:")
print(f"  Mean bids per vendor: {vendor_stats['n_bids'].mean():.2f}")
print(f"  Median bids per vendor: {vendor_stats['n_bids'].median():.0f}")
print(f"  Mean wins per vendor: {vendor_stats['n_wins'].mean():.2f}")
print(f"  Mean win rate: {vendor_stats['win_rate'].mean()*100:.2f}%")
print(f"  Mean revenue per vendor: ${vendor_stats['total_revenue'].mean()/100:,.2f}")

print("\nBid volume percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {vendor_stats['n_bids'].quantile(q):.0f} bids")

print("\nVendor categories by bid volume:")
print(f"  Very small (<100 bids):    {(vendor_stats['n_bids'] < 100).sum():,} ({(vendor_stats['n_bids'] < 100).mean()*100:.2f}%)")
print(f"  Small (100-1000):          {((vendor_stats['n_bids'] >= 100) & (vendor_stats['n_bids'] < 1000)).sum():,} ({((vendor_stats['n_bids'] >= 100) & (vendor_stats['n_bids'] < 1000)).mean()*100:.2f}%)")
print(f"  Medium (1000-10000):       {((vendor_stats['n_bids'] >= 1000) & (vendor_stats['n_bids'] < 10000)).sum():,} ({((vendor_stats['n_bids'] >= 1000) & (vendor_stats['n_bids'] < 10000)).mean()*100:.2f}%)")
print(f"  Large (10000+):            {(vendor_stats['n_bids'] >= 10000).sum():,} ({(vendor_stats['n_bids'] >= 10000).mean()*100:.2f}%)")

print("\n1.4 MULTI-CAMPAIGN VS SINGLE-CAMPAIGN VENDORS")
print("-" * 80)
single_campaign = vendor_stats[vendor_stats['n_campaigns'] == 1]
multi_campaign = vendor_stats[vendor_stats['n_campaigns'] > 1]

print(f"Single-campaign vendors: {len(single_campaign):,} ({len(single_campaign)/len(vendor_stats)*100:.2f}%)")
print(f"  Mean bids: {single_campaign['n_bids'].mean():.2f}")
print(f"  Mean wins: {single_campaign['n_wins'].mean():.2f}")
print(f"  Mean win rate: {single_campaign['win_rate'].mean()*100:.2f}%")
print(f"  Total revenue: ${single_campaign['total_revenue'].sum()/100:,.2f}")

print(f"\nMulti-campaign vendors: {len(multi_campaign):,} ({len(multi_campaign)/len(vendor_stats)*100:.2f}%)")
print(f"  Mean campaigns: {multi_campaign['n_campaigns'].mean():.2f}")
print(f"  Mean bids: {multi_campaign['n_bids'].mean():.2f}")
print(f"  Mean wins: {multi_campaign['n_wins'].mean():.2f}")
print(f"  Mean win rate: {multi_campaign['win_rate'].mean()*100:.2f}%")
print(f"  Total revenue: ${multi_campaign['total_revenue'].sum()/100:,.2f}")

print("\nCampaigns per vendor distribution:")
for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {vendor_stats['n_campaigns'].quantile(q):.0f} campaigns")

print("\n1.5 VENDOR PRODUCT PORTFOLIO")
print("-" * 80)
print(f"Mean products per vendor: {vendor_stats['n_products'].mean():.2f}")
print(f"Median products per vendor: {vendor_stats['n_products'].median():.0f}")

print("\nProducts per vendor percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {vendor_stats['n_products'].quantile(q):.0f} products")

print("\nVendor portfolio diversity:")
print(f"  Single-product vendors:    {(vendor_stats['n_products'] == 1).sum():,} ({(vendor_stats['n_products'] == 1).mean()*100:.2f}%)")
print(f"  Small portfolio (2-10):    {((vendor_stats['n_products'] >= 2) & (vendor_stats['n_products'] <= 10)).sum():,} ({((vendor_stats['n_products'] >= 2) & (vendor_stats['n_products'] <= 10)).mean()*100:.2f}%)")
print(f"  Medium portfolio (11-100): {((vendor_stats['n_products'] > 10) & (vendor_stats['n_products'] <= 100)).sum():,} ({((vendor_stats['n_products'] > 10) & (vendor_stats['n_products'] <= 100)).mean()*100:.2f}%)")
print(f"  Large portfolio (100+):    {(vendor_stats['n_products'] > 100).sum():,} ({(vendor_stats['n_products'] > 100).mean()*100:.2f}%)")

print("\n1.6 VENDOR WEEKLY ACTIVITY PATTERNS")
print("-" * 80)
vendor_weekly = df.groupby(['VENDOR_ID', 'week_start']).agg({
    'AUCTION_ID': 'count',
    'IS_WINNER': ['sum', 'mean'],
    'PRICE': 'sum'
}).reset_index()
vendor_weekly.columns = ['VENDOR_ID', 'week_start', 'n_bids', 'n_wins', 'win_rate', 'revenue']

print("Vendor-week observations:")
print(f"  Total vendor-weeks: {len(vendor_weekly):,}")
print(f"  Mean bids per vendor-week: {vendor_weekly['n_bids'].mean():.2f}")
print(f"  Mean wins per vendor-week: {vendor_weekly['n_wins'].mean():.2f}")
print(f"  Mean revenue per vendor-week: ${vendor_weekly['revenue'].mean()/100:,.2f}")

# Vendor persistence across weeks
weeks_active_per_vendor = vendor_weekly.groupby('VENDOR_ID')['week_start'].nunique()
print(f"\nVendor persistence:")
print(f"  Active 1 week only: {(weeks_active_per_vendor == 1).sum():,} vendors ({(weeks_active_per_vendor == 1).mean()*100:.2f}%)")
print(f"  Active 2 weeks: {(weeks_active_per_vendor == 2).sum():,} vendors ({(weeks_active_per_vendor == 2).mean()*100:.2f}%)")
print(f"  Active 3+ weeks: {(weeks_active_per_vendor >= 3).sum():,} vendors ({(weeks_active_per_vendor >= 3).mean()*100:.2f}%)")

print()

# ============================================================================
# SECTION 2: CAMPAIGN-LEVEL ANALYSIS
# ============================================================================

print("="*80)
print("SECTION 2: CAMPAIGN-LEVEL ANALYSIS (MIDDLE OF HIERARCHY)")
print("="*80)
print()

print("2.1 CAMPAIGN COUNTS AND STRUCTURE")
print("-" * 80)
total_campaigns = df['CAMPAIGN_ID'].nunique()
campaigns_with_wins = df[df['IS_WINNER'] == True]['CAMPAIGN_ID'].nunique()
print(f"Total unique campaigns: {total_campaigns:,}")
print(f"Campaigns with at least 1 win: {campaigns_with_wins:,} ({campaigns_with_wins/total_campaigns*100:.2f}%)")

campaign_stats = df.groupby('CAMPAIGN_ID').agg({
    'VENDOR_ID': 'first',
    'AUCTION_ID': 'count',
    'IS_WINNER': ['sum', 'mean'],
    'PRICE': 'sum',
    'PRODUCT_ID': 'nunique',
    'FINAL_BID': 'mean',
    'QUALITY': 'mean',
    'PACING': 'mean'
}).reset_index()
campaign_stats.columns = ['CAMPAIGN_ID', 'VENDOR_ID', 'n_bids', 'n_wins', 'win_rate',
                          'total_revenue', 'n_products', 'avg_bid', 'avg_quality', 'avg_pacing']

print(f"\nCampaign statistics:")
print(f"  Mean bids per campaign: {campaign_stats['n_bids'].mean():.2f}")
print(f"  Median bids per campaign: {campaign_stats['n_bids'].median():.0f}")
print(f"  Mean wins per campaign: {campaign_stats['n_wins'].mean():.2f}")
print(f"  Mean win rate: {campaign_stats['win_rate'].mean()*100:.2f}%")

print("\n2.2 CAMPAIGNS PER VENDOR")
print("-" * 80)
campaigns_per_vendor = campaign_stats.groupby('VENDOR_ID').size()
print(f"Mean campaigns per vendor: {campaigns_per_vendor.mean():.2f}")
print(f"Median campaigns per vendor: {campaigns_per_vendor.median():.0f}")
print(f"Max campaigns for single vendor: {campaigns_per_vendor.max():.0f}")

print("\nCampaigns per vendor distribution:")
for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {campaigns_per_vendor.quantile(q):.0f} campaigns")

print("\n2.3 CAMPAIGN PERFORMANCE DISTRIBUTION")
print("-" * 80)
print("Bid volume percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {campaign_stats['n_bids'].quantile(q):.0f} bids")

print("\nWin rate percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {campaign_stats['win_rate'].quantile(q)*100:.2f}%")

print("\nRevenue percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: ${campaign_stats['total_revenue'].quantile(q)/100:.2f}")

print("\n2.4 CAMPAIGN CONCENTRATION")
print("-" * 80)
campaign_stats_sorted = campaign_stats.sort_values('n_bids', ascending=False)
total_campaign_bids = campaign_stats_sorted['n_bids'].sum()
total_campaign_wins = campaign_stats_sorted['n_wins'].sum()
total_campaign_revenue = campaign_stats_sorted['total_revenue'].sum()

top_10_campaigns = campaign_stats_sorted.head(10)
top_100_campaigns = campaign_stats_sorted.head(100)
top_1000_campaigns = campaign_stats_sorted.head(1000)

print(f"Top 10 campaigns:")
print(f"  Bids: {top_10_campaigns['n_bids'].sum():,} ({top_10_campaigns['n_bids'].sum()/total_campaign_bids*100:.2f}%)")
print(f"  Wins: {top_10_campaigns['n_wins'].sum():,} ({top_10_campaigns['n_wins'].sum()/total_campaign_wins*100:.2f}%)")
print(f"  Revenue: ${top_10_campaigns['total_revenue'].sum()/100:,.2f} ({top_10_campaigns['total_revenue'].sum()/total_campaign_revenue*100:.2f}%)")

print(f"\nTop 100 campaigns:")
print(f"  Bids: {top_100_campaigns['n_bids'].sum():,} ({top_100_campaigns['n_bids'].sum()/total_campaign_bids*100:.2f}%)")
print(f"  Wins: {top_100_campaigns['n_wins'].sum():,} ({top_100_campaigns['n_wins'].sum()/total_campaign_wins*100:.2f}%)")
print(f"  Revenue: ${top_100_campaigns['total_revenue'].sum()/100:,.2f} ({top_100_campaigns['total_revenue'].sum()/total_campaign_revenue*100:.2f}%)")

print(f"\nTop 1000 campaigns:")
print(f"  Bids: {top_1000_campaigns['n_bids'].sum():,} ({top_1000_campaigns['n_bids'].sum()/total_campaign_bids*100:.2f}%)")
print(f"  Wins: {top_1000_campaigns['n_wins'].sum():,} ({top_1000_campaigns['n_wins'].sum()/total_campaign_wins*100:.2f}%)")
print(f"  Revenue: ${top_1000_campaigns['total_revenue'].sum()/100:,.2f} ({top_1000_campaigns['total_revenue'].sum()/total_campaign_revenue*100:.2f}%)")

print("\n2.5 PRODUCTS PER CAMPAIGN")
print("-" * 80)
print(f"Mean products per campaign: {campaign_stats['n_products'].mean():.2f}")
print(f"Median products per campaign: {campaign_stats['n_products'].median():.0f}")

print("\nProducts per campaign percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {campaign_stats['n_products'].quantile(q):.0f} products")

print("\nCampaign product portfolio:")
print(f"  Single-product campaigns:    {(campaign_stats['n_products'] == 1).sum():,} ({(campaign_stats['n_products'] == 1).mean()*100:.2f}%)")
print(f"  Small portfolio (2-10):      {((campaign_stats['n_products'] >= 2) & (campaign_stats['n_products'] <= 10)).sum():,} ({((campaign_stats['n_products'] >= 2) & (campaign_stats['n_products'] <= 10)).mean()*100:.2f}%)")
print(f"  Medium portfolio (11-100):   {((campaign_stats['n_products'] > 10) & (campaign_stats['n_products'] <= 100)).sum():,} ({((campaign_stats['n_products'] > 10) & (campaign_stats['n_products'] <= 100)).mean()*100:.2f}%)")
print(f"  Large portfolio (100+):      {(campaign_stats['n_products'] > 100).sum():,} ({(campaign_stats['n_products'] > 100).mean()*100:.2f}%)")

print("\n2.6 CAMPAIGN WEEKLY ACTIVITY")
print("-" * 80)
campaign_weekly = df.groupby(['CAMPAIGN_ID', 'week_start']).agg({
    'AUCTION_ID': 'count',
    'IS_WINNER': ['sum', 'mean'],
    'PRICE': 'sum',
    'PACING': 'mean'
}).reset_index()
campaign_weekly.columns = ['CAMPAIGN_ID', 'week_start', 'n_bids', 'n_wins', 'win_rate', 'revenue', 'avg_pacing']

print(f"Total campaign-weeks: {len(campaign_weekly):,}")
print(f"Mean bids per campaign-week: {campaign_weekly['n_bids'].mean():.2f}")
print(f"Mean wins per campaign-week: {campaign_weekly['n_wins'].mean():.2f}")
print(f"Mean revenue per campaign-week: ${campaign_weekly['revenue'].mean()/100:,.2f}")

weeks_active_per_campaign = campaign_weekly.groupby('CAMPAIGN_ID')['week_start'].nunique()
print(f"\nCampaign persistence:")
print(f"  Active 1 week only: {(weeks_active_per_campaign == 1).sum():,} campaigns ({(weeks_active_per_campaign == 1).mean()*100:.2f}%)")
print(f"  Active 2 weeks: {(weeks_active_per_campaign == 2).sum():,} campaigns ({(weeks_active_per_campaign == 2).mean()*100:.2f}%)")
print(f"  Active 3+ weeks: {(weeks_active_per_campaign >= 3).sum():,} campaigns ({(weeks_active_per_campaign >= 3).mean()*100:.2f}%)")

print()

# ============================================================================
# SECTION 3: PRODUCT-LEVEL ANALYSIS
# ============================================================================

print("="*80)
print("SECTION 3: PRODUCT-LEVEL ANALYSIS (BOTTOM OF HIERARCHY)")
print("="*80)
print()

print("3.1 PRODUCT USAGE IN CAMPAIGNS")
print("-" * 80)
total_products_advertised = df['PRODUCT_ID'].nunique()
print(f"Total unique products advertised: {total_products_advertised:,}")
print(f"Total products in catalog: {len(df_catalog):,}")
print(f"Catalog penetration: {total_products_advertised/len(df_catalog)*100:.2f}%")

product_stats = df.groupby('PRODUCT_ID').agg({
    'AUCTION_ID': 'count',
    'IS_WINNER': ['sum', 'mean'],
    'PRICE': 'sum',
    'CAMPAIGN_ID': 'nunique',
    'VENDOR_ID': 'nunique',
    'FINAL_BID': 'mean',
    'QUALITY': 'mean'
}).reset_index()
product_stats.columns = ['PRODUCT_ID', 'n_bids', 'n_wins', 'win_rate', 'total_revenue',
                         'n_campaigns', 'n_vendors', 'avg_bid', 'avg_quality']

print(f"\nProduct statistics:")
print(f"  Mean bids per product: {product_stats['n_bids'].mean():.2f}")
print(f"  Median bids per product: {product_stats['n_bids'].median():.0f}")
print(f"  Mean wins per product: {product_stats['n_wins'].mean():.2f}")
print(f"  Mean win rate: {product_stats['win_rate'].mean()*100:.2f}%")

print("\n3.2 PRODUCT PERFORMANCE DISTRIBUTION")
print("-" * 80)
print("Bid volume percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: {product_stats['n_bids'].quantile(q):.0f} bids")

print("\nRevenue percentiles:")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {q*100:5.1f}%: ${product_stats['total_revenue'].quantile(q)/100:.2f}")

print("\n3.3 PRODUCT CONCENTRATION")
print("-" * 80)
product_stats_sorted = product_stats.sort_values('n_bids', ascending=False)
total_product_bids = product_stats_sorted['n_bids'].sum()
total_product_wins = product_stats_sorted['n_wins'].sum()
total_product_revenue = product_stats_sorted['total_revenue'].sum()

top_10_products = product_stats_sorted.head(10)
top_100_products = product_stats_sorted.head(100)
top_1000_products = product_stats_sorted.head(1000)

print(f"Top 10 products:")
print(f"  Bids: {top_10_products['n_bids'].sum():,} ({top_10_products['n_bids'].sum()/total_product_bids*100:.2f}%)")
print(f"  Wins: {top_10_products['n_wins'].sum():,} ({top_10_products['n_wins'].sum()/total_product_wins*100:.2f}%)")
print(f"  Revenue: ${top_10_products['total_revenue'].sum()/100:,.2f} ({top_10_products['total_revenue'].sum()/total_product_revenue*100:.2f}%)")

print(f"\nTop 100 products:")
print(f"  Bids: {top_100_products['n_bids'].sum():,} ({top_100_products['n_bids'].sum()/total_product_bids*100:.2f}%)")
print(f"  Wins: {top_100_products['n_wins'].sum():,} ({top_100_products['n_wins'].sum()/total_product_wins*100:.2f}%)")
print(f"  Revenue: ${top_100_products['total_revenue'].sum()/100:,.2f} ({top_100_products['total_revenue'].sum()/total_product_revenue*100:.2f}%)")

print(f"\nTop 1000 products:")
print(f"  Bids: {top_1000_products['n_bids'].sum():,} ({top_1000_products['n_bids'].sum()/total_product_bids*100:.2f}%)")
print(f"  Wins: {top_1000_products['n_wins'].sum():,} ({top_1000_products['n_wins'].sum()/total_product_wins*100:.2f}%)")
print(f"  Revenue: ${top_1000_products['total_revenue'].sum()/100:,.2f} ({top_1000_products['total_revenue'].sum()/total_product_revenue*100:.2f}%)")

# Gini for product distribution
sorted_product_bids = np.sort(product_stats['n_bids'].values)
n_products = len(sorted_product_bids)
gini_products = (2 * np.sum(np.arange(1, n_products+1) * sorted_product_bids)) / (n_products * np.sum(sorted_product_bids)) - (n_products + 1) / n_products
print(f"\nGini coefficient (product bid distribution): {gini_products:.4f}")

print("\n3.4 CROSS-CAMPAIGN PRODUCT USAGE")
print("-" * 80)
print(f"Mean campaigns per product: {product_stats['n_campaigns'].mean():.2f}")
print(f"Median campaigns per product: {product_stats['n_campaigns'].median():.0f}")
print(f"Mean vendors per product: {product_stats['n_vendors'].mean():.2f}")

print("\nCampaigns per product distribution:")
print(f"  Single campaign: {(product_stats['n_campaigns'] == 1).sum():,} ({(product_stats['n_campaigns'] == 1).mean()*100:.2f}%)")
print(f"  2-5 campaigns: {((product_stats['n_campaigns'] >= 2) & (product_stats['n_campaigns'] <= 5)).sum():,} ({((product_stats['n_campaigns'] >= 2) & (product_stats['n_campaigns'] <= 5)).mean()*100:.2f}%)")
print(f"  6-10 campaigns: {((product_stats['n_campaigns'] > 5) & (product_stats['n_campaigns'] <= 10)).sum():,} ({((product_stats['n_campaigns'] > 5) & (product_stats['n_campaigns'] <= 10)).mean()*100:.2f}%)")
print(f"  11+ campaigns: {(product_stats['n_campaigns'] > 10).sum():,} ({(product_stats['n_campaigns'] > 10).mean()*100:.2f}%)")

print("\nVendors per product distribution:")
print(f"  Single vendor: {(product_stats['n_vendors'] == 1).sum():,} ({(product_stats['n_vendors'] == 1).mean()*100:.2f}%)")
print(f"  2-5 vendors: {((product_stats['n_vendors'] >= 2) & (product_stats['n_vendors'] <= 5)).sum():,} ({((product_stats['n_vendors'] >= 2) & (product_stats['n_vendors'] <= 5)).mean()*100:.2f}%)")
print(f"  6+ vendors: {(product_stats['n_vendors'] > 5).sum():,} ({(product_stats['n_vendors'] > 5).mean()*100:.2f}%)")

print("\n3.5 PRODUCT WEEKLY ACTIVITY")
print("-" * 80)
product_weekly = df.groupby(['PRODUCT_ID', 'week_start']).agg({
    'AUCTION_ID': 'count',
    'IS_WINNER': ['sum', 'mean'],
    'CAMPAIGN_ID': 'nunique'
}).reset_index()
product_weekly.columns = ['PRODUCT_ID', 'week_start', 'n_bids', 'n_wins', 'win_rate', 'n_campaigns']

print(f"Total product-weeks: {len(product_weekly):,}")
print(f"Mean bids per product-week: {product_weekly['n_bids'].mean():.2f}")

weeks_active_per_product = product_weekly.groupby('PRODUCT_ID')['week_start'].nunique()
print(f"\nProduct persistence:")
print(f"  Active 1 week only: {(weeks_active_per_product == 1).sum():,} products ({(weeks_active_per_product == 1).mean()*100:.2f}%)")
print(f"  Active 2 weeks: {(weeks_active_per_product == 2).sum():,} products ({(weeks_active_per_product == 2).mean()*100:.2f}%)")
print(f"  Active 3+ weeks: {(weeks_active_per_product >= 3).sum():,} products ({(weeks_active_per_product >= 3).mean()*100:.2f}%)")

print()

# ============================================================================
# SECTION 4: HIERARCHICAL RELATIONSHIPS
# ============================================================================

print("="*80)
print("SECTION 4: HIERARCHICAL RELATIONSHIPS")
print("="*80)
print()

print("4.1 VENDOR → CAMPAIGN MAPPING")
print("-" * 80)
vendor_campaign_map = df.groupby('VENDOR_ID')['CAMPAIGN_ID'].nunique()
campaign_vendor_map = df.groupby('CAMPAIGN_ID')['VENDOR_ID'].nunique()

print(f"Vendor → Campaign cardinality:")
print(f"  Mean campaigns per vendor: {vendor_campaign_map.mean():.2f}")
print(f"  Max campaigns per vendor: {vendor_campaign_map.max():.0f}")
print(f"\nCampaign → Vendor mapping:")
print(f"  Single-vendor campaigns: {(campaign_vendor_map == 1).sum():,} ({(campaign_vendor_map == 1).mean()*100:.2f}%)")
print(f"  Multi-vendor campaigns: {(campaign_vendor_map > 1).sum():,} ({(campaign_vendor_map > 1).mean()*100:.2f}%)")

print("\n4.2 CAMPAIGN → PRODUCT MAPPING")
print("-" * 80)
campaign_product_map = df.groupby('CAMPAIGN_ID')['PRODUCT_ID'].nunique()
product_campaign_map = df.groupby('PRODUCT_ID')['CAMPAIGN_ID'].nunique()

print(f"Campaign → Product cardinality:")
print(f"  Mean products per campaign: {campaign_product_map.mean():.2f}")
print(f"  Median products per campaign: {campaign_product_map.median():.0f}")
print(f"  Max products per campaign: {campaign_product_map.max():.0f}")

print(f"\nProduct → Campaign mapping:")
print(f"  Mean campaigns per product: {product_campaign_map.mean():.2f}")
print(f"  Median campaigns per product: {product_campaign_map.median():.0f}")
print(f"  Max campaigns per product: {product_campaign_map.max():.0f}")

print("\n4.3 VENDOR → PRODUCT DIRECT RELATIONSHIP")
print("-" * 80)
vendor_product_map = df.groupby('VENDOR_ID')['PRODUCT_ID'].nunique()
product_vendor_map = df.groupby('PRODUCT_ID')['VENDOR_ID'].nunique()

print(f"Vendor → Product cardinality:")
print(f"  Mean products per vendor: {vendor_product_map.mean():.2f}")
print(f"  Median products per vendor: {vendor_product_map.median():.0f}")
print(f"  Max products per vendor: {vendor_product_map.max():.0f}")

print(f"\nProduct → Vendor mapping:")
print(f"  Single-vendor products: {(product_vendor_map == 1).sum():,} ({(product_vendor_map == 1).mean()*100:.2f}%)")
print(f"  Multi-vendor products: {(product_vendor_map > 1).sum():,} ({(product_vendor_map > 1).mean()*100:.2f}%)")

print("\n4.4 HIERARCHY STABILITY ACROSS WEEKS")
print("-" * 80)
print("Computing week-over-week stability...")

weeks = sorted(df['week_start'].unique())
if len(weeks) >= 2:
    for i in range(len(weeks) - 1):
        week1 = weeks[i]
        week2 = weeks[i + 1]

        vendors_w1 = set(df[df['week_start'] == week1]['VENDOR_ID'].unique())
        vendors_w2 = set(df[df['week_start'] == week2]['VENDOR_ID'].unique())
        retention_vendors = len(vendors_w1 & vendors_w2) / len(vendors_w1) if len(vendors_w1) > 0 else 0

        campaigns_w1 = set(df[df['week_start'] == week1]['CAMPAIGN_ID'].unique())
        campaigns_w2 = set(df[df['week_start'] == week2]['CAMPAIGN_ID'].unique())
        retention_campaigns = len(campaigns_w1 & campaigns_w2) / len(campaigns_w1) if len(campaigns_w1) > 0 else 0

        products_w1 = set(df[df['week_start'] == week1]['PRODUCT_ID'].unique())
        products_w2 = set(df[df['week_start'] == week2]['PRODUCT_ID'].unique())
        retention_products = len(products_w1 & products_w2) / len(products_w1) if len(products_w1) > 0 else 0

        print(f"\nWeek {week1} → {week2}:")
        print(f"  Vendor retention: {retention_vendors*100:.2f}% ({len(vendors_w1 & vendors_w2):,}/{len(vendors_w1):,})")
        print(f"  Campaign retention: {retention_campaigns*100:.2f}% ({len(campaigns_w1 & campaigns_w2):,}/{len(campaigns_w1):,})")
        print(f"  Product retention: {retention_products*100:.2f}% ({len(products_w1 & products_w2):,}/{len(products_w1):,})")

print("\n4.5 PERFORMANCE ATTRIBUTION")
print("-" * 80)
print("Variance decomposition: How much of win rate variance is at each level?")

# Sample for efficiency
sample_size = min(50000, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Overall variance
overall_win_rate = df_sample['IS_WINNER'].mean()
total_variance = df_sample['IS_WINNER'].var()

# Vendor-level variance
vendor_win_rates = df_sample.groupby('VENDOR_ID')['IS_WINNER'].mean()
vendor_variance = ((vendor_win_rates - overall_win_rate) ** 2).mean()

# Campaign-level variance
campaign_win_rates = df_sample.groupby('CAMPAIGN_ID')['IS_WINNER'].mean()
campaign_variance = ((campaign_win_rates - overall_win_rate) ** 2).mean()

# Product-level variance
product_win_rates = df_sample.groupby('PRODUCT_ID')['IS_WINNER'].mean()
product_variance = ((product_win_rates - overall_win_rate) ** 2).mean()

print(f"Total variance in IS_WINNER: {total_variance:.6f}")
print(f"Vendor-level contribution: {vendor_variance:.6f} ({vendor_variance/total_variance*100:.2f}%)")
print(f"Campaign-level contribution: {campaign_variance:.6f} ({campaign_variance/total_variance*100:.2f}%)")
print(f"Product-level contribution: {product_variance:.6f} ({product_variance/total_variance*100:.2f}%)")

print()

# ============================================================================
# SECTION 5: WEEKLY TEMPORAL ANALYSIS
# ============================================================================

print("="*80)
print("SECTION 5: WEEKLY TEMPORAL ANALYSIS")
print("="*80)
print()

print("5.1 WEEK-OVER-WEEK GROWTH")
print("-" * 80)
weekly_summary = df.groupby('week_start').agg({
    'VENDOR_ID': 'nunique',
    'CAMPAIGN_ID': 'nunique',
    'PRODUCT_ID': 'nunique',
    'AUCTION_ID': 'count',
    'IS_WINNER': 'sum',
    'PRICE': 'sum'
}).reset_index()
weekly_summary.columns = ['week_start', 'n_vendors', 'n_campaigns', 'n_products',
                          'n_bids', 'n_wins', 'revenue']

print("Weekly metrics:")
for _, row in weekly_summary.iterrows():
    print(f"\nWeek {row['week_start']}:")
    print(f"  Vendors: {row['n_vendors']:,}")
    print(f"  Campaigns: {row['n_campaigns']:,}")
    print(f"  Products: {row['n_products']:,}")
    print(f"  Bids: {row['n_bids']:,}")
    print(f"  Wins: {row['n_wins']:,}")
    print(f"  Revenue: ${row['revenue']/100:,.2f}")

if len(weekly_summary) >= 2:
    print("\nWeek-over-week growth rates:")
    for i in range(1, len(weekly_summary)):
        prev_week = weekly_summary.iloc[i-1]
        curr_week = weekly_summary.iloc[i]

        vendor_growth = (curr_week['n_vendors'] - prev_week['n_vendors']) / prev_week['n_vendors'] * 100
        campaign_growth = (curr_week['n_campaigns'] - prev_week['n_campaigns']) / prev_week['n_campaigns'] * 100
        bid_growth = (curr_week['n_bids'] - prev_week['n_bids']) / prev_week['n_bids'] * 100

        print(f"\n{prev_week['week_start']} → {curr_week['week_start']}:")
        print(f"  Vendor growth: {vendor_growth:+.2f}%")
        print(f"  Campaign growth: {campaign_growth:+.2f}%")
        print(f"  Bid volume growth: {bid_growth:+.2f}%")

print("\n5.2 COHORT RETENTION ANALYSIS")
print("-" * 80)
print("Analyzing first-week cohorts...")

if len(weeks) >= 2:
    week1_vendors = set(df[df['week_start'] == weeks[0]]['VENDOR_ID'].unique())
    print(f"\nWeek 1 vendor cohort: {len(week1_vendors):,} vendors")

    for i, week in enumerate(weeks[1:], 2):
        week_n_vendors = set(df[df['week_start'] == week]['VENDOR_ID'].unique())
        retained = len(week1_vendors & week_n_vendors)
        retention_rate = retained / len(week1_vendors) * 100
        print(f"  Week {i} retention: {retained:,} vendors ({retention_rate:.2f}%)")

    week1_campaigns = set(df[df['week_start'] == weeks[0]]['CAMPAIGN_ID'].unique())
    print(f"\nWeek 1 campaign cohort: {len(week1_campaigns):,} campaigns")

    for i, week in enumerate(weeks[1:], 2):
        week_n_campaigns = set(df[df['week_start'] == week]['CAMPAIGN_ID'].unique())
        retained = len(week1_campaigns & week_n_campaigns)
        retention_rate = retained / len(week1_campaigns) * 100
        print(f"  Week {i} retention: {retained:,} campaigns ({retention_rate:.2f}%)")

print("\n5.3 PERFORMANCE TRENDS")
print("-" * 80)
print("Analyzing performance trends over weeks...")

weekly_performance = df.groupby('week_start').agg({
    'IS_WINNER': 'mean',
    'FINAL_BID': 'mean',
    'PRICE': lambda x: x.sum() / (x.notna().sum() + 1e-9),
    'QUALITY': 'mean',
    'PACING': 'mean'
}).reset_index()
weekly_performance.columns = ['week_start', 'win_rate', 'avg_bid', 'avg_price', 'avg_quality', 'avg_pacing']

print("\nWeekly performance metrics:")
for _, row in weekly_performance.iterrows():
    print(f"\nWeek {row['week_start']}:")
    print(f"  Win rate: {row['win_rate']*100:.2f}%")
    print(f"  Avg bid: ${row['avg_bid']/100:.4f}")
    print(f"  Avg price: ${row['avg_price']/100:.4f}")
    print(f"  Avg quality: {row['avg_quality']:.6f}")
    print(f"  Avg pacing: {row['avg_pacing']:.4f}")

print("\n5.4 MARKET CONCENTRATION TRENDS")
print("-" * 80)
print("Tracking HHI over weeks...")

for week in weeks:
    week_data = df[df['week_start'] == week]
    vendor_bids = week_data.groupby('VENDOR_ID').size()
    total_bids_week = vendor_bids.sum()
    vendor_shares = vendor_bids / total_bids_week
    hhi_week = (vendor_shares ** 2).sum()

    print(f"Week {week}: HHI = {hhi_week:.4f}")

print()

# ============================================================================
# SECTION 6: STATISTICAL MODELS
# ============================================================================

print("="*80)
print("SECTION 6: STATISTICAL MODELS")
print("="*80)
print()

print("MODEL 1: VENDOR PERFORMANCE REGRESSION")
print("-" * 80)
print("\nUnit of analysis: Vendor-week")
print("Dependent variable: log(wins)")
print("Independent variables: log(n_campaigns), log(n_products), log(avg_bid)")
print("Purpose: Identify drivers of vendor success")
print("Coefficients: Elasticities (% change in wins for % change in X)")
print()

# Prepare vendor-week data for regression
vendor_weekly_reg = df.groupby(['VENDOR_ID', 'week_start']).agg({
    'IS_WINNER': 'sum',
    'CAMPAIGN_ID': 'nunique',
    'PRODUCT_ID': 'nunique',
    'FINAL_BID': 'mean',
    'QUALITY': 'mean',
    'AUCTION_ID': 'count'
}).reset_index()
vendor_weekly_reg.columns = ['VENDOR_ID', 'week_start', 'wins', 'n_campaigns',
                              'n_products', 'avg_bid', 'avg_quality', 'n_bids']

# Filter to vendors with wins
vendor_weekly_reg = vendor_weekly_reg[vendor_weekly_reg['wins'] > 0]

# Log transform
vendor_weekly_reg['log_wins'] = np.log(vendor_weekly_reg['wins'])
vendor_weekly_reg['log_campaigns'] = np.log(vendor_weekly_reg['n_campaigns'])
vendor_weekly_reg['log_products'] = np.log(vendor_weekly_reg['n_products'])
vendor_weekly_reg['log_bid'] = np.log(vendor_weekly_reg['avg_bid'] + 1)

# Simple OLS
X = vendor_weekly_reg[['log_campaigns', 'log_products', 'log_bid']].values
X = np.column_stack([np.ones(len(X)), X])
y = vendor_weekly_reg['log_wins'].values

beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
r2 = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()

print(f"N observations: {len(vendor_weekly_reg):,}")
print(f"R²: {r2:.4f}")
print(f"\nCoefficients (elasticities):")
print(f"  Intercept: {beta[0]:.4f}")
print(f"  log(n_campaigns): {beta[1]:.4f} (1% ↑ campaigns → {beta[1]:.2f}% change in wins)")
print(f"  log(n_products): {beta[2]:.4f} (1% ↑ products → {beta[2]:.2f}% change in wins)")
print(f"  log(avg_bid): {beta[3]:.4f} (1% ↑ bid → {beta[3]:.2f}% change in wins)")

print("\n\nMODEL 2: CAMPAIGN EFFICIENCY MODEL")
print("-" * 80)
print("\nUnit of analysis: Campaign-week")
print("Dependent variable: win_rate")
print("Independent variables: log(n_products), log(avg_bid), avg_quality")
print("Purpose: Identify campaign-level efficiency drivers")
print()

# Prepare campaign-week data
campaign_weekly_reg = df.groupby(['CAMPAIGN_ID', 'week_start']).agg({
    'IS_WINNER': ['sum', 'mean'],
    'PRODUCT_ID': 'nunique',
    'FINAL_BID': 'mean',
    'QUALITY': 'mean',
    'AUCTION_ID': 'count'
}).reset_index()
campaign_weekly_reg.columns = ['CAMPAIGN_ID', 'week_start', 'wins', 'win_rate',
                                'n_products', 'avg_bid', 'avg_quality', 'n_bids']

# Filter to campaigns with sufficient activity
campaign_weekly_reg = campaign_weekly_reg[campaign_weekly_reg['n_bids'] >= 10]

# Transform
campaign_weekly_reg['log_products'] = np.log(campaign_weekly_reg['n_products'])
campaign_weekly_reg['log_bid'] = np.log(campaign_weekly_reg['avg_bid'] + 1)

# OLS
X_camp = campaign_weekly_reg[['log_products', 'log_bid', 'avg_quality']].values
X_camp = np.column_stack([np.ones(len(X_camp)), X_camp])
y_camp = campaign_weekly_reg['win_rate'].values

beta_camp = np.linalg.lstsq(X_camp, y_camp, rcond=None)[0]
resid_camp = y_camp - X_camp @ beta_camp
r2_camp = 1 - (resid_camp**2).sum() / ((y_camp - y_camp.mean())**2).sum()

print(f"N observations: {len(campaign_weekly_reg):,}")
print(f"R²: {r2_camp:.4f}")
print(f"\nCoefficients:")
print(f"  Intercept: {beta_camp[0]:.4f}")
print(f"  log(n_products): {beta_camp[1]:.4f}")
print(f"  log(avg_bid): {beta_camp[2]:.4f}")
print(f"  avg_quality: {beta_camp[3]:.4f}")

print("\n\nMODEL 3: HIERARCHICAL VARIANCE DECOMPOSITION")
print("-" * 80)
print("\nNested variance decomposition: Vendor > Campaign > Product")
print("Outcome: Wins (IS_WINNER)")
print("Purpose: Quantify variance contribution at each hierarchy level")
print()

# Use sample for computation efficiency
sample_size = min(100000, len(df))
df_decomp = df.sample(sample_size, random_state=42)

# Grand mean
grand_mean = df_decomp['IS_WINNER'].astype(float).mean()
total_ss = ((df_decomp['IS_WINNER'].astype(float) - grand_mean) ** 2).sum()

# Vendor-level
vendor_means = df_decomp.groupby('VENDOR_ID')['IS_WINNER'].transform('mean').astype(float)
vendor_ss = ((vendor_means - grand_mean) ** 2).sum()

# Campaign-level (within vendor)
campaign_means = df_decomp.groupby('CAMPAIGN_ID')['IS_WINNER'].transform('mean').astype(float)
campaign_ss = ((campaign_means - vendor_means) ** 2).sum()

# Residual (product + error)
residual_ss = total_ss - vendor_ss - campaign_ss

print(f"Sample size: {len(df_decomp):,}")
print(f"Grand mean win rate: {grand_mean:.4f}")
print(f"\nVariance decomposition:")
print(f"  Total SS: {total_ss:.2f}")
print(f"  Vendor-level SS: {vendor_ss:.2f} ({vendor_ss/total_ss*100:.2f}%)")
print(f"  Campaign-level SS (within vendor): {campaign_ss:.2f} ({campaign_ss/total_ss*100:.2f}%)")
print(f"  Residual SS (product + error): {residual_ss:.2f} ({residual_ss/total_ss*100:.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
