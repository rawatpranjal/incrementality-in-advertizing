"""
Extract single-day market for FPPE: B vector (budgets) and V matrix (valuations).

For a specific date and placement, this creates FPPE inputs by:
1. Sampling representative auctions from that day
2. Identifying all campaigns that bid on these auctions
3. Estimating daily budget B[i] for each campaign
4. Extracting valuations V[i,j] for each (campaign, auction) pair
5. Recording observed pacing λ_obs[i] for validation

Output: market_{date}_p5.npz with B, V, lambda_obs, campaign_ids, auction_ids
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

print("="*80)
print("EXTRACT SINGLE-DAY MARKET FOR FPPE")
print("="*80)
print()

# Parse arguments
parser = argparse.ArgumentParser(description='Extract daily market for FPPE')
parser.add_argument('--date', type=str, required=True, help='Date in YYYY-MM-DD format')
parser.add_argument('--num_auctions', type=int, default=50, help='Number of auctions to sample (ignored if --full_day)')
parser.add_argument('--min_campaigns', type=int, default=2, help='Min campaigns per auction (ignored if --full_day)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--full_day', action='store_true', help='Use ALL auctions from the day (no sampling)')
parser.add_argument('--budget_method', type=str, default='spend_pacing',
                    choices=['spend_pacing', 'sum_bids', 'hybrid'],
                    help='Budget estimation method: spend_pacing (spend/pacing), sum_bids (sum of all bids), hybrid (max of both)')
args = parser.parse_args()

np.random.seed(args.seed)

# Load data
print(f"Loading Placement 5 data...")
df = pd.read_parquet('placement5_data.parquet')
print(f"Total rows: {len(df):,}")
print()

# Filter to specific date
target_date = pd.to_datetime(args.date).date()
print(f"Filtering to date: {target_date}")

# Ensure we have date column
if 'date' not in df.columns:
    if 'datetime' in df.columns:
        df['date'] = df['datetime'].dt.date
    else:
        df['datetime'] = pd.to_datetime(df['CREATED_AT_auction'])
        df['date'] = df['datetime'].dt.date

df_day = df[df['date'] == target_date].copy()
print(f"Rows on {target_date}: {len(df_day):,}")

if len(df_day) == 0:
    print(f"ERROR: No data found for date {target_date}")
    print(f"Available dates: {sorted(df['date'].unique())}")
    exit(1)

print()

# Analyze day structure
print("="*80)
print("DAY STRUCTURE")
print("="*80)
print()

n_auctions = df_day['AUCTION_ID'].nunique()
n_campaigns = df_day['CAMPAIGN_ID'].nunique()
n_bids = len(df_day)

print(f"Unique auctions: {n_auctions:,}")
print(f"Unique campaigns: {n_campaigns:,}")
print(f"Total bids: {n_bids:,}")
print(f"Avg bids per auction: {n_bids/n_auctions:.1f}")
print()

# Decide whether to sample or use full day
if args.full_day:
    print("="*80)
    print("USING FULL DAY (NO SAMPLING)")
    print("="*80)
    print()
    print(f"Using ALL {n_auctions:,} auctions and ALL {n_campaigns:,} campaigns")
    print()
    selected_auctions = df_day['AUCTION_ID'].unique()
    df_market = df_day.copy()
else:
    # Sample representative auctions
    print("="*80)
    print(f"SAMPLING {args.num_auctions} REPRESENTATIVE AUCTIONS")
    print("="*80)
    print()

    # Get auction statistics
    auction_stats = df_day.groupby('AUCTION_ID').agg({
        'CAMPAIGN_ID': 'nunique',
        'FINAL_BID_DOLLARS': 'mean',
        'value': 'mean'
    }).reset_index()
    auction_stats.columns = ['AUCTION_ID', 'num_campaigns', 'avg_bid', 'avg_value']

    # Filter to auctions with min campaigns
    auction_stats = auction_stats[auction_stats['num_campaigns'] >= args.min_campaigns]
    print(f"Auctions with >={args.min_campaigns} campaigns: {len(auction_stats):,}")

    if len(auction_stats) < args.num_auctions:
        print(f"WARNING: Only {len(auction_stats)} auctions available, using all")
        selected_auctions = auction_stats['AUCTION_ID'].values
    else:
        # Stratified sampling by activity level
        auction_stats['activity_quartile'] = pd.qcut(auction_stats['num_campaigns'],
                                                       q=4, labels=False, duplicates='drop')

        # Sample proportionally from each quartile
        selected_auctions = []
        per_quartile = args.num_auctions // 4

        for q in range(4):
            q_auctions = auction_stats[auction_stats['activity_quartile'] == q]['AUCTION_ID'].values
            n_sample = min(per_quartile, len(q_auctions))
            selected = np.random.choice(q_auctions, size=n_sample, replace=False)
            selected_auctions.extend(selected)

        # Fill remaining if needed
        if len(selected_auctions) < args.num_auctions:
            remaining = args.num_auctions - len(selected_auctions)
            available = set(auction_stats['AUCTION_ID']) - set(selected_auctions)
            selected_auctions.extend(np.random.choice(list(available), size=remaining, replace=False))

        selected_auctions = np.array(selected_auctions[:args.num_auctions])

    print(f"Selected {len(selected_auctions)} auctions")
    print()

    # Filter to selected auctions
    df_market = df_day[df_day['AUCTION_ID'].isin(selected_auctions)].copy()

    print(f"Market bids: {len(df_market):,}")
    print(f"Market campaigns: {df_market['CAMPAIGN_ID'].nunique():,}")
    print()

# Build campaign-level data
print("="*80)
print("BUILDING CAMPAIGN-LEVEL DATA")
print("="*80)
print()

# Aggregate to campaign level (for this day)
campaign_agg = df_market.groupby('CAMPAIGN_ID').agg({
    'PACING': 'mean',
    'FINAL_BID_DOLLARS': 'sum',  # total spend in selected auctions
    'PRICE_DOLLARS': lambda x: (x * df_market.loc[x.index, 'IS_WINNER']).sum(),  # actual spend
    'value': 'mean',
    'AUCTION_ID': 'nunique'
}).reset_index()

campaign_agg.columns = ['CAMPAIGN_ID', 'pacing_mean', 'total_bid_amount',
                         'actual_spend', 'avg_value', 'num_auctions_bid']

# Estimate daily budget using specified method
if args.budget_method == 'spend_pacing':
    # Original method: budget = spend / pacing
    # Issue: underestimates for campaigns that lose most auctions
    campaign_agg['estimated_daily_budget'] = campaign_agg['actual_spend'] / np.maximum(campaign_agg['pacing_mean'], 0.05)
    campaign_agg.loc[campaign_agg['pacing_mean'] < 0.1, 'estimated_daily_budget'] = campaign_agg['actual_spend'] * 2

elif args.budget_method == 'sum_bids':
    # Alternative: budget = sum of all bids
    # Logic: if willing to bid X total, must have budget >= X
    campaign_agg['estimated_daily_budget'] = campaign_agg['total_bid_amount']

elif args.budget_method == 'hybrid':
    # Hybrid: max(spend/pacing, sum_bids)
    # Takes the more conservative (higher) estimate
    budget_from_spend = campaign_agg['actual_spend'] / np.maximum(campaign_agg['pacing_mean'], 0.05)
    budget_from_spend.loc[campaign_agg['pacing_mean'] < 0.1] = campaign_agg['actual_spend'] * 2
    budget_from_bids = campaign_agg['total_bid_amount']
    campaign_agg['estimated_daily_budget'] = np.maximum(budget_from_spend, budget_from_bids)

print(f"Campaigns in market: {len(campaign_agg)}")
print()
print(f"Budget estimation method: {args.budget_method}")
print("Budget estimates:")
print(f"  Mean: ${campaign_agg['estimated_daily_budget'].mean():.2f}")
print(f"  Median: ${campaign_agg['estimated_daily_budget'].median():.2f}")
print(f"  p10-p90: ${campaign_agg['estimated_daily_budget'].quantile(0.10):.2f} - ${campaign_agg['estimated_daily_budget'].quantile(0.90):.2f}")
print()
print("Pacing distribution:")
print(f"  Mean: {campaign_agg['pacing_mean'].mean():.4f}")
print(f"  Median: {campaign_agg['pacing_mean'].median():.4f}")
print(f"  % >= 0.95: {(campaign_agg['pacing_mean'] >= 0.95).mean()*100:.1f}%")
print()

# Build valuation matrix V[i,j]
print("="*80)
print("BUILDING VALUATION MATRIX V[i,j]")
print("="*80)
print()

# Get unique campaigns and auctions
campaigns = sorted(df_market['CAMPAIGN_ID'].unique())
auctions = selected_auctions

n_campaigns = len(campaigns)
n_auctions = len(auctions)

print(f"Matrix shape: {n_campaigns} campaigns × {n_auctions} auctions")

# Create campaign/auction index mappings
campaign_to_idx = {c: i for i, c in enumerate(campaigns)}
auction_to_idx = {a: j for j, a in enumerate(auctions)}

# Initialize V matrix
V = np.zeros((n_campaigns, n_auctions))

# Fill V matrix
for _, row in df_market.iterrows():
    i = campaign_to_idx[row['CAMPAIGN_ID']]
    j = auction_to_idx[row['AUCTION_ID']]
    # Use value (CONVERSION_RATE × AOV)
    V[i, j] = row['value']

# If campaign bid multiple times on same auction, keep max value
# (already handled by above loop - last value wins, but let's be explicit)
df_market_dedup = df_market.groupby(['CAMPAIGN_ID', 'AUCTION_ID'])['value'].max().reset_index()
for _, row in df_market_dedup.iterrows():
    i = campaign_to_idx[row['CAMPAIGN_ID']]
    j = auction_to_idx[row['AUCTION_ID']]
    V[i, j] = row['value']

# Compute sparsity
sparsity = (V == 0).mean()
print(f"Sparsity: {sparsity*100:.1f}% zero entries")
print(f"Non-zero entries: {(V > 0).sum():,}")
print()

print("Value statistics:")
print(f"  Mean (non-zero): ${V[V > 0].mean():.4f}")
print(f"  Median (non-zero): ${np.median(V[V > 0]):.4f}")
print(f"  p10-p90: ${np.quantile(V[V > 0], 0.10):.4f} - ${np.quantile(V[V > 0], 0.90):.4f}")
print()

# Extract B vector and lambda_obs
B = np.zeros(n_campaigns)
lambda_obs = np.zeros(n_campaigns)

for i, campaign_id in enumerate(campaigns):
    campaign_data = campaign_agg[campaign_agg['CAMPAIGN_ID'] == campaign_id].iloc[0]
    B[i] = campaign_data['estimated_daily_budget']
    lambda_obs[i] = campaign_data['pacing_mean']

print("="*80)
print("FPPE INPUTS SUMMARY")
print("="*80)
print()

print(f"B vector (budgets):")
print(f"  Shape: ({len(B)},)")
print(f"  Mean: ${B.mean():.2f}")
print(f"  Median: ${np.median(B):.2f}")
print(f"  Range: ${B.min():.2f} - ${B.max():.2f}")
print()

print(f"V matrix (valuations):")
print(f"  Shape: {V.shape}")
print(f"  Non-zero: {(V > 0).sum():,} ({(V > 0).mean()*100:.1f}%)")
print(f"  Mean (non-zero): ${V[V > 0].mean():.4f}")
print()

print(f"λ_obs vector (observed pacing):")
print(f"  Shape: ({len(lambda_obs)},)")
print(f"  Mean: {lambda_obs.mean():.4f}")
print(f"  Median: {np.median(lambda_obs):.4f}")
print(f"  % >= 0.95: {(lambda_obs >= 0.95).mean()*100:.1f}%")
print()

# Save to npz
suffix = "_full" if args.full_day else ""
budget_suffix = f"_{args.budget_method}" if args.budget_method != "spend_pacing" else ""
output_file = f"market_{args.date}_p5{suffix}{budget_suffix}.npz"
print(f"Saving to {output_file}...")

np.savez(output_file,
         B=B,
         V=V,
         lambda_obs=lambda_obs,
         campaign_ids=np.array(campaigns),
         auction_ids=auctions,
         date=str(target_date),
         num_campaigns=n_campaigns,
         num_auctions=n_auctions,
         sparsity=sparsity,
         budget_method=args.budget_method)

print(f"✓ Saved {output_file}")
print()

print("="*80)
print("MARKET EXTRACTION COMPLETE")
print("="*80)
print()
print(f"Use this file with FPPE solver (script 14) to find optimal pacing multipliers.")
print()
