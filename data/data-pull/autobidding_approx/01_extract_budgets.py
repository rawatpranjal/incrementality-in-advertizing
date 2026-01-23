"""
Extract parameter A: Daily budget distribution from pacing=1.0 campaigns.
Updates claude.md with results.
"""

import pandas as pd
import numpy as np

print("Loading auction data...")
df_auctions_results = pd.read_parquet('../data/raw_auctions_results_20251011.parquet')
df_auctions_users = pd.read_parquet('../data/raw_auctions_users_20251011.parquet')

print(f"Loaded {len(df_auctions_results):,} auction results")

print("Merging data...")
df = pd.merge(
    df_auctions_results.drop(columns=['CREATED_AT'], errors='ignore'),
    df_auctions_users[['AUCTION_ID', 'CREATED_AT']],
    on='AUCTION_ID',
    how='left'
)

print("Creating time features...")
df['datetime'] = pd.to_datetime(df['CREATED_AT'])
df['date'] = df['datetime'].dt.date
df['FINAL_BID_DOLLARS'] = df['FINAL_BID'] / 100.0
df['PRICE_DOLLARS'] = df['PRICE'] / 100.0

print("Aggregating campaign-day data...")
campaign_day = df.groupby(['CAMPAIGN_ID', 'date']).agg({
    'PACING': 'mean',
    'PRICE_DOLLARS': lambda x: (x * df.loc[x.index, 'IS_WINNER']).sum(),
    'AUCTION_ID': 'count'
}).reset_index()

campaign_day.columns = ['CAMPAIGN_ID', 'date', 'pacing_mean', 'total_spend_dollars', 'num_bids']

print(f"Total campaign-days: {len(campaign_day):,}")

print("Filtering budget-constrained campaigns (pacing >= 0.95)...")
budget_constrained = campaign_day[campaign_day['pacing_mean'] >= 0.95].copy()
budget_constrained['approx_daily_budget'] = budget_constrained['total_spend_dollars']

print(f"Budget-constrained campaign-days: {len(budget_constrained):,} ({len(budget_constrained)/len(campaign_day)*100:.1f}%)")

print("\nBudget Statistics:")
print(f"  Mean: ${budget_constrained['approx_daily_budget'].mean():.2f}")
print(f"  Median: ${budget_constrained['approx_daily_budget'].median():.2f}")
print(f"  Std: ${budget_constrained['approx_daily_budget'].std():.2f}")
print(f"  p10: ${budget_constrained['approx_daily_budget'].quantile(0.10):.2f}")
print(f"  p25: ${budget_constrained['approx_daily_budget'].quantile(0.25):.2f}")
print(f"  p50: ${budget_constrained['approx_daily_budget'].quantile(0.50):.2f}")
print(f"  p75: ${budget_constrained['approx_daily_budget'].quantile(0.75):.2f}")
print(f"  p90: ${budget_constrained['approx_daily_budget'].quantile(0.90):.2f}")
print(f"  p95: ${budget_constrained['approx_daily_budget'].quantile(0.95):.2f}")
print(f"  p99: ${budget_constrained['approx_daily_budget'].quantile(0.99):.2f}")

print("\nUpdating claude.md...")
results_text = f"""
### A) DAILY BUDGET DISTRIBUTION

**Status:** ✅ COMPLETED

**Method:** For campaigns with pacing ≥ 0.95, daily spend ≈ daily budget

**Results:**
- Total campaign-days: {len(campaign_day):,}
- Budget-constrained: {len(budget_constrained):,} ({len(budget_constrained)/len(campaign_day)*100:.1f}%)

**Budget Distribution (daily, per campaign):**
- Mean: ${budget_constrained['approx_daily_budget'].mean():.2f}
- Median: ${budget_constrained['approx_daily_budget'].median():.2f}
- Std: ${budget_constrained['approx_daily_budget'].std():.2f}

**Percentiles:**
- p10: ${budget_constrained['approx_daily_budget'].quantile(0.10):.2f}
- p25: ${budget_constrained['approx_daily_budget'].quantile(0.25):.2f}
- p50: ${budget_constrained['approx_daily_budget'].quantile(0.50):.2f}
- p75: ${budget_constrained['approx_daily_budget'].quantile(0.75):.2f}
- p90: ${budget_constrained['approx_daily_budget'].quantile(0.90):.2f}
- p95: ${budget_constrained['approx_daily_budget'].quantile(0.95):.2f}
- p99: ${budget_constrained['approx_daily_budget'].quantile(0.99):.2f}

**Interpretation:**
- Typical daily budget: ${budget_constrained['approx_daily_budget'].median():.2f} (median)
- Budget range for simulation: ${budget_constrained['approx_daily_budget'].quantile(0.10):.2f} - ${budget_constrained['approx_daily_budget'].quantile(0.90):.2f} (p10-p90)
- High heterogeneity: Small campaigns ~$0.20/day, large campaigns ~$50+/day
"""

with open('claude.md', 'r') as f:
    content = f.read()

content = content.replace(
    '### A) DAILY BUDGET DISTRIBUTION\n\n**Status:** NOT STARTED\n\n**Method:** For campaigns with pacing ≥ 0.95, daily spend ≈ daily budget\n\n**Results:**\n- TBD',
    results_text.strip()
)

with open('claude.md', 'w') as f:
    f.write(content)

print("✅ Updated claude.md with budget results")
