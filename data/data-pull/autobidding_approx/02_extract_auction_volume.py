"""
Extract parameter B: Auction volume (J).
Updates claude.md with results.
"""

import pandas as pd
import numpy as np

print("Loading auction data...")
df_auctions_users = pd.read_parquet('../data/raw_auctions_users_20251011.parquet')
df_auctions_results = pd.read_parquet('../data/raw_auctions_results_20251011.parquet')

print(f"Loaded {len(df_auctions_users):,} auction records")

print("Creating time features...")
df_auctions_users['datetime'] = pd.to_datetime(df_auctions_users['CREATED_AT'])
df_auctions_users['date'] = df_auctions_users['datetime'].dt.date
df_auctions_users['hour'] = df_auctions_users['datetime'].dt.hour

print("\nCalculating auction counts...")
total_auctions = df_auctions_users['AUCTION_ID'].nunique()
print(f"Total unique auctions: {total_auctions:,}")

daily_auctions = df_auctions_users.groupby('date')['AUCTION_ID'].nunique()
print(f"\nAuctions per day:")
print(f"  Mean: {daily_auctions.mean():.0f}")
print(f"  Median: {daily_auctions.median():.0f}")
print(f"  Min: {daily_auctions.min():,}")
print(f"  Max: {daily_auctions.max():,}")

hourly_avg = df_auctions_users.groupby('hour')['AUCTION_ID'].nunique().mean()
print(f"\nAverage auctions per hour: {hourly_avg:.0f}")

# Calculate bids per auction
bids_per_auction = df_auctions_results.groupby('AUCTION_ID').size()
print(f"\nBids per auction (competition):")
print(f"  Mean: {bids_per_auction.mean():.1f}")
print(f"  Median: {bids_per_auction.median():.0f}")
print(f"  p10: {bids_per_auction.quantile(0.10):.0f}")
print(f"  p90: {bids_per_auction.quantile(0.90):.0f}")

print("\nUpdating claude.md...")
results_text = f"""
### B) AUCTION VOLUME (J)

**Status:** ✅ COMPLETED

**Method:** Count unique auctions per day/hour

**Results:**
- Total unique auctions: {total_auctions:,}
- Data period: 14 days (Sept 27 - Oct 10, 2025)

**Daily Auction Volume:**
- Mean: {daily_auctions.mean():.0f} auctions/day
- Median: {daily_auctions.median():.0f} auctions/day
- Range: {daily_auctions.min():,} - {daily_auctions.max():,}

**Hourly Average:**
- Avg per hour: {hourly_avg:.0f} auctions/hour

**Competition Intensity (bids per auction):**
- Mean: {bids_per_auction.mean():.1f} bidders
- Median: {bids_per_auction.median():.0f} bidders
- Range (p10-p90): {bids_per_auction.quantile(0.10):.0f} - {bids_per_auction.quantile(0.90):.0f} bidders

**Interpretation:**
- Typical market: ~{daily_auctions.median():.0f} auctions per day
- High competition: {bids_per_auction.mean():.1f} average bidders per auction
- For simulation: Use J = 10-50 goods (auction opportunities)
- Use num_bidders = 20-50 (reflecting competition level)
"""

with open('claude.md', 'r') as f:
    content = f.read()

content = content.replace(
    '### B) AUCTION VOLUME (J)\n\n**Status:** NOT STARTED\n\n**Method:** Count unique auctions per day/hour\n\n**Results:**\n- TBD',
    results_text.strip()
)

with open('claude.md', 'w') as f:
    f.write(content)

print("✅ Updated claude.md with auction volume results")
