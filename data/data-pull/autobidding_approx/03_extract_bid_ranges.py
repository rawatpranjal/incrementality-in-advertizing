"""
Extract parameter C: Bid price ranges.
Updates claude.md with results.
"""

import pandas as pd
import numpy as np

print("Loading auction results...")
df = pd.read_parquet('../data/raw_auctions_results_20251011.parquet')

print(f"Loaded {len(df):,} bids")

print("\nCreating bid dollar values...")
df['FINAL_BID_DOLLARS'] = df['FINAL_BID'] / 100.0

print("\nBid Statistics (dollars):")
print(f"  Mean: ${df['FINAL_BID_DOLLARS'].mean():.4f}")
print(f"  Median: ${df['FINAL_BID_DOLLARS'].median():.4f}")
print(f"  Std: ${df['FINAL_BID_DOLLARS'].std():.4f}")
print(f"  Min: ${df['FINAL_BID_DOLLARS'].min():.4f}")
print(f"  Max: ${df['FINAL_BID_DOLLARS'].max():.4f}")

quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print(f"\nBid Percentiles:")
for q in quantiles:
    val = df['FINAL_BID_DOLLARS'].quantile(q)
    print(f"  p{int(q*100):02d}: ${val:.4f}")

print("\nBids by pacing level:")
high_pacing = df[df['PACING'] >= 0.95]
low_pacing = df[df['PACING'] < 0.5]
print(f"  High pacing (>=0.95): Mean=${high_pacing['FINAL_BID_DOLLARS'].mean():.4f}, Median=${high_pacing['FINAL_BID_DOLLARS'].median():.4f}")
print(f"  Low pacing (<0.5): Mean=${low_pacing['FINAL_BID_DOLLARS'].mean():.4f}, Median=${low_pacing['FINAL_BID_DOLLARS'].median():.4f}")

print("\nBids by win status:")
winners = df[df['IS_WINNER']]
losers = df[~df['IS_WINNER']]
print(f"  Winners: Mean=${winners['FINAL_BID_DOLLARS'].mean():.4f}, Median=${winners['FINAL_BID_DOLLARS'].median():.4f}")
print(f"  Losers: Mean=${losers['FINAL_BID_DOLLARS'].mean():.4f}, Median=${losers['FINAL_BID_DOLLARS'].median():.4f}")

print("\nUpdating claude.md...")
results_text = f"""
### C) BID PRICE RANGES

**Status:** ✅ COMPLETED

**Method:** Analyze FINAL_BID distribution

**Results:**
- Total bids: {len(df):,}

**Overall Bid Distribution (dollars):**
- Mean: ${df['FINAL_BID_DOLLARS'].mean():.4f}
- Median: ${df['FINAL_BID_DOLLARS'].median():.4f}
- Std: ${df['FINAL_BID_DOLLARS'].std():.4f}
- Range: ${df['FINAL_BID_DOLLARS'].min():.4f} - ${df['FINAL_BID_DOLLARS'].max():.4f}

**Percentiles:**
- p01: ${df['FINAL_BID_DOLLARS'].quantile(0.01):.4f}
- p05: ${df['FINAL_BID_DOLLARS'].quantile(0.05):.4f}
- p10: ${df['FINAL_BID_DOLLARS'].quantile(0.10):.4f}
- p25: ${df['FINAL_BID_DOLLARS'].quantile(0.25):.4f}
- p50: ${df['FINAL_BID_DOLLARS'].quantile(0.50):.4f}
- p75: ${df['FINAL_BID_DOLLARS'].quantile(0.75):.4f}
- p90: ${df['FINAL_BID_DOLLARS'].quantile(0.90):.4f}
- p95: ${df['FINAL_BID_DOLLARS'].quantile(0.95):.4f}
- p99: ${df['FINAL_BID_DOLLARS'].quantile(0.99):.4f}

**By Pacing Level:**
- High pacing (>=0.95): Mean=${high_pacing['FINAL_BID_DOLLARS'].mean():.4f}
- Low pacing (<0.5): Mean=${low_pacing['FINAL_BID_DOLLARS'].mean():.4f}

**Interpretation:**
- Typical bid: ${df['FINAL_BID_DOLLARS'].median():.4f} (median)
- Common range: ${df['FINAL_BID_DOLLARS'].quantile(0.10):.4f} - ${df['FINAL_BID_DOLLARS'].quantile(0.90):.4f} (p10-p90)
- Note: Bids already reflect pacing (FINAL_BID = base_bid × pacing)
- For FPPE: Valuations should be higher than observed bids (since bids = pacing × value)
"""

with open('claude.md', 'r') as f:
    content = f.read()

content = content.replace(
    '### C) BID PRICE RANGES\n\n**Status:** NOT STARTED\n\n**Method:** Analyze FINAL_BID distribution\n\n**From existing analysis:**\n- Mean bid: $0.1180\n- Median bid: $0.0600\n- Range: $0.00 - $1.00\n\n**Results:**\n- TBD',
    results_text.strip()
)

with open('claude.md', 'w') as f:
    f.write(content)

print("✅ Updated claude.md with bid range results")
