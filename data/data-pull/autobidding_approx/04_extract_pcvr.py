"""
Extract parameter D: Predicted conversion rate (pCVR).
Updates claude.md with results.
"""

import pandas as pd
import numpy as np

print("Loading auction results...")
df = pd.read_parquet('../data/raw_auctions_results_20251011.parquet')

print(f"Loaded {len(df):,} bids")

print("\npCVR Statistics:")
print(f"  Mean: {df['CONVERSION_RATE'].mean():.6f}")
print(f"  Median: {df['CONVERSION_RATE'].median():.6f}")
print(f"  Std: {df['CONVERSION_RATE'].std():.6f}")
print(f"  Min: {df['CONVERSION_RATE'].min():.6f}")
print(f"  Max: {df['CONVERSION_RATE'].max():.6f}")

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print(f"\npCVR Percentiles:")
for q in quantiles:
    val = df['CONVERSION_RATE'].quantile(q)
    print(f"  p{int(q*100):02d}: {val:.6f}")

print("\npCVR by pacing level:")
high_pacing = df[df['PACING'] >= 0.95]
low_pacing = df[df['PACING'] < 0.5]
print(f"  High pacing (>=0.95): Mean={high_pacing['CONVERSION_RATE'].mean():.6f}")
print(f"  Low pacing (<0.5): Mean={low_pacing['CONVERSION_RATE'].mean():.6f}")

print("\nCorrelation with other variables:")
print(f"  pCVR vs QUALITY: {df['CONVERSION_RATE'].corr(df['QUALITY']):.4f}")
print(f"  pCVR vs FINAL_BID: {df['CONVERSION_RATE'].corr(df['FINAL_BID']):.4f}")
print(f"  pCVR vs PACING: {df['CONVERSION_RATE'].corr(df['PACING']):.4f}")

print("\nUpdating claude.md...")
results_text = f"""
### D) PREDICTED CONVERSION RATE (pCVR)

**Status:** ✅ COMPLETED

**Method:** Use CONVERSION_RATE field

**Results:**
- Total bids with pCVR: {len(df):,}

**Overall pCVR Distribution:**
- Mean: {df['CONVERSION_RATE'].mean():.6f}
- Median: {df['CONVERSION_RATE'].median():.6f}
- Std: {df['CONVERSION_RATE'].std():.6f}
- Range: {df['CONVERSION_RATE'].min():.6f} - {df['CONVERSION_RATE'].max():.6f}

**Percentiles:**
- p10: {df['CONVERSION_RATE'].quantile(0.10):.6f}
- p25: {df['CONVERSION_RATE'].quantile(0.25):.6f}
- p50: {df['CONVERSION_RATE'].quantile(0.50):.6f}
- p75: {df['CONVERSION_RATE'].quantile(0.75):.6f}
- p90: {df['CONVERSION_RATE'].quantile(0.90):.6f}
- p95: {df['CONVERSION_RATE'].quantile(0.95):.6f}
- p99: {df['CONVERSION_RATE'].quantile(0.99):.6f}

**By Pacing Level:**
- High pacing (>=0.95): Mean={high_pacing['CONVERSION_RATE'].mean():.6f}
- Low pacing (<0.5): Mean={low_pacing['CONVERSION_RATE'].mean():.6f}

**Correlations:**
- pCVR vs QUALITY: {df['CONVERSION_RATE'].corr(df['QUALITY']):.4f}
- pCVR vs FINAL_BID: {df['CONVERSION_RATE'].corr(df['FINAL_BID']):.4f}
- pCVR vs PACING: {df['CONVERSION_RATE'].corr(df['PACING']):.4f}

**Interpretation:**
- Typical pCVR: {df['CONVERSION_RATE'].median():.6f} (1% conversion rate)
- Range for simulation: {df['CONVERSION_RATE'].quantile(0.10):.6f} - {df['CONVERSION_RATE'].quantile(0.90):.6f} (p10-p90)
- Moderate correlation with QUALITY suggests quality score partially captures CVR
"""

with open('claude.md', 'r') as f:
    content = f.read()

content = content.replace(
    '### D) PREDICTED CONVERSION RATE (pCVR)\n\n**Status:** NOT STARTED\n\n**Method:** Use CONVERSION_RATE field\n\n**From existing analysis:**\n- Mean pCVR: 0.010004\n- Median pCVR: 0.009010\n- Range: 0.000001 - 0.0565\n\n**Results:**\n- TBD',
    results_text.strip()
)

with open('claude.md', 'w') as f:
    f.write(content)

print("✅ Updated claude.md with pCVR results")
