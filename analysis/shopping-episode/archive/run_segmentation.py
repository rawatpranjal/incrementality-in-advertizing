"""
Run segmentation/heterogeneity analyses on the 28,619 row panel.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pyfixest as pf

DATA_DIR = Path('/Users/pranjal/Code/marketplace-incrementality/shopping-episode/archive/data')

print("=" * 80)
print("SEGMENTATION / HETEROGENEITY ANALYSES")
print("=" * 80)

# Load panel
panel = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
print(f"\nPanel: {len(panel):,} rows")

# ============================================================================
# 1. RANK HETEROGENEITY
# ============================================================================
print("\n" + "=" * 80)
print("1. RANK HETEROGENEITY")
print("   Model: Y ~ C + C×TopRank | user + week + vendor")
print("   TopRank = 1 if avg_rank ≤ 3")
print("=" * 80)

# Panel already has avg_rank from rebuild
print(f"Rows with avg_rank: {panel['avg_rank'].notna().sum():,}")

# Create TopRank indicator (handle NaN)
panel['TopRank'] = (panel['avg_rank'] <= 3).fillna(False).astype(int)
panel['C_x_TopRank'] = panel['C'] * panel['TopRank']

print(f"\nRows with rank data: {panel['avg_rank'].notna().sum():,}")
print(f"TopRank=1: {panel['TopRank'].sum():,} ({panel['TopRank'].mean()*100:.1f}%)")

# Run heterogeneity model
try:
    m = pf.feols("Y ~ C + C_x_TopRank | user_id + year_week + vendor_id",
                 data=panel.dropna(subset=['avg_rank']),
                 vcov={'CRV1': 'user_id'})

    print(f"\n| Coefficient | β | SE | p |")
    print(f"|-------------|---|----|---|")
    print(f"| C | {m.coef()['C']:.4f} | {m.se()['C']:.4f} | {m.pvalue()['C']:.4f} |")
    print(f"| C × TopRank | {m.coef()['C_x_TopRank']:.4f} | {m.se()['C_x_TopRank']:.4f} | {m.pvalue()['C_x_TopRank']:.4f} |")

    print(f"\nInterpretation:")
    print(f"  - Base effect (non-top rank): β = {m.coef()['C']:.3f}")
    print(f"  - Top rank effect: β = {m.coef()['C'] + m.coef()['C_x_TopRank']:.3f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# 2. CROSS-VENDOR EFFECTS
# ============================================================================
print("\n" + "=" * 80)
print("2. CROSS-VENDOR EFFECTS")
print("   Model: Y ~ C + C_other | user + week + vendor")
print("   C_other = clicks on OTHER vendors in same user-week")
print("=" * 80)

# Calculate total clicks per user-week
user_week_clicks = panel.groupby(['user_id', 'year_week'])['C'].sum().reset_index(name='C_total')

# Merge back
panel = panel.merge(user_week_clicks, on=['user_id', 'year_week'], how='left')
panel['C_other'] = panel['C_total'] - panel['C']

print(f"\nMean C_other: {panel['C_other'].mean():.2f}")

try:
    m = pf.feols("Y ~ C + C_other | user_id + year_week + vendor_id",
                 data=panel,
                 vcov={'CRV1': 'user_id'})

    print(f"\n| Coefficient | β | SE | p |")
    print(f"|-------------|---|----|---|")
    print(f"| C (own vendor) | {m.coef()['C']:.4f} | {m.se()['C']:.4f} | {m.pvalue()['C']:.4f} |")
    print(f"| C_other | {m.coef()['C_other']:.4f} | {m.se()['C_other']:.4f} | {m.pvalue()['C_other']:.4f} |")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# 3. LAGGED CLICKS
# ============================================================================
print("\n" + "=" * 80)
print("3. LAGGED CLICKS")
print("   Model: Y ~ C + C_total_lag | user + week + vendor")
print("   C_total_lag = total clicks in previous week")
print("=" * 80)

# Get lagged total clicks
user_week_total = panel.groupby(['user_id', 'year_week'])['C'].sum().reset_index(name='C_week_total')

# Parse week for lagging
user_week_total['year'] = user_week_total['year_week'].str[:4].astype(int)
user_week_total['week'] = user_week_total['year_week'].str[-2:].astype(int)

# Create lag (simplistic - just shift within user)
user_week_total = user_week_total.sort_values(['user_id', 'year', 'week'])
user_week_total['C_total_lag'] = user_week_total.groupby('user_id')['C_week_total'].shift(1)

# Merge back
panel = panel.merge(user_week_total[['user_id', 'year_week', 'C_total_lag']],
                    on=['user_id', 'year_week'], how='left')

panel_lag = panel.dropna(subset=['C_total_lag'])
print(f"\nRows with lag data: {len(panel_lag):,}")

try:
    m = pf.feols("Y ~ C + C_total_lag | user_id + year_week + vendor_id",
                 data=panel_lag,
                 vcov={'CRV1': 'user_id'})

    print(f"\n| Coefficient | β | SE | p |")
    print(f"|-------------|---|----|---|")
    print(f"| C | {m.coef()['C']:.4f} | {m.se()['C']:.4f} | {m.pvalue()['C']:.4f} |")
    print(f"| C_total_lag | {m.coef()['C_total_lag']:.4f} | {m.se()['C_total_lag']:.4f} | {m.pvalue()['C_total_lag']:.4f} |")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# 4. USER-WEEK AGGREGATION
# ============================================================================
print("\n" + "=" * 80)
print("4. USER-WEEK AGGREGATION")
print("   Model: Y_total ~ C_total | user + week")
print("   Aggregates across vendors within user-week")
print("=" * 80)

# Aggregate to user-week
user_week = panel.groupby(['user_id', 'year_week']).agg({
    'C': 'sum',
    'Y': 'sum'
}).reset_index()
user_week.columns = ['user_id', 'year_week', 'C_total', 'Y_total']

print(f"\nUser-week observations: {len(user_week):,}")
print(f"Mean C_total: {user_week['C_total'].mean():.2f}")
print(f"Mean Y_total: ${user_week['Y_total'].mean():.2f}")

try:
    m = pf.feols("Y_total ~ C_total | user_id + year_week",
                 data=user_week,
                 vcov={'CRV1': 'user_id'})

    print(f"\n| Coefficient | β | SE | p |")
    print(f"|-------------|---|----|---|")
    print(f"| C_total | {m.coef()['C_total']:.4f} | {m.se()['C_total']:.4f} | {m.pvalue()['C_total']:.4f} |")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# 5. CONVERSION BY RANK
# ============================================================================
print("\n" + "=" * 80)
print("5. CONVERSION BY RANK POSITION")
print("=" * 80)

# Reload clean panel (already has avg_rank)
panel = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')

panel['rank_bucket'] = pd.cut(panel['avg_rank'],
                               bins=[0, 1, 2, 3, 5, 10, 100],
                               labels=['1', '2', '3', '4-5', '6-10', '11+'])

conv_by_rank = panel.groupby('rank_bucket').agg({
    'C': 'sum',
    'Y': ['sum', 'count'],
    'D': 'sum'
}).reset_index()
conv_by_rank.columns = ['Rank', 'Clicks', 'Total_Spend', 'N', 'Conversions']
conv_by_rank['Conv_Rate'] = (conv_by_rank['Conversions'] / conv_by_rank['N'] * 100).round(2)

print(f"\n| Rank | N | Conversions | Conv Rate | Total Spend |")
print(f"|------|---|-------------|-----------|-------------|")
for _, row in conv_by_rank.iterrows():
    print(f"| {row['Rank']} | {row['N']:,} | {row['Conversions']:.0f} | {row['Conv_Rate']:.2f}% | ${row['Total_Spend']:.0f} |")

# ============================================================================
# 6. COMPARISON SHOPPING BEHAVIOR
# ============================================================================
print("\n" + "=" * 80)
print("6. COMPARISON SHOPPING BEHAVIOR")
print("=" * 80)

# User-level stats
user_stats = panel.groupby('user_id').agg({
    'C': 'sum',
    'Y': 'sum',
    'vendor_id': 'nunique',
    'D': 'max'  # 1 if any purchase
}).reset_index()
user_stats.columns = ['user_id', 'total_clicks', 'total_spend', 'n_vendors', 'is_converter']

converters = user_stats[user_stats['is_converter'] == 1]
non_converters = user_stats[user_stats['is_converter'] == 0]

print(f"\nConverters ({len(converters):,} users):")
print(f"  Mean clicks: {converters['total_clicks'].mean():.1f}")
print(f"  Mean vendors clicked: {converters['n_vendors'].mean():.1f}")
print(f"  Click-to-purchase ratio: {converters['total_clicks'].sum() / len(converters):.1f}")

print(f"\nNon-converters ({len(non_converters):,} users):")
print(f"  Mean clicks: {non_converters['total_clicks'].mean():.1f}")
print(f"  Mean vendors clicked: {non_converters['n_vendors'].mean():.1f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
Key Heterogeneity Findings:

1. RANK HETEROGENEITY: Top-rank clicks (rank ≤ 3) have POSITIVE effect
   - Base effect (non-top): ~0.24
   - Top rank interaction: ~0.23 additional
   - Combined for top rank: ~0.47

2. CROSS-VENDOR: Clicking other vendors doesn't affect focal vendor spend

3. LAGGED CLICKS: Past week's clicks don't predict current spend

4. USER-WEEK: At aggregated level, positive but weak effect (~0.09)

5. COMPARISON SHOPPING: Converters click 9x more than non-converters
   - Suggests clicking = shopping intent, not purchase intent for specific vendor
""")

print("\nDone.")
