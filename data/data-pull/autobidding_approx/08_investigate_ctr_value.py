"""
Investigate why value < bid by testing different CTR sources and formula interpretations.
Key question: Is CONVERSION_RATE = P(purchase|impression) or P(purchase|click)?
"""

import pandas as pd
import numpy as np

print("="*80)
print("INVESTIGATING VALUE < BID ISSUE")
print("="*80)
print()

# Load data
print("Loading data...")
df_results = pd.read_parquet('../data/raw_auctions_results_20251011.parquet')
df_impressions = pd.read_parquet('../data/raw_impressions_20251011.parquet')
df_clicks = pd.read_parquet('../data/raw_clicks_20251011.parquet')
df_purchases = pd.read_parquet('../data/raw_purchases_20251011.parquet')
df_catalog = pd.read_parquet('../data/catalog_20251011.parquet')

print(f"  Bids: {len(df_results):,}")
print(f"  Impressions: {len(df_impressions):,}")
print(f"  Clicks: {len(df_clicks):,}")
print(f"  Purchases: {len(df_purchases):,}")
print()

# Convert bid prices
df_results['FINAL_BID_DOLLARS'] = df_results['FINAL_BID'] / 100.0

# Map catalog prices
product_prices = df_catalog.set_index('PRODUCT_ID')['PRICE'].to_dict()
df_results['AOV'] = df_results['PRODUCT_ID'].map(product_prices)
median_aov = df_catalog['PRICE'].median()
df_results['AOV'] = df_results['AOV'].fillna(median_aov)

print("="*80)
print("STEP 1: CALCULATE ACTUAL CTRs")
print("="*80)
print()

# Calculate campaign-level CTR from actual impression/click data
campaign_impressions = df_impressions.groupby('CAMPAIGN_ID').size()
campaign_clicks = df_clicks.groupby('CAMPAIGN_ID').size()

print(f"Campaigns with impressions: {len(campaign_impressions):,}")
print(f"Campaigns with clicks: {len(campaign_clicks):,}")
print()

# Calculate CTR for campaigns with data
campaign_ctr = {}
for campaign_id in campaign_impressions.index:
    n_impressions = campaign_impressions[campaign_id]
    n_clicks = campaign_clicks.get(campaign_id, 0)
    if n_impressions > 0:
        campaign_ctr[campaign_id] = n_clicks / n_impressions

print(f"Campaigns with CTR data: {len(campaign_ctr):,}")
if len(campaign_ctr) > 0:
    ctr_values = list(campaign_ctr.values())
    print(f"  Mean actual CTR: {np.mean(ctr_values):.6f} ({np.mean(ctr_values)*100:.2f}%)")
    print(f"  Median actual CTR: {np.median(ctr_values):.6f} ({np.median(ctr_values)*100:.2f}%)")
    print(f"  p10: {np.quantile(ctr_values, 0.10):.6f}")
    print(f"  p90: {np.quantile(ctr_values, 0.90):.6f}")
print()

# Map actual CTR to bids
df_results['actual_CTR'] = df_results['CAMPAIGN_ID'].map(campaign_ctr)

# Calculate QUALITY proxy (what we used before)
quality_min = df_results['QUALITY'].min()
quality_max = df_results['QUALITY'].max()
quality_normalized = (df_results['QUALITY'] - quality_min) / (quality_max - quality_min + 1e-9)
df_results['pCTR_proxy'] = 0.01 + quality_normalized * 0.14

print("CTR Sources Comparison:")
print(f"  QUALITY proxy mean: {df_results['pCTR_proxy'].mean():.6f} ({df_results['pCTR_proxy'].mean()*100:.2f}%)")
print(f"  Actual CTR mean (where available): {df_results['actual_CTR'].mean():.6f} ({df_results['actual_CTR'].mean()*100:.2f}%)")
print(f"  Bids with actual CTR: {df_results['actual_CTR'].notna().sum():,} ({df_results['actual_CTR'].notna().mean()*100:.1f}%)")
print()

# Check correlation between QUALITY and actual CTR
has_both = df_results[df_results['actual_CTR'].notna()][['QUALITY', 'actual_CTR']].drop_duplicates()
if len(has_both) > 100:
    corr = has_both['QUALITY'].corr(has_both['actual_CTR'])
    print(f"Correlation (QUALITY, actual_CTR): {corr:.4f}")
    if abs(corr) < 0.3:
        print("  ⚠️  Weak correlation - QUALITY is poor proxy for CTR!")
    else:
        print("  ✓ Reasonable correlation - QUALITY captures some CTR signal")
print()

print("="*80)
print("STEP 2: TEST THREE VALUE FORMULAS")
print("="*80)
print()

# Scenario A: CONVERSION_RATE already includes CTR (full impression→purchase probability)
print("SCENARIO A: value = CONVERSION_RATE × AOV")
print("-"*80)
print("Assumption: CONVERSION_RATE = P(purchase|impression)")
print()

df_results['value_A'] = df_results['CONVERSION_RATE'] * df_results['AOV']
df_results['roas_A'] = df_results['value_A'] / (df_results['FINAL_BID_DOLLARS'] + 1e-9)

# Filter reasonable ROAS
roas_A = df_results[(df_results['roas_A'] > 0) & (df_results['roas_A'] < 100)]['roas_A']

print(f"Value Statistics:")
print(f"  Mean: ${df_results['value_A'].mean():.4f}")
print(f"  Median: ${df_results['value_A'].median():.4f}")
print(f"  p10: ${df_results['value_A'].quantile(0.10):.4f}")
print(f"  p90: ${df_results['value_A'].quantile(0.90):.4f}")
print()
print(f"ROAS Statistics:")
print(f"  Mean: {roas_A.mean():.2f}x")
print(f"  Median: {roas_A.median():.2f}x")
print(f"  p10: {roas_A.quantile(0.10):.2f}x")
print(f"  p90: {roas_A.quantile(0.90):.2f}x")
print()

if roas_A.median() >= 0.5 and roas_A.median() <= 5.0:
    print("✓ ECONOMICALLY RATIONAL! ROAS in reasonable range (0.5x - 5.0x)")
    print("  → This formula likely CORRECT")
else:
    print("✗ Economically questionable")
print()

# Scenario B: Use actual CTR with CVR (CVR is per-click)
print("SCENARIO B: value = actual_CTR × CONVERSION_RATE × AOV")
print("-"*80)
print("Assumption: CONVERSION_RATE = P(purchase|click)")
print()

df_with_ctr = df_results[df_results['actual_CTR'].notna()].copy()
df_with_ctr['value_B'] = df_with_ctr['actual_CTR'] * df_with_ctr['CONVERSION_RATE'] * df_with_ctr['AOV']
df_with_ctr['roas_B'] = df_with_ctr['value_B'] / (df_with_ctr['FINAL_BID_DOLLARS'] + 1e-9)

roas_B = df_with_ctr[(df_with_ctr['roas_B'] > 0) & (df_with_ctr['roas_B'] < 100)]['roas_B']

print(f"Sample size: {len(df_with_ctr):,} bids with actual CTR")
print()
print(f"Value Statistics:")
print(f"  Mean: ${df_with_ctr['value_B'].mean():.4f}")
print(f"  Median: ${df_with_ctr['value_B'].median():.4f}")
print(f"  p10: ${df_with_ctr['value_B'].quantile(0.10):.4f}")
print(f"  p90: ${df_with_ctr['value_B'].quantile(0.90):.4f}")
print()
print(f"ROAS Statistics:")
print(f"  Mean: {roas_B.mean():.2f}x")
print(f"  Median: {roas_B.median():.2f}x")
print(f"  p10: {roas_B.quantile(0.10):.2f}x")
print(f"  p90: {roas_B.quantile(0.90):.2f}x")
print()

if roas_B.median() >= 0.5 and roas_B.median() <= 5.0:
    print("✓ ECONOMICALLY RATIONAL! ROAS in reasonable range")
    print("  → This formula likely CORRECT")
else:
    print("✗ Economically questionable")
print()

# Scenario C: Use QUALITY proxy with CVR (what we did originally)
print("SCENARIO C: value = pCTR_proxy × CONVERSION_RATE × AOV (ORIGINAL)")
print("-"*80)
print("Assumption: CONVERSION_RATE = P(purchase|click), pCTR from QUALITY")
print()

df_results['value_C'] = df_results['pCTR_proxy'] * df_results['CONVERSION_RATE'] * df_results['AOV']
df_results['roas_C'] = df_results['value_C'] / (df_results['FINAL_BID_DOLLARS'] + 1e-9)

roas_C = df_results[(df_results['roas_C'] > 0) & (df_results['roas_C'] < 100)]['roas_C']

print(f"Value Statistics:")
print(f"  Mean: ${df_results['value_C'].mean():.4f}")
print(f"  Median: ${df_results['value_C'].median():.4f}")
print(f"  p10: ${df_results['value_C'].quantile(0.10):.4f}")
print(f"  p90: ${df_results['value_C'].quantile(0.90):.4f}")
print()
print(f"ROAS Statistics:")
print(f"  Mean: {roas_C.mean():.2f}x")
print(f"  Median: {roas_C.median():.2f}x")
print(f"  p10: {roas_C.quantile(0.10):.2f}x")
print(f"  p90: {roas_C.quantile(0.90):.2f}x")
print()

if roas_C.median() >= 0.5 and roas_C.median() <= 5.0:
    print("✓ ECONOMICALLY RATIONAL!")
else:
    print("✗ ECONOMICALLY IRRATIONAL - campaigns losing money")
    print("  → This formula is WRONG")
print()

print("="*80)
print("STEP 3: EMPIRICAL VALIDATION")
print("="*80)
print()

# Calculate actual revenue per impression (ground truth)
print("Calculating actual revenue per impression from purchases...")
print()

# Join impressions to purchases (within attribution window)
df_impressions['OCCURRED_AT'] = pd.to_datetime(df_impressions['OCCURRED_AT'])
df_purchases['PURCHASED_AT'] = pd.to_datetime(df_purchases['PURCHASED_AT'])

# For each impression, check if purchase occurred within 24 hours
attribution_window_hours = 24

# Sample for performance (full join would be huge)
sample_impressions = df_impressions.sample(min(10000, len(df_impressions)), random_state=42)

impressions_with_purchases = sample_impressions.merge(
    df_purchases[['USER_ID', 'PURCHASED_AT', 'UNIT_PRICE', 'QUANTITY']],
    left_on='USER_ID',
    right_on='USER_ID',
    how='left'
)

impressions_with_purchases['time_diff_hours'] = (
    impressions_with_purchases['PURCHASED_AT'] - impressions_with_purchases['OCCURRED_AT']
).dt.total_seconds() / 3600

# Count attributed purchases (within window, after impression)
impressions_with_purchases['attributed'] = (
    (impressions_with_purchases['time_diff_hours'] > 0) &
    (impressions_with_purchases['time_diff_hours'] <= attribution_window_hours)
)

impressions_with_purchases['purchase_value'] = (
    impressions_with_purchases['UNIT_PRICE'] * impressions_with_purchases['QUANTITY'] / 100.0
).fillna(0)

impressions_with_purchases.loc[~impressions_with_purchases['attributed'], 'purchase_value'] = 0

# Calculate actual revenue per impression
actual_revenue_per_impression = impressions_with_purchases.groupby('INTERACTION_ID')['purchase_value'].sum().mean()

print(f"Sample size: {len(sample_impressions):,} impressions")
print(f"Attributed purchases: {impressions_with_purchases['attributed'].sum():,}")
print(f"Attribution rate: {impressions_with_purchases['attributed'].mean()*100:.4f}%")
print()
print(f"ACTUAL revenue per impression: ${actual_revenue_per_impression:.4f}")
print()

print("Comparison to calculated values:")
print(f"  Scenario A (CVR only):     ${df_results['value_A'].median():.4f}")
print(f"  Scenario B (CTR × CVR):    ${df_with_ctr['value_B'].median():.4f}")
print(f"  Scenario C (proxy × CVR):  ${df_results['value_C'].median():.4f}")
print(f"  ACTUAL (from purchases):   ${actual_revenue_per_impression:.4f}")
print()

# Determine which is closest
diffs = {
    'A': abs(df_results['value_A'].median() - actual_revenue_per_impression),
    'B': abs(df_with_ctr['value_B'].median() - actual_revenue_per_impression),
    'C': abs(df_results['value_C'].median() - actual_revenue_per_impression)
}

best_scenario = min(diffs, key=diffs.get)
print(f"Closest to actual: Scenario {best_scenario}")
print()

print("="*80)
print("STEP 4: CONVERSION RATE INTERPRETATION")
print("="*80)
print()

# Try to infer what CONVERSION_RATE represents
print("Testing: Is CONVERSION_RATE the full impression→purchase probability?")
print()

# Calculate implied conversion rate from impressions
sample_campaigns = impressions_with_purchases.groupby('CAMPAIGN_ID').agg({
    'attributed': ['sum', 'count']
}).reset_index()
sample_campaigns.columns = ['CAMPAIGN_ID', 'purchases', 'impressions']
sample_campaigns['empirical_cvr'] = sample_campaigns['purchases'] / sample_campaigns['impressions']

# Compare to CONVERSION_RATE field
campaign_cvr_field = df_results.groupby('CAMPAIGN_ID')['CONVERSION_RATE'].mean()

comparison = sample_campaigns.set_index('CAMPAIGN_ID').join(
    campaign_cvr_field.rename('field_cvr')
)

comparison = comparison.dropna()

if len(comparison) > 10:
    corr = comparison['empirical_cvr'].corr(comparison['field_cvr'])
    print(f"Correlation (empirical CVR from purchases, CONVERSION_RATE field): {corr:.4f}")
    print()
    print(f"Empirical CVR statistics:")
    print(f"  Mean: {comparison['empirical_cvr'].mean():.6f}")
    print(f"  Median: {comparison['empirical_cvr'].median():.6f}")
    print()
    print(f"CONVERSION_RATE field statistics:")
    print(f"  Mean: {comparison['field_cvr'].mean():.6f}")
    print(f"  Median: {comparison['field_cvr'].median():.6f}")
    print()

    ratio = comparison['field_cvr'].mean() / (comparison['empirical_cvr'].mean() + 1e-9)
    print(f"Field / Empirical ratio: {ratio:.2f}x")
    print()

    if abs(ratio - 1.0) < 0.5:
        print("✓ CONVERSION_RATE appears to be P(purchase|impression)")
        print("  → Use formula: value = CONVERSION_RATE × AOV")
    else:
        print("⚠️  CONVERSION_RATE differs from empirical rate")
        print("  → May be prediction, not actual rate")
        print("  → Or may be click-based, not impression-based")

print()

print("="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print()

# Determine best approach
recommendations = []

if roas_A.median() >= 0.5 and roas_A.median() <= 5.0:
    recommendations.append(('A', 'value = CONVERSION_RATE × AOV', roas_A.median()))

if roas_B.median() >= 0.5 and roas_B.median() <= 5.0:
    recommendations.append(('B', 'value = actual_CTR × CONVERSION_RATE × AOV', roas_B.median()))

if len(recommendations) == 0:
    print("❌ NO FORMULA PRODUCES RATIONAL ROAS")
    print()
    print("FALLBACK RECOMMENDATION:")
    print("  Use observed bids as valuations directly")
    print("  Do not attempt to decompose value = pCTR × pCVR × AOV")
    print()
else:
    print("✓ ECONOMICALLY RATIONAL FORMULAS:")
    print()
    for scenario, formula, roas in recommendations:
        print(f"  Scenario {scenario}: {formula}")
        print(f"    Median ROAS: {roas:.2f}x")
        print()

    if len(recommendations) == 1:
        best = recommendations[0]
        print(f"RECOMMENDED FORMULA: {best[1]}")
    else:
        # Both work, choose simpler one
        print("Both formulas work! Recommend Scenario A (simpler):")
        print("  value = CONVERSION_RATE × AOV")

print()
print("This investigation will be documented in claude.md")
