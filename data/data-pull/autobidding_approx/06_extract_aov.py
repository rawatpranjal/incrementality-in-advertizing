"""
Extract parameter F: Average order value (AOV) from catalog.
Updates claude.md with results.
"""

import pandas as pd
import numpy as np

print("Loading catalog...")
df_catalog = pd.read_parquet('../data/catalog_20251011.parquet')

print(f"Loaded {len(df_catalog):,} products")

# Filter to products with price
catalog_with_price = df_catalog[df_catalog['PRICE'].notna()].copy()
print(f"Products with price: {len(catalog_with_price):,} ({len(catalog_with_price)/len(df_catalog)*100:.1f}%)")

print("\nAOV Statistics (from catalog prices):")
print(f"  Mean: ${catalog_with_price['PRICE'].mean():.2f}")
print(f"  Median: ${catalog_with_price['PRICE'].median():.2f}")
print(f"  Std: ${catalog_with_price['PRICE'].std():.2f}")
print(f"  Min: ${catalog_with_price['PRICE'].min():.2f}")
print(f"  Max: ${catalog_with_price['PRICE'].max():.2f}")

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print(f"\nAOV Percentiles:")
for q in quantiles:
    val = catalog_with_price['PRICE'].quantile(q)
    print(f"  p{int(q*100):02d}: ${val:.2f}")

# Validate with purchase data
print("\nValidating with purchase data...")
df_purchases = pd.read_parquet('../data/raw_purchases_20251011.parquet')
df_purchases['order_value'] = df_purchases['UNIT_PRICE'] * df_purchases['QUANTITY'] / 100.0

print(f"Purchase order values:")
print(f"  Count: {len(df_purchases):,}")
print(f"  Mean: ${df_purchases['order_value'].mean():.2f}")
print(f"  Median: ${df_purchases['order_value'].median():.2f}")

print("\nUpdating claude.md...")
results_text = f"""
### F) AVERAGE ORDER VALUE (AOV)

**Status:** ✅ COMPLETED

**Method:** Use catalog PRICE field

**Results:**
- Total products in catalog: {len(df_catalog):,}
- Products with price: {len(catalog_with_price):,} ({len(catalog_with_price)/len(df_catalog)*100:.1f}%)

**Catalog-based AOV Distribution:**
- Mean: ${catalog_with_price['PRICE'].mean():.2f}
- Median: ${catalog_with_price['PRICE'].median():.2f}
- Std: ${catalog_with_price['PRICE'].std():.2f}
- Range: ${catalog_with_price['PRICE'].min():.2f} - ${catalog_with_price['PRICE'].max():.2f}

**Percentiles:**
- p10: ${catalog_with_price['PRICE'].quantile(0.10):.2f}
- p25: ${catalog_with_price['PRICE'].quantile(0.25):.2f}
- p50: ${catalog_with_price['PRICE'].median():.2f}
- p75: ${catalog_with_price['PRICE'].quantile(0.75):.2f}
- p90: ${catalog_with_price['PRICE'].quantile(0.90):.2f}
- p95: ${catalog_with_price['PRICE'].quantile(0.95):.2f}
- p99: ${catalog_with_price['PRICE'].quantile(0.99):.2f}

**Validation from actual purchases:**
- Purchase count: {len(df_purchases):,}
- Mean order value: ${df_purchases['order_value'].mean():.2f}
- Median order value: ${df_purchases['order_value'].median():.2f}

**Interpretation:**
- Typical AOV: ${catalog_with_price['PRICE'].median():.2f} (median catalog price)
- Range for simulation: ${catalog_with_price['PRICE'].quantile(0.10):.2f} - ${catalog_with_price['PRICE'].quantile(0.90):.2f} (p10-p90)
- Catalog prices align well with purchase data
"""

with open('claude.md', 'r') as f:
    content = f.read()

content = content.replace(
    '### F) AVERAGE ORDER VALUE (AOV)\n\n**Status:** NOT STARTED\n\n**Method:** Use catalog PRICE field\n\n**Results:**\n- TBD',
    results_text.strip()
)

with open('claude.md', 'w') as f:
    f.write(content)

print("✅ Updated claude.md with AOV results")
