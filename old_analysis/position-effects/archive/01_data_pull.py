#!/usr/bin/env python3
"""
Position Effects Data Pull - Run with: python 01_data_pull.py
"""
import os
from pathlib import Path
import warnings
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector
from tqdm import tqdm

warnings.filterwarnings('ignore')
load_dotenv()

# =============================================================================
# CONFIG - ADJUST THESE
# =============================================================================
MINUTES_WINDOW = 5  # Change this as needed
PLACEMENT_FILTER = 5
USER_SAMPLE_PCT = 1  # Sample X% of users (1-100, use 100 for no sampling)

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Pulling {MINUTES_WINDOW} minutes of Placement {PLACEMENT_FILTER} data ({USER_SAMPLE_PCT}% user sample)...")

# =============================================================================
# CONNECT
# =============================================================================
print("\nConnecting to Snowflake...")
conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
    database='INCREMENTALITY',
    schema='INCREMENTALITY_RESEARCH'
)
print("[SUCCESS] Connected")

# =============================================================================
# PULL DATA
# =============================================================================

# 1. AUCTIONS_USERS (with user sampling)
print("\n1/5 AUCTIONS_USERS...")
user_sample_clause = f"AND MOD(ABS(HASH(OPAQUE_USER_ID)), 100) < {USER_SAMPLE_PCT}" if USER_SAMPLE_PCT < 100 else ""
auctions_users = pd.read_sql(f"""
SELECT
    LOWER(TO_VARCHAR(AUCTION_ID, 'HEX')) AS AUCTION_ID,
    OPAQUE_USER_ID AS USER_ID,
    PLACEMENT,
    CREATED_AT
FROM AUCTIONS_USERS
WHERE PLACEMENT = {PLACEMENT_FILTER}
  AND CREATED_AT >= DATEADD(minute, -{MINUTES_WINDOW}, CURRENT_TIMESTAMP())
  {user_sample_clause}
""", conn)
print(f"  {len(auctions_users):,} rows ({USER_SAMPLE_PCT}% sample)")

# 2. AUCTIONS_RESULTS (with same user sampling)
print("\n2/5 AUCTIONS_RESULTS...")
user_sample_clause_au = f"AND MOD(ABS(HASH(au.OPAQUE_USER_ID)), 100) < {USER_SAMPLE_PCT}" if USER_SAMPLE_PCT < 100 else ""
auctions_results = pd.read_sql(f"""
SELECT
    LOWER(TO_VARCHAR(ar.AUCTION_ID, 'HEX')) AS AUCTION_ID,
    LOWER(TO_VARCHAR(ar.VENDOR_ID, 'HEX')) AS VENDOR_ID,
    LOWER(TO_VARCHAR(ar.CAMPAIGN_ID, 'HEX')) AS CAMPAIGN_ID,
    LOWER(TRIM(ar.PRODUCT_ID)) AS PRODUCT_ID,
    ar.RANKING,
    ar.IS_WINNER,
    ar.FINAL_BID,
    ar.QUALITY,
    ar.CONVERSION_RATE,
    ar.PACING,
    ar.PRICE,
    ar.CREATED_AT
FROM AUCTIONS_RESULTS ar
JOIN AUCTIONS_USERS au ON ar.AUCTION_ID = au.AUCTION_ID
WHERE au.PLACEMENT = {PLACEMENT_FILTER}
  AND ar.CREATED_AT >= DATEADD(minute, -{MINUTES_WINDOW}, CURRENT_TIMESTAMP())
  {user_sample_clause_au}
""", conn)
print(f"  {len(auctions_results):,} rows")

# 3. IMPRESSIONS (filter to sampled users via auction join)
print("\n3/5 IMPRESSIONS...")
impressions = pd.read_sql(f"""
SELECT
    i.INTERACTION_ID,
    LOWER(REPLACE(i.AUCTION_ID, '-', '')) AS AUCTION_ID,
    LOWER(TRIM(i.PRODUCT_ID)) AS PRODUCT_ID,
    i.USER_ID,
    LOWER(REPLACE(i.CAMPAIGN_ID, '-', '')) AS CAMPAIGN_ID,
    LOWER(REPLACE(i.VENDOR_ID, '-', '')) AS VENDOR_ID,
    i.OCCURRED_AT
FROM IMPRESSIONS i
WHERE i.OCCURRED_AT >= DATEADD(minute, -{MINUTES_WINDOW}, CURRENT_TIMESTAMP())
  AND MOD(ABS(HASH(i.USER_ID)), 100) < {USER_SAMPLE_PCT}
""", conn)
print(f"  {len(impressions):,} rows")

# 4. CLICKS (filter to sampled users)
print("\n4/5 CLICKS...")
clicks = pd.read_sql(f"""
SELECT
    c.INTERACTION_ID,
    LOWER(REPLACE(c.AUCTION_ID, '-', '')) AS AUCTION_ID,
    LOWER(TRIM(c.PRODUCT_ID)) AS PRODUCT_ID,
    c.USER_ID,
    LOWER(REPLACE(c.CAMPAIGN_ID, '-', '')) AS CAMPAIGN_ID,
    LOWER(REPLACE(c.VENDOR_ID, '-', '')) AS VENDOR_ID,
    c.OCCURRED_AT
FROM CLICKS c
WHERE c.OCCURRED_AT >= DATEADD(minute, -{MINUTES_WINDOW}, CURRENT_TIMESTAMP())
  AND MOD(ABS(HASH(c.USER_ID)), 100) < {USER_SAMPLE_PCT}
""", conn)
print(f"  {len(clicks):,} rows")

# 5. CATALOG
print("\n5/5 CATALOG...")
product_ids = auctions_results['PRODUCT_ID'].dropna().unique().tolist()
print(f"  Products to fetch: {len(product_ids):,}")

if len(product_ids) > 0:
    # Batch if needed
    batch_size = 10000
    catalog_dfs = []
    for i in tqdm(range(0, len(product_ids), batch_size), desc="Catalog batches"):
        batch = product_ids[i:i+batch_size]
        placeholders = ', '.join(['%s'] * len(batch))
        batch_df = pd.read_sql(f"""
        SELECT
            LOWER(TRIM(PRODUCT_ID)) AS PRODUCT_ID,
            NAME,
            PRICE AS CATALOG_PRICE,
            ACTIVE,
            IS_DELETED,
            CATEGORIES,
            DESCRIPTION
        FROM CATALOG
        WHERE LOWER(TRIM(PRODUCT_ID)) IN ({placeholders})
        """, conn, params=batch)
        catalog_dfs.append(batch_df)
    catalog = pd.concat(catalog_dfs, ignore_index=True) if catalog_dfs else pd.DataFrame()
else:
    catalog = pd.DataFrame()
print(f"  {len(catalog):,} rows")

conn.close()
print("\n[SUCCESS] Connection closed")

# =============================================================================
# SAVE
# =============================================================================
print("\nSaving parquet files...")
auctions_results.to_parquet(OUTPUT_DIR / "auctions_results_p5_1d.parquet", index=False)
auctions_users.to_parquet(OUTPUT_DIR / "auctions_users_p5_1d.parquet", index=False)
impressions.to_parquet(OUTPUT_DIR / "impressions_p5_1d.parquet", index=False)
clicks.to_parquet(OUTPUT_DIR / "clicks_p5_1d.parquet", index=False)
catalog.to_parquet(OUTPUT_DIR / "catalog_p5_1d.parquet", index=False)

print("\n" + "="*50)
print("DONE")
print("="*50)
print(f"auctions_results: {len(auctions_results):,}")
print(f"auctions_users:   {len(auctions_users):,}")
print(f"impressions:      {len(impressions):,}")
print(f"clicks:           {len(clicks):,}")
print(f"catalog:          {len(catalog):,}")
