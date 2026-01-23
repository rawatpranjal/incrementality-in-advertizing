#!/usr/bin/env python3
"""
01_build_panel.py - Vendor-Week Panel Construction

Builds a dense vendor-week panel for Callaway-Sant'Anna DiD analysis.
Skips AUCTIONS_RESULTS (62B rows, too slow) and uses CLICKS as treatment proxy.

Output: data/vendor_weekly_panel.parquet
Log: results/01_build_panel.txt
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import snowflake.connector

# Configuration
ANALYSIS_START = '2025-03-24'
ANALYSIS_END = '2025-09-15'
N_WEEKS = 26

# Paths
BASE_DIR = Path(__file__).parent.parent  # Go up from scripts/ to project root
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

LOG_FILE = RESULTS_DIR / '01_build_panel.txt'

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)

def connect_snowflake():
    """Establish Snowflake connection."""
    load_dotenv()
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
        database='INCREMENTALITY',
        schema='INCREMENTALITY_RESEARCH'
    )
    return conn

def execute_query(conn, query, desc="Query"):
    """Execute query with progress indicator."""
    print(f"  Executing: {desc}...")
    start = datetime.now()
    df = pd.read_sql(query, conn)
    elapsed = datetime.now() - start
    print(f"  Returned {len(df):,} rows in {elapsed}")
    return df

def main():
    print("=" * 70)
    print("01_BUILD_PANEL.PY - Vendor-Week Panel Construction")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Analysis window: {ANALYSIS_START} to {ANALYSIS_END} ({N_WEEKS} weeks)")
    print()

    # Connect
    print("[STEP 0] Connecting to Snowflake...")
    conn = connect_snowflake()
    print("  [OK] Connected")
    print()

    # =========================================================================
    # STEP 1: Vendor Universe from Catalog
    # =========================================================================
    print("[STEP 1] Extracting Vendor Universe from CATALOG...")

    vendor_query = """
    SELECT DISTINCT v.value::STRING AS VENDOR_ID
    FROM CATALOG, LATERAL FLATTEN(input => VENDORS) v
    WHERE VENDORS IS NOT NULL
    """
    vendors = execute_query(conn, vendor_query, "Vendor Universe")
    print(f"  Unique vendors: {len(vendors):,}")
    print(f"  Sample: {vendors['VENDOR_ID'].head(3).tolist()}")
    print()

    # =========================================================================
    # STEP 2: Week Scaffold
    # =========================================================================
    print("[STEP 2] Creating Week Scaffold...")
    weeks = pd.date_range(start=ANALYSIS_START, periods=N_WEEKS, freq='W-MON')
    week_df = pd.DataFrame({'WEEK': weeks})
    print(f"  Weeks: {len(weeks)}")
    print(f"  Range: {weeks[0].date()} to {weeks[-1].date()}")
    print()

    # =========================================================================
    # STEP 3: Total GMV by Vendor-Week (via Catalog bridge)
    # =========================================================================
    print("[STEP 3] Aggregating Total GMV by Vendor-Week...")

    gmv_query = f"""
    SELECT
        v.value::STRING AS VENDOR_ID,
        DATE_TRUNC('week', p.PURCHASED_AT) AS WEEK,
        SUM(p.UNIT_PRICE * p.QUANTITY) AS TOTAL_GMV,
        COUNT(DISTINCT p.PURCHASE_ID) AS N_PURCHASES
    FROM PURCHASES p
    JOIN CATALOG c ON LOWER(TRIM(p.PRODUCT_ID)) = LOWER(TRIM(c.PRODUCT_ID))
    CROSS JOIN LATERAL FLATTEN(input => c.VENDORS) v
    WHERE p.PURCHASED_AT BETWEEN '{ANALYSIS_START}' AND '{ANALYSIS_END}'
    GROUP BY 1, 2
    """
    gmv = execute_query(conn, gmv_query, "Total GMV")
    print(f"  Total GMV: ${gmv['TOTAL_GMV'].sum():,.0f}")
    print(f"  Total Purchases: {gmv['N_PURCHASES'].sum():,}")
    print()

    # =========================================================================
    # STEP 4: Clicks by Vendor-Week (Treatment Proxy)
    # =========================================================================
    print("[STEP 4] Aggregating Clicks by Vendor-Week (Treatment Proxy)...")

    clicks_query = f"""
    SELECT
        LOWER(REPLACE(VENDOR_ID, '-', '')) AS VENDOR_ID,
        DATE_TRUNC('week', OCCURRED_AT) AS WEEK,
        COUNT(*) AS CLICKS
    FROM CLICKS
    WHERE OCCURRED_AT BETWEEN '{ANALYSIS_START}' AND '{ANALYSIS_END}'
    GROUP BY 1, 2
    """
    clicks = execute_query(conn, clicks_query, "Clicks")
    print(f"  Total Clicks: {clicks['CLICKS'].sum():,}")
    print()

    # =========================================================================
    # STEP 5: Impressions by Vendor-Week
    # =========================================================================
    print("[STEP 5] Aggregating Impressions by Vendor-Week...")

    impressions_query = f"""
    SELECT
        LOWER(REPLACE(VENDOR_ID, '-', '')) AS VENDOR_ID,
        DATE_TRUNC('week', OCCURRED_AT) AS WEEK,
        COUNT(*) AS IMPRESSIONS
    FROM IMPRESSIONS
    WHERE OCCURRED_AT BETWEEN '{ANALYSIS_START}' AND '{ANALYSIS_END}'
    GROUP BY 1, 2
    """
    impressions = execute_query(conn, impressions_query, "Impressions")
    print(f"  Total Impressions: {impressions['IMPRESSIONS'].sum():,}")
    print()

    # =========================================================================
    # STEP 6: Promoted GMV (Click-Attributed, 7-day window)
    # =========================================================================
    print("[STEP 6] Aggregating Promoted GMV (Click-Attributed)...")

    promoted_query = f"""
    SELECT
        LOWER(REPLACE(cl.VENDOR_ID, '-', '')) AS VENDOR_ID,
        DATE_TRUNC('week', p.PURCHASED_AT) AS WEEK,
        SUM(p.UNIT_PRICE * p.QUANTITY) AS PROMOTED_GMV,
        COUNT(DISTINCT p.PURCHASE_ID) AS N_ATTRIBUTED
    FROM CLICKS cl
    JOIN PURCHASES p
        ON cl.USER_ID = p.USER_ID
        AND LOWER(TRIM(cl.PRODUCT_ID)) = LOWER(TRIM(p.PRODUCT_ID))
        AND p.PURCHASED_AT BETWEEN cl.OCCURRED_AT AND DATEADD('day', 7, cl.OCCURRED_AT)
    WHERE cl.OCCURRED_AT BETWEEN '{ANALYSIS_START}' AND '{ANALYSIS_END}'
    GROUP BY 1, 2
    """
    promoted = execute_query(conn, promoted_query, "Promoted GMV")
    print(f"  Promoted GMV: ${promoted['PROMOTED_GMV'].sum():,.0f}")
    print(f"  Attributed Purchases: {promoted['N_ATTRIBUTED'].sum():,}")
    print()

    # =========================================================================
    # STEP 7: Build Dense Panel (Python-side Scaffold)
    # =========================================================================
    print("[STEP 7] Building Dense Vendor-Week Panel...")

    # Normalize vendor IDs
    vendors['VENDOR_ID'] = vendors['VENDOR_ID'].str.lower().str.replace('-', '', regex=False)
    gmv['VENDOR_ID'] = gmv['VENDOR_ID'].str.lower().str.replace('-', '', regex=False)

    # Create scaffold
    print("  Creating scaffold...")
    scaffold = pd.MultiIndex.from_product(
        [vendors['VENDOR_ID'].unique(), weeks],
        names=['VENDOR_ID', 'WEEK']
    ).to_frame(index=False)
    print(f"  Scaffold size: {len(scaffold):,} rows")

    # Normalize week columns
    gmv['WEEK'] = pd.to_datetime(gmv['WEEK'])
    clicks['WEEK'] = pd.to_datetime(clicks['WEEK'])
    impressions['WEEK'] = pd.to_datetime(impressions['WEEK'])
    promoted['WEEK'] = pd.to_datetime(promoted['WEEK'])

    # Left join all aggregates
    print("  Joining aggregates...")
    panel = scaffold.copy()

    panel = panel.merge(gmv, on=['VENDOR_ID', 'WEEK'], how='left')
    panel = panel.merge(clicks, on=['VENDOR_ID', 'WEEK'], how='left')
    panel = panel.merge(impressions, on=['VENDOR_ID', 'WEEK'], how='left')
    panel = panel.merge(promoted, on=['VENDOR_ID', 'WEEK'], how='left')

    # Fill NAs with zeros (explicit zeros for unobserved)
    print("  Filling zeros...")
    panel = panel.fillna({
        'TOTAL_GMV': 0,
        'N_PURCHASES': 0,
        'CLICKS': 0,
        'IMPRESSIONS': 0,
        'PROMOTED_GMV': 0,
        'N_ATTRIBUTED': 0
    })

    # Derived columns
    panel['ORGANIC_GMV'] = panel['TOTAL_GMV'] - panel['PROMOTED_GMV']
    panel['ORGANIC_GMV'] = panel['ORGANIC_GMV'].clip(lower=0)  # Handle attribution artifacts
    panel['HAS_CLICKS'] = (panel['CLICKS'] > 0).astype(int)
    panel['HAS_IMPRESSIONS'] = (panel['IMPRESSIONS'] > 0).astype(int)

    print()
    print("  Panel Summary:")
    print(f"    Rows: {len(panel):,}")
    print(f"    Vendors: {panel['VENDOR_ID'].nunique():,}")
    print(f"    Weeks: {panel['WEEK'].nunique()}")
    print()

    # =========================================================================
    # STEP 8: Panel Diagnostics
    # =========================================================================
    print("[STEP 8] Panel Diagnostics...")
    print()

    print("  Column Statistics:")
    for col in ['TOTAL_GMV', 'PROMOTED_GMV', 'ORGANIC_GMV', 'CLICKS', 'IMPRESSIONS']:
        zero_pct = (panel[col] == 0).mean() * 100
        mean_val = panel[col].mean()
        max_val = panel[col].max()
        print(f"    {col}: mean={mean_val:.2f}, max={max_val:,.0f}, zeros={zero_pct:.1f}%")
    print()

    print("  Treatment Distribution (HAS_CLICKS):")
    print(f"    Treated obs: {panel['HAS_CLICKS'].sum():,} ({panel['HAS_CLICKS'].mean()*100:.2f}%)")
    print(f"    Control obs: {(1-panel['HAS_CLICKS']).sum():,}")
    print()

    # Cohort assignment (first week with clicks > 0)
    first_click = panel[panel['HAS_CLICKS'] == 1].groupby('VENDOR_ID')['WEEK'].min()
    never_treated = set(panel['VENDOR_ID'].unique()) - set(first_click.index)

    print("  Cohort Summary:")
    print(f"    Treated vendors: {len(first_click):,}")
    print(f"    Never-treated vendors: {len(never_treated):,}")
    print(f"    Unique cohorts: {first_click.nunique()}")
    print()

    if len(first_click) > 0:
        cohort_sizes = first_click.value_counts().sort_index()
        print("  Cohort Sizes by Week:")
        for week, count in cohort_sizes.items():
            print(f"    {week.date()}: {count:,} vendors")
    print()

    # =========================================================================
    # STEP 9: Save Panel
    # =========================================================================
    print("[STEP 9] Saving Panel...")

    output_path = DATA_DIR / 'vendor_weekly_panel.parquet'
    panel.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved to: {output_path}")
    print(f"  File size: {size_mb:.1f} MB")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Analysis Period: {ANALYSIS_START} to {ANALYSIS_END}")
    print(f"Panel Dimensions: {len(panel):,} rows x {len(panel.columns)} columns")
    print(f"Vendors: {panel['VENDOR_ID'].nunique():,}")
    print(f"Weeks: {panel['WEEK'].nunique()}")
    print(f"Total GMV: ${panel['TOTAL_GMV'].sum():,.0f}")
    print(f"Promoted GMV: ${panel['PROMOTED_GMV'].sum():,.0f}")
    print(f"Organic GMV: ${panel['ORGANIC_GMV'].sum():,.0f}")
    print(f"Treated vendors: {len(first_click):,}")
    print(f"Never-treated: {len(never_treated):,}")
    print(f"Output: {output_path}")
    print()
    print(f"Completed: {datetime.now()}")

    conn.close()
    print("[OK] Snowflake connection closed")

if __name__ == '__main__':
    main()
