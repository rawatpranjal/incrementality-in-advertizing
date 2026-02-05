#!/usr/bin/env python3
"""
EDA Q1: What is the "Orphan Rate" of GMV?

Precisely what percentage of rows (and GMV volume) in the PURCHASES table
fail to join to the CATALOG table? If this exceeds ~5-10%, our proxy for
"Total GMV" is biased, likely undercounting organic sales for older/deleted products.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # staggered-adoption/
EDA_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "01_orphan_rate_gmv.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# SNOWFLAKE CONNECTION
# =============================================================================
def get_snowflake_connection():
    """Establish Snowflake connection using environment variables."""
    try:
        import snowflake.connector
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
    except Exception as e:
        return None, str(e)

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("EDA Q1: ORPHAN RATE OF GMV", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  What percentage of PURCHASES rows (and GMV) fail to join to CATALOG?", f)
        log("  Threshold: If >5-10%, Total GMV proxy is biased.", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. LEFT JOIN PURCHASES to CATALOG on PRODUCT_ID", f)
        log("  2. Count orphan rows (no catalog match)", f)
        log("  3. Sum orphan GMV (UNIT_PRICE * QUANTITY)", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Connect to Snowflake
        # -----------------------------------------------------------------
        log("CONNECTING TO SNOWFLAKE...", f)

        result = get_snowflake_connection()
        if isinstance(result, tuple):
            conn, error = None, result[1]
        else:
            conn, error = result, None

        if conn is None:
            log(f"  [ERROR] Could not connect to Snowflake: {error}", f)
            log("", f)
            log("  Falling back to sample data analysis...", f)
            log("", f)

            # Fallback: Use sample data
            analyze_sample_data(f)
            return

        log("  [SUCCESS] Snowflake connection established.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Run Snowflake Query
        # -----------------------------------------------------------------
        log("RUNNING ORPHAN RATE QUERY...", f)
        log("-" * 40, f)

        query = """
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT p.PURCHASE_ID) as total_purchases,
            SUM(CASE WHEN c.PRODUCT_ID IS NULL THEN 1 ELSE 0 END) as orphan_rows,
            COUNT(DISTINCT CASE WHEN c.PRODUCT_ID IS NULL THEN p.PURCHASE_ID END) as orphan_purchases,
            SUM(p.UNIT_PRICE * p.QUANTITY) as total_gmv,
            SUM(CASE WHEN c.PRODUCT_ID IS NULL THEN p.UNIT_PRICE * p.QUANTITY ELSE 0 END) as orphan_gmv,
            SUM(CASE WHEN c.PRODUCT_ID IS NOT NULL THEN p.UNIT_PRICE * p.QUANTITY ELSE 0 END) as matched_gmv
        FROM PURCHASES p
        LEFT JOIN CATALOG c ON LOWER(TRIM(p.PRODUCT_ID)) = LOWER(TRIM(c.PRODUCT_ID))
        WHERE p.PURCHASED_AT BETWEEN '2025-03-14' AND '2025-09-15'
        """

        log("  Query:", f)
        for line in query.strip().split('\n'):
            log(f"    {line}", f)
        log("", f)

        try:
            with tqdm(desc="Executing query") as pbar:
                df = pd.read_sql(query, conn)
                pbar.update(1)

            log("  [SUCCESS] Query completed.", f)
            log("", f)

            # -----------------------------------------------------------------
            # Results
            # -----------------------------------------------------------------
            log("=" * 80, f)
            log("RESULTS: ORPHAN RATE ANALYSIS", f)
            log("-" * 40, f)
            log("", f)

            total_rows = df['TOTAL_ROWS'].iloc[0]
            total_purchases = df['TOTAL_PURCHASES'].iloc[0]
            orphan_rows = df['ORPHAN_ROWS'].iloc[0]
            orphan_purchases = df['ORPHAN_PURCHASES'].iloc[0]
            total_gmv = df['TOTAL_GMV'].iloc[0]
            orphan_gmv = df['ORPHAN_GMV'].iloc[0]
            matched_gmv = df['MATCHED_GMV'].iloc[0]

            orphan_row_pct = (orphan_rows / total_rows * 100) if total_rows > 0 else 0
            orphan_gmv_pct = (orphan_gmv / total_gmv * 100) if total_gmv > 0 else 0

            log(f"  Total purchase rows:     {total_rows:,}", f)
            log(f"  Total unique purchases:  {total_purchases:,}", f)
            log(f"  Total GMV:               ${total_gmv:,.2f}", f)
            log("", f)

            log(f"  Orphan rows (no catalog): {orphan_rows:,} ({orphan_row_pct:.2f}%)", f)
            log(f"  Orphan purchases:         {orphan_purchases:,}", f)
            log(f"  Orphan GMV:               ${orphan_gmv:,.2f} ({orphan_gmv_pct:.2f}%)", f)
            log("", f)

            log(f"  Matched GMV:              ${matched_gmv:,.2f} ({100-orphan_gmv_pct:.2f}%)", f)
            log("", f)

            # -----------------------------------------------------------------
            # Interpretation
            # -----------------------------------------------------------------
            log("=" * 80, f)
            log("INTERPRETATION", f)
            log("-" * 40, f)
            log("", f)

            if orphan_gmv_pct > 10:
                log("  [WARNING] Orphan GMV rate exceeds 10%.", f)
                log("  This suggests significant bias in Total GMV measurement.", f)
                log("  Possible causes:", f)
                log("    - Deleted/inactive products not in current catalog", f)
                log("    - Products from organic (non-promoted) channels", f)
                log("    - Data quality issues in PRODUCT_ID matching", f)
            elif orphan_gmv_pct > 5:
                log("  [CAUTION] Orphan GMV rate is 5-10%.", f)
                log("  Moderate bias possible. Review product matching logic.", f)
            else:
                log("  [OK] Orphan GMV rate is within acceptable range (<5%).", f)
                log("  Total GMV proxy should be reasonably unbiased.", f)

            log("", f)

            # Additional breakdown by time
            log("=" * 80, f)
            log("ORPHAN RATE BY MONTH", f)
            log("-" * 40, f)

            monthly_query = """
            SELECT
                DATE_TRUNC('month', p.PURCHASED_AT) as month,
                COUNT(*) as total_rows,
                SUM(CASE WHEN c.PRODUCT_ID IS NULL THEN 1 ELSE 0 END) as orphan_rows,
                SUM(p.UNIT_PRICE * p.QUANTITY) as total_gmv,
                SUM(CASE WHEN c.PRODUCT_ID IS NULL THEN p.UNIT_PRICE * p.QUANTITY ELSE 0 END) as orphan_gmv
            FROM PURCHASES p
            LEFT JOIN CATALOG c ON LOWER(TRIM(p.PRODUCT_ID)) = LOWER(TRIM(c.PRODUCT_ID))
            WHERE p.PURCHASED_AT BETWEEN '2025-03-14' AND '2025-09-15'
            GROUP BY 1
            ORDER BY 1
            """

            monthly_df = pd.read_sql(monthly_query, conn)

            log("", f)
            log(f"  {'Month':<12} {'Rows':>10} {'Orphan%':>10} {'GMV':>15} {'Orphan GMV%':>12}", f)
            log(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*15} {'-'*12}", f)

            for _, row in monthly_df.iterrows():
                month_str = str(row['MONTH'])[:10]
                orphan_pct = (row['ORPHAN_ROWS'] / row['TOTAL_ROWS'] * 100) if row['TOTAL_ROWS'] > 0 else 0
                orphan_gmv_pct = (row['ORPHAN_GMV'] / row['TOTAL_GMV'] * 100) if row['TOTAL_GMV'] > 0 else 0
                log(f"  {month_str:<12} {row['TOTAL_ROWS']:>10,} {orphan_pct:>9.2f}% ${row['TOTAL_GMV']:>13,.0f} {orphan_gmv_pct:>10.2f}%", f)

            log("", f)

            conn.close()
            log("  [SUCCESS] Snowflake connection closed.", f)

        except Exception as e:
            log(f"  [ERROR] Query failed: {e}", f)
            log("", f)
            log("  Falling back to sample data analysis...", f)
            analyze_sample_data(f)
            return

        log("", f)
        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


def analyze_sample_data(f):
    """Fallback analysis using local sample data."""
    log("", f)
    log("=" * 80, f)
    log("SAMPLE DATA ANALYSIS (FALLBACK)", f)
    log("-" * 40, f)
    log("", f)

    RAW_DATA_DIR = BASE_DIR / ".." / "shopping-sessions" / "data"

    # Load sample data
    purchases_path = RAW_DATA_DIR / "raw_sample_purchases.parquet"
    catalog_path = RAW_DATA_DIR / "processed_sample_catalog.parquet"

    if not purchases_path.exists() or not catalog_path.exists():
        log("  [ERROR] Sample data files not found.", f)
        return

    log("  Loading sample purchases...", f)
    purchases = pd.read_parquet(purchases_path)
    log(f"    Loaded {len(purchases):,} purchase rows", f)

    log("  Loading catalog...", f)
    catalog = pd.read_parquet(catalog_path)
    log(f"    Loaded {len(catalog):,} catalog rows", f)
    log("", f)

    # Normalize product IDs
    purchases['PRODUCT_ID_CLEAN'] = purchases['PRODUCT_ID'].astype(str).str.lower().str.strip()
    catalog_products = set(catalog['PRODUCT_ID'].astype(str).str.lower().str.strip())

    # Calculate orphan rate
    purchases['is_orphan'] = ~purchases['PRODUCT_ID_CLEAN'].isin(catalog_products)
    purchases['gmv'] = purchases['UNIT_PRICE'] * purchases['QUANTITY']

    total_rows = len(purchases)
    orphan_rows = purchases['is_orphan'].sum()
    total_gmv = purchases['gmv'].sum()
    orphan_gmv = purchases.loc[purchases['is_orphan'], 'gmv'].sum()

    orphan_row_pct = (orphan_rows / total_rows * 100) if total_rows > 0 else 0
    orphan_gmv_pct = (orphan_gmv / total_gmv * 100) if total_gmv > 0 else 0

    log("RESULTS (SAMPLE DATA):", f)
    log("-" * 40, f)
    log(f"  Total purchase rows:     {total_rows:,}", f)
    log(f"  Total GMV:               ${total_gmv:,.2f}", f)
    log("", f)
    log(f"  Orphan rows (no catalog): {orphan_rows:,} ({orphan_row_pct:.2f}%)", f)
    log(f"  Orphan GMV:               ${orphan_gmv:,.2f} ({orphan_gmv_pct:.2f}%)", f)
    log("", f)

    log("  [NOTE] Sample is 0.1% of users; results may not be representative.", f)
    log("", f)


if __name__ == "__main__":
    main()
