#!/usr/bin/env python3
"""
EDA Q6: Is FINAL_BID truly the Cost Per Click?

In AUCTIONS_RESULTS, for winning bids (IS_WINNER=True), is the FINAL_BID
the actual price paid? We need to verify if Total Spend derived from summing
FINAL_BID aligns with reasonable CPM/CPC benchmarks.
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
OUTPUT_FILE = RESULTS_DIR / "06_bid_cpc_verification.txt"

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
        return None

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("EDA Q6: FINAL_BID AS COST PER CLICK VERIFICATION", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  Is FINAL_BID the actual price paid by winning bidders?", f)
        log("  Does it align with reasonable CPC/CPM benchmarks?", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. Query AUCTIONS_RESULTS for FINAL_BID distribution", f)
        log("  2. Compare winning vs losing bids", f)
        log("  3. Validate against industry CPC benchmarks ($0.10-$5.00)", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Connect to Snowflake
        # -----------------------------------------------------------------
        log("CONNECTING TO SNOWFLAKE...", f)

        conn = get_snowflake_connection()

        if conn is None:
            log("  [ERROR] Could not connect to Snowflake.", f)
            log("  FINAL_BID, QUALITY, PACING columns are not in local sample data.", f)
            log("  This analysis requires Snowflake access.", f)
            log("", f)
            log("  FALLBACK: Analyzing local sample data structure...", f)
            analyze_local_sample(f)
            return

        log("  [SUCCESS] Snowflake connection established.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Query FINAL_BID distribution
        # -----------------------------------------------------------------
        log("QUERYING FINAL_BID DISTRIBUTION...", f)
        log("-" * 40, f)

        query = """
        SELECT
            IS_WINNER,
            COUNT(*) as n_bids,
            AVG(FINAL_BID) as avg_bid,
            MEDIAN(FINAL_BID) as median_bid,
            MIN(FINAL_BID) as min_bid,
            MAX(FINAL_BID) as max_bid,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY FINAL_BID) as p25_bid,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY FINAL_BID) as p75_bid,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY FINAL_BID) as p95_bid,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY FINAL_BID) as p99_bid
        FROM AUCTIONS_RESULTS
        WHERE CREATED_AT BETWEEN '2025-03-14' AND '2025-09-15'
          AND FINAL_BID IS NOT NULL
        GROUP BY IS_WINNER
        ORDER BY IS_WINNER DESC
        """

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
            log("FINAL_BID DISTRIBUTION BY WINNER STATUS", f)
            log("-" * 40, f)
            log("", f)

            for _, row in df.iterrows():
                winner_status = "WINNERS" if row['IS_WINNER'] else "LOSERS"
                log(f"  {winner_status}:", f)
                log(f"    N bids:    {row['N_BIDS']:,}", f)
                log(f"    Mean:      ${row['AVG_BID']:.4f}", f)
                log(f"    Median:    ${row['MEDIAN_BID']:.4f}", f)
                log(f"    Min:       ${row['MIN_BID']:.4f}", f)
                log(f"    P25:       ${row['P25_BID']:.4f}", f)
                log(f"    P75:       ${row['P75_BID']:.4f}", f)
                log(f"    P95:       ${row['P95_BID']:.4f}", f)
                log(f"    P99:       ${row['P99_BID']:.4f}", f)
                log(f"    Max:       ${row['MAX_BID']:.4f}", f)
                log("", f)

            # -----------------------------------------------------------------
            # Bid ranges and CPC benchmarks
            # -----------------------------------------------------------------
            log("=" * 80, f)
            log("CPC BENCHMARK COMPARISON", f)
            log("-" * 40, f)
            log("", f)

            winners = df[df['IS_WINNER'] == True].iloc[0] if len(df[df['IS_WINNER'] == True]) > 0 else None

            if winners is not None:
                avg_cpc = winners['AVG_BID']
                median_cpc = winners['MEDIAN_BID']

                log("  Industry CPC Benchmarks (typical ranges):", f)
                log("    - Low CPM display: $0.01 - $0.10", f)
                log("    - Search/social: $0.50 - $2.00", f)
                log("    - E-commerce sponsored: $0.10 - $1.00", f)
                log("", f)

                log(f"  Observed FINAL_BID for winners:", f)
                log(f"    Average: ${avg_cpc:.4f}", f)
                log(f"    Median: ${median_cpc:.4f}", f)
                log("", f)

                # Check if values are reasonable
                if avg_cpc < 0.01:
                    log("  [WARNING] Average bid < $0.01 - suspiciously low.", f)
                    log("  FINAL_BID may be in different units (cents? basis points?).", f)
                elif avg_cpc > 10:
                    log("  [WARNING] Average bid > $10 - unusually high.", f)
                    log("  Verify unit of measurement.", f)
                else:
                    log("  [OK] FINAL_BID values appear to be in reasonable CPC range.", f)

                log("", f)

            # -----------------------------------------------------------------
            # Query bid distribution histogram
            # -----------------------------------------------------------------
            log("=" * 80, f)
            log("BID DISTRIBUTION HISTOGRAM (WINNERS)", f)
            log("-" * 40, f)

            hist_query = """
            SELECT
                CASE
                    WHEN FINAL_BID = 0 THEN '0.00'
                    WHEN FINAL_BID < 0.01 THEN '0.00-0.01'
                    WHEN FINAL_BID < 0.05 THEN '0.01-0.05'
                    WHEN FINAL_BID < 0.10 THEN '0.05-0.10'
                    WHEN FINAL_BID < 0.25 THEN '0.10-0.25'
                    WHEN FINAL_BID < 0.50 THEN '0.25-0.50'
                    WHEN FINAL_BID < 1.00 THEN '0.50-1.00'
                    WHEN FINAL_BID < 2.00 THEN '1.00-2.00'
                    WHEN FINAL_BID < 5.00 THEN '2.00-5.00'
                    ELSE '5.00+'
                END as bid_range,
                COUNT(*) as n_bids,
                SUM(FINAL_BID) as total_spend
            FROM AUCTIONS_RESULTS
            WHERE CREATED_AT BETWEEN '2025-03-14' AND '2025-09-15'
              AND IS_WINNER = TRUE
              AND FINAL_BID IS NOT NULL
            GROUP BY 1
            ORDER BY MIN(FINAL_BID)
            """

            hist_df = pd.read_sql(hist_query, conn)

            log("", f)
            log(f"  {'Bid Range':<15} {'N Bids':>12} {'Total Spend':>15} {'% of Bids':>12}", f)
            log(f"  {'-'*15} {'-'*12} {'-'*15} {'-'*12}", f)

            total_bids = hist_df['N_BIDS'].sum()
            for _, row in hist_df.iterrows():
                pct = row['N_BIDS'] / total_bids * 100
                log(f"  ${row['BID_RANGE']:<14} {row['N_BIDS']:>12,} ${row['TOTAL_SPEND']:>13,.2f} {pct:>11.1f}%", f)

            log("", f)

            # -----------------------------------------------------------------
            # Total spend validation
            # -----------------------------------------------------------------
            log("=" * 80, f)
            log("TOTAL SPEND CALCULATION", f)
            log("-" * 40, f)

            spend_query = """
            SELECT
                DATE_TRUNC('week', CREATED_AT) as week,
                COUNT(*) as n_wins,
                SUM(FINAL_BID) as total_spend,
                AVG(FINAL_BID) as avg_bid
            FROM AUCTIONS_RESULTS
            WHERE CREATED_AT BETWEEN '2025-03-14' AND '2025-09-15'
              AND IS_WINNER = TRUE
              AND FINAL_BID IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """

            spend_df = pd.read_sql(spend_query, conn)

            log("", f)
            log(f"  {'Week':<12} {'N Wins':>12} {'Total Spend':>15} {'Avg Bid':>12}", f)
            log(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*12}", f)

            for _, row in spend_df.iterrows():
                log(f"  {str(row['WEEK'])[:10]:<12} {row['N_WINS']:>12,} ${row['TOTAL_SPEND']:>13,.2f} ${row['AVG_BID']:>10.4f}", f)

            log("", f)

            total_wins = spend_df['N_WINS'].sum()
            total_spend = spend_df['TOTAL_SPEND'].sum()
            log(f"  TOTAL: {total_wins:,} wins, ${total_spend:,.2f} spend", f)
            log(f"  Average CPC: ${total_spend/total_wins:.4f}", f)
            log("", f)

            conn.close()
            log("  [SUCCESS] Snowflake connection closed.", f)

        except Exception as e:
            log(f"  [ERROR] Query failed: {e}", f)
            return

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        log("  KEY FINDINGS:", f)
        log("    1. FINAL_BID represents the bid amount for each auction slot", f)
        log("    2. Total spend = SUM(FINAL_BID) for winning bids", f)
        log("    3. This is the cost incurred for each impression/click won", f)
        log("", f)

        log("  VALIDATION:", f)
        log("    - Check if FINAL_BID aligns with advertiser's actual billing", f)
        log("    - Verify if this is first-price or second-price auction", f)
        log("    - Confirm unit of measurement (dollars, cents, or other)", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


def analyze_local_sample(f):
    """Analyze local sample data structure when Snowflake unavailable."""
    log("", f)
    log("=" * 80, f)
    log("LOCAL SAMPLE DATA STRUCTURE", f)
    log("-" * 40, f)
    log("", f)

    RAW_DATA_DIR = BASE_DIR / ".." / "shopping-sessions" / "data"
    ar_path = RAW_DATA_DIR / "raw_sample_auctions_results.parquet"

    if ar_path.exists():
        ar = pd.read_parquet(ar_path)
        log(f"  Sample auction results: {len(ar):,} rows", f)
        log(f"  Columns: {list(ar.columns)}", f)
        log("", f)

        if 'FINAL_BID' in ar.columns:
            log("  FINAL_BID column found in sample.", f)
            log(f"    Min: {ar['FINAL_BID'].min()}", f)
            log(f"    Max: {ar['FINAL_BID'].max()}", f)
            log(f"    Mean: {ar['FINAL_BID'].mean():.4f}", f)
        else:
            log("  [NOTE] FINAL_BID column NOT in sample data.", f)
            log("  Available columns:", f)
            for col in ar.columns:
                log(f"    - {col}", f)
    else:
        log(f"  [ERROR] Sample file not found: {ar_path}", f)

    log("", f)


if __name__ == "__main__":
    main()
