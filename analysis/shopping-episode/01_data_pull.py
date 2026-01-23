#!/usr/bin/env python3
"""
01_data_pull.py
Extract 2% deterministic user sample from Snowflake and construct shopping episodes.

RESEARCH HYPOTHESES:
- Episode-level analysis requires coherent user journeys (not random row samples)
- 48-hour inactivity gap defines episode boundaries
- Top 10 bids per auction sufficient for counterfactual analysis
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "01_data_pull.txt"

# Parameters
SAMPLE_RATE = 0.5  # 0.5% of users (small sample for testing)
ANALYSIS_DAYS = 7  # Analysis window (1 week for testing)
LOOKBACK_DAYS = 7  # For prior affinity features
EPISODE_GAP_HOURS = 48  # Inactivity threshold
TOP_BIDS_PER_AUCTION = 10  # Limit for AUCTIONS_RESULTS


def log(msg: str, file=None):
    """Print and optionally write to file."""
    print(msg)
    if file:
        file.write(msg + "\n")


def convert_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary/bytearray columns to hex strings for hashability."""
    for col in df.columns:
        if df[col].dtype == object and len(df) > 0:
            first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(first_val, (bytes, bytearray)):
                df[col] = df[col].apply(lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else x)
    return df


def get_snowflake_connection():
    """Create Snowflake connection from environment variables."""
    import snowflake.connector

    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )


def build_sampled_users_cte(start_date: str, end_date: str) -> str:
    """Build CTE for deterministic 2% user sample."""
    return f"""
    WITH SAMPLED_USERS AS (
        SELECT DISTINCT OPAQUE_USER_ID AS USER_ID
        FROM AUCTIONS_USERS
        WHERE CREATED_AT BETWEEN '{start_date}' AND '{end_date}'
        AND MOD(ABS(HASH(OPAQUE_USER_ID)), 100) < {SAMPLE_RATE}
    )
    """


def extract_auctions_users(conn, start_date: str, end_date: str, log_file) -> pd.DataFrame:
    """Extract AUCTIONS_USERS for sampled users."""
    log("\n--- Extracting AUCTIONS_USERS ---", log_file)

    cte = build_sampled_users_cte(start_date, end_date)
    query = f"""
    {cte}
    SELECT
        au.AUCTION_ID,
        au.OPAQUE_USER_ID AS USER_ID,
        au.CREATED_AT,
        au.PLACEMENT
    FROM AUCTIONS_USERS au
    INNER JOIN SAMPLED_USERS su ON au.OPAQUE_USER_ID = su.USER_ID
    WHERE au.CREATED_AT BETWEEN '{start_date}' AND '{end_date}'
    """

    df = pd.read_sql(query, conn)
    df = convert_binary_columns(df)
    log(f"  Rows: {len(df):,}", log_file)
    log(f"  Unique users: {df['USER_ID'].nunique():,}", log_file)
    log(f"  Unique auctions: {df['AUCTION_ID'].nunique():,}", log_file)

    return df


def extract_auctions_results(conn, start_date: str, end_date: str, log_file) -> pd.DataFrame:
    """Extract AUCTIONS_RESULTS (top 10 bids per auction) using JOIN."""
    log("\n--- Extracting AUCTIONS_RESULTS (top 10 per auction) ---", log_file)
    log("  Using JOIN approach (single query)...", log_file)

    # Use CTE to get sampled auctions, then JOIN to results
    cte = build_sampled_users_cte(start_date, end_date)
    query = f"""
    {cte},
    SAMPLED_AUCTIONS AS (
        SELECT DISTINCT au.AUCTION_ID
        FROM AUCTIONS_USERS au
        INNER JOIN SAMPLED_USERS su ON au.OPAQUE_USER_ID = su.USER_ID
        WHERE au.CREATED_AT BETWEEN '{start_date}' AND '{end_date}'
    )
    SELECT
        ar.AUCTION_ID,
        ar.VENDOR_ID,
        ar.CAMPAIGN_ID,
        ar.PRODUCT_ID,
        ar.RANKING,
        ar.IS_WINNER,
        ar.CREATED_AT,
        ar.QUALITY,
        ar.FINAL_BID,
        ar.PRICE,
        ar.CONVERSION_RATE,
        ar.PACING
    FROM AUCTIONS_RESULTS ar
    INNER JOIN SAMPLED_AUCTIONS sa ON ar.AUCTION_ID = sa.AUCTION_ID
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ar.AUCTION_ID ORDER BY ar.RANKING ASC) <= {TOP_BIDS_PER_AUCTION}
    """

    df = pd.read_sql(query, conn)
    df = convert_binary_columns(df)
    log(f"  Rows: {len(df):,}", log_file)
    log(f"  Unique auctions: {df['AUCTION_ID'].nunique():,}", log_file)
    log(f"  Unique vendors: {df['VENDOR_ID'].nunique():,}", log_file)

    return df


def extract_impressions(conn, start_date: str, end_date: str, log_file) -> pd.DataFrame:
    """Extract IMPRESSIONS for sampled users."""
    log("\n--- Extracting IMPRESSIONS ---", log_file)

    cte = build_sampled_users_cte(start_date, end_date)
    query = f"""
    {cte}
    SELECT
        i.INTERACTION_ID,
        i.AUCTION_ID,
        i.PRODUCT_ID,
        i.USER_ID,
        i.CAMPAIGN_ID,
        i.VENDOR_ID,
        i.OCCURRED_AT
    FROM IMPRESSIONS i
    INNER JOIN SAMPLED_USERS su ON i.USER_ID = su.USER_ID
    WHERE i.OCCURRED_AT BETWEEN '{start_date}' AND '{end_date}'
    """

    df = pd.read_sql(query, conn)
    df = convert_binary_columns(df)
    log(f"  Rows: {len(df):,}", log_file)
    log(f"  Unique users: {df['USER_ID'].nunique():,}", log_file)

    return df


def extract_clicks(conn, start_date: str, end_date: str, log_file) -> pd.DataFrame:
    """Extract CLICKS for sampled users."""
    log("\n--- Extracting CLICKS ---", log_file)

    cte = build_sampled_users_cte(start_date, end_date)
    query = f"""
    {cte}
    SELECT
        c.INTERACTION_ID,
        c.AUCTION_ID,
        c.PRODUCT_ID,
        c.USER_ID,
        c.CAMPAIGN_ID,
        c.VENDOR_ID,
        c.OCCURRED_AT
    FROM CLICKS c
    INNER JOIN SAMPLED_USERS su ON c.USER_ID = su.USER_ID
    WHERE c.OCCURRED_AT BETWEEN '{start_date}' AND '{end_date}'
    """

    df = pd.read_sql(query, conn)
    df = convert_binary_columns(df)
    log(f"  Rows: {len(df):,}", log_file)
    log(f"  Unique users: {df['USER_ID'].nunique():,}", log_file)

    return df


def extract_purchases(conn, start_date: str, end_date: str, log_file) -> pd.DataFrame:
    """Extract ALL purchases for sampled users (including organic)."""
    log("\n--- Extracting PURCHASES (all for sampled users) ---", log_file)

    cte = build_sampled_users_cte(start_date, end_date)
    query = f"""
    {cte}
    SELECT
        p.PURCHASE_ID,
        p.PURCHASED_AT,
        p.PRODUCT_ID,
        p.QUANTITY,
        p.UNIT_PRICE,
        p.USER_ID,
        p.PURCHASE_LINE
    FROM PURCHASES p
    INNER JOIN SAMPLED_USERS su ON p.USER_ID = su.USER_ID
    WHERE p.PURCHASED_AT BETWEEN '{start_date}' AND '{end_date}'
    """

    df = pd.read_sql(query, conn)
    df = convert_binary_columns(df)
    df['SPEND'] = df['QUANTITY'] * df['UNIT_PRICE']
    log(f"  Rows: {len(df):,}", log_file)
    log(f"  Unique users: {df['USER_ID'].nunique():,}", log_file)
    log(f"  Total spend: ${df['SPEND'].sum():,.2f}", log_file)

    return df


def extract_catalog(conn, product_ids: list, log_file) -> pd.DataFrame:
    """Extract CATALOG for products in events."""
    log("\n--- Extracting CATALOG ---", log_file)

    # Process in batches
    batch_size = 10000
    all_dfs = []

    for i in tqdm(range(0, len(product_ids), batch_size), desc="Extracting catalog"):
        batch = product_ids[i:i+batch_size]
        batch_str = "', '".join(batch)

        query = f"""
        SELECT
            PRODUCT_ID,
            NAME,
            ACTIVE,
            CATEGORIES,
            PRICE,
            VENDORS
        FROM CATALOG
        WHERE PRODUCT_ID IN ('{batch_str}')
        """

        df_batch = pd.read_sql(query, conn)
        df_batch = convert_binary_columns(df_batch)
        all_dfs.append(df_batch)

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    log(f"  Rows: {len(df):,}", log_file)

    return df


def construct_episodes(df_auctions: pd.DataFrame, df_impressions: pd.DataFrame,
                       df_clicks: pd.DataFrame, df_purchases: pd.DataFrame,
                       log_file) -> pd.DataFrame:
    """Construct episodes from user events using 48-hour gap."""
    log("\n" + "=" * 80, log_file)
    log("EPISODE CONSTRUCTION", log_file)
    log("=" * 80, log_file)

    # Collect all events with timestamps
    events = []

    # Auctions (earliest user engagement)
    if not df_auctions.empty:
        df_au = df_auctions[['USER_ID', 'CREATED_AT']].copy()
        df_au['EVENT_TYPE'] = 'auction'
        df_au = df_au.rename(columns={'CREATED_AT': 'OCCURRED_AT'})
        events.append(df_au)

    # Impressions
    if not df_impressions.empty:
        df_imp = df_impressions[['USER_ID', 'OCCURRED_AT']].copy()
        df_imp['EVENT_TYPE'] = 'impression'
        events.append(df_imp)

    # Clicks
    if not df_clicks.empty:
        df_clk = df_clicks[['USER_ID', 'OCCURRED_AT']].copy()
        df_clk['EVENT_TYPE'] = 'click'
        events.append(df_clk)

    # Purchases
    if not df_purchases.empty:
        df_pur = df_purchases[['USER_ID', 'PURCHASED_AT']].copy()
        df_pur['EVENT_TYPE'] = 'purchase'
        df_pur = df_pur.rename(columns={'PURCHASED_AT': 'OCCURRED_AT'})
        events.append(df_pur)

    if not events:
        log("  ERROR: No events to process", log_file)
        return pd.DataFrame()

    df_events = pd.concat(events, ignore_index=True)
    df_events['OCCURRED_AT'] = pd.to_datetime(df_events['OCCURRED_AT'])

    log(f"  Total events: {len(df_events):,}", log_file)
    log(f"  Unique users: {df_events['USER_ID'].nunique():,}", log_file)

    # Sort by user and time
    df_events = df_events.sort_values(['USER_ID', 'OCCURRED_AT']).reset_index(drop=True)

    # Calculate time diff within user
    df_events['TIME_DIFF'] = df_events.groupby('USER_ID')['OCCURRED_AT'].diff()
    df_events['TIME_DIFF_HOURS'] = df_events['TIME_DIFF'].dt.total_seconds() / 3600

    # Assign episode ID (increment when gap > 48 hours)
    df_events['NEW_EPISODE'] = (df_events['TIME_DIFF_HOURS'] > EPISODE_GAP_HOURS) | df_events['TIME_DIFF_HOURS'].isna()
    df_events['EPISODE_NUM'] = df_events.groupby('USER_ID')['NEW_EPISODE'].cumsum()
    df_events['EPISODE_ID'] = df_events['USER_ID'].astype(str) + '_' + df_events['EPISODE_NUM'].astype(str)

    # Episode statistics
    episode_stats = df_events.groupby('EPISODE_ID').agg({
        'USER_ID': 'first',
        'OCCURRED_AT': ['min', 'max', 'count'],
        'EVENT_TYPE': lambda x: (x == 'impression').sum()
    }).reset_index()
    episode_stats.columns = ['EPISODE_ID', 'USER_ID', 'START_TIME', 'END_TIME', 'EVENT_COUNT', 'IMPRESSION_COUNT']
    episode_stats['DURATION_HOURS'] = (episode_stats['END_TIME'] - episode_stats['START_TIME']).dt.total_seconds() / 3600

    log(f"\n  Total episodes: {len(episode_stats):,}", log_file)
    log(f"  Episodes with impressions: {(episode_stats['IMPRESSION_COUNT'] > 0).sum():,}", log_file)
    log(f"  Duration (median): {episode_stats['DURATION_HOURS'].median():.2f} hours", log_file)
    log(f"  Duration (mean): {episode_stats['DURATION_HOURS'].mean():.2f} hours", log_file)
    log(f"  Duration (95th): {episode_stats['DURATION_HOURS'].quantile(0.95):.2f} hours", log_file)

    # Filter to episodes with at least 1 impression
    valid_episodes = episode_stats[episode_stats['IMPRESSION_COUNT'] > 0]['EPISODE_ID'].tolist()
    df_events = df_events[df_events['EPISODE_ID'].isin(valid_episodes)]
    episode_stats = episode_stats[episode_stats['EPISODE_ID'].isin(valid_episodes)]

    log(f"\n  After filtering (>=1 impression):", log_file)
    log(f"    Episodes: {len(episode_stats):,}", log_file)
    log(f"    Events: {len(df_events):,}", log_file)

    return df_events, episode_stats


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("01_DATA_PULL (Shopping Episode Extraction)", f)
        log("=" * 80, f)
        log(f"\nTimestamp: {datetime.now()}", f)

        log("\n--- PARAMETERS ---", f)
        log(f"  Sample rate: {SAMPLE_RATE}%", f)
        log(f"  Analysis window: {ANALYSIS_DAYS} days", f)
        log(f"  Lookback window: {LOOKBACK_DAYS} days", f)
        log(f"  Episode gap: {EPISODE_GAP_HOURS} hours", f)
        log(f"  Top bids per auction: {TOP_BIDS_PER_AUCTION}", f)

        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=ANALYSIS_DAYS + LOOKBACK_DAYS)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        log(f"\n  Date range: {start_date_str} to {end_date_str}", f)

        try:
            log("\n--- CONNECTING TO SNOWFLAKE ---", f)
            conn = get_snowflake_connection()
            log("  Connected successfully", f)

            # Extract tables
            df_auctions = extract_auctions_users(conn, start_date_str, end_date_str, f)

            if df_auctions.empty:
                log("\nERROR: No auctions found for sampled users", f)
                return

            # Use JOIN-based extraction (much faster than passing IDs)
            df_bids = extract_auctions_results(conn, start_date_str, end_date_str, f)

            df_impressions = extract_impressions(conn, start_date_str, end_date_str, f)
            df_clicks = extract_clicks(conn, start_date_str, end_date_str, f)
            df_purchases = extract_purchases(conn, start_date_str, end_date_str, f)

            # Collect product IDs for catalog
            product_ids = set()
            if not df_impressions.empty:
                product_ids.update(df_impressions['PRODUCT_ID'].dropna().unique())
            if not df_clicks.empty:
                product_ids.update(df_clicks['PRODUCT_ID'].dropna().unique())
            if not df_purchases.empty:
                product_ids.update(df_purchases['PRODUCT_ID'].dropna().unique())

            product_ids = list(product_ids)
            df_catalog = extract_catalog(conn, product_ids, f)

            conn.close()
            log("\n  Connection closed", f)

            # Construct episodes
            df_events, episode_stats = construct_episodes(
                df_auctions, df_impressions, df_clicks, df_purchases, f
            )

            if df_events.empty:
                log("\nERROR: No valid episodes constructed", f)
                return

            # Save outputs
            log("\n--- SAVING DATA ---", f)

            df_auctions.to_parquet(DATA_DIR / "auctions_users.parquet", index=False)
            log(f"  Saved: auctions_users.parquet ({len(df_auctions):,} rows)", f)

            df_bids.to_parquet(DATA_DIR / "auctions_results.parquet", index=False)
            log(f"  Saved: auctions_results.parquet ({len(df_bids):,} rows)", f)

            df_impressions.to_parquet(DATA_DIR / "impressions.parquet", index=False)
            log(f"  Saved: impressions.parquet ({len(df_impressions):,} rows)", f)

            df_clicks.to_parquet(DATA_DIR / "clicks.parquet", index=False)
            log(f"  Saved: clicks.parquet ({len(df_clicks):,} rows)", f)

            df_purchases.to_parquet(DATA_DIR / "purchases.parquet", index=False)
            log(f"  Saved: purchases.parquet ({len(df_purchases):,} rows)", f)

            df_catalog.to_parquet(DATA_DIR / "catalog.parquet", index=False)
            log(f"  Saved: catalog.parquet ({len(df_catalog):,} rows)", f)

            df_events.to_parquet(DATA_DIR / "events.parquet", index=False)
            log(f"  Saved: events.parquet ({len(df_events):,} rows)", f)

            episode_stats.to_parquet(DATA_DIR / "episodes.parquet", index=False)
            log(f"  Saved: episodes.parquet ({len(episode_stats):,} rows)", f)

            log("\n" + "=" * 80, f)
            log("01_DATA_PULL COMPLETE", f)
            log("=" * 80, f)

        except Exception as e:
            log(f"\nERROR: {e}", f)
            import traceback
            log(traceback.format_exc(), f)
            raise


if __name__ == "__main__":
    main()
