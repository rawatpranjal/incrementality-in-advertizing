#!/usr/bin/env python3
"""
Data Inventory vs Measurement Wishlist Gap Analysis

Audits what measurements from the ideal incrementality wishlist we actually have
(or can derive) from existing datasets. Creates single source of truth for
data availability before any further modeling.

Output: 09_data_inventory.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "09_data_inventory.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# FIELD COVERAGE ANALYSIS
# =============================================================================
def analyze_field_coverage(df, table_name, f):
    """Analyze coverage rates and statistics for each field."""
    log(f"\n{'='*80}", f)
    log(f"TABLE: {table_name}", f)
    log(f"{'='*80}", f)
    log(f"Rows: {len(df):,}", f)
    log(f"Columns: {len(df.columns)}", f)

    log(f"\n{'Field':<30} {'Type':<15} {'Non-null %':<12} {'Unique':<12} {'Sample Values'}", f)
    log(f"{'-'*30} {'-'*15} {'-'*12} {'-'*12} {'-'*40}", f)

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_pct = (df[col].notna().sum() / len(df)) * 100

        # Handle array/list columns that can't use nunique
        try:
            n_unique = df[col].nunique()
        except (TypeError, ValueError):
            # Column contains unhashable types (lists, arrays)
            n_unique = -1  # Indicates array column

        # Sample values
        try:
            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = str(sample_vals)[:40] + "..." if len(str(sample_vals)) > 40 else str(sample_vals)
        except:
            sample_str = "[array/list data]"

        unique_str = "ARRAY" if n_unique == -1 else f"{n_unique:,}"
        log(f"{col:<30} {dtype:<15} {non_null_pct:>10.1f}% {unique_str:>10} {sample_str}", f)

    # Build fields dict with error handling
    fields_dict = {}
    for col in df.columns:
        try:
            n_uniq = df[col].nunique()
        except (TypeError, ValueError):
            n_uniq = -1
        fields_dict[col] = {
            'dtype': str(df[col].dtype),
            'non_null_pct': (df[col].notna().sum() / len(df)) * 100,
            'n_unique': n_uniq
        }

    return {
        'rows': len(df),
        'columns': len(df.columns),
        'fields': fields_dict
    }

def analyze_numeric_stats(df, table_name, f):
    """Detailed stats for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return

    log(f"\n--- Numeric Field Statistics ---", f)
    log(f"{'Field':<25} {'Min':>15} {'Max':>15} {'Mean':>15} {'Std':>15} {'Zeros %':>10}", f)
    log(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*10}", f)

    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        zero_pct = (vals == 0).sum() / len(vals) * 100
        log(f"{col:<25} {vals.min():>15.4g} {vals.max():>15.4g} {vals.mean():>15.4g} {vals.std():>15.4g} {zero_pct:>9.1f}%", f)

def analyze_temporal_range(df, table_name, f):
    """Analyze timestamp columns."""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) == 0:
        return

    log(f"\n--- Temporal Range ---", f)
    for col in datetime_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        log(f"{col}: {vals.min()} to {vals.max()} (span: {vals.max() - vals.min()})", f)

# =============================================================================
# DERIVABLE METRICS ANALYSIS
# =============================================================================
def analyze_derivable_metrics(data, f):
    """Compute what can be derived from available data."""
    log(f"\n{'='*80}", f)
    log(f"DERIVABLE METRICS ANALYSIS", f)
    log(f"{'='*80}", f)

    ar = data.get('auctions_results')
    au = data.get('auctions_users')
    imp = data.get('impressions')
    clicks = data.get('clicks')
    catalog = data.get('catalog')
    sessions = data.get('sessions')
    session_items = data.get('session_items')

    # 1. Second price (from bid distribution)
    log(f"\n--- 1. SECOND PRICE (from bid landscape) ---", f)
    if ar is not None:
        winners = ar[ar['IS_WINNER'] == True].copy()
        log(f"Total winning bids: {len(winners):,}", f)

        # Get second highest bid per auction
        bid_by_auction = ar.groupby('AUCTION_ID')['FINAL_BID'].apply(lambda x: sorted(x, reverse=True))
        auctions_with_2plus = bid_by_auction[bid_by_auction.apply(len) >= 2]
        log(f"Auctions with 2+ bids: {len(auctions_with_2plus):,}", f)

        if len(auctions_with_2plus) > 0:
            second_prices = auctions_with_2plus.apply(lambda x: x[1])
            first_prices = auctions_with_2plus.apply(lambda x: x[0])
            spread = first_prices - second_prices
            log(f"First price: mean={first_prices.mean():.2f}, median={first_prices.median():.2f}", f)
            log(f"Second price: mean={second_prices.mean():.2f}, median={second_prices.median():.2f}", f)
            log(f"Spread (1st - 2nd): mean={spread.mean():.2f}, median={spread.median():.2f}", f)
            log(f"DERIVABLE: YES - full bid landscape available", f)

    # 2. Session duration (from click timestamps)
    log(f"\n--- 2. SESSION DURATION (from click timestamps) ---", f)
    if clicks is not None and len(clicks) > 0:
        clicks_per_user = clicks.groupby('USER_ID').agg({
            'OCCURRED_AT': ['min', 'max', 'count']
        })
        clicks_per_user.columns = ['first_click', 'last_click', 'n_clicks']
        clicks_per_user['duration'] = (clicks_per_user['last_click'] - clicks_per_user['first_click']).dt.total_seconds()

        multi_click = clicks_per_user[clicks_per_user['n_clicks'] > 1]
        log(f"Users with clicks: {len(clicks_per_user):,}", f)
        log(f"Users with 2+ clicks: {len(multi_click):,}", f)
        if len(multi_click) > 0:
            log(f"Session duration (multi-click users): mean={multi_click['duration'].mean():.1f}s, median={multi_click['duration'].median():.1f}s", f)
        log(f"DERIVABLE: PARTIAL - only for clicking sessions, misses non-clicking sessions", f)
    else:
        log(f"No click data available", f)

    # 3. Time to purchase (click -> purchase join needed)
    log(f"\n--- 3. TIME TO PURCHASE ---", f)
    log(f"Would require PURCHASES table with timestamps", f)
    log(f"DERIVABLE: YES if purchases data available - join clicks to purchases on user_id + product_id", f)

    # 4. User purchase history / New vs returning
    log(f"\n--- 4. USER HISTORY (from purchase data) ---", f)
    log(f"Would require PURCHASES table with user history", f)
    log(f"DERIVABLE: YES if purchases data available - first purchase date vs session date", f)

    # 5. Neighboring items (from auction results)
    log(f"\n--- 5. NEIGHBORING ITEMS (from auction lineup) ---", f)
    if ar is not None:
        winners_per_auction = ar[ar['IS_WINNER'] == True].groupby('AUCTION_ID').size()
        log(f"Winners per auction: mean={winners_per_auction.mean():.2f}, max={winners_per_auction.max()}", f)
        log(f"Auctions: {len(winners_per_auction):,}", f)
        log(f"DERIVABLE: YES - full auction lineup available from AUCTIONS_RESULTS", f)

    # 6. Bid landscape / instruments
    log(f"\n--- 6. BID LANDSCAPE / INSTRUMENTS ---", f)
    if ar is not None:
        bids_per_auction = ar.groupby('AUCTION_ID').agg({
            'FINAL_BID': ['mean', 'std', 'count'],
            'QUALITY': ['mean', 'std'],
            'RANKING': ['min', 'max']
        })
        bids_per_auction.columns = ['bid_mean', 'bid_std', 'n_bids', 'qual_mean', 'qual_std', 'rank_min', 'rank_max']
        log(f"Bids per auction: mean={bids_per_auction['n_bids'].mean():.1f}, max={bids_per_auction['n_bids'].max()}", f)
        log(f"Bid variation (std): mean={bids_per_auction['bid_std'].mean():.2f}", f)
        log(f"Quality variation (std): mean={bids_per_auction['qual_std'].mean():.4f}", f)
        log(f"DERIVABLE: YES - competitor bids available for IV construction", f)

    # 7. Session-level aggregates from session_items
    log(f"\n--- 7. SESSION-LEVEL AGGREGATES ---", f)
    if session_items is not None:
        log(f"Session items table available: {len(session_items):,} rows", f)
        log(f"Fields: position, rank, clicked, quality, bid, n_items, n_clicks", f)
        log(f"Position range: {session_items['position'].min()} to {session_items['position'].max()}", f)
        log(f"Click rate: {session_items['clicked'].mean()*100:.2f}%", f)
        log(f"DERIVABLE: YES - session-level click models already supported", f)

# =============================================================================
# FUNNEL CROSS-VALIDATION
# =============================================================================
def cross_validate_funnel(data, f):
    """Cross-validate observed funnel statistics against claimed values from prior analysis."""
    log(f"\n{'='*80}", f)
    log(f"FUNNEL CROSS-VALIDATION", f)
    log(f"{'='*80}", f)
    log(f"Comparing observed statistics against claimed funnel metrics from 08_funnel_audit.txt", f)

    # Load _all.parquet files for full dataset validation
    all_data = {}
    all_files = {
        'auctions_results': 'auctions_results_all.parquet',
        'auctions_users': 'auctions_users_all.parquet',
        'impressions': 'impressions_all.parquet',
        'clicks': 'clicks_all.parquet',
    }

    for name, filename in tqdm(all_files.items(), desc="Loading _all.parquet for validation"):
        filepath = DATA_DIR / filename
        if filepath.exists():
            all_data[name] = pd.read_parquet(filepath)
        else:
            all_data[name] = None
            log(f"WARNING: Missing {filename} - using sampled data instead", f)

    # Use _all data if available, otherwise fall back to sampled
    ar = all_data.get('auctions_results') if all_data.get('auctions_results') is not None else data.get('auctions_results')
    au = all_data.get('auctions_users') if all_data.get('auctions_users') is not None else data.get('auctions_users')
    imp = all_data.get('impressions') if all_data.get('impressions') is not None else data.get('impressions')
    clicks = all_data.get('clicks') if all_data.get('clicks') is not None else data.get('clicks')

    if ar is None or au is None:
        log(f"ERROR: Required auction data not available for validation", f)
        return

    # Helper to format match indicator
    def match_indicator(actual, claimed, tolerance=0.15):
        """Return checkmark if within tolerance, X otherwise."""
        if claimed == 0:
            return "N/A"
        ratio = actual / claimed
        if 1 - tolerance <= ratio <= 1 + tolerance:
            return "✓"
        else:
            return "✗"

    # --- TABLE VOLUMES ---
    log(f"\n--- TABLE VOLUMES ---", f)
    log(f"{'Table':<25} {'Row Count':>15}", f)
    log(f"{'-'*25} {'-'*15}", f)

    tables_info = [
        ('AUCTIONS_USERS', au),
        ('AUCTIONS_RESULTS', ar),
        ('IMPRESSIONS', imp),
        ('CLICKS', clicks),
    ]

    for table_name, df in tables_info:
        if df is not None:
            log(f"{table_name:<25} {len(df):>15,}", f)
        else:
            log(f"{table_name:<25} {'N/A':>15}", f)

    # --- AUCTION STAGE ---
    log(f"\n--- AUCTION STAGE ---", f)
    log(f"{'Metric':<35} {'Claimed':>12} {'Actual':>12} {'Match?':>8}", f)
    log(f"{'-'*35} {'-'*12} {'-'*12} {'-'*8}", f)

    # Bids per auction
    bids_per_auction = ar.groupby('AUCTION_ID').size()
    avg_bids = bids_per_auction.mean()
    claimed_bids = 47
    log(f"{'Bids per auction':<35} {claimed_bids:>12} {avg_bids:>12.1f} {match_indicator(avg_bids, claimed_bids):>8}", f)

    # Winner rate (% of bids that are winners)
    winner_rate = ar['IS_WINNER'].mean() * 100
    claimed_winner_rate = 80
    log(f"{'Winner rate (%)':<35} {claimed_winner_rate:>11}% {winner_rate:>11.1f}% {match_indicator(winner_rate, claimed_winner_rate):>8}", f)

    # Winners per auction
    winners_per_auction = ar[ar['IS_WINNER'] == True].groupby('AUCTION_ID').size()
    avg_winners = winners_per_auction.mean()
    claimed_winners = 38
    log(f"{'Winners per auction':<35} {claimed_winners:>12} {avg_winners:>12.1f} {match_indicator(avg_winners, claimed_winners):>8}", f)

    # --- IMPRESSION SELECTION ---
    log(f"\n--- IMPRESSION SELECTION ---", f)
    log(f"{'Metric':<35} {'Claimed':>12} {'Actual':>12} {'Match?':>8}", f)
    log(f"{'-'*35} {'-'*12} {'-'*12} {'-'*8}", f)

    if imp is not None:
        # Total winners vs total impressions
        total_winners = ar['IS_WINNER'].sum()
        total_impressions = len(imp)

        # % of winners that get impressions
        # Note: This is approximate since winner and impression granularity may differ
        impression_rate = (total_impressions / total_winners) * 100 if total_winners > 0 else 0
        claimed_impression_rate = 6.4
        log(f"{'Winners getting impressions (%)':<35} {claimed_impression_rate:>11.1f}% {impression_rate:>11.1f}% {match_indicator(impression_rate, claimed_impression_rate):>8}", f)

        # Impressions per auction
        n_auctions = au['AUCTION_ID'].nunique() if 'AUCTION_ID' in au.columns else len(au)
        imp_per_auction = total_impressions / n_auctions if n_auctions > 0 else 0
        claimed_imp_per_auction = 7
        log(f"{'Impressions per auction':<35} {claimed_imp_per_auction:>12} {imp_per_auction:>12.1f} {match_indicator(imp_per_auction, claimed_imp_per_auction):>8}", f)
    else:
        log(f"{'Winners getting impressions (%)':<35} {'N/A':>12} {'N/A':>12} {'N/A':>8}", f)
        log(f"{'Impressions per auction':<35} {'N/A':>12} {'N/A':>12} {'N/A':>8}", f)

    # --- CLICK BEHAVIOR ---
    log(f"\n--- CLICK BEHAVIOR ---", f)
    log(f"{'Metric':<35} {'Claimed':>12} {'Actual':>12} {'Match?':>8}", f)
    log(f"{'-'*35} {'-'*12} {'-'*12} {'-'*8}", f)

    if clicks is not None and imp is not None:
        # CTR on sponsored impressions
        total_clicks = len(clicks)
        ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
        claimed_ctr = 2.8
        log(f"{'CTR on sponsored impressions (%)':<35} {claimed_ctr:>11.1f}% {ctr:>11.1f}% {match_indicator(ctr, claimed_ctr):>8}", f)

        # Clicks per auction distribution
        if 'AUCTION_ID' in clicks.columns:
            clicks_per_auction = clicks.groupby('AUCTION_ID').size()
            all_auction_ids = au['AUCTION_ID'].unique() if 'AUCTION_ID' in au.columns else []

            auctions_with_0_clicks = n_auctions - len(clicks_per_auction)
            pct_0_clicks = (auctions_with_0_clicks / n_auctions) * 100 if n_auctions > 0 else 0
            claimed_0_clicks = 86.6
            log(f"{'Auctions with 0 clicks (%)':<35} {claimed_0_clicks:>11.1f}% {pct_0_clicks:>11.1f}% {match_indicator(pct_0_clicks, claimed_0_clicks):>8}", f)

            auctions_with_2plus = (clicks_per_auction >= 2).sum()
            pct_2plus_clicks = (auctions_with_2plus / n_auctions) * 100 if n_auctions > 0 else 0
            claimed_2plus = 3.7
            log(f"{'Auctions with 2+ clicks (%)':<35} {claimed_2plus:>11.1f}% {pct_2plus_clicks:>11.1f}% {match_indicator(pct_2plus_clicks, claimed_2plus):>8}", f)
        else:
            log(f"{'Auctions with 0 clicks (%)':<35} {'N/A':>12} {'N/A':>12} {'N/A':>8}", f)
            log(f"{'Auctions with 2+ clicks (%)':<35} {'N/A':>12} {'N/A':>12} {'N/A':>8}", f)
    else:
        log(f"{'CTR on sponsored impressions (%)':<35} {'N/A':>12} {'N/A':>12} {'N/A':>8}", f)
        log(f"{'Auctions with 0 clicks (%)':<35} {'N/A':>12} {'N/A':>12} {'N/A':>8}", f)
        log(f"{'Auctions with 2+ clicks (%)':<35} {'N/A':>12} {'N/A':>12} {'N/A':>8}", f)

    # --- PLACEMENT DISTRIBUTION ---
    log(f"\n--- PLACEMENT DISTRIBUTION ---", f)
    if 'PLACEMENT' in au.columns:
        placement_counts = au['PLACEMENT'].value_counts()
        placement_pcts = au['PLACEMENT'].value_counts(normalize=True) * 100

        log(f"{'Placement':<15} {'Count':>15} {'Percentage':>15}", f)
        log(f"{'-'*15} {'-'*15} {'-'*15}", f)

        for placement in sorted(placement_counts.index):
            count = placement_counts[placement]
            pct = placement_pcts[placement]
            log(f"{str(placement):<15} {count:>15,} {pct:>14.1f}%", f)
    else:
        log(f"PLACEMENT column not found in AUCTIONS_USERS", f)

    # --- FUNNEL SUMMARY ---
    log(f"\n--- FUNNEL FLOW SUMMARY ---", f)
    log(f"Stage                              Count                   Rate", f)
    log(f"{'-'*35} {'-'*20} {'-'*20}", f)

    n_auctions_total = n_auctions
    n_bids_total = len(ar)
    n_winners_total = ar['IS_WINNER'].sum()
    n_impressions_total = len(imp) if imp is not None else 0
    n_clicks_total = len(clicks) if clicks is not None else 0

    log(f"{'Auctions':<35} {n_auctions_total:>20,} {'(base)':>20}", f)
    log(f"{'Bids':<35} {n_bids_total:>20,} {n_bids_total/n_auctions_total:>19.1f}/auction", f)
    log(f"{'Winners':<35} {n_winners_total:>20,} {n_winners_total/n_auctions_total:>19.1f}/auction", f)
    log(f"{'Impressions':<35} {n_impressions_total:>20,} {n_impressions_total/n_auctions_total:>19.1f}/auction", f)
    log(f"{'Clicks':<35} {n_clicks_total:>20,} {n_clicks_total/n_auctions_total:>19.2f}/auction", f)

    # Conversion rates through funnel
    log(f"\n--- FUNNEL CONVERSION RATES ---", f)
    log(f"{'Transition':<40} {'Rate':>15}", f)
    log(f"{'-'*40} {'-'*15}", f)

    if n_bids_total > 0:
        log(f"{'Bid -> Winner':<40} {(n_winners_total/n_bids_total)*100:>14.1f}%", f)
    if n_winners_total > 0:
        log(f"{'Winner -> Impression':<40} {(n_impressions_total/n_winners_total)*100:>14.1f}%", f)
    if n_impressions_total > 0:
        log(f"{'Impression -> Click':<40} {(n_clicks_total/n_impressions_total)*100:>14.1f}%", f)

# =============================================================================
# WISHLIST GAP ANALYSIS
# =============================================================================
def print_gap_analysis(data, f):
    """Print comprehensive gap analysis vs ideal wishlist."""
    log(f"\n{'='*80}", f)
    log(f"WISHLIST vs REALITY: GAP ANALYSIS", f)
    log(f"{'='*80}", f)

    # Helper to check field existence
    def has_field(table_name, field_name):
        if table_name not in data or data[table_name] is None:
            return False
        return field_name in data[table_name].columns or field_name.lower() in [c.lower() for c in data[table_name].columns]

    # Define wishlist with availability assessment
    wishlist = {
        'USER-LEVEL (Per Session)': [
            ('Scroll depth / max viewport', 'NO', '-', 'Not logged; would need SDK instrumentation'),
            ('Session duration', 'PARTIAL', 'Click timestamps', 'Can compute first-to-last click time, but misses non-clicking sessions'),
            ('Entry point', 'NO', '-', 'Not in data'),
            ('Query text', 'NO', '-', 'AUCTIONS_USERS has PLACEMENT but not query'),
            ('User history', 'PARTIAL', 'PURCHASES', 'Have purchase history if available, not browse/click history'),
            ('User ID', 'YES', 'USER_ID', 'Anonymized but trackable'),
            ('New vs returning', 'DERIVABLE', 'PURCHASES', 'First purchase date vs session date'),
        ],
        'VIEWPORT-LEVEL': [
            ('Viewport enter/exit timestamp', 'NO', '-', 'Not logged'),
            ('Viewport visible duration', 'NO', '-', 'Not logged'),
            ('Scroll direction', 'NO', '-', 'Not logged'),
            ('Viewport load success', 'NO', '-', 'Not logged'),
        ],
        'IMPRESSION-LEVEL': [
            ('Render timestamp', 'YES', 'IMPRESSIONS.OCCURRED_AT', 'But batched within auction'),
            ('Viewability', 'NO', '-', 'Not logged'),
            ('Creative variant', 'NO', '-', 'Not logged'),
            ('Price displayed', 'PARTIAL', 'AUCTIONS_RESULTS.PRICE, CATALOG.PRICE', 'Have product price, may not be displayed price'),
            ('Position within viewport', 'NO', '-', 'Only have RANKING'),
            ('Neighboring items', 'DERIVABLE', 'AUCTIONS_RESULTS', 'Can reconstruct auction lineup'),
            ('Ad label visibility', 'NO', '-', 'Not logged'),
        ],
        'CLICK-LEVEL': [
            ('Click timestamp', 'YES', 'CLICKS.OCCURRED_AT', 'Distinct per click'),
            ('Click position (on ad)', 'NO', '-', 'Only that click happened'),
            ('Post-click dwell', 'NO', '-', 'Not logged'),
            ('Post-click action', 'PARTIAL', 'PURCHASES', 'Can join clicks to purchases'),
            ('Return to SERP', 'NO', '-', 'Not logged'),
            ('Subsequent clicks', 'YES', 'CLICKS', 'Multiple clicks per user trackable'),
        ],
        'AUCTION/BID-LEVEL': [
            ('Winning bid', 'YES', 'IS_WINNER + FINAL_BID', 'Full availability'),
            ('Second price', 'DERIVABLE', 'AUCTIONS_RESULTS', 'Sort by FINAL_BID, take 2nd'),
            ('Bid landscape', 'YES', 'AUCTIONS_RESULTS', 'Full bid distribution per auction'),
            ('Reserve price', 'NO', '-', 'Not logged'),
            ('Quality score', 'YES', 'QUALITY', 'Full availability'),
            ('Eligible ads not shown', 'PARTIAL', 'AUCTIONS_RESULTS', 'Bids with IS_WINNER=FALSE'),
            ('Budget exhaustion', 'NO', '-', 'Not logged'),
            ('Bid strategy', 'NO', '-', 'Not logged'),
            ('Pacing', 'YES', 'PACING', 'Full availability'),
            ('Conversion rate', 'YES', 'CONVERSION_RATE', "Platform's estimate"),
        ],
        'ITEM/LISTING-LEVEL': [
            ('Organic rank', 'NO', '-', 'Only sponsored positions'),
            ('Item age', 'NO', '-', 'No listing date'),
            ('Item category', 'YES', 'CATEGORIES', 'Full availability'),
            ('Seller quality', 'NO', '-', 'Not in data'),
            ('Price', 'YES', 'CATALOG_PRICE', 'Full availability'),
            ('Image quality', 'NO', '-', 'Not logged'),
            ('Title/description', 'YES', 'NAME, DESCRIPTION', 'Full availability'),
            ('Inventory/scarcity', 'NO', '-', 'Not logged'),
            ('Vendors', 'NO', '-', 'Not in current data pull'),
        ],
        'PLATFORM/UI-LEVEL': [
            ('Device type', 'NO', '-', 'Not logged'),
            ('Screen resolution', 'NO', '-', 'Not logged'),
            ('App version', 'NO', '-', 'Not logged'),
            ('Ad slot configuration', 'PARTIAL', 'AUCTIONS_RESULTS', 'Know who won which rank'),
            ('Organic slot configuration', 'NO', '-', 'Only sponsored'),
            ('A/B test assignment', 'NO', '-', 'Not logged'),
            ('Page load time', 'NO', '-', 'Not logged'),
            ('Placement', 'YES', 'PLACEMENT', 'Full availability'),
        ],
        'TEMPORAL': [
            ('Time of day', 'YES', 'All timestamps', 'Full availability'),
            ('Days since last session', 'DERIVABLE', 'By user_id', 'If history available'),
            ('Time since query', 'NO', '-', 'No query logged'),
            ('Seasonality', 'YES', 'Timestamps', 'Full availability'),
        ],
        'OUTCOME-LEVEL': [
            ('Purchase (conversion)', 'PARTIAL', 'PURCHASES', 'If table available'),
            ('Add to cart', 'NO', '-', 'Not logged'),
            ('Purchase amount', 'PARTIAL', 'UNIT_PRICE * QUANTITY', 'If purchases available'),
            ('Return/refund', 'NO', '-', 'Not logged'),
            ('Time to purchase', 'DERIVABLE', 'Click -> Purchase timestamps', 'If purchases available'),
            ('Cross-device conversion', 'NO', '-', 'Not logged'),
            ('LTV', 'DERIVABLE', 'Aggregate PURCHASES', 'If purchases available'),
        ],
    }

    # Print formatted table
    for category, items in wishlist.items():
        log(f"\n### {category}", f)
        log(f"{'Measurement':<35} {'Available?':<12} {'Source':<30} {'Notes'}", f)
        log(f"{'-'*35} {'-'*12} {'-'*30} {'-'*40}", f)
        for measurement, available, source, notes in items:
            log(f"{measurement:<35} {available:<12} {source:<30} {notes}", f)

    # Summary counts
    log(f"\n{'='*80}", f)
    log(f"SUMMARY COUNTS", f)
    log(f"{'='*80}", f)

    all_items = []
    for items in wishlist.values():
        all_items.extend(items)

    yes_count = sum(1 for _, a, _, _ in all_items if a == 'YES')
    partial_count = sum(1 for _, a, _, _ in all_items if a == 'PARTIAL')
    derivable_count = sum(1 for _, a, _, _ in all_items if a == 'DERIVABLE')
    no_count = sum(1 for _, a, _, _ in all_items if a == 'NO')
    total = len(all_items)

    log(f"YES (directly available):     {yes_count:>3} ({yes_count/total*100:>5.1f}%)", f)
    log(f"PARTIAL (limited coverage):   {partial_count:>3} ({partial_count/total*100:>5.1f}%)", f)
    log(f"DERIVABLE (can compute):      {derivable_count:>3} ({derivable_count/total*100:>5.1f}%)", f)
    log(f"NO (not available):           {no_count:>3} ({no_count/total*100:>5.1f}%)", f)
    log(f"TOTAL:                        {total:>3}", f)

# =============================================================================
# CRITICAL GAPS ASSESSMENT
# =============================================================================
def print_critical_gaps(f):
    """Identify what's missing that would be game-changers."""
    log(f"\n{'='*80}", f)
    log(f"CRITICAL GAPS: HIGH-VALUE MISSING DATA", f)
    log(f"{'='*80}", f)

    gaps = [
        ("Scroll depth / viewport tracking",
         "Would enable examination probability estimation. Currently rely on position-only proxies.",
         "HIGH"),
        ("Viewability metrics",
         "Cannot distinguish 'shown but not seen' from 'seen but not clicked'. Batched impressions compound this.",
         "HIGH"),
        ("Post-click dwell time",
         "Key satisfaction signal. Cannot measure engagement quality after click.",
         "MEDIUM"),
        ("Organic positions/clicks",
         "Only see sponsored. Cannot compare promoted vs organic behavior. Limits counterfactual analysis.",
         "HIGH"),
        ("Query text",
         "Cannot segment by user intent. All auctions treated as homogeneous contexts.",
         "MEDIUM"),
        ("Device/screen info",
         "Cannot account for viewport size affecting examination. Mobile vs desktop effects unknown.",
         "MEDIUM"),
        ("Return to SERP signal",
         "Cannot detect click-then-bounce vs engaged click. Key for quality assessment.",
         "MEDIUM"),
    ]

    log(f"{'Gap':<45} {'Impact':<10} {'Implication'}", f)
    log(f"{'-'*45} {'-'*10} {'-'*50}", f)
    for gap, implication, impact in gaps:
        log(f"{gap:<45} {impact:<10} {implication[:50]}", f)
        if len(implication) > 50:
            log(f"{'':<45} {'':<10} {implication[50:]}", f)

# =============================================================================
# WHAT WE HAVE: STRENGTHS
# =============================================================================
def print_data_strengths(f):
    """Document what we DO have that's valuable."""
    log(f"\n{'='*80}", f)
    log(f"DATA STRENGTHS: WHAT WE CAN DO", f)
    log(f"{'='*80}", f)

    strengths = [
        ("Full auction/bid landscape",
         "AUCTIONS_RESULTS contains all bids (winners + losers), QUALITY, FINAL_BID, PACING, RANKING. "
         "Enables: bid landscape instruments, competitor bid controls, RDD at winner cutoff."),
        ("User ID panel structure",
         "OPAQUE_USER_ID allows tracking users across sessions. "
         "Enables: user fixed effects, repeat behavior analysis, within-user position variation."),
        ("Rich product catalog",
         "CATEGORIES, DESCRIPTION, NAME, PRICE available. "
         "Enables: product fixed effects, category controls, NLP-based quality proxies."),
        ("Placement context",
         "PLACEMENT field distinguishes page locations. "
         "Enables: stratified analysis, placement-specific position effects."),
        ("Click/impression linkage",
         "Can join clicks to impressions via AUCTION_ID + PRODUCT_ID. "
         "Enables: click-through rate computation at item level."),
        ("Prepared session data",
         "sessions.parquet and session_items.parquet already structured for click models. "
         "Enables: PBM, DBN, SDBN estimation without additional data prep."),
    ]

    for title, description in strengths:
        log(f"\n{title}:", f)
        log(f"  {description}", f)

# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("="*80, f)
        log("DATA INVENTORY vs MEASUREMENT WISHLIST GAP ANALYSIS", f)
        log("="*80, f)
        log(f"Purpose: Audit available data against ideal incrementality measurements", f)
        log(f"Output: Single source of truth for what's available before modeling", f)

        # Load all parquet files
        log(f"\n{'='*80}", f)
        log(f"LOADING DATASETS", f)
        log(f"{'='*80}", f)

        data = {}
        parquet_files = [
            ('auctions_results', 'auctions_results_p5.parquet'),
            ('auctions_users', 'auctions_users_p5.parquet'),
            ('impressions', 'impressions_p5.parquet'),
            ('clicks', 'clicks_p5.parquet'),
            ('catalog', 'catalog_p5.parquet'),
            ('sessions', 'sessions.parquet'),
            ('session_items', 'session_items.parquet'),
        ]

        for name, filename in tqdm(parquet_files, desc="Loading parquet files"):
            filepath = DATA_DIR / filename
            if filepath.exists():
                data[name] = pd.read_parquet(filepath)
                log(f"Loaded {name}: {len(data[name]):,} rows", f)
            else:
                data[name] = None
                log(f"Missing: {filename}", f)

        # Also check for _all versions
        all_files = [
            ('auctions_results_all', 'auctions_results_all.parquet'),
            ('auctions_users_all', 'auctions_users_all.parquet'),
            ('impressions_all', 'impressions_all.parquet'),
            ('clicks_all', 'clicks_all.parquet'),
            ('catalog_all', 'catalog_all.parquet'),
        ]

        log(f"\n--- Additional datasets (all placements) ---", f)
        for name, filename in all_files:
            filepath = DATA_DIR / filename
            if filepath.exists():
                df = pd.read_parquet(filepath)
                log(f"Available {name}: {len(df):,} rows", f)
            else:
                log(f"Missing: {filename}", f)

        # Field-level analysis for each table
        log(f"\n{'='*80}", f)
        log(f"FIELD-LEVEL INVENTORY", f)
        log(f"{'='*80}", f)

        for name, df in tqdm(data.items(), desc="Analyzing tables"):
            if df is not None:
                analyze_field_coverage(df, name.upper(), f)
                analyze_numeric_stats(df, name.upper(), f)
                analyze_temporal_range(df, name.upper(), f)

        # Derivable metrics
        analyze_derivable_metrics(data, f)

        # Funnel cross-validation
        cross_validate_funnel(data, f)

        # Gap analysis
        print_gap_analysis(data, f)

        # Critical gaps
        print_critical_gaps(f)

        # Strengths
        print_data_strengths(f)

        # Final summary
        log(f"\n{'='*80}", f)
        log(f"RECOMMENDATIONS", f)
        log(f"{'='*80}", f)
        log(f"""
1. PROCEED WITH AVAILABLE DATA
   - Position-based click models (PBM, DBN, SDBN) are fully supported
   - Auction/bid landscape provides strong instruments for IV approaches
   - User panel structure enables within-user analysis
   - Catalog data supports product quality controls

2. ACKNOWLEDGE LIMITATIONS
   - No scroll/viewport data: examination probability relies on position proxy
   - No viewability: cannot distinguish seen vs shown
   - No organic data: counterfactual limited to promoted-vs-promoted comparisons
   - Batched impressions: no sequential examination signal

3. PRIORITY DATA REQUESTS (if feasible)
   - Viewability metrics (most impactful for examination modeling)
   - Organic click/impression logs (enables promoted vs organic comparison)
   - Post-click dwell time (satisfaction signal)

4. MODELING IMPLICATIONS
   - PBM assumptions not directly testable without viewport data
   - Use within-auction bid variation as instruments (available)
   - Stratify by placement to account for different contexts (available)
   - Control for product quality via QUALITY score and catalog features (available)
""", f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"Output written to: {OUTPUT_FILE}", f)
        log(f"{'='*80}", f)

if __name__ == "__main__":
    main()
