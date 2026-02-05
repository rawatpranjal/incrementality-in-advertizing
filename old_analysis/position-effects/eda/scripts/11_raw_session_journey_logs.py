#!/usr/bin/env python3
"""
Raw Session Journey Logs

Dumps human-readable logs for random auctions with clicks,
showing the full user journey: auction → impressions → clicks,
with product catalog information.

Data model:
  User searches → Auction runs → Rankings frozen for session
                      ↓
  User scrolls from top → Impressions logged (with timestamps)
                      ↓
  User clicks → Click logged (with timestamp)

Usage:
    python 11_raw_session_journey_logs.py --round round1
    python 11_raw_session_journey_logs.py --round round2
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_BASE = BASE_DIR / "0_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_AUCTIONS = 10
SEED = 42


def get_data_paths(round_name):
    """Return data paths for specified round."""
    if round_name == "round1":
        return {
            'auctions_results': DATA_BASE / "round1/auctions_results_all.parquet",
            'auctions_users': DATA_BASE / "round1/auctions_users_all.parquet",
            'impressions': DATA_BASE / "round1/impressions_all.parquet",
            'clicks': DATA_BASE / "round1/clicks_all.parquet",
            'catalog': DATA_BASE / "round1/catalog_all.parquet",
        }
    elif round_name == "round2":
        return {
            'auctions_results': DATA_BASE / "round2/auctions_results_r2.parquet",
            'auctions_users': DATA_BASE / "round2/auctions_users_r2.parquet",
            'impressions': DATA_BASE / "round2/impressions_r2.parquet",
            'clicks': DATA_BASE / "round2/clicks_r2.parquet",
            'catalog': DATA_BASE / "round2/catalog_r2.parquet",
        }
    else:
        raise ValueError(f"Unknown round: {round_name}")


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# HELPERS
# =============================================================================
def truncate_name(name, max_len=40):
    """Truncate product name to max_len characters."""
    if pd.isna(name):
        return "[Unknown]"
    name = str(name)
    if len(name) > max_len:
        return name[:max_len-3] + "..."
    return name


def format_price(price):
    """Format price as currency string."""
    if pd.isna(price) or price == 0:
        return "--"
    return f"${price:.0f}"


def format_time(ts):
    """Format timestamp as HH:MM:SS.mmm"""
    if pd.isna(ts):
        return "--:--:--.---"
    return ts.strftime("%H:%M:%S.%f")[:-3]


def format_delta(seconds):
    """Format time delta in seconds."""
    if pd.isna(seconds):
        return "--"
    if seconds < 60:
        return f"+{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"+{minutes}m{secs:.0f}s"


def classify_scroll_pattern(impressions_df):
    """Classify scroll pattern based on timestamp uniqueness."""
    if len(impressions_df) <= 1:
        return "SINGLE", 0.0

    n_unique_ts = impressions_df['OCCURRED_AT'].nunique()
    n_total = len(impressions_df)
    unique_ratio = n_unique_ts / n_total

    if unique_ratio > 0.5:
        return "SEQUENTIAL", unique_ratio
    elif unique_ratio < 0.2:
        return "BATCHED", unique_ratio
    else:
        return "MIXED", unique_ratio


def get_price_column(catalog):
    """Get the price column name from catalog."""
    for col in ['CATALOG_PRICE', 'PRICE']:
        if col in catalog.columns:
            return col
    return None


def get_user_column(au):
    """Get the user column name from auctions_users."""
    for col in ['USER_ID', 'OPAQUE_USER_ID']:
        if col in au.columns:
            return col
    return None


# =============================================================================
# AUCTION PROCESSING
# =============================================================================
def process_auction(auction_id, ar, au, impressions, clicks, catalog, price_col, user_col, f):
    """Process and output logs for a single auction."""

    # Get auction metadata
    auction_meta = au[au['AUCTION_ID'] == auction_id].iloc[0]
    placement = auction_meta.get('PLACEMENT', 'N/A')
    user_id = auction_meta.get(user_col, 'N/A') if user_col else 'N/A'
    auction_created = auction_meta['CREATED_AT']

    # Get all bids for this auction
    auction_bids = ar[ar['AUCTION_ID'] == auction_id].copy()
    auction_bids = auction_bids.sort_values('RANKING')

    # Join catalog for product names
    catalog_cols = ['PRODUCT_ID', 'NAME']
    if price_col:
        catalog_cols.append(price_col)

    auction_bids = auction_bids.merge(
        catalog[catalog_cols],
        on='PRODUCT_ID',
        how='left'
    )

    # Get impressions for this auction
    auction_imps = impressions[impressions['AUCTION_ID'] == auction_id].copy()
    auction_imps = auction_imps.sort_values('OCCURRED_AT')
    auction_imps = auction_imps.merge(
        catalog[catalog_cols],
        on='PRODUCT_ID',
        how='left'
    )
    # Add ranking from bids
    auction_imps = auction_imps.merge(
        auction_bids[['PRODUCT_ID', 'RANKING']],
        on='PRODUCT_ID',
        how='left'
    )

    # Get clicks for this auction
    auction_clicks = clicks[clicks['AUCTION_ID'] == auction_id].copy()
    auction_clicks = auction_clicks.sort_values('OCCURRED_AT')
    auction_clicks = auction_clicks.merge(
        catalog[catalog_cols],
        on='PRODUCT_ID',
        how='left'
    )
    # Add ranking from bids
    auction_clicks = auction_clicks.merge(
        auction_bids[['PRODUCT_ID', 'RANKING']],
        on='PRODUCT_ID',
        how='left'
    )

    # Placement description
    placement_desc = {
        '1': 'Homepage/Search',
        '2': 'PDP (Product Detail Page)',
        '3': 'Category/Listing',
        '5': 'Cart/Checkout'
    }.get(str(placement), f'Type {placement}')

    # Header
    log("", f)
    log("-" * 80, f)
    log(f"AUCTION: {auction_id}", f)
    log("-" * 80, f)
    log(f"Placement: {placement} ({placement_desc})", f)
    log(f"User: {user_id}", f)
    log(f"Auction Created: {auction_created}", f)
    log("", f)

    # All bids
    n_winners = auction_bids['IS_WINNER'].sum()
    n_bidders = len(auction_bids)

    log(f"AUCTION RESULTS ({n_bidders} bids, {n_winners} winners, sorted by RANKING):", f)

    if 'QUALITY' in auction_bids.columns and 'FINAL_BID' in auction_bids.columns:
        log(f"{'Rank':>4} | {'Winner':>6} | {'Product Name':<42} | {'Price':>7} | {'Quality':>8} | {'Bid':>7}", f)
        log(f"{'-'*4}-+-{'-'*6}-+-{'-'*42}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}", f)

        for _, row in auction_bids.iterrows():
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            winner = "YES" if row['IS_WINNER'] else "NO"
            name = truncate_name(row['NAME'], 42)
            price = format_price(row.get(price_col)) if price_col else '--'
            quality = f"{row['QUALITY']:.4f}" if pd.notna(row.get('QUALITY')) else '--'
            bid = f"{row['FINAL_BID']:.2f}" if pd.notna(row.get('FINAL_BID')) else '--'

            log(f"{rank:>4} | {winner:>6} | {name:<42} | {price:>7} | {quality:>8} | {bid:>7}", f)
    else:
        log(f"{'Rank':>4} | {'Winner':>6} | {'Product Name':<42} | {'Price':>7}", f)
        log(f"{'-'*4}-+-{'-'*6}-+-{'-'*42}-+-{'-'*7}", f)

        for _, row in auction_bids.iterrows():
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            winner = "YES" if row['IS_WINNER'] else "NO"
            name = truncate_name(row['NAME'], 42)
            price = format_price(row.get(price_col)) if price_col else '--'

            log(f"{rank:>4} | {winner:>6} | {name:<42} | {price:>7}", f)

    log("", f)

    # Impressions (chronological)
    first_imp_time = None
    if len(auction_imps) > 0:
        first_imp_time = auction_imps['OCCURRED_AT'].min()

        log(f"IMPRESSIONS ({len(auction_imps)} total, chronological - shows scroll behavior):", f)
        log(f"{'Time':>12} | {'Rank':>4} | {'Product Name':<42} | {'Price':>7} | {'+Δt':>10}", f)
        log(f"{'-'*12}-+-{'-'*4}-+-{'-'*42}-+-{'-'*7}-+-{'-'*10}", f)

        prev_ts = None
        for _, row in auction_imps.iterrows():
            ts = format_time(row['OCCURRED_AT'])
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            name = truncate_name(row['NAME'], 42)
            price = format_price(row.get(price_col)) if price_col else '--'
            delta_secs = (row['OCCURRED_AT'] - first_imp_time).total_seconds()
            delta = format_delta(delta_secs)

            # Mark batching
            batch_marker = ""
            if prev_ts is not None and row['OCCURRED_AT'] == prev_ts:
                batch_marker = " <- batched"
            elif prev_ts is not None and delta_secs > 0:
                batch_marker = " <- scrolled"

            log(f"{ts:>12} | {rank:>4} | {name:<42} | {price:>7} | {delta:>10}{batch_marker}", f)
            prev_ts = row['OCCURRED_AT']
    else:
        log("IMPRESSIONS: None recorded", f)

    log("", f)

    # Clicks (chronological)
    if len(auction_clicks) > 0:
        log(f"CLICKS ({len(auction_clicks)} total, chronological):", f)
        log(f"{'Time':>12} | {'Rank':>4} | {'Product Name':<42} | {'Price':>7} | {'Since First Imp':>15}", f)
        log(f"{'-'*12}-+-{'-'*4}-+-{'-'*42}-+-{'-'*7}-+-{'-'*15}", f)

        for _, row in auction_clicks.iterrows():
            ts = format_time(row['OCCURRED_AT'])
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            name = truncate_name(row['NAME'], 42)
            price = format_price(row.get(price_col)) if price_col else '--'

            # Time since first impression
            if first_imp_time is not None:
                delta_secs = (row['OCCURRED_AT'] - first_imp_time).total_seconds()
                since_imp = format_delta(delta_secs)
            else:
                since_imp = "--"

            log(f"{ts:>12} | {rank:>4} | {name:<42} | {price:>7} | {since_imp:>15}", f)
    else:
        log("CLICKS: None recorded", f)

    log("", f)

    # Summary
    scroll_pattern, unique_ratio = classify_scroll_pattern(auction_imps)

    if len(auction_imps) > 0:
        session_duration = (auction_imps['OCCURRED_AT'].max() - auction_imps['OCCURRED_AT'].min()).total_seconds()
    else:
        session_duration = 0

    log("SUMMARY:", f)
    log(f"  - Total bids: {n_bidders}", f)
    log(f"  - Winners (shown): {n_winners}", f)
    log(f"  - Impressions: {len(auction_imps)}", f)
    log(f"  - Clicks: {len(auction_clicks)}", f)
    log(f"  - Session duration: {session_duration:.1f} seconds", f)
    log(f"  - Scroll pattern: {scroll_pattern} ({unique_ratio*100:.0f}% of impressions have unique timestamps)", f)
    log("", f)


# =============================================================================
# AGGREGATE ANALYSIS
# =============================================================================
def scroll_pattern_analysis(clicked_auctions, impressions, f):
    """Analyze scroll patterns across clicked auctions."""
    log("", f)
    log("=" * 80, f)
    log("AGGREGATE STATISTICS", f)
    log("=" * 80, f)
    log("", f)

    # Sample for speed
    sample_size = min(1000, len(clicked_auctions))
    np.random.seed(SEED)
    sample_auctions = np.random.choice(clicked_auctions, sample_size, replace=False)

    patterns = {'SEQUENTIAL': 0, 'BATCHED': 0, 'MIXED': 0, 'SINGLE': 0}
    for auction_id in tqdm(sample_auctions, desc="Analyzing scroll patterns"):
        auction_imps = impressions[impressions['AUCTION_ID'] == auction_id]
        pattern, _ = classify_scroll_pattern(auction_imps)
        patterns[pattern] += 1

    log("Scroll pattern distribution (sample of clicked auctions):", f)
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        pct = count / sample_size * 100
        log(f"  {pattern}: {count} ({pct:.1f}%)", f)
    log("", f)


def click_rate_by_ranking(impressions, clicks, ar, f):
    """Analyze click rate by ranking."""
    log("Click rate by ranking (all data):", f)

    # Build impression-level dataset with click indicator
    imp_click = impressions.merge(
        clicks[['AUCTION_ID', 'PRODUCT_ID']].drop_duplicates().assign(clicked=1),
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )
    imp_click['clicked'] = imp_click['clicked'].fillna(0)

    # Add ranking
    imp_click = imp_click.merge(
        ar[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    rank_ctr = imp_click.groupby('RANKING').agg(
        impressions=('clicked', 'count'),
        clicks=('clicked', 'sum')
    )
    rank_ctr['ctr'] = rank_ctr['clicks'] / rank_ctr['impressions'] * 100

    log(f"{'Rank':>4} | {'Impressions':>12} | {'Clicks':>8} | {'CTR':>8}", f)
    log(f"{'-'*4}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}", f)

    for rank in sorted(rank_ctr.index[:15]):  # Top 15 ranks
        row = rank_ctr.loc[rank]
        log(f"{int(rank):>4} | {int(row['impressions']):>12,} | {int(row['clicks']):>8,} | {row['ctr']:>7.2f}%", f)

    log("", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Raw session journey logs EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"11_raw_session_journey_logs_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("RAW SESSION JOURNEY LOGS", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)
        log("", f)
        log(f"Sampling {N_AUCTIONS} random auctions where clicks occurred.", f)
        log("Shows full user journey: auction -> impressions -> clicks.", f)
        log("", f)

        paths = get_data_paths(args.round)

        # Load data
        log("-" * 40, f)
        log("LOADING DATA", f)
        log("-" * 40, f)

        ar = pd.read_parquet(paths['auctions_results'])
        log(f"  Auctions results: {len(ar):,} bids", f)

        au = pd.read_parquet(paths['auctions_users'])
        log(f"  Auctions users: {len(au):,} records", f)

        impressions = pd.read_parquet(paths['impressions'])
        log(f"  Impressions: {len(impressions):,}", f)

        clicks = pd.read_parquet(paths['clicks'])
        log(f"  Clicks: {len(clicks):,}", f)

        catalog = None
        if paths['catalog'].exists():
            catalog = pd.read_parquet(paths['catalog'])
            log(f"  Catalog: {len(catalog):,} products", f)
        else:
            log("  Catalog: Not found", f)
            catalog = pd.DataFrame(columns=['PRODUCT_ID', 'NAME'])

        log("", f)

        # Determine column names
        price_col = get_price_column(catalog)
        user_col = get_user_column(au)

        # Find auctions with clicks
        clicked_auctions = clicks['AUCTION_ID'].unique()
        log(f"Auctions with clicks: {len(clicked_auctions):,}", f)
        log("", f)

        # Check for placement column
        has_placement = 'PLACEMENT' in au.columns

        if has_placement:
            # Get placement for clicked auctions
            clicked_with_placement = au[au['AUCTION_ID'].isin(clicked_auctions)][['AUCTION_ID', 'PLACEMENT']].drop_duplicates()
            log("Clicked auctions by placement:", f)
            placement_counts = clicked_with_placement['PLACEMENT'].value_counts().sort_index()
            for p, count in placement_counts.items():
                log(f"  Placement {p}: {count:,} auctions", f)
            log("", f)

            # Stratified sampling by placement
            np.random.seed(SEED)
            sampled_auctions = []

            # Try to get at least 2 from each placement, fill rest randomly
            for placement in sorted(clicked_with_placement['PLACEMENT'].unique()):
                placement_auctions = clicked_with_placement[clicked_with_placement['PLACEMENT'] == placement]['AUCTION_ID'].values
                n_sample = min(2, len(placement_auctions))
                sampled = np.random.choice(placement_auctions, n_sample, replace=False)
                sampled_auctions.extend(sampled)

            # Fill remaining slots randomly
            remaining = N_AUCTIONS - len(sampled_auctions)
            if remaining > 0:
                available = [a for a in clicked_auctions if a not in sampled_auctions]
                if len(available) > 0:
                    extra = np.random.choice(available, min(remaining, len(available)), replace=False)
                    sampled_auctions.extend(extra)

            sampled_auctions = sampled_auctions[:N_AUCTIONS]
        else:
            # Random sampling without placement stratification
            np.random.seed(SEED)
            sampled_auctions = list(np.random.choice(clicked_auctions, min(N_AUCTIONS, len(clicked_auctions)), replace=False))

        log(f"Sampled {len(sampled_auctions)} auctions for detailed logs.", f)
        log("", f)

        # Process each auction
        log("=" * 80, f)
        log("DETAILED SESSION LOGS", f)
        log("=" * 80, f)

        for i, auction_id in enumerate(tqdm(sampled_auctions, desc="Processing auctions")):
            log(f"\n[{i+1}/{len(sampled_auctions)}]", f)
            try:
                process_auction(auction_id, ar, au, impressions, clicks, catalog, price_col, user_col, f)
            except Exception as e:
                log(f"ERROR processing auction {auction_id}: {e}", f)

        # Aggregate statistics
        scroll_pattern_analysis(clicked_auctions, impressions, f)
        click_rate_by_ranking(impressions, clicks, ar, f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
