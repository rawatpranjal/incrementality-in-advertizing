#!/usr/bin/env python3
"""
Detailed Session Logs - Stored in Folder

Stores human-readable logs for auctions with clicks,
showing the full user journey: auction → impressions → clicks.
Creates individual log files per auction for easy browsing.

Output structure:
    eda/logs/{round}/
        summary.txt                 # Overview + aggregate stats
        auction_{id}_{placement}.txt  # Individual auction logs

Usage:
    python 13_session_logs_detailed.py --round round1
    python 13_session_logs_detailed.py --round round2
    python 13_session_logs_detailed.py --round round1 --n 200
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
LOGS_DIR = Path(__file__).parent.parent / "logs"

DEFAULT_N_AUCTIONS = 100
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


# =============================================================================
# HELPERS
# =============================================================================
def truncate_name(name, max_len=50):
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


def format_datetime(ts):
    """Format timestamp as YYYY-MM-DD HH:MM:SS"""
    if pd.isna(ts):
        return "----"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def format_delta(seconds):
    """Format time delta in seconds."""
    if pd.isna(seconds):
        return "--"
    if seconds < 60:
        return f"+{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"+{minutes}m{secs:.0f}s"


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


def get_placement_name(placement):
    """Get neutral placement name for filenames."""
    return f'p{placement}'


# =============================================================================
# AUCTION LOG GENERATION
# =============================================================================
def generate_auction_log(auction_id, ar, au, impressions, clicks, catalog, price_col, user_col):
    """Generate detailed log string for a single auction."""
    lines = []

    def log(msg):
        lines.append(msg)

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

    # Header
    log("=" * 80)
    log(f"AUCTION SESSION LOG")
    log("=" * 80)
    log("")
    log(f"Auction ID:      {auction_id}")
    log(f"Placement:       {placement}")
    log(f"User ID:         {user_id}")
    log(f"Auction Time:    {format_datetime(auction_created)}")
    log("")

    # Summary stats
    n_bidders = len(auction_bids)
    n_winners = auction_bids['IS_WINNER'].sum()
    n_impressions = len(auction_imps)
    n_clicks = len(auction_clicks)

    log("-" * 40)
    log("QUICK STATS")
    log("-" * 40)
    log(f"  Bids:        {n_bidders}")
    log(f"  Winners:     {n_winners}")
    log(f"  Impressions: {n_impressions}")
    log(f"  Clicks:      {n_clicks}")

    if n_impressions > 0:
        session_duration = (auction_imps['OCCURRED_AT'].max() - auction_imps['OCCURRED_AT'].min()).total_seconds()
        log(f"  Duration:    {session_duration:.1f} seconds")
    log("")

    # =================================================================
    # PART 1: AUCTION RESULTS (ALL BIDS)
    # =================================================================
    log("=" * 80)
    log("PART 1: AUCTION RESULTS (ALL BIDS, SORTED BY RANKING)")
    log("=" * 80)
    log("")
    log("The auction produced these rankings. Winners (IS_WINNER=True) get impression slots.")
    log("")

    if 'QUALITY' in auction_bids.columns and 'FINAL_BID' in auction_bids.columns:
        log(f"{'Rank':>4} | {'Win':>3} | {'Product Name':<52} | {'Price':>8} | {'Quality':>8} | {'Bid':>8}")
        log(f"{'-'*4}-+-{'-'*3}-+-{'-'*52}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        for _, row in auction_bids.iterrows():
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            winner = "Y" if row['IS_WINNER'] else "-"
            name = truncate_name(row['NAME'], 52)
            price = format_price(row.get(price_col)) if price_col else '--'
            quality = f"{row['QUALITY']:.4f}" if pd.notna(row.get('QUALITY')) else '--'
            bid = f"{row['FINAL_BID']:.2f}" if pd.notna(row.get('FINAL_BID')) else '--'

            log(f"{rank:>4} | {winner:>3} | {name:<52} | {price:>8} | {quality:>8} | {bid:>8}")
    else:
        log(f"{'Rank':>4} | {'Win':>3} | {'Product Name':<52} | {'Price':>8}")
        log(f"{'-'*4}-+-{'-'*3}-+-{'-'*52}-+-{'-'*8}")

        for _, row in auction_bids.iterrows():
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            winner = "Y" if row['IS_WINNER'] else "-"
            name = truncate_name(row['NAME'], 52)
            price = format_price(row.get(price_col)) if price_col else '--'

            log(f"{rank:>4} | {winner:>3} | {name:<52} | {price:>8}")

    log("")

    # =================================================================
    # PART 2: IMPRESSIONS (CHRONOLOGICAL - WHAT USER SAW)
    # =================================================================
    log("=" * 80)
    log("PART 2: IMPRESSIONS (CHRONOLOGICAL - WHAT USER ACTUALLY SAW)")
    log("=" * 80)
    log("")

    first_imp_time = None
    if len(auction_imps) > 0:
        first_imp_time = auction_imps['OCCURRED_AT'].min()
        log("User scrolled through these products (impressions logged as they enter viewport).")
        log("Same timestamp = batched (loaded together). Different timestamp = user scrolled.")
        log("")
        log(f"{'#':>3} | {'Time':>12} | {'+Δt':>8} | {'Rank':>4} | {'Product Name':<52} | {'Price':>8}")
        log(f"{'-'*3}-+-{'-'*12}-+-{'-'*8}-+-{'-'*4}-+-{'-'*52}-+-{'-'*8}")

        prev_ts = None
        for idx, (_, row) in enumerate(auction_imps.iterrows(), 1):
            ts = format_time(row['OCCURRED_AT'])
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            name = truncate_name(row['NAME'], 52)
            price = format_price(row.get(price_col)) if price_col else '--'
            delta_secs = (row['OCCURRED_AT'] - first_imp_time).total_seconds()
            delta = format_delta(delta_secs)

            # Mark batching
            note = ""
            if prev_ts is not None:
                if row['OCCURRED_AT'] == prev_ts:
                    note = "  <- same batch"
                elif delta_secs > 0:
                    note = "  <- scrolled"

            log(f"{idx:>3} | {ts:>12} | {delta:>8} | {rank:>4} | {name:<52} | {price:>8}{note}")
            prev_ts = row['OCCURRED_AT']
    else:
        log("No impressions recorded for this auction.")

    log("")

    # =================================================================
    # PART 3: CLICKS (CHRONOLOGICAL - WHAT USER CLICKED)
    # =================================================================
    log("=" * 80)
    log("PART 3: CLICKS (CHRONOLOGICAL - WHAT USER CLICKED)")
    log("=" * 80)
    log("")

    if len(auction_clicks) > 0:
        log("User clicked on these products after viewing them.")
        log("")
        log(f"{'#':>3} | {'Time':>12} | {'Since Imp':>10} | {'Rank':>4} | {'Product Name':<52} | {'Price':>8}")
        log(f"{'-'*3}-+-{'-'*12}-+-{'-'*10}-+-{'-'*4}-+-{'-'*52}-+-{'-'*8}")

        for idx, (_, row) in enumerate(auction_clicks.iterrows(), 1):
            ts = format_time(row['OCCURRED_AT'])
            rank = int(row['RANKING']) if pd.notna(row['RANKING']) else '--'
            name = truncate_name(row['NAME'], 52)
            price = format_price(row.get(price_col)) if price_col else '--'

            # Time since first impression
            if first_imp_time is not None:
                delta_secs = (row['OCCURRED_AT'] - first_imp_time).total_seconds()
                since_imp = format_delta(delta_secs)
            else:
                since_imp = "--"

            log(f"{idx:>3} | {ts:>12} | {since_imp:>10} | {rank:>4} | {name:<52} | {price:>8}")
    else:
        log("No clicks recorded for this auction.")

    log("")
    log("=" * 80)
    log("END OF SESSION LOG")
    log("=" * 80)

    return "\n".join(lines), placement


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate detailed session logs stored in folder')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze')
    parser.add_argument('--n', type=int, default=DEFAULT_N_AUCTIONS,
                        help=f'Number of auctions to log (default: {DEFAULT_N_AUCTIONS})')
    args = parser.parse_args()

    # Create output directory
    round_logs_dir = LOGS_DIR / args.round
    round_logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"GENERATING DETAILED SESSION LOGS ({args.round.upper()})")
    print("=" * 80)
    print(f"Output directory: {round_logs_dir}")
    print(f"Number of auctions: {args.n}")
    print("")

    paths = get_data_paths(args.round)

    # Load data
    print("Loading data...")
    ar = pd.read_parquet(paths['auctions_results'])
    print(f"  Auctions results: {len(ar):,} bids")

    au = pd.read_parquet(paths['auctions_users'])
    print(f"  Auctions users: {len(au):,} records")

    impressions = pd.read_parquet(paths['impressions'])
    print(f"  Impressions: {len(impressions):,}")

    clicks = pd.read_parquet(paths['clicks'])
    print(f"  Clicks: {len(clicks):,}")

    catalog = pd.read_parquet(paths['catalog']) if paths['catalog'].exists() else pd.DataFrame(columns=['PRODUCT_ID', 'NAME'])
    print(f"  Catalog: {len(catalog):,} products")
    print("")

    # Determine column names
    price_col = get_price_column(catalog)
    user_col = get_user_column(au)

    # Find auctions with clicks
    clicked_auctions = clicks['AUCTION_ID'].unique()
    print(f"Auctions with clicks: {len(clicked_auctions):,}")

    # Check for placement column
    has_placement = 'PLACEMENT' in au.columns

    # Sample auctions
    np.random.seed(SEED)

    if has_placement:
        # Stratified sampling by placement
        clicked_with_placement = au[au['AUCTION_ID'].isin(clicked_auctions)][['AUCTION_ID', 'PLACEMENT']].drop_duplicates()

        print("Clicked auctions by placement:")
        placement_counts = clicked_with_placement['PLACEMENT'].value_counts().sort_index()
        for p, count in placement_counts.items():
            print(f"  Placement {p}: {count:,}")
        print("")

        # Proportional sampling by placement
        sampled_auctions = []
        for placement in sorted(clicked_with_placement['PLACEMENT'].unique()):
            placement_auctions = clicked_with_placement[clicked_with_placement['PLACEMENT'] == placement]['AUCTION_ID'].values
            n_for_placement = max(1, int(args.n * len(placement_auctions) / len(clicked_with_placement)))
            n_sample = min(n_for_placement, len(placement_auctions))
            sampled = np.random.choice(placement_auctions, n_sample, replace=False)
            sampled_auctions.extend(sampled)

        # Trim or extend to exact count
        if len(sampled_auctions) > args.n:
            sampled_auctions = sampled_auctions[:args.n]
        elif len(sampled_auctions) < args.n:
            available = [a for a in clicked_auctions if a not in sampled_auctions]
            extra = np.random.choice(available, min(args.n - len(sampled_auctions), len(available)), replace=False)
            sampled_auctions.extend(extra)
    else:
        sampled_auctions = list(np.random.choice(clicked_auctions, min(args.n, len(clicked_auctions)), replace=False))

    print(f"Generating logs for {len(sampled_auctions)} auctions...")
    print("")

    # Generate individual logs
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"SESSION LOGS SUMMARY - {args.round.upper()}")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"Total auctions logged: {len(sampled_auctions)}")
    summary_lines.append(f"Directory: {round_logs_dir}")
    summary_lines.append("")
    summary_lines.append("-" * 80)
    summary_lines.append("FILES GENERATED:")
    summary_lines.append("-" * 80)
    summary_lines.append("")

    placement_counts = {}

    for i, auction_id in enumerate(tqdm(sampled_auctions, desc="Generating auction logs")):
        try:
            log_content, placement = generate_auction_log(
                auction_id, ar, au, impressions, clicks, catalog, price_col, user_col
            )

            # Create filename with placement info
            placement_name = get_placement_name(placement)
            # Truncate auction ID for filename
            short_id = str(auction_id)[:16]
            filename = f"auction_{i+1:03d}_{placement_name}_{short_id}.txt"

            filepath = round_logs_dir / filename
            with open(filepath, 'w') as f:
                f.write(log_content)

            summary_lines.append(f"  {filename}")

            # Track placement counts
            placement_counts[placement] = placement_counts.get(placement, 0) + 1

        except Exception as e:
            summary_lines.append(f"  ERROR: auction {auction_id[:16]}... - {e}")

    summary_lines.append("")
    summary_lines.append("-" * 80)
    summary_lines.append("BY PLACEMENT:")
    summary_lines.append("-" * 80)
    for p in sorted(placement_counts.keys()):
        pname = get_placement_name(p)
        summary_lines.append(f"  Placement {p} ({pname}): {placement_counts[p]} auctions")

    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("HOW TO READ THESE LOGS:")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("Each file shows ONE user session with a click:")
    summary_lines.append("")
    summary_lines.append("  PART 1 - AUCTION RESULTS:")
    summary_lines.append("    All products that bid in the auction, sorted by rank.")
    summary_lines.append("    'Win=Y' means the product won an impression slot.")
    summary_lines.append("")
    summary_lines.append("  PART 2 - IMPRESSIONS:")
    summary_lines.append("    Products the user actually SAW, in chronological order.")
    summary_lines.append("    Same timestamp = loaded together (initial viewport).")
    summary_lines.append("    Later timestamp = user scrolled to reveal more products.")
    summary_lines.append("")
    summary_lines.append("  PART 3 - CLICKS:")
    summary_lines.append("    Products the user clicked on.")
    summary_lines.append("    'Since Imp' shows time elapsed since first impression.")
    summary_lines.append("")
    summary_lines.append("=" * 80)

    # Write summary
    summary_path = round_logs_dir / "00_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))

    print("")
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Output directory: {round_logs_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Individual logs: {len(sampled_auctions)} files")
    print("")
    print("Files by placement:")
    for p in sorted(placement_counts.keys()):
        pname = get_placement_name(p)
        print(f"  {pname}: {placement_counts[p]}")


if __name__ == "__main__":
    main()
