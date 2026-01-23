#!/usr/bin/env python3
"""
07_auction_panel.py
Builds auction-vendor level panel with marginal win/loss indicators for IV analysis.
For each (auction, vendor), identifies whether vendor's best bid was at the margin (rank=K)
or just below (rank=K+1), creating quasi-experimental variation in impression eligibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/Users/pranjal/Code/marketplace-incrementality/eda/data")
OUTPUT_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "07_auction_panel.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("07_AUCTION_PANEL", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script constructs the auction-vendor level panel needed for IV analysis.", f)
        log("For each (auction, vendor) pair, we identify whether the vendor's best bid was", f)
        log("at the margin (rank = K, where K is the last winning position) or just below", f)
        log("(rank = K+1, marginal loser). This creates quasi-experimental variation in", f)
        log("impression eligibility: vendors at rank K got an impression slot, vendors at", f)
        log("rank K+1 did not. Conditional on being near the margin, which side you fall on", f)
        log("is quasi-random (determined by small bid/quality differences). This provides", f)
        log("an instrument for clicks that satisfies the exclusion restriction.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD DATA
        # ============================================================
        log("LOADING DATA", f)
        log("-" * 40, f)

        log("\nLoading auctions_results (bids)...", f)
        bids = pd.read_parquet(DATA_DIR / 'auctions_results_365d.parquet')
        log(f"  Rows: {len(bids):,}", f)
        log(f"  Columns: {list(bids.columns)}", f)

        log("\nLoading auctions_users...", f)
        auctions = pd.read_parquet(DATA_DIR / 'auctions_users_365d.parquet')
        log(f"  Rows: {len(auctions):,}", f)

        log("\nLoading promoted_events (for clicked indicator)...", f)
        promoted_events = pd.read_parquet(OUTPUT_DIR / 'promoted_events.parquet')
        log(f"  Rows: {len(promoted_events):,}", f)

        # ============================================================
        # 2. CALCULATE K PER AUCTION
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("CALCULATING K (NUMBER OF WINNER SLOTS) PER AUCTION", f)
        log("=" * 80, f)

        log("\nK = max(RANKING) where IS_WINNER = True for each auction", f)

        # Get max winning rank per auction
        winners = bids[bids['IS_WINNER'] == True].copy()
        k_per_auction = winners.groupby('AUCTION_ID')['RANKING'].max().reset_index()
        k_per_auction.columns = ['AUCTION_ID', 'K']

        log(f"\nAuctions with winners: {len(k_per_auction):,}", f)

        log("\n--- K Distribution ---", f)
        log(str(k_per_auction['K'].describe()), f)

        log("\nK value counts (top 10):", f)
        k_counts = k_per_auction['K'].value_counts().sort_index()
        for k_val in k_counts.index[:10]:
            log(f"  K={k_val}: {k_counts[k_val]:,} auctions", f)

        # ============================================================
        # 3. BUILD AUCTION-VENDOR PANEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING AUCTION-VENDOR PANEL", f)
        log("=" * 80, f)

        # Aggregate bids to (auction, vendor) level - take best rank
        log("\nAggregating bids to (auction, vendor) level...", f)

        auction_vendor = bids.groupby(['AUCTION_ID', 'VENDOR_ID']).agg({
            'RANKING': 'min',  # Best rank for this vendor in auction
            'IS_WINNER': 'max',  # Did vendor win at least one slot?
            'FINAL_BID': 'mean',  # Average bid amount
            'QUALITY': 'mean',  # Average quality score
            'PACING': 'mean',  # Average pacing
            'PRODUCT_ID': 'count'  # Number of products vendor had in auction
        }).reset_index()

        auction_vendor.columns = ['auction_id', 'vendor_id', 'best_rank', 'any_winner',
                                  'avg_bid', 'avg_quality', 'avg_pacing', 'n_products']

        log(f"  Unique (auction, vendor) pairs: {len(auction_vendor):,}", f)
        log(f"  Unique auctions: {auction_vendor['auction_id'].nunique():,}", f)
        log(f"  Unique vendors: {auction_vendor['vendor_id'].nunique():,}", f)

        # Merge K values
        log("\nMerging K values...", f)
        auction_vendor = auction_vendor.merge(
            k_per_auction,
            left_on='auction_id',
            right_on='AUCTION_ID',
            how='left'
        ).drop(columns=['AUCTION_ID'])

        # Auctions with no winners (K is NA) - should be 0
        no_k = auction_vendor['K'].isna().sum()
        log(f"  Auction-vendor pairs with missing K: {no_k:,}", f)
        if no_k > 0:
            auction_vendor = auction_vendor[auction_vendor['K'].notna()].copy()
            log(f"  After dropping: {len(auction_vendor):,}", f)

        # Create marginal indicators
        log("\nCreating marginal indicators...", f)

        auction_vendor['K'] = auction_vendor['K'].astype(int)
        auction_vendor['eligible'] = (auction_vendor['best_rank'] <= auction_vendor['K']).astype(int)
        auction_vendor['marginal_win'] = (auction_vendor['best_rank'] == auction_vendor['K']).astype(int)
        auction_vendor['marginal_loss'] = (auction_vendor['best_rank'] == auction_vendor['K'] + 1).astype(int)
        auction_vendor['close'] = auction_vendor['marginal_win'] + auction_vendor['marginal_loss']

        log("\n--- Marginal Status Distribution ---", f)
        log(f"  Eligible (best_rank <= K): {auction_vendor['eligible'].sum():,} ({auction_vendor['eligible'].mean()*100:.1f}%)", f)
        log(f"  Marginal winners (rank = K): {auction_vendor['marginal_win'].sum():,} ({auction_vendor['marginal_win'].mean()*100:.1f}%)", f)
        log(f"  Marginal losers (rank = K+1): {auction_vendor['marginal_loss'].sum():,} ({auction_vendor['marginal_loss'].mean()*100:.1f}%)", f)
        log(f"  Close (K or K+1): {auction_vendor['close'].sum():,} ({auction_vendor['close'].mean()*100:.1f}%)", f)

        # ============================================================
        # 4. ADD USER ID FROM AUCTIONS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("ADDING USER ID FROM AUCTIONS", f)
        log("=" * 80, f)

        auctions_users = auctions[['AUCTION_ID', 'OPAQUE_USER_ID', 'CREATED_AT']].copy()
        auctions_users.columns = ['auction_id', 'user_id', 'auction_time']
        auctions_users = auctions_users.drop_duplicates(subset=['auction_id'])

        auction_vendor = auction_vendor.merge(
            auctions_users,
            on='auction_id',
            how='left'
        )

        user_match = auction_vendor['user_id'].notna().mean() * 100
        log(f"  User ID match rate: {user_match:.1f}%", f)
        log(f"  Unique users: {auction_vendor['user_id'].nunique():,}", f)

        # ============================================================
        # 5. ADD CLICKED INDICATOR
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("ADDING CLICKED INDICATOR", f)
        log("=" * 80, f)

        # Get unique (auction, vendor) pairs that were clicked
        clicked_pairs = promoted_events[['auction_id', 'vendor_id']].drop_duplicates()
        clicked_pairs['clicked'] = 1

        auction_vendor = auction_vendor.merge(
            clicked_pairs,
            on=['auction_id', 'vendor_id'],
            how='left'
        )
        auction_vendor['clicked'] = auction_vendor['clicked'].fillna(0).astype(int)

        log(f"  Auction-vendor pairs with click: {auction_vendor['clicked'].sum():,} ({auction_vendor['clicked'].mean()*100:.3f}%)", f)

        # Cross-tab: eligible vs clicked
        log("\n--- Eligibility vs Click Cross-Tab ---", f)
        crosstab = pd.crosstab(auction_vendor['eligible'], auction_vendor['clicked'],
                               margins=True, margins_name='Total')
        log(str(crosstab), f)

        # Click rate by eligibility
        log("\n--- Click Rate by Eligibility ---", f)
        for elig in [0, 1]:
            subset = auction_vendor[auction_vendor['eligible'] == elig]
            click_rate = subset['clicked'].mean() * 100
            n = len(subset)
            log(f"  Eligible={elig}: {click_rate:.4f}% click rate (n={n:,})", f)

        # ============================================================
        # 6. VALIDATE DISCONTINUITY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("VALIDATING SHARP DISCONTINUITY AT K", f)
        log("=" * 80, f)

        log("\nVerifying IS_WINNER pattern at margin:", f)

        # Get raw bids at K and K+1 positions
        bids_at_margin = bids.merge(k_per_auction, on='AUCTION_ID', how='inner')
        bids_at_margin['at_K'] = bids_at_margin['RANKING'] == bids_at_margin['K']
        bids_at_margin['at_K_plus_1'] = bids_at_margin['RANKING'] == bids_at_margin['K'] + 1

        at_k = bids_at_margin[bids_at_margin['at_K']]
        at_k_plus_1 = bids_at_margin[bids_at_margin['at_K_plus_1']]

        log(f"\nBids at rank = K:", f)
        log(f"  Count: {len(at_k):,}", f)
        log(f"  IS_WINNER = True: {at_k['IS_WINNER'].sum():,} ({at_k['IS_WINNER'].mean()*100:.2f}%)", f)

        log(f"\nBids at rank = K+1:", f)
        log(f"  Count: {len(at_k_plus_1):,}", f)
        log(f"  IS_WINNER = True: {at_k_plus_1['IS_WINNER'].sum():,} ({at_k_plus_1['IS_WINNER'].mean()*100:.2f}%)", f)

        if at_k['IS_WINNER'].mean() > 0.99 and at_k_plus_1['IS_WINNER'].mean() < 0.01:
            log("\n  VERIFIED: Sharp discontinuity at K (100% win at K, 0% at K+1)", f)
        else:
            log("\n  WARNING: Discontinuity not perfectly sharp", f)

        # ============================================================
        # 7. BALANCE CHECK AT MARGIN
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BALANCE CHECK: MARGINAL WINNERS VS MARGINAL LOSERS", f)
        log("=" * 80, f)

        marginal_winners = auction_vendor[auction_vendor['marginal_win'] == 1]
        marginal_losers = auction_vendor[auction_vendor['marginal_loss'] == 1]

        log(f"\nN marginal winners: {len(marginal_winners):,}", f)
        log(f"N marginal losers: {len(marginal_losers):,}", f)

        balance_vars = ['avg_bid', 'avg_quality', 'avg_pacing', 'n_products']
        log("\n--- Variable Means by Marginal Status ---", f)
        log(f"{'Variable':<15} {'Winners':>12} {'Losers':>12} {'Diff':>12}", f)
        log("-" * 55, f)

        for var in balance_vars:
            mean_w = marginal_winners[var].mean()
            mean_l = marginal_losers[var].mean()
            diff = mean_w - mean_l
            log(f"{var:<15} {mean_w:>12.4f} {mean_l:>12.4f} {diff:>12.4f}", f)

        # ============================================================
        # 8. ADD TEMPORAL FEATURES
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("ADDING TEMPORAL FEATURES", f)
        log("=" * 80, f)

        auction_vendor['auction_time'] = pd.to_datetime(auction_vendor['auction_time'])
        auction_vendor['week'] = auction_vendor['auction_time'].dt.isocalendar().week
        auction_vendor['year'] = auction_vendor['auction_time'].dt.year
        auction_vendor['year_week'] = auction_vendor['year'].astype(str) + '_W' + auction_vendor['week'].astype(str).str.zfill(2)

        log(f"  Date range: {auction_vendor['auction_time'].min()} to {auction_vendor['auction_time'].max()}", f)
        log(f"  Weeks: {auction_vendor['year_week'].nunique()}", f)

        # ============================================================
        # 9. SAVE OUTPUT
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("SAVING OUTPUT", f)
        log("=" * 80, f)

        # Select final columns
        output_cols = [
            'auction_id', 'vendor_id', 'user_id',
            'auction_time', 'year_week',
            'K', 'best_rank',
            'eligible', 'marginal_win', 'marginal_loss', 'close',
            'any_winner', 'clicked',
            'avg_bid', 'avg_quality', 'avg_pacing', 'n_products'
        ]
        auction_panel = auction_vendor[output_cols].copy()

        log(f"\nFinal panel shape: {auction_panel.shape}", f)
        log(f"Columns: {list(auction_panel.columns)}", f)

        auction_panel.to_parquet(OUTPUT_DIR / 'auction_panel.parquet', index=False)
        log(f"\nSaved to {OUTPUT_DIR / 'auction_panel.parquet'}", f)

        # ============================================================
        # SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("AUCTION PANEL SUMMARY", f)
        log("=" * 80, f)

        log(f"\nDimensions:", f)
        log(f"  Rows (auction-vendor pairs): {len(auction_panel):,}", f)
        log(f"  Unique auctions: {auction_panel['auction_id'].nunique():,}", f)
        log(f"  Unique vendors: {auction_panel['vendor_id'].nunique():,}", f)
        log(f"  Unique users: {auction_panel['user_id'].nunique():,}", f)

        log(f"\nKey variables:", f)
        log(f"  Eligible: {auction_panel['eligible'].sum():,} ({auction_panel['eligible'].mean()*100:.1f}%)", f)
        log(f"  Marginal win (rank=K): {auction_panel['marginal_win'].sum():,} ({auction_panel['marginal_win'].mean()*100:.2f}%)", f)
        log(f"  Marginal loss (rank=K+1): {auction_panel['marginal_loss'].sum():,} ({auction_panel['marginal_loss'].mean()*100:.2f}%)", f)
        log(f"  Clicked: {auction_panel['clicked'].sum():,} ({auction_panel['clicked'].mean()*100:.4f}%)", f)

        log(f"\nFor IV analysis:", f)
        log(f"  Close auctions (K or K+1): {auction_panel['close'].sum():,}", f)
        log(f"  These provide quasi-experimental variation in eligibility", f)

        # File size
        size_mb = (OUTPUT_DIR / 'auction_panel.parquet').stat().st_size / 1e6
        log(f"\nFile size: {size_mb:.1f} MB", f)

        log("", f)
        log("=" * 80, f)
        log("07_AUCTION_PANEL COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
