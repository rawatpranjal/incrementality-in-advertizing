#!/usr/bin/env python3
"""
Bids, Winners, and Rankings per Auction

Bid volume, winner selection, ranking mechanics analysis.
Documents the auction structure and ranking formula behavior.

Usage:
    python 02_bids_winners_rankings_per_auction.py --round round1
    python 02_bids_winners_rankings_per_auction.py --round round2
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


def get_data_paths(round_name):
    """Return data paths for specified round."""
    if round_name == "round1":
        return {
            'auctions_results': DATA_BASE / "round1/auctions_results_all.parquet",
            'auctions_users': DATA_BASE / "round1/auctions_users_all.parquet",
            'impressions': DATA_BASE / "round1/impressions_all.parquet",
            'clicks': DATA_BASE / "round1/clicks_all.parquet",
        }
    elif round_name == "round2":
        return {
            'auctions_results': DATA_BASE / "round2/auctions_results_r2.parquet",
            'auctions_users': DATA_BASE / "round2/auctions_users_r2.parquet",
            'impressions': DATA_BASE / "round2/impressions_r2.parquet",
            'clicks': DATA_BASE / "round2/clicks_r2.parquet",
        }
    else:
        raise ValueError(f"Unknown round: {round_name}")


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def bids_per_auction_analysis(ar, f):
    """Analyze bid distribution per auction."""
    log(f"\n{'='*80}", f)
    log(f"BIDS PER AUCTION", f)
    log(f"{'='*80}", f)

    bids_per_auction = ar.groupby('AUCTION_ID').size()

    log(f"\nTotal auctions: {len(bids_per_auction):,}", f)
    log(f"Total bids: {len(ar):,}", f)

    log(f"\n--- Distribution Statistics ---", f)
    log(f"Mean: {bids_per_auction.mean():.2f}", f)
    log(f"Median: {bids_per_auction.median():.0f}", f)
    log(f"Std: {bids_per_auction.std():.2f}", f)
    log(f"Min: {bids_per_auction.min()}", f)
    log(f"Max: {bids_per_auction.max()}", f)

    log(f"\n--- Percentiles ---", f)
    for p in [25, 50, 75, 90, 95, 99]:
        val = bids_per_auction.quantile(p/100)
        log(f"P{p}: {val:.0f}", f)

    log(f"\n--- Distribution of Bid Counts ---", f)
    log(f"{'N Bids':>10} {'Auctions':>15} {'Percentage':>15} {'Cumulative':>15}", f)
    log(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15}", f)

    bid_dist = bids_per_auction.value_counts().sort_index()
    cumsum = 0
    for n_bids in sorted(bid_dist.index[:20]):
        count = bid_dist[n_bids]
        pct = count / len(bids_per_auction) * 100
        cumsum += pct
        log(f"{n_bids:>10} {count:>15,} {pct:>14.1f}% {cumsum:>14.1f}%", f)


def winners_per_auction_analysis(ar, f):
    """Analyze winner distribution per auction."""
    log(f"\n{'='*80}", f)
    log(f"WINNERS PER AUCTION", f)
    log(f"{'='*80}", f)

    winners_per_auction = ar[ar['IS_WINNER'] == True].groupby('AUCTION_ID').size()
    total_auctions = ar['AUCTION_ID'].nunique()

    auctions_with_winners = len(winners_per_auction)
    auctions_without_winners = total_auctions - auctions_with_winners

    log(f"\nAuctions with at least one winner: {auctions_with_winners:,} ({auctions_with_winners/total_auctions*100:.1f}%)", f)
    log(f"Auctions with no winners: {auctions_without_winners:,} ({auctions_without_winners/total_auctions*100:.1f}%)", f)

    log(f"\n--- Distribution Statistics (auctions with winners) ---", f)
    log(f"Mean: {winners_per_auction.mean():.2f}", f)
    log(f"Median: {winners_per_auction.median():.0f}", f)
    log(f"Std: {winners_per_auction.std():.2f}", f)
    log(f"Min: {winners_per_auction.min()}", f)
    log(f"Max: {winners_per_auction.max()}", f)

    log(f"\n--- Distribution of Winner Counts ---", f)
    log(f"{'N Winners':>12} {'Auctions':>15} {'Percentage':>15}", f)
    log(f"{'-'*12} {'-'*15} {'-'*15}", f)

    winner_dist = winners_per_auction.value_counts().sort_index()
    for n_winners in sorted(winner_dist.index[:20]):
        count = winner_dist[n_winners]
        pct = count / len(winners_per_auction) * 100
        log(f"{n_winners:>12} {count:>15,} {pct:>14.1f}%", f)


def ranking_distribution_analysis(ar, f):
    """Analyze RANKING field distribution."""
    log(f"\n{'='*80}", f)
    log(f"RANKING DISTRIBUTION", f)
    log(f"{'='*80}", f)

    log(f"\n--- Overall RANKING Statistics ---", f)
    ranking_stats = ar['RANKING'].describe()
    for stat, val in ranking_stats.items():
        log(f"{stat}: {val:.2f}", f)

    log(f"\n--- RANKING by Winner Status ---", f)

    for is_winner in [True, False]:
        subset = ar[ar['IS_WINNER'] == is_winner]['RANKING']
        label = "Winners" if is_winner else "Losers"
        log(f"\n{label} (N={len(subset):,}):", f)
        log(f"  Mean: {subset.mean():.2f}", f)
        log(f"  Median: {subset.median():.0f}", f)
        log(f"  Min: {subset.min()}", f)
        log(f"  Max: {subset.max()}", f)

    log(f"\n--- RANKING Distribution (Winners Only) ---", f)
    log(f"{'Rank':>8} {'Count':>15} {'Percentage':>15} {'Cumulative':>15}", f)
    log(f"{'-'*8} {'-'*15} {'-'*15} {'-'*15}", f)

    winner_ranks = ar[ar['IS_WINNER'] == True]['RANKING']
    rank_dist = winner_ranks.value_counts().sort_index()
    cumsum = 0
    for rank in sorted(rank_dist.index[:25]):
        count = rank_dist[rank]
        pct = count / len(winner_ranks) * 100
        cumsum += pct
        log(f"{int(rank):>8} {count:>15,} {pct:>14.1f}% {cumsum:>14.1f}%", f)


def ranking_formula_test(ar, f):
    """Test if RANKING = f(QUALITY, FINAL_BID, PACING)."""
    log(f"\n{'='*80}", f)
    log(f"RANKING FORMULA ANALYSIS", f)
    log(f"{'='*80}", f)

    ar_complete = ar[['AUCTION_ID', 'PRODUCT_ID', 'RANKING', 'IS_WINNER',
                      'FINAL_BID', 'QUALITY', 'PACING']].dropna(
        subset=['FINAL_BID', 'QUALITY', 'PACING', 'RANKING']
    )
    log(f"\nRows with complete scoring features: {len(ar_complete):,} / {len(ar):,} ({len(ar_complete)/len(ar)*100:.1f}%)", f)

    if len(ar_complete) < 1000:
        log("Insufficient data for ranking formula analysis", f)
        return

    ar_complete['score_v1'] = ar_complete['QUALITY'] * ar_complete['FINAL_BID']
    ar_complete['score_v2'] = ar_complete['QUALITY'] * ar_complete['FINAL_BID'] * ar_complete['PACING']

    log(f"\n--- Within-Auction Rank Prediction Test ---", f)
    log("Testing: If we sort by score descending, do we recover RANKING?", f)

    def compute_predicted_rank(group):
        group = group.copy()
        group['predicted_rank_v1'] = group['score_v1'].rank(ascending=False, method='min')
        group['predicted_rank_v2'] = group['score_v2'].rank(ascending=False, method='min')
        return group

    ar_with_pred = ar_complete.groupby('AUCTION_ID', group_keys=False).apply(compute_predicted_rank)

    exact_match_v1 = (ar_with_pred['RANKING'] == ar_with_pred['predicted_rank_v1']).mean()
    exact_match_v2 = (ar_with_pred['RANKING'] == ar_with_pred['predicted_rank_v2']).mean()

    mae_v1 = (ar_with_pred['RANKING'] - ar_with_pred['predicted_rank_v1']).abs().mean()
    mae_v2 = (ar_with_pred['RANKING'] - ar_with_pred['predicted_rank_v2']).abs().mean()

    log(f"\nUsing score = QUALITY x FINAL_BID:", f)
    log(f"  Exact rank match rate: {exact_match_v1*100:.2f}%", f)
    log(f"  Mean absolute error: {mae_v1:.2f} ranks", f)

    log(f"\nUsing score = QUALITY x FINAL_BID x PACING:", f)
    log(f"  Exact rank match rate: {exact_match_v2*100:.2f}%", f)
    log(f"  Mean absolute error: {mae_v2:.2f} ranks", f)

    log(f"\n--- Rank-1 Max Score Test ---", f)
    log("Testing: Does rank-1 always have the highest score in its auction?", f)

    auction_max_v1 = ar_with_pred.groupby('AUCTION_ID')['score_v1'].max().reset_index()
    auction_max_v1.columns = ['AUCTION_ID', 'max_score_v1']

    rank1 = ar_with_pred[ar_with_pred['RANKING'] == 1].copy()
    rank1 = rank1.merge(auction_max_v1, on='AUCTION_ID', how='left')

    rank1_has_max_v1 = (rank1['score_v1'] == rank1['max_score_v1']).mean()
    log(f"\nRank-1 has max score (QUALITY x BID): {rank1_has_max_v1*100:.2f}% of auctions", f)


def impressions_per_auction_analysis(ar, imp, f):
    """Analyze impressions per auction."""
    log(f"\n{'='*80}", f)
    log(f"IMPRESSIONS PER AUCTION", f)
    log(f"{'='*80}", f)

    if imp is None or len(imp) == 0:
        log("No impression data available", f)
        return

    imp_per_auction = imp.groupby('AUCTION_ID').size()
    total_auctions = ar['AUCTION_ID'].nunique()

    auctions_with_imp = len(imp_per_auction)
    auctions_without_imp = total_auctions - auctions_with_imp

    log(f"\nAuctions with impressions: {auctions_with_imp:,} ({auctions_with_imp/total_auctions*100:.1f}%)", f)
    log(f"Auctions without impressions: {auctions_without_imp:,} ({auctions_without_imp/total_auctions*100:.1f}%)", f)

    log(f"\n--- Distribution Statistics (auctions with impressions) ---", f)
    log(f"Mean: {imp_per_auction.mean():.2f}", f)
    log(f"Median: {imp_per_auction.median():.0f}", f)
    log(f"Std: {imp_per_auction.std():.2f}", f)
    log(f"Min: {imp_per_auction.min()}", f)
    log(f"Max: {imp_per_auction.max()}", f)

    log(f"\n--- Distribution of Impression Counts ---", f)
    log(f"{'N Imps':>10} {'Auctions':>15} {'Percentage':>15}", f)
    log(f"{'-'*10} {'-'*15} {'-'*15}", f)

    imp_dist = imp_per_auction.value_counts().sort_index()
    for n_imps in sorted(imp_dist.index[:15]):
        count = imp_dist[n_imps]
        pct = count / len(imp_per_auction) * 100
        log(f"{n_imps:>10} {count:>15,} {pct:>14.1f}%", f)


def winners_vs_impressions(ar, imp, f):
    """Compare winners to impressions per auction."""
    log(f"\n{'='*80}", f)
    log(f"WINNERS vs IMPRESSIONS COMPARISON", f)
    log(f"{'='*80}", f)

    if imp is None or len(imp) == 0:
        log("No impression data available", f)
        return

    winners_per_auction = ar[ar['IS_WINNER'] == True].groupby('AUCTION_ID').size()
    imp_per_auction = imp.groupby('AUCTION_ID').size()

    comparison = pd.DataFrame({
        'winners': winners_per_auction,
        'impressions': imp_per_auction
    }).dropna()

    if len(comparison) == 0:
        log("No overlapping auctions between winners and impressions", f)
        return

    log(f"\nAuctions with both winners and impressions: {len(comparison):,}", f)
    log(f"Correlation (winners, impressions): {comparison['winners'].corr(comparison['impressions']):.4f}", f)

    log(f"\n--- Comparison ---", f)
    log(f"Mean winners per auction: {comparison['winners'].mean():.2f}", f)
    log(f"Mean impressions per auction: {comparison['impressions'].mean():.2f}", f)
    log(f"Ratio (impressions/winners): {comparison['impressions'].sum() / comparison['winners'].sum():.3f}", f)

    log(f"\n--- Impression Rate by Winner Count ---", f)
    log(f"{'Winners':>10} {'Auctions':>12} {'Mean Imps':>12} {'Imp/Winner':>12}", f)
    log(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12}", f)

    for n_winners in sorted(comparison['winners'].unique())[:15]:
        subset = comparison[comparison['winners'] == n_winners]
        n_auctions = len(subset)
        mean_imps = subset['impressions'].mean()
        imp_per_winner = mean_imps / n_winners if n_winners > 0 else 0
        log(f"{int(n_winners):>10} {n_auctions:>12,} {mean_imps:>12.1f} {imp_per_winner:>12.2f}", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Bids, winners, rankings per auction EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"02_bids_winners_rankings_per_auction_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("BIDS, WINNERS, AND RANKINGS PER AUCTION", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)

        paths = get_data_paths(args.round)

        log(f"\n--- Loading Data ---", f)
        ar = pd.read_parquet(paths['auctions_results'])
        log(f"Auctions results: {len(ar):,} rows", f)

        imp = None
        if paths['impressions'].exists():
            imp = pd.read_parquet(paths['impressions'])
            log(f"Impressions: {len(imp):,} rows", f)

        bids_per_auction_analysis(ar, f)
        winners_per_auction_analysis(ar, f)
        ranking_distribution_analysis(ar, f)
        ranking_formula_test(ar, f)
        impressions_per_auction_analysis(ar, imp, f)
        winners_vs_impressions(ar, imp, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
