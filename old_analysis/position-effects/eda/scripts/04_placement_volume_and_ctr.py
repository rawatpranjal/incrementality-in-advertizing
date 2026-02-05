#!/usr/bin/env python3
"""
Placement Volume and CTR

Placement-level volume, CTR, session structure.
Documents characteristics of each placement type.

Usage:
    python 04_placement_volume_and_ctr.py --round round1
    python 04_placement_volume_and_ctr.py --round round2
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
def placement_overview(au, ar, imp, clicks, f):
    """Volume, CTR, and impression delivery by placement."""
    log(f"\n{'='*80}", f)
    log(f"PLACEMENT VOLUME OVERVIEW", f)
    log(f"{'='*80}", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return None, None, None

    ar_with_placement = ar.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')

    imp_with_placement = None
    if imp is not None:
        imp_with_placement = imp.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')

    clicks_with_placement = None
    if clicks is not None:
        clicks_with_placement = clicks.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')

    placements = sorted(au['PLACEMENT'].dropna().unique())

    log(f"\n--- Volume by Placement ---", f)
    log(f"{'Placement':<12} {'Auctions':>12} {'Bids':>12} {'Winners':>12} {'Impressions':>12} {'Clicks':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}", f)

    summary_data = []
    for p in placements:
        n_auctions = (au['PLACEMENT'] == p).sum()
        n_bids = (ar_with_placement['PLACEMENT'] == p).sum()
        n_winners = ((ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)).sum()
        n_imp = (imp_with_placement['PLACEMENT'] == p).sum() if imp_with_placement is not None else 0
        n_clicks = (clicks_with_placement['PLACEMENT'] == p).sum() if clicks_with_placement is not None else 0

        log(f"{str(p):<12} {n_auctions:>12,} {n_bids:>12,} {n_winners:>12,} {n_imp:>12,} {n_clicks:>10,}", f)
        summary_data.append({
            'placement': p,
            'auctions': n_auctions,
            'bids': n_bids,
            'winners': n_winners,
            'impressions': n_imp,
            'clicks': n_clicks
        })

    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}", f)
    log(f"{'TOTAL':<12} {len(au):>12,} {len(ar):>12,} {ar['IS_WINNER'].sum():>12,} {len(imp) if imp is not None else 0:>12,} {len(clicks) if clicks is not None else 0:>10,}", f)

    log(f"\n--- CTR and Delivery Rates by Placement ---", f)
    log(f"{'Placement':<12} {'CTR':>12} {'Imp/Auction':>14} {'Imp/Winner':>12} {'Auction w/ Imp%':>16}", f)
    log(f"{'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*16}", f)

    for row in summary_data:
        p = row['placement']
        ctr = (row['clicks'] / row['impressions'] * 100) if row['impressions'] > 0 else 0
        imp_per_auction = row['impressions'] / row['auctions'] if row['auctions'] > 0 else 0
        imp_per_winner = row['impressions'] / row['winners'] if row['winners'] > 0 else 0

        auctions_this_placement = au[au['PLACEMENT'] == p]['AUCTION_ID'].unique()
        if imp_with_placement is not None:
            auctions_with_imp = imp_with_placement[imp_with_placement['PLACEMENT'] == p]['AUCTION_ID'].nunique()
        else:
            auctions_with_imp = 0
        pct_with_imp = (auctions_with_imp / len(auctions_this_placement) * 100) if len(auctions_this_placement) > 0 else 0

        log(f"{str(p):<12} {ctr:>11.2f}% {imp_per_auction:>14.2f} {imp_per_winner:>12.3f} {pct_with_imp:>15.1f}%", f)

    return ar_with_placement, imp_with_placement, clicks_with_placement


def session_structure_by_placement(au, ar_with_placement, imp_with_placement, f):
    """Bids per auction, winners per auction, impressions per auction by placement."""
    log(f"\n{'='*80}", f)
    log(f"SESSION STRUCTURE BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None:
        return

    placements = sorted(au['PLACEMENT'].dropna().unique())

    log(f"\n--- Bids per Auction ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8} {'Std':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}", f)

    for p in placements:
        bids_per_auction = ar_with_placement[ar_with_placement['PLACEMENT'] == p].groupby('AUCTION_ID').size()
        if len(bids_per_auction) > 0:
            log(f"{str(p):<12} {bids_per_auction.mean():>10.1f} {bids_per_auction.median():>10.0f} {bids_per_auction.min():>8} {bids_per_auction.max():>8} {bids_per_auction.std():>10.1f}", f)

    log(f"\n--- Winners per Auction ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8} {'Std':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}", f)

    for p in placements:
        winners_per_auction = ar_with_placement[
            (ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)
        ].groupby('AUCTION_ID').size()
        if len(winners_per_auction) > 0:
            log(f"{str(p):<12} {winners_per_auction.mean():>10.1f} {winners_per_auction.median():>10.0f} {winners_per_auction.min():>8} {winners_per_auction.max():>8} {winners_per_auction.std():>10.1f}", f)

    if imp_with_placement is not None:
        log(f"\n--- Impressions per Auction (for auctions with any impression) ---", f)
        log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8} {'Std':>10}", f)
        log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}", f)

        for p in placements:
            imp_per_auction = imp_with_placement[imp_with_placement['PLACEMENT'] == p].groupby('AUCTION_ID').size()
            if len(imp_per_auction) > 0:
                log(f"{str(p):<12} {imp_per_auction.mean():>10.1f} {imp_per_auction.median():>10.0f} {imp_per_auction.min():>8} {imp_per_auction.max():>8} {imp_per_auction.std():>10.1f}", f)

        log(f"\n--- Distribution of Items Shown per Auction ---", f)
        log(f"{'Placement':<12} {'1 item':>10} {'2 items':>10} {'3 items':>10} {'4 items':>10} {'5+ items':>10}", f)
        log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", f)

        for p in placements:
            imp_counts = imp_with_placement[imp_with_placement['PLACEMENT'] == p].groupby('AUCTION_ID').size()
            total = len(imp_counts)
            if total > 0:
                pct_1 = (imp_counts == 1).sum() / total * 100
                pct_2 = (imp_counts == 2).sum() / total * 100
                pct_3 = (imp_counts == 3).sum() / total * 100
                pct_4 = (imp_counts == 4).sum() / total * 100
                pct_5plus = (imp_counts >= 5).sum() / total * 100
                log(f"{str(p):<12} {pct_1:>9.1f}% {pct_2:>9.1f}% {pct_3:>9.1f}% {pct_4:>9.1f}% {pct_5plus:>9.1f}%", f)


def ctr_by_rank_per_placement(ar_with_placement, imp_with_placement, clicks_with_placement, f):
    """CTR by rank for each placement."""
    log(f"\n{'='*80}", f)
    log(f"CTR BY RANK WITHIN PLACEMENT", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None or imp_with_placement is None or clicks_with_placement is None:
        log("Required data not available", f)
        return

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    clicks_with_rank = clicks_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    imp_with_rank = imp_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    for p in placements:
        log(f"\n--- Placement {p} ---", f)
        log(f"{'Rank':>8} {'Impressions':>15} {'Clicks':>10} {'CTR':>10}", f)
        log(f"{'-'*8} {'-'*15} {'-'*10} {'-'*10}", f)

        imp_p = imp_with_rank[imp_with_rank['PLACEMENT'] == p]
        clicks_p = clicks_with_rank[clicks_with_rank['PLACEMENT'] == p]

        all_ranks = sorted(imp_p['RANKING'].dropna().unique())

        for rank in all_ranks[:10]:
            n_imp = (imp_p['RANKING'] == rank).sum()
            n_clicks = (clicks_p['RANKING'] == rank).sum()
            ctr = n_clicks / n_imp * 100 if n_imp > 0 else 0
            log(f"{int(rank):>8} {n_imp:>15,} {n_clicks:>10,} {ctr:>9.2f}%", f)


def rank_distribution_by_placement(ar_with_placement, imp_with_placement, f):
    """What ranks get impressions by placement."""
    log(f"\n{'='*80}", f)
    log(f"RANK DISTRIBUTION OF IMPRESSIONS BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None or imp_with_placement is None:
        log("Required data not available", f)
        return

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    imp_with_rank = imp_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    log(f"\n--- Rank Distribution of Impressions ---", f)
    log(f"{'Placement':<12} {'Rank 1':>10} {'Rank 2':>10} {'Rank 3':>10} {'Rank 4-10':>12} {'Rank 11+':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}", f)

    for p in placements:
        ranks = imp_with_rank[imp_with_rank['PLACEMENT'] == p]['RANKING']
        total = len(ranks)
        if total > 0:
            pct_1 = (ranks == 1).sum() / total * 100
            pct_2 = (ranks == 2).sum() / total * 100
            pct_3 = (ranks == 3).sum() / total * 100
            pct_4_10 = ((ranks >= 4) & (ranks <= 10)).sum() / total * 100
            pct_11plus = (ranks > 10).sum() / total * 100
            log(f"{str(p):<12} {pct_1:>9.1f}% {pct_2:>9.1f}% {pct_3:>9.1f}% {pct_4_10:>11.1f}% {pct_11plus:>9.1f}%", f)

    log(f"\n--- Maximum Rank Shown (impression) by Placement ---", f)
    log(f"{'Placement':<12} {'Max Rank':>15}", f)
    log(f"{'-'*12} {'-'*15}", f)

    for p in placements:
        ranks = imp_with_rank[imp_with_rank['PLACEMENT'] == p]['RANKING']
        if len(ranks) > 0:
            log(f"{str(p):<12} {int(ranks.max()):>15}", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Placement volume and CTR EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"04_placement_volume_and_ctr_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("PLACEMENT VOLUME AND CTR", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)

        paths = get_data_paths(args.round)

        log(f"\n--- Loading Data ---", f)
        ar = pd.read_parquet(paths['auctions_results'])
        log(f"Auctions results: {len(ar):,} rows", f)

        au = pd.read_parquet(paths['auctions_users'])
        log(f"Auctions users: {len(au):,} rows", f)

        imp = None
        if paths['impressions'].exists():
            imp = pd.read_parquet(paths['impressions'])
            log(f"Impressions: {len(imp):,} rows", f)

        clicks = None
        if paths['clicks'].exists():
            clicks = pd.read_parquet(paths['clicks'])
            log(f"Clicks: {len(clicks):,} rows", f)

        ar_with_placement, imp_with_placement, clicks_with_placement = placement_overview(au, ar, imp, clicks, f)
        session_structure_by_placement(au, ar_with_placement, imp_with_placement, f)
        ctr_by_rank_per_placement(ar_with_placement, imp_with_placement, clicks_with_placement, f)
        rank_distribution_by_placement(ar_with_placement, imp_with_placement, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
