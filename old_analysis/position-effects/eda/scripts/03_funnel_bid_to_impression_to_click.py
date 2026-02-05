#!/usr/bin/env python3
"""
Funnel: Bid to Impression to Click

Conversion rates through the ad funnel.
Documents bid->winner->impression->click transitions.

Usage:
    python 03_funnel_bid_to_impression_to_click.py --round round1
    python 03_funnel_bid_to_impression_to_click.py --round round2
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
def overall_funnel(ar, au, imp, clicks, f):
    """Overall funnel conversion rates."""
    log(f"\n{'='*80}", f)
    log(f"OVERALL FUNNEL", f)
    log(f"{'='*80}", f)

    n_auctions = au['AUCTION_ID'].nunique()
    n_bids = len(ar)
    n_winners = ar['IS_WINNER'].sum()
    n_impressions = len(imp) if imp is not None else 0
    n_clicks = len(clicks) if clicks is not None else 0

    log(f"\n--- Absolute Counts ---", f)
    log(f"{'Stage':<25} {'Count':>20}", f)
    log(f"{'-'*25} {'-'*20}", f)
    log(f"{'Auctions':<25} {n_auctions:>20,}", f)
    log(f"{'Bids':<25} {n_bids:>20,}", f)
    log(f"{'Winners (IS_WINNER=True)':<25} {n_winners:>20,}", f)
    log(f"{'Impressions':<25} {n_impressions:>20,}", f)
    log(f"{'Clicks':<25} {n_clicks:>20,}", f)

    log(f"\n--- Per-Auction Rates ---", f)
    log(f"{'Metric':<40} {'Value':>15}", f)
    log(f"{'-'*40} {'-'*15}", f)
    log(f"{'Bids per auction':<40} {n_bids/n_auctions:>15.1f}", f)
    log(f"{'Winners per auction':<40} {n_winners/n_auctions:>15.1f}", f)
    log(f"{'Impressions per auction':<40} {n_impressions/n_auctions:>15.2f}", f)
    log(f"{'Clicks per auction':<40} {n_clicks/n_auctions:>15.4f}", f)

    log(f"\n--- Funnel Conversion Rates ---", f)
    log(f"{'Transition':<40} {'Rate':>15}", f)
    log(f"{'-'*40} {'-'*15}", f)

    if n_bids > 0:
        log(f"{'Bid -> Winner':<40} {(n_winners/n_bids)*100:>14.2f}%", f)
    if n_winners > 0:
        log(f"{'Winner -> Impression':<40} {(n_impressions/n_winners)*100:>14.2f}%", f)
    if n_impressions > 0:
        log(f"{'Impression -> Click (CTR)':<40} {(n_clicks/n_impressions)*100:>14.2f}%", f)
    if n_winners > 0:
        log(f"{'Winner -> Click':<40} {(n_clicks/n_winners)*100:>14.2f}%", f)


def funnel_by_placement(ar, au, imp, clicks, f):
    """Funnel metrics by placement."""
    log(f"\n{'='*80}", f)
    log(f"FUNNEL BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available in auctions_users", f)
        return

    ar_with_placement = ar.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')

    if imp is not None:
        imp_with_placement = imp.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
    else:
        imp_with_placement = None

    if clicks is not None:
        clicks_with_placement = clicks.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
    else:
        clicks_with_placement = None

    placements = sorted(au['PLACEMENT'].dropna().unique())

    log(f"\n--- Volume by Placement ---", f)
    log(f"{'Placement':<12} {'Auctions':>12} {'Bids':>12} {'Winners':>12} {'Imps':>12} {'Clicks':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}", f)

    for p in placements:
        n_auctions = (au['PLACEMENT'] == p).sum()
        n_bids = (ar_with_placement['PLACEMENT'] == p).sum()
        n_winners = ((ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)).sum()
        n_imp = (imp_with_placement['PLACEMENT'] == p).sum() if imp_with_placement is not None else 0
        n_clicks = (clicks_with_placement['PLACEMENT'] == p).sum() if clicks_with_placement is not None else 0

        log(f"{str(p):<12} {n_auctions:>12,} {n_bids:>12,} {n_winners:>12,} {n_imp:>12,} {n_clicks:>10,}", f)

    log(f"\n--- Rates by Placement ---", f)
    log(f"{'Placement':<12} {'CTR':>10} {'Imp/Auction':>14} {'Imp/Winner':>12} {'Click/Auction':>15}", f)
    log(f"{'-'*12} {'-'*10} {'-'*14} {'-'*12} {'-'*15}", f)

    for p in placements:
        n_auctions = (au['PLACEMENT'] == p).sum()
        n_winners = ((ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)).sum()
        n_imp = (imp_with_placement['PLACEMENT'] == p).sum() if imp_with_placement is not None else 0
        n_clicks = (clicks_with_placement['PLACEMENT'] == p).sum() if clicks_with_placement is not None else 0

        ctr = (n_clicks / n_imp * 100) if n_imp > 0 else 0
        imp_per_auction = n_imp / n_auctions if n_auctions > 0 else 0
        imp_per_winner = n_imp / n_winners if n_winners > 0 else 0
        click_per_auction = n_clicks / n_auctions if n_auctions > 0 else 0

        log(f"{str(p):<12} {ctr:>9.2f}% {imp_per_auction:>14.2f} {imp_per_winner:>12.3f} {click_per_auction:>15.4f}", f)


def ctr_by_rank(ar, imp, clicks, f):
    """CTR by ranking."""
    log(f"\n{'='*80}", f)
    log(f"CTR BY RANKING", f)
    log(f"{'='*80}", f)

    if imp is None or clicks is None:
        log("Impression or click data not available", f)
        return

    winners = ar[ar['IS_WINNER'] == True].copy()

    click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
    winners['clicked'] = winners.apply(
        lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
    )

    log(f"\nTotal winners: {len(winners):,}", f)
    log(f"Total clicked: {winners['clicked'].sum():,}", f)

    ctr_by_rank = winners.groupby('RANKING').agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    ctr_by_rank.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

    log(f"\n--- CTR by Rank ---", f)
    log(f"{'Rank':>8} {'Impressions':>15} {'Clicks':>12} {'CTR %':>12}", f)
    log(f"{'-'*8} {'-'*15} {'-'*12} {'-'*12}", f)

    for _, row in ctr_by_rank.head(25).iterrows():
        log(f"{int(row['RANKING']):>8} {int(row['impressions']):>15,} {int(row['clicks']):>12,} {row['CTR']*100:>11.3f}%", f)

    ctr_values = ctr_by_rank.head(10)['CTR'].values
    is_monotonic = all(ctr_values[i] >= ctr_values[i+1] for i in range(len(ctr_values)-1))
    log(f"\nCTR monotonically decreasing (top 10 positions): {is_monotonic}", f)


def winner_to_impression_rate_by_rank(ar, imp, f):
    """What fraction of winners at each rank get impressions?"""
    log(f"\n{'='*80}", f)
    log(f"WINNER -> IMPRESSION RATE BY RANK", f)
    log(f"{'='*80}", f)

    if imp is None:
        log("Impression data not available", f)
        return

    winners = ar[ar['IS_WINNER'] == True].copy()

    winners_with_imp = winners.merge(
        imp[['AUCTION_ID', 'PRODUCT_ID']].drop_duplicates(),
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left',
        indicator=True
    )
    winners_with_imp['got_impression'] = winners_with_imp['_merge'] == 'both'

    log(f"\nTotal winners: {len(winners):,}", f)
    log(f"Winners with impressions: {winners_with_imp['got_impression'].sum():,} ({winners_with_imp['got_impression'].mean()*100:.2f}%)", f)

    log(f"\n--- Impression Rate by Rank ---", f)
    log(f"{'Rank':>8} {'Winners':>15} {'With Imp':>15} {'Imp Rate %':>12}", f)
    log(f"{'-'*8} {'-'*15} {'-'*15} {'-'*12}", f)

    imp_rate_by_rank = winners_with_imp.groupby('RANKING').agg({
        'got_impression': ['sum', 'count', 'mean']
    }).reset_index()
    imp_rate_by_rank.columns = ['RANKING', 'n_impressed', 'n_winners', 'imp_rate']

    for _, row in imp_rate_by_rank.head(20).iterrows():
        log(f"{int(row['RANKING']):>8} {int(row['n_winners']):>15,} {int(row['n_impressed']):>15,} {row['imp_rate']*100:>11.2f}%", f)

    imp_rates = imp_rate_by_rank.head(10)['imp_rate'].values
    is_monotonic = all(imp_rates[i] >= imp_rates[i+1] for i in range(len(imp_rates)-1))
    log(f"\nImpression rate monotonically decreasing (top 10): {is_monotonic}", f)


def click_distribution(clicks, au, f):
    """Clicks per auction distribution."""
    log(f"\n{'='*80}", f)
    log(f"CLICK DISTRIBUTION", f)
    log(f"{'='*80}", f)

    if clicks is None or len(clicks) == 0:
        log("No click data available", f)
        return

    n_auctions = au['AUCTION_ID'].nunique()
    clicks_per_auction = clicks.groupby('AUCTION_ID').size()

    auctions_with_clicks = len(clicks_per_auction)
    auctions_with_0_clicks = n_auctions - auctions_with_clicks

    log(f"\nTotal auctions: {n_auctions:,}", f)
    log(f"Auctions with 0 clicks: {auctions_with_0_clicks:,} ({auctions_with_0_clicks/n_auctions*100:.1f}%)", f)
    log(f"Auctions with 1+ clicks: {auctions_with_clicks:,} ({auctions_with_clicks/n_auctions*100:.1f}%)", f)

    log(f"\n--- Distribution (auctions with clicks) ---", f)
    log(f"Mean clicks/auction: {clicks_per_auction.mean():.2f}", f)
    log(f"Median: {clicks_per_auction.median():.0f}", f)
    log(f"Max: {clicks_per_auction.max()}", f)

    log(f"\n--- Click Count Distribution ---", f)
    log(f"{'N Clicks':>10} {'Auctions':>15} {'Percentage':>15}", f)
    log(f"{'-'*10} {'-'*15} {'-'*15}", f)

    log(f"{'0':>10} {auctions_with_0_clicks:>15,} {auctions_with_0_clicks/n_auctions*100:>14.1f}%", f)

    click_dist = clicks_per_auction.value_counts().sort_index()
    for n_clicks in sorted(click_dist.index[:10]):
        count = click_dist[n_clicks]
        pct = count / n_auctions * 100
        log(f"{n_clicks:>10} {count:>15,} {pct:>14.1f}%", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Funnel bid to impression to click EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"03_funnel_bid_to_impression_to_click_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("FUNNEL: BID TO IMPRESSION TO CLICK", f)
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

        overall_funnel(ar, au, imp, clicks, f)
        funnel_by_placement(ar, au, imp, clicks, f)
        ctr_by_rank(ar, imp, clicks, f)
        winner_to_impression_rate_by_rank(ar, imp, f)
        click_distribution(clicks, au, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
