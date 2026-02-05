#!/usr/bin/env python3
"""
CTR by Rank and Position

Click-through by ranking, position effects analysis.
Detailed position-based click patterns.

Usage:
    python 09_ctr_by_rank_and_position.py --round round1
    python 09_ctr_by_rank_and_position.py --round round2
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
def overall_ctr_by_rank(ar, imp, clicks, f):
    """Overall CTR by rank."""
    log(f"\n{'='*80}", f)
    log(f"OVERALL CTR BY RANK", f)
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
    log(f"Overall CTR: {winners['clicked'].mean()*100:.3f}%", f)

    ctr_by_rank = winners.groupby('RANKING').agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    ctr_by_rank.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

    log(f"\n--- CTR by Rank (Top 30) ---", f)
    log(f"{'Rank':>8} {'Impressions':>15} {'Clicks':>12} {'CTR %':>12} {'Rel to R1':>12}", f)
    log(f"{'-'*8} {'-'*15} {'-'*12} {'-'*12} {'-'*12}", f)

    rank1_ctr = ctr_by_rank[ctr_by_rank['RANKING'] == 1]['CTR'].values[0] if 1 in ctr_by_rank['RANKING'].values else ctr_by_rank['CTR'].max()

    for _, row in ctr_by_rank.head(30).iterrows():
        rel_to_r1 = row['CTR'] / rank1_ctr if rank1_ctr > 0 else 0
        log(f"{int(row['RANKING']):>8} {int(row['impressions']):>15,} {int(row['clicks']):>12,} {row['CTR']*100:>11.3f}% {rel_to_r1:>11.3f}", f)

    ctr_values = ctr_by_rank.head(10)['CTR'].values
    is_monotonic = all(ctr_values[i] >= ctr_values[i+1] for i in range(len(ctr_values)-1))
    log(f"\nCTR monotonically decreasing (top 10): {is_monotonic}", f)


def position_effect_magnitude(ar, clicks, f):
    """Quantify position effect magnitude."""
    log(f"\n{'='*80}", f)
    log(f"POSITION EFFECT MAGNITUDE", f)
    log(f"{'='*80}", f)

    if clicks is None:
        log("Click data not available", f)
        return

    winners = ar[ar['IS_WINNER'] == True].copy()
    click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
    winners['clicked'] = winners.apply(
        lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
    )

    ctr_by_rank = winners.groupby('RANKING').agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    ctr_by_rank.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

    rank1_ctr = ctr_by_rank[ctr_by_rank['RANKING'] == 1]['CTR'].values[0] if 1 in ctr_by_rank['RANKING'].values else 0

    log(f"\n--- Position Effect Relative to Rank 1 ---", f)
    log(f"{'Rank':>8} {'CTR %':>12} {'Ratio to R1':>14} {'CTR Decay':>12}", f)
    log(f"{'-'*8} {'-'*12} {'-'*14} {'-'*12}", f)

    for _, row in ctr_by_rank.head(15).iterrows():
        if row['impressions'] >= 100:
            ratio = row['CTR'] / rank1_ctr if rank1_ctr > 0 else 0
            decay = (1 - ratio) * 100
            log(f"{int(row['RANKING']):>8} {row['CTR']*100:>11.3f}% {ratio:>13.3f} {decay:>11.1f}%", f)

    log(f"\n--- Position Effect Summary ---", f)
    rank5_ctr = ctr_by_rank[ctr_by_rank['RANKING'] == 5]['CTR'].values[0] if 5 in ctr_by_rank['RANKING'].values else 0
    rank10_ctr = ctr_by_rank[ctr_by_rank['RANKING'] == 10]['CTR'].values[0] if 10 in ctr_by_rank['RANKING'].values else 0

    log(f"Rank 1 CTR: {rank1_ctr*100:.3f}%", f)
    if rank5_ctr > 0:
        log(f"Rank 5 CTR: {rank5_ctr*100:.3f}% (ratio: {rank5_ctr/rank1_ctr:.3f})", f)
    if rank10_ctr > 0:
        log(f"Rank 10 CTR: {rank10_ctr*100:.3f}% (ratio: {rank10_ctr/rank1_ctr:.3f})", f)


def ctr_by_rank_per_placement(ar, au, clicks, f):
    """CTR by rank for each placement."""
    log(f"\n{'='*80}", f)
    log(f"CTR BY RANK PER PLACEMENT", f)
    log(f"{'='*80}", f)

    if clicks is None:
        log("Click data not available", f)
        return

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    ar_with_placement = ar.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
    winners = ar_with_placement[ar_with_placement['IS_WINNER'] == True].copy()

    click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
    winners['clicked'] = winners.apply(
        lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
    )

    placements = sorted(au['PLACEMENT'].dropna().unique())

    for p in placements:
        log(f"\n--- Placement {p} ---", f)

        winners_p = winners[winners['PLACEMENT'] == p]
        if len(winners_p) == 0:
            log("No winners for this placement", f)
            continue

        ctr_by_rank = winners_p.groupby('RANKING').agg({
            'clicked': ['sum', 'count', 'mean']
        }).reset_index()
        ctr_by_rank.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

        log(f"{'Rank':>8} {'Impressions':>15} {'Clicks':>10} {'CTR %':>10}", f)
        log(f"{'-'*8} {'-'*15} {'-'*10} {'-'*10}", f)

        for _, row in ctr_by_rank.head(15).iterrows():
            log(f"{int(row['RANKING']):>8} {int(row['impressions']):>15,} {int(row['clicks']):>10,} {row['CTR']*100:>9.3f}%", f)

        if len(ctr_by_rank) >= 3:
            ctr_vals = ctr_by_rank.head(5)['CTR'].values
            trend = "DECLINING" if ctr_vals[0] > ctr_vals[-1] else ("INCREASING" if ctr_vals[-1] > ctr_vals[0] else "FLAT")
            log(f"\nPattern: {trend}", f)


def click_rank_distribution(ar, clicks, f):
    """Distribution of ranks that get clicked."""
    log(f"\n{'='*80}", f)
    log(f"CLICK RANK DISTRIBUTION", f)
    log(f"{'='*80}", f)

    if clicks is None:
        log("Click data not available", f)
        return

    winners = ar[ar['IS_WINNER'] == True].copy()

    clicked_winners = winners.merge(
        clicks[['AUCTION_ID', 'PRODUCT_ID']].drop_duplicates(),
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='inner'
    )

    log(f"\nTotal clicks: {len(clicked_winners):,}", f)

    click_rank_dist = clicked_winners['RANKING'].value_counts().sort_index()

    log(f"\n--- Rank Distribution of Clicks ---", f)
    log(f"{'Rank':>8} {'Clicks':>12} {'%':>10} {'Cumulative %':>15}", f)
    log(f"{'-'*8} {'-'*12} {'-'*10} {'-'*15}", f)

    total_clicks = len(clicked_winners)
    cumsum = 0
    for rank in sorted(click_rank_dist.index[:20]):
        count = click_rank_dist[rank]
        pct = count / total_clicks * 100
        cumsum += pct
        log(f"{int(rank):>8} {count:>12,} {pct:>9.1f}% {cumsum:>14.1f}%", f)

    log(f"\n--- Click Concentration ---", f)
    rank1_clicks = click_rank_dist.get(1, 0)
    top3_clicks = sum(click_rank_dist.get(r, 0) for r in range(1, 4))
    top5_clicks = sum(click_rank_dist.get(r, 0) for r in range(1, 6))
    top10_clicks = sum(click_rank_dist.get(r, 0) for r in range(1, 11))

    log(f"Rank 1 captures: {rank1_clicks/total_clicks*100:.1f}% of clicks", f)
    log(f"Ranks 1-3 capture: {top3_clicks/total_clicks*100:.1f}% of clicks", f)
    log(f"Ranks 1-5 capture: {top5_clicks/total_clicks*100:.1f}% of clicks", f)
    log(f"Ranks 1-10 capture: {top10_clicks/total_clicks*100:.1f}% of clicks", f)


def position_variation_analysis(ar, clicks, f):
    """Analyze products that appear at multiple positions."""
    log(f"\n{'='*80}", f)
    log(f"POSITION VARIATION ANALYSIS", f)
    log(f"{'='*80}", f)
    log("Analyzing same products appearing at different ranks", f)

    if clicks is None:
        log("Click data not available", f)
        return

    winners = ar[ar['IS_WINNER'] == True].copy()

    click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
    winners['clicked'] = winners.apply(
        lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
    )

    product_positions = winners.groupby('PRODUCT_ID').agg({
        'RANKING': ['nunique', 'min', 'max', 'mean', 'count'],
        'clicked': ['sum', 'mean']
    })
    product_positions.columns = ['n_positions', 'min_rank', 'max_rank', 'mean_rank', 'n_impressions', 'clicks', 'ctr']
    product_positions['position_range'] = product_positions['max_rank'] - product_positions['min_rank']

    log(f"\nTotal unique products (winners): {len(product_positions):,}", f)

    products_multi_pos = product_positions[product_positions['n_positions'] >= 2]
    log(f"Products with 2+ positions: {len(products_multi_pos):,} ({len(products_multi_pos)/len(product_positions)*100:.1f}%)", f)

    products_3plus_pos = product_positions[product_positions['n_positions'] >= 3]
    log(f"Products with 3+ positions: {len(products_3plus_pos):,} ({len(products_3plus_pos)/len(product_positions)*100:.1f}%)", f)

    log(f"\n--- Position Range Statistics (products with 2+ positions) ---", f)
    log(f"Mean position range: {products_multi_pos['position_range'].mean():.2f}", f)
    log(f"Median position range: {products_multi_pos['position_range'].median():.0f}", f)
    log(f"Max position range: {products_multi_pos['position_range'].max():.0f}", f)

    log(f"\n--- CTR vs Position for Multi-Position Products ---", f)

    products_sample = products_multi_pos[products_multi_pos['n_impressions'] >= 10].head(1000).index
    sample_data = winners[winners['PRODUCT_ID'].isin(products_sample)]

    ctr_by_rank_sample = sample_data.groupby('RANKING').agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    ctr_by_rank_sample.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

    log(f"{'Rank':>8} {'Impressions':>15} {'Clicks':>10} {'CTR %':>10}", f)
    log(f"{'-'*8} {'-'*15} {'-'*10} {'-'*10}", f)

    for _, row in ctr_by_rank_sample[ctr_by_rank_sample['impressions'] >= 50].head(15).iterrows():
        log(f"{int(row['RANKING']):>8} {int(row['impressions']):>15,} {int(row['clicks']):>10,} {row['CTR']*100:>9.3f}%", f)


def hazard_rate_analysis(ar, clicks, f):
    """Compute hazard rates (conditional click probability)."""
    log(f"\n{'='*80}", f)
    log(f"HAZARD RATE ANALYSIS", f)
    log(f"{'='*80}", f)
    log("Hazard = P(click at rank k | no click before rank k)", f)

    if clicks is None:
        log("Click data not available", f)
        return

    winners = ar[ar['IS_WINNER'] == True].copy()
    click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
    winners['clicked'] = winners.apply(
        lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
    )

    imp_by_rank = winners.groupby('RANKING').size().sort_index()
    click_by_rank = winners[winners['clicked']].groupby('RANKING').size().sort_index()

    log(f"\n--- Hazard Rates by Rank ---", f)
    log(f"{'Rank':>8} {'At Risk':>15} {'Clicks':>10} {'Hazard':>10} {'Survival':>10}", f)
    log(f"{'-'*8} {'-'*15} {'-'*10} {'-'*10} {'-'*10}", f)

    survival = 1.0
    for rank in sorted(imp_by_rank.index[:20]):
        at_risk = imp_by_rank.get(rank, 0)
        n_clicks = click_by_rank.get(rank, 0)
        hazard = n_clicks / at_risk if at_risk > 0 else 0
        survival = survival * (1 - hazard)
        log(f"{int(rank):>8} {at_risk:>15,} {n_clicks:>10,} {hazard:>9.4f} {survival:>9.4f}", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='CTR by rank and position EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"09_ctr_by_rank_and_position_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("CTR BY RANK AND POSITION", f)
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

        overall_ctr_by_rank(ar, imp, clicks, f)
        position_effect_magnitude(ar, clicks, f)
        ctr_by_rank_per_placement(ar, au, clicks, f)
        click_rank_distribution(ar, clicks, f)
        position_variation_analysis(ar, clicks, f)
        hazard_rate_analysis(ar, clicks, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
