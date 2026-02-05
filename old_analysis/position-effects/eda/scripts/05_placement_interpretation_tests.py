#!/usr/bin/env python3
"""
Placement Interpretation Tests

Hypothesis tests for placement meanings (e.g., browse vs search vs PDP).
Statistical evidence for placement type interpretations.

Usage:
    python 05_placement_interpretation_tests.py --round round1
    python 05_placement_interpretation_tests.py --round round2
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
# TEST FUNCTIONS
# =============================================================================
def test_time_gaps(au, f):
    """Test time gaps between consecutive auctions by placement."""
    log(f"\n{'='*80}", f)
    log(f"TEST: TIME GAPS BETWEEN AUCTIONS", f)
    log(f"{'='*80}", f)
    log("Hypothesis: Different placements have different browsing tempo", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    placements = sorted(au['PLACEMENT'].dropna().unique())

    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()

    log(f"\n--- Time Gap Statistics (seconds) ---", f)
    log(f"{'Placement':<12} {'Mean':>12} {'Median':>12} {'P25':>10} {'P75':>10} {'P95':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)

    for p in placements:
        gaps = au_sorted[(au_sorted['PLACEMENT'] == p) & (au_sorted['TIME_GAP'].notna())]['TIME_GAP']
        if len(gaps) > 0:
            log(f"{str(p):<12} {gaps.mean():>12.1f} {gaps.median():>12.1f} {gaps.quantile(0.25):>10.1f} {gaps.quantile(0.75):>10.1f} {gaps.quantile(0.95):>10.1f}", f)

    log(f"\n--- Rapid-fire Analysis (gaps < 5 seconds) ---", f)
    log(f"{'Placement':<12} {'Total Gaps':>15} {'<5s':>12} {'%':>10}", f)
    log(f"{'-'*12} {'-'*15} {'-'*12} {'-'*10}", f)

    for p in placements:
        gaps = au_sorted[(au_sorted['PLACEMENT'] == p) & (au_sorted['TIME_GAP'].notna())]['TIME_GAP']
        if len(gaps) > 0:
            rapid_fire = (gaps < 5).sum()
            pct = rapid_fire / len(gaps) * 100
            log(f"{str(p):<12} {len(gaps):>15,} {rapid_fire:>12,} {pct:>9.1f}%", f)


def test_self_transitions(au, f):
    """Test self-transition rates (user stays in same placement)."""
    log(f"\n{'='*80}", f)
    log(f"TEST: SELF-TRANSITION RATES", f)
    log(f"{'='*80}", f)
    log("Hypothesis: Search/pagination shows high self-transition, PDP shows low", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    placements = sorted(au['PLACEMENT'].dropna().unique())

    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['NEXT_PLACEMENT'] = au_sorted.groupby('USER_ID')['PLACEMENT'].shift(-1)

    log(f"\n--- Self-Transition Rates ---", f)
    log(f"{'From':>12} {'Total':>15} {'Same':>12} {'Self-Rate %':>12}", f)
    log(f"{'-'*12} {'-'*15} {'-'*12} {'-'*12}", f)

    for p in placements:
        p_transitions = au_sorted[au_sorted['PLACEMENT'] == p]
        p_to_p = (p_transitions['NEXT_PLACEMENT'] == p).sum()
        p_total = p_transitions['NEXT_PLACEMENT'].notna().sum()
        self_rate = p_to_p / p_total * 100 if p_total > 0 else 0
        log(f"{str(p):>12} {p_total:>15,} {p_to_p:>12,} {self_rate:>11.1f}%", f)

    log(f"\n--- Full Transition Matrix ---", f)
    header = "From/To".ljust(10) + " " + " ".join([str(p).rjust(10) for p in placements])
    log(header, f)
    log("-"*10 + " " + " ".join(["-"*10 for _ in placements]), f)

    for from_p in placements:
        p_transitions = au_sorted[au_sorted['PLACEMENT'] == from_p]
        total = p_transitions['NEXT_PLACEMENT'].notna().sum()
        row_values = []
        for to_p in placements:
            count = (p_transitions['NEXT_PLACEMENT'] == to_p).sum()
            pct = count / total * 100 if total > 0 else 0
            row_values.append(f"{pct:.1f}%".rjust(10))
        log(str(from_p).ljust(10) + " " + " ".join(row_values), f)


def test_impression_delivery(au, imp, f):
    """Test impression delivery rate by placement."""
    log(f"\n{'='*80}", f)
    log(f"TEST: IMPRESSION DELIVERY RATE", f)
    log(f"{'='*80}", f)
    log("Hypothesis: Browse/feed has high delivery, search may have low (user doesn't scroll)", f)

    if 'PLACEMENT' not in au.columns or imp is None:
        log("Required data not available", f)
        return

    placements = sorted(au['PLACEMENT'].dropna().unique())

    log(f"\n--- Auction Delivery Rate (% of auctions with at least one impression) ---", f)
    log(f"{'Placement':<12} {'Auctions':>15} {'With Imp':>15} {'Delivery %':>12}", f)
    log(f"{'-'*12} {'-'*15} {'-'*15} {'-'*12}", f)

    for p in placements:
        auctions_p = set(au[au['PLACEMENT'] == p]['AUCTION_ID'])
        auctions_with_imp = set(imp[imp['AUCTION_ID'].isin(auctions_p)]['AUCTION_ID'])
        delivery_rate = len(auctions_with_imp) / len(auctions_p) * 100 if len(auctions_p) > 0 else 0
        log(f"{str(p):<12} {len(auctions_p):>15,} {len(auctions_with_imp):>15,} {delivery_rate:>11.1f}%", f)


def test_user_concentration(au, f):
    """Test user concentration by placement."""
    log(f"\n{'='*80}", f)
    log(f"TEST: USER CONCENTRATION", f)
    log(f"{'='*80}", f)
    log("Hypothesis: Bot/scraper placements show extreme concentration", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    placements = sorted(au['PLACEMENT'].dropna().unique())

    log(f"\n--- User Activity by Placement ---", f)
    log(f"{'Placement':<12} {'Users':>12} {'Mean Aucs':>12} {'Max Aucs':>12} {'Top 1% Share':>14}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*14}", f)

    for p in placements:
        au_p = au[au['PLACEMENT'] == p]
        auctions_per_user = au_p.groupby('USER_ID').size()
        n_users = len(auctions_per_user)
        mean_aucs = auctions_per_user.mean()
        max_aucs = auctions_per_user.max()

        threshold = auctions_per_user.quantile(0.99)
        top1pct_share = auctions_per_user[auctions_per_user >= threshold].sum() / auctions_per_user.sum() * 100

        log(f"{str(p):<12} {n_users:>12,} {mean_aucs:>12.1f} {max_aucs:>12,} {top1pct_share:>13.1f}%", f)


def test_ctr_pattern(au, ar, imp, clicks, f):
    """Test CTR pattern by rank (declining vs flat vs increasing)."""
    log(f"\n{'='*80}", f)
    log(f"TEST: CTR PATTERN BY RANK", f)
    log(f"{'='*80}", f)
    log("Hypothesis: Browse shows declining CTR, PDP may show flat/increasing", f)

    if 'PLACEMENT' not in au.columns or imp is None or clicks is None:
        log("Required data not available", f)
        return

    placements = sorted(au['PLACEMENT'].dropna().unique())

    ar_with_placement = ar.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
    imp_with_placement = imp.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
    clicks_with_placement = clicks.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')

    imp_with_rank = imp_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )
    clicks_with_rank = clicks_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    log(f"\n--- CTR Trend Analysis by Placement ---", f)

    for p in placements:
        imp_p = imp_with_rank[imp_with_rank['PLACEMENT'] == p]
        clicks_p = clicks_with_rank[clicks_with_rank['PLACEMENT'] == p]

        ctrs = []
        for rank in range(1, 6):
            n_imp = (imp_p['RANKING'] == rank).sum()
            n_clicks = (clicks_p['RANKING'] == rank).sum()
            ctr = n_clicks / n_imp * 100 if n_imp > 0 else 0
            ctrs.append((rank, ctr, n_imp))

        log(f"\nPlacement {p}:", f)
        log(f"  {'Rank':>6} {'CTR %':>10} {'N Imps':>12}", f)
        log(f"  {'-'*6} {'-'*10} {'-'*12}", f)

        for rank, ctr, n_imp in ctrs:
            log(f"  {rank:>6} {ctr:>9.2f}% {n_imp:>12,}", f)

        if len(ctrs) >= 3:
            early_ctr = np.mean([c for _, c, n in ctrs[:2] if n > 100])
            late_ctr = np.mean([c for _, c, n in ctrs[2:5] if n > 100])
            trend = late_ctr - early_ctr

            if abs(trend) < 0.2:
                pattern = "FLAT"
            elif trend < 0:
                pattern = "DECLINING"
            else:
                pattern = "INCREASING"

            log(f"  Pattern: {pattern} (R1-2 avg: {early_ctr:.2f}%, R3-5 avg: {late_ctr:.2f}%)", f)


def test_multi_click_rate(au, clicks, f):
    """Test multi-click rate by placement."""
    log(f"\n{'='*80}", f)
    log(f"TEST: MULTI-CLICK RATE", f)
    log(f"{'='*80}", f)
    log("Hypothesis: Carousel/PDP placements may have higher multi-click", f)

    if 'PLACEMENT' not in au.columns or clicks is None:
        log("Required data not available", f)
        return

    placements = sorted(au['PLACEMENT'].dropna().unique())

    clicks_with_placement = clicks.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
    clicks_per_auction = clicks_with_placement.groupby(['AUCTION_ID', 'PLACEMENT']).size().reset_index(name='n_clicks')

    log(f"\n--- Multi-Click Analysis ---", f)
    log(f"{'Placement':<12} {'Auctions w/ Click':>18} {'Multi-Click':>15} {'Rate %':>10}", f)
    log(f"{'-'*12} {'-'*18} {'-'*15} {'-'*10}", f)

    for p in placements:
        subset = clicks_per_auction[clicks_per_auction['PLACEMENT'] == p]
        total = len(subset)
        multi = (subset['n_clicks'] >= 2).sum()
        rate = multi / total * 100 if total > 0 else 0
        log(f"{str(p):<12} {total:>18,} {multi:>15,} {rate:>9.1f}%", f)


def test_session_start_placement(au, f):
    """Test which placement typically starts user sessions."""
    log(f"\n{'='*80}", f)
    log(f"TEST: SESSION START PLACEMENT", f)
    log(f"{'='*80}", f)
    log("Hypothesis: Browse/feed typically starts sessions", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])

    first_auctions = au_sorted.groupby('USER_ID').first()
    first_placement_dist = first_auctions['PLACEMENT'].value_counts(normalize=True) * 100

    log(f"\n--- First Placement per User Session ---", f)
    log(f"{'Placement':<12} {'Percentage':>15}", f)
    log(f"{'-'*12} {'-'*15}", f)

    for p in sorted(first_placement_dist.index):
        log(f"{str(p):<12} {first_placement_dist[p]:>14.1f}%", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Placement interpretation tests')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"05_placement_interpretation_tests_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("PLACEMENT INTERPRETATION TESTS", f)
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

        test_time_gaps(au, f)
        test_self_transitions(au, f)
        test_impression_delivery(au, imp, f)
        test_user_concentration(au, f)
        test_ctr_pattern(au, ar, imp, clicks, f)
        test_multi_click_rate(au, clicks, f)
        test_session_start_placement(au, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
