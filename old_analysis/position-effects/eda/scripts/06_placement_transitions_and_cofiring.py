#!/usr/bin/env python3
"""
Placement Transitions and Co-firing

User transitions between placements, simultaneous fires, pagination evidence.
Documents how users move between placement types.

Usage:
    python 06_placement_transitions_and_cofiring.py --round round1
    python 06_placement_transitions_and_cofiring.py --round round2
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
def transition_matrix(au, f):
    """Full placement transition matrix."""
    log(f"\n{'='*80}", f)
    log(f"PLACEMENT TRANSITION MATRIX", f)
    log(f"{'='*80}", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['PREV_PLACEMENT'] = au_sorted.groupby('USER_ID')['PLACEMENT'].shift(1)

    transitions = au_sorted[au_sorted['PREV_PLACEMENT'].notna()].copy()

    placements = sorted(au['PLACEMENT'].dropna().unique())

    log(f"\n--- Raw Transition Counts ---", f)
    transition_counts = transitions.groupby(['PREV_PLACEMENT', 'PLACEMENT']).size().unstack(fill_value=0)

    header = "From/To".ljust(10) + " " + " ".join([str(p).rjust(10) for p in placements])
    log(header, f)
    log("-"*10 + " " + " ".join(["-"*10 for _ in placements]), f)

    for from_p in placements:
        if from_p in transition_counts.index:
            row = [str(int(transition_counts.loc[from_p, to_p])).rjust(10) if to_p in transition_counts.columns else "0".rjust(10) for to_p in placements]
            log(str(from_p).ljust(10) + " " + " ".join(row), f)

    log(f"\n--- Row-Normalized Transition Probabilities (%) ---", f)
    transition_pcts = transition_counts.div(transition_counts.sum(axis=1), axis=0) * 100

    header = "From/To".ljust(10) + " " + " ".join([str(p).rjust(10) for p in placements])
    log(header, f)
    log("-"*10 + " " + " ".join(["-"*10 for _ in placements]), f)

    for from_p in placements:
        if from_p in transition_pcts.index:
            row = [f"{transition_pcts.loc[from_p, to_p]:.1f}%".rjust(10) if to_p in transition_pcts.columns else "0.0%".rjust(10) for to_p in placements]
            log(str(from_p).ljust(10) + " " + " ".join(row), f)

    log(f"\n--- Summary ---", f)
    log(f"Total transitions: {len(transitions):,}", f)
    for from_p in placements:
        if from_p in transition_pcts.index and from_p in transition_pcts.columns:
            self_rate = transition_pcts.loc[from_p, from_p]
            log(f"P{from_p} self-transition rate: {self_rate:.1f}%", f)


def cofiring_analysis(au, f):
    """Analyze placements that fire in close temporal proximity."""
    log(f"\n{'='*80}", f)
    log(f"CO-FIRING ANALYSIS", f)
    log(f"{'='*80}", f)
    log("Placements that fire within short time windows of each other", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
    au_sorted['PREV_PLACEMENT'] = au_sorted.groupby('USER_ID')['PLACEMENT'].shift(1)

    placements = sorted(au['PLACEMENT'].dropna().unique())

    for threshold in [1, 5, 10]:
        log(f"\n--- Co-fires within {threshold} seconds ---", f)

        cofires = au_sorted[(au_sorted['TIME_GAP'].notna()) & (au_sorted['TIME_GAP'] <= threshold) & (au_sorted['PREV_PLACEMENT'].notna())]

        if len(cofires) == 0:
            log(f"No co-fires within {threshold}s", f)
            continue

        log(f"Total co-fires: {len(cofires):,}", f)

        cofire_matrix = cofires.groupby(['PREV_PLACEMENT', 'PLACEMENT']).size().unstack(fill_value=0)

        header = "From/To".ljust(10) + " " + " ".join([str(p).rjust(8) for p in placements])
        log(header, f)
        log("-"*10 + " " + " ".join(["-"*8 for _ in placements]), f)

        for from_p in placements:
            if from_p in cofire_matrix.index:
                row = [str(int(cofire_matrix.loc[from_p, to_p])).rjust(8) if to_p in cofire_matrix.columns else "0".rjust(8) for to_p in placements]
                log(str(from_p).ljust(10) + " " + " ".join(row), f)


def pagination_evidence(au, ar, f):
    """Look for evidence of pagination (same products appearing at different ranks across consecutive auctions)."""
    log(f"\n{'='*80}", f)
    log(f"PAGINATION EVIDENCE", f)
    log(f"{'='*80}", f)
    log("Looking for same products at shifted ranks in consecutive auctions", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    placements = sorted(au['PLACEMENT'].dropna().unique())

    for p in placements:
        log(f"\n--- Placement {p} ---", f)

        au_p = au[au['PLACEMENT'] == p].copy()
        if not pd.api.types.is_datetime64_any_dtype(au_p['CREATED_AT']):
            au_p['CREATED_AT'] = pd.to_datetime(au_p['CREATED_AT'])

        au_p_sorted = au_p.sort_values(['USER_ID', 'CREATED_AT'])

        au_p_sorted['PREV_AUCTION'] = au_p_sorted.groupby('USER_ID')['AUCTION_ID'].shift(1)
        au_p_sorted['TIME_GAP'] = au_p_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()

        consecutive = au_p_sorted[(au_p_sorted['PREV_AUCTION'].notna()) & (au_p_sorted['TIME_GAP'] < 30)]

        if len(consecutive) == 0:
            log(f"No consecutive auctions within 30s", f)
            continue

        log(f"Consecutive auction pairs (within 30s): {len(consecutive):,}", f)

        ar_subset = ar[ar['AUCTION_ID'].isin(set(consecutive['AUCTION_ID']) | set(consecutive['PREV_AUCTION'].dropna()))]

        sample_pairs = consecutive.head(min(1000, len(consecutive)))
        overlap_counts = []

        for _, row in tqdm(sample_pairs.iterrows(), total=len(sample_pairs), desc=f"Analyzing P{p}", leave=False):
            current_auction = row['AUCTION_ID']
            prev_auction = row['PREV_AUCTION']

            current_products = set(ar_subset[ar_subset['AUCTION_ID'] == current_auction]['PRODUCT_ID'])
            prev_products = set(ar_subset[ar_subset['AUCTION_ID'] == prev_auction]['PRODUCT_ID'])

            overlap = len(current_products & prev_products)
            overlap_counts.append(overlap)

        if overlap_counts:
            mean_overlap = np.mean(overlap_counts)
            pct_with_overlap = (np.array(overlap_counts) > 0).mean() * 100
            log(f"Mean product overlap: {mean_overlap:.2f}", f)
            log(f"Pairs with any overlap: {pct_with_overlap:.1f}%", f)


def burst_analysis(au, f):
    """Analyze burst patterns (multiple consecutive auctions in same placement)."""
    log(f"\n{'='*80}", f)
    log(f"BURST ANALYSIS", f)
    log(f"{'='*80}", f)
    log("Consecutive auctions in same placement with short gaps", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    placements = sorted(au['PLACEMENT'].dropna().unique())

    for p in placements:
        log(f"\n--- Placement {p} ---", f)

        au_p = au[au['PLACEMENT'] == p].copy()
        au_p_sorted = au_p.sort_values(['USER_ID', 'CREATED_AT'])

        au_p_sorted['TIME_GAP'] = au_p_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
        au_p_sorted['NEW_BURST'] = (au_p_sorted['TIME_GAP'] > 30) | (au_p_sorted['TIME_GAP'].isna())
        au_p_sorted['BURST_ID'] = au_p_sorted.groupby('USER_ID')['NEW_BURST'].cumsum()

        burst_sizes = au_p_sorted.groupby(['USER_ID', 'BURST_ID']).size()

        if len(burst_sizes) == 0:
            log("No bursts found", f)
            continue

        log(f"Total bursts: {len(burst_sizes):,}", f)
        log(f"Mean burst size: {burst_sizes.mean():.2f}", f)
        log(f"Median burst size: {burst_sizes.median():.0f}", f)
        log(f"Max burst size: {burst_sizes.max()}", f)

        log(f"\nBurst size distribution:", f)
        log(f"{'Size':>8} {'Count':>12} {'%':>10}", f)
        log(f"{'-'*8} {'-'*12} {'-'*10}", f)

        burst_dist = burst_sizes.value_counts().sort_index()
        for size in sorted(burst_dist.index[:10]):
            count = burst_dist[size]
            pct = count / len(burst_sizes) * 100
            log(f"{size:>8} {count:>12,} {pct:>9.1f}%", f)


def multi_placement_users(au, f):
    """Analyze users who visit multiple placements."""
    log(f"\n{'='*80}", f)
    log(f"MULTI-PLACEMENT USERS", f)
    log(f"{'='*80}", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    user_placements = au.groupby('USER_ID')['PLACEMENT'].nunique()
    total_users = len(user_placements)

    log(f"\n--- Users by Number of Placements Visited ---", f)
    log(f"{'Placements':>12} {'Users':>15} {'%':>10}", f)
    log(f"{'-'*12} {'-'*15} {'-'*10}", f)

    for n in sorted(user_placements.unique()):
        count = (user_placements == n).sum()
        pct = count / total_users * 100
        log(f"{n:>12} {count:>15,} {pct:>9.1f}%", f)

    log(f"\n--- Placement Combinations for Multi-Placement Users ---", f)

    multi_users = user_placements[user_placements > 1].index
    if len(multi_users) > 0:
        user_placement_sets = au[au['USER_ID'].isin(multi_users)].groupby('USER_ID')['PLACEMENT'].apply(lambda x: tuple(sorted(set(x))))
        combo_counts = user_placement_sets.value_counts().head(15)

        log(f"{'Combination':<30} {'Users':>12} {'%':>10}", f)
        log(f"{'-'*30} {'-'*12} {'-'*10}", f)

        for combo, count in combo_counts.items():
            pct = count / len(multi_users) * 100
            combo_str = ", ".join(str(p) for p in combo)
            log(f"{combo_str:<30} {count:>12,} {pct:>9.1f}%", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Placement transitions and co-firing EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"06_placement_transitions_and_cofiring_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("PLACEMENT TRANSITIONS AND CO-FIRING", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)

        paths = get_data_paths(args.round)

        log(f"\n--- Loading Data ---", f)
        ar = pd.read_parquet(paths['auctions_results'])
        log(f"Auctions results: {len(ar):,} rows", f)

        au = pd.read_parquet(paths['auctions_users'])
        log(f"Auctions users: {len(au):,} rows", f)

        transition_matrix(au, f)
        cofiring_analysis(au, f)
        pagination_evidence(au, ar, f)
        burst_analysis(au, f)
        multi_placement_users(au, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
