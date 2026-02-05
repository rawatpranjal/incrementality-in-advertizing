#!/usr/bin/env python3
"""
User Sessions and Power Users

Multi-placement behavior, extreme users, time gaps, session patterns.
Documents user-level behavior patterns.

Usage:
    python 08_user_sessions_and_power_users.py --round round1
    python 08_user_sessions_and_power_users.py --round round2
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
def user_activity_distribution(au, f):
    """Distribution of auctions per user."""
    log(f"\n{'='*80}", f)
    log(f"USER ACTIVITY DISTRIBUTION", f)
    log(f"{'='*80}", f)

    auctions_per_user = au.groupby('USER_ID').size()
    total_users = len(auctions_per_user)

    log(f"\nTotal unique users: {total_users:,}", f)
    log(f"Total auctions: {len(au):,}", f)

    log(f"\n--- Auctions per User Statistics ---", f)
    log(f"Mean: {auctions_per_user.mean():.2f}", f)
    log(f"Median: {auctions_per_user.median():.0f}", f)
    log(f"Std: {auctions_per_user.std():.2f}", f)
    log(f"Min: {auctions_per_user.min()}", f)
    log(f"Max: {auctions_per_user.max()}", f)

    log(f"\n--- Percentiles ---", f)
    for pct in [50, 75, 90, 95, 99, 99.9]:
        val = auctions_per_user.quantile(pct/100)
        log(f"P{pct}: {val:.0f}", f)

    log(f"\n--- Distribution ---", f)
    log(f"{'Auctions':>12} {'Users':>15} {'%':>10} {'Cumulative %':>15}", f)
    log(f"{'-'*12} {'-'*15} {'-'*10} {'-'*15}", f)

    bins = [1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
    bin_labels = ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '100+']

    cumsum = 0
    for i, (low, high) in enumerate(zip([0] + bins[:-1], bins)):
        if i < len(bins) - 1:
            count = ((auctions_per_user > low) & (auctions_per_user <= high)).sum()
        else:
            count = (auctions_per_user > low).sum()

        pct = count / total_users * 100
        cumsum += pct
        label = bin_labels[i] if i < len(bin_labels) else f">{int(low)}"
        log(f"{label:>12} {count:>15,} {pct:>9.1f}% {cumsum:>14.1f}%", f)


def power_user_analysis(au, f):
    """Detailed analysis of power users (top 1%)."""
    log(f"\n{'='*80}", f)
    log(f"POWER USER ANALYSIS", f)
    log(f"{'='*80}", f)

    auctions_per_user = au.groupby('USER_ID').size()
    threshold_99 = auctions_per_user.quantile(0.99)
    power_users = auctions_per_user[auctions_per_user >= threshold_99].index

    log(f"\nPower user threshold (P99): {threshold_99:.0f} auctions", f)
    log(f"Number of power users: {len(power_users):,}", f)

    power_auctions = auctions_per_user[auctions_per_user >= threshold_99]
    power_share = power_auctions.sum() / auctions_per_user.sum() * 100

    log(f"Power users control: {power_share:.1f}% of all auctions", f)

    log(f"\n--- Power User Activity Statistics ---", f)
    log(f"Mean auctions: {power_auctions.mean():.1f}", f)
    log(f"Median auctions: {power_auctions.median():.0f}", f)
    log(f"Max auctions: {power_auctions.max()}", f)

    if 'PLACEMENT' in au.columns:
        log(f"\n--- Power User Placement Distribution ---", f)

        power_user_auctions = au[au['USER_ID'].isin(power_users)]
        overall_dist = au['PLACEMENT'].value_counts(normalize=True) * 100
        power_dist = power_user_auctions['PLACEMENT'].value_counts(normalize=True) * 100

        placements = sorted(au['PLACEMENT'].dropna().unique())

        log(f"{'Placement':<12} {'Overall %':>12} {'Power User %':>14} {'Diff':>10}", f)
        log(f"{'-'*12} {'-'*12} {'-'*14} {'-'*10}", f)

        for p in placements:
            overall_pct = overall_dist.get(p, 0)
            power_pct = power_dist.get(p, 0)
            diff = power_pct - overall_pct
            log(f"{str(p):<12} {overall_pct:>11.1f}% {power_pct:>13.1f}% {diff:>+9.1f}%", f)

    log(f"\n--- Top 20 Power Users ---", f)
    top_users = auctions_per_user.nlargest(20)

    log(f"{'User (truncated)':<25} {'Auctions':>12}", f)
    log(f"{'-'*25} {'-'*12}", f)

    for user_id, n_auctions in top_users.items():
        log(f"{str(user_id)[:25]:<25} {n_auctions:>12,}", f)


def time_gap_analysis(au, f):
    """Analyze time gaps between consecutive auctions."""
    log(f"\n{'='*80}", f)
    log(f"TIME GAP ANALYSIS", f)
    log(f"{'='*80}", f)

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()

    gaps = au_sorted['TIME_GAP'].dropna()

    log(f"\n--- Overall Time Gap Statistics (seconds) ---", f)
    log(f"Total gaps: {len(gaps):,}", f)
    log(f"Mean: {gaps.mean():.1f}", f)
    log(f"Median: {gaps.median():.1f}", f)
    log(f"Std: {gaps.std():.1f}", f)

    log(f"\n--- Percentiles ---", f)
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = gaps.quantile(pct/100)
        log(f"P{pct}: {val:.1f}s", f)

    log(f"\n--- Gap Distribution ---", f)
    log(f"{'Gap Range':>20} {'Count':>15} {'%':>10}", f)
    log(f"{'-'*20} {'-'*15} {'-'*10}", f)

    gap_bins = [
        (0, 1, "< 1 second"),
        (1, 5, "1-5 seconds"),
        (5, 10, "5-10 seconds"),
        (10, 30, "10-30 seconds"),
        (30, 60, "30-60 seconds"),
        (60, 300, "1-5 minutes"),
        (300, 3600, "5-60 minutes"),
        (3600, float('inf'), "> 1 hour"),
    ]

    for low, high, label in gap_bins:
        count = ((gaps >= low) & (gaps < high)).sum()
        pct = count / len(gaps) * 100
        log(f"{label:>20} {count:>15,} {pct:>9.1f}%", f)

    if 'PLACEMENT' in au.columns:
        log(f"\n--- Median Time Gap by Placement ---", f)
        placements = sorted(au['PLACEMENT'].dropna().unique())

        log(f"{'Placement':<12} {'Median Gap':>15} {'Mean Gap':>15}", f)
        log(f"{'-'*12} {'-'*15} {'-'*15}", f)

        for p in placements:
            p_gaps = au_sorted[(au_sorted['PLACEMENT'] == p) & (au_sorted['TIME_GAP'].notna())]['TIME_GAP']
            if len(p_gaps) > 0:
                log(f"{str(p):<12} {p_gaps.median():>14.1f}s {p_gaps.mean():>14.1f}s", f)


def session_analysis(au, f):
    """Define and analyze sessions (bursts of activity)."""
    log(f"\n{'='*80}", f)
    log(f"SESSION ANALYSIS", f)
    log(f"{'='*80}", f)
    log("Sessions defined as bursts of activity with gaps < 30 minutes", f)

    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
    au_sorted['NEW_SESSION'] = (au_sorted['TIME_GAP'] > 1800) | (au_sorted['TIME_GAP'].isna())
    au_sorted['SESSION_ID'] = au_sorted.groupby('USER_ID')['NEW_SESSION'].cumsum()

    session_sizes = au_sorted.groupby(['USER_ID', 'SESSION_ID']).size()

    log(f"\n--- Session Statistics ---", f)
    log(f"Total sessions: {len(session_sizes):,}", f)
    log(f"Mean session size: {session_sizes.mean():.2f} auctions", f)
    log(f"Median session size: {session_sizes.median():.0f} auctions", f)
    log(f"Max session size: {session_sizes.max()} auctions", f)

    log(f"\n--- Session Size Distribution ---", f)
    log(f"{'Size':>10} {'Sessions':>15} {'%':>10}", f)
    log(f"{'-'*10} {'-'*15} {'-'*10}", f)

    session_dist = session_sizes.value_counts().sort_index()
    for size in sorted(session_dist.index[:15]):
        count = session_dist[size]
        pct = count / len(session_sizes) * 100
        log(f"{size:>10} {count:>15,} {pct:>9.1f}%", f)

    sessions_per_user = au_sorted.groupby('USER_ID')['SESSION_ID'].max()

    log(f"\n--- Sessions per User ---", f)
    log(f"Mean sessions per user: {sessions_per_user.mean():.2f}", f)
    log(f"Median sessions per user: {sessions_per_user.median():.0f}", f)
    log(f"Max sessions per user: {sessions_per_user.max()}", f)


def click_behavior_by_user(au, clicks, f):
    """Analyze click behavior patterns by user."""
    log(f"\n{'='*80}", f)
    log(f"CLICK BEHAVIOR BY USER", f)
    log(f"{'='*80}", f)

    if clicks is None or len(clicks) == 0:
        log("No click data available", f)
        return

    user_col = 'USER_ID' if 'USER_ID' in clicks.columns else None
    if user_col is None:
        clicks_with_user = clicks.merge(au[['AUCTION_ID', 'USER_ID']], on='AUCTION_ID', how='left')
    else:
        clicks_with_user = clicks.copy()

    clicks_per_user = clicks_with_user.groupby('USER_ID').size()
    total_users = au['USER_ID'].nunique()
    users_with_clicks = len(clicks_per_user)

    log(f"\nTotal users: {total_users:,}", f)
    log(f"Users with clicks: {users_with_clicks:,} ({users_with_clicks/total_users*100:.1f}%)", f)
    log(f"Users without clicks: {total_users - users_with_clicks:,} ({(total_users - users_with_clicks)/total_users*100:.1f}%)", f)

    log(f"\n--- Clicks per User (users with clicks) ---", f)
    log(f"Mean: {clicks_per_user.mean():.2f}", f)
    log(f"Median: {clicks_per_user.median():.0f}", f)
    log(f"Max: {clicks_per_user.max()}", f)

    log(f"\n--- Click Count Distribution ---", f)
    log(f"{'Clicks':>10} {'Users':>15} {'%':>10}", f)
    log(f"{'-'*10} {'-'*15} {'-'*10}", f)

    click_dist = clicks_per_user.value_counts().sort_index()
    for n_clicks in sorted(click_dist.index[:15]):
        count = click_dist[n_clicks]
        pct = count / users_with_clicks * 100
        log(f"{n_clicks:>10} {count:>15,} {pct:>9.1f}%", f)


def multi_placement_user_behavior(au, f):
    """Analyze users who visit multiple placements."""
    log(f"\n{'='*80}", f)
    log(f"MULTI-PLACEMENT USER BEHAVIOR", f)
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

    multi_users = user_placements[user_placements > 1].index
    log(f"\n--- Auctions per User by Multi-Placement Status ---", f)

    auctions_per_user = au.groupby('USER_ID').size()

    single_users = user_placements[user_placements == 1].index
    single_auctions = auctions_per_user[auctions_per_user.index.isin(single_users)]
    multi_auctions = auctions_per_user[auctions_per_user.index.isin(multi_users)]

    log(f"{'Group':<20} {'Users':>12} {'Mean Aucs':>12} {'Median Aucs':>14}", f)
    log(f"{'-'*20} {'-'*12} {'-'*12} {'-'*14}", f)

    log(f"{'Single placement':<20} {len(single_auctions):>12,} {single_auctions.mean():>12.1f} {single_auctions.median():>14.0f}", f)
    log(f"{'Multi placement':<20} {len(multi_auctions):>12,} {multi_auctions.mean():>12.1f} {multi_auctions.median():>14.0f}", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='User sessions and power users EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"08_user_sessions_and_power_users_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("USER SESSIONS AND POWER USERS", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)

        paths = get_data_paths(args.round)

        log(f"\n--- Loading Data ---", f)
        au = pd.read_parquet(paths['auctions_users'])
        log(f"Auctions users: {len(au):,} rows", f)

        clicks = None
        if paths['clicks'].exists():
            clicks = pd.read_parquet(paths['clicks'])
            log(f"Clicks: {len(clicks):,} rows", f)

        user_activity_distribution(au, f)
        power_user_analysis(au, f)
        time_gap_analysis(au, f)
        session_analysis(au, f)
        click_behavior_by_user(au, clicks, f)
        multi_placement_user_behavior(au, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
