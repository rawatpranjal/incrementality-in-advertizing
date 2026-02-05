#!/usr/bin/env python3
"""
Device Detection Heuristics

Analyzes impression batch sizes as a signal for device viewport width.
Impressions with same (AUCTION_ID, OCCURRED_AT) = same viewport batch.

Batch size patterns:
  1-2 impressions = narrow viewport
  3-4 impressions = medium viewport
  5+  impressions = wide viewport

Raw statistics only, no device interpretations.

Usage:
    python 14_device_detection_heuristics.py --round round1
    python 14_device_detection_heuristics.py --round round2
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
            'auctions_users': DATA_BASE / "round1/auctions_users_all.parquet",
            'impressions': DATA_BASE / "round1/impressions_all.parquet",
            'clicks': DATA_BASE / "round1/clicks_all.parquet",
        }
    elif round_name == "round2":
        return {
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
def batch_size_distribution(impressions, f):
    """Analyze batch size distribution."""
    log(f"\n{'='*80}", f)
    log(f"BATCH SIZE DISTRIBUTION", f)
    log(f"{'='*80}", f)
    log("Batch = impressions with same (AUCTION_ID, OCCURRED_AT)", f)

    batches = impressions.groupby(['AUCTION_ID', 'OCCURRED_AT']).size().reset_index(name='batch_size')

    log(f"\nTotal batches: {len(batches):,}", f)
    log(f"Total impressions: {len(impressions):,}", f)
    log(f"Mean impressions per batch: {len(impressions) / len(batches):.2f}", f)

    log(f"\n--- Batch Size Statistics ---", f)
    log(f"Mean batch size:   {batches['batch_size'].mean():.3f}", f)
    log(f"Std batch size:    {batches['batch_size'].std():.3f}", f)
    log(f"Min batch size:    {batches['batch_size'].min()}", f)
    log(f"P25 batch size:    {batches['batch_size'].quantile(0.25):.0f}", f)
    log(f"Median batch size: {batches['batch_size'].median():.0f}", f)
    log(f"P75 batch size:    {batches['batch_size'].quantile(0.75):.0f}", f)
    log(f"P90 batch size:    {batches['batch_size'].quantile(0.90):.0f}", f)
    log(f"P95 batch size:    {batches['batch_size'].quantile(0.95):.0f}", f)
    log(f"P99 batch size:    {batches['batch_size'].quantile(0.99):.0f}", f)
    log(f"Max batch size:    {batches['batch_size'].max()}", f)

    log(f"\n--- Batch Size Frequency ---", f)
    log(f"{'Size':>8} {'Count':>15} {'%':>10} {'Cumulative %':>15}", f)
    log(f"{'-'*8} {'-'*15} {'-'*10} {'-'*15}", f)

    size_counts = batches['batch_size'].value_counts().sort_index()
    total = len(batches)
    cumsum = 0
    for size in sorted(size_counts.index[:30]):
        count = size_counts[size]
        pct = count / total * 100
        cumsum += pct
        log(f"{size:>8} {count:>15,} {pct:>9.2f}% {cumsum:>14.2f}%", f)

    log(f"\n--- Batch Size Buckets ---", f)
    size_1_2 = len(batches[batches['batch_size'] <= 2])
    size_3_4 = len(batches[(batches['batch_size'] >= 3) & (batches['batch_size'] <= 4)])
    size_5_plus = len(batches[batches['batch_size'] >= 5])

    log(f"Batch size 1-2:  {size_1_2:>12,} ({size_1_2/total*100:>6.2f}%)", f)
    log(f"Batch size 3-4:  {size_3_4:>12,} ({size_3_4/total*100:>6.2f}%)", f)
    log(f"Batch size 5+:   {size_5_plus:>12,} ({size_5_plus/total*100:>6.2f}%)", f)

    return batches


def batch_size_by_placement(impressions, au, f):
    """Batch size distribution by placement."""
    log(f"\n{'='*80}", f)
    log(f"BATCH SIZE BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if 'PLACEMENT' not in au.columns:
        log("PLACEMENT column not available", f)
        return

    imp_with_placement = impressions.merge(
        au[['AUCTION_ID', 'PLACEMENT']].drop_duplicates(),
        on='AUCTION_ID',
        how='left'
    )

    batches = imp_with_placement.groupby(['AUCTION_ID', 'OCCURRED_AT', 'PLACEMENT']).size().reset_index(name='batch_size')

    placements = sorted(batches['PLACEMENT'].dropna().unique())

    log(f"\n{'Placement':>12} {'N Batches':>12} {'Mean':>10} {'Median':>10} {'P25':>10} {'P75':>10} {'P95':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", f)

    for p in placements:
        p_batches = batches[batches['PLACEMENT'] == p]
        if len(p_batches) == 0:
            continue
        log(f"{p:>12} {len(p_batches):>12,} {p_batches['batch_size'].mean():>10.2f} "
            f"{p_batches['batch_size'].median():>10.0f} {p_batches['batch_size'].quantile(0.25):>10.0f} "
            f"{p_batches['batch_size'].quantile(0.75):>10.0f} {p_batches['batch_size'].quantile(0.95):>10.0f}", f)

    log(f"\n--- Batch Size Buckets by Placement ---", f)
    log(f"{'Placement':>12} {'1-2':>15} {'3-4':>15} {'5+':>15}", f)
    log(f"{'-'*12} {'-'*15} {'-'*15} {'-'*15}", f)

    for p in placements:
        p_batches = batches[batches['PLACEMENT'] == p]
        if len(p_batches) == 0:
            continue
        total = len(p_batches)
        s_1_2 = len(p_batches[p_batches['batch_size'] <= 2])
        s_3_4 = len(p_batches[(p_batches['batch_size'] >= 3) & (p_batches['batch_size'] <= 4)])
        s_5_plus = len(p_batches[p_batches['batch_size'] >= 5])
        log(f"{p:>12} {s_1_2:>8,} ({s_1_2/total*100:>5.1f}%) "
            f"{s_3_4:>8,} ({s_3_4/total*100:>5.1f}%) "
            f"{s_5_plus:>8,} ({s_5_plus/total*100:>5.1f}%)", f)


def scroll_gap_timing(impressions, f):
    """Analyze time gaps between scroll events (consecutive batches)."""
    log(f"\n{'='*80}", f)
    log(f"SCROLL GAP TIMING", f)
    log(f"{'='*80}", f)
    log("Time between consecutive batches within an auction", f)

    batches = impressions.groupby(['AUCTION_ID', 'OCCURRED_AT']).size().reset_index(name='batch_size')
    batches = batches.sort_values(['AUCTION_ID', 'OCCURRED_AT'])

    auctions_with_multiple_batches = batches.groupby('AUCTION_ID').size()
    auctions_with_multiple_batches = auctions_with_multiple_batches[auctions_with_multiple_batches > 1].index

    log(f"\nAuctions with 1 batch: {len(batches['AUCTION_ID'].unique()) - len(auctions_with_multiple_batches):,}", f)
    log(f"Auctions with 2+ batches: {len(auctions_with_multiple_batches):,}", f)

    multi_batch = batches[batches['AUCTION_ID'].isin(auctions_with_multiple_batches)].copy()
    multi_batch['prev_time'] = multi_batch.groupby('AUCTION_ID')['OCCURRED_AT'].shift(1)
    multi_batch['gap_seconds'] = (multi_batch['OCCURRED_AT'] - multi_batch['prev_time']).dt.total_seconds()

    gaps = multi_batch['gap_seconds'].dropna()

    if len(gaps) == 0:
        log("No scroll gaps to analyze", f)
        return

    log(f"\n--- Scroll Gap Statistics ---", f)
    log(f"Total scroll gaps: {len(gaps):,}", f)
    log(f"Mean gap:   {gaps.mean():.3f} seconds", f)
    log(f"Std gap:    {gaps.std():.3f} seconds", f)
    log(f"Min gap:    {gaps.min():.3f} seconds", f)
    log(f"P25 gap:    {gaps.quantile(0.25):.3f} seconds", f)
    log(f"Median gap: {gaps.median():.3f} seconds", f)
    log(f"P75 gap:    {gaps.quantile(0.75):.3f} seconds", f)
    log(f"P90 gap:    {gaps.quantile(0.90):.3f} seconds", f)
    log(f"P95 gap:    {gaps.quantile(0.95):.3f} seconds", f)
    log(f"P99 gap:    {gaps.quantile(0.99):.3f} seconds", f)
    log(f"Max gap:    {gaps.max():.3f} seconds", f)

    log(f"\n--- Scroll Gap Distribution ---", f)
    log(f"Gap < 1s:   {(gaps < 1).sum():>12,} ({(gaps < 1).mean()*100:>6.2f}%)", f)
    log(f"Gap 1-5s:   {((gaps >= 1) & (gaps < 5)).sum():>12,} ({((gaps >= 1) & (gaps < 5)).mean()*100:>6.2f}%)", f)
    log(f"Gap 5-10s:  {((gaps >= 5) & (gaps < 10)).sum():>12,} ({((gaps >= 5) & (gaps < 10)).mean()*100:>6.2f}%)", f)
    log(f"Gap 10-30s: {((gaps >= 10) & (gaps < 30)).sum():>12,} ({((gaps >= 10) & (gaps < 30)).mean()*100:>6.2f}%)", f)
    log(f"Gap 30-60s: {((gaps >= 30) & (gaps < 60)).sum():>12,} ({((gaps >= 30) & (gaps < 60)).mean()*100:>6.2f}%)", f)
    log(f"Gap 60s+:   {(gaps >= 60).sum():>12,} ({(gaps >= 60).mean()*100:>6.2f}%)", f)


def session_level_aggregates(impressions, au, f):
    """Session-level batch statistics."""
    log(f"\n{'='*80}", f)
    log(f"SESSION-LEVEL AGGREGATES", f)
    log(f"{'='*80}", f)
    log("Aggregating batch metrics per user session (USER_ID)", f)

    # Use USER_ID from impressions directly if available
    if 'USER_ID' in impressions.columns:
        user_col = 'USER_ID'
    elif 'OPAQUE_USER_ID' in au.columns:
        # Need to merge from au
        user_col = 'OPAQUE_USER_ID'
        impressions = impressions.merge(
            au[['AUCTION_ID', user_col]].drop_duplicates(),
            on='AUCTION_ID',
            how='left'
        )
    else:
        log(f"User column not found", f)
        return

    batches = impressions.groupby(['AUCTION_ID', 'OCCURRED_AT']).agg({
        user_col: 'first',
        'PRODUCT_ID': 'count'
    }).reset_index()
    batches.columns = ['AUCTION_ID', 'OCCURRED_AT', user_col, 'batch_size']

    session_stats = batches.groupby(user_col).agg({
        'batch_size': ['mean', 'std', 'min', 'max', 'count'],
        'AUCTION_ID': 'nunique'
    })
    session_stats.columns = ['mean_batch_size', 'std_batch_size', 'min_batch_size', 'max_batch_size', 'n_batches', 'n_auctions']
    session_stats = session_stats.reset_index()

    log(f"\nTotal users: {len(session_stats):,}", f)
    log(f"Total batches: {session_stats['n_batches'].sum():,}", f)

    log(f"\n--- Mean Batch Size per User ---", f)
    log(f"Mean of user means:   {session_stats['mean_batch_size'].mean():.3f}", f)
    log(f"Std of user means:    {session_stats['mean_batch_size'].std():.3f}", f)
    log(f"Median of user means: {session_stats['mean_batch_size'].median():.3f}", f)
    log(f"P25 of user means:    {session_stats['mean_batch_size'].quantile(0.25):.3f}", f)
    log(f"P75 of user means:    {session_stats['mean_batch_size'].quantile(0.75):.3f}", f)

    log(f"\n--- Scroll Events per User ---", f)
    log(f"Mean batches per user:   {session_stats['n_batches'].mean():.2f}", f)
    log(f"Median batches per user: {session_stats['n_batches'].median():.0f}", f)
    log(f"P25 batches per user:    {session_stats['n_batches'].quantile(0.25):.0f}", f)
    log(f"P75 batches per user:    {session_stats['n_batches'].quantile(0.75):.0f}", f)
    log(f"P95 batches per user:    {session_stats['n_batches'].quantile(0.95):.0f}", f)
    log(f"Max batches per user:    {session_stats['n_batches'].max():.0f}", f)

    log(f"\n--- User Batch Consistency ---", f)
    consistent_users = session_stats[session_stats['std_batch_size'].fillna(0) == 0]
    log(f"Users with constant batch size: {len(consistent_users):,} ({len(consistent_users)/len(session_stats)*100:.1f}%)", f)

    variable_users = session_stats[session_stats['std_batch_size'].fillna(0) > 0]
    if len(variable_users) > 0:
        log(f"Users with variable batch size: {len(variable_users):,} ({len(variable_users)/len(session_stats)*100:.1f}%)", f)
        log(f"  Mean std of batch size: {variable_users['std_batch_size'].mean():.3f}", f)


def batch_size_ctr(impressions, clicks, f):
    """CTR by batch size."""
    log(f"\n{'='*80}", f)
    log(f"CTR BY BATCH SIZE", f)
    log(f"{'='*80}", f)

    if clicks is None or len(clicks) == 0:
        log("Click data not available", f)
        return

    batches = impressions.groupby(['AUCTION_ID', 'OCCURRED_AT']).size().reset_index(name='batch_size')

    imp_with_batch = impressions.merge(batches, on=['AUCTION_ID', 'OCCURRED_AT'], how='left')

    click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
    imp_with_batch['clicked'] = imp_with_batch.apply(
        lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
    )

    ctr_by_batch = imp_with_batch.groupby('batch_size').agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    ctr_by_batch.columns = ['batch_size', 'clicks', 'impressions', 'CTR']

    log(f"\n--- CTR by Batch Size ---", f)
    log(f"{'Batch Size':>12} {'Impressions':>15} {'Clicks':>12} {'CTR %':>10}", f)
    log(f"{'-'*12} {'-'*15} {'-'*12} {'-'*10}", f)

    for _, row in ctr_by_batch.head(20).iterrows():
        log(f"{int(row['batch_size']):>12} {int(row['impressions']):>15,} {int(row['clicks']):>12,} {row['CTR']*100:>9.3f}%", f)

    log(f"\n--- CTR by Batch Size Bucket ---", f)
    imp_with_batch['batch_bucket'] = pd.cut(
        imp_with_batch['batch_size'],
        bins=[0, 2, 4, float('inf')],
        labels=['1-2', '3-4', '5+']
    )

    ctr_by_bucket = imp_with_batch.groupby('batch_bucket').agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    ctr_by_bucket.columns = ['batch_bucket', 'clicks', 'impressions', 'CTR']

    log(f"{'Bucket':>12} {'Impressions':>15} {'Clicks':>12} {'CTR %':>10}", f)
    log(f"{'-'*12} {'-'*15} {'-'*12} {'-'*10}", f)

    for _, row in ctr_by_bucket.iterrows():
        log(f"{row['batch_bucket']:>12} {int(row['impressions']):>15,} {int(row['clicks']):>12,} {row['CTR']*100:>9.3f}%", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Device detection heuristics via batch size analysis')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"14_device_detection_heuristics_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("DEVICE DETECTION HEURISTICS", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)
        log("", f)
        log("Core signal: Batch size = impressions with same (AUCTION_ID, OCCURRED_AT)", f)
        log("Hypothesis: Batch size correlates with viewport width", f)

        paths = get_data_paths(args.round)

        log(f"\n--- Loading Data ---", f)

        if not paths['impressions'].exists():
            log(f"ERROR: Impressions file not found: {paths['impressions']}", f)
            return

        impressions = pd.read_parquet(paths['impressions'])
        log(f"Impressions: {len(impressions):,} rows", f)

        au = pd.read_parquet(paths['auctions_users'])
        log(f"Auctions users: {len(au):,} rows", f)

        clicks = None
        if paths['clicks'].exists():
            clicks = pd.read_parquet(paths['clicks'])
            log(f"Clicks: {len(clicks):,} rows", f)

        batch_size_distribution(impressions, f)
        batch_size_by_placement(impressions, au, f)
        scroll_gap_timing(impressions, f)
        session_level_aggregates(impressions, au, f)
        batch_size_ctr(impressions, clicks, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
