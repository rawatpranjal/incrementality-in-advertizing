#!/usr/bin/env python3
"""
Three-Way Placement Comparison: P1 vs P2 vs P3

Compares placement types across key dimensions:
- P1 (Browse/Feed): 108K rows
- P2 (Category/Collection): 61K rows
- P3 (Search Pagination): 48K rows

Output: results/12_p1_vs_p3_comparison.txt
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR_R1 = BASE_DIR / "0_data" / "round1"
DATA_DIR_R2 = BASE_DIR / "0_data" / "round2"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "12_p1_vs_p3_comparison.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# SECTION 1: DATA LOADING & MERGE
# =============================================================================
def load_data(f):
    """Load and filter data to P1, P2, P3."""
    log(f"\n{'='*80}", f)
    log("SECTION 1: DATA LOADING & MERGE", f)
    log(f"{'='*80}", f)

    # Load session_items (round1)
    session_items_path = DATA_DIR_R1 / "session_items.parquet"
    log(f"\nLoading: {session_items_path}", f)
    si = pd.read_parquet(session_items_path)
    log(f"  Loaded session_items: {len(si):,} rows", f)
    log(f"  Columns: {si.columns.tolist()}", f)

    # Load auctions_users (round2) for user IDs and timestamps
    au_path = DATA_DIR_R2 / "auctions_users_r2.parquet"
    log(f"\nLoading: {au_path}", f)
    au = pd.read_parquet(au_path)
    log(f"  Loaded auctions_users: {len(au):,} rows", f)

    # Load catalog (round2) for product info
    catalog_path = DATA_DIR_R2 / "catalog_r2.parquet"
    log(f"\nLoading: {catalog_path}", f)
    catalog = pd.read_parquet(catalog_path)
    log(f"  Loaded catalog: {len(catalog):,} rows", f)

    # Placement distribution before filtering
    log(f"\n--- Placement Distribution (All) ---", f)
    log(f"{'Placement':<12} {'Count':>12} {'Percentage':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12}", f)
    for p, count in si['placement'].value_counts().sort_index().items():
        pct = count / len(si) * 100
        log(f"{p:<12} {count:>12,} {pct:>11.1f}%", f)

    # Filter to P1, P2, P3 (placement is string type)
    log(f"\nFiltering to placements 1, 2, 3...", f)
    si_filtered = si[si['placement'].isin(['1', '2', '3'])].copy()
    # Convert placement to int for easier comparison
    si_filtered['placement'] = si_filtered['placement'].astype(int)
    log(f"  Filtered session_items: {len(si_filtered):,} rows", f)

    # Split by placement
    p1 = si_filtered[si_filtered['placement'] == 1].copy()
    p2 = si_filtered[si_filtered['placement'] == 2].copy()
    p3 = si_filtered[si_filtered['placement'] == 3].copy()

    log(f"\n--- Filtered Placement Counts ---", f)
    log(f"  P1 (Browse/Feed): {len(p1):,} rows", f)
    log(f"  P2 (Category/Collection): {len(p2):,} rows", f)
    log(f"  P3 (Search Pagination): {len(p3):,} rows", f)

    # Merge with auctions_users to get user IDs and timestamps
    log(f"\nMerging with auctions_users for user IDs...", f)

    # Lowercase auction_id in auctions_users to match session_items
    au['auction_id_lower'] = au['AUCTION_ID'].str.lower()

    si_merged = si_filtered.merge(
        au[['auction_id_lower', 'USER_ID', 'CREATED_AT']],
        left_on='auction_id',
        right_on='auction_id_lower',
        how='left'
    )

    merge_rate = si_merged['USER_ID'].notna().sum() / len(si_merged) * 100
    log(f"  Merge success rate: {merge_rate:.1f}%", f)
    log(f"  Rows with user ID: {si_merged['USER_ID'].notna().sum():,}", f)

    # Merge with catalog for product info
    log(f"\nMerging with catalog for product info...", f)
    si_with_catalog = si_merged.merge(
        catalog[['PRODUCT_ID', 'NAME', 'CATEGORIES', 'DESCRIPTION']],
        left_on='product_id',
        right_on='PRODUCT_ID',
        how='left'
    )

    catalog_merge_rate = si_with_catalog['NAME'].notna().sum() / len(si_with_catalog) * 100
    log(f"  Catalog merge rate: {catalog_merge_rate:.1f}%", f)
    log(f"  Rows with product name: {si_with_catalog['NAME'].notna().sum():,}", f)

    return si_with_catalog, p1, p2, p3, au, catalog

# =============================================================================
# SECTION 2: AGGREGATE STATISTICS
# =============================================================================
def aggregate_statistics(si, f):
    """Compute aggregate statistics by placement."""
    log(f"\n{'='*80}", f)
    log("SECTION 2: AGGREGATE STATISTICS", f)
    log(f"{'='*80}", f)

    placements = [1, 2, 3]

    # Basic counts
    log(f"\n--- Basic Counts ---", f)
    log(f"{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
    log(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}", f)

    for p in placements:
        pass  # Build stats dict

    stats = {}
    for p in placements:
        subset = si[si['placement'] == p]
        stats[p] = {
            'rows': len(subset),
            'sessions': subset['auction_id'].nunique(),
            'products': subset['product_id'].nunique(),
            'clicks': subset['clicked'].sum(),
            'users': subset['USER_ID'].nunique() if 'USER_ID' in subset.columns else 0,
        }

    log(f"{'Total rows':<30} {stats[1]['rows']:>15,} {stats[2]['rows']:>15,} {stats[3]['rows']:>15,}", f)
    log(f"{'Unique sessions':<30} {stats[1]['sessions']:>15,} {stats[2]['sessions']:>15,} {stats[3]['sessions']:>15,}", f)
    log(f"{'Unique products':<30} {stats[1]['products']:>15,} {stats[2]['products']:>15,} {stats[3]['products']:>15,}", f)
    log(f"{'Total clicks':<30} {stats[1]['clicks']:>15,} {stats[2]['clicks']:>15,} {stats[3]['clicks']:>15,}", f)
    log(f"{'Unique users':<30} {stats[1]['users']:>15,} {stats[2]['users']:>15,} {stats[3]['users']:>15,}", f)

    # Derived metrics
    log(f"\n--- Derived Metrics ---", f)
    log(f"{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
    log(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}", f)

    for p in placements:
        stats[p]['items_per_session'] = stats[p]['rows'] / stats[p]['sessions'] if stats[p]['sessions'] > 0 else 0
        stats[p]['ctr'] = stats[p]['clicks'] / stats[p]['rows'] * 100 if stats[p]['rows'] > 0 else 0
        stats[p]['sessions_per_user'] = stats[p]['sessions'] / stats[p]['users'] if stats[p]['users'] > 0 else 0

    log(f"{'Items per session (mean)':<30} {stats[1]['items_per_session']:>15.2f} {stats[2]['items_per_session']:>15.2f} {stats[3]['items_per_session']:>15.2f}", f)
    log(f"{'CTR (%)':<30} {stats[1]['ctr']:>14.2f}% {stats[2]['ctr']:>14.2f}% {stats[3]['ctr']:>14.2f}%", f)
    log(f"{'Sessions per user (mean)':<30} {stats[1]['sessions_per_user']:>15.2f} {stats[2]['sessions_per_user']:>15.2f} {stats[3]['sessions_per_user']:>15.2f}", f)

    # Session-level stats
    log(f"\n--- Session-Level Statistics ---", f)

    for p in placements:
        subset = si[si['placement'] == p]
        session_stats = subset.groupby('auction_id').agg({
            'clicked': 'sum',
            'position': 'max',
            'product_id': 'count'
        }).rename(columns={'product_id': 'n_items', 'clicked': 'n_clicks'})
        stats[p]['session_df'] = session_stats

    log(f"\n{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
    log(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}", f)

    log(f"{'Items per session (median)':<30} {stats[1]['session_df']['n_items'].median():>15.1f} {stats[2]['session_df']['n_items'].median():>15.1f} {stats[3]['session_df']['n_items'].median():>15.1f}", f)
    log(f"{'Items per session (max)':<30} {stats[1]['session_df']['n_items'].max():>15} {stats[2]['session_df']['n_items'].max():>15} {stats[3]['session_df']['n_items'].max():>15}", f)
    log(f"{'Max position shown (max)':<30} {stats[1]['session_df']['position'].max():>15} {stats[2]['session_df']['position'].max():>15} {stats[3]['session_df']['position'].max():>15}", f)
    log(f"{'Sessions with clicks (%)':<30} {(stats[1]['session_df']['n_clicks'] > 0).mean()*100:>14.1f}% {(stats[2]['session_df']['n_clicks'] > 0).mean()*100:>14.1f}% {(stats[3]['session_df']['n_clicks'] > 0).mean()*100:>14.1f}%", f)

    return stats

# =============================================================================
# SECTION 3: POSITION/RANK DISTRIBUTIONS
# =============================================================================
def position_rank_distributions(si, f):
    """Position and rank distributions by placement."""
    log(f"\n{'='*80}", f)
    log("SECTION 3: POSITION/RANK DISTRIBUTIONS", f)
    log(f"{'='*80}", f)

    placements = [1, 2, 3]

    # Position distribution
    log(f"\n--- Position Distribution ---", f)
    log(f"{'Position':<12} {'P1 Count':>12} {'P1 %':>8} {'P2 Count':>12} {'P2 %':>8} {'P3 Count':>12} {'P3 %':>8}", f)
    log(f"{'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8}", f)

    max_pos = 15
    for pos in range(1, max_pos + 1):
        row = f"{pos:<12}"
        for p in placements:
            subset = si[si['placement'] == p]
            count = (subset['position'] == pos).sum()
            pct = count / len(subset) * 100 if len(subset) > 0 else 0
            row += f" {count:>12,} {pct:>7.1f}%"
        log(row, f)

    # Rank distribution
    log(f"\n--- Rank Distribution ---", f)
    log(f"{'Rank':<12} {'P1 Count':>12} {'P1 %':>8} {'P2 Count':>12} {'P2 %':>8} {'P3 Count':>12} {'P3 %':>8}", f)
    log(f"{'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8}", f)

    max_rank = 15
    for rank in range(1, max_rank + 1):
        row = f"{rank:<12}"
        for p in placements:
            subset = si[si['placement'] == p]
            count = (subset['rank'] == rank).sum()
            pct = count / len(subset) * 100 if len(subset) > 0 else 0
            row += f" {count:>12,} {pct:>7.1f}%"
        log(row, f)

    # CTR by position
    log(f"\n--- CTR by Position ---", f)
    log(f"{'Position':<12} {'P1 CTR':>12} {'P2 CTR':>12} {'P3 CTR':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

    for pos in range(1, max_pos + 1):
        row = f"{pos:<12}"
        for p in placements:
            subset = si[(si['placement'] == p) & (si['position'] == pos)]
            if len(subset) > 0:
                ctr = subset['clicked'].mean() * 100
                row += f" {ctr:>11.2f}%"
            else:
                row += f" {'N/A':>12}"
        log(row, f)

    # CTR by rank
    log(f"\n--- CTR by Rank ---", f)
    log(f"{'Rank':<12} {'P1 CTR':>12} {'P2 CTR':>12} {'P3 CTR':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

    for rank in range(1, max_rank + 1):
        row = f"{rank:<12}"
        for p in placements:
            subset = si[(si['placement'] == p) & (si['rank'] == rank)]
            if len(subset) > 0:
                ctr = subset['clicked'].mean() * 100
                row += f" {ctr:>11.2f}%"
            else:
                row += f" {'N/A':>12}"
        log(row, f)

# =============================================================================
# SECTION 4: TEMPORAL PATTERNS
# =============================================================================
def temporal_patterns(si, f):
    """Time gap distributions by placement."""
    log(f"\n{'='*80}", f)
    log("SECTION 4: TEMPORAL PATTERNS", f)
    log(f"{'='*80}", f)

    placements = [1, 2, 3]

    # Check if we have timestamps
    if 'CREATED_AT' not in si.columns or si['CREATED_AT'].isna().all():
        log(f"\nNo timestamp data available for temporal analysis.", f)
        return

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(si['CREATED_AT']):
        si['CREATED_AT'] = pd.to_datetime(si['CREATED_AT'])

    # Compute time gaps between sessions for same user
    log(f"\n--- Time Gap Between Sessions (Same User) ---", f)

    # Get session-level data with timestamps
    sessions = si.groupby(['auction_id', 'placement', 'USER_ID']).agg({
        'CREATED_AT': 'first'
    }).reset_index()

    sessions = sessions.sort_values(['USER_ID', 'CREATED_AT'])
    sessions['TIME_GAP'] = sessions.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()

    log(f"\n{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
    log(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}", f)

    gap_stats = {}
    for p in placements:
        gaps = sessions[(sessions['placement'] == p) & (sessions['TIME_GAP'].notna())]['TIME_GAP']
        gap_stats[p] = gaps

    if all(len(gap_stats[p]) > 0 for p in placements):
        log(f"{'Gap count':<30} {len(gap_stats[1]):>15,} {len(gap_stats[2]):>15,} {len(gap_stats[3]):>15,}", f)
        log(f"{'Mean gap (seconds)':<30} {gap_stats[1].mean():>15.1f} {gap_stats[2].mean():>15.1f} {gap_stats[3].mean():>15.1f}", f)
        log(f"{'Median gap (seconds)':<30} {gap_stats[1].median():>15.1f} {gap_stats[2].median():>15.1f} {gap_stats[3].median():>15.1f}", f)
        log(f"{'P25 gap (seconds)':<30} {gap_stats[1].quantile(0.25):>15.1f} {gap_stats[2].quantile(0.25):>15.1f} {gap_stats[3].quantile(0.25):>15.1f}", f)
        log(f"{'P75 gap (seconds)':<30} {gap_stats[1].quantile(0.75):>15.1f} {gap_stats[2].quantile(0.75):>15.1f} {gap_stats[3].quantile(0.75):>15.1f}", f)
        log(f"{'P95 gap (seconds)':<30} {gap_stats[1].quantile(0.95):>15.1f} {gap_stats[2].quantile(0.95):>15.1f} {gap_stats[3].quantile(0.95):>15.1f}", f)

    # Gap bucket distribution
    log(f"\n--- Gap Bucket Distribution ---", f)
    log(f"{'Gap Bucket':<20} {'P1':>12} {'P2':>12} {'P3':>12}", f)
    log(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}", f)

    buckets = [
        ('<5 seconds', 0, 5),
        ('5-30 seconds', 5, 30),
        ('30-60 seconds', 30, 60),
        ('1-5 minutes', 60, 300),
        ('5-30 minutes', 300, 1800),
        ('30+ minutes', 1800, float('inf'))
    ]

    for bucket_name, low, high in buckets:
        row = f"{bucket_name:<20}"
        for p in placements:
            gaps = gap_stats[p]
            if len(gaps) > 0:
                count = ((gaps >= low) & (gaps < high)).sum()
                pct = count / len(gaps) * 100
                row += f" {pct:>11.1f}%"
            else:
                row += f" {'N/A':>12}"
        log(row, f)

# =============================================================================
# SECTION 5: USER BEHAVIOR
# =============================================================================
def user_behavior(si, f):
    """User overlap and behavior across placements."""
    log(f"\n{'='*80}", f)
    log("SECTION 5: USER BEHAVIOR", f)
    log(f"{'='*80}", f)

    placements = [1, 2, 3]

    # Check if we have user data
    if 'USER_ID' not in si.columns or si['USER_ID'].isna().all():
        log(f"\nNo user data available for user behavior analysis.", f)
        return

    # Filter to rows with user IDs
    si_users = si[si['USER_ID'].notna()].copy()

    # User counts by placement
    log(f"\n--- User Statistics by Placement ---", f)
    log(f"{'Metric':<35} {'P1':>15} {'P2':>15} {'P3':>15}", f)
    log(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15}", f)

    user_stats = {}
    for p in placements:
        subset = si_users[si_users['placement'] == p]
        users = subset['USER_ID'].unique()
        user_stats[p] = {
            'users': set(users),
            'n_users': len(users),
            'sessions': subset['auction_id'].nunique(),
            'items': len(subset),
        }
        user_stats[p]['sessions_per_user'] = user_stats[p]['sessions'] / user_stats[p]['n_users'] if user_stats[p]['n_users'] > 0 else 0
        user_stats[p]['items_per_user'] = user_stats[p]['items'] / user_stats[p]['n_users'] if user_stats[p]['n_users'] > 0 else 0

    log(f"{'Unique users':<35} {user_stats[1]['n_users']:>15,} {user_stats[2]['n_users']:>15,} {user_stats[3]['n_users']:>15,}", f)
    log(f"{'Sessions per user (mean)':<35} {user_stats[1]['sessions_per_user']:>15.2f} {user_stats[2]['sessions_per_user']:>15.2f} {user_stats[3]['sessions_per_user']:>15.2f}", f)
    log(f"{'Items seen per user (mean)':<35} {user_stats[1]['items_per_user']:>15.2f} {user_stats[2]['items_per_user']:>15.2f} {user_stats[3]['items_per_user']:>15.2f}", f)

    # User overlap analysis
    log(f"\n--- User Overlap Analysis ---", f)

    p1_users = user_stats[1]['users']
    p2_users = user_stats[2]['users']
    p3_users = user_stats[3]['users']

    # Pairwise overlaps
    p1_p2 = p1_users & p2_users
    p1_p3 = p1_users & p3_users
    p2_p3 = p2_users & p3_users

    # Three-way overlap
    all_three = p1_users & p2_users & p3_users

    # Users in any placement
    any_user = p1_users | p2_users | p3_users

    log(f"\nPairwise overlaps:", f)
    log(f"  P1 ∩ P2: {len(p1_p2):,} users ({len(p1_p2)/len(any_user)*100:.1f}% of all users)", f)
    log(f"  P1 ∩ P3: {len(p1_p3):,} users ({len(p1_p3)/len(any_user)*100:.1f}% of all users)", f)
    log(f"  P2 ∩ P3: {len(p2_p3):,} users ({len(p2_p3)/len(any_user)*100:.1f}% of all users)", f)

    log(f"\nThree-way overlap:", f)
    log(f"  P1 ∩ P2 ∩ P3: {len(all_three):,} users ({len(all_three)/len(any_user)*100:.1f}% of all users)", f)

    log(f"\nExclusive users (only in one placement):", f)
    p1_only = p1_users - p2_users - p3_users
    p2_only = p2_users - p1_users - p3_users
    p3_only = p3_users - p1_users - p2_users
    log(f"  P1 only: {len(p1_only):,} users ({len(p1_only)/len(any_user)*100:.1f}% of all users)", f)
    log(f"  P2 only: {len(p2_only):,} users ({len(p2_only)/len(any_user)*100:.1f}% of all users)", f)
    log(f"  P3 only: {len(p3_only):,} users ({len(p3_only)/len(any_user)*100:.1f}% of all users)", f)

    log(f"\nTotal unique users across all placements: {len(any_user):,}", f)

    # User placement combinations
    log(f"\n--- User Placement Combinations ---", f)

    user_placement_combos = si_users.groupby('USER_ID')['placement'].apply(lambda x: frozenset(x.unique())).value_counts()

    log(f"{'Placements':<25} {'Users':>12} {'Percentage':>12}", f)
    log(f"{'-'*25} {'-'*12} {'-'*12}", f)

    for combo, count in user_placement_combos.head(10).items():
        combo_str = ', '.join(map(str, sorted(combo)))
        pct = count / len(any_user) * 100
        log(f"P{combo_str.replace(', ', ', P'):<23} {count:>12,} {pct:>11.1f}%", f)

# =============================================================================
# SECTION 6: RAW SESSION LOGS
# =============================================================================
def raw_session_logs(si, f):
    """Raw session examples for each placement."""
    log(f"\n{'='*80}", f)
    log("SECTION 6: RAW SESSION LOGS", f)
    log(f"{'='*80}", f)

    placements = [1, 2, 3]
    placement_names = {1: 'P1 (Browse/Feed)', 2: 'P2 (Category/Collection)', 3: 'P3 (Search Pagination)'}

    for p in placements:
        log(f"\n{'-'*80}", f)
        log(f"{placement_names[p]} - SAMPLE SESSIONS", f)
        log(f"{'-'*80}", f)

        subset = si[si['placement'] == p]

        # Sessions with clicks
        sessions_with_clicks = subset[subset['clicked'] == 1]['auction_id'].unique()

        log(f"\n--- Sessions WITH Clicks (10 examples) ---", f)
        for i, session_id in enumerate(sessions_with_clicks[:10]):
            session_data = subset[subset['auction_id'] == session_id].sort_values('position')
            log(f"\nSession {i+1}: {session_id}", f)
            log(f"{'Pos':>4} {'Rank':>6} {'Clicked':>8} {'Quality':>10} {'Bid':>8} {'Product ID':<26}", f)
            log(f"{'-'*4} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*26}", f)
            for _, row in session_data.iterrows():
                click_marker = '***' if row['clicked'] == 1 else ''
                log(f"{row['position']:>4} {row['rank']:>6} {row['clicked']:>8} {row['quality']:>10.4f} {row['bid']:>8} {row['product_id']:<26} {click_marker}", f)

        # Sessions without clicks
        sessions_no_clicks = subset.groupby('auction_id')['clicked'].sum()
        sessions_no_clicks = sessions_no_clicks[sessions_no_clicks == 0].index

        log(f"\n--- Sessions WITHOUT Clicks (10 examples) ---", f)
        for i, session_id in enumerate(sessions_no_clicks[:10]):
            session_data = subset[subset['auction_id'] == session_id].sort_values('position')
            log(f"\nSession {i+1}: {session_id}", f)
            log(f"{'Pos':>4} {'Rank':>6} {'Clicked':>8} {'Quality':>10} {'Bid':>8} {'Product ID':<26}", f)
            log(f"{'-'*4} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*26}", f)
            for _, row in session_data.iterrows():
                log(f"{row['position']:>4} {row['rank']:>6} {row['clicked']:>8} {row['quality']:>10.4f} {row['bid']:>8} {row['product_id']:<26}", f)

    # Same-user comparison across all three placements
    log(f"\n{'-'*80}", f)
    log(f"SAME-USER COMPARISON ACROSS P1, P2, P3", f)
    log(f"{'-'*80}", f)

    if 'USER_ID' in si.columns and si['USER_ID'].notna().any():
        # Find users with sessions in all three placements
        user_placements = si[si['USER_ID'].notna()].groupby('USER_ID')['placement'].apply(lambda x: set(x.unique()))
        users_all_three = user_placements[user_placements.apply(lambda x: {1, 2, 3}.issubset(x))].index

        log(f"\nUsers with sessions in all three placements: {len(users_all_three):,}", f)

        for i, user_id in enumerate(users_all_three[:5]):
            log(f"\n{'='*60}", f)
            log(f"User {i+1}: {user_id[:50]}...", f)
            log(f"{'='*60}", f)

            user_data = si[si['USER_ID'] == user_id]

            for p in placements:
                p_data = user_data[user_data['placement'] == p]
                n_sessions = p_data['auction_id'].nunique()
                n_items = len(p_data)
                n_clicks = p_data['clicked'].sum()

                log(f"\n{placement_names[p]}: {n_sessions} sessions, {n_items} items, {n_clicks} clicks", f)

                # Show first session
                first_session = p_data['auction_id'].iloc[0] if len(p_data) > 0 else None
                if first_session:
                    session_data = p_data[p_data['auction_id'] == first_session].sort_values('position')
                    log(f"  Sample session: {first_session}", f)
                    log(f"  {'Pos':>4} {'Rank':>6} {'Clicked':>8} {'Product ID':<26}", f)
                    for _, row in session_data.head(5).iterrows():
                        click_marker = '***' if row['clicked'] == 1 else ''
                        log(f"  {row['position']:>4} {row['rank']:>6} {row['clicked']:>8} {row['product_id']:<26} {click_marker}", f)
                    if len(session_data) > 5:
                        log(f"  ... ({len(session_data) - 5} more items)", f)

# =============================================================================
# SECTION 7: PRODUCT TYPE ANALYSIS
# =============================================================================
def product_type_analysis(si, f):
    """Bag-of-words and category analysis by placement."""
    log(f"\n{'='*80}", f)
    log("SECTION 7: PRODUCT TYPE ANALYSIS", f)
    log(f"{'='*80}", f)

    placements = [1, 2, 3]
    placement_names = {1: 'P1', 2: 'P2', 3: 'P3'}

    # Check if we have product names
    if 'NAME' not in si.columns or si['NAME'].isna().all():
        log(f"\nNo product name data available for product type analysis.", f)
        return

    # Tokenize product names
    log(f"\n--- Bag-of-Words on Product Names ---", f)

    def tokenize(text):
        if pd.isna(text):
            return []
        # Lowercase, remove punctuation, split on whitespace
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Filter short tokens and numbers
        tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
        return tokens

    # Count words by placement
    word_counts = {}
    for p in placements:
        subset = si[si['placement'] == p]
        all_words = []
        for name in tqdm(subset['NAME'].dropna(), desc=f"Tokenizing P{p}"):
            all_words.extend(tokenize(name))
        word_counts[p] = Counter(all_words)

    # Get top 30 words for each placement
    log(f"\n--- Top 30 Words in Product Names ---", f)
    log(f"{'Rank':<6} {'P1 Word (count)':<25} {'P2 Word (count)':<25} {'P3 Word (count)':<25}", f)
    log(f"{'-'*6} {'-'*25} {'-'*25} {'-'*25}", f)

    top_words = {p: word_counts[p].most_common(30) for p in placements}

    for i in range(30):
        row = f"{i+1:<6}"
        for p in placements:
            if i < len(top_words[p]):
                word, count = top_words[p][i]
                row += f" {word} ({count:,})".ljust(25)
            else:
                row += " ".ljust(25)
        log(row, f)

    # Words unique to each placement (appear 2x+ more often than others)
    log(f"\n--- Words Over-Represented by Placement ---", f)
    log(f"(Words appearing 2x+ more often in one placement vs others)", f)

    for p in placements:
        other_placements = [op for op in placements if op != p]

        # Calculate word frequency (normalized by total words)
        total_words_p = sum(word_counts[p].values())
        word_freq_p = {w: c / total_words_p for w, c in word_counts[p].items()}

        # Average frequency in other placements
        other_totals = [sum(word_counts[op].values()) for op in other_placements]

        over_rep = []
        for word in word_freq_p:
            freq_p = word_freq_p[word]
            other_freqs = []
            for op, ot in zip(other_placements, other_totals):
                other_freqs.append(word_counts[op].get(word, 0) / ot if ot > 0 else 0)
            avg_other_freq = np.mean(other_freqs) if other_freqs else 0

            if freq_p > 2 * avg_other_freq and word_counts[p][word] >= 50:  # At least 50 occurrences
                ratio = freq_p / avg_other_freq if avg_other_freq > 0 else float('inf')
                over_rep.append((word, word_counts[p][word], ratio))

        over_rep.sort(key=lambda x: x[2], reverse=True)

        log(f"\n{placement_names[p]} over-represented words (top 15):", f)
        log(f"{'Word':<20} {'Count':>10} {'Ratio':>10}", f)
        log(f"{'-'*20} {'-'*10} {'-'*10}", f)
        for word, count, ratio in over_rep[:15]:
            ratio_str = f"{ratio:.1f}x" if ratio < float('inf') else "unique"
            log(f"{word:<20} {count:>10,} {ratio_str:>10}", f)

    # Category analysis
    log(f"\n--- Category Tag Analysis ---", f)

    if 'CATEGORIES' not in si.columns or si['CATEGORIES'].isna().all():
        log(f"No category data available.", f)
        return

    def parse_categories(cat_str):
        """Parse CATEGORIES string/list into structured tags."""
        if pd.isna(cat_str):
            return {}

        # Handle string representation of list
        if isinstance(cat_str, str):
            # Try to parse as JSON-like list
            try:
                import json
                cat_list = json.loads(cat_str.replace("'", '"'))
            except:
                # Split by newlines or commas
                cat_list = [c.strip().strip('"').strip("'") for c in cat_str.replace('[', '').replace(']', '').split(',')]
        else:
            cat_list = list(cat_str) if hasattr(cat_str, '__iter__') else []

        tags = {
            'brand': [],
            'color': [],
            'department': [],
            'category': [],
            'style_tag': [],
            'size': []
        }

        for item in cat_list:
            item = str(item).strip()
            for tag_type in tags:
                if item.startswith(f'{tag_type}#'):
                    value = item.split('#', 1)[1] if '#' in item else item
                    tags[tag_type].append(value)

        return tags

    # Extract tags by placement
    tag_counts = {p: {tag: Counter() for tag in ['brand', 'color', 'department', 'style_tag']} for p in placements}

    for p in placements:
        subset = si[si['placement'] == p]
        for cat_str in tqdm(subset['CATEGORIES'].dropna(), desc=f"Parsing categories P{p}"):
            tags = parse_categories(cat_str)
            for tag_type in ['brand', 'color', 'department', 'style_tag']:
                for value in tags.get(tag_type, []):
                    tag_counts[p][tag_type][value] += 1

    # Top brands by placement
    log(f"\n--- Top 20 Brands by Placement ---", f)
    log(f"{'Rank':<6} {'P1 Brand (%)':<25} {'P2 Brand (%)':<25} {'P3 Brand (%)':<25}", f)
    log(f"{'-'*6} {'-'*25} {'-'*25} {'-'*25}", f)

    for i in range(20):
        row = f"{i+1:<6}"
        for p in placements:
            brands = tag_counts[p]['brand'].most_common(20)
            total = sum(tag_counts[p]['brand'].values())
            if i < len(brands) and total > 0:
                brand, count = brands[i]
                pct = count / total * 100
                row += f" {brand[:15]} ({pct:.1f}%)".ljust(25)
            else:
                row += " ".ljust(25)
        log(row, f)

    # Top colors by placement
    log(f"\n--- Top 15 Colors by Placement ---", f)
    log(f"{'Rank':<6} {'P1 Color (%)':<25} {'P2 Color (%)':<25} {'P3 Color (%)':<25}", f)
    log(f"{'-'*6} {'-'*25} {'-'*25} {'-'*25}", f)

    for i in range(15):
        row = f"{i+1:<6}"
        for p in placements:
            colors = tag_counts[p]['color'].most_common(15)
            total = sum(tag_counts[p]['color'].values())
            if i < len(colors) and total > 0:
                color, count = colors[i]
                pct = count / total * 100
                row += f" {color[:15]} ({pct:.1f}%)".ljust(25)
            else:
                row += " ".ljust(25)
        log(row, f)

    # Top style tags by placement
    log(f"\n--- Top 20 Style Tags by Placement ---", f)
    log(f"{'Rank':<6} {'P1 Style Tag (%)':<25} {'P2 Style Tag (%)':<25} {'P3 Style Tag (%)':<25}", f)
    log(f"{'-'*6} {'-'*25} {'-'*25} {'-'*25}", f)

    for i in range(20):
        row = f"{i+1:<6}"
        for p in placements:
            style_tags = tag_counts[p]['style_tag'].most_common(20)
            total = sum(tag_counts[p]['style_tag'].values())
            if i < len(style_tags) and total > 0:
                tag, count = style_tags[i]
                pct = count / total * 100
                row += f" {tag[:15]} ({pct:.1f}%)".ljust(25)
            else:
                row += " ".ljust(25)
        log(row, f)

# =============================================================================
# SECTION 8: KEY DIFFERENCES SUMMARY
# =============================================================================
def key_differences_summary(si, f):
    """Summary of key differences between placements."""
    log(f"\n{'='*80}", f)
    log("SECTION 8: KEY DIFFERENCES SUMMARY", f)
    log(f"{'='*80}", f)

    placements = [1, 2, 3]
    placement_names = {
        1: 'P1 (Browse/Feed)',
        2: 'P2 (Category/Collection)',
        3: 'P3 (Search Pagination)'
    }

    log(f"\n--- Volume ---", f)
    for p in placements:
        subset = si[si['placement'] == p]
        log(f"  {placement_names[p]}: {len(subset):,} items, {subset['auction_id'].nunique():,} sessions", f)

    log(f"\n--- CTR ---", f)
    for p in placements:
        subset = si[si['placement'] == p]
        ctr = subset['clicked'].mean() * 100
        log(f"  {placement_names[p]}: {ctr:.2f}%", f)

    log(f"\n--- Session Size ---", f)
    for p in placements:
        subset = si[si['placement'] == p]
        items_per_session = subset.groupby('auction_id').size()
        log(f"  {placement_names[p]}: mean={items_per_session.mean():.1f}, median={items_per_session.median():.0f}, max={items_per_session.max()}", f)

    log(f"\n--- Position Range ---", f)
    for p in placements:
        subset = si[si['placement'] == p]
        log(f"  {placement_names[p]}: positions 1-{subset['position'].max()}", f)

    log(f"\n--- Click Position Distribution ---", f)
    for p in placements:
        subset = si[(si['placement'] == p) & (si['clicked'] == 1)]
        if len(subset) > 0:
            mean_pos = subset['position'].mean()
            median_pos = subset['position'].median()
            log(f"  {placement_names[p]}: mean clicked position={mean_pos:.1f}, median={median_pos:.0f}", f)
        else:
            log(f"  {placement_names[p]}: no clicks", f)

    log(f"\n--- User Overlap Summary ---", f)
    if 'USER_ID' in si.columns and si['USER_ID'].notna().any():
        si_users = si[si['USER_ID'].notna()]

        p1_users = set(si_users[si_users['placement'] == 1]['USER_ID'].unique())
        p2_users = set(si_users[si_users['placement'] == 2]['USER_ID'].unique())
        p3_users = set(si_users[si_users['placement'] == 3]['USER_ID'].unique())

        all_users = p1_users | p2_users | p3_users
        all_three = p1_users & p2_users & p3_users

        log(f"  Total unique users: {len(all_users):,}", f)
        log(f"  Users in P1: {len(p1_users):,} ({len(p1_users)/len(all_users)*100:.1f}%)", f)
        log(f"  Users in P2: {len(p2_users):,} ({len(p2_users)/len(all_users)*100:.1f}%)", f)
        log(f"  Users in P3: {len(p3_users):,} ({len(p3_users)/len(all_users)*100:.1f}%)", f)
        log(f"  Users in all three: {len(all_three):,} ({len(all_three)/len(all_users)*100:.1f}%)", f)

# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("="*80, f)
        log("THREE-WAY PLACEMENT COMPARISON: P1 vs P2 vs P3", f)
        log("="*80, f)
        log(f"P1: Browse/Feed (largest)", f)
        log(f"P2: Category/Collection", f)
        log(f"P3: Search Pagination", f)
        log(f"\nData sources:", f)
        log(f"  - session_items.parquet (item-level with position, rank, clicked)", f)
        log(f"  - auctions_users_r2.parquet (user IDs and timestamps)", f)
        log(f"  - catalog_r2.parquet (product NAME, CATEGORIES)", f)

        # Section 1: Load data
        si, p1, p2, p3, au, catalog = load_data(f)

        # Section 2: Aggregate statistics
        stats = aggregate_statistics(si, f)

        # Section 3: Position/rank distributions
        position_rank_distributions(si, f)

        # Section 4: Temporal patterns
        temporal_patterns(si, f)

        # Section 5: User behavior
        user_behavior(si, f)

        # Section 6: Raw session logs
        raw_session_logs(si, f)

        # Section 7: Product type analysis
        product_type_analysis(si, f)

        # Section 8: Key differences summary
        key_differences_summary(si, f)

        # Final
        log(f"\n{'='*80}", f)
        log("ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output saved to: {OUTPUT_FILE}", f)

if __name__ == "__main__":
    main()
