#!/usr/bin/env python3
"""
Agnostic Placement Characterization EDA

Compares P1, P2, and P3 placements WITHOUT presupposing what they represent.
Let the data reveal the nature of each placement through behavioral metrics.

Data volumes:
- P1: 108,078 rows
- P2: 61,386 rows
- P3: 48,420 rows

Output: results/12_p1_vs_p3_comparison.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Up to position-effects/
DATA_R1_DIR = BASE_DIR / "0_data" / "round1"
DATA_R2_DIR = BASE_DIR / "0_data" / "round2"
RESULTS_DIR = Path(__file__).parent / "results"
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
# MAIN
# =============================================================================
def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("AGNOSTIC PLACEMENT CHARACTERIZATION EDA", f)
        log("P1 vs P2 vs P3 Three-Way Comparison", f)
        log("=" * 80, f)
        log(f"Output: {OUTPUT_FILE}", f)
        log("", f)

        # =====================================================================
        # SECTION 1: DATA LOADING & MERGE
        # =====================================================================
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING & MERGE", f)
        log("=" * 80, f)

        log("\nLoading session_items.parquet...", f)
        session_items = pd.read_parquet(DATA_R1_DIR / "session_items.parquet")
        log(f"  Shape: {session_items.shape}", f)
        log(f"  Columns: {list(session_items.columns)}", f)

        log("\nLoading auctions_users_r2.parquet...", f)
        auctions_users = pd.read_parquet(DATA_R2_DIR / "auctions_users_r2.parquet")
        log(f"  Shape: {auctions_users.shape}", f)
        log(f"  Columns: {list(auctions_users.columns)}", f)

        log("\nLoading catalog_r2.parquet...", f)
        catalog = pd.read_parquet(DATA_R2_DIR / "catalog_r2.parquet")
        log(f"  Shape: {catalog.shape}", f)
        log(f"  Columns: {list(catalog.columns)}", f)

        # Normalize auction_id format (lowercase for join)
        session_items['auction_id'] = session_items['auction_id'].str.lower()
        auctions_users['AUCTION_ID'] = auctions_users['AUCTION_ID'].str.lower()
        catalog['PRODUCT_ID'] = catalog['PRODUCT_ID'].str.lower()
        session_items['product_id'] = session_items['product_id'].str.lower()

        # Merge to get user_id and timestamp
        log("\nMerging session_items with auctions_users...", f)
        merged = session_items.merge(
            auctions_users[['AUCTION_ID', 'USER_ID', 'CREATED_AT']],
            left_on='auction_id',
            right_on='AUCTION_ID',
            how='left'
        )
        log(f"  Merged shape: {merged.shape}", f)
        log(f"  Rows with user_id: {merged['USER_ID'].notna().sum():,}", f)

        # Merge with catalog for price data
        log("\nMerging with catalog for price data...", f)
        merged = merged.merge(
            catalog[['PRODUCT_ID', 'CATALOG_PRICE']],
            left_on='product_id',
            right_on='PRODUCT_ID',
            how='left'
        )
        log(f"  Merged shape with catalog: {merged.shape}", f)
        log(f"  Rows with price: {merged['CATALOG_PRICE'].notna().sum():,}", f)

        # Filter to P1, P2, and P3
        log("\nFiltering to P1, P2, and P3...", f)
        p1_data = merged[merged['placement'] == '1'].copy()
        p2_data = merged[merged['placement'] == '2'].copy()
        p3_data = merged[merged['placement'] == '3'].copy()

        log(f"  P1 rows: {len(p1_data):,}", f)
        log(f"  P2 rows: {len(p2_data):,}", f)
        log(f"  P3 rows: {len(p3_data):,}", f)

        placements = {'P1': p1_data, 'P2': p2_data, 'P3': p3_data}

        # =====================================================================
        # SECTION 2: AGGREGATE STATISTICS SIDE-BY-SIDE
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 2: AGGREGATE STATISTICS SIDE-BY-SIDE", f)
        log("=" * 80, f)

        def compute_stats(df, name):
            stats = {}
            stats['Total items shown'] = len(df)
            stats['Total auctions'] = df['auction_id'].nunique()
            stats['Total clicks'] = df['clicked'].sum()
            stats['CTR (%)'] = 100 * df['clicked'].mean() if len(df) > 0 else 0
            stats['Unique users'] = df['USER_ID'].nunique()
            stats['Unique products'] = df['product_id'].nunique()

            items_per_auction = df.groupby('auction_id').size()
            stats['Avg items per auction'] = items_per_auction.mean()
            stats['Median items per auction'] = items_per_auction.median()
            stats['Max items per auction'] = items_per_auction.max()

            clicks_per_auction = df.groupby('auction_id')['clicked'].sum()
            stats['% auctions with clicks'] = 100 * (clicks_per_auction > 0).mean()
            stats['% multi-click auctions'] = 100 * (clicks_per_auction > 1).mean()

            clicked_items = df[df['clicked'] == 1]
            stats['Avg position of clicked items'] = clicked_items['position'].mean() if len(clicked_items) > 0 else np.nan
            stats['Avg rank of clicked items'] = clicked_items['rank'].mean() if len(clicked_items) > 0 else np.nan

            return stats

        all_stats = {name: compute_stats(df, name) for name, df in placements.items()}

        log("\n" + "-" * 75, f)
        log(f"{'Metric':<35} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 75, f)

        for metric in all_stats['P1'].keys():
            v1 = all_stats['P1'][metric]
            v2 = all_stats['P2'][metric]
            v3 = all_stats['P3'][metric]
            if isinstance(v1, float):
                log(f"{metric:<35} {v1:>12.2f} {v2:>12.2f} {v3:>12.2f}", f)
            else:
                log(f"{metric:<35} {v1:>12,} {v2:>12,} {v3:>12,}", f)

        log("-" * 75, f)

        # =====================================================================
        # SECTION 3: POSITION/RANK DISTRIBUTIONS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 3: POSITION/RANK DISTRIBUTIONS", f)
        log("=" * 80, f)

        # Position distribution
        log("\n--- Position Distribution (1-20) ---", f)
        log(f"{'Position':<8} {'P1 Count':>10} {'P1 %':>8} {'P2 Count':>10} {'P2 %':>8} {'P3 Count':>10} {'P3 %':>8}", f)
        log("-" * 70, f)

        pos_counts = {name: df['position'].value_counts().sort_index() for name, df in placements.items()}

        for pos in range(1, 21):
            vals = []
            for name, df in placements.items():
                cnt = pos_counts[name].get(pos, 0)
                pct = 100 * cnt / len(df) if len(df) > 0 else 0
                vals.extend([cnt, pct])
            log(f"{pos:<8} {vals[0]:>10,} {vals[1]:>7.1f}% {vals[2]:>10,} {vals[3]:>7.1f}% {vals[4]:>10,} {vals[5]:>7.1f}%", f)

        # Rank distribution
        log("\n--- Rank Distribution (1-20) ---", f)
        log(f"{'Rank':<8} {'P1 Count':>10} {'P1 %':>8} {'P2 Count':>10} {'P2 %':>8} {'P3 Count':>10} {'P3 %':>8}", f)
        log("-" * 70, f)

        rank_counts = {name: df['rank'].value_counts().sort_index() for name, df in placements.items()}

        for rank in range(1, 21):
            vals = []
            for name, df in placements.items():
                cnt = rank_counts[name].get(rank, 0)
                pct = 100 * cnt / len(df) if len(df) > 0 else 0
                vals.extend([cnt, pct])
            log(f"{rank:<8} {vals[0]:>10,} {vals[1]:>7.1f}% {vals[2]:>10,} {vals[3]:>7.1f}% {vals[4]:>10,} {vals[5]:>7.1f}%", f)

        # CTR by position
        log("\n--- CTR by Position (1-20) ---", f)
        log(f"{'Position':<8} {'P1 CTR':>10} {'P1 N':>8} {'P2 CTR':>10} {'P2 N':>8} {'P3 CTR':>10} {'P3 N':>8}", f)
        log("-" * 70, f)

        for pos in range(1, 21):
            vals = []
            for name, df in placements.items():
                pos_data = df[df['position'] == pos]
                ctr = 100 * pos_data['clicked'].mean() if len(pos_data) > 0 else np.nan
                n = len(pos_data)
                ctr_str = f"{ctr:.2f}%" if not np.isnan(ctr) else "N/A"
                vals.extend([ctr_str, n])
            log(f"{pos:<8} {vals[0]:>10} {vals[1]:>8,} {vals[2]:>10} {vals[3]:>8,} {vals[4]:>10} {vals[5]:>8,}", f)

        # CTR by rank
        log("\n--- CTR by Rank (1-20) ---", f)
        log(f"{'Rank':<8} {'P1 CTR':>10} {'P1 N':>8} {'P2 CTR':>10} {'P2 N':>8} {'P3 CTR':>10} {'P3 N':>8}", f)
        log("-" * 70, f)

        for rank in range(1, 21):
            vals = []
            for name, df in placements.items():
                rank_data = df[df['rank'] == rank]
                ctr = 100 * rank_data['clicked'].mean() if len(rank_data) > 0 else np.nan
                n = len(rank_data)
                ctr_str = f"{ctr:.2f}%" if not np.isnan(ctr) else "N/A"
                vals.extend([ctr_str, n])
            log(f"{rank:<8} {vals[0]:>10} {vals[1]:>8,} {vals[2]:>10} {vals[3]:>8,} {vals[4]:>10} {vals[5]:>8,}", f)

        # =====================================================================
        # SECTION 4: TEMPORAL PATTERNS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 4: TEMPORAL PATTERNS", f)
        log("=" * 80, f)

        # Get auction-level data with timestamps
        auction_data = {}
        gaps_data = {}
        for name, df in placements.items():
            auctions = df[['auction_id', 'USER_ID', 'CREATED_AT']].drop_duplicates()
            auctions = auctions.sort_values(['USER_ID', 'CREATED_AT'])
            auction_data[name] = auctions

            # Compute inter-auction gaps
            auctions_copy = auctions.copy()
            auctions_copy['prev_time'] = auctions_copy.groupby('USER_ID')['CREATED_AT'].shift(1)
            auctions_copy['gap_seconds'] = (auctions_copy['CREATED_AT'] - auctions_copy['prev_time']).dt.total_seconds()
            gaps_data[name] = auctions_copy[auctions_copy['gap_seconds'].notna()]['gap_seconds']

        log("\n--- Inter-Auction Gap Distribution (seconds) ---", f)
        log(f"{'Statistic':<20} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 67, f)

        stats_to_compute = [
            ('N transitions', lambda x: len(x)),
            ('Mean', lambda x: x.mean()),
            ('Median', lambda x: x.median()),
            ('Std', lambda x: x.std()),
            ('Min', lambda x: x.min()),
            ('25th percentile', lambda x: x.quantile(0.25)),
            ('75th percentile', lambda x: x.quantile(0.75)),
            ('90th percentile', lambda x: x.quantile(0.90)),
            ('Max', lambda x: x.max()),
            ('% < 1 sec', lambda x: 100 * (x < 1).mean()),
            ('% < 5 sec', lambda x: 100 * (x < 5).mean()),
            ('% < 30 sec', lambda x: 100 * (x < 30).mean()),
        ]

        for stat_name, stat_func in stats_to_compute:
            vals = []
            for name in ['P1', 'P2', 'P3']:
                val = stat_func(gaps_data[name]) if len(gaps_data[name]) > 0 else np.nan
                vals.append(val)
            log(f"{stat_name:<20} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # Gap distribution buckets
        log("\n--- Gap Distribution Buckets ---", f)
        buckets = [(0, 0.1), (0.1, 1), (1, 5), (5, 30), (30, 60), (60, 300), (300, float('inf'))]
        log(f"{'Gap Range':<20} {'P1 %':>12} {'P2 %':>12} {'P3 %':>12}", f)
        log("-" * 58, f)

        for low, high in buckets:
            if high == float('inf'):
                label = f">= {low}s"
            else:
                label = f"{low}-{high}s"

            vals = []
            for name in ['P1', 'P2', 'P3']:
                pct = 100 * ((gaps_data[name] >= low) & (gaps_data[name] < high)).mean() if len(gaps_data[name]) > 0 else 0
                vals.append(pct)
            log(f"{label:<20} {vals[0]:>11.1f}% {vals[1]:>11.1f}% {vals[2]:>11.1f}%", f)

        # =====================================================================
        # SECTION 5: USER BEHAVIOR
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 5: USER BEHAVIOR", f)
        log("=" * 80, f)

        # Auctions per user
        user_auctions = {name: auction_data[name].groupby('USER_ID').size() for name in placements}

        log("\n--- Auctions Per User ---", f)
        log(f"{'Statistic':<20} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 67, f)

        user_stats = [
            ('Unique users', lambda x: len(x)),
            ('Mean auctions', lambda x: x.mean()),
            ('Median auctions', lambda x: x.median()),
            ('Max auctions', lambda x: x.max()),
            ('% single-auction', lambda x: 100 * (x == 1).mean()),
            ('% 2+ auctions', lambda x: 100 * (x >= 2).mean()),
            ('% 5+ auctions', lambda x: 100 * (x >= 5).mean()),
            ('% 10+ auctions', lambda x: 100 * (x >= 10).mean()),
        ]

        for stat_name, stat_func in user_stats:
            vals = []
            for name in ['P1', 'P2', 'P3']:
                val = stat_func(user_auctions[name]) if len(user_auctions[name]) > 0 else np.nan
                vals.append(val)
            log(f"{stat_name:<20} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # User overlap analysis
        p1_users = set(p1_data['USER_ID'].dropna().unique())
        p2_users = set(p2_data['USER_ID'].dropna().unique())
        p3_users = set(p3_data['USER_ID'].dropna().unique())

        log(f"\n--- User Overlap Matrix ---", f)
        log(f"{'Users in...':<25} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 63, f)
        log(f"{'Only this placement':<25} {len(p1_users - p2_users - p3_users):>12,} {len(p2_users - p1_users - p3_users):>12,} {len(p3_users - p1_users - p2_users):>12,}", f)
        log(f"{'Also in P1':<25} {'-':>12} {len(p1_users & p2_users):>12,} {len(p1_users & p3_users):>12,}", f)
        log(f"{'Also in P2':<25} {len(p1_users & p2_users):>12,} {'-':>12} {len(p2_users & p3_users):>12,}", f)
        log(f"{'Also in P3':<25} {len(p1_users & p3_users):>12,} {len(p2_users & p3_users):>12,} {'-':>12}", f)
        log(f"{'In all three':<25} {len(p1_users & p2_users & p3_users):>12,} {len(p1_users & p2_users & p3_users):>12,} {len(p1_users & p2_users & p3_users):>12,}", f)

        # Click patterns per user
        log(f"\n--- Clicks Per User ---", f)
        log(f"{'Statistic':<25} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 72, f)

        user_clicks = {name: df.groupby('USER_ID')['clicked'].sum() for name, df in placements.items()}

        click_stats = [
            ('Users with 0 clicks', lambda x: (x == 0).sum()),
            ('Users with 1+ clicks', lambda x: (x >= 1).sum()),
            ('Users with 2+ clicks', lambda x: (x >= 2).sum()),
            ('Mean clicks per user', lambda x: x.mean()),
            ('Max clicks per user', lambda x: x.max()),
            ('% users with clicks', lambda x: 100 * (x > 0).mean()),
        ]

        for stat_name, stat_func in click_stats:
            vals = []
            for name in ['P1', 'P2', 'P3']:
                val = stat_func(user_clicks[name]) if len(user_clicks[name]) > 0 else np.nan
                vals.append(val)
            if 'Users with' in stat_name:
                log(f"{stat_name:<25} {int(vals[0]):>15,} {int(vals[1]):>15,} {int(vals[2]):>15,}", f)
            else:
                log(f"{stat_name:<25} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # =====================================================================
        # SECTION 6: RAW SESSION LOGS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 6: RAW SESSION LOGS (SAMPLE SESSIONS)", f)
        log("=" * 80, f)

        def print_session(auction_id, df, f):
            """Print a single session/auction in tabular format."""
            session_data = df[df['auction_id'] == auction_id].sort_values('position')

            if len(session_data) == 0:
                return

            user_id = session_data['USER_ID'].iloc[0]
            timestamp = session_data['CREATED_AT'].iloc[0]
            n_items = len(session_data)
            n_clicks = session_data['clicked'].sum()

            log(f"\n=== AUCTION: {auction_id[:20]}... ===", f)
            log(f"User: {user_id}", f)
            log(f"Time: {timestamp}", f)
            log(f"Items: {n_items} | Clicks: {n_clicks}", f)
            log("", f)
            log(f"{'Pos':<5} {'Rank':<6} {'Product_ID':<26} {'Quality':>10} {'Bid':>6} {'Clicked':<8}", f)
            log("-" * 65, f)

            for _, row in session_data.iterrows():
                clicked_marker = "*" if row['clicked'] == 1 else ""
                quality_str = f"{row['quality']:.4f}" if pd.notna(row['quality']) else "N/A"
                bid_str = f"{int(row['bid'])}" if pd.notna(row['bid']) else "N/A"
                log(f"{int(row['position']):<5} {int(row['rank']):<6} {row['product_id'][:24]:<26} {quality_str:>10} {bid_str:>6} {clicked_marker:<8}", f)

        np.random.seed(42)

        for name, df in placements.items():
            auction_clicks = df.groupby('auction_id')['clicked'].sum()
            with_clicks = auction_clicks[auction_clicks > 0].index.tolist()
            without_clicks = auction_clicks[auction_clicks == 0].index.tolist()

            # With clicks
            log(f"\n{'-' * 80}", f)
            log(f"{name} EXAMPLES - WITH CLICKS (5 samples)", f)
            log("-" * 80, f)

            samples = np.random.choice(with_clicks, min(5, len(with_clicks)), replace=False) if len(with_clicks) > 0 else []
            for auction_id in samples:
                print_session(auction_id, df, f)

            # Without clicks
            log(f"\n{'-' * 80}", f)
            log(f"{name} EXAMPLES - NO CLICKS (3 samples)", f)
            log("-" * 80, f)

            samples = np.random.choice(without_clicks, min(3, len(without_clicks)), replace=False) if len(without_clicks) > 0 else []
            for auction_id in samples:
                print_session(auction_id, df, f)

        # =====================================================================
        # SECTION 7: KEY DIFFERENCES SUMMARY (ORIGINAL)
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 7: KEY DIFFERENCES SUMMARY (BASIC)", f)
        log("=" * 80, f)

        log("\n--- Structural Differences ---", f)
        for name in ['P1', 'P2', 'P3']:
            log(f"- {name}: {all_stats[name]['Avg items per auction']:.1f} items/auction, {all_stats[name]['Total auctions']:,} auctions, {all_stats[name]['Unique users']:,} users", f)

        log("\n--- Engagement Differences ---", f)
        for name in ['P1', 'P2', 'P3']:
            log(f"- {name} CTR: {all_stats[name]['CTR (%)']:.2f}%, {all_stats[name]['% auctions with clicks']:.1f}% auctions with clicks", f)

        # =====================================================================
        # SECTION 8: POSITION/RANK CORRELATIONS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 8: POSITION/RANK CORRELATIONS", f)
        log("=" * 80, f)

        log("\n--- Position vs Rank Correlation ---", f)
        for name, df in placements.items():
            corr = df[['position', 'rank']].corr().iloc[0, 1]
            log(f"{name} Position-Rank correlation: {corr:.4f}", f)

        # =====================================================================
        # SECTION 9: SESSION DEPTH ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 9: SESSION DEPTH ANALYSIS", f)
        log("=" * 80, f)

        log("\n--- Max Position Reached Per Session ---", f)
        log(f"{'Statistic':<25} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 72, f)

        max_pos_per_session = {name: df.groupby('auction_id')['position'].max() for name, df in placements.items()}

        depth_stats = [
            ('Mean max position', lambda x: x.mean()),
            ('Median max position', lambda x: x.median()),
            ('P95 max position', lambda x: x.quantile(0.95)),
            ('Max max position', lambda x: x.max()),
        ]

        for stat_name, stat_func in depth_stats:
            vals = [stat_func(max_pos_per_session[name]) for name in ['P1', 'P2', 'P3']]
            log(f"{stat_name:<25} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # Session length buckets
        log("\n--- Session Length Buckets (items per session) ---", f)
        session_lengths = {name: df.groupby('auction_id').size() for name, df in placements.items()}

        buckets = [(1, 4), (5, 10), (11, 20), (21, 40), (41, float('inf'))]
        log(f"{'Bucket':<20} {'P1 %':>12} {'P1 N':>10} {'P2 %':>12} {'P2 N':>10} {'P3 %':>12} {'P3 N':>10}", f)
        log("-" * 90, f)

        for low, high in buckets:
            if high == float('inf'):
                label = f"{low}+ items"
            else:
                label = f"{low}-{high} items"

            vals = []
            for name in ['P1', 'P2', 'P3']:
                if high == float('inf'):
                    mask = session_lengths[name] >= low
                else:
                    mask = (session_lengths[name] >= low) & (session_lengths[name] <= high)
                cnt = mask.sum()
                pct = 100 * cnt / len(session_lengths[name]) if len(session_lengths[name]) > 0 else 0
                vals.extend([pct, cnt])
            log(f"{label:<20} {vals[0]:>11.1f}% {vals[1]:>10,} {vals[2]:>11.1f}% {vals[3]:>10,} {vals[4]:>11.1f}% {vals[5]:>10,}", f)

        # Pagination patterns
        log("\n--- Pagination Patterns ---", f)
        log(f"{'Pattern':<35} {'P1 %':>12} {'P2 %':>12} {'P3 %':>12}", f)
        log("-" * 75, f)

        patterns = [
            ('Sessions reaching only pos 1-4', lambda x: x <= 4),
            ('Sessions reaching pos 5+', lambda x: x >= 5),
            ('Sessions reaching pos 10+', lambda x: x >= 10),
            ('Sessions reaching pos 20+', lambda x: x >= 20),
        ]

        for pattern_name, pattern_func in patterns:
            vals = []
            for name in ['P1', 'P2', 'P3']:
                pct = 100 * pattern_func(max_pos_per_session[name]).mean()
                vals.append(pct)
            log(f"{pattern_name:<35} {vals[0]:>11.1f}% {vals[1]:>11.1f}% {vals[2]:>11.1f}%", f)

        # =====================================================================
        # SECTION 10: RANK VS POSITION ALIGNMENT
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 10: RANK VS POSITION ALIGNMENT", f)
        log("=" * 80, f)

        log("\n--- Position-Rank Gap Analysis ---", f)
        log(f"{'Metric':<35} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 82, f)

        for name, df in placements.items():
            df['pos_rank_gap'] = df['position'] - df['rank']

        gap_stats = [
            ('Mean(position - rank)', lambda df: df['pos_rank_gap'].mean()),
            ('Median(position - rank)', lambda df: df['pos_rank_gap'].median()),
            ('Std(position - rank)', lambda df: df['pos_rank_gap'].std()),
            ('Min gap', lambda df: df['pos_rank_gap'].min()),
            ('Max gap', lambda df: df['pos_rank_gap'].max()),
            ('% where position == rank', lambda df: 100 * (df['pos_rank_gap'] == 0).mean()),
            ('% where position > rank', lambda df: 100 * (df['pos_rank_gap'] > 0).mean()),
            ('% where position < rank', lambda df: 100 * (df['pos_rank_gap'] < 0).mean()),
            ('Correlation(position, rank)', lambda df: df[['position', 'rank']].corr().iloc[0, 1]),
        ]

        for stat_name, stat_func in gap_stats:
            vals = [stat_func(df) for df in [p1_data, p2_data, p3_data]]
            log(f"{stat_name:<35} {vals[0]:>15.4f} {vals[1]:>15.4f} {vals[2]:>15.4f}", f)

        # Position-rank alignment by position
        log("\n--- % Items Where Rank == Position (by position) ---", f)
        log(f"{'Position':<10} {'P1 %':>12} {'P2 %':>12} {'P3 %':>12}", f)
        log("-" * 50, f)

        for pos in range(1, 11):
            vals = []
            for df in [p1_data, p2_data, p3_data]:
                pos_df = df[df['position'] == pos]
                if len(pos_df) > 0:
                    pct = 100 * (pos_df['rank'] == pos).mean()
                else:
                    pct = np.nan
                vals.append(pct)
            val_strs = [f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in vals]
            log(f"{pos:<10} {val_strs[0]:>12} {val_strs[1]:>12} {val_strs[2]:>12}", f)

        # =====================================================================
        # SECTION 11: BID/QUALITY DISTRIBUTIONS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 11: BID/QUALITY DISTRIBUTIONS", f)
        log("=" * 80, f)

        log("\n--- Bid Distribution by Placement ---", f)
        log(f"{'Statistic':<20} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 67, f)

        bid_stats = [
            ('Mean bid', lambda df: df['bid'].mean()),
            ('Median bid', lambda df: df['bid'].median()),
            ('Std bid', lambda df: df['bid'].std()),
            ('P10 bid', lambda df: df['bid'].quantile(0.10)),
            ('P25 bid', lambda df: df['bid'].quantile(0.25)),
            ('P75 bid', lambda df: df['bid'].quantile(0.75)),
            ('P90 bid', lambda df: df['bid'].quantile(0.90)),
            ('Max bid', lambda df: df['bid'].max()),
        ]

        for stat_name, stat_func in bid_stats:
            vals = [stat_func(df) for df in [p1_data, p2_data, p3_data]]
            log(f"{stat_name:<20} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        log("\n--- Quality Score Distribution by Placement ---", f)
        log(f"{'Statistic':<20} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 67, f)

        quality_stats = [
            ('Mean quality', lambda df: df['quality'].mean()),
            ('Median quality', lambda df: df['quality'].median()),
            ('Std quality', lambda df: df['quality'].std()),
            ('P10 quality', lambda df: df['quality'].quantile(0.10)),
            ('P25 quality', lambda df: df['quality'].quantile(0.25)),
            ('P75 quality', lambda df: df['quality'].quantile(0.75)),
            ('P90 quality', lambda df: df['quality'].quantile(0.90)),
            ('Max quality', lambda df: df['quality'].max()),
        ]

        for stat_name, stat_func in quality_stats:
            vals = [stat_func(df) for df in [p1_data, p2_data, p3_data]]
            log(f"{stat_name:<20} {vals[0]:>15.6f} {vals[1]:>15.6f} {vals[2]:>15.6f}", f)

        log("\n--- Bid-Quality Correlation by Placement ---", f)
        for name, df in placements.items():
            corr = df[['bid', 'quality']].corr().iloc[0, 1]
            log(f"{name} Bid-Quality correlation: {corr:.4f}", f)

        log("\n--- Bid by Position (mean) ---", f)
        log(f"{'Position':<10} {'P1 Bid':>12} {'P2 Bid':>12} {'P3 Bid':>12}", f)
        log("-" * 50, f)

        for pos in range(1, 11):
            vals = []
            for df in [p1_data, p2_data, p3_data]:
                pos_df = df[df['position'] == pos]
                if len(pos_df) > 0:
                    val = pos_df['bid'].mean()
                else:
                    val = np.nan
                vals.append(val)
            val_strs = [f"{v:.2f}" if not np.isnan(v) else "N/A" for v in vals]
            log(f"{pos:<10} {val_strs[0]:>12} {val_strs[1]:>12} {val_strs[2]:>12}", f)

        log("\n--- Quality by Position (mean) ---", f)
        log(f"{'Position':<10} {'P1 Quality':>12} {'P2 Quality':>12} {'P3 Quality':>12}", f)
        log("-" * 50, f)

        for pos in range(1, 11):
            vals = []
            for df in [p1_data, p2_data, p3_data]:
                pos_df = df[df['position'] == pos]
                if len(pos_df) > 0:
                    val = pos_df['quality'].mean()
                else:
                    val = np.nan
                vals.append(val)
            val_strs = [f"{v:.4f}" if not np.isnan(v) else "N/A" for v in vals]
            log(f"{pos:<10} {val_strs[0]:>12} {val_strs[1]:>12} {val_strs[2]:>12}", f)

        # =====================================================================
        # SECTION 12: CLICK DEPTH PATTERNS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 12: CLICK DEPTH PATTERNS", f)
        log("=" * 80, f)

        clicked_items = {name: df[df['clicked'] == 1] for name, df in placements.items()}

        log("\n--- Click Position Distribution ---", f)
        log(f"{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 77, f)

        click_pos_stats = [
            ('Total clicks', lambda df: len(df)),
            ('Mean click position', lambda df: df['position'].mean() if len(df) > 0 else np.nan),
            ('Median click position', lambda df: df['position'].median() if len(df) > 0 else np.nan),
            ('P25 click position', lambda df: df['position'].quantile(0.25) if len(df) > 0 else np.nan),
            ('P75 click position', lambda df: df['position'].quantile(0.75) if len(df) > 0 else np.nan),
            ('P90 click position', lambda df: df['position'].quantile(0.90) if len(df) > 0 else np.nan),
        ]

        for stat_name, stat_func in click_pos_stats:
            vals = [stat_func(clicked_items[name]) for name in ['P1', 'P2', 'P3']]
            if 'Total' in stat_name:
                log(f"{stat_name:<30} {int(vals[0]):>15,} {int(vals[1]):>15,} {int(vals[2]):>15,}", f)
            else:
                log(f"{stat_name:<30} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        log("\n--- Click Rank Distribution ---", f)
        log(f"{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 77, f)

        click_rank_stats = [
            ('Mean click rank', lambda df: df['rank'].mean() if len(df) > 0 else np.nan),
            ('Median click rank', lambda df: df['rank'].median() if len(df) > 0 else np.nan),
            ('P25 click rank', lambda df: df['rank'].quantile(0.25) if len(df) > 0 else np.nan),
            ('P75 click rank', lambda df: df['rank'].quantile(0.75) if len(df) > 0 else np.nan),
        ]

        for stat_name, stat_func in click_rank_stats:
            vals = [stat_func(clicked_items[name]) for name in ['P1', 'P2', 'P3']]
            log(f"{stat_name:<30} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # First click position per session
        log("\n--- First Click Position Per Session ---", f)

        first_click_pos = {}
        for name, df in placements.items():
            clicked_df = df[df['clicked'] == 1].copy()
            if len(clicked_df) > 0:
                first_click_pos[name] = clicked_df.groupby('auction_id')['position'].min()
            else:
                first_click_pos[name] = pd.Series(dtype=float)

        log(f"{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 77, f)

        first_click_stats = [
            ('N sessions with clicks', lambda x: len(x)),
            ('Mean first click pos', lambda x: x.mean() if len(x) > 0 else np.nan),
            ('Median first click pos', lambda x: x.median() if len(x) > 0 else np.nan),
            ('P25 first click pos', lambda x: x.quantile(0.25) if len(x) > 0 else np.nan),
            ('P75 first click pos', lambda x: x.quantile(0.75) if len(x) > 0 else np.nan),
        ]

        for stat_name, stat_func in first_click_stats:
            vals = [stat_func(first_click_pos[name]) for name in ['P1', 'P2', 'P3']]
            if 'N sessions' in stat_name:
                log(f"{stat_name:<30} {int(vals[0]):>15,} {int(vals[1]):>15,} {int(vals[2]):>15,}", f)
            else:
                log(f"{stat_name:<30} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # Click concentration
        log("\n--- Click Concentration by Position ---", f)
        log(f"{'Metric':<35} {'P1 %':>12} {'P2 %':>12} {'P3 %':>12}", f)
        log("-" * 75, f)

        concentration_stats = [
            ('% clicks in positions 1-3', lambda df: 100 * (df['position'] <= 3).mean() if len(df) > 0 else np.nan),
            ('% clicks in positions 4+', lambda df: 100 * (df['position'] >= 4).mean() if len(df) > 0 else np.nan),
            ('% clicks in positions 1', lambda df: 100 * (df['position'] == 1).mean() if len(df) > 0 else np.nan),
            ('% clicks in positions 5+', lambda df: 100 * (df['position'] >= 5).mean() if len(df) > 0 else np.nan),
            ('% clicks in positions 10+', lambda df: 100 * (df['position'] >= 10).mean() if len(df) > 0 else np.nan),
        ]

        for stat_name, stat_func in concentration_stats:
            vals = [stat_func(clicked_items[name]) for name in ['P1', 'P2', 'P3']]
            val_strs = [f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in vals]
            log(f"{stat_name:<35} {val_strs[0]:>12} {val_strs[1]:>12} {val_strs[2]:>12}", f)

        # =====================================================================
        # SECTION 13: PRODUCT DIVERSITY
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 13: PRODUCT DIVERSITY", f)
        log("=" * 80, f)

        log("\n--- Products Per Session ---", f)
        log(f"{'Metric':<30} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 77, f)

        products_per_session = {name: df.groupby('auction_id')['product_id'].nunique() for name, df in placements.items()}

        prod_stats = [
            ('Mean unique products', lambda x: x.mean()),
            ('Median unique products', lambda x: x.median()),
            ('Max unique products', lambda x: x.max()),
            ('% sessions w/ 1 product', lambda x: 100 * (x == 1).mean()),
            ('% sessions w/ 5+ products', lambda x: 100 * (x >= 5).mean()),
        ]

        for stat_name, stat_func in prod_stats:
            vals = [stat_func(products_per_session[name]) for name in ['P1', 'P2', 'P3']]
            log(f"{stat_name:<30} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # Product repeat rate (how often same product appears across sessions)
        log("\n--- Product Repeat Rate ---", f)
        log(f"{'Metric':<40} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 70, f)

        for name, df in placements.items():
            product_session_counts = df.groupby('product_id')['auction_id'].nunique()
            avg_sessions_per_product = product_session_counts.mean()
            pct_products_in_multiple = 100 * (product_session_counts > 1).mean()
            log(f"{'Avg sessions per product':<40} {avg_sessions_per_product:>12.2f}" if name == 'P1' else "", f)

        for stat_name, stat_func in [
            ('Avg sessions per product', lambda df: df.groupby('product_id')['auction_id'].nunique().mean()),
            ('% products in 2+ sessions', lambda df: 100 * (df.groupby('product_id')['auction_id'].nunique() > 1).mean()),
            ('% products in 5+ sessions', lambda df: 100 * (df.groupby('product_id')['auction_id'].nunique() > 5).mean()),
        ]:
            vals = [stat_func(df) for df in [p1_data, p2_data, p3_data]]
            log(f"{stat_name:<40} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f}", f)

        # Price distribution from catalog
        log("\n--- Price Distribution (from catalog) ---", f)
        log(f"{'Metric':<25} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 72, f)

        price_stats = [
            ('N with price', lambda df: df['CATALOG_PRICE'].notna().sum()),
            ('Mean price', lambda df: df['CATALOG_PRICE'].mean()),
            ('Median price', lambda df: df['CATALOG_PRICE'].median()),
            ('P10 price', lambda df: df['CATALOG_PRICE'].quantile(0.10)),
            ('P25 price', lambda df: df['CATALOG_PRICE'].quantile(0.25)),
            ('P75 price', lambda df: df['CATALOG_PRICE'].quantile(0.75)),
            ('P90 price', lambda df: df['CATALOG_PRICE'].quantile(0.90)),
        ]

        for stat_name, stat_func in price_stats:
            vals = [stat_func(df) for df in [p1_data, p2_data, p3_data]]
            if 'N with' in stat_name:
                log(f"{stat_name:<25} {int(vals[0]):>15,} {int(vals[1]):>15,} {int(vals[2]):>15,}", f)
            else:
                log(f"{stat_name:<25} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # =====================================================================
        # SECTION 14: TEMPORAL FINGERPRINT
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 14: TEMPORAL FINGERPRINT", f)
        log("=" * 80, f)

        # Hour of day distribution
        log("\n--- Hour of Day Distribution ---", f)
        log(f"{'Hour':<8} {'P1 %':>10} {'P1 N':>10} {'P2 %':>10} {'P2 N':>10} {'P3 %':>10} {'P3 N':>10}", f)
        log("-" * 72, f)

        for name, df in placements.items():
            df['hour'] = pd.to_datetime(df['CREATED_AT']).dt.hour

        for hour in range(24):
            vals = []
            for name, df in placements.items():
                cnt = (df['hour'] == hour).sum()
                pct = 100 * cnt / len(df) if len(df) > 0 else 0
                vals.extend([pct, cnt])
            log(f"{hour:<8} {vals[0]:>9.1f}% {vals[1]:>10,} {vals[2]:>9.1f}% {vals[3]:>10,} {vals[4]:>9.1f}% {vals[5]:>10,}", f)

        # Day of week distribution
        log("\n--- Day of Week Distribution ---", f)
        log(f"{'Day':<12} {'P1 %':>10} {'P1 N':>10} {'P2 %':>10} {'P2 N':>10} {'P3 %':>10} {'P3 N':>10}", f)
        log("-" * 76, f)

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for name, df in placements.items():
            df['dayofweek'] = pd.to_datetime(df['CREATED_AT']).dt.dayofweek

        for dow, day_name in enumerate(days):
            vals = []
            for name, df in placements.items():
                cnt = (df['dayofweek'] == dow).sum()
                pct = 100 * cnt / len(df) if len(df) > 0 else 0
                vals.extend([pct, cnt])
            log(f"{day_name:<12} {vals[0]:>9.1f}% {vals[1]:>10,} {vals[2]:>9.1f}% {vals[3]:>10,} {vals[4]:>9.1f}% {vals[5]:>10,}", f)

        # Session burst patterns
        log("\n--- Session Burst Patterns (inter-auction gap) ---", f)
        log(f"{'Pattern':<40} {'P1 %':>12} {'P2 %':>12} {'P3 %':>12}", f)
        log("-" * 80, f)

        burst_patterns = [
            ('Rapid-fire (<1s between auctions)', lambda x: 100 * (x < 1).mean()),
            ('Quick (1-5s between auctions)', lambda x: 100 * ((x >= 1) & (x < 5)).mean()),
            ('Moderate (5-30s between auctions)', lambda x: 100 * ((x >= 5) & (x < 30)).mean()),
            ('Deliberate (30s-5m between auctions)', lambda x: 100 * ((x >= 30) & (x < 300)).mean()),
            ('New session (>5m between auctions)', lambda x: 100 * (x >= 300).mean()),
        ]

        for pattern_name, pattern_func in burst_patterns:
            vals = [pattern_func(gaps_data[name]) if len(gaps_data[name]) > 0 else np.nan for name in ['P1', 'P2', 'P3']]
            val_strs = [f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in vals]
            log(f"{pattern_name:<40} {val_strs[0]:>12} {val_strs[1]:>12} {val_strs[2]:>12}", f)

        # =====================================================================
        # SECTION 15: SUMMARY FINGERPRINT TABLE
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 15: SUMMARY FINGERPRINT TABLE", f)
        log("=" * 80, f)

        log("\n" + "-" * 80, f)
        log(f"{'Metric':<40} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 80, f)

        # Compile fingerprint metrics
        fingerprint_metrics = []

        # Volume
        fingerprint_metrics.append(('Total items', [len(p1_data), len(p2_data), len(p3_data)], 'int'))
        fingerprint_metrics.append(('Total sessions', [p1_data['auction_id'].nunique(), p2_data['auction_id'].nunique(), p3_data['auction_id'].nunique()], 'int'))
        fingerprint_metrics.append(('Total users', [p1_data['USER_ID'].nunique(), p2_data['USER_ID'].nunique(), p3_data['USER_ID'].nunique()], 'int'))

        # Session structure
        fingerprint_metrics.append(('Median session length', [session_lengths['P1'].median(), session_lengths['P2'].median(), session_lengths['P3'].median()], 'float1'))
        fingerprint_metrics.append(('Max position reached (P95)', [max_pos_per_session['P1'].quantile(0.95), max_pos_per_session['P2'].quantile(0.95), max_pos_per_session['P3'].quantile(0.95)], 'float1'))

        # Engagement
        fingerprint_metrics.append(('CTR (%)', [100 * p1_data['clicked'].mean(), 100 * p2_data['clicked'].mean(), 100 * p3_data['clicked'].mean()], 'pct'))
        fingerprint_metrics.append(('% sessions with click', [100 * (p1_data.groupby('auction_id')['clicked'].sum() > 0).mean(),
                                                              100 * (p2_data.groupby('auction_id')['clicked'].sum() > 0).mean(),
                                                              100 * (p3_data.groupby('auction_id')['clicked'].sum() > 0).mean()], 'pct'))

        # Click depth
        fingerprint_metrics.append(('Click depth (median pos)', [clicked_items['P1']['position'].median() if len(clicked_items['P1']) > 0 else np.nan,
                                                                  clicked_items['P2']['position'].median() if len(clicked_items['P2']) > 0 else np.nan,
                                                                  clicked_items['P3']['position'].median() if len(clicked_items['P3']) > 0 else np.nan], 'float1'))

        # Position-rank alignment
        fingerprint_metrics.append(('Position-Rank correlation', [p1_data[['position', 'rank']].corr().iloc[0, 1],
                                                                   p2_data[['position', 'rank']].corr().iloc[0, 1],
                                                                   p3_data[['position', 'rank']].corr().iloc[0, 1]], 'float3'))

        # Bid and quality
        fingerprint_metrics.append(('Bid (median)', [p1_data['bid'].median(), p2_data['bid'].median(), p3_data['bid'].median()], 'float1'))
        fingerprint_metrics.append(('Quality (median)', [p1_data['quality'].median(), p2_data['quality'].median(), p3_data['quality'].median()], 'float4'))

        # Temporal
        fingerprint_metrics.append(('Median inter-auction gap (s)', [gaps_data['P1'].median() if len(gaps_data['P1']) > 0 else np.nan,
                                                                      gaps_data['P2'].median() if len(gaps_data['P2']) > 0 else np.nan,
                                                                      gaps_data['P3'].median() if len(gaps_data['P3']) > 0 else np.nan], 'float1'))
        fingerprint_metrics.append(('% rapid-fire (<1s)', [100 * (gaps_data['P1'] < 1).mean() if len(gaps_data['P1']) > 0 else np.nan,
                                                            100 * (gaps_data['P2'] < 1).mean() if len(gaps_data['P2']) > 0 else np.nan,
                                                            100 * (gaps_data['P3'] < 1).mean() if len(gaps_data['P3']) > 0 else np.nan], 'pct'))

        # Print fingerprint table
        for metric_name, vals, fmt in fingerprint_metrics:
            if fmt == 'int':
                log(f"{metric_name:<40} {int(vals[0]):>12,} {int(vals[1]):>12,} {int(vals[2]):>12,}", f)
            elif fmt == 'float1':
                log(f"{metric_name:<40} {vals[0]:>12.1f} {vals[1]:>12.1f} {vals[2]:>12.1f}", f)
            elif fmt == 'float3':
                log(f"{metric_name:<40} {vals[0]:>12.3f} {vals[1]:>12.3f} {vals[2]:>12.3f}", f)
            elif fmt == 'float4':
                log(f"{metric_name:<40} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}", f)
            elif fmt == 'pct':
                log(f"{metric_name:<40} {vals[0]:>11.1f}% {vals[1]:>11.1f}% {vals[2]:>11.1f}%", f)

        log("-" * 80, f)

        # =====================================================================
        # SECTION 16: USER CROSS-PLACEMENT BEHAVIOR
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 16: USER CROSS-PLACEMENT BEHAVIOR", f)
        log("=" * 80, f)

        # Detailed user overlap analysis
        log("\n--- User Placement Exposure ---", f)
        log(f"{'Category':<45} {'Count':>12} {'%':>10}", f)
        log("-" * 70, f)

        # Users in each combination
        only_p1 = p1_users - p2_users - p3_users
        only_p2 = p2_users - p1_users - p3_users
        only_p3 = p3_users - p1_users - p2_users
        p1_and_p2_only = (p1_users & p2_users) - p3_users
        p1_and_p3_only = (p1_users & p3_users) - p2_users
        p2_and_p3_only = (p2_users & p3_users) - p1_users
        all_three = p1_users & p2_users & p3_users

        total_users = len(p1_users | p2_users | p3_users)

        exposure_data = [
            ('Users seeing only P1', len(only_p1)),
            ('Users seeing only P2', len(only_p2)),
            ('Users seeing only P3', len(only_p3)),
            ('Users seeing exactly 2 placements', len(p1_and_p2_only) + len(p1_and_p3_only) + len(p2_and_p3_only)),
            ('  - P1 + P2 only', len(p1_and_p2_only)),
            ('  - P1 + P3 only', len(p1_and_p3_only)),
            ('  - P2 + P3 only', len(p2_and_p3_only)),
            ('Users seeing all 3 placements', len(all_three)),
        ]

        for label, count in exposure_data:
            pct = 100 * count / total_users if total_users > 0 else 0
            log(f"{label:<45} {count:>12,} {pct:>9.1f}%", f)

        log(f"\n{'Total unique users across all placements':<45} {total_users:>12,}", f)

        # User journey patterns: do users start in one placement then move to another?
        log("\n--- User Journey Patterns (First Placement Seen) ---", f)

        # For users in multiple placements, find their first placement
        multi_placement_users = (p1_users | p2_users | p3_users) - only_p1 - only_p2 - only_p3

        if len(multi_placement_users) > 0:
            # Get first timestamp per user per placement
            first_ts = {}
            for name, df in placements.items():
                user_first = df.groupby('USER_ID')['CREATED_AT'].min().reset_index()
                user_first.columns = ['USER_ID', f'first_{name}']
                first_ts[name] = user_first

            # Merge all first timestamps
            user_journey = first_ts['P1'].merge(first_ts['P2'], on='USER_ID', how='outer')
            user_journey = user_journey.merge(first_ts['P3'], on='USER_ID', how='outer')

            # For multi-placement users, determine first placement
            user_journey = user_journey[user_journey['USER_ID'].isin(multi_placement_users)]

            def get_first_placement(row):
                times = {
                    'P1': row['first_P1'] if pd.notna(row.get('first_P1')) else pd.Timestamp.max,
                    'P2': row['first_P2'] if pd.notna(row.get('first_P2')) else pd.Timestamp.max,
                    'P3': row['first_P3'] if pd.notna(row.get('first_P3')) else pd.Timestamp.max,
                }
                return min(times, key=times.get)

            user_journey['first_placement'] = user_journey.apply(get_first_placement, axis=1)

            first_placement_counts = user_journey['first_placement'].value_counts()
            log(f"{'First Placement':<20} {'Count':>12} {'%':>10}", f)
            log("-" * 45, f)
            for p in ['P1', 'P2', 'P3']:
                cnt = first_placement_counts.get(p, 0)
                pct = 100 * cnt / len(user_journey) if len(user_journey) > 0 else 0
                log(f"{p:<20} {cnt:>12,} {pct:>9.1f}%", f)
        else:
            log("No multi-placement users found.", f)

        # Cross-placement click rates
        log("\n--- Cross-Placement Click Rates ---", f)
        log(f"{'User Group':<45} {'Users':>10} {'Clicks':>10} {'CTR (%)':>10}", f)
        log("-" * 78, f)

        # Overall CTR by user exposure level
        user_click_totals = {}
        user_item_totals = {}

        for name, df in placements.items():
            clicks_by_user = df.groupby('USER_ID')['clicked'].sum()
            items_by_user = df.groupby('USER_ID').size()
            for user, clicks in clicks_by_user.items():
                user_click_totals[user] = user_click_totals.get(user, 0) + clicks
                user_item_totals[user] = user_item_totals.get(user, 0) + items_by_user[user]

        def calc_group_ctr(user_set):
            if len(user_set) == 0:
                return 0, 0, 0
            total_clicks = sum(user_click_totals.get(u, 0) for u in user_set)
            total_items = sum(user_item_totals.get(u, 0) for u in user_set)
            ctr = 100 * total_clicks / total_items if total_items > 0 else 0
            return len(user_set), total_clicks, ctr

        single_placement = only_p1 | only_p2 | only_p3
        two_placements = p1_and_p2_only | p1_and_p3_only | p2_and_p3_only

        groups = [
            ('Users in 1 placement only', single_placement),
            ('  - Only P1', only_p1),
            ('  - Only P2', only_p2),
            ('  - Only P3', only_p3),
            ('Users in exactly 2 placements', two_placements),
            ('  - P1 + P2 only', p1_and_p2_only),
            ('  - P1 + P3 only', p1_and_p3_only),
            ('  - P2 + P3 only', p2_and_p3_only),
            ('Users in all 3 placements', all_three),
        ]

        for label, user_set in groups:
            n_users, n_clicks, ctr = calc_group_ctr(user_set)
            log(f"{label:<45} {n_users:>10,} {n_clicks:>10,} {ctr:>9.2f}%", f)

        # =====================================================================
        # SECTION 17: CATEGORY DIVERSITY ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 17: CATEGORY DIVERSITY ANALYSIS", f)
        log("=" * 80, f)

        # Extract category# values from CATEGORIES column
        log("\nParsing category information from catalog...", f)

        def extract_categories(cat_list):
            """Extract category# values from CATEGORIES list."""
            if cat_list is None or (isinstance(cat_list, float) and pd.isna(cat_list)):
                return []
            if isinstance(cat_list, str):
                import json
                try:
                    cat_list = json.loads(cat_list)
                except:
                    return []
            return [c.split('#')[1] for c in cat_list if isinstance(c, str) and c.startswith('category#')]

        def extract_departments(cat_list):
            """Extract department# values from CATEGORIES list."""
            if cat_list is None or (isinstance(cat_list, float) and pd.isna(cat_list)):
                return []
            if isinstance(cat_list, str):
                import json
                try:
                    cat_list = json.loads(cat_list)
                except:
                    return []
            return [c.split('#')[1] for c in cat_list if isinstance(c, str) and c.startswith('department#')]

        def extract_brands(cat_list):
            """Extract brand# values from CATEGORIES list."""
            if cat_list is None or (isinstance(cat_list, float) and pd.isna(cat_list)):
                return []
            if isinstance(cat_list, str):
                import json
                try:
                    cat_list = json.loads(cat_list)
                except:
                    return []
            return [c.split('#')[1] for c in cat_list if isinstance(c, str) and c.startswith('brand#')]

        # Apply extraction to catalog
        catalog['categories_list'] = catalog['CATEGORIES'].apply(extract_categories)
        catalog['departments_list'] = catalog['CATEGORIES'].apply(extract_departments)
        catalog['brands_list'] = catalog['CATEGORIES'].apply(extract_brands)
        catalog['primary_category'] = catalog['categories_list'].apply(lambda x: x[0] if len(x) > 0 else None)
        catalog['primary_department'] = catalog['departments_list'].apply(lambda x: x[0] if len(x) > 0 else None)
        catalog['primary_brand'] = catalog['brands_list'].apply(lambda x: x[0] if len(x) > 0 else None)

        # Merge category info into placement data
        catalog_subset = catalog[['PRODUCT_ID', 'primary_category', 'primary_department', 'primary_brand']].copy()
        catalog_subset['PRODUCT_ID'] = catalog_subset['PRODUCT_ID'].str.lower()

        for name in placements:
            placements[name] = placements[name].merge(catalog_subset, left_on='product_id', right_on='PRODUCT_ID', how='left')

        log(f"Category coverage: {placements['P1']['primary_category'].notna().sum():,}/{len(placements['P1']):,} P1 items have category", f)
        log(f"Category coverage: {placements['P2']['primary_category'].notna().sum():,}/{len(placements['P2']):,} P2 items have category", f)
        log(f"Category coverage: {placements['P3']['primary_category'].notna().sum():,}/{len(placements['P3']):,} P3 items have category", f)

        # Category homogeneity per session
        log("\n--- Category Homogeneity Per Session ---", f)
        log(f"{'Metric':<45} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 85, f)

        cat_per_session = {}
        dept_per_session = {}
        brand_per_session = {}

        for name, df in placements.items():
            cat_per_session[name] = df.groupby('auction_id')['primary_category'].nunique()
            dept_per_session[name] = df.groupby('auction_id')['primary_department'].nunique()
            brand_per_session[name] = df.groupby('auction_id')['primary_brand'].nunique()

        homogeneity_stats = [
            ('Mean unique categories/session', lambda name: cat_per_session[name].mean()),
            ('Median unique categories/session', lambda name: cat_per_session[name].median()),
            ('% sessions with 1 category', lambda name: 100 * (cat_per_session[name] == 1).mean()),
            ('% sessions with 2 categories', lambda name: 100 * (cat_per_session[name] == 2).mean()),
            ('% sessions with 3+ categories', lambda name: 100 * (cat_per_session[name] >= 3).mean()),
            ('', lambda name: None),  # spacer
            ('Mean unique departments/session', lambda name: dept_per_session[name].mean()),
            ('Median unique departments/session', lambda name: dept_per_session[name].median()),
            ('% sessions with 1 department', lambda name: 100 * (dept_per_session[name] == 1).mean()),
            ('% sessions with 2+ departments', lambda name: 100 * (dept_per_session[name] >= 2).mean()),
            ('', lambda name: None),  # spacer
            ('Mean unique brands/session', lambda name: brand_per_session[name].mean()),
            ('Median unique brands/session', lambda name: brand_per_session[name].median()),
            ('% sessions with 1 brand', lambda name: 100 * (brand_per_session[name] == 1).mean()),
            ('% sessions with 5+ brands', lambda name: 100 * (brand_per_session[name] >= 5).mean()),
        ]

        for stat_name, stat_func in homogeneity_stats:
            if stat_name == '':
                log("", f)
                continue
            vals = [stat_func(name) for name in ['P1', 'P2', 'P3']]
            log(f"{stat_name:<45} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f}", f)

        # Top categories per placement
        log("\n--- Top 10 Categories by Placement ---", f)

        for name, df in placements.items():
            log(f"\n{name} Top Categories:", f)
            cat_counts = df['primary_category'].value_counts().head(10)
            total_with_cat = df['primary_category'].notna().sum()
            log(f"{'Rank':<6} {'Category ID':<35} {'Count':>10} {'%':>10}", f)
            log("-" * 65, f)
            for rank, (cat, cnt) in enumerate(cat_counts.items(), 1):
                pct = 100 * cnt / total_with_cat if total_with_cat > 0 else 0
                cat_str = str(cat)[:33] if cat else 'N/A'
                log(f"{rank:<6} {cat_str:<35} {cnt:>10,} {pct:>9.1f}%", f)

        # Category concentration (share of top N categories)
        log("\n--- Category Concentration ---", f)
        log(f"{'Metric':<45} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 85, f)

        concentration_metrics = [
            ('Top 1 category share (%)', lambda df: 100 * df['primary_category'].value_counts().iloc[0] / df['primary_category'].notna().sum() if df['primary_category'].notna().sum() > 0 else 0),
            ('Top 5 categories share (%)', lambda df: 100 * df['primary_category'].value_counts().head(5).sum() / df['primary_category'].notna().sum() if df['primary_category'].notna().sum() > 0 else 0),
            ('Top 10 categories share (%)', lambda df: 100 * df['primary_category'].value_counts().head(10).sum() / df['primary_category'].notna().sum() if df['primary_category'].notna().sum() > 0 else 0),
            ('Total unique categories', lambda df: df['primary_category'].nunique()),
        ]

        for stat_name, stat_func in concentration_metrics:
            vals = [stat_func(placements[name]) for name in ['P1', 'P2', 'P3']]
            if 'unique' in stat_name.lower():
                log(f"{stat_name:<45} {int(vals[0]):>12,} {int(vals[1]):>12,} {int(vals[2]):>12,}", f)
            else:
                log(f"{stat_name:<45} {vals[0]:>12.1f} {vals[1]:>12.1f} {vals[2]:>12.1f}", f)

        # =====================================================================
        # SECTION 18: TIME SPENT ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 18: TIME SPENT ANALYSIS", f)
        log("=" * 80, f)

        log("\nNote: CREATED_AT in auctions_users is at the auction level, not item level.", f)
        log("All items in a session share the same timestamp, so within-session duration = 0.", f)
        log("Inter-auction gap (Section 4) is the closest proxy for user engagement time.", f)

        # Summarize inter-auction gap metrics (already computed in Section 4)
        log("\n--- Inter-Auction Gap Summary (from Section 4) ---", f)
        log("This measures time between consecutive sessions for the same user.", f)
        log(f"{'Metric':<40} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 88, f)

        gap_summary_stats = [
            ('N user transitions', lambda name: len(gaps_data[name])),
            ('Mean inter-auction gap (s)', lambda name: gaps_data[name].mean() if len(gaps_data[name]) > 0 else np.nan),
            ('Median inter-auction gap (s)', lambda name: gaps_data[name].median() if len(gaps_data[name]) > 0 else np.nan),
            ('P25 inter-auction gap (s)', lambda name: gaps_data[name].quantile(0.25) if len(gaps_data[name]) > 0 else np.nan),
            ('P75 inter-auction gap (s)', lambda name: gaps_data[name].quantile(0.75) if len(gaps_data[name]) > 0 else np.nan),
            ('% rapid-fire (<1s)', lambda name: 100 * (gaps_data[name] < 1).mean() if len(gaps_data[name]) > 0 else np.nan),
            ('% quick (1-5s)', lambda name: 100 * ((gaps_data[name] >= 1) & (gaps_data[name] < 5)).mean() if len(gaps_data[name]) > 0 else np.nan),
            ('% moderate (5-30s)', lambda name: 100 * ((gaps_data[name] >= 5) & (gaps_data[name] < 30)).mean() if len(gaps_data[name]) > 0 else np.nan),
            ('% new session (>5m)', lambda name: 100 * (gaps_data[name] >= 300).mean() if len(gaps_data[name]) > 0 else np.nan),
        ]

        for stat_name, stat_func in gap_summary_stats:
            vals = [stat_func(name) for name in ['P1', 'P2', 'P3']]
            if 'N user' in stat_name:
                log(f"{stat_name:<40} {int(vals[0]):>15,} {int(vals[1]):>15,} {int(vals[2]):>15,}", f)
            else:
                log(f"{stat_name:<40} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # User session counts as proxy for engagement
        log("\n--- User Session Engagement ---", f)
        log("Sessions per user as a proxy for time spent in each placement.", f)
        log(f"{'Metric':<40} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 88, f)

        sessions_per_user_engagement = [
            ('Mean sessions per user', lambda name: user_auctions[name].mean()),
            ('Median sessions per user', lambda name: user_auctions[name].median()),
            ('P75 sessions per user', lambda name: user_auctions[name].quantile(0.75)),
            ('P90 sessions per user', lambda name: user_auctions[name].quantile(0.90)),
            ('Max sessions per user', lambda name: user_auctions[name].max()),
            ('% users with 1 session', lambda name: 100 * (user_auctions[name] == 1).mean()),
            ('% users with 2-5 sessions', lambda name: 100 * ((user_auctions[name] >= 2) & (user_auctions[name] <= 5)).mean()),
            ('% users with 6+ sessions', lambda name: 100 * (user_auctions[name] >= 6).mean()),
        ]

        for stat_name, stat_func in sessions_per_user_engagement:
            vals = [stat_func(name) for name in ['P1', 'P2', 'P3']]
            log(f"{stat_name:<40} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # Items seen per user as another engagement proxy
        log("\n--- Items Seen Per User ---", f)
        log(f"{'Metric':<40} {'P1':>15} {'P2':>15} {'P3':>15}", f)
        log("-" * 88, f)

        items_per_user = {name: df.groupby('USER_ID').size() for name, df in placements.items()}

        items_stats = [
            ('Mean items per user', lambda name: items_per_user[name].mean()),
            ('Median items per user', lambda name: items_per_user[name].median()),
            ('P75 items per user', lambda name: items_per_user[name].quantile(0.75)),
            ('P90 items per user', lambda name: items_per_user[name].quantile(0.90)),
            ('Max items per user', lambda name: items_per_user[name].max()),
        ]

        for stat_name, stat_func in items_stats:
            vals = [stat_func(name) for name in ['P1', 'P2', 'P3']]
            log(f"{stat_name:<40} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}", f)

        # Multi-placement user session allocation
        log("\n--- Session Allocation for Multi-Placement Users ---", f)

        if len(multi_placement_users) > 0:
            # For users in all 3 placements
            if len(all_three) > 0:
                log(f"\nUsers in all 3 placements (n={len(all_three):,}):", f)
                log(f"{'Placement':<20} {'Mean Sessions':>15} {'% of Total':>15}", f)
                log("-" * 52, f)

                all_three_sessions = {}
                for p in ['P1', 'P2', 'P3']:
                    sessions = user_auctions[p]
                    sessions_subset = sessions[sessions.index.isin(all_three)]
                    all_three_sessions[p] = sessions_subset.mean() if len(sessions_subset) > 0 else 0

                total_sessions = sum(all_three_sessions.values())
                for p in ['P1', 'P2', 'P3']:
                    pct = 100 * all_three_sessions[p] / total_sessions if total_sessions > 0 else 0
                    log(f"{p:<20} {all_three_sessions[p]:>15.2f} {pct:>14.1f}%", f)
                log(f"{'Total':<20} {total_sessions:>15.2f} {'100.0':>14}%", f)

            # For users in exactly 2 placements
            for combo, combo_users in [('P1+P2', p1_and_p2_only), ('P1+P3', p1_and_p3_only), ('P2+P3', p2_and_p3_only)]:
                if len(combo_users) > 10:
                    p_list = combo.split('+')
                    mean_sessions = {}
                    for p in p_list:
                        sessions = user_auctions[p]
                        sessions_subset = sessions[sessions.index.isin(combo_users)]
                        mean_sessions[p] = sessions_subset.mean() if len(sessions_subset) > 0 else 0

                    total_sessions = sum(mean_sessions.values())

                    log(f"\nUsers in {combo} only (n={len(combo_users):,}):", f)
                    log(f"{'Placement':<20} {'Mean Sessions':>15} {'% of Total':>15}", f)
                    log("-" * 52, f)
                    for p in p_list:
                        pct = 100 * mean_sessions[p] / total_sessions if total_sessions > 0 else 0
                        log(f"{p:<20} {mean_sessions[p]:>15.2f} {pct:>14.1f}%", f)
        else:
            log("No multi-placement users found for session allocation analysis.", f)

        # =====================================================================
        # SECTION 19: PLACEMENT TYPE VALIDATION
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 19: PLACEMENT TYPE VALIDATION", f)
        log("=" * 80, f)
        log("\nValidating hypotheses about placement types based on behavioral patterns:", f)
        log("- P1 = Search Page: More scrolling, more exploration/tinkering", f)
        log("- P3 = Category Page: Fixed layout, no tinkering (users know what they want)", f)
        log("- P2 = Product Page: Should appear AFTER search or category page in journey", f)

        # ---------------------------------------------------------------------
        # 19.1: SCROLLING/TINKERING METRICS
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("19.1: SCROLLING/TINKERING METRICS", f)
        log("-" * 80, f)
        log("\nHypothesis: Search page (P1) shows more exploration, Category page (P3) is fixed.", f)

        log("\n--- Scrolling Behavior ---", f)
        log(f"{'Metric':<45} {'P1':>12} {'P2':>12} {'P3':>12} {'Expected':<20}", f)
        log("-" * 95, f)

        scrolling_metrics = [
            ('Mean items per session',
             [session_lengths['P1'].mean(), session_lengths['P2'].mean(), session_lengths['P3'].mean()],
             'P1 > P3'),
            ('Variance in items per session',
             [session_lengths['P1'].var(), session_lengths['P2'].var(), session_lengths['P3'].var()],
             'P1 high, P3 low'),
            ('Max position cap (99th pctl)',
             [max_pos_per_session['P1'].quantile(0.99), max_pos_per_session['P2'].quantile(0.99), max_pos_per_session['P3'].quantile(0.99)],
             'P3 ~ 10-12'),
            ('% sessions with exactly 4 items',
             [100 * (session_lengths['P1'] == 4).mean(), 100 * (session_lengths['P2'] == 4).mean(), 100 * (session_lengths['P3'] == 4).mean()],
             'P3 highest (2x2 grid?)'),
        ]

        for metric_name, vals, expected in scrolling_metrics:
            log(f"{metric_name:<45} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f} {expected:<20}", f)

        # Tinkering Index = (unique_products_viewed / total_items) * (category_diversity)
        log("\n--- Tinkering Index ---", f)
        log("Tinkering Index = (unique_products / total_items) * (unique_categories / session)", f)
        log("High = more exploration (search), Low = focused (category)", f)
        log(f"{'Metric':<45} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 85, f)

        tinkering_indices = {}
        for name, df in placements.items():
            # Per session: unique_products / items * unique_categories
            session_stats = df.groupby('auction_id').agg({
                'product_id': 'nunique',
                'primary_category': 'nunique'
            }).rename(columns={'product_id': 'unique_products', 'primary_category': 'unique_cats'})

            session_stats = session_stats.merge(
                session_lengths[name].reset_index().rename(columns={'auction_id': 'auction_id', 0: 'n_items'}),
                left_index=True, right_on='auction_id'
            )

            # Handle cases where n_items might be 0 or NaN
            session_stats['product_ratio'] = session_stats['unique_products'] / session_stats['n_items'].replace(0, np.nan)
            session_stats['tinkering_index'] = session_stats['product_ratio'] * session_stats['unique_cats']
            tinkering_indices[name] = session_stats['tinkering_index']

        tinkering_stats = [
            ('Mean Tinkering Index', [tinkering_indices['P1'].mean(), tinkering_indices['P2'].mean(), tinkering_indices['P3'].mean()]),
            ('Median Tinkering Index', [tinkering_indices['P1'].median(), tinkering_indices['P2'].median(), tinkering_indices['P3'].median()]),
            ('P75 Tinkering Index', [tinkering_indices['P1'].quantile(0.75), tinkering_indices['P2'].quantile(0.75), tinkering_indices['P3'].quantile(0.75)]),
        ]

        for metric_name, vals in tinkering_stats:
            log(f"{metric_name:<45} {vals[0]:>12.3f} {vals[1]:>12.3f} {vals[2]:>12.3f}", f)

        # ---------------------------------------------------------------------
        # 19.2: LAYOUT FIXEDNESS ANALYSIS
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("19.2: LAYOUT FIXEDNESS ANALYSIS", f)
        log("-" * 80, f)
        log("\nHypothesis: Category pages have fixed grid layouts (4, 6, 8 items).", f)

        log("\n--- Session Length Distribution ---", f)
        log(f"{'Items':<15} {'P1 %':>12} {'P1 N':>10} {'P2 %':>12} {'P2 N':>10} {'P3 %':>12} {'P3 N':>10} {'Expected':<15}", f)
        log("-" * 110, f)

        length_buckets = [
            ((1, 4), 'P3 highest'),
            ((5, 8), 'P2 highest'),
            ((9, 12), 'P3 drop'),
            ((13, float('inf')), 'P3 = 0%'),
        ]

        for (low, high), expected in length_buckets:
            if high == float('inf'):
                label = f"{low}+ items"
                mask_func = lambda x, lo=low: x >= lo
            else:
                label = f"{low}-{int(high)} items"
                mask_func = lambda x, lo=low, hi=high: (x >= lo) & (x <= hi)

            vals = []
            for name in ['P1', 'P2', 'P3']:
                mask = mask_func(session_lengths[name])
                cnt = mask.sum()
                pct = 100 * cnt / len(session_lengths[name]) if len(session_lengths[name]) > 0 else 0
                vals.extend([pct, cnt])
            log(f"{label:<15} {vals[0]:>11.1f}% {vals[1]:>10,} {vals[2]:>11.1f}% {vals[3]:>10,} {vals[4]:>11.1f}% {vals[5]:>10,} {expected:<15}", f)

        # Layout Entropy
        log("\n--- Layout Entropy ---", f)
        log("Entropy measures variability in session lengths. Low = fixed grid pattern.", f)
        log(f"{'Placement':<15} {'Entropy':>15} {'Interpretation':<30}", f)
        log("-" * 62, f)

        from scipy.stats import entropy as scipy_entropy

        for name in ['P1', 'P2', 'P3']:
            length_counts = session_lengths[name].value_counts(normalize=True).sort_index()
            ent = scipy_entropy(length_counts.values, base=2) if len(length_counts) > 1 else 0
            interp = "Fixed layout" if ent < 2 else ("Variable" if ent < 3 else "Highly variable")
            log(f"{name:<15} {ent:>15.3f} {interp:<30}", f)

        # Specific grid sizes
        log("\n--- Common Grid Sizes ---", f)
        log(f"{'Grid Size':<15} {'P1 %':>12} {'P2 %':>12} {'P3 %':>12}", f)
        log("-" * 55, f)

        for grid in [4, 6, 8, 10, 12]:
            vals = []
            for name in ['P1', 'P2', 'P3']:
                pct = 100 * (session_lengths[name] == grid).mean()
                vals.append(pct)
            log(f"{grid:<15} {vals[0]:>11.1f}% {vals[1]:>11.1f}% {vals[2]:>11.1f}%", f)

        # ---------------------------------------------------------------------
        # 19.3: USER JOURNEY SEQUENCING
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("19.3: USER JOURNEY SEQUENCING", f)
        log("-" * 80, f)
        log("\nHypothesis: Product pages (P2) appear AFTER search (P1) or category (P3).", f)

        # Build user journey sequences
        log("\n--- First-to-Second Placement Transition ---", f)

        # Combine all auctions with timestamps
        all_auctions = []
        for name, df in placements.items():
            auctions = df[['auction_id', 'USER_ID', 'CREATED_AT']].drop_duplicates()
            auctions['placement'] = name
            all_auctions.append(auctions)

        all_auctions_df = pd.concat(all_auctions, ignore_index=True)
        all_auctions_df = all_auctions_df.sort_values(['USER_ID', 'CREATED_AT'])

        # Compute transitions
        all_auctions_df['next_placement'] = all_auctions_df.groupby('USER_ID')['placement'].shift(-1)
        transitions = all_auctions_df[all_auctions_df['next_placement'].notna()].copy()

        transition_counts = transitions.groupby(['placement', 'next_placement']).size().unstack(fill_value=0)

        log(f"{'Transition':<20} {'Count':>12} {'% of From':>12} {'Expected':<25}", f)
        log("-" * 75, f)

        expected_transitions = {
            ('P1', 'P2'): 'Common (search -> product)',
            ('P1', 'P3'): 'Less common',
            ('P2', 'P1'): 'Low (product -> search unlikely)',
            ('P2', 'P3'): 'Low',
            ('P3', 'P1'): 'Low',
            ('P3', 'P2'): 'Common (category -> product)',
        }

        for from_p in ['P1', 'P2', 'P3']:
            from_total = transition_counts.loc[from_p].sum() if from_p in transition_counts.index else 0
            for to_p in ['P1', 'P2', 'P3']:
                if from_p == to_p:
                    continue
                cnt = transition_counts.loc[from_p, to_p] if (from_p in transition_counts.index and to_p in transition_counts.columns) else 0
                pct = 100 * cnt / from_total if from_total > 0 else 0
                expected = expected_transitions.get((from_p, to_p), '')
                log(f"{from_p} -> {to_p:<14} {cnt:>12,} {pct:>11.1f}% {expected:<25}", f)

        # Placement Order in Session for users seeing all 3
        log("\n--- Placement Order for Users Seeing All 3 Placements ---", f)

        if len(all_three) > 0:
            # For users in all 3 placements, get first timestamp per placement
            first_ts_all_three = {}
            for name, df in placements.items():
                user_first = df[df['USER_ID'].isin(all_three)].groupby('USER_ID')['CREATED_AT'].min().reset_index()
                user_first.columns = ['USER_ID', f'first_{name}']
                first_ts_all_three[name] = user_first

            # Merge all first timestamps
            user_order = first_ts_all_three['P1'].merge(first_ts_all_three['P2'], on='USER_ID', how='outer')
            user_order = user_order.merge(first_ts_all_three['P3'], on='USER_ID', how='outer')
            user_order = user_order.dropna()

            def get_order(row):
                times = {'P1': row['first_P1'], 'P2': row['first_P2'], 'P3': row['first_P3']}
                sorted_times = sorted(times.items(), key=lambda x: x[1])
                return tuple([p for p, t in sorted_times])

            user_order['order'] = user_order.apply(get_order, axis=1)
            order_counts = user_order['order'].value_counts()

            log(f"{'Order (First -> Last)':<30} {'Count':>12} {'%':>10}", f)
            log("-" * 55, f)

            for order, cnt in order_counts.head(10).items():
                pct = 100 * cnt / len(user_order) if len(user_order) > 0 else 0
                order_str = ' -> '.join(order)
                log(f"{order_str:<30} {cnt:>12,} {pct:>9.1f}%", f)

            # Summary stats
            log(f"\nTotal users seeing all 3 placements: {len(user_order):,}", f)

            # Which placement comes first most often?
            first_placement = user_order['order'].apply(lambda x: x[0])
            log(f"\n{'First Placement':<20} {'%':>10}", f)
            log("-" * 32, f)
            for p in ['P1', 'P2', 'P3']:
                pct = 100 * (first_placement == p).mean()
                log(f"{p:<20} {pct:>9.1f}%", f)

            # Which placement comes last most often?
            last_placement = user_order['order'].apply(lambda x: x[-1])
            log(f"\n{'Last Placement':<20} {'%':>10}", f)
            log("-" * 32, f)
            for p in ['P1', 'P2', 'P3']:
                pct = 100 * (last_placement == p).mean()
                log(f"{p:<20} {pct:>9.1f}%", f)
        else:
            log("No users found seeing all 3 placements.", f)

        # ---------------------------------------------------------------------
        # 19.4: ENGAGEMENT PATTERN METRICS
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("19.4: ENGAGEMENT PATTERN METRICS", f)
        log("-" * 80, f)
        log("\nHypothesis: Search = exploratory (low initial CTR), Category = intentional (higher CTR).", f)

        log("\n--- CTR at Position 1 ---", f)
        log(f"{'Metric':<45} {'P1':>12} {'P2':>12} {'P3':>12} {'Expected':<20}", f)
        log("-" * 95, f)

        ctr_metrics = []

        # CTR at position 1
        for name, df in placements.items():
            pos1 = df[df['position'] == 1]
            ctr = 100 * pos1['clicked'].mean() if len(pos1) > 0 else np.nan
            ctr_metrics.append(ctr)
        log(f"{'CTR at position 1':<45} {ctr_metrics[0]:>11.2f}% {ctr_metrics[1]:>11.2f}% {ctr_metrics[2]:>11.2f}% {'P3 > P1':<20}", f)

        # CTR at positions 1-4
        ctr_14 = []
        for name, df in placements.items():
            pos14 = df[df['position'] <= 4]
            ctr = 100 * pos14['clicked'].mean() if len(pos14) > 0 else np.nan
            ctr_14.append(ctr)
        log(f"{'CTR at positions 1-4':<45} {ctr_14[0]:>11.2f}% {ctr_14[1]:>11.2f}% {ctr_14[2]:>11.2f}% {'P3 highest':<20}", f)

        # Click depth (mean position of clicks)
        click_depth = []
        for name in ['P1', 'P2', 'P3']:
            if len(clicked_items[name]) > 0:
                click_depth.append(clicked_items[name]['position'].mean())
            else:
                click_depth.append(np.nan)
        log(f"{'Click depth (mean position of clicks)':<45} {click_depth[0]:>12.2f} {click_depth[1]:>12.2f} {click_depth[2]:>12.2f} {'P1 deeper':<20}", f)

        log("\n--- Multi-Click Sessions ---", f)
        log(f"{'Metric':<45} {'P1':>12} {'P2':>12} {'P3':>12} {'Expected':<20}", f)
        log("-" * 95, f)

        # % sessions with 2+ clicks
        multi_click_pct = []
        for name, df in placements.items():
            session_clicks = df.groupby('auction_id')['clicked'].sum()
            pct = 100 * (session_clicks >= 2).mean() if len(session_clicks) > 0 else 0
            multi_click_pct.append(pct)
        log(f"{'% sessions with 2+ clicks':<45} {multi_click_pct[0]:>11.1f}% {multi_click_pct[1]:>11.1f}% {multi_click_pct[2]:>11.1f}% {'P1 > P3':<20}", f)

        # Mean clicks per session (if any click)
        mean_clicks = []
        for name, df in placements.items():
            session_clicks = df.groupby('auction_id')['clicked'].sum()
            sessions_with_clicks = session_clicks[session_clicks > 0]
            mean_c = sessions_with_clicks.mean() if len(sessions_with_clicks) > 0 else np.nan
            mean_clicks.append(mean_c)
        log(f"{'Mean clicks per session (if any)':<45} {mean_clicks[0]:>12.2f} {mean_clicks[1]:>12.2f} {mean_clicks[2]:>12.2f} {'P1 > P3':<20}", f)

        # ---------------------------------------------------------------------
        # 19.5: SUMMARY VALIDATION TABLE
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("19.5: SUMMARY VALIDATION TABLE", f)
        log("-" * 80, f)

        log("\n" + "=" * 100, f)
        log("=== PLACEMENT TYPE VALIDATION SUMMARY ===", f)
        log("=" * 100, f)

        log(f"\n{'Hypothesis':<55} {'Evidence For':<25} {'Verdict':<15}", f)
        log("-" * 100, f)

        # P3 = Category Page validation
        log("\nP3 = Category Page", f)

        # Fixed layout: Max pos = 12, low entropy
        p3_max_99 = max_pos_per_session['P3'].quantile(0.99)
        p3_entropy = scipy_entropy(session_lengths['P3'].value_counts(normalize=True).values, base=2)
        p3_fixed_evidence = f"99th pctl pos={p3_max_99:.0f}, entropy={p3_entropy:.2f}"
        p3_fixed_verdict = "SUPPORTED" if p3_max_99 <= 12 and p3_entropy < 3 else "INCONCLUSIVE"
        log(f"  {'Fixed layout':<53} {p3_fixed_evidence:<25} {p3_fixed_verdict:<15}", f)

        # Single category/session: 99.95%
        p3_single_cat = 100 * (cat_per_session['P3'] == 1).mean()
        p3_cat_evidence = f"{p3_single_cat:.1f}% single-cat"
        p3_cat_verdict = "SUPPORTED" if p3_single_cat > 95 else "INCONCLUSIVE"
        log(f"  {'Single category/session':<53} {p3_cat_evidence:<25} {p3_cat_verdict:<15}", f)

        # No deep scrolling: 0% reach pos 20+
        p3_deep = 100 * (max_pos_per_session['P3'] >= 20).mean()
        p3_deep_evidence = f"{p3_deep:.1f}% reach pos 20+"
        p3_deep_verdict = "SUPPORTED" if p3_deep < 1 else "NOT SUPPORTED"
        log(f"  {'No deep scrolling':<53} {p3_deep_evidence:<25} {p3_deep_verdict:<15}", f)

        # P1 = Search Page validation
        log("\nP1 = Search Page", f)

        # More exploration: higher category diversity
        p1_cats = cat_per_session['P1'].mean()
        p3_cats_mean = cat_per_session['P3'].mean()
        p1_explore_evidence = f"{p1_cats:.2f} cats vs P3's {p3_cats_mean:.2f}"
        p1_explore_verdict = "SUPPORTED" if p1_cats > p3_cats_mean else "NOT SUPPORTED"
        log(f"  {'More exploration (categories)':<53} {p1_explore_evidence:<25} {p1_explore_verdict:<15}", f)

        # More scrolling: reach pos 20+
        p1_deep = 100 * (max_pos_per_session['P1'] >= 20).mean()
        p1_scroll_evidence = f"{p1_deep:.1f}% reach pos 20+"
        p1_scroll_verdict = "SUPPORTED" if p1_deep > 5 else "INCONCLUSIVE"
        log(f"  {'More scrolling':<53} {p1_scroll_evidence:<25} {p1_scroll_verdict:<15}", f)

        # Variable session lengths (high entropy)
        p1_entropy = scipy_entropy(session_lengths['P1'].value_counts(normalize=True).values, base=2)
        p1_entropy_evidence = f"Entropy={p1_entropy:.2f}"
        p1_entropy_verdict = "SUPPORTED" if p1_entropy > p3_entropy else "NOT SUPPORTED"
        log(f"  {'Variable session lengths':<53} {p1_entropy_evidence:<25} {p1_entropy_verdict:<15}", f)

        # Higher tinkering index
        p1_tinkering = tinkering_indices['P1'].mean()
        p3_tinkering = tinkering_indices['P3'].mean()
        p1_tinkering_evidence = f"{p1_tinkering:.3f} vs P3's {p3_tinkering:.3f}"
        p1_tinkering_verdict = "SUPPORTED" if p1_tinkering > p3_tinkering else "NOT SUPPORTED"
        log(f"  {'Higher tinkering index':<53} {p1_tinkering_evidence:<25} {p1_tinkering_verdict:<15}", f)

        # P2 = Product Page validation
        log("\nP2 = Product Page", f)

        # Single brand focus: 99.2% 1-brand
        p2_single_brand = 100 * (brand_per_session['P2'] == 1).mean()
        p2_brand_evidence = f"{p2_single_brand:.1f}% single-brand"
        p2_brand_verdict = "SUPPORTED" if p2_single_brand > 90 else "INCONCLUSIVE"
        log(f"  {'Single brand focus':<53} {p2_brand_evidence:<25} {p2_brand_verdict:<15}", f)

        # Appears after P1/P3: check journey analysis
        if len(all_three) > 0:
            p2_last_pct = 100 * (last_placement == 'P2').mean()
            p2_journey_evidence = f"P2 last in {p2_last_pct:.1f}%"
            p2_journey_verdict = "SUPPORTED" if p2_last_pct > 30 else "INCONCLUSIVE"
        else:
            p2_journey_evidence = "Insufficient data"
            p2_journey_verdict = "INCONCLUSIVE"
        log(f"  {'Appears after P1/P3':<53} {p2_journey_evidence:<25} {p2_journey_verdict:<15}", f)

        # Higher engagement (CTR)
        p2_ctr = 100 * p2_data['clicked'].mean()
        p1_ctr = 100 * p1_data['clicked'].mean()
        p3_ctr = 100 * p3_data['clicked'].mean()
        p2_ctr_evidence = f"{p2_ctr:.2f}% (P1={p1_ctr:.2f}%, P3={p3_ctr:.2f}%)"
        p2_ctr_verdict = "SUPPORTED" if p2_ctr > p1_ctr and p2_ctr > p3_ctr else "NOT SUPPORTED"
        log(f"  {'Higher engagement (CTR)':<53} {p2_ctr_evidence:<25} {p2_ctr_verdict:<15}", f)

        log("\n" + "=" * 100, f)

        # Final summary table with all key metrics
        log("\n--- Final Metrics Comparison Table ---", f)
        log(f"{'Metric':<50} {'P1':>12} {'P2':>12} {'P3':>12}", f)
        log("-" * 90, f)

        final_metrics = [
            ('% sessions with 1 category', 100 * (cat_per_session['P1'] == 1).mean(), 100 * (cat_per_session['P2'] == 1).mean(), 100 * (cat_per_session['P3'] == 1).mean()),
            ('% sessions with 1 brand', 100 * (brand_per_session['P1'] == 1).mean(), 100 * (brand_per_session['P2'] == 1).mean(), 100 * (brand_per_session['P3'] == 1).mean()),
            ('Max max position', max_pos_per_session['P1'].max(), max_pos_per_session['P2'].max(), max_pos_per_session['P3'].max()),
            ('99th percentile max position', max_pos_per_session['P1'].quantile(0.99), max_pos_per_session['P2'].quantile(0.99), max_pos_per_session['P3'].quantile(0.99)),
            ('% reaching pos 20+', 100 * (max_pos_per_session['P1'] >= 20).mean(), 100 * (max_pos_per_session['P2'] >= 20).mean(), 100 * (max_pos_per_session['P3'] >= 20).mean()),
            ('% rapid-fire (<1s)', 100 * (gaps_data['P1'] < 1).mean() if len(gaps_data['P1']) > 0 else np.nan, 100 * (gaps_data['P2'] < 1).mean() if len(gaps_data['P2']) > 0 else np.nan, 100 * (gaps_data['P3'] < 1).mean() if len(gaps_data['P3']) > 0 else np.nan),
            ('Session length entropy (bits)', scipy_entropy(session_lengths['P1'].value_counts(normalize=True).values, base=2), scipy_entropy(session_lengths['P2'].value_counts(normalize=True).values, base=2), scipy_entropy(session_lengths['P3'].value_counts(normalize=True).values, base=2)),
            ('Mean tinkering index', tinkering_indices['P1'].mean(), tinkering_indices['P2'].mean(), tinkering_indices['P3'].mean()),
            ('CTR (%)', p1_ctr, p2_ctr, p3_ctr),
            ('CTR at position 1 (%)', ctr_metrics[0], ctr_metrics[1], ctr_metrics[2]),
            ('Mean click position', click_depth[0], click_depth[1], click_depth[2]),
            ('% multi-click sessions', multi_click_pct[0], multi_click_pct[1], multi_click_pct[2]),
        ]

        for metric_name, v1, v2, v3 in final_metrics:
            if 'Max max' in metric_name:
                log(f"{metric_name:<50} {int(v1):>12} {int(v2):>12} {int(v3):>12}", f)
            else:
                log(f"{metric_name:<50} {v1:>12.2f} {v2:>12.2f} {v3:>12.2f}", f)

        log("-" * 90, f)

        # =====================================================================
        # SECTION 20: P3 AUCTION ARCHITECTURE
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 20: P3 AUCTION ARCHITECTURE", f)
        log("=" * 80, f)
        log("\nInvestigating P3's internal structure: auction clustering, pagination, and timing.", f)

        # ---------------------------------------------------------------------
        # 20.1: AUCTION CLUSTERING BY TIMESTAMP
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("20.1: AUCTION CLUSTERING BY TIMESTAMP", f)
        log("-" * 80, f)
        log("\nGroup P3 auctions that fire within 0.1s of each other (same page load).", f)

        # Get P3 auctions with timestamps per user
        p3_auctions = p3_data[['auction_id', 'USER_ID', 'CREATED_AT']].drop_duplicates()
        p3_auctions = p3_auctions.sort_values(['USER_ID', 'CREATED_AT'])

        # Compute time gap to previous auction for same user
        p3_auctions['prev_time'] = p3_auctions.groupby('USER_ID')['CREATED_AT'].shift(1)
        p3_auctions['gap_ms'] = (p3_auctions['CREATED_AT'] - p3_auctions['prev_time']).dt.total_seconds() * 1000

        # Assign cluster IDs: new cluster if gap > 100ms or first auction
        p3_auctions['new_cluster'] = (p3_auctions['gap_ms'].isna() | (p3_auctions['gap_ms'] > 100)).astype(int)
        p3_auctions['cluster_id'] = p3_auctions.groupby('USER_ID')['new_cluster'].cumsum()

        # Create unique cluster identifier
        p3_auctions['full_cluster_id'] = p3_auctions['USER_ID'].astype(str) + '_' + p3_auctions['cluster_id'].astype(str)

        # Count auctions per cluster
        cluster_sizes = p3_auctions.groupby('full_cluster_id').size()

        log("\n--- Auction Cluster Size Distribution ---", f)
        log(f"{'Cluster Size':<15} {'Count':>12} {'%':>10} {'Interpretation':<30}", f)
        log("-" * 70, f)

        cluster_interpretations = {
            1: 'Single slot',
            2: '2 ad blocks',
            3: '3 ad blocks',
            4: '4 ad blocks',
            6: 'Full page (6 blocks)',
        }

        cluster_size_dist = cluster_sizes.value_counts().sort_index()
        total_clusters = len(cluster_sizes)

        for size in sorted(cluster_size_dist.index):
            cnt = cluster_size_dist[size]
            pct = 100 * cnt / total_clusters if total_clusters > 0 else 0
            interp = cluster_interpretations.get(size, 'Complex/unusual')
            if size <= 10:
                log(f"{size:<15} {cnt:>12,} {pct:>9.1f}% {interp:<30}", f)

        # Aggregate stats
        log(f"\nCluster size statistics:", f)
        log(f"  Mean cluster size: {cluster_sizes.mean():.2f}", f)
        log(f"  Median cluster size: {cluster_sizes.median():.0f}", f)
        log(f"  Max cluster size: {cluster_sizes.max()}", f)
        log(f"  Total clusters: {total_clusters:,}", f)

        # Rank distribution within clusters (for multi-auction clusters)
        log("\n--- Rank Distribution Within Clusters (size >= 2) ---", f)
        multi_auction_clusters = p3_auctions[p3_auctions['full_cluster_id'].isin(
            cluster_sizes[cluster_sizes >= 2].index
        )].copy()

        if len(multi_auction_clusters) > 0:
            # Get ranks for items in these clusters
            multi_cluster_items = p3_data[p3_data['auction_id'].isin(multi_auction_clusters['auction_id'])]

            # Per-cluster rank patterns
            cluster_rank_stats = multi_cluster_items.merge(
                multi_auction_clusters[['auction_id', 'full_cluster_id']],
                on='auction_id'
            ).groupby('full_cluster_id')['rank'].agg(['min', 'max', 'nunique'])

            log(f"{'Metric':<35} {'Value':>15}", f)
            log("-" * 52, f)
            log(f"{'Mean min rank in cluster':<35} {cluster_rank_stats['min'].mean():>15.2f}", f)
            log(f"{'Mean max rank in cluster':<35} {cluster_rank_stats['max'].mean():>15.2f}", f)
            log(f"{'Mean unique ranks in cluster':<35} {cluster_rank_stats['nunique'].mean():>15.2f}", f)

            # Are ranks consecutive?
            cluster_rank_stats['range'] = cluster_rank_stats['max'] - cluster_rank_stats['min'] + 1
            cluster_rank_stats['is_contiguous'] = cluster_rank_stats['range'] == cluster_rank_stats['nunique']
            log(f"{'% clusters with contiguous ranks':<35} {100 * cluster_rank_stats['is_contiguous'].mean():>14.1f}%", f)
        else:
            log("No multi-auction clusters found.", f)

        # ---------------------------------------------------------------------
        # 20.2: RANK OFFSET ANALYSIS (PAGINATION DETECTION)
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("20.2: RANK OFFSET ANALYSIS (PAGINATION DETECTION)", f)
        log("-" * 80, f)
        log("\nExamine minimum rank per session to detect pagination patterns.", f)
        log("If paginated: min_rank = 1, 13, 25, ... (jumps of 12)", f)
        log("If scroll-based: continuous distribution", f)

        # Min rank per P3 session
        p3_min_rank = p3_data.groupby('auction_id')['rank'].min()

        log("\n--- Min Rank Distribution in P3 Sessions ---", f)
        log(f"{'Min Rank':<15} {'Count':>12} {'%':>10} {'Interpretation':<25}", f)
        log("-" * 65, f)

        rank_interpretations = {
            1: 'First page',
            13: 'Page 2 (if 12/page)',
            25: 'Page 3',
            37: 'Page 4',
        }

        min_rank_dist = p3_min_rank.value_counts().sort_index()
        total_sessions = len(p3_min_rank)

        # Show top 20 most common min ranks
        top_min_ranks = min_rank_dist.head(20)
        for rank in top_min_ranks.index:
            cnt = min_rank_dist[rank]
            pct = 100 * cnt / total_sessions if total_sessions > 0 else 0
            interp = rank_interpretations.get(rank, 'Scroll position')
            log(f"{rank:<15} {cnt:>12,} {pct:>9.1f}% {interp:<25}", f)

        # Check for pagination pattern (ranks 1, 13, 25, 37, ...)
        pagination_ranks = [1, 13, 25, 37, 49, 61]
        pagination_count = sum(min_rank_dist.get(r, 0) for r in pagination_ranks)
        pagination_pct = 100 * pagination_count / total_sessions if total_sessions > 0 else 0

        log(f"\nPagination pattern analysis:", f)
        log(f"  Sessions starting at pagination ranks (1,13,25,37,...): {pagination_count:,} ({pagination_pct:.1f}%)", f)
        log(f"  Sessions starting at rank 1: {min_rank_dist.get(1, 0):,} ({100 * min_rank_dist.get(1, 0) / total_sessions:.1f}%)", f)

        verdict = "Mostly first page" if min_rank_dist.get(1, 0) / total_sessions > 0.9 else \
                  "Paginated" if pagination_pct > 80 else "Scroll-based"
        log(f"  Verdict: {verdict}", f)

        # ---------------------------------------------------------------------
        # 20.3: TIMESTAMP MICROSECOND ANALYSIS
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("20.3: TIMESTAMP MICROSECOND ANALYSIS", f)
        log("-" * 80, f)
        log("\nAnalyze P3 inter-auction gaps at millisecond precision.", f)

        # Use gaps already computed
        p3_gaps = p3_auctions[p3_auctions['gap_ms'].notna()]['gap_ms']

        log("\n--- P3 Rapid-Fire Timing Precision ---", f)
        log(f"{'Gap Range (ms)':<20} {'Count':>12} {'%':>10} {'Meaning':<30}", f)
        log("-" * 75, f)

        gap_buckets = [
            (0, 1, 'Truly simultaneous'),
            (1, 50, 'Sequential API calls'),
            (50, 100, 'Lazy loading'),
            (100, 500, 'User scroll'),
            (500, 1000, 'User pause'),
            (1000, 5000, 'Short break'),
            (5000, float('inf'), 'New page/action'),
        ]

        for low, high, meaning in gap_buckets:
            if high == float('inf'):
                mask = p3_gaps >= low
                label = f"{low}ms+"
            else:
                mask = (p3_gaps >= low) & (p3_gaps < high)
                label = f"{low}-{int(high)}ms"

            cnt = mask.sum()
            pct = 100 * cnt / len(p3_gaps) if len(p3_gaps) > 0 else 0
            log(f"{label:<20} {cnt:>12,} {pct:>9.1f}% {meaning:<30}", f)

        # Summary statistics
        log(f"\nP3 gap statistics (ms):", f)
        log(f"  Mean gap: {p3_gaps.mean():.2f}ms", f)
        log(f"  Median gap: {p3_gaps.median():.2f}ms", f)
        log(f"  P5: {p3_gaps.quantile(0.05):.2f}ms", f)
        log(f"  P95: {p3_gaps.quantile(0.95):.2f}ms", f)

        # Compare to P1
        p1_auctions = p1_data[['auction_id', 'USER_ID', 'CREATED_AT']].drop_duplicates()
        p1_auctions = p1_auctions.sort_values(['USER_ID', 'CREATED_AT'])
        p1_auctions['prev_time'] = p1_auctions.groupby('USER_ID')['CREATED_AT'].shift(1)
        p1_auctions['gap_ms'] = (p1_auctions['CREATED_AT'] - p1_auctions['prev_time']).dt.total_seconds() * 1000
        p1_gaps = p1_auctions[p1_auctions['gap_ms'].notna()]['gap_ms']

        log(f"\nComparison: P1 vs P3 gap statistics (ms):", f)
        log(f"{'Metric':<25} {'P1':>15} {'P3':>15}", f)
        log("-" * 57, f)
        log(f"{'Mean gap':<25} {p1_gaps.mean():>15.2f} {p3_gaps.mean():>15.2f}", f)
        log(f"{'Median gap':<25} {p1_gaps.median():>15.2f} {p3_gaps.median():>15.2f}", f)
        log(f"{'% < 100ms':<25} {100 * (p1_gaps < 100).mean():>14.1f}% {100 * (p3_gaps < 100).mean():>14.1f}%", f)
        log(f"{'% < 1s':<25} {100 * (p1_gaps < 1000).mean():>14.1f}% {100 * (p3_gaps < 1000).mean():>14.1f}%", f)

        # =====================================================================
        # SECTION 21: CROSS-PLACEMENT JOURNEY ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 21: CROSS-PLACEMENT JOURNEY ANALYSIS", f)
        log("=" * 80, f)
        log("\nAnalyzing how users move between placements and product overlap.", f)

        # ---------------------------------------------------------------------
        # 21.1: CLICK -> NEXT PLACEMENT TIMING
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("21.1: CLICK -> NEXT PLACEMENT TIMING", f)
        log("-" * 80, f)
        log("\nMeasure time from a click to the next placement event.", f)
        log("If clicking -> PDP -> P2: expect short gap after click to P2 appearance.", f)

        # Build combined dataset of all clicked items with their timestamps
        all_clicks = []
        for name, df in placements.items():
            clicks = df[df['clicked'] == 1][['auction_id', 'USER_ID', 'CREATED_AT', 'product_id']].copy()
            clicks['placement'] = name
            clicks['event_type'] = 'click'
            all_clicks.append(clicks)
        all_clicks_df = pd.concat(all_clicks, ignore_index=True)

        # Get all auction first events (first item shown per auction as proxy for auction start)
        all_auction_events = []
        for name, df in placements.items():
            first_items = df.groupby('auction_id').agg({
                'USER_ID': 'first',
                'CREATED_AT': 'min',
                'product_id': 'first'
            }).reset_index()
            first_items['placement'] = name
            first_items['event_type'] = 'auction'
            all_auction_events.append(first_items)
        all_auction_df = pd.concat(all_auction_events, ignore_index=True)

        # Combine clicks and auctions, sort by time
        combined_events = pd.concat([
            all_clicks_df[['USER_ID', 'CREATED_AT', 'placement', 'event_type', 'product_id']],
            all_auction_df[['USER_ID', 'CREATED_AT', 'placement', 'event_type', 'product_id']]
        ], ignore_index=True)
        combined_events = combined_events.sort_values(['USER_ID', 'CREATED_AT'])

        # For each click, find the next auction event
        combined_events['next_event_type'] = combined_events.groupby('USER_ID')['event_type'].shift(-1)
        combined_events['next_placement'] = combined_events.groupby('USER_ID')['placement'].shift(-1)
        combined_events['next_time'] = combined_events.groupby('USER_ID')['CREATED_AT'].shift(-1)
        combined_events['gap_to_next'] = (combined_events['next_time'] - combined_events['CREATED_AT']).dt.total_seconds()

        # Filter to clicks followed by auctions
        click_to_auction = combined_events[
            (combined_events['event_type'] == 'click') &
            (combined_events['next_event_type'] == 'auction')
        ].copy()

        log("\n--- Time from Click to Next Auction (median, seconds) ---", f)
        log(f"{'Click Placement':<15} {'Next Placement':<15} {'Median Gap':>12} {'Count':>12}", f)
        log("-" * 57, f)

        click_transitions = click_to_auction.groupby(['placement', 'next_placement'])['gap_to_next'].agg(['median', 'count'])
        click_transitions = click_transitions.reset_index()

        for _, row in click_transitions.sort_values(['placement', 'next_placement']).iterrows():
            if row['count'] >= 10:  # Only show if meaningful sample
                log(f"{row['placement']:<15} {row['next_placement']:<15} {row['median']:>12.2f} {int(row['count']):>12,}", f)

        # Specifically check P1/P3 click -> P2 pattern
        log("\n--- Key Journey Patterns ---", f)

        for src in ['P1', 'P3']:
            for dst in ['P2']:
                subset = click_to_auction[(click_to_auction['placement'] == src) & (click_to_auction['next_placement'] == dst)]
                if len(subset) > 0:
                    log(f"\n{src} click -> {dst} auction:", f)
                    log(f"  Count: {len(subset):,}", f)
                    log(f"  Median gap: {subset['gap_to_next'].median():.2f}s", f)
                    log(f"  Mean gap: {subset['gap_to_next'].mean():.2f}s", f)
                    log(f"  % under 5s: {100 * (subset['gap_to_next'] < 5).mean():.1f}%", f)
                    log(f"  % under 30s: {100 * (subset['gap_to_next'] < 30).mean():.1f}%", f)

        # ---------------------------------------------------------------------
        # 21.2: PRODUCT OVERLAP ANALYSIS
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("21.2: PRODUCT OVERLAP ANALYSIS", f)
        log("-" * 80, f)
        log("\nWithin same user, check product/brand overlap across placements.", f)

        # Get unique products per user per placement
        user_products = {}
        for name, df in placements.items():
            user_products[name] = df.groupby('USER_ID')['product_id'].apply(set)

        # Find users who appear in multiple placements
        users_p1_p2 = set(user_products['P1'].index) & set(user_products['P2'].index)
        users_p3_p2 = set(user_products['P3'].index) & set(user_products['P2'].index)
        users_p1_p3 = set(user_products['P1'].index) & set(user_products['P3'].index)

        log("\n--- Product Overlap Between Placements (Same User) ---", f)
        log(f"{'Comparison':<30} {'Users':>12} {'With Overlap':>15} {'% Overlap':>12}", f)
        log("-" * 72, f)

        # P1 -> P2 overlap
        p1_p2_overlap_count = 0
        for user in tqdm(list(users_p1_p2)[:10000], desc="P1-P2 overlap"):  # Sample for speed
            p1_prods = user_products['P1'].get(user, set())
            p2_prods = user_products['P2'].get(user, set())
            if len(p1_prods & p2_prods) > 0:
                p1_p2_overlap_count += 1
        p1_p2_pct = 100 * p1_p2_overlap_count / min(len(users_p1_p2), 10000) if len(users_p1_p2) > 0 else 0
        log(f"{'P1 products also in P2':<30} {len(users_p1_p2):>12,} {p1_p2_overlap_count:>15,} {p1_p2_pct:>11.1f}%", f)

        # P3 -> P2 overlap
        p3_p2_overlap_count = 0
        for user in tqdm(list(users_p3_p2)[:10000], desc="P3-P2 overlap"):
            p3_prods = user_products['P3'].get(user, set())
            p2_prods = user_products['P2'].get(user, set())
            if len(p3_prods & p2_prods) > 0:
                p3_p2_overlap_count += 1
        p3_p2_pct = 100 * p3_p2_overlap_count / min(len(users_p3_p2), 10000) if len(users_p3_p2) > 0 else 0
        log(f"{'P3 products also in P2':<30} {len(users_p3_p2):>12,} {p3_p2_overlap_count:>15,} {p3_p2_pct:>11.1f}%", f)

        # P1 -> P3 overlap
        p1_p3_overlap_count = 0
        for user in tqdm(list(users_p1_p3)[:10000], desc="P1-P3 overlap"):
            p1_prods = user_products['P1'].get(user, set())
            p3_prods = user_products['P3'].get(user, set())
            if len(p1_prods & p3_prods) > 0:
                p1_p3_overlap_count += 1
        p1_p3_pct = 100 * p1_p3_overlap_count / min(len(users_p1_p3), 10000) if len(users_p1_p3) > 0 else 0
        log(f"{'P1 products also in P3':<30} {len(users_p1_p3):>12,} {p1_p3_overlap_count:>15,} {p1_p3_pct:>11.1f}%", f)

        # Brand overlap for clicked items
        log("\n--- Brand Overlap (Clicked Items) ---", f)
        log("Does clicking on a brand in P1/P3 lead to seeing that brand in P2?", f)

        # Get clicked brands per user per placement
        user_clicked_brands = {}
        for name, df in placements.items():
            clicked = df[df['clicked'] == 1]
            if 'primary_brand' in clicked.columns:
                user_clicked_brands[name] = clicked.groupby('USER_ID')['primary_brand'].apply(lambda x: set(x.dropna()))
            else:
                user_clicked_brands[name] = pd.Series(dtype=object)

        # Get shown brands per user in P2
        if 'primary_brand' in p2_data.columns:
            user_p2_brands = p2_data.groupby('USER_ID')['primary_brand'].apply(lambda x: set(x.dropna()))

            # Check if clicked brand in P1 appears in P2
            users_clicked_p1 = set(user_clicked_brands['P1'].index) & set(user_p2_brands.index)
            brand_match_p1_p2 = 0
            for user in tqdm(list(users_clicked_p1)[:5000], desc="P1 click brand -> P2"):
                clicked_brands = user_clicked_brands['P1'].get(user, set())
                p2_brands = user_p2_brands.get(user, set())
                if len(clicked_brands & p2_brands) > 0:
                    brand_match_p1_p2 += 1
            p1_p2_brand_pct = 100 * brand_match_p1_p2 / min(len(users_clicked_p1), 5000) if len(users_clicked_p1) > 0 else 0
            log(f"P1 clicked brand -> appears in P2: {brand_match_p1_p2:,} / {min(len(users_clicked_p1), 5000):,} ({p1_p2_brand_pct:.1f}%)", f)

            # Check if clicked brand in P3 appears in P2
            users_clicked_p3 = set(user_clicked_brands['P3'].index) & set(user_p2_brands.index)
            brand_match_p3_p2 = 0
            for user in tqdm(list(users_clicked_p3)[:5000], desc="P3 click brand -> P2"):
                clicked_brands = user_clicked_brands['P3'].get(user, set())
                p2_brands = user_p2_brands.get(user, set())
                if len(clicked_brands & p2_brands) > 0:
                    brand_match_p3_p2 += 1
            p3_p2_brand_pct = 100 * brand_match_p3_p2 / min(len(users_clicked_p3), 5000) if len(users_clicked_p3) > 0 else 0
            log(f"P3 clicked brand -> appears in P2: {brand_match_p3_p2:,} / {min(len(users_clicked_p3), 5000):,} ({p3_p2_brand_pct:.1f}%)", f)
        else:
            log("Brand data not available for overlap analysis.", f)

        # =====================================================================
        # SECTION 22: POSITION/RANK STRUCTURE ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 22: POSITION/RANK STRUCTURE ANALYSIS", f)
        log("=" * 80, f)
        log("\nAnalyzing position patterns to understand grid layouts and P2 sub-types.", f)

        # ---------------------------------------------------------------------
        # 22.1: POSITION DISTRIBUTION BY SESSION LENGTH
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("22.1: POSITION DISTRIBUTION BY SESSION LENGTH", f)
        log("-" * 80, f)
        log("\nExamine position patterns for sessions of specific lengths.", f)

        for placement_name in ['P1', 'P2', 'P3']:
            df = placements[placement_name]
            sess_len = session_lengths[placement_name]

            log(f"\n=== {placement_name} Position Patterns ===", f)

            # Sessions with exactly 2 items
            sessions_2 = sess_len[sess_len == 2].index
            if len(sessions_2) > 0:
                log(f"\n--- {placement_name} Sessions with Exactly 2 Items (n={len(sessions_2):,}) ---", f)
                pos_pairs = df[df['auction_id'].isin(sessions_2)].groupby('auction_id')['position'].apply(tuple)
                pair_counts = pos_pairs.value_counts().head(10)
                log(f"{'Position Pair':<20} {'Count':>12} {'%':>10} {'Pattern':<20}", f)
                log("-" * 65, f)
                for pair, cnt in pair_counts.items():
                    pct = 100 * cnt / len(sessions_2)
                    if pair == (1, 2):
                        pattern = "Contiguous row"
                    elif pair[1] - pair[0] == 2:
                        pattern = "Grid skip (stride=2)"
                    elif pair[1] - pair[0] == 1:
                        pattern = "Contiguous"
                    else:
                        pattern = f"Gap of {pair[1] - pair[0]}"
                    log(f"{str(pair):<20} {cnt:>12,} {pct:>9.1f}% {pattern:<20}", f)

            # Sessions with exactly 4 items
            sessions_4 = sess_len[sess_len == 4].index
            if len(sessions_4) > 0:
                log(f"\n--- {placement_name} Sessions with Exactly 4 Items (n={len(sessions_4):,}) ---", f)
                pos_quads = df[df['auction_id'].isin(sessions_4)].groupby('auction_id')['position'].apply(lambda x: tuple(sorted(x)))
                quad_counts = pos_quads.value_counts().head(10)
                log(f"{'Positions':<25} {'Count':>12} {'%':>10} {'Pattern':<20}", f)
                log("-" * 70, f)
                for positions, cnt in quad_counts.items():
                    pct = 100 * cnt / len(sessions_4)
                    if positions == (1, 2, 3, 4):
                        pattern = "Contiguous 1-4"
                    elif positions == (1, 2, 5, 6):
                        pattern = "Two rows (skip 3-4)"
                    elif max(positions) - min(positions) + 1 == 4:
                        pattern = "Contiguous block"
                    else:
                        pattern = "Scattered"
                    log(f"{str(positions):<25} {cnt:>12,} {pct:>9.1f}% {pattern:<20}", f)

            # Sessions with exactly 6 items
            sessions_6 = sess_len[sess_len == 6].index
            if len(sessions_6) > 0:
                log(f"\n--- {placement_name} Sessions with Exactly 6 Items (n={len(sessions_6):,}) ---", f)
                pos_six = df[df['auction_id'].isin(sessions_6)].groupby('auction_id')['position'].apply(lambda x: tuple(sorted(x)))
                six_counts = pos_six.value_counts().head(5)
                log(f"{'Positions':<30} {'Count':>12} {'%':>10}", f)
                log("-" * 55, f)
                for positions, cnt in six_counts.items():
                    pct = 100 * cnt / len(sessions_6)
                    log(f"{str(positions):<30} {cnt:>12,} {pct:>9.1f}%", f)

        # ---------------------------------------------------------------------
        # 22.2: P2 SUB-TYPE DETECTION
        # ---------------------------------------------------------------------
        log("\n" + "-" * 80, f)
        log("22.2: P2 SUB-TYPE DETECTION", f)
        log("-" * 80, f)
        log("\nCluster P2 sessions to detect distinct layouts.", f)

        # Use placements['P2'] which has catalog data merged
        p2_enriched = placements['P2']

        p2_session_stats = p2_enriched.groupby('auction_id').agg({
            'position': ['count', 'min', 'max'],
            'rank': ['min', 'max'],
            'clicked': 'sum',
            'primary_brand': 'nunique'
        })
        p2_session_stats.columns = ['_'.join(col).strip() for col in p2_session_stats.columns.values]
        p2_session_stats = p2_session_stats.rename(columns={
            'position_count': 'session_length',
            'position_min': 'min_position',
            'position_max': 'max_position',
            'rank_min': 'min_rank',
            'rank_max': 'max_rank',
            'clicked_sum': 'clicks',
            'primary_brand_nunique': 'unique_brands'
        })

        # Classify by session length
        p2_session_stats['length_bucket'] = pd.cut(
            p2_session_stats['session_length'],
            bins=[0, 2, 4, 6, 8, 12, float('inf')],
            labels=['1-2', '3-4', '5-6', '7-8', '9-12', '13+']
        )

        log("\n--- P2 Session Length Distribution ---", f)
        log(f"{'Length Bucket':<15} {'N Sessions':>12} {'%':>10} {'Mean CTR':>12} {'Mean Brands':>12}", f)
        log("-" * 65, f)

        for bucket in ['1-2', '3-4', '5-6', '7-8', '9-12', '13+']:
            bucket_data = p2_session_stats[p2_session_stats['length_bucket'] == bucket]
            n = len(bucket_data)
            if n > 0:
                pct = 100 * n / len(p2_session_stats)
                # Calculate CTR for this bucket
                bucket_clicks = bucket_data['clicks'].sum()
                bucket_items = bucket_data['session_length'].sum()
                ctr = 100 * bucket_clicks / bucket_items if bucket_items > 0 else 0
                mean_brands = bucket_data['unique_brands'].mean()
                log(f"{bucket:<15} {n:>12,} {pct:>9.1f}% {ctr:>11.2f}% {mean_brands:>12.2f}", f)

        # CTR by position within P2 length buckets
        log("\n--- CTR by Position: Short (1-4 items) vs Long (8+ items) P2 Sessions ---", f)
        short_sessions = p2_session_stats[p2_session_stats['session_length'] <= 4].index
        long_sessions = p2_session_stats[p2_session_stats['session_length'] >= 8].index

        log(f"{'Position':<10} {'Short CTR':>15} {'Short N':>10} {'Long CTR':>15} {'Long N':>10}", f)
        log("-" * 65, f)

        for pos in range(1, 13):
            short_data = p2_enriched[(p2_enriched['auction_id'].isin(short_sessions)) & (p2_enriched['position'] == pos)]
            long_data = p2_enriched[(p2_enriched['auction_id'].isin(long_sessions)) & (p2_enriched['position'] == pos)]

            short_ctr = 100 * short_data['clicked'].mean() if len(short_data) > 0 else np.nan
            long_ctr = 100 * long_data['clicked'].mean() if len(long_data) > 0 else np.nan

            short_ctr_str = f"{short_ctr:.2f}%" if not np.isnan(short_ctr) else "N/A"
            long_ctr_str = f"{long_ctr:.2f}%" if not np.isnan(long_ctr) else "N/A"

            log(f"{pos:<10} {short_ctr_str:>15} {len(short_data):>10,} {long_ctr_str:>15} {len(long_data):>10,}", f)

        # P2 sub-type interpretation
        log("\n--- P2 Sub-Type Interpretation ---", f)
        short_pct = 100 * len(short_sessions) / len(p2_session_stats) if len(p2_session_stats) > 0 else 0
        long_pct = 100 * len(long_sessions) / len(p2_session_stats) if len(p2_session_stats) > 0 else 0

        short_total_ctr = 100 * p2_enriched[p2_enriched['auction_id'].isin(short_sessions)]['clicked'].mean() if len(short_sessions) > 0 else np.nan
        long_total_ctr = 100 * p2_enriched[p2_enriched['auction_id'].isin(long_sessions)]['clicked'].mean() if len(long_sessions) > 0 else np.nan

        log(f"\nShort P2 sessions (1-4 items): {len(short_sessions):,} ({short_pct:.1f}%)", f)
        log(f"  Possible interpretation: 'Featured Items' or 'Similar Products' widget", f)
        log(f"  Total CTR: {short_total_ctr:.2f}%", f)

        log(f"\nLong P2 sessions (8+ items): {len(long_sessions):,} ({long_pct:.1f}%)", f)
        log(f"  Possible interpretation: 'Other Listings' or full product grid", f)
        log(f"  Total CTR: {long_total_ctr:.2f}%", f)

        # Brand concentration difference
        short_single_brand = (p2_session_stats.loc[short_sessions, 'unique_brands'] == 1).mean() if len(short_sessions) > 0 else np.nan
        long_single_brand = (p2_session_stats.loc[long_sessions, 'unique_brands'] == 1).mean() if len(long_sessions) > 0 else np.nan

        log(f"\n--- Brand Concentration ---", f)
        log(f"% single-brand sessions (short): {100 * short_single_brand:.1f}%", f)
        log(f"% single-brand sessions (long): {100 * long_single_brand:.1f}%", f)

        # =====================================================================
        # END OF REPORT
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("END OF REPORT", f)
        log(f"Output saved to: {OUTPUT_FILE}", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
