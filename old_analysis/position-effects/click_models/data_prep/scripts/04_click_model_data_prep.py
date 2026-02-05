#!/usr/bin/env python3
"""
Click Model Data Preparation

Transforms raw auction data into session-based format suitable for click models.
Each session = one auction. For each auction, we have:
- auction_id: unique identifier
- placement: UI context
- items: list of products shown (in display order by RANKING)
- ranks: list of rankings (1-indexed)
- clicks: binary vector (1 if product clicked, 0 otherwise)

This format supports PBM, DBN, and SDBN estimation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_r2"
OUTPUT_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "04_click_model_data_prep.txt"

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
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("CLICK MODEL DATA PREPARATION", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Load Data
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("-" * 40, f)
        log("", f)

        ar = pd.read_parquet(DATA_DIR / "auctions_results_r2.parquet")
        au = pd.read_parquet(DATA_DIR / "auctions_users_r2.parquet")
        imp = pd.read_parquet(DATA_DIR / "impressions_r2.parquet")
        clicks = pd.read_parquet(DATA_DIR / "clicks_r2.parquet")

        log(f"Loaded auctions_results: {len(ar):,} rows", f)
        log(f"Loaded auctions_users: {len(au):,} rows", f)
        log(f"Loaded impressions: {len(imp):,} rows", f)
        log(f"Loaded clicks: {len(clicks):,} rows", f)
        log("", f)

        # Basic stats
        log(f"Unique auctions in AR: {ar['AUCTION_ID'].nunique():,}", f)
        log(f"Unique auctions in AU: {au['AUCTION_ID'].nunique():,}", f)
        log(f"Unique auctions in IMP: {imp['AUCTION_ID'].nunique():,}", f)
        log(f"Unique auctions in CLICKS: {clicks['AUCTION_ID'].nunique():,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Filter to Winners with Impressions
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: FILTER TO IMPRESSED ITEMS", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Build sessions from impressions (products actually shown)", f)
        log("", f)

        # Create impression set for quick lookup
        imp_keys = set(zip(imp['AUCTION_ID'], imp['PRODUCT_ID']))
        log(f"Unique (auction, product) pairs with impression: {len(imp_keys):,}", f)

        # Create click set for quick lookup
        click_keys = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
        log(f"Unique (auction, product) pairs with click: {len(click_keys):,}", f)
        log("", f)

        # Get winners only (IS_WINNER = True)
        winners = ar[ar['IS_WINNER'] == True].copy()
        log(f"Winners (IS_WINNER=True): {len(winners):,}", f)

        # Check which winners have impressions
        winners['has_impression'] = winners.apply(
            lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in imp_keys, axis=1
        )
        winners_with_imp = winners[winners['has_impression']].copy()
        log(f"Winners with impressions: {len(winners_with_imp):,}", f)
        log(f"Impression rate for winners: {len(winners_with_imp)/len(winners)*100:.2f}%", f)
        log("", f)

        # Add click indicator
        winners_with_imp['clicked'] = winners_with_imp.apply(
            lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_keys, axis=1
        )

        total_clicks_matched = winners_with_imp['clicked'].sum()
        log(f"Clicks matched to winners with impressions: {total_clicks_matched:,}", f)
        log(f"Total clicks in data: {len(clicks):,}", f)
        log(f"Match rate: {total_clicks_matched/len(clicks)*100:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Add Placement Information
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: ADD PLACEMENT", f)
        log("-" * 40, f)
        log("", f)

        # Merge with auctions_users to get placement
        au_lookup = au[['AUCTION_ID', 'PLACEMENT']].drop_duplicates('AUCTION_ID')
        winners_with_imp = winners_with_imp.merge(au_lookup, on='AUCTION_ID', how='left')

        missing_placement = winners_with_imp['PLACEMENT'].isna().sum()
        log(f"Rows missing placement: {missing_placement:,} ({missing_placement/len(winners_with_imp)*100:.2f}%)", f)

        # Fill missing with 'unknown'
        winners_with_imp['PLACEMENT'] = winners_with_imp['PLACEMENT'].fillna('unknown')

        # Placement distribution
        log("", f)
        log("Placement distribution:", f)
        placement_dist = winners_with_imp.groupby('PLACEMENT').agg({
            'AUCTION_ID': 'count',
            'clicked': 'sum'
        }).rename(columns={'AUCTION_ID': 'impressions', 'clicked': 'clicks'})
        placement_dist['CTR'] = placement_dist['clicks'] / placement_dist['impressions'] * 100

        for placement, row in placement_dist.iterrows():
            log(f"  Placement {placement}: {row['impressions']:,} impressions, {row['clicks']:,} clicks, CTR={row['CTR']:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Build Sessions
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: BUILD SESSIONS", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: One session per auction, with items sorted by RANKING", f)
        log("", f)

        # Sort by auction and ranking
        winners_with_imp = winners_with_imp.sort_values(['AUCTION_ID', 'RANKING'])

        # Group by auction
        sessions = []
        auction_groups = winners_with_imp.groupby('AUCTION_ID')

        for auction_id, group in tqdm(auction_groups, desc="Building sessions"):
            group = group.sort_values('RANKING')

            session = {
                'auction_id': auction_id,
                'placement': group['PLACEMENT'].iloc[0],
                'n_items': len(group),
                'items': group['PRODUCT_ID'].tolist(),
                'ranks': group['RANKING'].tolist(),
                'clicks': group['clicked'].astype(int).tolist(),
                'qualities': group['QUALITY'].tolist(),
                'bids': group['FINAL_BID'].tolist(),
                'n_clicks': group['clicked'].sum()
            }
            sessions.append(session)

        sessions_df = pd.DataFrame(sessions)

        log(f"Total sessions: {len(sessions_df):,}", f)
        log(f"Total items across sessions: {sessions_df['n_items'].sum():,}", f)
        log(f"Total clicks: {sessions_df['n_clicks'].sum():,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Session Statistics
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: SESSION STATISTICS", f)
        log("-" * 40, f)
        log("", f)

        # Items per session
        log("Items per session distribution:", f)
        log(f"  Min: {sessions_df['n_items'].min()}", f)
        log(f"  Max: {sessions_df['n_items'].max()}", f)
        log(f"  Mean: {sessions_df['n_items'].mean():.2f}", f)
        log(f"  Median: {sessions_df['n_items'].median():.2f}", f)
        log("", f)

        # Items per session histogram
        log("Items per session histogram:", f)
        item_bins = [1, 2, 3, 4, 5, 6, 10, 20, 50, 100]
        prev_bin = 0
        for bin_edge in item_bins:
            count = ((sessions_df['n_items'] > prev_bin) & (sessions_df['n_items'] <= bin_edge)).sum()
            pct = count / len(sessions_df) * 100
            log(f"  {prev_bin+1}-{bin_edge} items: {count:,} ({pct:.1f}%)", f)
            prev_bin = bin_edge
        remaining = (sessions_df['n_items'] > item_bins[-1]).sum()
        log(f"  {item_bins[-1]+1}+ items: {remaining:,} ({remaining/len(sessions_df)*100:.1f}%)", f)
        log("", f)

        # Clicks per session
        log("Clicks per session distribution:", f)
        clicks_dist = sessions_df['n_clicks'].value_counts().sort_index()
        total_sessions = len(sessions_df)
        for n_clicks, count in clicks_dist.head(10).items():
            pct = count / total_sessions * 100
            log(f"  {n_clicks} clicks: {count:,} ({pct:.1f}%)", f)

        # Multi-click rate
        multi_click = (sessions_df['n_clicks'] >= 2).sum()
        multi_click_rate = multi_click / total_sessions * 100
        log("", f)
        log(f"Multi-click sessions (2+ clicks): {multi_click:,} ({multi_click_rate:.1f}%)", f)
        log("", f)

        # Sessions by placement
        log("Sessions by placement:", f)
        for placement, count in sessions_df['placement'].value_counts().items():
            pct = count / len(sessions_df) * 100
            log(f"  Placement {placement}: {count:,} ({pct:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Rank Distribution in Sessions
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: RANK DISTRIBUTION", f)
        log("-" * 40, f)
        log("", f)

        # Flatten to item level
        item_level = []
        for _, session in sessions_df.iterrows():
            for i, (rank, clicked) in enumerate(zip(session['ranks'], session['clicks'])):
                item_level.append({
                    'auction_id': session['auction_id'],
                    'placement': session['placement'],
                    'position': i + 1,  # 1-indexed position within session
                    'rank': rank,
                    'clicked': clicked
                })

        item_df = pd.DataFrame(item_level)

        # CTR by position (within session)
        log("CTR by position within session (display order):", f)
        log(f"  {'Position':<10} {'N':<12} {'Clicks':<10} {'CTR %':<10}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10}", f)

        ctr_by_pos = item_df.groupby('position').agg({
            'clicked': ['count', 'sum', 'mean']
        }).reset_index()
        ctr_by_pos.columns = ['position', 'n', 'clicks', 'ctr']

        for _, row in ctr_by_pos.head(20).iterrows():
            log(f"  {int(row['position']):<10} {int(row['n']):<12,} {int(row['clicks']):<10,} {row['ctr']*100:<10.2f}", f)
        log("", f)

        # CTR by RANKING (auction ranking)
        log("CTR by RANKING (auction rank):", f)
        log(f"  {'Rank':<10} {'N':<12} {'Clicks':<10} {'CTR %':<10}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10}", f)

        ctr_by_rank = item_df.groupby('rank').agg({
            'clicked': ['count', 'sum', 'mean']
        }).reset_index()
        ctr_by_rank.columns = ['rank', 'n', 'clicks', 'ctr']

        for _, row in ctr_by_rank.head(20).iterrows():
            log(f"  {int(row['rank']):<10} {int(row['n']):<12,} {int(row['clicks']):<10,} {row['ctr']*100:<10.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Click Position Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: CLICK POSITION ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Understand click patterns for click model assumptions", f)
        log("", f)

        # First click position
        sessions_with_clicks = sessions_df[sessions_df['n_clicks'] > 0].copy()

        def get_first_click_pos(clicks_list):
            for i, c in enumerate(clicks_list):
                if c == 1:
                    return i + 1
            return None

        sessions_with_clicks['first_click_pos'] = sessions_with_clicks['clicks'].apply(get_first_click_pos)

        log("First click position distribution:", f)
        first_click_dist = sessions_with_clicks['first_click_pos'].value_counts().sort_index()
        for pos, count in first_click_dist.head(10).items():
            pct = count / len(sessions_with_clicks) * 100
            log(f"  Position {pos}: {count:,} ({pct:.1f}%)", f)
        log("", f)

        # Last click position (for multi-click sessions)
        multi_click_sessions = sessions_df[sessions_df['n_clicks'] >= 2].copy()

        def get_last_click_pos(clicks_list):
            for i in range(len(clicks_list) - 1, -1, -1):
                if clicks_list[i] == 1:
                    return i + 1
            return None

        if len(multi_click_sessions) > 0:
            multi_click_sessions['last_click_pos'] = multi_click_sessions['clicks'].apply(get_last_click_pos)
            multi_click_sessions['first_click_pos'] = multi_click_sessions['clicks'].apply(get_first_click_pos)
            multi_click_sessions['click_span'] = multi_click_sessions['last_click_pos'] - multi_click_sessions['first_click_pos']

            log("Multi-click sessions analysis:", f)
            log(f"  N multi-click sessions: {len(multi_click_sessions):,}", f)
            log(f"  Mean first click position: {multi_click_sessions['first_click_pos'].mean():.2f}", f)
            log(f"  Mean last click position: {multi_click_sessions['last_click_pos'].mean():.2f}", f)
            log(f"  Mean click span: {multi_click_sessions['click_span'].mean():.2f}", f)
            log("", f)

            # Clicks jumping vs sequential
            def analyze_click_pattern(clicks_list):
                click_positions = [i for i, c in enumerate(clicks_list) if c == 1]
                if len(click_positions) < 2:
                    return 'single'
                gaps = [click_positions[i+1] - click_positions[i] for i in range(len(click_positions)-1)]
                if all(g == 1 for g in gaps):
                    return 'sequential'
                elif any(g < 0 for g in gaps):
                    return 'backwards'
                else:
                    return 'jumping'

            multi_click_sessions['pattern'] = multi_click_sessions['clicks'].apply(analyze_click_pattern)
            pattern_dist = multi_click_sessions['pattern'].value_counts()

            log("Multi-click patterns:", f)
            for pattern, count in pattern_dist.items():
                pct = count / len(multi_click_sessions) * 100
                log(f"  {pattern}: {count:,} ({pct:.1f}%)", f)
            log("", f)

        # -----------------------------------------------------------------
        # Section 8: Product-Level Statistics
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: PRODUCT-LEVEL STATISTICS", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Understand product variation for alpha_i estimation", f)
        log("", f)

        # Count impressions and clicks per product
        product_stats = item_df.groupby(
            item_df.apply(lambda row: sessions_df.loc[sessions_df['auction_id'] == row['auction_id'], 'items'].values[0][row['position']-1], axis=1)
        ).agg({
            'clicked': ['count', 'sum', 'mean']
        })

        # Alternative: flatten properly
        product_level = []
        for _, session in sessions_df.iterrows():
            for i, (item, clicked) in enumerate(zip(session['items'], session['clicks'])):
                product_level.append({
                    'product_id': item,
                    'rank': session['ranks'][i],
                    'clicked': clicked
                })

        product_df = pd.DataFrame(product_level)

        product_stats = product_df.groupby('product_id').agg({
            'clicked': ['count', 'sum', 'mean']
        }).reset_index()
        product_stats.columns = ['product_id', 'impressions', 'clicks', 'ctr']

        log(f"Total unique products: {len(product_stats):,}", f)
        log("", f)

        log("Product impression distribution:", f)
        log(f"  Min: {product_stats['impressions'].min()}", f)
        log(f"  Max: {product_stats['impressions'].max()}", f)
        log(f"  Mean: {product_stats['impressions'].mean():.2f}", f)
        log(f"  Median: {product_stats['impressions'].median():.2f}", f)
        log("", f)

        # Products with enough data for alpha estimation
        products_5plus = (product_stats['impressions'] >= 5).sum()
        products_10plus = (product_stats['impressions'] >= 10).sum()
        products_20plus = (product_stats['impressions'] >= 20).sum()

        log(f"Products with 5+ impressions: {products_5plus:,} ({products_5plus/len(product_stats)*100:.1f}%)", f)
        log(f"Products with 10+ impressions: {products_10plus:,} ({products_10plus/len(product_stats)*100:.1f}%)", f)
        log(f"Products with 20+ impressions: {products_20plus:,} ({products_20plus/len(product_stats)*100:.1f}%)", f)
        log("", f)

        # Position variation per product
        product_rank_variation = product_df.groupby('product_id')['rank'].agg(['nunique', 'min', 'max', 'std']).reset_index()
        product_rank_variation.columns = ['product_id', 'n_positions', 'min_rank', 'max_rank', 'rank_std']
        product_rank_variation['rank_range'] = product_rank_variation['max_rank'] - product_rank_variation['min_rank']

        log("Product position variation:", f)
        log(f"  Products with 2+ positions: {(product_rank_variation['n_positions'] >= 2).sum():,}", f)
        log(f"  Products with 3+ positions: {(product_rank_variation['n_positions'] >= 3).sum():,}", f)
        log(f"  Mean rank range: {product_rank_variation['rank_range'].mean():.2f}", f)
        log(f"  Max rank range: {product_rank_variation['rank_range'].max()}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 9: Save Output
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: SAVE OUTPUT", f)
        log("-" * 40, f)
        log("", f)

        # Save sessions
        output_path = OUTPUT_DIR / "sessions.parquet"
        sessions_df.to_parquet(output_path, index=False)
        log(f"Saved sessions to: {output_path}", f)
        log(f"  Sessions: {len(sessions_df):,}", f)
        log("", f)

        # Save item-level data for easier analysis
        item_output_path = OUTPUT_DIR / "session_items.parquet"

        item_level_full = []
        for _, session in sessions_df.iterrows():
            for i, (item, rank, clicked, quality, bid) in enumerate(zip(
                session['items'], session['ranks'], session['clicks'],
                session['qualities'], session['bids']
            )):
                item_level_full.append({
                    'auction_id': session['auction_id'],
                    'placement': session['placement'],
                    'position': i + 1,
                    'product_id': item,
                    'rank': rank,
                    'clicked': clicked,
                    'quality': quality,
                    'bid': bid,
                    'n_items': session['n_items'],
                    'n_clicks': session['n_clicks']
                })

        item_level_df = pd.DataFrame(item_level_full)
        item_level_df.to_parquet(item_output_path, index=False)
        log(f"Saved item-level data to: {item_output_path}", f)
        log(f"  Items: {len(item_level_df):,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 10: Summary for Click Models
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 10: SUMMARY FOR CLICK MODELS", f)
        log("=" * 80, f)
        log("", f)

        log("DATA SUMMARY:", f)
        log(f"  Total sessions: {len(sessions_df):,}", f)
        log(f"  Total impressions: {len(item_level_df):,}", f)
        log(f"  Total clicks: {sessions_df['n_clicks'].sum():,}", f)
        log(f"  Overall CTR: {sessions_df['n_clicks'].sum() / len(item_level_df) * 100:.3f}%", f)
        log("", f)

        log("CLICK MODEL VIABILITY:", f)
        log("", f)

        # PBM viability
        log("  Position-Based Model (PBM):", f)
        log(f"    - Max position in sessions: {sessions_df['n_items'].max()}", f)
        log(f"    - Mean items per session: {sessions_df['n_items'].mean():.1f}", f)
        log(f"    - Sessions with 3+ items: {(sessions_df['n_items'] >= 3).sum():,}", f)
        log("    - Status: VIABLE", f)
        log("", f)

        # DBN viability
        log("  Dynamic Bayesian Network (DBN):", f)
        log(f"    - Multi-click rate: {multi_click_rate:.1f}%", f)
        log(f"    - Multi-click sessions: {multi_click:,}", f)
        log("    - Note: High multi-click rate suggests DBN may fit better than cascade", f)
        log("    - Status: VIABLE", f)
        log("", f)

        # SDBN viability
        log("  Simplified DBN (SDBN):", f)
        log("    - Assumes no continuation after click (sigma * gamma = 0)", f)
        log("    - Simpler MLE estimation", f)
        log("    - Status: VIABLE (baseline comparison)", f)
        log("", f)

        log("OUTPUT FILES:", f)
        log(f"  - sessions.parquet: {len(sessions_df):,} sessions", f)
        log(f"  - session_items.parquet: {len(item_level_df):,} items", f)
        log("", f)

        log("=" * 80, f)
        log("DATA PREPARATION COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
