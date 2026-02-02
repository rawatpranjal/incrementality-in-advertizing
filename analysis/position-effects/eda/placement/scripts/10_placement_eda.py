#!/usr/bin/env python3
"""
Placement EDA: Comprehensive characterization of placement types from raw user sessions.

Documents all placement characteristics: volume, session structure, user behavior,
rank distributions, product characteristics, and click patterns.

Output: results/10_placement_eda_results.txt
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
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_r2"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "10_placement_eda_results.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# SECTION 1: PLACEMENT OVERVIEW
# =============================================================================
def placement_overview(au, ar, imp, clicks, f):
    """Volume, CTR, and impression delivery by placement."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 1: PLACEMENT OVERVIEW", f)
    log(f"{'='*80}", f)

    # Add placement to auction results
    ar_with_placement = ar.merge(
        au[['AUCTION_ID', 'PLACEMENT']],
        on='AUCTION_ID',
        how='left'
    )

    # Add placement to impressions
    imp_with_placement = imp.merge(
        au[['AUCTION_ID', 'PLACEMENT']],
        left_on='AUCTION_ID',
        right_on='AUCTION_ID',
        how='left'
    )

    # Add placement to clicks
    clicks_with_placement = clicks.merge(
        au[['AUCTION_ID', 'PLACEMENT']],
        left_on='AUCTION_ID',
        right_on='AUCTION_ID',
        how='left'
    )

    placements = sorted(au['PLACEMENT'].dropna().unique())

    log(f"\n--- Volume by Placement ---", f)
    log(f"{'Placement':<12} {'Auctions':>12} {'Bids':>12} {'Winners':>12} {'Impressions':>12} {'Clicks':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}", f)

    summary_data = []
    for p in placements:
        n_auctions = (au['PLACEMENT'] == p).sum()
        n_bids = (ar_with_placement['PLACEMENT'] == p).sum()
        n_winners = ((ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)).sum()
        n_imp = (imp_with_placement['PLACEMENT'] == p).sum()
        n_clicks = (clicks_with_placement['PLACEMENT'] == p).sum()

        log(f"{str(p):<12} {n_auctions:>12,} {n_bids:>12,} {n_winners:>12,} {n_imp:>12,} {n_clicks:>10,}", f)
        summary_data.append({
            'placement': p,
            'auctions': n_auctions,
            'bids': n_bids,
            'winners': n_winners,
            'impressions': n_imp,
            'clicks': n_clicks
        })

    # Totals
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}", f)
    log(f"{'TOTAL':<12} {len(au):>12,} {len(ar):>12,} {ar['IS_WINNER'].sum():>12,} {len(imp):>12,} {len(clicks):>10,}", f)

    # CTR by placement
    log(f"\n--- CTR and Delivery Rates by Placement ---", f)
    log(f"{'Placement':<12} {'CTR':>12} {'Imp/Auction':>14} {'Imp/Winner':>12} {'Auction w/ Imp%':>16}", f)
    log(f"{'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*16}", f)

    for row in summary_data:
        p = row['placement']
        ctr = (row['clicks'] / row['impressions'] * 100) if row['impressions'] > 0 else 0
        imp_per_auction = row['impressions'] / row['auctions'] if row['auctions'] > 0 else 0
        imp_per_winner = row['impressions'] / row['winners'] if row['winners'] > 0 else 0

        # % auctions that have at least one impression
        auctions_this_placement = au[au['PLACEMENT'] == p]['AUCTION_ID'].unique()
        auctions_with_imp = imp_with_placement[imp_with_placement['PLACEMENT'] == p]['AUCTION_ID'].nunique()
        pct_with_imp = (auctions_with_imp / len(auctions_this_placement) * 100) if len(auctions_this_placement) > 0 else 0

        log(f"{str(p):<12} {ctr:>11.2f}% {imp_per_auction:>14.2f} {imp_per_winner:>12.3f} {pct_with_imp:>15.1f}%", f)

    return ar_with_placement, imp_with_placement, clicks_with_placement

# =============================================================================
# SECTION 2: SESSION STRUCTURE BY PLACEMENT
# =============================================================================
def session_structure(au, ar_with_placement, imp_with_placement, f):
    """Bids per auction, winners per auction, impressions per auction by placement."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 2: SESSION STRUCTURE BY PLACEMENT", f)
    log(f"{'='*80}", f)

    placements = sorted(au['PLACEMENT'].dropna().unique())

    # Bids per auction
    log(f"\n--- Bids per Auction ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8} {'Std':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}", f)

    for p in placements:
        bids_per_auction = ar_with_placement[ar_with_placement['PLACEMENT'] == p].groupby('AUCTION_ID').size()
        if len(bids_per_auction) > 0:
            log(f"{str(p):<12} {bids_per_auction.mean():>10.1f} {bids_per_auction.median():>10.0f} {bids_per_auction.min():>8} {bids_per_auction.max():>8} {bids_per_auction.std():>10.1f}", f)

    # Winners per auction
    log(f"\n--- Winners per Auction ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8} {'Std':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}", f)

    for p in placements:
        winners_per_auction = ar_with_placement[
            (ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)
        ].groupby('AUCTION_ID').size()
        if len(winners_per_auction) > 0:
            log(f"{str(p):<12} {winners_per_auction.mean():>10.1f} {winners_per_auction.median():>10.0f} {winners_per_auction.min():>8} {winners_per_auction.max():>8} {winners_per_auction.std():>10.1f}", f)

    # Impressions per auction
    log(f"\n--- Impressions per Auction (for auctions with any impression) ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8} {'Std':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}", f)

    for p in placements:
        imp_per_auction = imp_with_placement[imp_with_placement['PLACEMENT'] == p].groupby('AUCTION_ID').size()
        if len(imp_per_auction) > 0:
            log(f"{str(p):<12} {imp_per_auction.mean():>10.1f} {imp_per_auction.median():>10.0f} {imp_per_auction.min():>8} {imp_per_auction.max():>8} {imp_per_auction.std():>10.1f}", f)

    # Distribution of n_items shown (impressions per auction)
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

# =============================================================================
# SECTION 3: USER BEHAVIOR PATTERNS
# =============================================================================
def user_behavior(au, clicks_with_placement, f):
    """User patterns: multi-placement visits, transitions, time gaps."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 3: USER BEHAVIOR PATTERNS", f)
    log(f"{'='*80}", f)

    # Ensure timestamp column is datetime
    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    placements = sorted(au['PLACEMENT'].dropna().unique())

    # Users visiting multiple placements
    log(f"\n--- Multi-Placement Users ---", f)
    user_placements = au.groupby('USER_ID')['PLACEMENT'].nunique()
    total_users = len(user_placements)

    log(f"Total unique users: {total_users:,}", f)
    log(f"\n{'Num Placements':>15} {'Users':>12} {'Percentage':>12}", f)
    log(f"{'-'*15} {'-'*12} {'-'*12}", f)

    for n in sorted(user_placements.unique()):
        count = (user_placements == n).sum()
        pct = count / total_users * 100
        log(f"{n:>15} {count:>12,} {pct:>11.1f}%", f)

    # Placement visit distribution per user
    log(f"\n--- Auctions per User by Placement ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Max':>10} {'Users':>12}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12}", f)

    for p in placements:
        user_auction_counts = au[au['PLACEMENT'] == p].groupby('USER_ID').size()
        if len(user_auction_counts) > 0:
            log(f"{str(p):<12} {user_auction_counts.mean():>10.1f} {user_auction_counts.median():>10.0f} {user_auction_counts.max():>10} {len(user_auction_counts):>12,}", f)

    # Placement transition matrix
    log(f"\n--- Placement Transition Matrix (From -> To, same user, ordered by time) ---", f)

    # Sort by user and time
    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['PREV_PLACEMENT'] = au_sorted.groupby('USER_ID')['PLACEMENT'].shift(1)

    transitions = au_sorted[au_sorted['PREV_PLACEMENT'].notna()].copy()

    if len(transitions) > 0:
        # Build transition counts
        transition_counts = transitions.groupby(['PREV_PLACEMENT', 'PLACEMENT']).size().unstack(fill_value=0)

        # Normalize to percentages (row-wise)
        transition_pcts = transition_counts.div(transition_counts.sum(axis=1), axis=0) * 100

        log(f"\nRaw counts:", f)
        header = "From/To".ljust(10) + " " + " ".join([str(p).rjust(10) for p in placements])
        log(header, f)
        log("-"*10 + " " + " ".join(["-"*10 for _ in placements]), f)

        for from_p in placements:
            if from_p in transition_counts.index:
                row = [str(transition_counts.loc[from_p, to_p]).rjust(10) if to_p in transition_counts.columns else "0".rjust(10) for to_p in placements]
                log(str(from_p).ljust(10) + " " + " ".join(row), f)

        log(f"\nRow percentages (probability of next placement):", f)
        header = "From/To".ljust(10) + " " + " ".join([str(p).rjust(10) for p in placements])
        log(header, f)
        log("-"*10 + " " + " ".join(["-"*10 for _ in placements]), f)

        for from_p in placements:
            if from_p in transition_pcts.index:
                row = [f"{transition_pcts.loc[from_p, to_p]:.1f}%".rjust(10) if to_p in transition_pcts.columns else "0.0%".rjust(10) for to_p in placements]
                log(str(from_p).ljust(10) + " " + " ".join(row), f)

    # Time gaps between auctions by placement
    log(f"\n--- Time Gap Between Consecutive Auctions (same user, seconds) ---", f)

    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()

    log(f"{'Placement':<12} {'Mean':>12} {'Median':>12} {'P25':>10} {'P75':>10} {'P95':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)

    for p in placements:
        gaps = au_sorted[(au_sorted['PLACEMENT'] == p) & (au_sorted['TIME_GAP'].notna())]['TIME_GAP']
        if len(gaps) > 0:
            log(f"{str(p):<12} {gaps.mean():>12.1f} {gaps.median():>12.1f} {gaps.quantile(0.25):>10.1f} {gaps.quantile(0.75):>10.1f} {gaps.quantile(0.95):>10.1f}", f)

    # Time from auction to click
    log(f"\n--- Time from Auction to Click (seconds) ---", f)

    # Need to merge auction time with click time
    if len(clicks_with_placement) > 0:
        clicks_with_auction_time = clicks_with_placement.merge(
            au[['AUCTION_ID', 'CREATED_AT']].rename(columns={'CREATED_AT': 'AUCTION_TIME'}),
            on='AUCTION_ID',
            how='left'
        )

        if 'OCCURRED_AT' in clicks_with_auction_time.columns:
            if not pd.api.types.is_datetime64_any_dtype(clicks_with_auction_time['OCCURRED_AT']):
                clicks_with_auction_time['OCCURRED_AT'] = pd.to_datetime(clicks_with_auction_time['OCCURRED_AT'])

            clicks_with_auction_time['TIME_TO_CLICK'] = (
                clicks_with_auction_time['OCCURRED_AT'] - clicks_with_auction_time['AUCTION_TIME']
            ).dt.total_seconds()

            log(f"{'Placement':<12} {'Mean':>12} {'Median':>12} {'P25':>10} {'P75':>10} {'P95':>10}", f)
            log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)

            for p in placements:
                times = clicks_with_auction_time[
                    (clicks_with_auction_time['PLACEMENT'] == p) &
                    (clicks_with_auction_time['TIME_TO_CLICK'].notna()) &
                    (clicks_with_auction_time['TIME_TO_CLICK'] >= 0)
                ]['TIME_TO_CLICK']
                if len(times) > 0:
                    log(f"{str(p):<12} {times.mean():>12.1f} {times.median():>12.1f} {times.quantile(0.25):>10.1f} {times.quantile(0.75):>10.1f} {times.quantile(0.95):>10.1f}", f)

# =============================================================================
# SECTION 4: RANK/POSITION ANALYSIS
# =============================================================================
def rank_position_analysis(ar_with_placement, imp_with_placement, f):
    """What ranks get impressions by placement."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 4: RANK/POSITION ANALYSIS", f)
    log(f"{'='*80}", f)

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    # Maximum rank that wins (IS_WINNER=True) by placement
    log(f"\n--- Winner Rank Statistics by Placement ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Max':>10} {'Mode':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", f)

    for p in placements:
        winner_ranks = ar_with_placement[
            (ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)
        ]['RANKING']
        if len(winner_ranks) > 0:
            mode_val = winner_ranks.mode().iloc[0] if len(winner_ranks.mode()) > 0 else np.nan
            log(f"{str(p):<12} {winner_ranks.mean():>10.1f} {winner_ranks.median():>10.0f} {winner_ranks.max():>10} {mode_val:>10.0f}", f)

    # Rank distribution of impressions
    log(f"\n--- Rank Distribution of Impressions ---", f)

    # Join impressions to get rank
    imp_with_rank = imp_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

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

    # Maximum rank that gets an impression by placement
    log(f"\n--- Maximum Rank Shown (impression) by Placement ---", f)
    log(f"{'Placement':<12} {'Max Rank Shown':>15}", f)
    log(f"{'-'*12} {'-'*15}", f)

    for p in placements:
        ranks = imp_with_rank[imp_with_rank['PLACEMENT'] == p]['RANKING']
        if len(ranks) > 0:
            log(f"{str(p):<12} {ranks.max():>15}", f)

    # Detailed rank distribution
    log(f"\n--- Detailed Rank Distribution of Impressions ---", f)
    for p in placements:
        ranks = imp_with_rank[imp_with_rank['PLACEMENT'] == p]['RANKING'].dropna()
        if len(ranks) > 0:
            log(f"\nPlacement {p}:", f)
            rank_counts = ranks.value_counts().sort_index()
            total = len(ranks)
            log(f"{'Rank':>8} {'Count':>12} {'Percentage':>12} {'Cumulative':>12}", f)
            log(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}", f)
            cumsum = 0
            for rank in sorted(rank_counts.index)[:15]:  # Top 15 ranks
                count = rank_counts[rank]
                pct = count / total * 100
                cumsum += pct
                log(f"{rank:>8} {count:>12,} {pct:>11.1f}% {cumsum:>11.1f}%", f)

# =============================================================================
# SECTION 5: PLACEMENT INTERPRETATION
# =============================================================================
def placement_interpretation(au, ar_with_placement, imp_with_placement, clicks_with_placement, f):
    """Evidence-based interpretation of each placement type."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 5: PLACEMENT INTERPRETATION (Evidence-Based)", f)
    log(f"{'='*80}", f)

    placements = sorted(au['PLACEMENT'].dropna().unique())

    for p in placements:
        log(f"\n{'='*60}", f)
        log(f"PLACEMENT {p}", f)
        log(f"{'='*60}", f)

        # Volume
        n_auctions = (au['PLACEMENT'] == p).sum()
        n_users = au[au['PLACEMENT'] == p]['USER_ID'].nunique()
        auctions_per_user = n_auctions / n_users if n_users > 0 else 0

        log(f"\nVolume:", f)
        log(f"  Auctions: {n_auctions:,} ({n_auctions/len(au)*100:.1f}% of total)", f)
        log(f"  Users: {n_users:,}", f)
        log(f"  Auctions per user: {auctions_per_user:.1f}", f)

        # Session structure
        bids_per_auction = ar_with_placement[ar_with_placement['PLACEMENT'] == p].groupby('AUCTION_ID').size()
        winners_per_auction = ar_with_placement[
            (ar_with_placement['PLACEMENT'] == p) & (ar_with_placement['IS_WINNER'] == True)
        ].groupby('AUCTION_ID').size()
        imp_per_auction = imp_with_placement[imp_with_placement['PLACEMENT'] == p].groupby('AUCTION_ID').size()

        log(f"\nSession structure:", f)
        log(f"  Bids per auction: mean={bids_per_auction.mean():.1f}, median={bids_per_auction.median():.0f}", f)
        log(f"  Winners per auction: mean={winners_per_auction.mean():.1f}, median={winners_per_auction.median():.0f}", f)
        if len(imp_per_auction) > 0:
            log(f"  Impressions per auction: mean={imp_per_auction.mean():.1f}, median={imp_per_auction.median():.0f}", f)

        # Click behavior
        n_imp = (imp_with_placement['PLACEMENT'] == p).sum()
        n_clicks = (clicks_with_placement['PLACEMENT'] == p).sum()
        ctr = n_clicks / n_imp * 100 if n_imp > 0 else 0

        log(f"\nClick behavior:", f)
        log(f"  Impressions: {n_imp:,}", f)
        log(f"  Clicks: {n_clicks:,}", f)
        log(f"  CTR: {ctr:.2f}%", f)

        # Time patterns
        au_p = au[au['PLACEMENT'] == p].copy()
        if not pd.api.types.is_datetime64_any_dtype(au_p['CREATED_AT']):
            au_p['CREATED_AT'] = pd.to_datetime(au_p['CREATED_AT'])

        au_p_sorted = au_p.sort_values(['USER_ID', 'CREATED_AT'])
        au_p_sorted['TIME_GAP'] = au_p_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
        gaps = au_p_sorted['TIME_GAP'].dropna()

        log(f"\nTime patterns:", f)
        if len(gaps) > 0:
            log(f"  Time gap between consecutive auctions: median={gaps.median():.1f}s, mean={gaps.mean():.1f}s", f)
            rapid_fire = (gaps < 5).sum() / len(gaps) * 100
            log(f"  Rapid-fire (<5s gap): {rapid_fire:.1f}%", f)

        # Evidence summary
        log(f"\nCharacteristics summary:", f)

        if auctions_per_user > 3:
            log(f"  - HIGH auction frequency per user ({auctions_per_user:.1f})", f)
        elif auctions_per_user < 1.5:
            log(f"  - LOW auction frequency per user ({auctions_per_user:.1f})", f)

        if len(bids_per_auction) > 0 and bids_per_auction.mean() > 50:
            log(f"  - HIGH bid volume per auction ({bids_per_auction.mean():.1f})", f)
        elif len(bids_per_auction) > 0 and bids_per_auction.mean() < 30:
            log(f"  - LOW bid volume per auction ({bids_per_auction.mean():.1f})", f)

        if ctr > 4:
            log(f"  - HIGH CTR ({ctr:.2f}%)", f)
        elif ctr < 2:
            log(f"  - LOW CTR ({ctr:.2f}%)", f)

        if len(gaps) > 0 and gaps.median() < 10:
            log(f"  - RAPID session tempo (median gap {gaps.median():.1f}s)", f)
        elif len(gaps) > 0 and gaps.median() > 60:
            log(f"  - SLOW session tempo (median gap {gaps.median():.1f}s)", f)

# =============================================================================
# SECTION 6: POWER USER ANALYSIS
# =============================================================================
def power_user_analysis(au, f):
    """Users with extreme auction counts."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 6: POWER USER ANALYSIS", f)
    log(f"{'='*80}", f)

    # Ensure timestamp column is datetime
    au = au.copy()
    if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

    placements = sorted(au['PLACEMENT'].dropna().unique())

    # Auctions per user
    auctions_per_user = au.groupby('USER_ID').size()

    log(f"\n--- Auction Count Distribution Across Users ---", f)
    log(f"Total users: {len(auctions_per_user):,}", f)
    log(f"Mean auctions/user: {auctions_per_user.mean():.2f}", f)
    log(f"Median auctions/user: {auctions_per_user.median():.0f}", f)
    log(f"Max auctions/user: {auctions_per_user.max()}", f)

    log(f"\n{'Percentile':>12} {'Auctions':>12}", f)
    log(f"{'-'*12} {'-'*12}", f)
    for pct in [50, 75, 90, 95, 99, 99.9]:
        val = auctions_per_user.quantile(pct/100)
        log(f"{pct:>11}% {val:>12.0f}", f)

    # Power users (top 1%)
    threshold_99 = auctions_per_user.quantile(0.99)
    power_users = auctions_per_user[auctions_per_user >= threshold_99].index

    log(f"\n--- Power Users (Top 1%, >= {threshold_99:.0f} auctions) ---", f)
    log(f"Number of power users: {len(power_users):,}", f)

    # Placement distribution for power users
    power_user_auctions = au[au['USER_ID'].isin(power_users)]

    log(f"\nPlacement distribution for power users:", f)
    log(f"{'Placement':<12} {'Count':>12} {'Percentage':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12}", f)

    placement_counts = power_user_auctions['PLACEMENT'].value_counts()
    for p in sorted(placement_counts.index):
        count = placement_counts[p]
        pct = count / len(power_user_auctions) * 100
        log(f"{str(p):<12} {count:>12,} {pct:>11.1f}%", f)

    # Compare to overall
    log(f"\nComparison to overall distribution:", f)
    overall_dist = au['PLACEMENT'].value_counts(normalize=True) * 100
    power_dist = power_user_auctions['PLACEMENT'].value_counts(normalize=True) * 100

    log(f"{'Placement':<12} {'Overall %':>12} {'Power User %':>14} {'Diff':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*14} {'-'*10}", f)

    for p in placements:
        overall_pct = overall_dist.get(p, 0)
        power_pct = power_dist.get(p, 0)
        diff = power_pct - overall_pct
        log(f"{str(p):<12} {overall_pct:>11.1f}% {power_pct:>13.1f}% {diff:>+9.1f}%", f)

    # Time patterns for power users
    log(f"\n--- Time Patterns for Power Users ---", f)

    power_sorted = power_user_auctions.sort_values(['USER_ID', 'CREATED_AT'])
    power_sorted['TIME_GAP'] = power_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()

    for p in placements:
        gaps = power_sorted[(power_sorted['PLACEMENT'] == p) & (power_sorted['TIME_GAP'].notna())]['TIME_GAP']
        if len(gaps) > 0:
            rapid_fire = (gaps < 5).sum() / len(gaps) * 100
            log(f"Placement {p}: median gap={gaps.median():.1f}s, rapid-fire (<5s)={rapid_fire:.1f}%", f)

    # Top 10 power users detail
    log(f"\n--- Top 10 Power Users ---", f)
    top_10_users = auctions_per_user.nlargest(10)

    log(f"{'User (truncated)':>20} {'Auctions':>10} {'Placements':>12} {'Most Common':>12}", f)
    log(f"{'-'*20} {'-'*10} {'-'*12} {'-'*12}", f)

    for user_id, n_auctions in top_10_users.items():
        user_data = au[au['USER_ID'] == user_id]
        n_placements = user_data['PLACEMENT'].nunique()
        most_common = user_data['PLACEMENT'].mode().iloc[0] if len(user_data['PLACEMENT'].mode()) > 0 else '-'
        log(f"{str(user_id)[:20]:>20} {n_auctions:>10} {n_placements:>12} {str(most_common):>12}", f)

# =============================================================================
# SECTION 7: PRODUCT CHARACTERISTICS
# =============================================================================
def product_characteristics(ar_with_placement, catalog, f):
    """Price distribution and brand/category patterns by placement."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 7: PRODUCT CHARACTERISTICS BY PLACEMENT", f)
    log(f"{'='*80}", f)

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    # Join to get catalog info for winning bids
    winners = ar_with_placement[ar_with_placement['IS_WINNER'] == True].copy()
    winners_with_catalog = winners.merge(
        catalog[['PRODUCT_ID', 'CATALOG_PRICE', 'NAME', 'CATEGORIES']],
        on='PRODUCT_ID',
        how='left',
        suffixes=('', '_catalog')
    )

    # Price distribution by placement
    log(f"\n--- Price Distribution by Placement (Winners Only) ---", f)
    log(f"{'Placement':<12} {'Mean':>12} {'Median':>12} {'P25':>10} {'P75':>10} {'P95':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}", f)

    for p in placements:
        prices = winners_with_catalog[winners_with_catalog['PLACEMENT'] == p]['CATALOG_PRICE'].dropna()
        if len(prices) > 0:
            log(f"{str(p):<12} ${prices.mean():>10.2f} ${prices.median():>10.2f} ${prices.quantile(0.25):>8.2f} ${prices.quantile(0.75):>8.2f} ${prices.quantile(0.95):>10.2f}", f)

    # Price range distribution
    log(f"\n--- Price Range Distribution by Placement ---", f)
    log(f"{'Placement':<12} {'<$20':>10} {'$20-50':>10} {'$50-100':>10} {'$100-500':>12} {'$500+':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}", f)

    for p in placements:
        prices = winners_with_catalog[winners_with_catalog['PLACEMENT'] == p]['CATALOG_PRICE'].dropna()
        total = len(prices)
        if total > 0:
            pct_under_20 = (prices < 20).sum() / total * 100
            pct_20_50 = ((prices >= 20) & (prices < 50)).sum() / total * 100
            pct_50_100 = ((prices >= 50) & (prices < 100)).sum() / total * 100
            pct_100_500 = ((prices >= 100) & (prices < 500)).sum() / total * 100
            pct_500plus = (prices >= 500).sum() / total * 100
            log(f"{str(p):<12} {pct_under_20:>9.1f}% {pct_20_50:>9.1f}% {pct_50_100:>9.1f}% {pct_100_500:>11.1f}% {pct_500plus:>9.1f}%", f)

    # Top products by placement
    log(f"\n--- Top 5 Products by Placement (by frequency in winning bids) ---", f)

    for p in placements:
        log(f"\nPlacement {p}:", f)
        top_products = winners_with_catalog[winners_with_catalog['PLACEMENT'] == p]['PRODUCT_ID'].value_counts().head(5)

        log(f"{'Product ID (truncated)':<30} {'Count':>10} {'Price':>10}", f)
        log(f"{'-'*30} {'-'*10} {'-'*10}", f)

        for prod_id, count in top_products.items():
            price_info = winners_with_catalog[winners_with_catalog['PRODUCT_ID'] == prod_id]['CATALOG_PRICE'].iloc[0] if len(winners_with_catalog[winners_with_catalog['PRODUCT_ID'] == prod_id]) > 0 else np.nan
            price_str = f"${price_info:.2f}" if pd.notna(price_info) else "N/A"
            log(f"{str(prod_id)[:30]:<30} {count:>10,} {price_str:>10}", f)

    # Category analysis (if available)
    log(f"\n--- Category Presence by Placement ---", f)
    log(f"Note: CATEGORIES is an array field; counting products with non-empty categories", f)

    for p in placements:
        subset = winners_with_catalog[winners_with_catalog['PLACEMENT'] == p]
        has_categories = subset['CATEGORIES'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)
        pct_with_cats = has_categories.sum() / len(subset) * 100 if len(subset) > 0 else 0
        log(f"Placement {p}: {pct_with_cats:.1f}% have categories", f)

# =============================================================================
# SECTION 8: CLICK PATTERNS
# =============================================================================
def click_patterns(au, ar_with_placement, imp_with_placement, clicks_with_placement, f):
    """Click behavior by rank and placement."""
    log(f"\n{'='*80}", f)
    log(f"SECTION 8: CLICK PATTERNS", f)
    log(f"{'='*80}", f)

    placements = sorted(au['PLACEMENT'].dropna().unique())

    # Join clicks to get rank
    clicks_with_rank = clicks_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    # Impressions with rank
    imp_with_rank = imp_with_placement.merge(
        ar_with_placement[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    # CTR by rank within each placement
    log(f"\n--- CTR by Rank within Placement ---", f)

    for p in placements:
        log(f"\nPlacement {p}:", f)
        log(f"{'Rank':>8} {'Impressions':>15} {'Clicks':>10} {'CTR':>10}", f)
        log(f"{'-'*8} {'-'*15} {'-'*10} {'-'*10}", f)

        imp_p = imp_with_rank[imp_with_rank['PLACEMENT'] == p]
        clicks_p = clicks_with_rank[clicks_with_rank['PLACEMENT'] == p]

        # Get unique ranks
        all_ranks = sorted(imp_p['RANKING'].dropna().unique())

        for rank in all_ranks[:10]:  # Top 10 ranks
            n_imp = (imp_p['RANKING'] == rank).sum()
            n_clicks = (clicks_p['RANKING'] == rank).sum()
            ctr = n_clicks / n_imp * 100 if n_imp > 0 else 0
            log(f"{rank:>8} {n_imp:>15,} {n_clicks:>10,} {ctr:>9.2f}%", f)

    # Multi-click sessions by placement
    log(f"\n--- Multi-Click Sessions by Placement ---", f)

    clicks_per_auction = clicks_with_placement.groupby(['AUCTION_ID', 'PLACEMENT']).size().reset_index(name='n_clicks')

    log(f"{'Placement':<12} {'1 click':>12} {'2 clicks':>12} {'3+ clicks':>12} {'Total':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}", f)

    for p in placements:
        subset = clicks_per_auction[clicks_per_auction['PLACEMENT'] == p]
        total = len(subset)
        if total > 0:
            pct_1 = (subset['n_clicks'] == 1).sum() / total * 100
            pct_2 = (subset['n_clicks'] == 2).sum() / total * 100
            pct_3plus = (subset['n_clicks'] >= 3).sum() / total * 100
            log(f"{str(p):<12} {pct_1:>11.1f}% {pct_2:>11.1f}% {pct_3plus:>11.1f}% {total:>10,}", f)

    # Click timing patterns
    log(f"\n--- Click Timing Patterns (time between clicks in same auction) ---", f)

    # Ensure timestamp is datetime
    if 'OCCURRED_AT' in clicks_with_placement.columns:
        clicks_sorted = clicks_with_placement.copy()
        if not pd.api.types.is_datetime64_any_dtype(clicks_sorted['OCCURRED_AT']):
            clicks_sorted['OCCURRED_AT'] = pd.to_datetime(clicks_sorted['OCCURRED_AT'])

        clicks_sorted = clicks_sorted.sort_values(['AUCTION_ID', 'OCCURRED_AT'])
        clicks_sorted['TIME_SINCE_PREV_CLICK'] = clicks_sorted.groupby('AUCTION_ID')['OCCURRED_AT'].diff().dt.total_seconds()

        log(f"{'Placement':<12} {'Mean':>12} {'Median':>12} {'Min':>10} {'Max':>10}", f)
        log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", f)

        for p in placements:
            times = clicks_sorted[
                (clicks_sorted['PLACEMENT'] == p) &
                (clicks_sorted['TIME_SINCE_PREV_CLICK'].notna())
            ]['TIME_SINCE_PREV_CLICK']
            if len(times) > 0:
                log(f"{str(p):<12} {times.mean():>12.1f} {times.median():>12.1f} {times.min():>10.1f} {times.max():>10.1f}", f)

    # Rank of clicked items vs all shown items
    log(f"\n--- Rank Comparison: Clicked vs Shown ---", f)

    for p in placements:
        imp_ranks = imp_with_rank[imp_with_rank['PLACEMENT'] == p]['RANKING'].dropna()
        click_ranks = clicks_with_rank[clicks_with_rank['PLACEMENT'] == p]['RANKING'].dropna()

        if len(imp_ranks) > 0 and len(click_ranks) > 0:
            log(f"\nPlacement {p}:", f)
            log(f"  Shown items: mean rank={imp_ranks.mean():.2f}, median={imp_ranks.median():.0f}", f)
            log(f"  Clicked items: mean rank={click_ranks.mean():.2f}, median={click_ranks.median():.0f}", f)
            log(f"  Difference (clicked - shown): {click_ranks.mean() - imp_ranks.mean():.2f}", f)

# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("="*80, f)
        log("PLACEMENT EDA: COMPREHENSIVE CHARACTERIZATION", f)
        log("="*80, f)
        log(f"Purpose: Document all placement characteristics from raw user sessions", f)
        log(f"Data source: data_r2/*.parquet", f)

        # Load data
        log(f"\n{'='*80}", f)
        log(f"LOADING DATA", f)
        log(f"{'='*80}", f)

        files_to_load = {
            'auctions_users': 'auctions_users_r2.parquet',
            'auctions_results': 'auctions_results_r2.parquet',
            'impressions': 'impressions_r2.parquet',
            'clicks': 'clicks_r2.parquet',
            'catalog': 'catalog_r2.parquet',
        }

        data = {}
        for name, filename in tqdm(files_to_load.items(), desc="Loading parquet files"):
            filepath = DATA_DIR / filename
            if filepath.exists():
                data[name] = pd.read_parquet(filepath)
                log(f"Loaded {name}: {len(data[name]):,} rows", f)
            else:
                log(f"ERROR: Missing {filename}", f)
                return

        au = data['auctions_users']
        ar = data['auctions_results']
        imp = data['impressions']
        clicks = data['clicks']
        catalog = data['catalog']

        # Data overview
        log(f"\n--- Data Overview ---", f)
        log(f"Auctions: {len(au):,}", f)
        log(f"Bids: {len(ar):,}", f)
        log(f"Impressions: {len(imp):,}", f)
        log(f"Clicks: {len(clicks):,}", f)
        log(f"Catalog products: {len(catalog):,}", f)
        log(f"Unique users: {au['USER_ID'].nunique():,}", f)
        log(f"Unique placements: {au['PLACEMENT'].nunique()}", f)
        log(f"Placement values: {sorted(au['PLACEMENT'].dropna().unique())}", f)

        # Section 1: Placement Overview
        ar_with_placement, imp_with_placement, clicks_with_placement = placement_overview(au, ar, imp, clicks, f)

        # Section 2: Session Structure
        session_structure(au, ar_with_placement, imp_with_placement, f)

        # Section 3: User Behavior
        user_behavior(au, clicks_with_placement, f)

        # Section 4: Rank/Position Analysis
        rank_position_analysis(ar_with_placement, imp_with_placement, f)

        # Section 5: Placement Interpretation
        placement_interpretation(au, ar_with_placement, imp_with_placement, clicks_with_placement, f)

        # Section 6: Power User Analysis
        power_user_analysis(au, f)

        # Section 7: Product Characteristics
        product_characteristics(ar_with_placement, catalog, f)

        # Section 8: Click Patterns
        click_patterns(au, ar_with_placement, imp_with_placement, clicks_with_placement, f)

        # Final summary
        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {OUTPUT_FILE}", f)

if __name__ == "__main__":
    main()
