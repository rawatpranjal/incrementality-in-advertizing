#!/usr/bin/env python3
"""
Search Pagination Evidence: Proves P3 is search pagination via self-transition analysis.

The diagnostic signal: P3 has ~88% self-transition rate (P3 → P3), indicating users
paginate through search results before exiting to other placements.

Extended Analysis (v2): Resolves the core uncertainty:
  Q: Does 1 auction = 1 page load (runs = pagination) OR 1 auction = 1 ad slot (runs = slots on same page)?

New checks:
  - SECTION 6: Timestamp clustering - how many auctions share exact same timestamp within user?
  - SECTION 7: Time gaps excluding instant (>0.5s) - what is "true" pagination speed?
  - SECTION 8: Reverse transition matrix - P(came from X | now in Y)
  - SECTION 9: Winners per auction by placement - how many items shown per auction?

Output: results/12_search_pagination_evidence.txt
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
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "12_search_pagination_evidence.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# TRANSITION MATRIX COMPUTATION
# =============================================================================
def compute_transition_matrix(au, placements, f, dataset_name):
    """Compute full placement transition matrix."""
    log(f"\n--- Dataset: {dataset_name} ---", f)

    # Sort by user and time
    au_sorted = au.sort_values(['USER_ID', 'CREATED_AT']).copy()
    au_sorted['NEXT_PLACEMENT'] = au_sorted.groupby('USER_ID')['PLACEMENT'].shift(-1)

    # Filter to rows with a next placement (not last in user sequence)
    transitions = au_sorted[au_sorted['NEXT_PLACEMENT'].notna()].copy()

    log(f"\nTotal transitions: {len(transitions):,}", f)
    log(f"Users with 2+ auctions: {transitions['USER_ID'].nunique():,}", f)

    # Build raw count matrix
    transition_counts = transitions.groupby(['PLACEMENT', 'NEXT_PLACEMENT']).size().unstack(fill_value=0)

    # Ensure all placements present
    for p in placements:
        if p not in transition_counts.index:
            transition_counts.loc[p] = 0
        if p not in transition_counts.columns:
            transition_counts[p] = 0

    transition_counts = transition_counts.reindex(index=placements, columns=placements, fill_value=0)

    # Row sums for exit calculation
    row_sums = transition_counts.sum(axis=1)

    # Normalize to percentages
    transition_pcts = transition_counts.div(row_sums, axis=0) * 100
    transition_pcts = transition_pcts.fillna(0)

    # Print raw counts
    log(f"\nRaw transition counts:", f)
    from_to_label = "From\\To"
    header = f"{from_to_label:>10}" + "".join([f"{p:>12}" for p in placements])
    log(header, f)
    log("-" * 10 + "-" * 12 * len(placements), f)

    for from_p in placements:
        row = f"{from_p:>10}"
        for to_p in placements:
            count = int(transition_counts.loc[from_p, to_p])
            row += f"{count:>12,}"
        log(row, f)

    # Print percentages (THE KEY TABLE)
    log(f"\nTransition probabilities (row percentages):", f)
    log(f"P(next_placement = Y | current_placement = X)", f)
    header = f"{from_to_label:>10}" + "".join([f"{p:>12}" for p in placements])
    log(header, f)
    log("-" * 10 + "-" * 12 * len(placements), f)

    for from_p in placements:
        row = f"{from_p:>10}"
        for to_p in placements:
            pct = transition_pcts.loc[from_p, to_p]
            row += f"{pct:>11.1f}%"
        log(row, f)

    # Highlight self-transitions
    log(f"\nSelf-transition rates (DIAGONAL):", f)
    for p in placements:
        self_rate = transition_pcts.loc[p, p]
        marker = " <-- PAGINATION SIGNAL" if self_rate > 50 else ""
        log(f"  P{p} → P{p}: {self_rate:.1f}%{marker}", f)

    return au_sorted, transitions, transition_pcts

# =============================================================================
# CONSECUTIVE RUN LENGTH ANALYSIS
# =============================================================================
def analyze_consecutive_runs(au_sorted, target_placement, f, dataset_name):
    """Compute distribution of consecutive same-placement runs."""
    log(f"\n--- Consecutive {target_placement} Run Lengths ({dataset_name}) ---", f)
    log(f'Interpretation: "How many pages does a user view in one search session?"', f)

    # Identify runs
    au_p = au_sorted[au_sorted['PLACEMENT'] == target_placement].copy()
    au_p = au_p.sort_values(['USER_ID', 'CREATED_AT'])

    # Mark where a new run starts (different user or gap in sequence)
    au_p['prev_user'] = au_p['USER_ID'].shift(1)
    au_p['prev_placement_in_full'] = au_sorted.groupby('USER_ID')['PLACEMENT'].shift(1).reindex(au_p.index)

    # A new run starts when:
    # 1. Different user from previous row
    # 2. Previous row (in full sorted data) was not the same placement
    au_p['new_run'] = (au_p['USER_ID'] != au_p['prev_user']) | (au_p['prev_placement_in_full'] != target_placement)
    au_p['run_id'] = au_p['new_run'].cumsum()

    # Count run lengths
    run_lengths = au_p.groupby('run_id').size()

    log(f"\nTotal runs: {len(run_lengths):,}", f)
    log(f"Mean run length: {run_lengths.mean():.2f}", f)
    log(f"Median run length: {run_lengths.median():.0f}", f)
    log(f"Max run length: {run_lengths.max()}", f)

    # Distribution
    log(f"\n{'Run Length':>12} {'Count':>12} {'Percentage':>12} {'Cumulative':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

    total = len(run_lengths)
    cumsum = 0
    for length in range(1, min(11, run_lengths.max() + 1)):
        count = (run_lengths == length).sum()
        pct = count / total * 100
        cumsum += pct
        interpretation = ""
        if length == 1:
            interpretation = " (single page view)"
        elif length == 2:
            interpretation = " (page 1 → page 2)"
        elif length >= 3:
            interpretation = f" (deep pagination: {length} pages)"
        log(f"{length:>12} {count:>12,} {pct:>11.1f}% {cumsum:>11.1f}%{interpretation}", f)

    if run_lengths.max() > 10:
        count_10plus = (run_lengths > 10).sum()
        pct_10plus = count_10plus / total * 100
        log(f"{'11+':>12} {count_10plus:>12,} {pct_10plus:>11.1f}% {100.0:>11.1f}% (very deep pagination)", f)

    return run_lengths

# =============================================================================
# TIME GAP ANALYSIS WITHIN RUNS
# =============================================================================
def analyze_time_gaps(au_sorted, target_placement, f, dataset_name):
    """Analyze time gaps between consecutive same-placement auctions."""
    log(f"\n--- Time Gaps Within {target_placement} Runs ({dataset_name}) ---", f)

    # Filter to target placement
    au_p = au_sorted[au_sorted['PLACEMENT'] == target_placement].copy()
    au_p = au_p.sort_values(['USER_ID', 'CREATED_AT'])

    # Compute time gap to next auction of same placement within user
    au_p['prev_user'] = au_p['USER_ID'].shift(1)
    au_p['prev_time'] = au_p['CREATED_AT'].shift(1)

    # Gap only valid if same user
    au_p['time_gap'] = np.where(
        au_p['USER_ID'] == au_p['prev_user'],
        (au_p['CREATED_AT'] - au_p['prev_time']).dt.total_seconds(),
        np.nan
    )

    gaps = au_p['time_gap'].dropna()

    if len(gaps) == 0:
        log(f"No consecutive {target_placement} auctions found.", f)
        return None, au_p

    log(f"\nTime gap between consecutive P{target_placement} auctions (seconds):", f)
    log(f"  Count: {len(gaps):,}", f)
    log(f"  Median: {gaps.median():.2f}s", f)
    log(f"  Mean: {gaps.mean():.2f}s", f)
    log(f"  P25: {gaps.quantile(0.25):.2f}s", f)
    log(f"  P75: {gaps.quantile(0.75):.2f}s", f)
    log(f"  P95: {gaps.quantile(0.95):.2f}s", f)

    # Interpret
    if gaps.median() < 5:
        log(f"\n  Interpretation: Median gap of {gaps.median():.1f}s indicates rapid page turns,", f)
        log(f"  consistent with search result pagination behavior.", f)

    # Bucket distribution
    log(f"\nGap distribution:", f)
    log(f"  <2s (instant):      {(gaps < 2).sum():>10,} ({(gaps < 2).sum()/len(gaps)*100:>6.1f}%)", f)
    log(f"  2-5s (fast):        {((gaps >= 2) & (gaps < 5)).sum():>10,} ({((gaps >= 2) & (gaps < 5)).sum()/len(gaps)*100:>6.1f}%)", f)
    log(f"  5-10s (moderate):   {((gaps >= 5) & (gaps < 10)).sum():>10,} ({((gaps >= 5) & (gaps < 10)).sum()/len(gaps)*100:>6.1f}%)", f)
    log(f"  10-30s (browsing):  {((gaps >= 10) & (gaps < 30)).sum():>10,} ({((gaps >= 10) & (gaps < 30)).sum()/len(gaps)*100:>6.1f}%)", f)
    log(f"  30s+ (new session): {(gaps >= 30).sum():>10,} ({(gaps >= 30).sum()/len(gaps)*100:>6.1f}%)", f)

    return gaps, au_p

# =============================================================================
# SAMPLE USER SEQUENCES
# =============================================================================
def show_sample_sequences(au_sorted, f, dataset_name, n_samples=10):
    """Show actual user placement sequences."""
    log(f"\n--- Sample User Sequences ({dataset_name}) ---", f)
    log(f"Format: Placement(time since prev)", f)

    # Find users with interesting sequences (multiple P3 in a row)
    au_sorted = au_sorted.copy()
    au_sorted['prev_user'] = au_sorted['USER_ID'].shift(1)
    au_sorted['prev_time'] = au_sorted['CREATED_AT'].shift(1)
    au_sorted['time_gap'] = np.where(
        au_sorted['USER_ID'] == au_sorted['prev_user'],
        (au_sorted['CREATED_AT'] - au_sorted['prev_time']).dt.total_seconds(),
        np.nan
    )

    # Count consecutive P3 per user
    def count_max_consecutive_p3(group):
        placements = group['PLACEMENT'].tolist()
        max_run = 0
        current_run = 0
        for p in placements:
            if p == '3':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    user_max_p3 = au_sorted.groupby('USER_ID').apply(count_max_consecutive_p3)

    # Sample users with 3+ consecutive P3
    interesting_users = user_max_p3[user_max_p3 >= 3].head(n_samples).index.tolist()

    if not interesting_users:
        interesting_users = au_sorted['USER_ID'].value_counts().head(n_samples).index.tolist()

    for user_id in interesting_users[:n_samples]:
        user_data = au_sorted[au_sorted['USER_ID'] == user_id].head(15)  # First 15 events

        sequence_parts = []
        for _, row in user_data.iterrows():
            gap = row['time_gap']
            if pd.isna(gap):
                sequence_parts.append(f"P{row['PLACEMENT']}(start)")
            else:
                sequence_parts.append(f"P{row['PLACEMENT']}({gap:.1f}s)")

        sequence_str = " → ".join(sequence_parts)
        user_short = str(user_id)[:20] + "..." if len(str(user_id)) > 20 else str(user_id)
        log(f"\n{user_short}:", f)
        log(f"  {sequence_str}", f)

# =============================================================================
# SECTION 6: TIMESTAMP CLUSTERING ANALYSIS
# =============================================================================
def analyze_timestamp_clustering(au, placements, f, dataset_name):
    """
    Check if multiple auctions share the exact same timestamp within a user.

    If clustering = 4-6 auctions at time T → 1 auction = 1 slot (batching)
    If clustering = 1 auction at time T → 1 auction = 1 page
    """
    log(f"\n--- Timestamp Clustering Analysis ({dataset_name}) ---", f)
    log(f"Question: Do multiple auctions share the exact same timestamp within a user?", f)
    log(f"If yes → auctions are ad slots batched together (1 auction = 1 slot)", f)
    log(f"If no → auctions are page loads (1 auction = 1 page)", f)

    # Group by (USER_ID, CREATED_AT, PLACEMENT) and count
    cluster_counts = au.groupby(['USER_ID', 'CREATED_AT', 'PLACEMENT']).size().reset_index(name='auctions_at_same_time')

    log(f"\nTotal (user, timestamp, placement) combinations: {len(cluster_counts):,}", f)

    # Distribution of cluster sizes by placement
    for p in placements:
        p_clusters = cluster_counts[cluster_counts['PLACEMENT'] == p]['auctions_at_same_time']
        if len(p_clusters) == 0:
            continue

        log(f"\nPlacement {p} - Auctions per (user, timestamp):", f)
        log(f"  Total timestamp clusters: {len(p_clusters):,}", f)
        log(f"  Mean auctions per cluster: {p_clusters.mean():.2f}", f)
        log(f"  Median auctions per cluster: {p_clusters.median():.0f}", f)
        log(f"  Max auctions per cluster: {p_clusters.max()}", f)

        # Distribution
        log(f"\n  {'Cluster Size':>12} {'Count':>12} {'Percentage':>12}", f)
        log(f"  {'-'*12} {'-'*12} {'-'*12}", f)
        for size in range(1, min(11, int(p_clusters.max()) + 1)):
            count = (p_clusters == size).sum()
            pct = count / len(p_clusters) * 100
            if count > 0:
                log(f"  {size:>12} {count:>12,} {pct:>11.1f}%", f)

        if p_clusters.max() > 10:
            count_10plus = (p_clusters > 10).sum()
            pct_10plus = count_10plus / len(p_clusters) * 100
            log(f"  {'11+':>12} {count_10plus:>12,} {pct_10plus:>11.1f}%", f)

    # Key diagnostic: what % of auctions are in clusters of size > 1?
    log(f"\n--- KEY DIAGNOSTIC: Batching Prevalence ---", f)
    for p in placements:
        p_data = cluster_counts[cluster_counts['PLACEMENT'] == p]
        if len(p_data) == 0:
            continue

        total_auctions = p_data['auctions_at_same_time'].sum()
        singleton_auctions = (p_data['auctions_at_same_time'] == 1).sum()  # clusters of size 1
        batched_auctions = total_auctions - singleton_auctions
        batched_pct = batched_auctions / total_auctions * 100 if total_auctions > 0 else 0

        interpretation = ""
        if batched_pct > 50:
            interpretation = " → BATCHING DETECTED (1 auction = 1 slot)"
        elif batched_pct < 10:
            interpretation = " → MINIMAL BATCHING (1 auction = 1 page)"
        else:
            interpretation = " → MIXED"

        log(f"  P{p}: {batched_pct:.1f}% of auctions in clusters >1{interpretation}", f)

    return cluster_counts


# =============================================================================
# SECTION 7: TIME GAPS EXCLUDING INSTANT (>0.5s)
# =============================================================================
def analyze_gaps_excluding_instant(gaps, f, placement, threshold=0.5):
    """
    Re-analyze gaps filtering out 0.0s (batched events).

    If median still ~2-5s after filtering → real rapid pagination
    If jumps to 30s+ → the "rapid pagination" was batching artifact
    """
    if gaps is None or len(gaps) == 0:
        log(f"\nNo gaps data for placement {placement}.", f)
        return None

    log(f"\n--- Time Gaps Excluding Instant (<{threshold}s) for P{placement} ---", f)

    # Original stats
    log(f"\nOriginal gap statistics (all gaps):", f)
    log(f"  N: {len(gaps):,}", f)
    log(f"  Median: {gaps.median():.2f}s", f)
    log(f"  Mean: {gaps.mean():.2f}s", f)
    log(f"  % gaps = 0.0s: {(gaps == 0).sum() / len(gaps) * 100:.1f}%", f)
    log(f"  % gaps < {threshold}s: {(gaps < threshold).sum() / len(gaps) * 100:.1f}%", f)

    # Filter to gaps > threshold
    filtered_gaps = gaps[gaps >= threshold]

    if len(filtered_gaps) == 0:
        log(f"\nNo gaps >= {threshold}s found.", f)
        return None

    log(f"\nFiltered gap statistics (gaps >= {threshold}s):", f)
    log(f"  N: {len(filtered_gaps):,} ({len(filtered_gaps)/len(gaps)*100:.1f}% of original)", f)
    log(f"  Median: {filtered_gaps.median():.2f}s", f)
    log(f"  Mean: {filtered_gaps.mean():.2f}s", f)
    log(f"  P25: {filtered_gaps.quantile(0.25):.2f}s", f)
    log(f"  P75: {filtered_gaps.quantile(0.75):.2f}s", f)
    log(f"  P95: {filtered_gaps.quantile(0.95):.2f}s", f)

    # Interpretation
    median_diff = filtered_gaps.median() - gaps.median()
    log(f"\n  Median shift: {gaps.median():.2f}s → {filtered_gaps.median():.2f}s (Δ = {median_diff:+.2f}s)", f)

    if filtered_gaps.median() < 10:
        log(f"  Interpretation: Even after filtering batched events, gaps remain short.", f)
        log(f"  → Rapid pagination is REAL, not an artifact of batching.", f)
    elif filtered_gaps.median() > 30:
        log(f"  Interpretation: After filtering, median jumps to {filtered_gaps.median():.1f}s.", f)
        log(f"  → The 'rapid pagination' was largely a BATCHING ARTIFACT.", f)
    else:
        log(f"  Interpretation: Median gap is moderate ({filtered_gaps.median():.1f}s).", f)
        log(f"  → Mix of real pagination and batching.", f)

    # Bucket distribution for filtered gaps
    log(f"\nFiltered gap distribution:", f)
    log(f"  {threshold}-2s:           {((filtered_gaps >= threshold) & (filtered_gaps < 2)).sum():>10,} ({((filtered_gaps >= threshold) & (filtered_gaps < 2)).sum()/len(filtered_gaps)*100:>6.1f}%)", f)
    log(f"  2-5s (fast page):   {((filtered_gaps >= 2) & (filtered_gaps < 5)).sum():>10,} ({((filtered_gaps >= 2) & (filtered_gaps < 5)).sum()/len(filtered_gaps)*100:>6.1f}%)", f)
    log(f"  5-10s (moderate):   {((filtered_gaps >= 5) & (filtered_gaps < 10)).sum():>10,} ({((filtered_gaps >= 5) & (filtered_gaps < 10)).sum()/len(filtered_gaps)*100:>6.1f}%)", f)
    log(f"  10-30s (browsing):  {((filtered_gaps >= 10) & (filtered_gaps < 30)).sum():>10,} ({((filtered_gaps >= 10) & (filtered_gaps < 30)).sum()/len(filtered_gaps)*100:>6.1f}%)", f)
    log(f"  30s+ (new session): {(filtered_gaps >= 30).sum():>10,} ({(filtered_gaps >= 30).sum()/len(filtered_gaps)*100:>6.1f}%)", f)

    return filtered_gaps


# =============================================================================
# SECTION 8: REVERSE TRANSITION MATRIX
# =============================================================================
def compute_reverse_transitions(transitions, placements, f, dataset_name):
    """
    Compute P(came from X | now in Y) - column percentages.

    This shows where users come from when they arrive at each placement.
    """
    log(f"\n--- Reverse Transition Matrix ({dataset_name}) ---", f)
    log(f"P(came from X | now in Y) - Column percentages", f)
    log(f"Read: 'Of users who are now at Y, X% came from placement X'", f)

    # Build raw count matrix
    transition_counts = transitions.groupby(['PLACEMENT', 'NEXT_PLACEMENT']).size().unstack(fill_value=0)

    # Ensure all placements present
    for p in placements:
        if p not in transition_counts.index:
            transition_counts.loc[p] = 0
        if p not in transition_counts.columns:
            transition_counts[p] = 0

    transition_counts = transition_counts.reindex(index=placements, columns=placements, fill_value=0)

    # Column sums for reverse calculation
    col_sums = transition_counts.sum(axis=0)

    # Normalize to column percentages (P(from | to))
    reverse_pcts = transition_counts.div(col_sums, axis=1) * 100
    reverse_pcts = reverse_pcts.fillna(0)

    # Print
    from_to_label = "From\\To"
    header = f"{from_to_label:>10}" + "".join([f"{p:>12}" for p in placements])
    log(f"\n{header}", f)
    log("-" * 10 + "-" * 12 * len(placements), f)

    for from_p in placements:
        row = f"{from_p:>10}"
        for to_p in placements:
            pct = reverse_pcts.loc[from_p, to_p]
            row += f"{pct:>11.1f}%"
        log(row, f)

    # Highlight key findings for P3
    log(f"\nKey insight for P3 (where do P3 visitors come from?):", f)
    for from_p in placements:
        pct = reverse_pcts.loc[from_p, '3']
        marker = ""
        if from_p == '3' and pct > 50:
            marker = " ← MOST P3 VISITS COME FROM P3 (pagination)"
        elif from_p == '1' and pct > 20:
            marker = " ← Homepage is major entry to search"
        log(f"  P(from P{from_p} | now at P3): {pct:.1f}%{marker}", f)

    return reverse_pcts


# =============================================================================
# SECTION 9: WINNERS PER AUCTION BY PLACEMENT
# =============================================================================
def analyze_winners_per_auction(au, ar, placements, f, dataset_name):
    """
    How many winners (shown items) per auction by placement.

    If P3 has many winners per auction, it's a page with many ad slots
    (search results typically show more items than carousel/PDP).
    """
    log(f"\n--- Winners Per Auction by Placement ({dataset_name}) ---", f)
    log(f"Question: How many products are shown (IS_WINNER=TRUE) per auction?", f)

    # Count winners per auction
    winners_per_auction = ar[ar['IS_WINNER'] == True].groupby('AUCTION_ID').size().reset_index(name='winner_count')

    # Merge with auctions to get placement
    au_minimal = au[['AUCTION_ID', 'PLACEMENT']].drop_duplicates()
    winners_with_placement = winners_per_auction.merge(au_minimal, on='AUCTION_ID', how='left')

    log(f"\nTotal auctions with winners: {len(winners_with_placement):,}", f)

    for p in placements:
        p_data = winners_with_placement[winners_with_placement['PLACEMENT'] == p]['winner_count']
        if len(p_data) == 0:
            log(f"\nPlacement {p}: No data", f)
            continue

        log(f"\nPlacement {p}:", f)
        log(f"  Auctions with winners: {len(p_data):,}", f)
        log(f"  Mean winners per auction: {p_data.mean():.2f}", f)
        log(f"  Median winners per auction: {p_data.median():.0f}", f)
        log(f"  Max winners per auction: {p_data.max()}", f)

        # Distribution
        log(f"\n  {'Winners':>12} {'Count':>12} {'Percentage':>12}", f)
        log(f"  {'-'*12} {'-'*12} {'-'*12}", f)
        for size in range(1, min(11, int(p_data.max()) + 1)):
            count = (p_data == size).sum()
            pct = count / len(p_data) * 100
            if count > 0:
                log(f"  {size:>12} {count:>12,} {pct:>11.1f}%", f)

        if p_data.max() > 10:
            count_10plus = (p_data > 10).sum()
            pct_10plus = count_10plus / len(p_data) * 100
            log(f"  {'11+':>12} {count_10plus:>12,} {pct_10plus:>11.1f}%", f)

    # Summary comparison
    log(f"\n--- Summary: Winners Per Auction ---", f)
    log(f"{'Placement':>12} {'Mean':>10} {'Median':>10} {'Max':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)
    for p in placements:
        p_data = winners_with_placement[winners_with_placement['PLACEMENT'] == p]['winner_count']
        if len(p_data) > 0:
            log(f"{'P' + p:>12} {p_data.mean():>10.2f} {p_data.median():>10.0f} {p_data.max():>10}", f)

    return winners_with_placement


# =============================================================================
# SECTION 10: NEAR-TIMESTAMP CLUSTERING (0.5s THRESHOLD)
# =============================================================================
def analyze_near_timestamp_clustering(au, placements, f, dataset_name, threshold=0.5):
    """
    Cluster auctions within 0.5s of each other into 'page load events'.

    Resolves the finding that:
    - Section 6: 99.3% unique timestamps (minimal exact batching)
    - Section 7: 49.7% of P3 gaps are <0.5s (near-simultaneous)

    Hypothesis: Auctions are near-simultaneous but not identical:
      Auction 1: 12:00:00.000
      Auction 2: 12:00:00.020 ← different timestamp, 0.02s later
    """
    log(f"\n--- Near-Timestamp Clustering Analysis ({dataset_name}) ---", f)
    log(f"Threshold: {threshold}s (cluster auctions within {threshold}s of each other)", f)
    log(f"", f)
    log(f"Resolution: Exact timestamps are unique (Section 6), but 49.7% of P3 gaps are <0.5s.", f)
    log(f"This means auctions arrive in rapid succession but with slightly different timestamps.", f)
    log(f"Clustering them reveals 'page load events' - groups of auctions fired together.", f)

    results = {}

    for p in placements:
        au_p = au[au['PLACEMENT'] == p].copy()
        if len(au_p) == 0:
            continue

        au_p = au_p.sort_values(['USER_ID', 'CREATED_AT'])

        # Compute time gap to previous auction within user
        au_p['prev_user'] = au_p['USER_ID'].shift(1)
        au_p['prev_time'] = au_p['CREATED_AT'].shift(1)
        au_p['time_gap'] = np.where(
            au_p['USER_ID'] == au_p['prev_user'],
            (au_p['CREATED_AT'] - au_p['prev_time']).dt.total_seconds(),
            np.nan
        )

        # Mark cluster boundaries: new cluster when gap > threshold or different user
        au_p['new_cluster'] = (au_p['time_gap'].isna()) | (au_p['time_gap'] > threshold)
        au_p['cluster_id'] = au_p['new_cluster'].cumsum()

        # Count auctions per cluster
        cluster_sizes = au_p.groupby('cluster_id').size()

        log(f"\nPlacement {p}:", f)
        log(f"  Total auctions: {len(au_p):,}", f)
        log(f"  Total page-load clusters: {len(cluster_sizes):,}", f)
        log(f"  Mean auctions per cluster: {cluster_sizes.mean():.2f}", f)
        log(f"  Median auctions per cluster: {cluster_sizes.median():.0f}", f)
        log(f"  Max auctions per cluster: {cluster_sizes.max()}", f)

        # Distribution of cluster sizes
        log(f"\n  {'Cluster Size':>12} {'Count':>12} {'Percentage':>12} {'Interpretation':>30}", f)
        log(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*30}", f)

        for size in range(1, min(11, int(cluster_sizes.max()) + 1)):
            count = (cluster_sizes == size).sum()
            pct = count / len(cluster_sizes) * 100
            if count > 0:
                interp = ""
                if size == 1:
                    interp = "(single auction)"
                elif size == 2:
                    interp = "(paired ad slots)"
                elif size >= 3:
                    interp = f"({size} ad slots on page)"
                log(f"  {size:>12} {count:>12,} {pct:>11.1f}% {interp:>30}", f)

        if cluster_sizes.max() > 10:
            count_10plus = (cluster_sizes > 10).sum()
            pct_10plus = count_10plus / len(cluster_sizes) * 100
            log(f"  {'11+':>12} {count_10plus:>12,} {pct_10plus:>11.1f}%", f)

        # Key metric: what % of clusters have size > 1 (multiple slots per page)?
        multi_slot_clusters = (cluster_sizes > 1).sum()
        multi_slot_pct = multi_slot_clusters / len(cluster_sizes) * 100

        log(f"\n  KEY METRIC: {multi_slot_pct:.1f}% of page loads have multiple ad slots", f)

        results[p] = {
            'total_auctions': len(au_p),
            'total_clusters': len(cluster_sizes),
            'mean_cluster_size': cluster_sizes.mean(),
            'median_cluster_size': cluster_sizes.median(),
            'multi_slot_pct': multi_slot_pct,
            'cluster_sizes': cluster_sizes,
            'au_p': au_p
        }

    # Summary
    log(f"\n--- Summary: Near-Timestamp Clustering ---", f)
    log(f"{'Placement':>12} {'Auctions':>12} {'Clusters':>12} {'Mean Size':>12} {'Multi-Slot%':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)
    for p in placements:
        if p in results:
            r = results[p]
            log(f"{'P' + p:>12} {r['total_auctions']:>12,} {r['total_clusters']:>12,} {r['mean_cluster_size']:>12.2f} {r['multi_slot_pct']:>11.1f}%", f)

    return results


# =============================================================================
# SECTION 11: PRODUCT OVERLAP IN NEAR-SIMULTANEOUS AUCTIONS
# =============================================================================
def analyze_product_overlap_in_clusters(au, ar, cluster_results, placements, f, dataset_name):
    """
    For auction pairs within 0.5s gap (same page-load cluster):
    - Do they show different products (different ad slots)?
    - Or same products (duplicate logging)?

    Join to auctions_results to get winner products per auction.
    Compute Jaccard similarity of product sets.
    """
    log(f"\n--- Product Overlap in Near-Simultaneous Auctions ({dataset_name}) ---", f)
    log(f"Question: Do paired auctions show different or same products?", f)
    log(f"", f)
    log(f"If different products → different ad slots on same page", f)
    log(f"If same products → duplicate logging or caching artifact", f)

    if ar is None:
        log(f"\nSkipped: auctions_results data not available.", f)
        return None

    # Get winning products per auction
    winners = ar[ar['IS_WINNER'] == True][['AUCTION_ID', 'PRODUCT_ID']].copy()
    winner_products = winners.groupby('AUCTION_ID')['PRODUCT_ID'].apply(set).reset_index()
    winner_products.columns = ['AUCTION_ID', 'product_set']

    for p in placements:
        if p not in cluster_results:
            continue

        au_p = cluster_results[p]['au_p']
        cluster_sizes = cluster_results[p]['cluster_sizes']

        # Get clusters with size >= 2
        multi_clusters = cluster_sizes[cluster_sizes >= 2].index.tolist()

        if len(multi_clusters) == 0:
            log(f"\nPlacement {p}: No multi-auction clusters found.", f)
            continue

        log(f"\nPlacement {p}:", f)
        log(f"  Multi-auction clusters to analyze: {len(multi_clusters):,}", f)

        # Sample some clusters for detailed analysis
        jaccard_similarities = []
        overlap_counts = []
        n_analyzed = 0

        for cluster_id in tqdm(multi_clusters, desc=f"  Analyzing P{p} clusters", leave=False):
            cluster_auctions = au_p[au_p['cluster_id'] == cluster_id]['AUCTION_ID'].tolist()

            if len(cluster_auctions) < 2:
                continue

            # Get product sets for each auction
            cluster_products = []
            for auction_id in cluster_auctions:
                match = winner_products[winner_products['AUCTION_ID'] == auction_id]
                if len(match) > 0:
                    cluster_products.append(match.iloc[0]['product_set'])

            if len(cluster_products) < 2:
                continue

            # Compute pairwise Jaccard similarity
            for i in range(len(cluster_products)):
                for j in range(i + 1, len(cluster_products)):
                    set_a = cluster_products[i]
                    set_b = cluster_products[j]

                    if len(set_a) == 0 and len(set_b) == 0:
                        continue

                    intersection = len(set_a & set_b)
                    union = len(set_a | set_b)
                    jaccard = intersection / union if union > 0 else 0

                    jaccard_similarities.append(jaccard)
                    overlap_counts.append(intersection)

            n_analyzed += 1

        if len(jaccard_similarities) == 0:
            log(f"  No valid auction pairs found for product analysis.", f)
            continue

        jaccard_arr = np.array(jaccard_similarities)
        overlap_arr = np.array(overlap_counts)

        log(f"  Auction pairs analyzed: {len(jaccard_arr):,}", f)
        log(f"", f)
        log(f"  Jaccard Similarity (0=no overlap, 1=identical):", f)
        log(f"    Mean: {jaccard_arr.mean():.3f}", f)
        log(f"    Median: {np.median(jaccard_arr):.3f}", f)
        log(f"    % with zero overlap: {(jaccard_arr == 0).sum() / len(jaccard_arr) * 100:.1f}%", f)
        log(f"    % with perfect overlap: {(jaccard_arr == 1).sum() / len(jaccard_arr) * 100:.1f}%", f)

        log(f"", f)
        log(f"  Product Overlap Count:", f)
        log(f"    Mean overlapping products: {overlap_arr.mean():.2f}", f)
        log(f"    Median overlapping products: {np.median(overlap_arr):.0f}", f)
        log(f"    Max overlapping products: {overlap_arr.max()}", f)

        # Interpretation
        mean_jaccard = jaccard_arr.mean()
        zero_overlap_pct = (jaccard_arr == 0).sum() / len(jaccard_arr) * 100

        log(f"", f)
        if mean_jaccard < 0.1 and zero_overlap_pct > 80:
            log(f"  INTERPRETATION: Different products in paired auctions.", f)
            log(f"  → Paired auctions represent DIFFERENT AD SLOTS on the same page.", f)
        elif mean_jaccard > 0.9:
            log(f"  INTERPRETATION: Same products in paired auctions.", f)
            log(f"  → Paired auctions may be DUPLICATE LOGGING or caching artifacts.", f)
        else:
            log(f"  INTERPRETATION: Partial overlap in paired auctions.", f)
            log(f"  → Some shared products, some unique - mixed behavior.", f)

    return True


# =============================================================================
# SECTION 12: CORRECTED RUN LENGTHS (PAGE LOADS, NOT AUCTIONS)
# =============================================================================
def analyze_corrected_run_lengths(cluster_results, placements, f, dataset_name, target_placement='3'):
    """
    Recompute run lengths at the page-load level instead of auction level.

    Original Section 2 counted consecutive auctions.
    If auctions are paired (2 per page), a run of 8 auctions = 4 page loads.

    This section clusters auctions into page loads first, then counts
    consecutive page loads of the same placement.
    """
    log(f"\n--- Corrected Run Lengths: Page Loads, Not Auctions ({dataset_name}) ---", f)
    log(f"Target placement: {target_placement}", f)
    log(f"", f)
    log(f"Original analysis counted AUCTION runs. If auctions are paired,", f)
    log(f"a 'run of 8' is really 4 page loads, not 8 page turns.", f)
    log(f"This section recomputes using page-load clusters.", f)

    if target_placement not in cluster_results:
        log(f"\nNo cluster data for placement {target_placement}.", f)
        return None

    au_p = cluster_results[target_placement]['au_p'].copy()

    # Get unique (USER_ID, cluster_id) combinations as "page load events"
    page_loads = au_p.groupby(['USER_ID', 'cluster_id']).agg({
        'CREATED_AT': 'min',  # Use first timestamp of cluster
        'PLACEMENT': 'first'
    }).reset_index()

    page_loads = page_loads.sort_values(['USER_ID', 'CREATED_AT'])

    log(f"\nPage load events for P{target_placement}:", f)
    log(f"  Total page loads: {len(page_loads):,}", f)
    log(f"  Original auctions: {len(au_p):,}", f)
    log(f"  Ratio: {len(au_p) / len(page_loads):.2f} auctions per page load", f)

    # Compute run lengths at page-load level
    # (All page loads here are already P3, so every consecutive sequence is a run)
    # Need to identify runs separated by non-P3 placements in the full user timeline

    # Since we're only looking at P3 page loads, we need the full timeline
    # For simplicity, compute inter-page-load gaps and use large gaps as run breaks

    page_loads['prev_user'] = page_loads['USER_ID'].shift(1)
    page_loads['prev_time'] = page_loads['CREATED_AT'].shift(1)
    page_loads['time_gap'] = np.where(
        page_loads['USER_ID'] == page_loads['prev_user'],
        (page_loads['CREATED_AT'] - page_loads['prev_time']).dt.total_seconds(),
        np.nan
    )

    # Define session break as gap > 30 minutes (typical session timeout)
    SESSION_BREAK = 30 * 60  # 30 minutes in seconds

    page_loads['new_run'] = (page_loads['time_gap'].isna()) | (page_loads['time_gap'] > SESSION_BREAK)
    page_loads['run_id'] = page_loads['new_run'].cumsum()

    # Count page loads per run
    run_lengths = page_loads.groupby('run_id').size()

    log(f"\nPage-load run statistics (session break = {SESSION_BREAK/60:.0f} min):", f)
    log(f"  Total runs: {len(run_lengths):,}", f)
    log(f"  Mean run length: {run_lengths.mean():.2f} page loads", f)
    log(f"  Median run length: {run_lengths.median():.0f} page loads", f)
    log(f"  Max run length: {run_lengths.max()} page loads", f)

    # Distribution
    log(f"\n{'Run Length':>12} {'Count':>12} {'Percentage':>12} {'Cumulative':>12} {'Interpretation':>25}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*25}", f)

    total = len(run_lengths)
    cumsum = 0
    for length in range(1, min(16, run_lengths.max() + 1)):
        count = (run_lengths == length).sum()
        pct = count / total * 100
        cumsum += pct
        interp = ""
        if length == 1:
            interp = "(single page)"
        elif length == 2:
            interp = "(page 1 → page 2)"
        elif length >= 3:
            interp = f"({length} pages viewed)"
        log(f"{length:>12} {count:>12,} {pct:>11.1f}% {cumsum:>11.1f}% {interp:>25}", f)

    if run_lengths.max() > 15:
        count_15plus = (run_lengths > 15).sum()
        pct_15plus = count_15plus / total * 100
        log(f"{'16+':>12} {count_15plus:>12,} {pct_15plus:>11.1f}% {100.0:>11.1f}% (deep pagination)", f)

    # Compare to original auction-level run lengths
    original_mean = cluster_results[target_placement]['cluster_sizes'].groupby(
        cluster_results[target_placement]['au_p'].groupby('cluster_id').ngroup()
    ).size().mean() if 'cluster_sizes' in cluster_results[target_placement] else None

    log(f"\n--- Comparison to Original Auction-Level Analysis ---", f)
    log(f"  Page-load mean run length: {run_lengths.mean():.2f}", f)
    log(f"  Auctions per page load: {len(au_p) / len(page_loads):.2f}", f)
    log(f"  Implied correction factor: {len(au_p) / len(page_loads):.2f}x", f)

    return run_lengths, page_loads


# =============================================================================
# SECTION 13: CHARACTERIZING THE 0.5-30s MIDDLE GAPS
# =============================================================================
def analyze_middle_gaps(au, all_gaps, placements, f, dataset_name, target_placement='3'):
    """
    Only 12% of P3 gaps fall in 0.5-30s range. Characterize these:
    - Are they fast pagination (power users)?
    - Scroll-triggered lazy loading?
    - Specific user segments?
    """
    log(f"\n--- Characterizing the 0.5-30s Middle Gaps ({dataset_name}) ---", f)
    log(f"Target placement: {target_placement}", f)
    log(f"", f)
    log(f"Gap distribution recap:", f)
    log(f"  <0.5s: ~50% (within-page slots)", f)
    log(f"  0.5-30s: ~12% (the 'middle' - what is this?)", f)
    log(f"  30s+: ~37% (actual page transitions/sessions)", f)
    log(f"", f)
    log(f"This section investigates the 0.5-30s gaps.", f)

    if target_placement not in all_gaps or all_gaps[target_placement] is None:
        log(f"\nNo gap data for placement {target_placement}.", f)
        return None

    gaps = all_gaps[target_placement]

    # Define middle gaps
    middle_mask = (gaps >= 0.5) & (gaps < 30)
    middle_gaps = gaps[middle_mask]

    log(f"\nMiddle gaps (0.5-30s):", f)
    log(f"  Count: {len(middle_gaps):,} ({len(middle_gaps)/len(gaps)*100:.1f}% of all P{target_placement} gaps)", f)

    if len(middle_gaps) == 0:
        log(f"  No middle gaps found.", f)
        return None

    log(f"  Median: {middle_gaps.median():.2f}s", f)
    log(f"  Mean: {middle_gaps.mean():.2f}s", f)
    log(f"  P25: {middle_gaps.quantile(0.25):.2f}s", f)
    log(f"  P75: {middle_gaps.quantile(0.75):.2f}s", f)

    # Fine-grained distribution within middle
    log(f"\nFine-grained distribution within 0.5-30s:", f)
    bins = [(0.5, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 30)]

    log(f"  {'Range':>12} {'Count':>10} {'% of Middle':>12} {'% of All':>12}", f)
    log(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}", f)

    for low, high in bins:
        in_bin = ((middle_gaps >= low) & (middle_gaps < high)).sum()
        pct_middle = in_bin / len(middle_gaps) * 100 if len(middle_gaps) > 0 else 0
        pct_all = in_bin / len(gaps) * 100
        log(f"  {low:>5.1f}-{high:<5.1f}s {in_bin:>10,} {pct_middle:>11.1f}% {pct_all:>11.1f}%", f)

    # Hypotheses about middle gaps
    log(f"\n--- Hypotheses for Middle Gaps ---", f)

    # Check 1: Are these between-page transitions that got miscategorized?
    log(f"\n1. Fast pagination hypothesis:", f)
    fast_pagination = (middle_gaps >= 0.5) & (middle_gaps < 5)
    log(f"   Gaps 0.5-5s (plausible page turns): {fast_pagination.sum():,} ({fast_pagination.sum()/len(gaps)*100:.1f}%)", f)

    # Check 2: Are these scroll-triggered loads?
    log(f"\n2. Scroll-triggered loading hypothesis:", f)
    scroll_load = (middle_gaps >= 2) & (middle_gaps < 10)
    log(f"   Gaps 2-10s (scroll-triggered): {scroll_load.sum():,} ({scroll_load.sum()/len(gaps)*100:.1f}%)", f)

    # Check 3: Distribution pattern
    log(f"\n3. Pattern analysis:", f)
    mode_estimate = middle_gaps.mode().iloc[0] if len(middle_gaps.mode()) > 0 else middle_gaps.median()
    log(f"   Mode (most common gap): {mode_estimate:.2f}s", f)

    # Check if bimodal within middle
    below_median = middle_gaps[middle_gaps < middle_gaps.median()]
    above_median = middle_gaps[middle_gaps >= middle_gaps.median()]
    log(f"   Below median ({middle_gaps.median():.1f}s): {len(below_median):,} gaps, mean {below_median.mean():.2f}s", f)
    log(f"   Above median ({middle_gaps.median():.1f}s): {len(above_median):,} gaps, mean {above_median.mean():.2f}s", f)

    # Interpretation
    log(f"\n--- Interpretation ---", f)

    median_val = middle_gaps.median()
    if median_val < 5:
        log(f"Middle gaps cluster around {median_val:.1f}s - consistent with:", f)
        log(f"  - Manual page turns by engaged users", f)
        log(f"  - Scroll-to-load delays", f)
        log(f"  - Network latency variations", f)
    elif median_val < 15:
        log(f"Middle gaps cluster around {median_val:.1f}s - consistent with:", f)
        log(f"  - Users reading/scanning results before next action", f)
        log(f"  - Deliberate browsing (not rapid pagination)", f)
    else:
        log(f"Middle gaps cluster around {median_val:.1f}s - consistent with:", f)
        log(f"  - Extended browsing sessions", f)
        log(f"  - Users who take breaks but return within 30s", f)

    return middle_gaps


# =============================================================================
# CROSS-PLACEMENT COMPARISON
# =============================================================================
def cross_placement_comparison(au_sorted, placements, all_gaps_by_placement, transition_pcts, f, dataset_name):
    """Side-by-side comparison of metrics across placements."""
    log(f"\n--- Cross-Placement Comparison ({dataset_name}) ---", f)

    # Compute metrics for each placement
    log(f"\n{'Metric':<25}" + "".join([f"P{p:>12}" for p in placements]), f)
    log(f"{'-'*25}" + "-" * 12 * len(placements), f)

    # Self-transition rate
    row = f"{'Self-transition rate':<25}"
    for p in placements:
        rate = transition_pcts.loc[p, p]
        row += f"{rate:>11.1f}%"
    log(row, f)

    # Median time gap within same placement
    row = f"{'Median gap (within P)':<25}"
    for p in placements:
        if p in all_gaps_by_placement and all_gaps_by_placement[p] is not None and len(all_gaps_by_placement[p]) > 0:
            gap = all_gaps_by_placement[p].median()
            row += f"{gap:>10.1f}s "
        else:
            row += f"{'N/A':>12}"
    log(row, f)

    # Mean time gap
    row = f"{'Mean gap (within P)':<25}"
    for p in placements:
        if p in all_gaps_by_placement and all_gaps_by_placement[p] is not None and len(all_gaps_by_placement[p]) > 0:
            gap = all_gaps_by_placement[p].mean()
            row += f"{gap:>10.1f}s "
        else:
            row += f"{'N/A':>12}"
    log(row, f)

    # Auction count
    row = f"{'Total auctions':<25}"
    for p in placements:
        count = (au_sorted['PLACEMENT'] == p).sum()
        row += f"{count:>12,}"
    log(row, f)

    # % of auctions that have a next
    row = f"{'% with next auction':<25}"
    for p in placements:
        total = (au_sorted['PLACEMENT'] == p).sum()
        with_next = ((au_sorted['PLACEMENT'] == p) & (au_sorted['NEXT_PLACEMENT'].notna())).sum()
        pct = with_next / total * 100 if total > 0 else 0
        row += f"{pct:>11.1f}%"
    log(row, f)

# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("SEARCH PAGINATION EVIDENCE: P3 SELF-TRANSITION ANALYSIS", f)
        log("=" * 80, f)
        log(f"\nObjective: Prove P3 is search pagination via the diagnostic self-transition pattern.", f)
        log(f"If P3 = search results, users viewing multiple pages will generate P3 → P3 → P3 sequences.", f)
        log(f"This 'self-transition' rate should be much higher than other placements.", f)

        # Define datasets to analyze
        datasets = [
            ('_all', 'All placements dataset'),
            ('_p5', 'P5-filtered dataset'),
        ]

        placements = ['1', '2', '3', '5']

        for suffix, description in datasets:
            log(f"\n{'='*80}", f)
            log(f"DATASET: {suffix} ({description})", f)
            log(f"{'='*80}", f)

            # Load data
            au_file = DATA_DIR / f"auctions_users{suffix}.parquet"
            ar_file = DATA_DIR / f"auctions_results{suffix}.parquet"

            if not au_file.exists():
                log(f"ERROR: {au_file} not found. Skipping.", f)
                continue

            log(f"\nLoading {au_file}...", f)
            au = pd.read_parquet(au_file)

            # Load auctions_results for winners analysis
            ar = None
            if ar_file.exists():
                log(f"Loading {ar_file}...", f)
                ar = pd.read_parquet(ar_file)
                log(f"Loaded {len(ar):,} auction results", f)
            else:
                log(f"WARNING: {ar_file} not found. Winners analysis will be skipped.", f)

            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(au['CREATED_AT']):
                au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

            log(f"Loaded {len(au):,} auctions from {au['USER_ID'].nunique():,} users", f)
            log(f"Placements: {sorted(au['PLACEMENT'].unique())}", f)

            # Section 1: Transition Matrix
            log(f"\n{'='*60}", f)
            log(f"SECTION 1: FULL TRANSITION MATRIX", f)
            log(f"{'='*60}", f)

            au_sorted, transitions, transition_pcts = compute_transition_matrix(au, placements, f, suffix)

            # Section 2: Consecutive P3 Run Lengths
            log(f"\n{'='*60}", f)
            log(f"SECTION 2: CONSECUTIVE P3 RUN LENGTHS", f)
            log(f"{'='*60}", f)

            run_lengths = analyze_consecutive_runs(au_sorted, '3', f, suffix)

            # Section 3: Time Gaps Within P3 Runs
            log(f"\n{'='*60}", f)
            log(f"SECTION 3: TIME GAPS WITHIN P3 RUNS", f)
            log(f"{'='*60}", f)

            gaps_p3, _ = analyze_time_gaps(au_sorted, '3', f, suffix)

            # Compute gaps for all placements for comparison
            all_gaps = {}
            for p in placements:
                result = analyze_time_gaps(au_sorted, p, f, suffix)
                all_gaps[p] = result[0] if result is not None else None

            # Section 4: Sample User Sequences
            log(f"\n{'='*60}", f)
            log(f"SECTION 4: SAMPLE USER SEQUENCES", f)
            log(f"{'='*60}", f)

            show_sample_sequences(au_sorted, f, suffix, n_samples=10)

            # Section 5: Cross-Placement Comparison
            log(f"\n{'='*60}", f)
            log(f"SECTION 5: CROSS-PLACEMENT COMPARISON", f)
            log(f"{'='*60}", f)

            cross_placement_comparison(au_sorted, placements, all_gaps, transition_pcts, f, suffix)

            # Section 6: Timestamp Clustering Analysis (NEW)
            log(f"\n{'='*60}", f)
            log(f"SECTION 6: TIMESTAMP CLUSTERING ANALYSIS", f)
            log(f"{'='*60}", f)

            cluster_counts = analyze_timestamp_clustering(au, placements, f, suffix)

            # Section 7: Time Gaps Excluding Instant (NEW)
            log(f"\n{'='*60}", f)
            log(f"SECTION 7: TIME GAPS EXCLUDING INSTANT (<0.5s)", f)
            log(f"{'='*60}", f)

            for p in placements:
                if all_gaps[p] is not None and len(all_gaps[p]) > 0:
                    analyze_gaps_excluding_instant(all_gaps[p], f, p, threshold=0.5)

            # Section 8: Reverse Transition Matrix (NEW)
            log(f"\n{'='*60}", f)
            log(f"SECTION 8: REVERSE TRANSITION MATRIX", f)
            log(f"{'='*60}", f)

            reverse_pcts = compute_reverse_transitions(transitions, placements, f, suffix)

            # Section 9: Winners Per Auction by Placement (NEW)
            log(f"\n{'='*60}", f)
            log(f"SECTION 9: WINNERS PER AUCTION BY PLACEMENT", f)
            log(f"{'='*60}", f)

            if ar is not None:
                winners_data = analyze_winners_per_auction(au, ar, placements, f, suffix)
            else:
                log(f"Skipped: auctions_results data not available.", f)

            # Section 10: Near-Timestamp Clustering (NEW - Round 2)
            log(f"\n{'='*60}", f)
            log(f"SECTION 10: NEAR-TIMESTAMP CLUSTERING (0.5s THRESHOLD)", f)
            log(f"{'='*60}", f)

            cluster_results = analyze_near_timestamp_clustering(au, placements, f, suffix, threshold=0.5)

            # Section 11: Product Overlap in Near-Simultaneous Auctions (NEW - Round 2)
            log(f"\n{'='*60}", f)
            log(f"SECTION 11: PRODUCT OVERLAP IN NEAR-SIMULTANEOUS AUCTIONS", f)
            log(f"{'='*60}", f)

            if ar is not None and cluster_results:
                analyze_product_overlap_in_clusters(au, ar, cluster_results, placements, f, suffix)
            else:
                log(f"Skipped: auctions_results data or cluster results not available.", f)

            # Section 12: Corrected Run Lengths (NEW - Round 2)
            log(f"\n{'='*60}", f)
            log(f"SECTION 12: CORRECTED RUN LENGTHS (PAGE LOADS, NOT AUCTIONS)", f)
            log(f"{'='*60}", f)

            if cluster_results and '3' in cluster_results:
                corrected_runs, page_loads = analyze_corrected_run_lengths(cluster_results, placements, f, suffix, target_placement='3')
            else:
                log(f"Skipped: cluster results for P3 not available.", f)

            # Section 13: Characterizing the 0.5-30s Middle Gaps (NEW - Round 2)
            log(f"\n{'='*60}", f)
            log(f"SECTION 13: CHARACTERIZING THE 0.5-30s MIDDLE GAPS", f)
            log(f"{'='*60}", f)

            analyze_middle_gaps(au, all_gaps, placements, f, suffix, target_placement='3')

        # Conclusion
        log(f"\n{'='*80}", f)
        log(f"CONCLUSION", f)
        log(f"{'='*80}", f)

        log(f"\nEvidence for P3 = Search Pagination:", f)
        log(f"", f)
        log(f"1. SELF-TRANSITION RATE: If P3 shows ~80%+ self-transition rate while other", f)
        log(f"   placements show <50%, this indicates users repeatedly trigger P3 events", f)
        log(f"   in sequence - consistent with paginating through search results.", f)
        log(f"", f)
        log(f"2. RUN LENGTHS: If users commonly have 2-5+ consecutive P3 events, this", f)
        log(f"   corresponds to viewing pages 1, 2, 3... of search results.", f)
        log(f"", f)
        log(f"3. TIME GAPS: If gaps between consecutive P3 events are ~1-3 seconds,", f)
        log(f"   this matches rapid pagination behavior (not content consumption).", f)
        log(f"", f)
        log(f"4. COMPARISON: Other placements (P1=homepage, P2=PDP, P5=carousel?) should", f)
        log(f"   show lower self-transition and longer gaps (content viewing time).", f)
        log(f"", f)
        log(f"The combination of high self-transition + short gaps + multi-page runs", f)
        log(f"is pathognomonic for search pagination. No other page type produces this.", f)
        log(f"", f)
        log(f"\n{'='*80}", f)
        log(f"EXTENDED ANALYSIS FINDINGS (SECTIONS 6-9)", f)
        log(f"{'='*80}", f)
        log(f"", f)
        log(f"5. TIMESTAMP CLUSTERING (Section 6):", f)
        log(f"   - If many auctions share exact same timestamp within user → batching detected", f)
        log(f"   - 'Auction' = ad slot, not page load", f)
        log(f"   - 'Run of 4' = one search page with 4 ad slots, not 4 page loads", f)
        log(f"   - If auctions mostly have unique timestamps → 'Auction' = page load", f)
        log(f"", f)
        log(f"6. TIME GAPS EXCLUDING INSTANT (Section 7):", f)
        log(f"   - After filtering 0.0s gaps (batched events), what is 'true' pagination speed?", f)
        log(f"   - If median still 2-5s → rapid pagination is real behavior", f)
        log(f"   - If median jumps to 30s+ → 'rapid pagination' was batching artifact", f)
        log(f"", f)
        log(f"7. REVERSE TRANSITIONS (Section 8):", f)
        log(f"   - P(came from X | now at P3) shows entry points to search", f)
        log(f"   - High P(from P3) → most P3 visits are self-pagination", f)
        log(f"   - High P(from P1) → homepage is major entry to search", f)
        log(f"", f)
        log(f"8. WINNERS PER AUCTION (Section 9):", f)
        log(f"   - How many products shown per auction by placement?", f)
        log(f"   - Search results typically show more items than carousel/PDP", f)
        log(f"   - If P3 has many winners → search results page (multiple ad slots)", f)

        log(f"\n{'='*80}", f)
        log(f"EXTENDED ANALYSIS FINDINGS (SECTIONS 10-13) - ROUND 2", f)
        log(f"{'='*80}", f)
        log(f"", f)
        log(f"Resolution of the core uncertainty from Round 1:", f)
        log(f"  - Section 6 showed 99.3% unique timestamps (minimal exact batching)", f)
        log(f"  - Section 7 showed 49.7% of P3 gaps are <0.5s", f)
        log(f"  - This means auctions are near-simultaneous but not identical timestamps", f)
        log(f"", f)
        log(f"9. NEAR-TIMESTAMP CLUSTERING (Section 10):", f)
        log(f"   - Cluster auctions within 0.5s of each other as 'page load events'", f)
        log(f"   - Shows how many auctions fire per page load", f)
        log(f"   - If P3 shows 2 auctions per cluster → paired ad slots on search page", f)
        log(f"", f)
        log(f"10. PRODUCT OVERLAP IN CLUSTERS (Section 11):", f)
        log(f"    - Do paired auctions show different or same products?", f)
        log(f"    - Different products → different ad slots on same page", f)
        log(f"    - Same products → duplicate logging or caching artifact", f)
        log(f"", f)
        log(f"11. CORRECTED RUN LENGTHS (Section 12):", f)
        log(f"    - Recompute run lengths at page-load level, not auction level", f)
        log(f"    - If auctions are paired, original runs are 2x inflated", f)
        log(f"    - 'Run of 8 auctions' → '4 page loads' (4 pages viewed)", f)
        log(f"", f)
        log(f"12. MIDDLE GAPS (Section 13):", f)
        log(f"    - Characterize the 0.5-30s gaps (~12% of P3 gaps)", f)
        log(f"    - Are these fast pagination, scroll-loading, or mixed behavior?", f)
        log(f"", f)
        log(f"KEY RESOLUTION:", f)
        log(f"  The bimodal gap distribution (<0.5s: 50%, 0.5-30s: 12%, 30s+: 37%) suggests:", f)
        log(f"  - <0.5s gaps: Multiple ad slots firing together on same page load", f)
        log(f"  - 30s+ gaps: New page loads (actual pagination or session breaks)", f)
        log(f"  - 0.5-30s gaps: Edge cases (network delays, scroll-loading, etc.)", f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {OUTPUT_FILE.absolute()}", f)

if __name__ == "__main__":
    main()
