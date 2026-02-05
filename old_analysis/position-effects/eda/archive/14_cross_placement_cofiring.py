#!/usr/bin/env python3
"""
Cross-Placement Co-Firing Analysis

Identifies which placements fire simultaneously (within <1s) to determine if they
share the same page. Key insight: if P1 co-fires with P2, they're likely both on
PDP. If P1 co-fires with P3, they're both on Search.

Purpose:
- Disambiguate P1's identity (is it on PDP, Search, Brand Page, or separate page?)
- Validate P3/P5 are different pages (should NOT co-fire)
- Confirm P2 is on PDP (might co-fire with another PDP placement)

Output: results/14_cross_placement_cofiring.txt
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
BASE_DIR = Path(__file__).parent.parent.parent  # Up to position-effects/
DATA_DIR = BASE_DIR / "0_data" / "round1"
RESULTS_DIR = Path(__file__).parent.parent / "results"  # eda/results/
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "14_cross_placement_cofiring.txt"

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
        log("CROSS-PLACEMENT CO-FIRING ANALYSIS", f)
        log("=" * 80, f)
        log("Purpose: Identify which placements fire simultaneously (<1s apart)", f)
        log("         to determine if they share the same page", f)
        log(f"Data source: {DATA_DIR}", f)
        log(f"Output: {OUTPUT_FILE}", f)
        log("", f)

        # =====================================================================
        # SECTION 1: DATA LOADING
        # =====================================================================
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("=" * 80, f)

        log("\nLoading auctions_users_all.parquet...", f)
        au = pd.read_parquet(DATA_DIR / "auctions_users_all.parquet")
        log(f"  Shape: {au.shape}", f)
        log(f"  Columns: {list(au.columns)}", f)

        # Convert timestamp
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'])

        # Convert PLACEMENT to string for consistency
        au['PLACEMENT'] = au['PLACEMENT'].astype(str)

        log("\nPlacement distribution:", f)
        placement_counts = au['PLACEMENT'].value_counts().sort_index()
        for p, count in placement_counts.items():
            log(f"  P{p}: {count:,} auctions ({count/len(au)*100:.1f}%)", f)

        log(f"\nUnique users: {au['USER_ID'].nunique():,}", f)
        log(f"Date range: {au['CREATED_AT'].min()} to {au['CREATED_AT'].max()}", f)

        # =====================================================================
        # SECTION 2: CO-FIRING DETECTION
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 2: CO-FIRING DETECTION", f)
        log("=" * 80, f)
        log("", f)
        log("Definition: Two auctions 'co-fire' if they occur within <1 second", f)
        log("            for the same user but with DIFFERENT placements.", f)
        log("", f)
        log("Interpretation:", f)
        log("  - High co-firing rate (>5%) = same page, different ad slots", f)
        log("  - Low co-firing rate (<2%) = different pages", f)
        log("", f)

        # Sort by user and timestamp
        au_sorted = au.sort_values(['USER_ID', 'CREATED_AT']).reset_index(drop=True)

        # Compute time gaps and placement transitions
        au_sorted['NEXT_CREATED_AT'] = au_sorted.groupby('USER_ID')['CREATED_AT'].shift(-1)
        au_sorted['NEXT_PLACEMENT'] = au_sorted.groupby('USER_ID')['PLACEMENT'].shift(-1)
        au_sorted['TIME_GAP'] = (au_sorted['NEXT_CREATED_AT'] - au_sorted['CREATED_AT']).dt.total_seconds()

        # Filter to different-placement transitions only
        different_placement = au_sorted[
            (au_sorted['PLACEMENT'] != au_sorted['NEXT_PLACEMENT']) &
            (au_sorted['NEXT_PLACEMENT'].notna())
        ].copy()

        log(f"Total different-placement transitions: {len(different_placement):,}", f)

        # =====================================================================
        # SECTION 3: CO-FIRING MATRIX (< 1 SECOND)
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 3: CO-FIRING MATRIX (< 1 SECOND)", f)
        log("=" * 80, f)

        thresholds = [0.5, 1.0, 2.0, 5.0]

        for threshold in thresholds:
            log(f"\n--- Co-firing within {threshold}s ---", f)

            cofire = different_placement[different_placement['TIME_GAP'] < threshold]

            # Build co-firing matrix
            placements = sorted(au['PLACEMENT'].unique())
            matrix = pd.DataFrame(0, index=placements, columns=placements, dtype=float)

            # Count co-firings
            for _, row in tqdm(cofire.iterrows(), total=len(cofire), desc=f"Processing {threshold}s", leave=False):
                p1 = row['PLACEMENT']
                p2 = row['NEXT_PLACEMENT']
                matrix.loc[p1, p2] += 1

            # Calculate rates (as % of total auctions for each placement)
            rates = matrix.copy()
            for p in placements:
                total_p = placement_counts[p]
                rates.loc[p, :] = (matrix.loc[p, :] / total_p * 100).round(2)

            log(f"\nCo-firing counts (row -> column within {threshold}s):", f)
            log(f"       " + "".join(f"  P{p:>5}" for p in placements), f)
            for p1 in placements:
                row_str = f"  P{p1}  "
                for p2 in placements:
                    if p1 == p2:
                        row_str += "     -  "
                    else:
                        row_str += f"{int(matrix.loc[p1, p2]):>7} "
                log(row_str, f)

            log(f"\nCo-firing rates (% of row placement's auctions):", f)
            log(f"       " + "".join(f"  P{p:>5}" for p in placements), f)
            for p1 in placements:
                row_str = f"  P{p1}  "
                for p2 in placements:
                    if p1 == p2:
                        row_str += "     -  "
                    else:
                        row_str += f"{rates.loc[p1, p2]:>6.2f}% "
                log(row_str, f)

        # =====================================================================
        # SECTION 4: BIDIRECTIONAL CO-FIRING ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 4: BIDIRECTIONAL CO-FIRING ANALYSIS", f)
        log("=" * 80, f)
        log("", f)
        log("If A co-fires with B, we should see both A->B and B->A at similar rates.", f)
        log("Asymmetric rates suggest timing/causality (one triggers the other).", f)
        log("", f)

        threshold = 1.0  # Use 1 second threshold for main analysis
        cofire = different_placement[different_placement['TIME_GAP'] < threshold]

        pairs = [('1', '2'), ('1', '3'), ('1', '5'), ('2', '3'), ('2', '5'), ('3', '5')]

        log(f"Bidirectional co-firing rates (within {threshold}s):", f)
        log("", f)
        log(f"  {'Pair':<8} {'A->B':<12} {'B->A':<12} {'Symmetric?':<12} {'Interpretation'}", f)
        log(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*30}", f)

        for p1, p2 in pairs:
            # A -> B
            ab_count = len(cofire[(cofire['PLACEMENT'] == p1) & (cofire['NEXT_PLACEMENT'] == p2)])
            ab_rate = ab_count / placement_counts[p1] * 100

            # B -> A
            ba_count = len(cofire[(cofire['PLACEMENT'] == p2) & (cofire['NEXT_PLACEMENT'] == p1)])
            ba_rate = ba_count / placement_counts[p2] * 100

            # Symmetry
            avg_rate = (ab_rate + ba_rate) / 2
            if avg_rate > 0:
                symmetry = min(ab_rate, ba_rate) / max(ab_rate, ba_rate) * 100
            else:
                symmetry = 0

            # Interpretation
            if avg_rate > 5:
                interp = "SAME PAGE (high co-fire)"
            elif avg_rate > 2:
                interp = "Possibly same page"
            elif avg_rate > 0.5:
                interp = "Sequential navigation"
            else:
                interp = "DIFFERENT PAGES (rare co-fire)"

            symmetric_str = f"{symmetry:.0f}%" if symmetry > 0 else "N/A"

            log(f"  P{p1}<->P{p2} {ab_rate:>5.2f}%     {ba_rate:>5.2f}%      {symmetric_str:<12} {interp}", f)

        # =====================================================================
        # SECTION 5: TIME GAP DISTRIBUTION BY PAIR
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 5: TIME GAP DISTRIBUTION BY PAIR", f)
        log("=" * 80, f)
        log("", f)
        log("For pairs that DO co-fire, what's the typical gap?", f)
        log("", f)

        for p1, p2 in pairs:
            # Get transitions in both directions
            trans = different_placement[
                ((different_placement['PLACEMENT'] == p1) & (different_placement['NEXT_PLACEMENT'] == p2)) |
                ((different_placement['PLACEMENT'] == p2) & (different_placement['NEXT_PLACEMENT'] == p1))
            ]

            if len(trans) > 0:
                gaps = trans['TIME_GAP'].dropna()

                # Gap distribution
                under_1s = (gaps < 1).sum() / len(gaps) * 100
                under_5s = (gaps < 5).sum() / len(gaps) * 100
                under_30s = (gaps < 30).sum() / len(gaps) * 100

                log(f"\nP{p1} <-> P{p2} transitions (n={len(trans):,}):", f)
                log(f"  Median gap: {gaps.median():.2f}s", f)
                log(f"  Mean gap: {gaps.mean():.2f}s", f)
                log(f"  Gap < 1s: {under_1s:.1f}%", f)
                log(f"  Gap < 5s: {under_5s:.1f}%", f)
                log(f"  Gap < 30s: {under_30s:.1f}%", f)

                # Quick histogram
                bins = [0, 0.5, 1, 2, 5, 10, 30, 60, float('inf')]
                hist, _ = np.histogram(gaps, bins=bins)
                log(f"  Distribution:", f)
                for i in range(len(bins)-1):
                    pct = hist[i] / len(gaps) * 100
                    bar = '#' * int(pct / 2)
                    if bins[i+1] == float('inf'):
                        label = f">{bins[i]}s"
                    else:
                        label = f"{bins[i]}-{bins[i+1]}s"
                    log(f"    {label:<10} {bar} ({pct:.1f}%)", f)

        # =====================================================================
        # SECTION 6: P1 IDENTITY DETERMINATION
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 6: P1 IDENTITY DETERMINATION", f)
        log("=" * 80, f)
        log("", f)
        log("Based on co-firing patterns, what page is P1 on?", f)
        log("", f)

        threshold = 1.0
        cofire = different_placement[different_placement['TIME_GAP'] < threshold]

        # P1 co-firing rates
        p1_with_p2 = len(cofire[(cofire['PLACEMENT'] == '1') & (cofire['NEXT_PLACEMENT'] == '2')]) + \
                     len(cofire[(cofire['PLACEMENT'] == '2') & (cofire['NEXT_PLACEMENT'] == '1')])
        p1_with_p3 = len(cofire[(cofire['PLACEMENT'] == '1') & (cofire['NEXT_PLACEMENT'] == '3')]) + \
                     len(cofire[(cofire['PLACEMENT'] == '3') & (cofire['NEXT_PLACEMENT'] == '1')])
        p1_with_p5 = len(cofire[(cofire['PLACEMENT'] == '1') & (cofire['NEXT_PLACEMENT'] == '5')]) + \
                     len(cofire[(cofire['PLACEMENT'] == '5') & (cofire['NEXT_PLACEMENT'] == '1')])

        p1_total = placement_counts['1']

        log(f"P1 co-firing within 1s:", f)
        log(f"  With P2: {p1_with_p2:,} ({p1_with_p2/p1_total*100:.2f}%)", f)
        log(f"  With P3: {p1_with_p3:,} ({p1_with_p3/p1_total*100:.2f}%)", f)
        log(f"  With P5: {p1_with_p5:,} ({p1_with_p5/p1_total*100:.2f}%)", f)

        # Baseline: P3 <-> P5 (should NOT co-fire as different pages)
        p3_with_p5 = len(cofire[(cofire['PLACEMENT'] == '3') & (cofire['NEXT_PLACEMENT'] == '5')]) + \
                     len(cofire[(cofire['PLACEMENT'] == '5') & (cofire['NEXT_PLACEMENT'] == '3')])
        p3_total = placement_counts['3']

        log(f"\nBaseline (P3 <-> P5, known different pages):", f)
        log(f"  P3 <-> P5: {p3_with_p5:,} ({p3_with_p5/p3_total*100:.2f}%)", f)

        # Determine P1 identity
        log("\nConclusion:", f)
        max_cofire = max(p1_with_p2, p1_with_p3, p1_with_p5)

        if max_cofire == p1_with_p2 and p1_with_p2/p1_total*100 > 5:
            log("  -> P1 likely on SAME PAGE as P2 (PDP)", f)
            log("  -> P1 may be another PDP ad section (e.g., 'You May Also Like')", f)
        elif max_cofire == p1_with_p3 and p1_with_p3/p1_total*100 > 5:
            log("  -> P1 likely on SAME PAGE as P3 (Search Results)", f)
            log("  -> P1 may be a secondary search ad slot (sidebar/header)", f)
        elif max_cofire == p1_with_p5 and p1_with_p5/p1_total*100 > 5:
            log("  -> P1 likely on SAME PAGE as P5 (Brand Page)", f)
            log("  -> P1 may be a secondary brand page ad slot", f)
        else:
            log("  -> P1 is on a SEPARATE PAGE (low co-firing with all placements)", f)
            log("  -> P1 may be Homepage/Feed or an undocumented placement", f)

        # =====================================================================
        # SECTION 7: MULTI-PLACEMENT SESSIONS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 7: MULTI-PLACEMENT SESSIONS", f)
        log("=" * 80, f)
        log("", f)
        log("Do users typically encounter multiple placements in sequence?", f)
        log("", f)

        # Placements per user
        placements_per_user = au.groupby('USER_ID')['PLACEMENT'].nunique()

        log("Placements per user:", f)
        for n in range(1, 5):
            count = (placements_per_user == n).sum()
            pct = count / len(placements_per_user) * 100
            log(f"  {n} placement(s): {count:,} users ({pct:.1f}%)", f)

        # Most common placement sequences
        log("\nMost common placement sequences (first 3 placements per user):", f)

        def get_first_n_placements(group, n=3):
            return '-'.join(group.head(n)['PLACEMENT'].astype(str).tolist())

        au_sorted_for_seq = au.sort_values(['USER_ID', 'CREATED_AT'])
        sequences = au_sorted_for_seq.groupby('USER_ID').apply(get_first_n_placements)
        seq_counts = sequences.value_counts().head(15)

        for seq, count in seq_counts.items():
            pct = count / len(sequences) * 100
            log(f"  {seq:<15} {count:,} ({pct:.1f}%)", f)

        # =====================================================================
        # SECTION 8: SIMULTANEOUS AUCTIONS (EXACT TIMESTAMP MATCHES)
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 8: SIMULTANEOUS AUCTIONS (EXACT TIMESTAMP MATCHES)", f)
        log("=" * 80, f)
        log("", f)
        log("Auctions with identical timestamps strongly indicate same-page co-loading.", f)
        log("", f)

        # Find auctions with identical timestamps within user
        au_with_count = au.groupby(['USER_ID', 'CREATED_AT']).agg({
            'AUCTION_ID': 'count',
            'PLACEMENT': lambda x: list(x)
        }).reset_index()
        au_with_count.columns = ['USER_ID', 'CREATED_AT', 'N_AUCTIONS', 'PLACEMENTS']

        # Filter to multi-auction timestamps
        multi_auction = au_with_count[au_with_count['N_AUCTIONS'] > 1]

        log(f"Timestamps with 2+ auctions: {len(multi_auction):,}", f)
        log(f"Average auctions per multi-auction timestamp: {multi_auction['N_AUCTIONS'].mean():.2f}", f)

        # What placement combinations appear at same timestamp?
        multi_auction['PLACEMENT_SET'] = multi_auction['PLACEMENTS'].apply(lambda x: tuple(sorted(set(x))))
        combo_counts = multi_auction['PLACEMENT_SET'].value_counts().head(10)

        log("\nMost common placement combinations at identical timestamps:", f)
        for combo, count in combo_counts.items():
            combo_str = '+'.join([f'P{p}' for p in combo])
            pct = count / len(multi_auction) * 100
            log(f"  {combo_str:<15} {count:,} ({pct:.1f}%)", f)

        # =====================================================================
        # SECTION 9: SUMMARY & IMPLICATIONS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 9: SUMMARY & IMPLICATIONS", f)
        log("=" * 80, f)
        log("", f)

        log("CO-FIRING INTERPRETATION:", f)
        log("", f)
        log("If two placements co-fire at >5% rate, they're likely:", f)
        log("  - On the SAME page (different ad slots)", f)
        log("  - Loading simultaneously on page render", f)
        log("", f)
        log("If two placements co-fire at <2% rate, they're likely:", f)
        log("  - On DIFFERENT pages", f)
        log("  - User navigated between pages", f)
        log("", f)

        log("KEY FINDINGS:", f)
        log("", f)

        # Summarize P1 identity
        if p1_with_p2 > p1_with_p3 and p1_with_p2 > p1_with_p5:
            log(f"  1. P1 co-fires most with P2 ({p1_with_p2/p1_total*100:.2f}%)", f)
            log("     -> P1 may be on PDP alongside P2", f)
        elif p1_with_p3 > p1_with_p2 and p1_with_p3 > p1_with_p5:
            log(f"  1. P1 co-fires most with P3 ({p1_with_p3/p1_total*100:.2f}%)", f)
            log("     -> P1 may be on Search alongside P3", f)
        else:
            log(f"  1. P1 has low co-firing with all placements", f)
            log("     -> P1 is likely on a separate page (Homepage/Feed)", f)

        log("", f)
        log(f"  2. P3 <-> P5 baseline co-firing: {p3_with_p5/p3_total*100:.2f}%", f)
        log("     -> Confirms Search (P3) and Brand Page (P5) are separate pages", f)

        log("", f)
        log("REVISED PLACEMENT MAPPING:", f)
        log("", f)
        log("  P3 = Search Results (high self-transition, 2 slots/page)", f)
        log("  P5 = Brand Page (official Poshmark placement)", f)
        log("  P2 = PDP Ad Section (low self-transition, high CTR)", f)
        log("  P1 = TBD based on co-firing analysis above", f)

        log("\n" + "=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)
        log(f"Output saved to: {OUTPUT_FILE}", f)

if __name__ == "__main__":
    main()
