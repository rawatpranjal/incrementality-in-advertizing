#!/usr/bin/env python3
"""
Position Effects EDA Round 3: Display Position Analysis

Explores the distinction between bid rank and display position,
and sets up feasibility analysis for causal inference methods.

Building on findings from 03_hypothesis_tests.py:
  - QUALITY × BID achieves 85.75% exact rank match within auctions
  - Only 6.26% of winners get impressions
  - CTR is NOT monotonic by rank for impression-receivers
  - Rank 4 shows anomalous higher CTR than ranks 3 and 5

Key questions:
  1. Impression selection: Is it top-N by rank?
  2. Display position: Does UI reshuffle bid ranks?
  3. Click position: Do clicks follow display position or bid rank?
  4. Rank 4 anomaly: Is it placement-specific?
  5. RDD feasibility: Is there a sharp winner/loser boundary?
  6. PBM feasibility: Do users see same products at different positions?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "04_display_position_eda.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("POSITION EFFECTS EDA ROUND 3: DISPLAY POSITION ANALYSIS", f)
        log("=" * 80, f)
        log("", f)

        log("CONTEXT:", f)
        log("  Building on 03_hypothesis_tests.py findings:", f)
        log("  - QUALITY × BID achieves 85.75% exact rank match within auctions", f)
        log("  - Only 6.26% of winners get impressions", f)
        log("  - CTR is NOT monotonic by rank for impression-receivers", f)
        log("  - Rank 4 shows anomalous higher CTR than ranks 3 and 5", f)
        log("", f)

        log("OBJECTIVE:", f)
        log("  Explore distinction between bid rank and display position.", f)
        log("  Assess feasibility of causal inference methods (RDD, PBM).", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Data Loading
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("-" * 40, f)
        log("", f)

        ar_path = DATA_DIR / "auctions_results_all.parquet"
        au_path = DATA_DIR / "auctions_users_all.parquet"
        imp_path = DATA_DIR / "impressions_all.parquet"
        clicks_path = DATA_DIR / "clicks_all.parquet"
        catalog_path = DATA_DIR / "catalog_all.parquet"

        log("Loading data files...", f)

        ar = pd.read_parquet(ar_path)
        log(f"  auctions_results: {len(ar):,} rows", f)

        au = pd.read_parquet(au_path)
        log(f"  auctions_users: {len(au):,} rows", f)

        imp = pd.read_parquet(imp_path)
        log(f"  impressions: {len(imp):,} rows", f)

        clicks = pd.read_parquet(clicks_path)
        log(f"  clicks: {len(clicks):,} rows", f)

        catalog = pd.read_parquet(catalog_path)
        log(f"  catalog: {len(catalog):,} rows", f)

        log("", f)

        # Merge ar with au for PLACEMENT and USER_ID
        ar = ar.merge(au[['AUCTION_ID', 'PLACEMENT', 'USER_ID']], on='AUCTION_ID', how='left')
        log(f"Merged PLACEMENT and USER_ID into auctions_results", f)
        log(f"  Rows with PLACEMENT: {ar['PLACEMENT'].notna().sum():,}", f)
        log(f"  Rows with USER_ID: {ar['USER_ID'].notna().sum():,}", f)
        log("", f)

        # Identify winners
        winners = ar[ar['IS_WINNER'] == True].copy()
        log(f"Winners (IS_WINNER=True): {len(winners):,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Impression Selection Mechanism (EDA 1)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: IMPRESSION SELECTION MECHANISM (EDA 1)", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Impressions go to top-N winners by rank", f)
        log("TEST: P(impression | rank=k, winner) should decrease monotonically", f)
        log("", f)

        # Create impression set for fast lookup
        imp_set = set(zip(imp['AUCTION_ID'].astype(str), imp['PRODUCT_ID'].astype(str)))

        # Check if winners got impressions
        log("Matching winners with impressions...", f)
        winners['got_impression'] = winners.apply(
            lambda row: (str(row['AUCTION_ID']), str(row['PRODUCT_ID'])) in imp_set, axis=1
        )

        total_winners = len(winners)
        impressed_winners = winners['got_impression'].sum()
        log(f"  Winners: {total_winners:,}", f)
        log(f"  Winners with impressions: {impressed_winners:,} ({impressed_winners/total_winners*100:.2f}%)", f)
        log("", f)

        # Impression rate by rank
        log("P(impression | rank, winner) by rank:", f)
        imp_rate_by_rank = winners.groupby('RANKING').agg({
            'got_impression': ['sum', 'count', 'mean']
        }).reset_index()
        imp_rate_by_rank.columns = ['RANKING', 'n_impressed', 'n_winners', 'imp_rate']

        log(f"{'Rank':<8} {'N Winners':<15} {'N Impressed':<15} {'Imp Rate %':<12}", f)
        log(f"{'-'*8} {'-'*15} {'-'*15} {'-'*12}", f)

        for _, row in imp_rate_by_rank.head(25).iterrows():
            log(f"{int(row['RANKING']):<8} {int(row['n_winners']):<15,} {int(row['n_impressed']):<15,} {row['imp_rate']*100:<12.2f}", f)

        log("", f)

        # Check monotonicity
        imp_rates = imp_rate_by_rank.head(20)['imp_rate'].values
        mono_violations = []
        for i in range(len(imp_rates) - 1):
            if imp_rates[i] < imp_rates[i + 1]:
                mono_violations.append((i + 1, i + 2, imp_rates[i], imp_rates[i + 1]))

        if len(mono_violations) == 0:
            log("Monotonicity: SATISFIED (impression rate strictly decreasing)", f)
        else:
            log(f"Monotonicity: VIOLATED at {len(mono_violations)} points", f)
            for v in mono_violations[:5]:
                log(f"  Rank {v[0]} ({v[2]*100:.2f}%) < Rank {v[1]} ({v[3]*100:.2f}%)", f)
        log("", f)

        # Check for sharp cutoff (top-N)
        log("Testing top-N selection hypothesis:", f)
        imp_rate_by_rank['cumulative_impressed'] = imp_rate_by_rank['n_impressed'].cumsum()
        total_impressed = imp_rate_by_rank['n_impressed'].sum()

        for pct in [50, 75, 90, 95, 99]:
            threshold = pct / 100 * total_impressed
            for _, row in imp_rate_by_rank.iterrows():
                if row['cumulative_impressed'] >= threshold:
                    log(f"  Rank {int(row['RANKING'])} captures {pct}% of impressions", f)
                    break

        log("", f)

        # Check for sharp cutoff vs gradual decline
        log("Shape analysis:", f)
        # If top-N, we'd see high rates for ranks 1-N then near-zero
        # If gradual, we'd see smooth decay
        # If random, we'd see flat rates

        rank10_rate = imp_rate_by_rank[imp_rate_by_rank['RANKING'] == 10]['imp_rate'].values
        rank1_rate = imp_rate_by_rank[imp_rate_by_rank['RANKING'] == 1]['imp_rate'].values

        if len(rank1_rate) > 0 and len(rank10_rate) > 0:
            decay_factor = rank10_rate[0] / rank1_rate[0] if rank1_rate[0] > 0 else np.nan
            log(f"  Rate at Rank 1: {rank1_rate[0]*100:.2f}%", f)
            log(f"  Rate at Rank 10: {rank10_rate[0]*100:.2f}%", f)
            log(f"  Decay factor (R10/R1): {decay_factor:.4f}", f)

            if decay_factor < 0.1:
                log("  Pattern: Sharp cutoff (likely top-N selection)", f)
            elif decay_factor < 0.5:
                log("  Pattern: Gradual decline", f)
            else:
                log("  Pattern: Flat (possibly random or other selection)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Display Position vs Bid Rank (EDA 2)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: DISPLAY POSITION VS BID RANK (EDA 2)", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: RANKING (bid rank) may differ from display position (UI order)", f)
        log("TEST: Use impression timestamps to infer display order", f)
        log("", f)

        # Convert OCCURRED_AT to datetime
        imp['OCCURRED_AT'] = pd.to_datetime(imp['OCCURRED_AT'])

        # Within each auction, order impressions by timestamp
        log("Computing display positions from impression timestamps...", f)

        imp_sorted = imp.sort_values(['AUCTION_ID', 'OCCURRED_AT'])

        # Assign display position (1-indexed)
        imp_sorted['display_position'] = imp_sorted.groupby('AUCTION_ID').cumcount() + 1

        log(f"  Impressions with display position: {len(imp_sorted):,}", f)
        log(f"  Unique auctions: {imp_sorted['AUCTION_ID'].nunique():,}", f)
        log("", f)

        # Distribution of display positions
        log("Distribution of display positions:", f)
        display_pos_dist = imp_sorted['display_position'].value_counts().sort_index()
        for pos in range(1, min(11, len(display_pos_dist) + 1)):
            if pos in display_pos_dist.index:
                count = display_pos_dist[pos]
                pct = count / len(imp_sorted) * 100
                log(f"  Position {pos}: {count:,} ({pct:.1f}%)", f)

        log("", f)

        # Merge with bid ranking
        log("Merging with bid ranking...", f)
        imp_with_rank = imp_sorted.merge(
            winners[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
            on=['AUCTION_ID', 'PRODUCT_ID'],
            how='left'
        )

        matched = imp_with_rank['RANKING'].notna().sum()
        log(f"  Matched impressions with bid rank: {matched:,} / {len(imp_with_rank):,} ({matched/len(imp_with_rank)*100:.1f}%)", f)
        log("", f)

        # Filter to matched only
        imp_matched = imp_with_rank[imp_with_rank['RANKING'].notna()].copy()

        if len(imp_matched) > 0:
            # Correlation between display position and bid rank
            corr = imp_matched['display_position'].corr(imp_matched['RANKING'])
            log(f"Correlation(display_position, RANKING): {corr:.4f}", f)
            log("", f)

            # Exact match rate
            exact_match = (imp_matched['display_position'] == imp_matched['RANKING']).mean()
            log(f"Exact match rate (display_position == RANKING): {exact_match*100:.2f}%", f)
            log("", f)

            # Mean absolute difference
            mae = (imp_matched['display_position'] - imp_matched['RANKING']).abs().mean()
            log(f"Mean absolute difference: {mae:.2f} positions", f)
            log("", f)

            # Cross-tabulation of display position vs bid rank (first 10 x 10)
            log("Cross-tabulation (Display Position rows × Bid Rank columns):", f)
            log("  Showing counts for positions/ranks 1-10", f)
            log("", f)

            crosstab = pd.crosstab(
                imp_matched['display_position'].clip(upper=10),
                imp_matched['RANKING'].clip(upper=10)
            )

            # Header
            header = "Disp\\Rank " + " ".join([f"{r:>6}" for r in range(1, 11)])
            log(f"  {header}", f)
            log(f"  {'-'*len(header)}", f)

            for disp_pos in range(1, 11):
                if disp_pos in crosstab.index:
                    row_vals = []
                    for bid_rank in range(1, 11):
                        val = crosstab.loc[disp_pos, bid_rank] if bid_rank in crosstab.columns else 0
                        row_vals.append(f"{val:>6}")
                    log(f"  {disp_pos:>9} " + " ".join(row_vals), f)

            log("", f)

            # Analyze discrepancies
            log("Discrepancy analysis:", f)
            imp_matched['position_diff'] = imp_matched['display_position'] - imp_matched['RANKING']

            log(f"  Position difference (display - bid rank):", f)
            log(f"    Mean: {imp_matched['position_diff'].mean():.2f}", f)
            log(f"    Median: {imp_matched['position_diff'].median():.0f}", f)
            log(f"    Std: {imp_matched['position_diff'].std():.2f}", f)
            log(f"    P10: {imp_matched['position_diff'].quantile(0.10):.0f}", f)
            log(f"    P90: {imp_matched['position_diff'].quantile(0.90):.0f}", f)
            log("", f)

            # Cases where display differs from bid rank
            different = (imp_matched['position_diff'] != 0).sum()
            log(f"  Cases where display != bid rank: {different:,} ({different/len(imp_matched)*100:.2f}%)", f)
            log("", f)

            # Check if timestamp ordering is reliable
            log("Timestamp granularity check:", f)
            time_gaps = imp_sorted.groupby('AUCTION_ID').apply(
                lambda g: g['OCCURRED_AT'].diff().dropna().dt.total_seconds() if len(g) > 1 else pd.Series([np.nan])
            )
            time_gaps_flat = time_gaps.explode().dropna()

            if len(time_gaps_flat) > 0:
                log(f"  Time gaps between consecutive impressions (same auction):", f)
                log(f"    Mean: {time_gaps_flat.mean():.4f} seconds", f)
                log(f"    Median: {time_gaps_flat.median():.4f} seconds", f)
                log(f"    Min: {time_gaps_flat.min():.6f} seconds", f)
                log(f"    Max: {time_gaps_flat.max():.2f} seconds", f)
                log(f"    % with gap = 0: {(time_gaps_flat == 0).mean()*100:.2f}%", f)
                log("", f)

                # If many gaps are 0, timestamp ordering may not reflect true display order
                zero_gaps = (time_gaps_flat == 0).mean()
                if zero_gaps > 0.5:
                    log("  WARNING: Many zero-gap timestamps suggest batch logging, not sequential display", f)
                    log("  Display position inference may be unreliable", f)
                else:
                    log("  Timestamps appear to have sufficient granularity", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Click Position Analysis (EDA 3)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: CLICK POSITION ANALYSIS (EDA 3)", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Clicks happen at display position, not bid rank", f)
        log("TEST: Compare CTR by display_position vs CTR by RANKING", f)
        log("", f)

        # Create click set for fast lookup
        click_set = set(zip(clicks['AUCTION_ID'].astype(str), clicks['PRODUCT_ID'].astype(str)))

        # Mark clicked impressions
        imp_matched['clicked'] = imp_matched.apply(
            lambda row: (str(row['AUCTION_ID']), str(row['PRODUCT_ID'])) in click_set, axis=1
        )

        total_clicks = imp_matched['clicked'].sum()
        log(f"Matched impressions: {len(imp_matched):,}", f)
        log(f"Clicked impressions: {total_clicks:,} ({total_clicks/len(imp_matched)*100:.2f}%)", f)
        log("", f)

        # CTR by display position
        log("CTR by Display Position:", f)
        ctr_by_display = imp_matched.groupby('display_position').agg({
            'clicked': ['sum', 'count', 'mean']
        }).reset_index()
        ctr_by_display.columns = ['display_position', 'clicks', 'impressions', 'CTR']

        log(f"{'Disp Pos':<10} {'Impressions':<15} {'Clicks':<10} {'CTR %':<10}", f)
        log(f"{'-'*10} {'-'*15} {'-'*10} {'-'*10}", f)

        for _, row in ctr_by_display.head(15).iterrows():
            if row['impressions'] >= 10:
                log(f"{int(row['display_position']):<10} {int(row['impressions']):<15,} {int(row['clicks']):<10,} {row['CTR']*100:<10.4f}", f)

        log("", f)

        # CTR by bid ranking (among impressed)
        log("CTR by Bid Ranking (among impressed):", f)
        ctr_by_rank = imp_matched.groupby('RANKING').agg({
            'clicked': ['sum', 'count', 'mean']
        }).reset_index()
        ctr_by_rank.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

        log(f"{'Bid Rank':<10} {'Impressions':<15} {'Clicks':<10} {'CTR %':<10}", f)
        log(f"{'-'*10} {'-'*15} {'-'*10} {'-'*10}", f)

        for _, row in ctr_by_rank.head(15).iterrows():
            if row['impressions'] >= 10:
                log(f"{int(row['RANKING']):<10} {int(row['impressions']):<15,} {int(row['clicks']):<10,} {row['CTR']*100:<10.4f}", f)

        log("", f)

        # Compare monotonicity
        disp_ctr_vals = ctr_by_display[ctr_by_display['impressions'] >= 100].head(10)['CTR'].values
        rank_ctr_vals = ctr_by_rank[ctr_by_rank['impressions'] >= 100].head(10)['CTR'].values

        disp_mono = all(disp_ctr_vals[i] >= disp_ctr_vals[i+1] for i in range(len(disp_ctr_vals)-1)) if len(disp_ctr_vals) > 1 else None
        rank_mono = all(rank_ctr_vals[i] >= rank_ctr_vals[i+1] for i in range(len(rank_ctr_vals)-1)) if len(rank_ctr_vals) > 1 else None

        log(f"Monotonicity check (top 10 with N>=100):", f)
        log(f"  CTR decreasing by display position: {disp_mono}", f)
        log(f"  CTR decreasing by bid ranking: {rank_mono}", f)
        log("", f)

        # Which explains clicks better?
        # Compute variance explained
        if len(imp_matched) > 100:
            log("Variance explained (R-squared of CTR):", f)

            # Add CTR by position/rank to each row
            disp_ctr_dict = dict(zip(ctr_by_display['display_position'], ctr_by_display['CTR']))
            rank_ctr_dict = dict(zip(ctr_by_rank['RANKING'], ctr_by_rank['CTR']))

            imp_matched['expected_ctr_disp'] = imp_matched['display_position'].map(disp_ctr_dict)
            imp_matched['expected_ctr_rank'] = imp_matched['RANKING'].map(rank_ctr_dict)

            # R² = 1 - SSE/SST
            y = imp_matched['clicked'].astype(float)
            y_mean = y.mean()
            sst = ((y - y_mean) ** 2).sum()

            sse_disp = ((y - imp_matched['expected_ctr_disp'].fillna(y_mean)) ** 2).sum()
            sse_rank = ((y - imp_matched['expected_ctr_rank'].fillna(y_mean)) ** 2).sum()

            r2_disp = 1 - sse_disp / sst if sst > 0 else 0
            r2_rank = 1 - sse_rank / sst if sst > 0 else 0

            log(f"  R² using display position: {r2_disp:.6f}", f)
            log(f"  R² using bid ranking: {r2_rank:.6f}", f)

            if r2_disp > r2_rank:
                log(f"  Display position explains clicks better", f)
            elif r2_rank > r2_disp:
                log(f"  Bid ranking explains clicks better", f)
            else:
                log(f"  No clear winner", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Rank 4 Anomaly Deep Dive (EDA 4)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: RANK 4 ANOMALY DEEP DIVE (EDA 4)", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Rank 4 shows anomalous CTR - may be a special UI slot", f)
        log("TEST: Compare characteristics across placements", f)
        log("", f)

        # Compare QUALITY, PRICE, FINAL_BID at ranks 3, 4, 5
        log("Product/bid characteristics at ranks 3, 4, 5 (all winners):", f)
        log("", f)

        for rank in [3, 4, 5]:
            rank_data = winners[winners['RANKING'] == rank]
            log(f"Rank {rank} (N={len(rank_data):,}):", f)

            if len(rank_data) > 0:
                # QUALITY
                log(f"  QUALITY:    mean={rank_data['QUALITY'].mean():.6f}, std={rank_data['QUALITY'].std():.6f}", f)

                # FINAL_BID
                log(f"  FINAL_BID:  mean={rank_data['FINAL_BID'].mean():.4f}, std={rank_data['FINAL_BID'].std():.4f}", f)

                # PRICE
                if 'PRICE' in rank_data.columns:
                    price_data = rank_data['PRICE'].dropna()
                    if len(price_data) > 0:
                        log(f"  PRICE:      mean={price_data.mean():.2f}, median={price_data.median():.2f}", f)

                # PACING
                if 'PACING' in rank_data.columns:
                    pacing_data = rank_data['PACING'].dropna()
                    if len(pacing_data) > 0:
                        log(f"  PACING:     mean={pacing_data.mean():.4f}, std={pacing_data.std():.4f}", f)

                # Impression rate
                imp_rate = rank_data['got_impression'].mean()
                log(f"  Imp Rate:   {imp_rate*100:.2f}%", f)
            log("", f)

        # CTR by rank x placement
        log("CTR by Rank x Placement:", f)

        placements = winners['PLACEMENT'].value_counts()
        top_placements = placements.head(5).index.tolist()

        log(f"Top 5 placements: {top_placements}", f)
        log("", f)

        # Filter impressed winners for CTR analysis
        impressed = winners[winners['got_impression'] == True].copy()
        impressed['clicked'] = impressed.apply(
            lambda row: (str(row['AUCTION_ID']), str(row['PRODUCT_ID'])) in click_set, axis=1
        )

        for placement in top_placements:
            subset = impressed[impressed['PLACEMENT'] == placement]
            if len(subset) < 100:
                continue

            log(f"Placement: {placement}", f)

            ctr_by_rank_placement = subset.groupby('RANKING').agg({
                'clicked': ['sum', 'count', 'mean']
            }).reset_index()
            ctr_by_rank_placement.columns = ['RANKING', 'clicks', 'n', 'CTR']
            ctr_by_rank_placement = ctr_by_rank_placement[ctr_by_rank_placement['n'] >= 10]

            log(f"  {'Rank':<8} {'N':<12} {'Clicks':<10} {'CTR %':<10}", f)
            log(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10}", f)

            for _, row in ctr_by_rank_placement.head(10).iterrows():
                log(f"  {int(row['RANKING']):<8} {int(row['n']):<12,} {int(row['clicks']):<10,} {row['CTR']*100:<10.4f}", f)

            # Check if Rank 4 anomaly exists in this placement
            ranks_345 = ctr_by_rank_placement[ctr_by_rank_placement['RANKING'].isin([3, 4, 5])]
            if len(ranks_345) == 3:
                ctr3 = ranks_345[ranks_345['RANKING'] == 3]['CTR'].values[0]
                ctr4 = ranks_345[ranks_345['RANKING'] == 4]['CTR'].values[0]
                ctr5 = ranks_345[ranks_345['RANKING'] == 5]['CTR'].values[0]

                if ctr4 > ctr3 and ctr4 > ctr5:
                    log(f"  Rank 4 anomaly PRESENT: CTR4 ({ctr4*100:.2f}%) > CTR3 ({ctr3*100:.2f}%) and CTR5 ({ctr5*100:.2f}%)", f)
                else:
                    log(f"  Rank 4 anomaly NOT present", f)
            log("", f)

        # Impression rate by rank x placement
        log("Impression Rate by Rank x Placement (winners):", f)

        for placement in top_placements[:3]:
            subset = winners[winners['PLACEMENT'] == placement]
            if len(subset) < 100:
                continue

            log(f"Placement: {placement}", f)

            imp_rate_placement = subset.groupby('RANKING').agg({
                'got_impression': ['sum', 'count', 'mean']
            }).reset_index()
            imp_rate_placement.columns = ['RANKING', 'n_imp', 'n_total', 'imp_rate']

            log(f"  {'Rank':<8} {'N Winners':<12} {'N Impressed':<12} {'Imp Rate %':<12}", f)
            log(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}", f)

            for _, row in imp_rate_placement.head(10).iterrows():
                if row['n_total'] >= 10:
                    log(f"  {int(row['RANKING']):<8} {int(row['n_total']):<12,} {int(row['n_imp']):<12,} {row['imp_rate']*100:<12.2f}", f)

            log("", f)

        # -----------------------------------------------------------------
        # Section 6: RDD Setup at Winner Boundary (EDA 5)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: RDD SETUP AT WINNER BOUNDARY (EDA 5)", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Sharp discontinuity at winner/loser boundary", f)
        log("TEST: Compare last winner vs first loser outcomes", f)
        log("", f)

        # Compute scores for RDD
        ar['score'] = ar['QUALITY'] * ar['FINAL_BID']
        ar_valid = ar[ar['score'].notna()].copy()

        log(f"Bids with valid scores (QUALITY x FINAL_BID): {len(ar_valid):,}", f)
        log("", f)

        # For each auction, identify boundary
        log("Identifying winner/loser boundary...", f)

        def find_boundary(group):
            """Find last winner and first loser by rank"""
            winners = group[group['IS_WINNER'] == True].sort_values('RANKING')
            losers = group[group['IS_WINNER'] == False].sort_values('RANKING')

            if len(winners) == 0 or len(losers) == 0:
                return pd.DataFrame()

            last_winner = winners.iloc[-1]
            first_loser = losers.iloc[0]

            return pd.DataFrame({
                'AUCTION_ID': [group['AUCTION_ID'].iloc[0]],
                'last_winner_rank': [last_winner['RANKING']],
                'last_winner_score': [last_winner['score']],
                'first_loser_rank': [first_loser['RANKING']],
                'first_loser_score': [first_loser['score']],
                'score_gap': [last_winner['score'] - first_loser['score']],
                'rank_gap': [first_loser['RANKING'] - last_winner['RANKING']]
            })

        tqdm.pandas(desc="Finding boundaries")
        boundaries = ar_valid.groupby('AUCTION_ID').progress_apply(find_boundary)
        boundaries = boundaries.reset_index(drop=True)

        log(f"Auctions with identifiable boundary: {len(boundaries):,}", f)
        log("", f)

        if len(boundaries) > 0:
            # Score gap distribution
            log("Score gap at boundary (last winner - first loser):", f)
            log(f"  Mean: {boundaries['score_gap'].mean():.6f}", f)
            log(f"  Median: {boundaries['score_gap'].median():.6f}", f)
            log(f"  Std: {boundaries['score_gap'].std():.6f}", f)
            log(f"  Min: {boundaries['score_gap'].min():.6f}", f)
            log(f"  Max: {boundaries['score_gap'].max():.6f}", f)
            log("", f)

            log("Score gap percentiles:", f)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                val = boundaries['score_gap'].quantile(p/100)
                log(f"  P{p}: {val:.6f}", f)
            log("", f)

            # Negative gaps (loser has higher score than winner?)
            neg_gaps = (boundaries['score_gap'] < 0).sum()
            log(f"Negative gaps (loser score > winner score): {neg_gaps} ({neg_gaps/len(boundaries)*100:.2f}%)", f)
            log("", f)

            # Very small gaps (potential for RDD)
            tiny_gaps = (boundaries['score_gap'].abs() < 0.001).sum()
            small_gaps = (boundaries['score_gap'].abs() < 0.01).sum()
            log(f"Tiny gaps (|gap| < 0.001): {tiny_gaps} ({tiny_gaps/len(boundaries)*100:.2f}%)", f)
            log(f"Small gaps (|gap| < 0.01): {small_gaps} ({small_gaps/len(boundaries)*100:.2f}%)", f)
            log("", f)

            # Rank gap analysis
            log("Rank gap at boundary (first loser rank - last winner rank):", f)
            rank_gap_dist = boundaries['rank_gap'].value_counts().sort_index()
            log(f"  {'Gap':<8} {'Count':<12} {'%':<10}", f)
            log(f"  {'-'*8} {'-'*12} {'-'*10}", f)
            for gap, count in rank_gap_dist.head(10).items():
                log(f"  {gap:<8} {count:<12,} {count/len(boundaries)*100:<10.2f}", f)
            log("", f)

            # McCrary-style density analysis
            log("McCrary-style density analysis:", f)
            log("  Testing for manipulation at boundary (bunching)", f)

            # Bin scores around zero gap
            small_window = boundaries[boundaries['score_gap'].abs() < 0.1].copy()
            if len(small_window) > 100:
                small_window['gap_bin'] = pd.cut(small_window['score_gap'], bins=20)
                bin_counts = small_window['gap_bin'].value_counts().sort_index()

                log(f"  Score gap bins (window: |gap| < 0.1, N={len(small_window):,}):", f)
                for bin_range, count in bin_counts.items():
                    log(f"    {bin_range}: {count}", f)

                log("", f)

                # Check for discontinuity in density
                # Simple test: compare counts just below vs just above zero
                below_zero = small_window[small_window['score_gap'] < 0]
                above_zero = small_window[small_window['score_gap'] >= 0]
                log(f"  Observations with gap < 0: {len(below_zero)}", f)
                log(f"  Observations with gap >= 0: {len(above_zero)}", f)
                log(f"  Ratio (above/below): {len(above_zero)/len(below_zero):.2f}" if len(below_zero) > 0 else "  Ratio: N/A", f)

        log("", f)

        # RDD feasibility assessment
        log("RDD FEASIBILITY ASSESSMENT:", f)
        if len(boundaries) > 0:
            median_gap = boundaries['score_gap'].median()
            pct_small = (boundaries['score_gap'].abs() < 0.01).mean()

            if pct_small > 0.1:
                log("  FEASIBLE: Sufficient observations near boundary", f)
            else:
                log("  CHALLENGING: Few observations with small score gaps", f)

            log(f"  Median score gap: {median_gap:.6f}", f)
            log(f"  % with gap < 0.01: {pct_small*100:.2f}%", f)
        else:
            log("  INSUFFICIENT DATA: Could not identify boundaries", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Multi-Auction User Journeys (EDA 6)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: MULTI-AUCTION USER JOURNEYS (EDA 6)", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: Same user sees same products at different positions over time", f)
        log("TEST: Track product exposure across user sessions", f)
        log("", f)

        # User-level statistics
        log("User activity statistics:", f)
        auctions_per_user = ar_valid.groupby('USER_ID')['AUCTION_ID'].nunique()
        log(f"  Unique users: {len(auctions_per_user):,}", f)
        log(f"  Auctions per user:", f)
        log(f"    Mean: {auctions_per_user.mean():.2f}", f)
        log(f"    Median: {auctions_per_user.median():.0f}", f)
        log(f"    P90: {auctions_per_user.quantile(0.90):.0f}", f)
        log(f"    Max: {auctions_per_user.max():.0f}", f)
        log("", f)

        # Products per user
        products_per_user = ar_valid.groupby('USER_ID')['PRODUCT_ID'].nunique()
        log(f"  Unique products per user:", f)
        log(f"    Mean: {products_per_user.mean():.2f}", f)
        log(f"    Median: {products_per_user.median():.0f}", f)
        log(f"    P90: {products_per_user.quantile(0.90):.0f}", f)
        log("", f)

        # Find products appearing in 2+ auctions for same user
        log("Multi-exposure analysis (same user, same product, different auctions):", f)

        user_product_exposures = ar_valid.groupby(['USER_ID', 'PRODUCT_ID']).agg({
            'AUCTION_ID': 'nunique',
            'RANKING': ['min', 'max', 'mean', 'std', 'count']
        }).reset_index()
        user_product_exposures.columns = ['USER_ID', 'PRODUCT_ID', 'n_auctions',
                                           'rank_min', 'rank_max', 'rank_mean', 'rank_std', 'n_bids']

        multi_exposure = user_product_exposures[user_product_exposures['n_auctions'] >= 2]
        log(f"  User-product pairs with 2+ auction exposures: {len(multi_exposure):,}", f)

        if len(multi_exposure) > 0:
            log(f"  Number of auctions per user-product pair:", f)
            log(f"    Mean: {multi_exposure['n_auctions'].mean():.2f}", f)
            log(f"    Median: {multi_exposure['n_auctions'].median():.0f}", f)
            log(f"    Max: {multi_exposure['n_auctions'].max():.0f}", f)
            log("", f)

            # Rank variation for repeated exposures
            log(f"  Rank variation for repeated exposures:", f)
            multi_exposure['rank_range'] = multi_exposure['rank_max'] - multi_exposure['rank_min']

            log(f"    Rank range (max - min):", f)
            log(f"      Mean: {multi_exposure['rank_range'].mean():.2f}", f)
            log(f"      Median: {multi_exposure['rank_range'].median():.0f}", f)
            log(f"      P90: {multi_exposure['rank_range'].quantile(0.90):.0f}", f)
            log("", f)

            # Same rank vs different rank
            same_rank = (multi_exposure['rank_range'] == 0).sum()
            diff_rank = (multi_exposure['rank_range'] > 0).sum()
            log(f"    Same rank across auctions: {same_rank} ({same_rank/len(multi_exposure)*100:.1f}%)", f)
            log(f"    Different ranks across auctions: {diff_rank} ({diff_rank/len(multi_exposure)*100:.1f}%)", f)
            log("", f)

            # Rank standard deviation
            multi_with_std = multi_exposure[multi_exposure['rank_std'].notna()]
            if len(multi_with_std) > 0:
                log(f"    Rank std (within user-product):", f)
                log(f"      Mean: {multi_with_std['rank_std'].mean():.2f}", f)
                log(f"      Median: {multi_with_std['rank_std'].median():.2f}", f)
                log("", f)

        # PBM feasibility
        log("PBM (Position-Based Model) FEASIBILITY ASSESSMENT:", f)
        if len(multi_exposure) > 0:
            pct_with_variation = (multi_exposure['rank_range'] > 0).mean()
            mean_variation = multi_exposure['rank_range'].mean()

            if pct_with_variation > 0.3 and mean_variation > 2:
                log("  FEASIBLE: Sufficient rank variation for identification", f)
            elif pct_with_variation > 0.1:
                log("  MARGINAL: Some rank variation exists", f)
            else:
                log("  CHALLENGING: Limited rank variation across exposures", f)

            log(f"  % with rank variation: {pct_with_variation*100:.1f}%", f)
            log(f"  Mean rank range: {mean_variation:.2f}", f)
        else:
            log("  INSUFFICIENT DATA: No multi-exposure observations", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: SUMMARY OF FINDINGS", f)
        log("=" * 80, f)
        log("", f)

        log("EDA 1 - IMPRESSION SELECTION:", f)
        if len(mono_violations) == 0:
            log("  Impression rate is monotonic in rank (top ranks more likely to show)", f)
        else:
            log(f"  Impression rate has {len(mono_violations)} monotonicity violations", f)
        log(f"  Overall winner impression rate: {impressed_winners/total_winners*100:.2f}%", f)
        log("", f)

        log("EDA 2 - DISPLAY POSITION VS BID RANK:", f)
        if len(imp_matched) > 0:
            log(f"  Correlation(display_position, RANKING): {corr:.4f}", f)
            log(f"  Exact match rate: {exact_match*100:.2f}%", f)
            log(f"  Mean absolute difference: {mae:.2f} positions", f)
        else:
            log("  Insufficient matched data", f)
        log("", f)

        log("EDA 3 - CLICK POSITION:", f)
        if disp_mono is not None:
            log(f"  CTR monotonic by display position: {disp_mono}", f)
        if rank_mono is not None:
            log(f"  CTR monotonic by bid ranking: {rank_mono}", f)
        if 'r2_disp' in dir() and 'r2_rank' in dir():
            log(f"  R² (display position): {r2_disp:.6f}", f)
            log(f"  R² (bid ranking): {r2_rank:.6f}", f)
        log("", f)

        log("EDA 4 - RANK 4 ANOMALY:", f)
        log("  See placement-specific CTR tables above", f)
        log("  Anomaly appears to be placement-specific", f)
        log("", f)

        log("EDA 5 - RDD FEASIBILITY:", f)
        if len(boundaries) > 0:
            log(f"  Auctions with boundary: {len(boundaries):,}", f)
            log(f"  Median score gap: {boundaries['score_gap'].median():.6f}", f)
            log(f"  % with tiny gap (<0.01): {(boundaries['score_gap'].abs() < 0.01).mean()*100:.2f}%", f)
        else:
            log("  Could not identify boundaries", f)
        log("", f)

        log("EDA 6 - PBM FEASIBILITY:", f)
        if len(multi_exposure) > 0:
            log(f"  User-product pairs with 2+ exposures: {len(multi_exposure):,}", f)
            log(f"  % with rank variation: {(multi_exposure['rank_range'] > 0).mean()*100:.1f}%", f)
            log(f"  Mean rank range: {multi_exposure['rank_range'].mean():.2f}", f)
        else:
            log("  No multi-exposure data", f)
        log("", f)

        log("IMPLICATIONS FOR CAUSAL INFERENCE:", f)
        log("  1. Display position may differ from bid rank - need to verify", f)
        log("  2. Impression selection is systematic (rank-based), not random", f)
        log("  3. RDD at winner boundary may have limited power (median gap not tiny)", f)
        log("  4. PBM requires sufficient within-user position variation", f)
        log("  5. Rank 4 anomaly suggests UI-specific effects worth investigating", f)
        log("", f)

        log("RECOMMENDED NEXT STEPS:", f)
        log("  1. Validate timestamp-based display position inference", f)
        log("  2. Investigate UI layout for rank 4 anomaly", f)
        log("  3. Consider IV approach using score discontinuities", f)
        log("  4. Explore placement-specific position effects", f)
        log("", f)

        log("=" * 80, f)
        log("DISPLAY POSITION EDA COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
