#!/usr/bin/env python3
"""
Auction Pressure Analysis: Competitive Metrics as Potential Instruments

Studies competition measures within auctions as potential instrumental variables
for identifying causal position effects. Good instruments should:
1. Correlate with RANKING (first stage)
2. Not directly affect clicks conditional on ranking (exclusion)
3. Be independent of focal product quality (independence)

Measures computed:
- Count-based: n_bidders, n_winners, winner_rate
- Score-based: mean/max/median/std competitor scores, focal percentile/zscore
- Gap-based: gap_above, gap_below, gap_ratio, local_density
- Concentration: HHI, Gini, top3_share, effective_competitors
- Threat: n_better, threat_score, closest_challenger
- Decomposition: correlation/variance decomposition of quality vs bid
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from scipy import stats

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # Go up one level to position-effects/
DATA_DIR = BASE_DIR / "data_r2"
RESULTS_DIR = Path(__file__).parent / "results"  # Local results in iv-analysis/results/
OUTPUT_FILE = RESULTS_DIR / "10_auction_pressure.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# COMPETITION MEASURES
# =============================================================================
def compute_competition_measures(ar, f):
    """
    Compute all competition measures for each bid in the auction.
    Returns dataframe with one row per bid, augmented with competition metrics.
    """
    log("=" * 80, f)
    log("COMPUTING COMPETITION MEASURES", f)
    log("=" * 80, f)
    log("", f)

    df = ar.copy()

    # Compute score = QUALITY * FINAL_BID * PACING
    df['score'] = df['QUALITY'] * df['FINAL_BID'] * df['PACING']

    log(f"Total bids: {len(df):,}", f)
    log(f"Total auctions: {df['AUCTION_ID'].nunique():,}", f)
    log("", f)

    # Pre-compute auction-level aggregates
    log("Computing auction-level aggregates...", f)

    auction_agg = df.groupby('AUCTION_ID').agg({
        'PRODUCT_ID': 'count',
        'IS_WINNER': 'sum',
        'score': ['sum', 'mean', 'std', 'max', 'median'],
        'QUALITY': ['mean', 'std', 'var'],
        'FINAL_BID': ['mean', 'std', 'var']
    }).reset_index()
    auction_agg.columns = ['AUCTION_ID', 'n_bidders', 'n_winners',
                           'total_score', 'mean_score', 'std_score', 'max_score', 'median_score',
                           'mean_quality', 'std_quality', 'var_quality',
                           'mean_bid', 'std_bid', 'var_bid']

    # Merge auction aggregates
    df = df.merge(auction_agg, on='AUCTION_ID', how='left')

    # -------------------------------------------------------------------------
    # COUNT-BASED MEASURES (3)
    # -------------------------------------------------------------------------
    log("Computing count-based measures...", f)

    # n_bidders already computed
    # n_winners already computed
    df['winner_rate'] = df['n_winners'] / df['n_bidders']

    # -------------------------------------------------------------------------
    # SCORE-BASED MEASURES (7)
    # -------------------------------------------------------------------------
    log("Computing score-based measures...", f)

    # mean_competitor_score = (total_score - focal_score) / (n_bidders - 1)
    df['mean_competitor_score'] = np.where(
        df['n_bidders'] > 1,
        (df['total_score'] - df['score']) / (df['n_bidders'] - 1),
        0
    )

    # For max_competitor_score and median, we need per-bid computation
    # This is expensive, so we use a vectorized approach with groupby

    # Compute rank of score within auction (for percentile)
    df['score_rank'] = df.groupby('AUCTION_ID')['score'].rank(method='average', ascending=True)
    df['score_percentile'] = (df['score_rank'] - 1) / (df['n_bidders'] - 1).clip(lower=1)

    # z-score of focal score
    df['score_zscore'] = np.where(
        df['std_score'] > 0,
        (df['score'] - df['mean_score']) / df['std_score'],
        0
    )

    # For max_competitor_score: max in auction minus focal if focal is max
    # We need a different approach - compute for each bid
    log("  Computing max_competitor_score (this may take a moment)...", f)

    # Sort by auction and score descending
    df_sorted = df.sort_values(['AUCTION_ID', 'score'], ascending=[True, False])

    # Within each auction, the max competitor score for a bid is:
    # - If bid has highest score: second highest
    # - Otherwise: the max score
    def compute_max_competitor(group):
        scores = group['score'].values
        n = len(scores)
        if n == 1:
            return np.zeros(n)
        # scores are sorted descending
        max_comp = np.zeros(n)
        max_comp[0] = scores[1]  # For highest, competitor max is 2nd highest
        max_comp[1:] = scores[0]  # For others, competitor max is the highest
        return max_comp

    max_comp_values = df_sorted.groupby('AUCTION_ID', group_keys=False).apply(
        lambda g: pd.Series(compute_max_competitor(g), index=g.index)
    )
    df['max_competitor_score'] = max_comp_values

    # median_competitor_score: approximate with auction median (small bias)
    df['median_competitor_score'] = df['median_score']

    # -------------------------------------------------------------------------
    # GAP-BASED MEASURES (4)
    # -------------------------------------------------------------------------
    log("Computing gap-based measures...", f)

    # Sort by auction and ranking
    df = df.sort_values(['AUCTION_ID', 'RANKING'])

    # gap_above = score of next better product - focal score
    # gap_below = focal score - score of next worse product

    df['score_above'] = df.groupby('AUCTION_ID')['score'].shift(-1)  # Better rank = lower ranking number
    df['score_below'] = df.groupby('AUCTION_ID')['score'].shift(1)   # Worse rank = higher ranking number

    # Wait - RANKING is position (1=best). Better products have LOWER ranking numbers.
    # So shift(-1) gives the NEXT row which has WORSE ranking (higher number = worse position)
    # We need to reverse: shift(1) gives better products, shift(-1) gives worse products

    # Re-sort: RANKING 1 is best (highest score), RANKING increases = worse
    df = df.sort_values(['AUCTION_ID', 'RANKING'])

    # Better product = lower RANKING = previous row after sorting by RANKING asc
    df['score_better'] = df.groupby('AUCTION_ID')['score'].shift(1)  # Previous row has better rank
    df['score_worse'] = df.groupby('AUCTION_ID')['score'].shift(-1)  # Next row has worse rank

    df['gap_above'] = df['score_better'] - df['score']  # How much better is the product above
    df['gap_below'] = df['score'] - df['score_worse']   # How much better than product below

    # Fill NaN with 0 for products at boundaries
    df['gap_above'] = df['gap_above'].fillna(0)
    df['gap_below'] = df['gap_below'].fillna(0)

    # gap_ratio = gap_above / gap_below (competitive pressure from above vs cushion below)
    df['gap_ratio'] = np.where(
        df['gap_below'] > 0,
        df['gap_above'] / df['gap_below'],
        np.where(df['gap_above'] > 0, np.inf, 1)
    )
    df['gap_ratio'] = df['gap_ratio'].replace([np.inf, -np.inf], 10)  # Cap at 10

    # local_density = 1 / (gap_above + gap_below + epsilon)
    df['local_density'] = 1 / (df['gap_above'].abs() + df['gap_below'].abs() + 0.001)

    # -------------------------------------------------------------------------
    # CONCENTRATION MEASURES (4)
    # -------------------------------------------------------------------------
    log("Computing concentration measures...", f)

    # HHI of scores
    def compute_hhi(scores):
        total = scores.sum()
        if total == 0:
            return 0
        shares = scores / total
        return (shares ** 2).sum()

    hhi_by_auction = df.groupby('AUCTION_ID')['score'].apply(compute_hhi)
    df['hhi_score'] = df['AUCTION_ID'].map(hhi_by_auction)

    # Gini coefficient of scores
    def compute_gini(scores):
        scores = np.sort(scores)
        n = len(scores)
        if n == 0 or scores.sum() == 0:
            return 0
        cumsum = np.cumsum(scores)
        return (2 * np.sum((np.arange(1, n+1) * scores)) / (n * scores.sum())) - (n + 1) / n

    gini_by_auction = df.groupby('AUCTION_ID')['score'].apply(compute_gini)
    df['gini_score'] = df['AUCTION_ID'].map(gini_by_auction)

    # Top 3 share
    def compute_top3_share(scores):
        total = scores.sum()
        if total == 0:
            return 0
        top3 = np.sort(scores)[-3:].sum()
        return top3 / total

    top3_by_auction = df.groupby('AUCTION_ID')['score'].apply(compute_top3_share)
    df['top3_share'] = df['AUCTION_ID'].map(top3_by_auction)

    # Effective competitors = 1 / HHI
    df['effective_competitors'] = np.where(df['hhi_score'] > 0, 1 / df['hhi_score'], df['n_bidders'])

    # -------------------------------------------------------------------------
    # THREAT MEASURES (3)
    # -------------------------------------------------------------------------
    log("Computing threat measures...", f)

    # n_better = count of products with higher score than focal
    df['n_better'] = df['RANKING'] - 1  # Since RANKING 1 = best, products with better rank = RANKING - 1

    # threat_score = sum of (competitor_score - focal_score) for competitors scoring higher
    # This requires per-bid computation
    log("  Computing threat_score (this may take a moment)...", f)

    def compute_threat(group):
        scores = group['score'].values
        n = len(scores)
        threats = np.zeros(n)
        for i in range(n):
            focal = scores[i]
            better_scores = scores[scores > focal]
            threats[i] = (better_scores - focal).sum() if len(better_scores) > 0 else 0
        return pd.Series(threats, index=group.index)

    # Sample for speed if dataset is large
    if len(df) > 500000:
        log("  Using vectorized approximation for large dataset...", f)
        # Approximate: threat = (n_better) * (mean_score_above_focal - focal_score)
        # mean_score_above = (total_score - sum_of_scores_below) / n_better
        # This is complex, so use simpler approximation:
        # threat ≈ n_better * (max_competitor_score - focal_score) * 0.5
        df['threat_score'] = df['n_better'] * np.maximum(df['max_competitor_score'] - df['score'], 0) * 0.5
    else:
        threat_values = df.groupby('AUCTION_ID', group_keys=False).apply(compute_threat)
        df['threat_score'] = threat_values

    # closest_challenger = score gap to best product with worse ranking
    df['closest_challenger'] = df['gap_below']

    # -------------------------------------------------------------------------
    # QUALITY vs BID DECOMPOSITION (3)
    # -------------------------------------------------------------------------
    log("Computing quality vs bid decomposition...", f)

    # Correlation between quality and bid within auction
    def compute_corr_quality_bid(group):
        if len(group) < 3:
            return np.nan
        q = group['QUALITY'].values
        b = group['FINAL_BID'].values
        if q.std() == 0 or b.std() == 0:
            return np.nan
        return np.corrcoef(q, b)[0, 1]

    corr_by_auction = df.groupby('AUCTION_ID').apply(compute_corr_quality_bid)
    df['corr_quality_bid'] = df['AUCTION_ID'].map(corr_by_auction)

    # Variance decomposition: how much of score variance is due to quality vs bid
    # score = quality * bid * pacing
    # log(score) ≈ log(quality) + log(bid) + log(pacing)
    # var(log_score) = var(log_q) + var(log_b) + var(log_p) + covariances

    # Simplified: compute share of variance
    # quality_dominance = var(quality) / var(score) (after standardizing)
    df['var_score'] = df['std_score'] ** 2

    # Use auction-level variances
    df['quality_dominance'] = np.where(
        df['var_score'] > 0,
        df['var_quality'] / (df['var_quality'] + df['var_bid'] + 0.001),
        0.5
    )
    df['bid_dominance'] = 1 - df['quality_dominance']

    log("", f)
    log("Competition measures computed.", f)
    log("", f)

    return df


# =============================================================================
# SECTION 1: MEASURE SUMMARY STATISTICS
# =============================================================================
def section_1_summary_stats(df, f):
    """Summary statistics for all competition measures."""
    log("=" * 80, f)
    log("SECTION 1: COMPETITION MEASURE SUMMARY STATISTICS", f)
    log("=" * 80, f)
    log("", f)

    measures = [
        # Count-based
        ('n_bidders', 'Count: Total bidders in auction'),
        ('n_winners', 'Count: Total winners in auction'),
        ('winner_rate', 'Count: Winner rate (n_winners/n_bidders)'),
        # Score-based
        ('mean_competitor_score', 'Score: Mean competitor score'),
        ('max_competitor_score', 'Score: Max competitor score'),
        ('median_competitor_score', 'Score: Median competitor score'),
        ('std_score', 'Score: Std dev of scores in auction'),
        ('score_percentile', 'Score: Focal percentile (0=worst, 1=best)'),
        ('score_zscore', 'Score: Focal z-score in auction'),
        # Gap-based
        ('gap_above', 'Gap: Score gap to better product'),
        ('gap_below', 'Gap: Score gap to worse product'),
        ('gap_ratio', 'Gap: gap_above / gap_below'),
        ('local_density', 'Gap: 1/(gap_above + gap_below)'),
        # Concentration
        ('hhi_score', 'Conc: HHI of scores'),
        ('gini_score', 'Conc: Gini coefficient'),
        ('top3_share', 'Conc: Top 3 bidders share'),
        ('effective_competitors', 'Conc: 1/HHI'),
        # Threat
        ('n_better', 'Threat: Count scoring higher'),
        ('threat_score', 'Threat: Sum of score gaps above'),
        ('closest_challenger', 'Threat: Gap to next worse'),
        # Decomposition
        ('corr_quality_bid', 'Decomp: Corr(quality, bid) in auction'),
        ('quality_dominance', 'Decomp: Quality share of variance'),
        ('bid_dominance', 'Decomp: Bid share of variance'),
    ]

    log(f"{'Measure':<30} {'N':>12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}", f)
    log(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

    for measure, desc in measures:
        if measure in df.columns:
            vals = df[measure].dropna()
            n = len(vals)
            mean = vals.mean()
            std = vals.std()
            min_v = vals.min()
            max_v = vals.max()
            log(f"{measure:<30} {n:>12,} {mean:>12.4f} {std:>12.4f} {min_v:>12.4f} {max_v:>12.4f}", f)
        else:
            log(f"{measure:<30} NOT COMPUTED", f)

    log("", f)
    log("Measure descriptions:", f)
    for measure, desc in measures:
        log(f"  {measure}: {desc}", f)
    log("", f)


# =============================================================================
# SECTION 2: FIRST STAGE CORRELATIONS
# =============================================================================
def section_2_first_stage(df, f):
    """Correlations of competition measures with RANKING."""
    log("=" * 80, f)
    log("SECTION 2: FIRST STAGE - CORRELATION WITH RANKING", f)
    log("=" * 80, f)
    log("", f)

    log("For a valid instrument, we need strong correlation with RANKING.", f)
    log("", f)

    measures = [
        'n_bidders', 'n_winners', 'winner_rate',
        'mean_competitor_score', 'max_competitor_score', 'median_competitor_score',
        'std_score', 'score_percentile', 'score_zscore',
        'gap_above', 'gap_below', 'gap_ratio', 'local_density',
        'hhi_score', 'gini_score', 'top3_share', 'effective_competitors',
        'n_better', 'threat_score', 'closest_challenger',
        'corr_quality_bid', 'quality_dominance', 'bid_dominance'
    ]

    log(f"{'Measure':<30} {'Corr(M,Rank)':>15} {'p-value':>12} {'Partial|Q':>15} {'Partial|Q,B':>15}", f)
    log(f"{'-'*30} {'-'*15} {'-'*12} {'-'*15} {'-'*15}", f)

    # Prepare data for partial correlations
    df_clean = df[['RANKING', 'QUALITY', 'FINAL_BID'] + measures].dropna(subset=['RANKING', 'QUALITY', 'FINAL_BID'])

    for measure in measures:
        if measure not in df.columns:
            continue

        # Simple correlation
        valid = df[[measure, 'RANKING']].dropna()
        if len(valid) < 100:
            log(f"{measure:<30} Insufficient data", f)
            continue

        corr, pval = stats.pearsonr(valid[measure], valid['RANKING'])

        # Partial correlation controlling for QUALITY
        try:
            # Residualize both on QUALITY
            valid_q = df_clean[[measure, 'RANKING', 'QUALITY']].dropna()
            resid_m_q = valid_q[measure] - valid_q['QUALITY'] * np.cov(valid_q[measure], valid_q['QUALITY'])[0,1] / valid_q['QUALITY'].var()
            resid_r_q = valid_q['RANKING'] - valid_q['QUALITY'] * np.cov(valid_q['RANKING'], valid_q['QUALITY'])[0,1] / valid_q['QUALITY'].var()
            partial_q = np.corrcoef(resid_m_q, resid_r_q)[0,1]
        except:
            partial_q = np.nan

        # Partial correlation controlling for QUALITY and FINAL_BID
        try:
            from sklearn.linear_model import LinearRegression
            valid_qb = df_clean[[measure, 'RANKING', 'QUALITY', 'FINAL_BID']].dropna()
            X = valid_qb[['QUALITY', 'FINAL_BID']].values

            lr = LinearRegression()
            lr.fit(X, valid_qb[measure])
            resid_m = valid_qb[measure] - lr.predict(X)

            lr.fit(X, valid_qb['RANKING'])
            resid_r = valid_qb['RANKING'] - lr.predict(X)

            partial_qb = np.corrcoef(resid_m, resid_r)[0,1]
        except:
            partial_qb = np.nan

        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
        log(f"{measure:<30} {corr:>15.4f}{sig:<3} {pval:>12.2e} {partial_q:>15.4f} {partial_qb:>15.4f}", f)

    log("", f)
    log("Interpretation:", f)
    log("  Strong instruments: |Corr| > 0.3, significant p-value", f)
    log("  Partial|Q: Correlation after controlling for QUALITY", f)
    log("  Partial|Q,B: Correlation after controlling for QUALITY and BID (should be ~0)", f)
    log("", f)


# =============================================================================
# SECTION 3: EXCLUSION CHECK - CLICK CORRELATION
# =============================================================================
def section_3_exclusion_check(df, f):
    """Correlations with click outcome, conditioning on RANKING."""
    log("=" * 80, f)
    log("SECTION 3: EXCLUSION CHECK - CORRELATION WITH CLICK", f)
    log("=" * 80, f)
    log("", f)

    log("For exclusion restriction, we need measures to NOT correlate with clicks", f)
    log("AFTER controlling for RANKING (the endogenous variable).", f)
    log("", f)

    if 'clicked' not in df.columns:
        log("ERROR: clicked column not found. Skipping exclusion check.", f)
        log("", f)
        return

    measures = [
        'n_bidders', 'n_winners', 'winner_rate',
        'mean_competitor_score', 'max_competitor_score', 'std_score',
        'score_percentile', 'score_zscore',
        'gap_above', 'gap_below', 'gap_ratio', 'local_density',
        'hhi_score', 'gini_score', 'top3_share', 'effective_competitors',
        'n_better', 'threat_score', 'closest_challenger',
        'corr_quality_bid', 'quality_dominance'
    ]

    # Only winners have clicks
    winners = df[df['IS_WINNER'] == True].copy()
    log(f"Analyzing {len(winners):,} winners ({winners['clicked'].sum():,} clicks)", f)
    log("", f)

    log(f"{'Measure':<30} {'Corr(M,Click)':>15} {'p-value':>12} {'Partial|Rank':>15}", f)
    log(f"{'-'*30} {'-'*15} {'-'*12} {'-'*15}", f)

    for measure in measures:
        if measure not in winners.columns:
            continue

        # Simple correlation with click
        valid = winners[[measure, 'clicked', 'RANKING']].dropna()
        if len(valid) < 100:
            log(f"{measure:<30} Insufficient data", f)
            continue

        corr, pval = stats.pearsonr(valid[measure], valid['clicked'])

        # Partial correlation controlling for RANKING
        try:
            from sklearn.linear_model import LinearRegression
            X = valid[['RANKING']].values

            lr = LinearRegression()
            lr.fit(X, valid[measure])
            resid_m = valid[measure] - lr.predict(X)

            lr.fit(X, valid['clicked'])
            resid_c = valid['clicked'] - lr.predict(X)

            partial_r = np.corrcoef(resid_m, resid_c)[0,1]
        except:
            partial_r = np.nan

        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
        excl = 'GOOD' if abs(partial_r) < 0.05 else ('WEAK' if abs(partial_r) < 0.10 else 'BAD')
        log(f"{measure:<30} {corr:>15.4f}{sig:<3} {pval:>12.2e} {partial_r:>15.4f} [{excl}]", f)

    log("", f)
    log("Interpretation:", f)
    log("  GOOD: |Partial|Rank| < 0.05 - measure doesn't directly affect clicks", f)
    log("  WEAK: |Partial|Rank| < 0.10 - borderline exclusion", f)
    log("  BAD: |Partial|Rank| >= 0.10 - measure directly affects clicks, invalid instrument", f)
    log("", f)


# =============================================================================
# SECTION 4: INDEPENDENCE CHECKS
# =============================================================================
def section_4_independence(df, f):
    """Correlations with focal product characteristics."""
    log("=" * 80, f)
    log("SECTION 4: INDEPENDENCE - CORRELATION WITH FOCAL CHARACTERISTICS", f)
    log("=" * 80, f)
    log("", f)

    log("Good instruments should be independent of focal product quality/bid.", f)
    log("High correlation suggests the instrument is not exogenous.", f)
    log("", f)

    measures = [
        'n_bidders', 'n_winners', 'winner_rate',
        'mean_competitor_score', 'max_competitor_score', 'std_score',
        'gap_above', 'gap_below', 'local_density',
        'hhi_score', 'gini_score', 'effective_competitors',
        'n_better', 'threat_score',
        'corr_quality_bid', 'quality_dominance'
    ]

    log(f"{'Measure':<30} {'Corr(M,QUALITY)':>18} {'Corr(M,FINAL_BID)':>18} {'Corr(M,PLACEMENT)':>18}", f)
    log(f"{'-'*30} {'-'*18} {'-'*18} {'-'*18}", f)

    # Add PLACEMENT as numeric if available
    if 'PLACEMENT' in df.columns:
        df['PLACEMENT_num'] = pd.to_numeric(df['PLACEMENT'], errors='coerce')
    else:
        df['PLACEMENT_num'] = np.nan

    for measure in measures:
        if measure not in df.columns:
            continue

        valid = df[[measure, 'QUALITY', 'FINAL_BID', 'PLACEMENT_num']].dropna(subset=[measure, 'QUALITY', 'FINAL_BID'])
        if len(valid) < 100:
            log(f"{measure:<30} Insufficient data", f)
            continue

        corr_q, _ = stats.pearsonr(valid[measure], valid['QUALITY'])
        corr_b, _ = stats.pearsonr(valid[measure], valid['FINAL_BID'])

        if valid['PLACEMENT_num'].notna().sum() > 100:
            valid_p = valid.dropna(subset=['PLACEMENT_num'])
            corr_p, _ = stats.pearsonr(valid_p[measure], valid_p['PLACEMENT_num'])
        else:
            corr_p = np.nan

        indep = 'GOOD' if (abs(corr_q) < 0.3 and abs(corr_b) < 0.3) else 'WEAK'
        log(f"{measure:<30} {corr_q:>18.4f} {corr_b:>18.4f} {corr_p:>18.4f} [{indep}]", f)

    log("", f)
    log("Interpretation:", f)
    log("  GOOD: |Corr| < 0.3 for both QUALITY and FINAL_BID", f)
    log("  WEAK: Higher correlations suggest endogeneity concerns", f)
    log("", f)


# =============================================================================
# SECTION 5: WITHIN-PRODUCT VARIATION
# =============================================================================
def section_5_within_product(df, f):
    """Analyze within-product variation in competition measures."""
    log("=" * 80, f)
    log("SECTION 5: WITHIN-PRODUCT VARIATION", f)
    log("=" * 80, f)
    log("", f)

    log("Products appearing in multiple auctions provide within-product variation.", f)
    log("This helps isolate competition effects from product characteristics.", f)
    log("", f)

    # Count products by auction frequency
    product_counts = df.groupby('PRODUCT_ID').size()

    log("Product auction frequency:", f)
    log(f"  1 auction: {(product_counts == 1).sum():,} products", f)
    log(f"  2-4 auctions: {((product_counts >= 2) & (product_counts <= 4)).sum():,} products", f)
    log(f"  5-9 auctions: {((product_counts >= 5) & (product_counts <= 9)).sum():,} products", f)
    log(f"  10+ auctions: {(product_counts >= 10).sum():,} products", f)
    log("", f)

    # Filter to products with 5+ auctions
    multi_products = product_counts[product_counts >= 5].index
    df_multi = df[df['PRODUCT_ID'].isin(multi_products)].copy()

    log(f"Analyzing {len(multi_products):,} products with 5+ auctions ({len(df_multi):,} bids)", f)
    log("", f)

    if len(df_multi) < 1000:
        log("Insufficient data for within-product analysis.", f)
        log("", f)
        return

    measures = [
        'n_bidders', 'n_winners',
        'mean_competitor_score', 'max_competitor_score', 'std_score',
        'gap_above', 'gap_below', 'local_density',
        'hhi_score', 'effective_competitors',
        'n_better', 'threat_score'
    ]

    log(f"{'Measure':<30} {'Within-Prod Var':>18} {'Across-Prod Var':>18} {'Within Corr(M,Rank)':>20}", f)
    log(f"{'-'*30} {'-'*18} {'-'*18} {'-'*20}", f)

    for measure in measures:
        if measure not in df_multi.columns:
            continue

        valid = df_multi[[measure, 'RANKING', 'PRODUCT_ID']].dropna()
        if len(valid) < 500:
            log(f"{measure:<30} Insufficient data", f)
            continue

        # Within-product variance (demeaned)
        product_means = valid.groupby('PRODUCT_ID')[measure].transform('mean')
        within_var = (valid[measure] - product_means).var()

        # Across-product variance
        across_var = product_means.var()

        # Within-product correlation with RANKING
        # Demean both by product
        rank_means = valid.groupby('PRODUCT_ID')['RANKING'].transform('mean')
        demeaned_m = valid[measure] - product_means
        demeaned_r = valid['RANKING'] - rank_means

        within_corr = np.corrcoef(demeaned_m, demeaned_r)[0,1]

        log(f"{measure:<30} {within_var:>18.4f} {across_var:>18.4f} {within_corr:>20.4f}", f)

    log("", f)
    log("Interpretation:", f)
    log("  High within-product variance: Measure varies across auctions for same product", f)
    log("  Within Corr(M,Rank): Correlation after removing product fixed effects", f)
    log("  Strong within-product correlation suggests measure can identify position effects", f)
    log("", f)


# =============================================================================
# SECTION 6: COMPETITION BY PLACEMENT
# =============================================================================
def section_6_by_placement(df, f):
    """Stratify competition measures by PLACEMENT."""
    log("=" * 80, f)
    log("SECTION 6: COMPETITION BY PLACEMENT", f)
    log("=" * 80, f)
    log("", f)

    if 'PLACEMENT' not in df.columns:
        log("PLACEMENT not available. Skipping.", f)
        log("", f)
        return

    placements = sorted(df['PLACEMENT'].unique())
    log(f"Placements: {placements}", f)
    log("", f)

    measures = ['n_bidders', 'n_winners', 'mean_competitor_score', 'hhi_score', 'effective_competitors']

    # Build header row
    header = f"{'Placement':<12} {'N':>12}"
    for m in measures:
        header += f" {m[:15]:>15}"
    log(header, f)

    # Build separator
    sep = f"{'-'*12} {'-'*12}"
    for _ in measures:
        sep += f" {'-'*15}"
    log(sep, f)

    for placement in placements:
        subset = df[df['PLACEMENT'] == placement]
        n = len(subset)

        row = f"{str(placement):<12} {n:>12,}"
        for m in measures:
            if m in subset.columns:
                val = subset[m].mean()
                row += f" {val:>15.4f}"
            else:
                row += f" {'--':>15}"
        log(row, f)

    log("", f)

    # First stage by placement
    log("First stage correlation (Corr with RANKING) by placement:", f)
    log("", f)

    key_measures = ['n_bidders', 'mean_competitor_score', 'hhi_score', 'n_better']

    # Build header
    header = f"{'Placement':<12}"
    for m in key_measures:
        header += f" {m[:15]:>15}"
    log(header, f)

    # Build separator
    sep = f"{'-'*12}"
    for _ in key_measures:
        sep += f" {'-'*15}"
    log(sep, f)

    for placement in placements:
        subset = df[df['PLACEMENT'] == placement]

        row = f"{str(placement):<12}"
        for m in key_measures:
            if m in subset.columns:
                valid = subset[[m, 'RANKING']].dropna()
                if len(valid) > 100:
                    corr, _ = stats.pearsonr(valid[m], valid['RANKING'])
                    row += f" {corr:>15.4f}"
                else:
                    row += f" {'--':>15}"
            else:
                row += f" {'--':>15}"
        log(row, f)

    log("", f)


# =============================================================================
# SECTION 7: BEST INSTRUMENT SELECTION
# =============================================================================
def section_7_instrument_ranking(df, f):
    """Rank measures by instrument quality criteria."""
    log("=" * 80, f)
    log("SECTION 7: INSTRUMENT QUALITY RANKING", f)
    log("=" * 80, f)
    log("", f)

    log("Ranking measures by:", f)
    log("  1. First stage strength: |Corr(M, RANKING)|", f)
    log("  2. Exclusion plausibility: |Corr(M, Click | RANKING)| (lower is better)", f)
    log("  3. Independence: |Corr(M, QUALITY)| (lower is better)", f)
    log("", f)

    measures = [
        'n_bidders', 'n_winners', 'winner_rate',
        'mean_competitor_score', 'max_competitor_score', 'std_score',
        'score_percentile', 'score_zscore',
        'gap_above', 'gap_below', 'gap_ratio', 'local_density',
        'hhi_score', 'gini_score', 'top3_share', 'effective_competitors',
        'n_better', 'threat_score', 'closest_challenger',
        'corr_quality_bid', 'quality_dominance'
    ]

    # Only winners for exclusion check
    winners = df[df['IS_WINNER'] == True].copy() if 'clicked' in df.columns else None

    results = []

    for measure in tqdm(measures, desc="Evaluating measures"):
        if measure not in df.columns:
            continue

        # First stage
        valid = df[[measure, 'RANKING', 'QUALITY']].dropna()
        if len(valid) < 1000:
            continue

        corr_rank, _ = stats.pearsonr(valid[measure], valid['RANKING'])
        corr_quality, _ = stats.pearsonr(valid[measure], valid['QUALITY'])

        # Exclusion (if click data available)
        if winners is not None and 'clicked' in winners.columns:
            valid_w = winners[[measure, 'clicked', 'RANKING']].dropna()
            if len(valid_w) > 100:
                # Partial correlation with click | RANKING
                try:
                    from sklearn.linear_model import LinearRegression
                    X = valid_w[['RANKING']].values

                    lr = LinearRegression()
                    lr.fit(X, valid_w[measure])
                    resid_m = valid_w[measure] - lr.predict(X)

                    lr.fit(X, valid_w['clicked'])
                    resid_c = valid_w['clicked'] - lr.predict(X)

                    partial_click = np.corrcoef(resid_m, resid_c)[0,1]
                except:
                    partial_click = np.nan
            else:
                partial_click = np.nan
        else:
            partial_click = np.nan

        results.append({
            'measure': measure,
            'first_stage': abs(corr_rank),
            'exclusion': abs(partial_click) if not np.isnan(partial_click) else 1.0,
            'independence': abs(corr_quality),
            'corr_rank': corr_rank,
            'corr_quality': corr_quality,
            'partial_click': partial_click
        })

    results_df = pd.DataFrame(results)

    # Score: high first stage, low exclusion, low independence
    # Normalize each to 0-1 scale
    results_df['first_stage_norm'] = results_df['first_stage'] / results_df['first_stage'].max()
    results_df['exclusion_norm'] = 1 - (results_df['exclusion'] / results_df['exclusion'].max())
    results_df['independence_norm'] = 1 - (results_df['independence'] / results_df['independence'].max())

    # Composite score (equal weights)
    results_df['score'] = (results_df['first_stage_norm'] +
                           results_df['exclusion_norm'] +
                           results_df['independence_norm']) / 3

    results_df = results_df.sort_values('score', ascending=False)

    log(f"{'Rank':>4} {'Measure':<25} {'FirstStage':>12} {'Exclusion':>12} {'Independence':>12} {'Score':>10}", f)
    log(f"{'-'*4} {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}", f)

    for i, row in enumerate(results_df.itertuples(), 1):
        log(f"{i:>4} {row.measure:<25} {row.first_stage:>12.4f} {row.exclusion:>12.4f} {row.independence:>12.4f} {row.score:>10.4f}", f)

    log("", f)
    log("TOP INSTRUMENT CANDIDATES:", f)
    log("", f)

    for i, row in enumerate(results_df.head(5).itertuples(), 1):
        log(f"{i}. {row.measure}", f)
        log(f"   First stage: Corr(M, RANKING) = {row.corr_rank:.4f}", f)
        log(f"   Exclusion: Partial Corr(M, Click | RANKING) = {row.partial_click:.4f}", f)
        log(f"   Independence: Corr(M, QUALITY) = {row.corr_quality:.4f}", f)
        log("", f)

    log("INTERPRETATION:", f)
    log("  Good instruments have:", f)
    log("    - Strong first stage (|Corr with RANKING| > 0.3)", f)
    log("    - Low exclusion violation (|Partial with Click| < 0.05)", f)
    log("    - Low correlation with focal quality (|Corr with QUALITY| < 0.3)", f)
    log("", f)

    return results_df


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("AUCTION PRESSURE ANALYSIS", f)
        log("Competition Measures as Potential Instrumental Variables", f)
        log("=" * 80, f)
        log("", f)

        log("OBJECTIVE:", f)
        log("  Identify competition measures that can serve as instruments for RANKING", f)
        log("  in estimating causal position effects on click-through rates.", f)
        log("", f)

        log("INSTRUMENT REQUIREMENTS:", f)
        log("  1. Relevance: Strong correlation with RANKING (first stage)", f)
        log("  2. Exclusion: No direct effect on clicks conditional on RANKING", f)
        log("  3. Independence: Uncorrelated with focal product quality", f)
        log("", f)

        # Load data
        log("-" * 40, f)
        log("LOADING DATA", f)
        log("-" * 40, f)

        ar_path = DATA_DIR / "auctions_results_r2.parquet"
        clicks_path = DATA_DIR / "clicks_r2.parquet"
        au_path = DATA_DIR / "auctions_users_r2.parquet"

        log(f"  Loading {ar_path}...", f)
        ar = pd.read_parquet(ar_path)
        log(f"    {len(ar):,} bids in {ar['AUCTION_ID'].nunique():,} auctions", f)

        log(f"  Loading {clicks_path}...", f)
        clicks = pd.read_parquet(clicks_path)
        log(f"    {len(clicks):,} clicks", f)

        log(f"  Loading {au_path}...", f)
        au = pd.read_parquet(au_path)
        log(f"    {len(au):,} auction-user records", f)
        log("", f)

        # Merge placement
        log("Merging PLACEMENT from auctions_users...", f)
        placement_map = au[['AUCTION_ID', 'PLACEMENT']].drop_duplicates()
        ar = ar.merge(placement_map, on='AUCTION_ID', how='left')
        log(f"  Matched placements: {ar['PLACEMENT'].notna().sum():,} / {len(ar):,}", f)
        log("", f)

        # Create click indicator for winners
        log("Creating click indicator for winners...", f)
        clicks_dedup = clicks[['AUCTION_ID', 'PRODUCT_ID']].drop_duplicates()
        clicks_dedup['clicked'] = 1
        ar = ar.merge(clicks_dedup, on=['AUCTION_ID', 'PRODUCT_ID'], how='left')
        ar['clicked'] = ar['clicked'].fillna(0).astype(int)

        winners = ar[ar['IS_WINNER'] == True]
        log(f"  Winners: {len(winners):,}", f)
        log(f"  Clicked: {winners['clicked'].sum():,} ({winners['clicked'].mean()*100:.3f}%)", f)
        log("", f)

        # Compute competition measures
        df = compute_competition_measures(ar, f)

        # Run all sections
        section_1_summary_stats(df, f)
        section_2_first_stage(df, f)
        section_3_exclusion_check(df, f)
        section_4_independence(df, f)
        section_5_within_product(df, f)
        section_6_by_placement(df, f)
        results = section_7_instrument_ranking(df, f)

        # Final summary
        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)
        log("", f)
        log(f"Output saved to: {OUTPUT_FILE}", f)
        log("", f)


if __name__ == "__main__":
    main()
