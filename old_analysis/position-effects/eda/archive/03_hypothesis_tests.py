#!/usr/bin/env python3
"""
Position Effects EDA Round 2: Hypothesis Testing

Tests specific hypotheses about ranking mechanics, funnel logic, and quality scores
based on contradictions discovered in first EDA (R² = 0.01 for QUALITY × BID → RANKING).

Hypotheses tested:
  A1: Within-auction exact rank match (predicted vs actual)
  A2: Does rank-1 have max score in its auction?
  B2: Is QUALITY a good pCTR predictor?
  B3: Is QUALITY calibrated?
  C3: Does FINAL_BID ≈ CONVERSION_RATE × PRICE?
  D2: Is PACING campaign-level?
  D5: Does QUALITY × FINAL_BID × PACING fix R²?
  D6: Is PACING already baked into FINAL_BID?
  F2: Winners per auction distribution
  F3: Impressions per auction distribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "03_hypothesis_tests.txt"

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
        log("POSITION EFFECTS EDA ROUND 2: HYPOTHESIS TESTING", f)
        log("=" * 80, f)
        log("", f)

        log("CONTEXT:", f)
        log("  First EDA revealed contradictions:", f)
        log("  - QUALITY × FINAL_BID explains only 1% of RANKING variance (R² = 0.01)", f)
        log("  - This contradicts claimed ranking mechanics", f)
        log("", f)

        log("OBJECTIVE:", f)
        log("  Test specific hypotheses about ranking formula, funnel logic,", f)
        log("  quality score meaning, and pacing mechanism.", f)
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

        log("Loading data files...", f)

        ar = pd.read_parquet(ar_path)
        log(f"  auctions_results: {len(ar):,} rows", f)

        au = pd.read_parquet(au_path)
        log(f"  auctions_users: {len(au):,} rows", f)

        imp = pd.read_parquet(imp_path)
        log(f"  impressions: {len(imp):,} rows", f)

        clicks = pd.read_parquet(clicks_path)
        log(f"  clicks: {len(clicks):,} rows", f)

        log("", f)

        # Merge ar with au for PLACEMENT
        ar = ar.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')
        log(f"Merged PLACEMENT into auctions_results: {ar['PLACEMENT'].notna().sum():,} have PLACEMENT", f)
        log("", f)

        # Filter to complete rows for scoring tests
        ar_complete = ar[['AUCTION_ID', 'PRODUCT_ID', 'RANKING', 'IS_WINNER', 'FINAL_BID',
                          'QUALITY', 'PACING', 'CONVERSION_RATE', 'PRICE', 'CAMPAIGN_ID', 'PLACEMENT']].dropna(
            subset=['FINAL_BID', 'QUALITY', 'PACING', 'RANKING']
        )
        log(f"Rows with complete scoring features: {len(ar_complete):,} / {len(ar):,} ({len(ar_complete)/len(ar)*100:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Ranking Formula Tests (A1, A2, D5, D6)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: RANKING FORMULA TESTS (A1, A2, D5, D6)", f)
        log("-" * 40, f)
        log("", f)

        # Compute scores
        ar_complete['score_v1'] = ar_complete['QUALITY'] * ar_complete['FINAL_BID']
        ar_complete['score_v2'] = ar_complete['QUALITY'] * ar_complete['FINAL_BID'] * ar_complete['PACING']

        # ---------------------------
        # TEST A1: Within-auction exact rank match
        # ---------------------------
        log("TEST A1: WITHIN-AUCTION EXACT RANK MATCH", f)
        log("  Question: If we sort by QUALITY × FINAL_BID desc, do we recover RANKING?", f)
        log("", f)

        # For each auction, predict rank by sorting score_v1 desc
        log("  Computing predicted ranks within each auction...", f)

        def compute_predicted_rank(group):
            group = group.copy()
            group['predicted_rank_v1'] = group['score_v1'].rank(ascending=False, method='min')
            group['predicted_rank_v2'] = group['score_v2'].rank(ascending=False, method='min')
            return group

        ar_with_pred = ar_complete.groupby('AUCTION_ID', group_keys=False).apply(compute_predicted_rank)

        # Exact match rate
        exact_match_v1 = (ar_with_pred['RANKING'] == ar_with_pred['predicted_rank_v1']).mean()
        exact_match_v2 = (ar_with_pred['RANKING'] == ar_with_pred['predicted_rank_v2']).mean()

        # Mean absolute error
        mae_v1 = (ar_with_pred['RANKING'] - ar_with_pred['predicted_rank_v1']).abs().mean()
        mae_v2 = (ar_with_pred['RANKING'] - ar_with_pred['predicted_rank_v2']).abs().mean()

        log(f"  Using score = QUALITY × FINAL_BID:", f)
        log(f"    Exact rank match rate: {exact_match_v1*100:.2f}%", f)
        log(f"    Mean absolute error: {mae_v1:.2f} ranks", f)
        log("", f)

        log(f"  Using score = QUALITY × FINAL_BID × PACING:", f)
        log(f"    Exact rank match rate: {exact_match_v2*100:.2f}%", f)
        log(f"    Mean absolute error: {mae_v2:.2f} ranks", f)
        log("", f)

        log(f"  INTERPRETATION:", f)
        if exact_match_v1 > 0.9:
            log(f"    >90% exact match: RANKING = f(QUALITY × BID) is deterministic", f)
        elif exact_match_v1 > 0.5:
            log(f"    50-90% exact match: Ranking partially determined by score", f)
        else:
            log(f"    <50% exact match: Other factors dominate ranking", f)
        log("", f)

        # ---------------------------
        # TEST A2: Does rank-1 have max score?
        # ---------------------------
        log("TEST A2: DOES RANK-1 HAVE MAX SCORE IN ITS AUCTION?", f)
        log("  Question: Is the highest-ranked bid always the highest score?", f)
        log("", f)

        # Get max score per auction
        auction_max_v1 = ar_with_pred.groupby('AUCTION_ID')['score_v1'].max().reset_index()
        auction_max_v1.columns = ['AUCTION_ID', 'max_score_v1']

        auction_max_v2 = ar_with_pred.groupby('AUCTION_ID')['score_v2'].max().reset_index()
        auction_max_v2.columns = ['AUCTION_ID', 'max_score_v2']

        # Get rank-1 rows
        rank1 = ar_with_pred[ar_with_pred['RANKING'] == 1].copy()
        rank1 = rank1.merge(auction_max_v1, on='AUCTION_ID', how='left')
        rank1 = rank1.merge(auction_max_v2, on='AUCTION_ID', how='left')

        # Check if rank-1 has max score
        rank1_has_max_v1 = (rank1['score_v1'] == rank1['max_score_v1']).mean()
        rank1_has_max_v2 = (rank1['score_v2'] == rank1['max_score_v2']).mean()

        log(f"  Using score = QUALITY × FINAL_BID:", f)
        log(f"    Rank-1 has max score: {rank1_has_max_v1*100:.2f}% of auctions", f)
        log("", f)

        log(f"  Using score = QUALITY × FINAL_BID × PACING:", f)
        log(f"    Rank-1 has max score: {rank1_has_max_v2*100:.2f}% of auctions", f)
        log("", f)

        # ---------------------------
        # TEST D5: Does PACING improve R²?
        # ---------------------------
        log("TEST D5: DOES ADDING PACING IMPROVE R²?", f)
        log("  Question: Does QUALITY × FINAL_BID × PACING explain more variance?", f)
        log("", f)

        # Sample for computational efficiency
        sample_auctions = ar_complete['AUCTION_ID'].drop_duplicates().sample(
            min(20000, ar_complete['AUCTION_ID'].nunique()), random_state=42
        )
        ar_sample = ar_complete[ar_complete['AUCTION_ID'].isin(sample_auctions)].copy()

        # Within-auction R²
        # Normalize scores within auction for proper R² computation
        def compute_r2_within_auction(group, score_col):
            if len(group) < 2:
                return np.nan
            y = group['RANKING'].values
            x = group[score_col].values
            if np.std(x) == 0:
                return np.nan
            # Correlation-based R² (rank vs score)
            corr = np.corrcoef(x, y)[0, 1]
            return corr ** 2

        r2_within_v1 = ar_sample.groupby('AUCTION_ID').apply(
            lambda g: compute_r2_within_auction(g, 'score_v1')
        ).dropna()

        r2_within_v2 = ar_sample.groupby('AUCTION_ID').apply(
            lambda g: compute_r2_within_auction(g, 'score_v2')
        ).dropna()

        log(f"  Within-auction R² (score vs RANKING):", f)
        log(f"  Sample: {len(r2_within_v1):,} auctions", f)
        log("", f)

        log(f"  score_v1 = QUALITY × FINAL_BID:", f)
        log(f"    Mean R²: {r2_within_v1.mean():.4f}", f)
        log(f"    Median R²: {r2_within_v1.median():.4f}", f)
        log(f"    P25: {r2_within_v1.quantile(0.25):.4f}", f)
        log(f"    P75: {r2_within_v1.quantile(0.75):.4f}", f)
        log("", f)

        log(f"  score_v2 = QUALITY × FINAL_BID × PACING:", f)
        log(f"    Mean R²: {r2_within_v2.mean():.4f}", f)
        log(f"    Median R²: {r2_within_v2.median():.4f}", f)
        log(f"    P25: {r2_within_v2.quantile(0.25):.4f}", f)
        log(f"    P75: {r2_within_v2.quantile(0.75):.4f}", f)
        log("", f)

        r2_improvement = r2_within_v2.mean() - r2_within_v1.mean()
        log(f"  R² improvement from adding PACING: {r2_improvement:.4f}", f)
        log("", f)

        # ---------------------------
        # TEST D6: Is PACING redundant?
        # ---------------------------
        log("TEST D6: IS PACING ALREADY BAKED INTO FINAL_BID?", f)
        log("  Question: Is CORR(FINAL_BID, FINAL_BID × PACING) ≈ 1?", f)
        log("", f)

        corr_bid_pacingbid = ar_complete['FINAL_BID'].corr(ar_complete['FINAL_BID'] * ar_complete['PACING'])
        corr_bid_pacing = ar_complete['FINAL_BID'].corr(ar_complete['PACING'])

        log(f"  CORR(FINAL_BID, FINAL_BID × PACING): {corr_bid_pacingbid:.6f}", f)
        log(f"  CORR(FINAL_BID, PACING): {corr_bid_pacing:.6f}", f)
        log("", f)

        if corr_bid_pacingbid > 0.99:
            log(f"  INTERPRETATION: PACING is redundant (correlation > 0.99)", f)
        else:
            log(f"  INTERPRETATION: PACING adds information beyond FINAL_BID", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Funnel Verification (F2, F3)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: FUNNEL VERIFICATION (F2, F3)", f)
        log("-" * 40, f)
        log("", f)

        # ---------------------------
        # TEST F2: Winners per auction
        # ---------------------------
        log("TEST F2: WINNERS PER AUCTION", f)
        log("  Question: How many bids win (IS_WINNER=True) per auction?", f)
        log("", f)

        winners_per_auction = ar[ar['IS_WINNER'] == True].groupby('AUCTION_ID').size()

        log(f"  N auctions with winners: {len(winners_per_auction):,}", f)
        log(f"  Winners per auction:", f)
        log(f"    Mean: {winners_per_auction.mean():.2f}", f)
        log(f"    Median: {winners_per_auction.median():.0f}", f)
        log(f"    P25: {winners_per_auction.quantile(0.25):.0f}", f)
        log(f"    P75: {winners_per_auction.quantile(0.75):.0f}", f)
        log(f"    Max: {winners_per_auction.max():.0f}", f)
        log("", f)

        # Distribution
        log(f"  Distribution of winner counts:", f)
        winner_dist = winners_per_auction.value_counts().sort_index().head(20)
        for n_winners, count in winner_dist.items():
            pct = count / len(winners_per_auction) * 100
            log(f"    {n_winners} winners: {count:,} auctions ({pct:.1f}%)", f)
        log("", f)

        # ---------------------------
        # TEST F3: Impressions per auction
        # ---------------------------
        log("TEST F3: IMPRESSIONS PER AUCTION", f)
        log("  Question: How many impressions per auction?", f)
        log("", f)

        imp_per_auction = imp.groupby('AUCTION_ID').size()

        log(f"  N auctions with impressions: {len(imp_per_auction):,}", f)
        log(f"  Impressions per auction:", f)
        log(f"    Mean: {imp_per_auction.mean():.2f}", f)
        log(f"    Median: {imp_per_auction.median():.0f}", f)
        log(f"    P25: {imp_per_auction.quantile(0.25):.0f}", f)
        log(f"    P75: {imp_per_auction.quantile(0.75):.0f}", f)
        log(f"    Max: {imp_per_auction.max():.0f}", f)
        log("", f)

        # Distribution
        log(f"  Distribution of impression counts:", f)
        imp_dist = imp_per_auction.value_counts().sort_index().head(20)
        for n_imp, count in imp_dist.items():
            pct = count / len(imp_per_auction) * 100
            log(f"    {n_imp} impressions: {count:,} auctions ({pct:.1f}%)", f)
        log("", f)

        # Compare winners vs impressions
        log("  COMPARISON: Winners vs Impressions per auction", f)
        # Merge
        comparison = pd.DataFrame({
            'winners': winners_per_auction,
            'impressions': imp_per_auction
        }).dropna()

        if len(comparison) > 0:
            log(f"    N auctions with both: {len(comparison):,}", f)
            log(f"    CORR(winners, impressions): {comparison['winners'].corr(comparison['impressions']):.4f}", f)
            log(f"    Mean winners: {comparison['winners'].mean():.2f}", f)
            log(f"    Mean impressions: {comparison['impressions'].mean():.2f}", f)
            log(f"    Ratio (imp/winners): {comparison['impressions'].mean() / comparison['winners'].mean():.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Quality Score Validation (B2, B3)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: QUALITY SCORE VALIDATION (B2, B3)", f)
        log("-" * 40, f)
        log("", f)

        # Merge winners with clicks
        winners = ar[ar['IS_WINNER'] == True].copy()
        click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
        winners['clicked'] = winners.apply(
            lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
        )

        log(f"  Winners: {len(winners):,}", f)
        log(f"  Clicked: {winners['clicked'].sum():,} ({winners['clicked'].mean()*100:.2f}%)", f)
        log("", f)

        # ---------------------------
        # TEST B2: Is QUALITY a good pCTR?
        # ---------------------------
        log("TEST B2: IS QUALITY A GOOD PCTR PREDICTOR?", f)
        log("  Question: Does QUALITY predict clicks (AUC)?", f)
        log("", f)

        winners_with_quality = winners[winners['QUALITY'].notna()].copy()
        log(f"  Winners with QUALITY: {len(winners_with_quality):,}", f)

        if winners_with_quality['clicked'].sum() >= 10:
            y_true = winners_with_quality['clicked'].astype(int)
            y_score = winners_with_quality['QUALITY']

            auc = roc_auc_score(y_true, y_score)
            log(f"  AUC of QUALITY predicting click: {auc:.4f}", f)
            log("", f)

            if auc > 0.7:
                log(f"  INTERPRETATION: QUALITY is a good pCTR (AUC > 0.7)", f)
            elif auc > 0.55:
                log(f"  INTERPRETATION: QUALITY has weak predictive power (0.55 < AUC < 0.7)", f)
            else:
                log(f"  INTERPRETATION: QUALITY is not a useful pCTR (AUC ≈ 0.5)", f)
        else:
            log(f"  Insufficient clicks for AUC computation", f)
        log("", f)

        # ---------------------------
        # TEST B3: Is QUALITY calibrated?
        # ---------------------------
        log("TEST B3: IS QUALITY CALIBRATED?", f)
        log("  Question: Does QUALITY match actual CTR across deciles?", f)
        log("", f)

        if len(winners_with_quality) > 1000:
            winners_with_quality['quality_decile'] = pd.qcut(
                winners_with_quality['QUALITY'], q=10, labels=False, duplicates='drop'
            )

            calibration = winners_with_quality.groupby('quality_decile').agg({
                'QUALITY': 'mean',
                'clicked': ['sum', 'count', 'mean']
            }).reset_index()
            calibration.columns = ['decile', 'mean_quality', 'clicks', 'n', 'actual_ctr']

            log(f"  Calibration table:", f)
            log(f"  {'Decile':<10} {'Mean QUALITY':<15} {'N':<12} {'Clicks':<10} {'Actual CTR':<12}", f)
            log(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*10} {'-'*12}", f)

            for _, row in calibration.iterrows():
                log(f"  {int(row['decile']):<10} {row['mean_quality']:<15.6f} {int(row['n']):<12,} {int(row['clicks']):<10,} {row['actual_ctr']*100:<12.4f}%", f)

            log("", f)

            # Check monotonicity
            ctr_values = calibration['actual_ctr'].values
            quality_values = calibration['mean_quality'].values
            ctr_monotonic = all(ctr_values[i] <= ctr_values[i+1] for i in range(len(ctr_values)-1))
            log(f"  CTR monotonically increasing with QUALITY: {ctr_monotonic}", f)

            # Calibration slope
            if np.std(quality_values) > 0:
                slope = np.corrcoef(quality_values, ctr_values)[0, 1]
                log(f"  Correlation(mean_quality, actual_ctr): {slope:.4f}", f)
        else:
            log(f"  Insufficient data for calibration analysis", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Bid Formula Test (C3)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: BID FORMULA TEST (C3)", f)
        log("-" * 40, f)
        log("", f)

        log("TEST C3: DOES FINAL_BID ≈ CONVERSION_RATE × PRICE?", f)
        log("  Question: Is FINAL_BID derived from pCVR × Price?", f)
        log("", f)

        bid_test = ar[['FINAL_BID', 'CONVERSION_RATE', 'PRICE']].dropna()
        log(f"  Rows with FINAL_BID, CONVERSION_RATE, PRICE: {len(bid_test):,}", f)

        if len(bid_test) > 1000:
            bid_test['predicted_bid'] = bid_test['CONVERSION_RATE'] * bid_test['PRICE']

            # R²
            r2 = r2_score(bid_test['FINAL_BID'], bid_test['predicted_bid'])
            log(f"  R² of FINAL_BID ~ CONVERSION_RATE × PRICE: {r2:.4f}", f)

            # Correlation
            corr = bid_test['FINAL_BID'].corr(bid_test['predicted_bid'])
            log(f"  Correlation: {corr:.4f}", f)

            # Ratio
            log("", f)
            log(f"  Summary of FINAL_BID / (CONVERSION_RATE × PRICE):", f)
            ratio = bid_test['FINAL_BID'] / bid_test['predicted_bid'].clip(lower=0.0001)
            log(f"    Mean: {ratio.mean():.4f}", f)
            log(f"    Median: {ratio.median():.4f}", f)
            log(f"    Std: {ratio.std():.4f}", f)
            log(f"    P10: {ratio.quantile(0.10):.4f}", f)
            log(f"    P90: {ratio.quantile(0.90):.4f}", f)

            log("", f)
            if r2 > 0.8:
                log(f"  INTERPRETATION: FINAL_BID ≈ CONVERSION_RATE × PRICE (R² > 0.8)", f)
            elif corr > 0.5:
                log(f"  INTERPRETATION: Moderate relationship (corr > 0.5)", f)
            else:
                log(f"  INTERPRETATION: FINAL_BID is NOT simply CONVERSION_RATE × PRICE", f)
        else:
            log(f"  Insufficient data for bid formula test", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Pacing Mechanism (D2)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: PACING MECHANISM (D2)", f)
        log("-" * 40, f)
        log("", f)

        log("TEST D2: IS PACING CAMPAIGN-LEVEL?", f)
        log("  Question: Does PACING vary within or between campaigns?", f)
        log("", f)

        pacing_data = ar[['CAMPAIGN_ID', 'PACING']].dropna()
        log(f"  Rows with CAMPAIGN_ID, PACING: {len(pacing_data):,}", f)
        log(f"  Unique campaigns: {pacing_data['CAMPAIGN_ID'].nunique():,}", f)
        log("", f)

        if pacing_data['CAMPAIGN_ID'].nunique() > 10:
            # ICC (Intraclass Correlation Coefficient)
            # Simplified: variance decomposition
            campaign_stats = pacing_data.groupby('CAMPAIGN_ID')['PACING'].agg(['mean', 'var', 'count'])
            campaign_stats = campaign_stats[campaign_stats['count'] >= 5]

            if len(campaign_stats) > 10:
                overall_mean = pacing_data['PACING'].mean()
                overall_var = pacing_data['PACING'].var()

                # Between-campaign variance
                between_var = ((campaign_stats['mean'] - overall_mean) ** 2 * campaign_stats['count']).sum() / campaign_stats['count'].sum()

                # Within-campaign variance (pooled)
                within_var = (campaign_stats['var'] * (campaign_stats['count'] - 1)).sum() / (campaign_stats['count'].sum() - len(campaign_stats))

                # ICC
                icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0

                log(f"  Variance decomposition:", f)
                log(f"    Overall variance: {overall_var:.6f}", f)
                log(f"    Between-campaign variance: {between_var:.6f}", f)
                log(f"    Within-campaign variance: {within_var:.6f}", f)
                log(f"    ICC (intraclass correlation): {icc:.4f}", f)
                log("", f)

                if icc > 0.8:
                    log(f"  INTERPRETATION: PACING is mostly campaign-level (ICC > 0.8)", f)
                elif icc > 0.5:
                    log(f"  INTERPRETATION: PACING varies both within and between campaigns", f)
                else:
                    log(f"  INTERPRETATION: PACING varies mostly within campaigns (not campaign-level)", f)

                # Show sample campaign pacing values
                log("", f)
                log(f"  Sample campaigns (top 10 by count):", f)
                top_campaigns = campaign_stats.nlargest(10, 'count')
                log(f"  {'Campaign':<40} {'N':<10} {'Mean PACING':<15} {'Std PACING':<15}", f)
                log(f"  {'-'*40} {'-'*10} {'-'*15} {'-'*15}", f)
                for cid, row in top_campaigns.iterrows():
                    std = np.sqrt(row['var']) if row['var'] > 0 else 0
                    log(f"  {str(cid)[:38]:<40} {int(row['count']):<10} {row['mean']:<15.4f} {std:<15.4f}", f)
        else:
            log(f"  Insufficient campaigns for variance decomposition", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: CTR by Placement × Rank
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: CTR BY PLACEMENT × RANK", f)
        log("-" * 40, f)
        log("", f)

        log("Question: Does CTR by rank differ across placements?", f)
        log("", f)

        winners_with_placement = winners[winners['PLACEMENT'].notna()].copy()
        log(f"  Winners with PLACEMENT: {len(winners_with_placement):,}", f)

        if len(winners_with_placement) > 1000:
            placements = winners_with_placement['PLACEMENT'].value_counts()
            log(f"  Unique placements: {len(placements)}", f)
            log("", f)

            # Show top placements
            log(f"  Top placements by volume:", f)
            for placement, count in placements.head(10).items():
                pct = count / len(winners_with_placement) * 100
                log(f"    {placement}: {count:,} ({pct:.1f}%)", f)
            log("", f)

            # CTR by rank for each major placement
            log(f"  CTR by Rank for top 5 placements:", f)
            log("", f)

            for placement in placements.head(5).index:
                subset = winners_with_placement[winners_with_placement['PLACEMENT'] == placement]
                ctr_by_rank = subset.groupby('RANKING').agg({
                    'clicked': ['sum', 'count', 'mean']
                }).reset_index()
                ctr_by_rank.columns = ['RANKING', 'clicks', 'n', 'CTR']
                ctr_by_rank = ctr_by_rank[ctr_by_rank['n'] >= 10].head(10)

                log(f"  Placement: {placement}", f)
                log(f"    {'Rank':<8} {'N':<12} {'Clicks':<10} {'CTR %':<10}", f)
                log(f"    {'-'*8} {'-'*12} {'-'*10} {'-'*10}", f)

                for _, row in ctr_by_rank.iterrows():
                    log(f"    {int(row['RANKING']):<8} {int(row['n']):<12,} {int(row['clicks']):<10,} {row['CTR']*100:<10.3f}", f)

                # Check monotonicity
                if len(ctr_by_rank) >= 3:
                    ctr_vals = ctr_by_rank['CTR'].values
                    mono = all(ctr_vals[i] >= ctr_vals[i+1] for i in range(min(5, len(ctr_vals)-1)))
                    log(f"    Monotonic (top 5): {mono}", f)
                log("", f)
        else:
            log(f"  Insufficient data for placement analysis", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Multi-Click Investigation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: MULTI-CLICK INVESTIGATION", f)
        log("-" * 40, f)
        log("", f)

        log("Question: What's happening in auctions with 2+ clicks?", f)
        log("", f)

        clicks_per_auction = clicks.groupby('AUCTION_ID').size()
        multi_click_auctions = clicks_per_auction[clicks_per_auction >= 2].index

        log(f"  Total auctions with clicks: {len(clicks_per_auction):,}", f)
        log(f"  Auctions with 2+ clicks: {len(multi_click_auctions):,} ({len(multi_click_auctions)/len(clicks_per_auction)*100:.1f}%)", f)
        log("", f)

        if len(multi_click_auctions) > 0:
            # Distribution of click counts
            log(f"  Distribution of clicks per auction:", f)
            click_dist = clicks_per_auction.value_counts().sort_index()
            for n_clicks, count in click_dist.head(10).items():
                pct = count / len(clicks_per_auction) * 100
                log(f"    {n_clicks} clicks: {count:,} ({pct:.1f}%)", f)
            log("", f)

            # Time gap between clicks in multi-click auctions
            multi_clicks = clicks[clicks['AUCTION_ID'].isin(multi_click_auctions)].copy()
            multi_clicks = multi_clicks.sort_values(['AUCTION_ID', 'OCCURRED_AT'])

            log(f"  Time gap analysis in multi-click auctions:", f)

            def compute_time_gaps(group):
                if len(group) < 2:
                    return pd.Series({'min_gap_seconds': np.nan, 'max_gap_seconds': np.nan})
                times = pd.to_datetime(group['OCCURRED_AT']).sort_values()
                gaps = times.diff().dropna().dt.total_seconds()
                return pd.Series({
                    'min_gap_seconds': gaps.min(),
                    'max_gap_seconds': gaps.max()
                })

            time_gaps = multi_clicks.groupby('AUCTION_ID').apply(compute_time_gaps)

            if len(time_gaps) > 0:
                log(f"    Min gap between clicks:", f)
                log(f"      Mean: {time_gaps['min_gap_seconds'].mean():.2f} seconds", f)
                log(f"      Median: {time_gaps['min_gap_seconds'].median():.2f} seconds", f)
                log(f"      P10: {time_gaps['min_gap_seconds'].quantile(0.10):.2f} seconds", f)
                log(f"      P90: {time_gaps['min_gap_seconds'].quantile(0.90):.2f} seconds", f)
                log("", f)

                # Check for potential duplicates (< 1 second gap)
                potential_dups = (time_gaps['min_gap_seconds'] < 1).sum()
                log(f"    Potential duplicate clicks (gap < 1 sec): {potential_dups} ({potential_dups/len(time_gaps)*100:.1f}%)", f)
                log("", f)

            # Same product or different?
            log(f"  Same product vs different products:", f)

            same_product = multi_clicks.groupby('AUCTION_ID')['PRODUCT_ID'].nunique() == 1
            same_product_count = same_product.sum()
            diff_product_count = len(same_product) - same_product_count

            log(f"    Same product clicked multiple times: {same_product_count} ({same_product_count/len(same_product)*100:.1f}%)", f)
            log(f"    Different products clicked: {diff_product_count} ({diff_product_count/len(same_product)*100:.1f}%)", f)
            log("", f)

            # Ranks of clicked products in multi-click auctions
            multi_click_ranks = multi_clicks.merge(
                winners[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
                on=['AUCTION_ID', 'PRODUCT_ID'],
                how='left'
            )

            if multi_click_ranks['RANKING'].notna().sum() > 0:
                log(f"  Ranks of products in multi-click auctions:", f)
                rank_dist = multi_click_ranks['RANKING'].value_counts().sort_index().head(10)
                for rank, count in rank_dist.items():
                    log(f"    Rank {int(rank)}: {count} clicks", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 9: PACING Distribution
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: PACING DISTRIBUTION", f)
        log("-" * 40, f)
        log("", f)

        pacing_vals = ar['PACING'].dropna()
        log(f"  N rows with PACING: {len(pacing_vals):,}", f)
        log("", f)

        log(f"  PACING range:", f)
        log(f"    Min: {pacing_vals.min():.6f}", f)
        log(f"    Max: {pacing_vals.max():.6f}", f)
        log(f"    Mean: {pacing_vals.mean():.6f}", f)
        log(f"    Median: {pacing_vals.median():.6f}", f)
        log(f"    Std: {pacing_vals.std():.6f}", f)
        log("", f)

        # Check if bounded [0,1]
        in_01 = ((pacing_vals >= 0) & (pacing_vals <= 1)).mean()
        log(f"  Fraction in [0, 1]: {in_01*100:.2f}%", f)
        log("", f)

        # Distribution shape
        log(f"  PACING percentiles:", f)
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = pacing_vals.quantile(p/100)
            log(f"    P{p}: {val:.6f}", f)
        log("", f)

        # By campaign
        log(f"  PACING by campaign (top 10 campaigns by N):", f)
        pacing_by_campaign = ar.groupby('CAMPAIGN_ID')['PACING'].agg(['mean', 'std', 'min', 'max', 'count'])
        pacing_by_campaign = pacing_by_campaign[pacing_by_campaign['count'] >= 100].nlargest(10, 'count')

        log(f"  {'Campaign':<40} {'N':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}", f)
        log(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", f)

        for cid, row in pacing_by_campaign.iterrows():
            log(f"  {str(cid)[:38]:<40} {int(row['count']):<10} {row['mean']:<10.4f} {row['std']:<10.4f} {row['min']:<10.4f} {row['max']:<10.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 10: Winner → Impression Selection (H3)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 10: WINNER → IMPRESSION SELECTION (H3)", f)
        log("-" * 40, f)
        log("", f)

        log("Question: Among IS_WINNER=True, who gets impressions?", f)
        log("  Is it top-N by rank? Random? Something else?", f)
        log("", f)

        # Merge winners with impressions
        winners_with_imp = winners.merge(
            imp[['AUCTION_ID', 'PRODUCT_ID', 'OCCURRED_AT']],
            on=['AUCTION_ID', 'PRODUCT_ID'],
            how='left'
        )
        winners_with_imp['got_impression'] = winners_with_imp['OCCURRED_AT'].notna()

        log(f"  Winners: {len(winners):,}", f)
        log(f"  Winners with impressions: {winners_with_imp['got_impression'].sum():,} ({winners_with_imp['got_impression'].mean()*100:.2f}%)", f)
        log("", f)

        # P(impression | rank, winner) by rank
        log(f"  P(impression | rank) for winners:", f)
        imp_rate_by_rank = winners_with_imp.groupby('RANKING').agg({
            'got_impression': ['sum', 'count', 'mean']
        }).reset_index()
        imp_rate_by_rank.columns = ['RANKING', 'n_impressed', 'n_winners', 'imp_rate']

        log(f"  {'Rank':<8} {'N Winners':<15} {'N Impressed':<15} {'Imp Rate %':<12}", f)
        log(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*12}", f)

        for _, row in imp_rate_by_rank.head(20).iterrows():
            log(f"  {int(row['RANKING']):<8} {int(row['n_winners']):<15,} {int(row['n_impressed']):<15,} {row['imp_rate']*100:<12.2f}", f)

        log("", f)

        # Check if impression rate is monotonic in rank
        imp_rates = imp_rate_by_rank.head(10)['imp_rate'].values
        imp_rate_monotonic = all(imp_rates[i] >= imp_rates[i+1] for i in range(len(imp_rates)-1))
        log(f"  Impression rate monotonically decreasing (top 10): {imp_rate_monotonic}", f)
        log("", f)

        # Is it top-N selection?
        log(f"  Testing top-N selection hypothesis:", f)

        # If top-N, we'd expect sharp cutoff
        # Check cumulative impression rate
        imp_rate_by_rank['cumulative_winners'] = imp_rate_by_rank['n_winners'].cumsum()
        imp_rate_by_rank['cumulative_impressed'] = imp_rate_by_rank['n_impressed'].cumsum()
        imp_rate_by_rank['cumulative_imp_rate'] = imp_rate_by_rank['cumulative_impressed'] / imp_rate_by_rank['cumulative_winners']

        log(f"  {'Up to Rank':<12} {'Cum Winners':<15} {'Cum Impressed':<15} {'Cum Rate %':<12}", f)
        log(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*12}", f)

        for _, row in imp_rate_by_rank.head(15).iterrows():
            log(f"  {int(row['RANKING']):<12} {int(row['cumulative_winners']):<15,} {int(row['cumulative_impressed']):<15,} {row['cumulative_imp_rate']*100:<12.2f}", f)

        log("", f)

        # Check for sharp cutoff: what rank captures 90% of impressions?
        total_impressed = imp_rate_by_rank['n_impressed'].sum()
        for _, row in imp_rate_by_rank.iterrows():
            if row['cumulative_impressed'] >= 0.9 * total_impressed:
                log(f"  Rank {int(row['RANKING'])} captures 90% of impressions", f)
                break

        for _, row in imp_rate_by_rank.iterrows():
            if row['cumulative_impressed'] >= 0.99 * total_impressed:
                log(f"  Rank {int(row['RANKING'])} captures 99% of impressions", f)
                break

        log("", f)

        # -----------------------------------------------------------------
        # Section 11: Rank 4 Anomaly Investigation (H7)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 11: RANK 4 ANOMALY INVESTIGATION (H7)", f)
        log("-" * 40, f)
        log("", f)

        log("Question: Is Rank 4 a special slot? Different characteristics?", f)
        log("", f)

        # Compare characteristics at ranks 3, 4, 5
        for compare_rank in [3, 4, 5]:
            rank_data = winners[winners['RANKING'] == compare_rank]
            log(f"  Rank {compare_rank} characteristics (N={len(rank_data):,}):", f)

            if len(rank_data) > 0:
                log(f"    QUALITY:    mean={rank_data['QUALITY'].mean():.6f}, median={rank_data['QUALITY'].median():.6f}, std={rank_data['QUALITY'].std():.6f}", f)
                log(f"    FINAL_BID:  mean={rank_data['FINAL_BID'].mean():.4f}, median={rank_data['FINAL_BID'].median():.4f}, std={rank_data['FINAL_BID'].std():.4f}", f)
                if 'PRICE' in rank_data.columns and rank_data['PRICE'].notna().sum() > 0:
                    log(f"    PRICE:      mean={rank_data['PRICE'].mean():.2f}, median={rank_data['PRICE'].median():.2f}, std={rank_data['PRICE'].std():.2f}", f)
                if 'PACING' in rank_data.columns and rank_data['PACING'].notna().sum() > 0:
                    log(f"    PACING:     mean={rank_data['PACING'].mean():.4f}, median={rank_data['PACING'].median():.4f}, std={rank_data['PACING'].std():.4f}", f)

                # CTR at this rank
                clicked_at_rank = rank_data['clicked'].sum()
                ctr_at_rank = rank_data['clicked'].mean()
                log(f"    CTR:        {ctr_at_rank*100:.4f}% ({clicked_at_rank:,} clicks)", f)

                # Impression rate at this rank
                rank_with_imp = winners_with_imp[winners_with_imp['RANKING'] == compare_rank]
                imp_rate_at_rank = rank_with_imp['got_impression'].mean() if len(rank_with_imp) > 0 else 0
                log(f"    Imp Rate:   {imp_rate_at_rank*100:.2f}%", f)
            log("", f)

        # Statistical test: is Rank 4 different from Rank 3?
        log(f"  Comparing Rank 3 vs Rank 4:", f)
        rank3 = winners[winners['RANKING'] == 3]
        rank4 = winners[winners['RANKING'] == 4]

        if len(rank3) > 100 and len(rank4) > 100:
            # Quality difference
            quality_diff = rank4['QUALITY'].mean() - rank3['QUALITY'].mean()
            quality_ratio = rank4['QUALITY'].mean() / rank3['QUALITY'].mean() if rank3['QUALITY'].mean() > 0 else np.nan
            log(f"    QUALITY diff (R4-R3): {quality_diff:.6f}", f)
            log(f"    QUALITY ratio (R4/R3): {quality_ratio:.4f}", f)

            # Bid difference
            bid_diff = rank4['FINAL_BID'].mean() - rank3['FINAL_BID'].mean()
            bid_ratio = rank4['FINAL_BID'].mean() / rank3['FINAL_BID'].mean() if rank3['FINAL_BID'].mean() > 0 else np.nan
            log(f"    FINAL_BID diff (R4-R3): {bid_diff:.4f}", f)
            log(f"    FINAL_BID ratio (R4/R3): {bid_ratio:.4f}", f)

            # CTR difference
            ctr3 = rank3['clicked'].mean()
            ctr4 = rank4['clicked'].mean()
            ctr_diff = ctr4 - ctr3
            ctr_ratio = ctr4 / ctr3 if ctr3 > 0 else np.nan
            log(f"    CTR diff (R4-R3): {ctr_diff*100:.4f}pp", f)
            log(f"    CTR ratio (R4/R3): {ctr_ratio:.4f}", f)

        log("", f)

        # Check if Rank 4 has more variance (mixing different types?)
        log(f"  Variance comparison (coefficient of variation):", f)
        for compare_rank in [3, 4, 5]:
            rank_data = winners[winners['RANKING'] == compare_rank]
            if len(rank_data) > 100:
                cv_quality = rank_data['QUALITY'].std() / rank_data['QUALITY'].mean() if rank_data['QUALITY'].mean() > 0 else np.nan
                cv_bid = rank_data['FINAL_BID'].std() / rank_data['FINAL_BID'].mean() if rank_data['FINAL_BID'].mean() > 0 else np.nan
                log(f"    Rank {compare_rank}: CV(QUALITY)={cv_quality:.4f}, CV(BID)={cv_bid:.4f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 12: Position Effects Within Impression-Receivers (H4)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 12: POSITION EFFECTS WITHIN IMPRESSION-RECEIVERS (H4)", f)
        log("-" * 40, f)
        log("", f)

        log("Question: Among products that got impressions, is CTR monotonic in rank?", f)
        log("", f)

        # Filter to winners who got impressions
        impressed_winners = winners_with_imp[winners_with_imp['got_impression'] == True].copy()
        log(f"  Winners with impressions: {len(impressed_winners):,}", f)
        log("", f)

        # CTR by rank for impressed only
        log(f"  CTR by Rank (impression-receivers only):", f)
        ctr_by_rank_impressed = impressed_winners.groupby('RANKING').agg({
            'clicked': ['sum', 'count', 'mean']
        }).reset_index()
        ctr_by_rank_impressed.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

        log(f"  {'Rank':<8} {'Impressions':<15} {'Clicks':<12} {'CTR %':<10}", f)
        log(f"  {'-'*8} {'-'*15} {'-'*12} {'-'*10}", f)

        for _, row in ctr_by_rank_impressed.head(20).iterrows():
            if row['impressions'] >= 10:
                log(f"  {int(row['RANKING']):<8} {int(row['impressions']):<15,} {int(row['clicks']):<12,} {row['CTR']*100:<10.4f}", f)

        log("", f)

        # Check monotonicity
        ctr_impressed_vals = ctr_by_rank_impressed[ctr_by_rank_impressed['impressions'] >= 100].head(10)['CTR'].values
        ctr_impressed_monotonic = None
        if len(ctr_impressed_vals) > 1:
            ctr_impressed_monotonic = all(ctr_impressed_vals[i] >= ctr_impressed_vals[i+1] for i in range(len(ctr_impressed_vals)-1))
            log(f"  CTR monotonically decreasing (top 10 with N>=100): {ctr_impressed_monotonic}", f)
        log("", f)

        # Compare to all winners
        log(f"  Comparison: CTR by Rank (all winners vs impression-receivers):", f)
        ctr_all_winners = winners.groupby('RANKING').agg({
            'clicked': ['sum', 'count', 'mean']
        }).reset_index()
        ctr_all_winners.columns = ['RANKING', 'clicks', 'n', 'CTR']

        log(f"  {'Rank':<8} {'All Winners CTR %':<20} {'Impressed CTR %':<20} {'Ratio':<10}", f)
        log(f"  {'-'*8} {'-'*20} {'-'*20} {'-'*10}", f)

        for rank in range(1, 11):
            all_row = ctr_all_winners[ctr_all_winners['RANKING'] == rank]
            imp_row = ctr_by_rank_impressed[ctr_by_rank_impressed['RANKING'] == rank]

            if len(all_row) > 0 and len(imp_row) > 0:
                all_ctr = all_row['CTR'].values[0]
                imp_ctr = imp_row['CTR'].values[0]
                ratio = imp_ctr / all_ctr if all_ctr > 0 else np.nan
                log(f"  {rank:<8} {all_ctr*100:<20.4f} {imp_ctr*100:<20.4f} {ratio:<10.2f}", f)

        log("", f)

        # Position effect magnitude
        log(f"  Position effect magnitude (relative to Rank 1):", f)
        rank1_ctr_arr = ctr_by_rank_impressed[ctr_by_rank_impressed['RANKING'] == 1]['CTR'].values
        rank1_ctr_impressed = rank1_ctr_arr[0] if len(rank1_ctr_arr) > 0 else None
        if rank1_ctr_impressed is not None:
            log(f"  {'Rank':<8} {'CTR %':<15} {'Relative to R1':<15} {'Decay':<10}", f)
            log(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*10}", f)

            for _, row in ctr_by_rank_impressed.head(10).iterrows():
                if row['impressions'] >= 10:
                    relative = row['CTR'] / rank1_ctr_impressed if rank1_ctr_impressed > 0 else np.nan
                    decay = 1 - relative if relative is not np.nan else np.nan
                    log(f"  {int(row['RANKING']):<8} {row['CTR']*100:<15.4f} {relative:<15.4f} {decay:<10.4f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 13: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 13: SUMMARY OF FINDINGS", f)
        log("=" * 80, f)
        log("", f)

        log("RANKING FORMULA TESTS:", f)
        log(f"  A1: Exact rank match rate (QUALITY × BID): {exact_match_v1*100:.2f}%", f)
        log(f"  A2: Rank-1 has max score: {rank1_has_max_v1*100:.2f}%", f)
        log(f"  D5: R² improvement from PACING: {r2_improvement:.4f}", f)
        log(f"  D6: CORR(BID, BID×PACING): {corr_bid_pacingbid:.4f}", f)
        log("", f)

        log("FUNNEL VERIFICATION:", f)
        log(f"  F2: Mean winners per auction: {winners_per_auction.mean():.2f}", f)
        log(f"  F3: Mean impressions per auction: {imp_per_auction.mean():.2f}", f)
        log("", f)

        if 'auc' in dir():
            log("QUALITY SCORE:", f)
            log(f"  B2: AUC for click prediction: {auc:.4f}", f)
        log("", f)

        log("WINNER → IMPRESSION SELECTION (H3):", f)
        log(f"  Overall impression rate for winners: {winners_with_imp['got_impression'].mean()*100:.2f}%", f)
        log(f"  Impression rate monotonic in rank: {imp_rate_monotonic}", f)
        log("", f)

        log("POSITION EFFECTS (H4 - Impressed Only):", f)
        if ctr_impressed_monotonic is not None:
            log(f"  CTR monotonic for impression-receivers: {ctr_impressed_monotonic}", f)
        if rank1_ctr_impressed is not None and rank1_ctr_impressed > 0:
            log(f"  Rank 1 CTR (impressed): {rank1_ctr_impressed*100:.4f}%", f)
        log("", f)

        log("=" * 80, f)
        log("HYPOTHESIS TESTING COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
