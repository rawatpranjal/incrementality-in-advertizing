#!/usr/bin/env python3
"""
Causal Inference Viability EDA for Round 2 Data

Comprehensive assessment of causal inference methods:
- Selection into impression
- RDD feasibility
- Display position vs bid rank
- CTR by rank
- Within-product position variation
- Placement deep dive
- Multi-click patterns
- Purchase funnel
- Score mechanics validation
- Competition as instrument
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_r2"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "03_causal_eda.txt"

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
        log("CAUSAL INFERENCE VIABILITY EDA - ROUND 2 DATA", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 0: Data Loading
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 0: DATA LOADING", f)
        log("-" * 40, f)
        log("", f)

        # Load all files
        ar_path = DATA_DIR / "auctions_results_r2.parquet"
        au_path = DATA_DIR / "auctions_users_r2.parquet"
        imp_path = DATA_DIR / "impressions_r2.parquet"
        clicks_path = DATA_DIR / "clicks_r2.parquet"
        purchases_path = DATA_DIR / "purchases_r2.parquet"
        catalog_path = DATA_DIR / "catalog_r2.parquet"

        log("Loading data files...", f)

        ar = pd.read_parquet(ar_path)
        log(f"  auctions_results: {len(ar):,} rows, {ar.shape[1]} columns", f)

        au = pd.read_parquet(au_path)
        log(f"  auctions_users: {len(au):,} rows, {au.shape[1]} columns", f)

        imp = pd.read_parquet(imp_path)
        log(f"  impressions: {len(imp):,} rows, {imp.shape[1]} columns", f)

        clicks = pd.read_parquet(clicks_path)
        log(f"  clicks: {len(clicks):,} rows, {clicks.shape[1]} columns", f)

        purchases = pd.read_parquet(purchases_path)
        log(f"  purchases: {len(purchases):,} rows, {purchases.shape[1]} columns", f)

        catalog = pd.read_parquet(catalog_path)
        log(f"  catalog: {len(catalog):,} rows, {catalog.shape[1]} columns", f)
        log("", f)

        # Column info
        log("AUCTIONS_RESULTS columns:", f)
        for col in ar.columns:
            dtype = ar[col].dtype
            n_null = ar[col].isna().sum()
            if ar[col].dtype in ['float64', 'int64']:
                log(f"  {col}: dtype={dtype}, null={n_null:,}, min={ar[col].min():.4f}, max={ar[col].max():.4f}", f)
            else:
                log(f"  {col}: dtype={dtype}, null={n_null:,}, unique={ar[col].nunique():,}", f)
        log("", f)

        # Date ranges
        log("Date ranges:", f)
        log(f"  auctions_results: {ar['CREATED_AT'].min()} to {ar['CREATED_AT'].max()}", f)
        log(f"  auctions_users: {au['CREATED_AT'].min()} to {au['CREATED_AT'].max()}", f)
        log(f"  impressions: {imp['OCCURRED_AT'].min()} to {imp['OCCURRED_AT'].max()}", f)
        log(f"  clicks: {clicks['OCCURRED_AT'].min()} to {clicks['OCCURRED_AT'].max()}", f)
        log(f"  purchases: {purchases['PURCHASED_AT'].min()} to {purchases['PURCHASED_AT'].max()}", f)
        log("", f)

        # Basic counts
        log("Basic counts:", f)
        log(f"  Unique auctions (AR): {ar['AUCTION_ID'].nunique():,}", f)
        log(f"  Unique auctions (AU): {au['AUCTION_ID'].nunique():,}", f)
        log(f"  Unique vendors: {ar['VENDOR_ID'].nunique():,}", f)
        log(f"  Unique products (AR): {ar['PRODUCT_ID'].nunique():,}", f)
        log(f"  Unique products (catalog): {catalog['PRODUCT_ID'].nunique():,}", f)
        log(f"  Unique users (AU): {au['USER_ID'].nunique():,}", f)
        log(f"  Unique users (purchases): {purchases['USER_ID'].nunique():,}", f)
        log("", f)

        # Join validation
        log("Join validation (auction_id overlaps):", f)
        ar_auctions = set(ar['AUCTION_ID'].unique())
        au_auctions = set(au['AUCTION_ID'].unique())
        imp_auctions = set(imp['AUCTION_ID'].unique())
        click_auctions = set(clicks['AUCTION_ID'].unique())

        ar_au_overlap = len(ar_auctions & au_auctions) / len(ar_auctions) * 100
        imp_ar_overlap = len(imp_auctions & ar_auctions) / len(imp_auctions) * 100 if len(imp_auctions) > 0 else 0
        click_ar_overlap = len(click_auctions & ar_auctions) / len(click_auctions) * 100 if len(click_auctions) > 0 else 0

        log(f"  AR ∩ AU: {ar_au_overlap:.1f}%", f)
        log(f"  IMP ∩ AR: {imp_ar_overlap:.1f}%", f)
        log(f"  CLICK ∩ AR: {click_ar_overlap:.1f}%", f)
        log("", f)

        # Create merged analysis dataframe (impressions + rankings + clicks)
        log("Creating merged analysis dataframe...", f)

        # Merge impressions with auction results to get RANKING
        imp_with_rank = imp.merge(
            ar[['AUCTION_ID', 'PRODUCT_ID', 'RANKING', 'IS_WINNER', 'QUALITY', 'FINAL_BID', 'PACING', 'CONVERSION_RATE']],
            on=['AUCTION_ID', 'PRODUCT_ID'],
            how='left'
        )

        # Merge with auctions_users for placement
        imp_with_rank = imp_with_rank.merge(
            au[['AUCTION_ID', 'USER_ID', 'PLACEMENT']],
            on=['AUCTION_ID', 'USER_ID'],
            how='left'
        )

        # Flag clicks
        click_keys = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
        imp_with_rank['clicked'] = imp_with_rank.apply(
            lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_keys, axis=1
        )

        log(f"  Merged impressions: {len(imp_with_rank):,} rows", f)
        log(f"  With RANKING: {imp_with_rank['RANKING'].notna().sum():,} ({imp_with_rank['RANKING'].notna().mean()*100:.1f}%)", f)
        log(f"  With PLACEMENT: {imp_with_rank['PLACEMENT'].notna().sum():,} ({imp_with_rank['PLACEMENT'].notna().mean()*100:.1f}%)", f)
        log(f"  Total clicks matched: {imp_with_rank['clicked'].sum():,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Selection Into Impression (Priority 1)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: SELECTION INTO IMPRESSION", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Understand what determines whether a bid gets an impression.", f)
        log("", f)

        # Add impression flag to auction results
        imp_keys = set(zip(imp['AUCTION_ID'], imp['PRODUCT_ID']))
        ar['got_impression'] = ar.apply(
            lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in imp_keys, axis=1
        )

        log("Overall impression rate:", f)
        total_bids = len(ar)
        total_impressions = ar['got_impression'].sum()
        log(f"  Total bids: {total_bids:,}", f)
        log(f"  Bids with impression: {total_impressions:,} ({total_impressions/total_bids*100:.2f}%)", f)
        log(f"  Winners: {ar['IS_WINNER'].sum():,} ({ar['IS_WINNER'].mean()*100:.2f}%)", f)
        log("", f)

        # Impression rate by IS_WINNER
        log("Impression rate by IS_WINNER:", f)
        for winner in [True, False]:
            subset = ar[ar['IS_WINNER'] == winner]
            rate = subset['got_impression'].mean() * 100
            log(f"  IS_WINNER={winner}: {rate:.2f}% (N={len(subset):,})", f)
        log("", f)

        # Impression rate by RANKING
        log("Impression rate by RANKING (top 64):", f)
        log(f"  {'Rank':<8} {'N Bids':<15} {'Imp Rate %':<12} {'Winner Rate %':<15}", f)
        log(f"  {'-'*8} {'-'*15} {'-'*12} {'-'*15}", f)

        for rank in range(1, 65):
            subset = ar[ar['RANKING'] == rank]
            if len(subset) > 0:
                imp_rate = subset['got_impression'].mean() * 100
                win_rate = subset['IS_WINNER'].mean() * 100
                log(f"  {rank:<8} {len(subset):<15,} {imp_rate:<12.2f} {win_rate:<15.2f}", f)
        log("", f)

        # Impression rate by QUALITY decile
        log("Impression rate by QUALITY decile:", f)
        ar['quality_decile'] = pd.qcut(ar['QUALITY'], 10, labels=False, duplicates='drop')
        quality_imp = ar.groupby('quality_decile').agg({
            'got_impression': ['mean', 'count'],
            'IS_WINNER': 'mean'
        }).reset_index()
        quality_imp.columns = ['decile', 'imp_rate', 'n', 'win_rate']

        log(f"  {'Decile':<10} {'N':<12} {'Imp Rate %':<15} {'Win Rate %':<15}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*15} {'-'*15}", f)
        for _, row in quality_imp.iterrows():
            log(f"  {int(row['decile']):<10} {int(row['n']):<12,} {row['imp_rate']*100:<15.2f} {row['win_rate']*100:<15.2f}", f)
        log("", f)

        # Impression rate by FINAL_BID decile
        log("Impression rate by FINAL_BID decile:", f)
        ar['bid_decile'] = pd.qcut(ar['FINAL_BID'], 10, labels=False, duplicates='drop')
        bid_imp = ar.groupby('bid_decile').agg({
            'got_impression': ['mean', 'count'],
            'IS_WINNER': 'mean'
        }).reset_index()
        bid_imp.columns = ['decile', 'imp_rate', 'n', 'win_rate']

        log(f"  {'Decile':<10} {'N':<12} {'Imp Rate %':<15} {'Win Rate %':<15}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*15} {'-'*15}", f)
        for _, row in bid_imp.iterrows():
            log(f"  {int(row['decile']):<10} {int(row['n']):<12,} {row['imp_rate']*100:<15.2f} {row['win_rate']*100:<15.2f}", f)
        log("", f)

        # Impression rate by auction size
        log("Impression rate by auction size (num bidders):", f)
        auction_size = ar.groupby('AUCTION_ID').size().reset_index(name='n_bidders')
        ar = ar.merge(auction_size, on='AUCTION_ID', how='left')

        max_bidders = ar['n_bidders'].max()
        # Build bins dynamically based on actual data
        size_bins = [0, 10, 20, 30, 40, 50, 60, max_bidders + 1]
        size_labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '60+']
        ar['size_bin'] = pd.cut(ar['n_bidders'], bins=size_bins, labels=size_labels)
        size_imp = ar.groupby('size_bin').agg({
            'got_impression': ['mean', 'count'],
            'IS_WINNER': 'mean'
        }).reset_index()
        size_imp.columns = ['size_bin', 'imp_rate', 'n', 'win_rate']

        log(f"  {'Size Bin':<12} {'N':<15} {'Imp Rate %':<15} {'Win Rate %':<15}", f)
        log(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15}", f)
        for _, row in size_imp.iterrows():
            if row['n'] > 0:
                log(f"  {str(row['size_bin']):<12} {int(row['n']):<15,} {row['imp_rate']*100:<15.2f} {row['win_rate']*100:<15.2f}", f)
        log("", f)

        # Impression rate by PACING
        log("Impression rate by PACING decile:", f)
        ar['pacing_decile'] = pd.qcut(ar['PACING'], 10, labels=False, duplicates='drop')
        pacing_imp = ar.groupby('pacing_decile').agg({
            'got_impression': ['mean', 'count'],
            'IS_WINNER': 'mean'
        }).reset_index()
        pacing_imp.columns = ['decile', 'imp_rate', 'n', 'win_rate']

        log(f"  {'Decile':<10} {'N':<12} {'Imp Rate %':<15} {'Win Rate %':<15}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*15} {'-'*15}", f)
        for _, row in pacing_imp.iterrows():
            log(f"  {int(row['decile']):<10} {int(row['n']):<12,} {row['imp_rate']*100:<15.2f} {row['win_rate']*100:<15.2f}", f)
        log("", f)

        # Logistic regression: P(impression | winner) ~ rank + quality + bid + pacing + auction_size
        log("Logistic regression: P(impression) ~ features", f)
        log("", f)

        # Prepare data
        model_data = ar[['got_impression', 'IS_WINNER', 'RANKING', 'QUALITY', 'FINAL_BID', 'PACING', 'n_bidders']].dropna()
        X = model_data[['RANKING', 'QUALITY', 'FINAL_BID', 'PACING', 'n_bidders']]
        y = model_data['got_impression'].astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        log("  Logistic regression coefficients (standardized):", f)
        for feat, coef in zip(X.columns, model.coef_[0]):
            log(f"    {feat}: {coef:.4f}", f)
        log(f"  Intercept: {model.intercept_[0]:.4f}", f)
        log("", f)

        # Pseudo-R2
        from sklearn.metrics import log_loss
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        ll_model = -log_loss(y, y_pred_proba, normalize=False)
        ll_null = -log_loss(y, [y.mean()] * len(y), normalize=False)
        pseudo_r2 = 1 - (ll_model / ll_null)
        log(f"  McFadden's pseudo-R2: {pseudo_r2:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: RDD Feasibility (Priority 2)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: RDD FEASIBILITY", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Assess feasibility of regression discontinuity design.", f)
        log("", f)

        # Compute score = QUALITY * FINAL_BID
        ar['score'] = ar['QUALITY'] * ar['FINAL_BID']

        log("Score distribution (QUALITY * FINAL_BID):", f)
        score_stats = ar['score'].describe()
        for stat, val in score_stats.items():
            log(f"  {stat}: {val:.4f}", f)
        log("", f)

        # Within-auction score gaps between adjacent ranks
        log("Within-auction score gaps between adjacent ranks:", f)

        def compute_gaps(group):
            group = group.sort_values('RANKING')
            group['score_next'] = group['score'].shift(-1)
            group['rank_next'] = group['RANKING'].shift(-1)
            group['gap'] = group['score'] - group['score_next']
            return group

        ar_with_gaps = ar.groupby('AUCTION_ID', group_keys=False).apply(compute_gaps)
        ar_with_gaps = ar_with_gaps[ar_with_gaps['gap'].notna()]

        log(f"  Total gap observations: {len(ar_with_gaps):,}", f)
        log("", f)

        # Gap distribution
        log("Gap distribution:", f)
        gaps = ar_with_gaps['gap']
        log(f"  Mean: {gaps.mean():.6f}", f)
        log(f"  Median: {gaps.median():.6f}", f)
        log(f"  Std: {gaps.std():.6f}", f)
        log(f"  P25: {gaps.quantile(0.25):.6f}", f)
        log(f"  P75: {gaps.quantile(0.75):.6f}", f)
        log(f"  P95: {gaps.quantile(0.95):.6f}", f)
        log("", f)

        # Percentage of gaps below thresholds
        log("Gap concentration:", f)
        for thresh in [0.1, 0.01, 0.001, 0.0001]:
            pct = (gaps.abs() < thresh).mean() * 100
            log(f"  |gap| < {thresh}: {pct:.2f}%", f)
        log("", f)

        # Sample size at each rank boundary within bandwidths
        log("Sample size at rank boundaries within bandwidths:", f)
        for bw in [0.01, 0.001, 0.0001]:
            for boundary in [1, 2, 3, 4, 5]:
                # Bids at rank = boundary with small gap to next
                subset = ar_with_gaps[(ar_with_gaps['RANKING'] == boundary) & (ar_with_gaps['gap'].abs() < bw)]
                n = len(subset)
                log(f"  Rank {boundary}, |gap| < {bw}: N = {n:,}", f)
        log("", f)

        # McCrary density test proxy (bunching near 0)
        log("McCrary density test proxy:", f)
        # Count observations in bins around 0
        bin_width = 0.0001
        bins = np.arange(-0.01, 0.01 + bin_width, bin_width)
        hist, bin_edges = np.histogram(gaps, bins=bins)
        mid_idx = len(hist) // 2

        left_density = hist[:mid_idx].mean() if mid_idx > 0 else 0
        right_density = hist[mid_idx:].mean() if mid_idx < len(hist) else 0

        log(f"  Left of 0 density (mean count): {left_density:.2f}", f)
        log(f"  Right of 0 density (mean count): {right_density:.2f}", f)
        log(f"  Ratio (left/right): {left_density/right_density:.3f}" if right_density > 0 else "  Ratio: undefined", f)
        log("", f)

        # Covariate balance at cutoff
        log("Covariate balance at winner/loser cutoff:", f)

        # Compare winners at rank boundary vs non-winners at rank boundary
        # Focus on the IS_WINNER cutoff
        winners_df = ar[ar['IS_WINNER'] == True]
        losers_df = ar[ar['IS_WINNER'] == False]

        # Get max winner rank per auction
        max_winner_rank = winners_df.groupby('AUCTION_ID')['RANKING'].max().reset_index(name='max_winner_rank')
        ar = ar.merge(max_winner_rank, on='AUCTION_ID', how='left')

        # Marginal winners: IS_WINNER=True at max_winner_rank
        marginal_winners = ar[(ar['IS_WINNER'] == True) & (ar['RANKING'] == ar['max_winner_rank'])]
        # Marginal losers: IS_WINNER=False at max_winner_rank + 1
        ar['is_marginal_loser'] = (ar['IS_WINNER'] == False) & (ar['RANKING'] == ar['max_winner_rank'] + 1)
        marginal_losers = ar[ar['is_marginal_loser']]

        log(f"  Marginal winners (rank = max winner rank): {len(marginal_winners):,}", f)
        log(f"  Marginal losers (rank = max winner rank + 1): {len(marginal_losers):,}", f)
        log("", f)

        if len(marginal_winners) > 100 and len(marginal_losers) > 100:
            for var in ['QUALITY', 'FINAL_BID', 'PACING', 'CONVERSION_RATE']:
                winner_mean = marginal_winners[var].mean()
                loser_mean = marginal_losers[var].mean()
                diff = winner_mean - loser_mean
                pooled_std = np.sqrt((marginal_winners[var].var() + marginal_losers[var].var()) / 2)
                std_diff = diff / pooled_std if pooled_std > 0 else 0
                log(f"  {var}: winner={winner_mean:.4f}, loser={loser_mean:.4f}, diff={diff:.4f}, std_diff={std_diff:.3f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Display Position vs Bid Rank (Priority 3)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: DISPLAY POSITION VS BID RANK", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Assess if RANKING corresponds to display order.", f)
        log("", f)

        # For non-batched auctions: correlation of timestamp order vs RANKING
        log("Timestamp analysis of impressions:", f)

        # Group impressions by auction
        imp_auction = imp.groupby('AUCTION_ID').agg({
            'OCCURRED_AT': ['min', 'max', 'nunique', 'count'],
            'PRODUCT_ID': 'nunique'
        }).reset_index()
        imp_auction.columns = ['AUCTION_ID', 'min_time', 'max_time', 'n_unique_times', 'n_impressions', 'n_products']

        imp_auction['time_span_seconds'] = (imp_auction['max_time'] - imp_auction['min_time']).dt.total_seconds()

        log(f"  Total auctions with impressions: {len(imp_auction):,}", f)
        log(f"  Auctions with unique timestamps per impression: {(imp_auction['n_unique_times'] == imp_auction['n_impressions']).sum():,}", f)
        log(f"  Mean time span (seconds): {imp_auction['time_span_seconds'].mean():.4f}", f)
        log(f"  Auctions with time span > 1s: {(imp_auction['time_span_seconds'] > 1).sum():,}", f)
        log("", f)

        # Define "good timestamp" auctions
        good_timestamp_auctions = set(
            imp_auction[
                (imp_auction['n_unique_times'] == imp_auction['n_impressions']) &
                (imp_auction['time_span_seconds'] > 1)
            ]['AUCTION_ID']
        )
        log(f"  'Good timestamp' auctions (unique times, span > 1s): {len(good_timestamp_auctions):,}", f)
        log("", f)

        # For good timestamp auctions, compute correlation of timestamp order vs RANKING
        if len(good_timestamp_auctions) > 10:
            good_imp = imp_with_rank[imp_with_rank['AUCTION_ID'].isin(good_timestamp_auctions)]

            # Within each auction, compute timestamp rank
            def compute_display_position(group):
                group = group.sort_values('OCCURRED_AT')
                group['display_position'] = range(1, len(group) + 1)
                return group

            good_imp = good_imp.groupby('AUCTION_ID', group_keys=False).apply(compute_display_position)

            # Correlation of display_position vs RANKING
            corr = good_imp[['display_position', 'RANKING']].dropna()
            if len(corr) > 10:
                correlation = corr['display_position'].corr(corr['RANKING'])
                log(f"  Correlation (display_position vs RANKING): {correlation:.4f}", f)

                # Distribution of |display_position - RANKING|
                corr['position_diff'] = (corr['display_position'] - corr['RANKING']).abs()
                log(f"  Mean |display - RANKING|: {corr['position_diff'].mean():.2f}", f)
                log(f"  Median |display - RANKING|: {corr['position_diff'].median():.2f}", f)
                log(f"  % exact match: {(corr['position_diff'] == 0).mean()*100:.1f}%", f)
        else:
            log("  Insufficient good timestamp auctions for analysis.", f)
        log("", f)

        # Compare CTR in good vs bad timestamp auctions
        log("CTR comparison by timestamp quality:", f)

        imp_with_rank['good_timestamp'] = imp_with_rank['AUCTION_ID'].isin(good_timestamp_auctions)

        for quality in [True, False]:
            subset = imp_with_rank[imp_with_rank['good_timestamp'] == quality]
            if len(subset) > 0:
                ctr = subset['clicked'].mean() * 100
                log(f"  Good timestamp = {quality}: CTR = {ctr:.3f}% (N = {len(subset):,})", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: CTR by Rank (Priority 4)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: CTR BY RANK", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Measure click-through rate by ranking position.", f)
        log("", f)

        # CTR by RANKING positions 1-20
        log("CTR by RANKING (positions 1-20):", f)
        log(f"  {'Rank':<8} {'Impressions':<15} {'Clicks':<12} {'CTR %':<10}", f)
        log(f"  {'-'*8} {'-'*15} {'-'*12} {'-'*10}", f)

        ctr_by_rank = imp_with_rank.groupby('RANKING').agg({
            'clicked': ['sum', 'count', 'mean']
        }).reset_index()
        ctr_by_rank.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

        for _, row in ctr_by_rank[ctr_by_rank['RANKING'] <= 20].iterrows():
            log(f"  {int(row['RANKING']):<8} {int(row['impressions']):<15,} {int(row['clicks']):<12,} {row['CTR']*100:<10.3f}", f)
        log("", f)

        # CTR by placement (stratified)
        log("CTR by RANKING stratified by PLACEMENT:", f)

        for placement in sorted(imp_with_rank['PLACEMENT'].dropna().unique()):
            subset = imp_with_rank[imp_with_rank['PLACEMENT'] == placement]
            log(f"", f)
            log(f"  PLACEMENT = {placement}:", f)

            ctr_subset = subset.groupby('RANKING').agg({
                'clicked': ['sum', 'count', 'mean']
            }).reset_index()
            ctr_subset.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

            log(f"    {'Rank':<8} {'Impressions':<15} {'Clicks':<10} {'CTR %':<8}", f)
            log(f"    {'-'*8} {'-'*15} {'-'*10} {'-'*8}", f)

            for _, row in ctr_subset[ctr_subset['RANKING'] <= 10].iterrows():
                log(f"    {int(row['RANKING']):<8} {int(row['impressions']):<15,} {int(row['clicks']):<10,} {row['CTR']*100:<8.3f}", f)
        log("", f)

        # Check Rank 4 anomaly
        log("Rank 4 anomaly check:", f)
        rank_3_ctr = ctr_by_rank[ctr_by_rank['RANKING'] == 3]['CTR'].values
        rank_4_ctr = ctr_by_rank[ctr_by_rank['RANKING'] == 4]['CTR'].values
        rank_5_ctr = ctr_by_rank[ctr_by_rank['RANKING'] == 5]['CTR'].values

        if len(rank_3_ctr) > 0 and len(rank_4_ctr) > 0 and len(rank_5_ctr) > 0:
            expected_4 = (rank_3_ctr[0] + rank_5_ctr[0]) / 2
            actual_4 = rank_4_ctr[0]
            log(f"  Rank 3 CTR: {rank_3_ctr[0]*100:.3f}%", f)
            log(f"  Rank 4 CTR: {actual_4*100:.3f}%", f)
            log(f"  Rank 5 CTR: {rank_5_ctr[0]*100:.3f}%", f)
            log(f"  Expected Rank 4 (linear interp): {expected_4*100:.3f}%", f)
            log(f"  Anomaly: {(actual_4 - expected_4)*100:.3f}pp", f)
        log("", f)

        # CTR by rank conditional on auction size
        log("CTR by rank conditional on auction size:", f)

        # Merge auction size to impressions
        imp_with_rank = imp_with_rank.merge(
            ar[['AUCTION_ID', 'PRODUCT_ID', 'n_bidders']].drop_duplicates(),
            on=['AUCTION_ID', 'PRODUCT_ID'],
            how='left'
        )

        for size_range, label in [((1, 10), '1-10'), ((10, 50), '10-50'), ((50, 200), '50-200')]:
            subset = imp_with_rank[
                (imp_with_rank['n_bidders'] >= size_range[0]) &
                (imp_with_rank['n_bidders'] < size_range[1])
            ]
            if len(subset) > 100:
                log(f"  Auction size {label} bidders:", f)
                ctr_subset = subset.groupby('RANKING').agg({
                    'clicked': ['sum', 'count', 'mean']
                }).reset_index()
                ctr_subset.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

                for _, row in ctr_subset[ctr_subset['RANKING'] <= 5].iterrows():
                    log(f"    Rank {int(row['RANKING'])}: CTR = {row['CTR']*100:.3f}% (N = {int(row['impressions']):,})", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Within-Product Position Variation (Priority 5)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: WITHIN-PRODUCT POSITION VARIATION", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Identify products appearing at multiple positions.", f)
        log("", f)

        # Per product: count auctions, unique positions, position range
        product_variation = imp_with_rank.groupby('PRODUCT_ID').agg({
            'AUCTION_ID': 'nunique',
            'RANKING': ['nunique', 'min', 'max'],
            'clicked': ['sum', 'count']
        }).reset_index()
        product_variation.columns = ['PRODUCT_ID', 'n_auctions', 'n_positions', 'min_rank', 'max_rank', 'clicks', 'impressions']
        product_variation['position_range'] = product_variation['max_rank'] - product_variation['min_rank']
        product_variation['ctr'] = product_variation['clicks'] / product_variation['impressions']

        log("Product position variation summary:", f)
        log(f"  Total products with impressions: {len(product_variation):,}", f)
        log(f"  Products with 2+ positions: {(product_variation['n_positions'] >= 2).sum():,}", f)
        log(f"  Products with 5+ auctions: {(product_variation['n_auctions'] >= 5).sum():,}", f)
        log(f"  Products with range > 3: {(product_variation['position_range'] > 3).sum():,}", f)
        log("", f)

        # Products with 5+ auctions AND range > 3
        good_variation = product_variation[
            (product_variation['n_auctions'] >= 5) &
            (product_variation['position_range'] > 3)
        ]
        log(f"Products with 5+ auctions AND range > 3: {len(good_variation):,}", f)
        log("", f)

        # For these: CTR at rank 1-5 vs CTR at rank 6-10
        if len(good_variation) > 10:
            good_products = set(good_variation['PRODUCT_ID'])
            good_imp = imp_with_rank[imp_with_rank['PRODUCT_ID'].isin(good_products)]

            log("CTR comparison for high-variation products:", f)

            # Ranks 1-5
            rank_1_5 = good_imp[good_imp['RANKING'] <= 5]
            ctr_1_5 = rank_1_5['clicked'].mean() * 100 if len(rank_1_5) > 0 else 0

            # Ranks 6-10
            rank_6_10 = good_imp[(good_imp['RANKING'] >= 6) & (good_imp['RANKING'] <= 10)]
            ctr_6_10 = rank_6_10['clicked'].mean() * 100 if len(rank_6_10) > 0 else 0

            log(f"  CTR at ranks 1-5: {ctr_1_5:.3f}% (N = {len(rank_1_5):,})", f)
            log(f"  CTR at ranks 6-10: {ctr_6_10:.3f}% (N = {len(rank_6_10):,})", f)
            log(f"  Difference: {ctr_1_5 - ctr_6_10:.3f}pp", f)
        log("", f)

        # User-product pairs with variation
        log("User-product pairs with position variation:", f)

        user_product = imp_with_rank.groupby(['USER_ID', 'PRODUCT_ID']).agg({
            'AUCTION_ID': 'nunique',
            'RANKING': ['nunique', 'min', 'max']
        }).reset_index()
        user_product.columns = ['USER_ID', 'PRODUCT_ID', 'n_auctions', 'n_positions', 'min_rank', 'max_rank']
        user_product['position_range'] = user_product['max_rank'] - user_product['min_rank']

        log(f"  Total user-product pairs: {len(user_product):,}", f)
        log(f"  Pairs with 2+ positions: {(user_product['n_positions'] >= 2).sum():,}", f)
        log(f"  Pairs with range > 3: {(user_product['position_range'] > 3).sum():,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Placement Deep Dive (Priority 6)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: PLACEMENT DEEP DIVE", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Understand placement-level behavior.", f)
        log("", f)

        # Merge placement to impressions (already done in imp_with_rank)
        placement_stats = imp_with_rank.groupby('PLACEMENT').agg({
            'AUCTION_ID': 'nunique',
            'PRODUCT_ID': 'nunique',
            'clicked': ['sum', 'count', 'mean'],
            'RANKING': 'mean'
        }).reset_index()
        placement_stats.columns = ['PLACEMENT', 'n_auctions', 'n_products', 'clicks', 'impressions', 'ctr', 'avg_rank']

        log("Placement summary:", f)
        log(f"  {'Placement':<12} {'Auctions':<12} {'Impressions':<15} {'CTR %':<10} {'Avg Rank':<10}", f)
        log(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*10} {'-'*10}", f)

        for _, row in placement_stats.iterrows():
            log(f"  {str(row['PLACEMENT']):<12} {int(row['n_auctions']):<12,} {int(row['impressions']):<15,} {row['ctr']*100:<10.3f} {row['avg_rank']:<10.2f}", f)
        log("", f)

        # Impressions per auction by placement
        log("Impressions per auction by placement:", f)

        imp_per_auction = imp_with_rank.groupby(['PLACEMENT', 'AUCTION_ID']).size().reset_index(name='n_imp')
        placement_imp_stats = imp_per_auction.groupby('PLACEMENT')['n_imp'].agg(['mean', 'median', 'max']).reset_index()
        placement_imp_stats.columns = ['PLACEMENT', 'mean_imp', 'median_imp', 'max_imp']

        log(f"  {'Placement':<12} {'Mean':<10} {'Median':<10} {'Max':<10}", f)
        log(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)

        for _, row in placement_imp_stats.iterrows():
            log(f"  {str(row['PLACEMENT']):<12} {row['mean_imp']:<10.2f} {row['median_imp']:<10.0f} {int(row['max_imp']):<10}", f)
        log("", f)

        # Average RANKING of impressed items by placement
        log("Average RANKING of impressed items by placement:", f)
        for placement in sorted(imp_with_rank['PLACEMENT'].dropna().unique()):
            subset = imp_with_rank[imp_with_rank['PLACEMENT'] == placement]
            avg_rank = subset['RANKING'].mean()
            log(f"  Placement {placement}: avg RANKING = {avg_rank:.2f}", f)
        log("", f)

        # Multi-click rate by placement
        log("Multi-click rate by placement:", f)

        clicks_per_auction_placement = imp_with_rank.groupby(['PLACEMENT', 'AUCTION_ID'])['clicked'].sum().reset_index(name='n_clicks')
        multi_click_by_placement = clicks_per_auction_placement.groupby('PLACEMENT').apply(
            lambda x: (x['n_clicks'] >= 2).sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index(name='multi_click_rate')

        for _, row in multi_click_by_placement.iterrows():
            log(f"  Placement {row['PLACEMENT']}: {row['multi_click_rate']:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Multi-Click Patterns (Priority 7)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: MULTI-CLICK PATTERNS", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Understand click behavior within auctions.", f)
        log("", f)

        # Distribution: 0, 1, 2+ clicks per auction
        clicks_per_auction = imp_with_rank.groupby('AUCTION_ID')['clicked'].sum().reset_index(name='n_clicks')

        log("Clicks per auction distribution:", f)
        click_dist = clicks_per_auction['n_clicks'].value_counts().sort_index()
        total_auctions = len(clicks_per_auction)

        for n_clicks in range(min(5, click_dist.index.max() + 1)):
            cnt = click_dist.get(n_clicks, 0)
            pct = cnt / total_auctions * 100
            log(f"  {n_clicks} clicks: {cnt:,} ({pct:.1f}%)", f)

        cnt_2plus = (clicks_per_auction['n_clicks'] >= 2).sum()
        log(f"  2+ clicks: {cnt_2plus:,} ({cnt_2plus/total_auctions*100:.1f}%)", f)
        log("", f)

        # In multi-click auctions: analyze click patterns
        multi_click_auctions = set(clicks_per_auction[clicks_per_auction['n_clicks'] >= 2]['AUCTION_ID'])
        if len(multi_click_auctions) > 10:
            log(f"Multi-click auctions: {len(multi_click_auctions):,}", f)
            log("", f)

            # Get clicked impressions in multi-click auctions
            multi_click_imp = imp_with_rank[
                (imp_with_rank['AUCTION_ID'].isin(multi_click_auctions)) &
                (imp_with_rank['clicked'] == True)
            ].copy()

            # Sort by auction and time
            multi_click_imp = multi_click_imp.sort_values(['AUCTION_ID', 'OCCURRED_AT'])

            # Add click order within auction
            multi_click_imp['click_order'] = multi_click_imp.groupby('AUCTION_ID').cumcount() + 1

            log("Rank of first vs second click:", f)
            first_clicks = multi_click_imp[multi_click_imp['click_order'] == 1]['RANKING']
            second_clicks = multi_click_imp[multi_click_imp['click_order'] == 2]['RANKING']

            if len(first_clicks) > 0 and len(second_clicks) > 0:
                log(f"  First click avg rank: {first_clicks.mean():.2f}", f)
                log(f"  Second click avg rank: {second_clicks.mean():.2f}", f)

                # Sequential vs jumping patterns
                merged = multi_click_imp[multi_click_imp['click_order'] <= 2].pivot(
                    index='AUCTION_ID', columns='click_order', values='RANKING'
                ).dropna()
                if len(merged) > 0:
                    merged.columns = ['first_rank', 'second_rank']
                    merged['rank_diff'] = merged['second_rank'] - merged['first_rank']

                    log("", f)
                    log("Rank difference (second - first):", f)
                    log(f"  Mean: {merged['rank_diff'].mean():.2f}", f)
                    log(f"  % sequential (diff = 1): {(merged['rank_diff'] == 1).mean()*100:.1f}%", f)
                    log(f"  % jumping (|diff| > 1): {(merged['rank_diff'].abs() > 1).mean()*100:.1f}%", f)
                    log(f"  % backwards (diff < 0): {(merged['rank_diff'] < 0).mean()*100:.1f}%", f)
            log("", f)

            # Time between clicks
            log("Time between clicks:", f)
            multi_click_imp['time_diff'] = multi_click_imp.groupby('AUCTION_ID')['OCCURRED_AT'].diff().dt.total_seconds()
            time_diffs = multi_click_imp['time_diff'].dropna()

            if len(time_diffs) > 0:
                log(f"  Mean: {time_diffs.mean():.2f} seconds", f)
                log(f"  Median: {time_diffs.median():.2f} seconds", f)
                log(f"  P95: {time_diffs.quantile(0.95):.2f} seconds", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Purchase Funnel (Priority 8)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: PURCHASE FUNNEL", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Connect impressions/clicks to purchases.", f)
        log("", f)

        log("Purchase data summary:", f)
        log(f"  Total purchases: {len(purchases):,}", f)
        log(f"  Unique users: {purchases['USER_ID'].nunique():,}", f)
        log(f"  Unique products: {purchases['PRODUCT_ID'].nunique():,}", f)
        log("", f)

        # Match purchases to impressions (by user_id + product_id)
        log("Matching purchases to impressions:", f)

        imp_keys = set(zip(imp_with_rank['USER_ID'], imp_with_rank['PRODUCT_ID']))
        purchases['had_impression'] = purchases.apply(
            lambda row: (row['USER_ID'], row['PRODUCT_ID']) in imp_keys, axis=1
        )

        click_keys_up = set(zip(clicks['USER_ID'], clicks['PRODUCT_ID']))
        purchases['had_click'] = purchases.apply(
            lambda row: (row['USER_ID'], row['PRODUCT_ID']) in click_keys_up, axis=1
        )

        log(f"  Purchases with prior impression: {purchases['had_impression'].sum():,} ({purchases['had_impression'].mean()*100:.1f}%)", f)
        log(f"  Purchases with prior click: {purchases['had_click'].sum():,} ({purchases['had_click'].mean()*100:.1f}%)", f)
        log(f"  Purchases with click but no impression: {((~purchases['had_impression']) & purchases['had_click']).sum():,}", f)
        log("", f)

        # Conversion rate by click rank
        log("Conversion rate by click rank:", f)

        # Merge clicks with rank info
        clicks_with_rank = clicks.merge(
            imp_with_rank[['AUCTION_ID', 'PRODUCT_ID', 'USER_ID', 'RANKING']].drop_duplicates(),
            on=['AUCTION_ID', 'PRODUCT_ID', 'USER_ID'],
            how='left'
        )

        # Mark conversions
        purchase_keys = set(zip(purchases['USER_ID'], purchases['PRODUCT_ID']))
        clicks_with_rank['converted'] = clicks_with_rank.apply(
            lambda row: (row['USER_ID'], row['PRODUCT_ID']) in purchase_keys, axis=1
        )

        conv_by_rank = clicks_with_rank.groupby('RANKING').agg({
            'converted': ['sum', 'count', 'mean']
        }).reset_index()
        conv_by_rank.columns = ['RANKING', 'conversions', 'clicks', 'conversion_rate']

        log(f"  {'Rank':<8} {'Clicks':<12} {'Conversions':<15} {'Conv Rate %':<12}", f)
        log(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12}", f)

        for _, row in conv_by_rank[conv_by_rank['RANKING'] <= 10].iterrows():
            log(f"  {int(row['RANKING']):<8} {int(row['clicks']):<12,} {int(row['conversions']):<15,} {row['conversion_rate']*100:<12.2f}", f)
        log("", f)

        # Conversion rate for non-clicked impressed items
        log("Conversion rate: clicked vs non-clicked impressions:", f)

        imp_with_rank['converted'] = imp_with_rank.apply(
            lambda row: (row['USER_ID'], row['PRODUCT_ID']) in purchase_keys, axis=1
        )

        clicked_conv = imp_with_rank[imp_with_rank['clicked'] == True]['converted'].mean() * 100
        not_clicked_conv = imp_with_rank[imp_with_rank['clicked'] == False]['converted'].mean() * 100

        log(f"  Clicked impressions: conversion rate = {clicked_conv:.3f}%", f)
        log(f"  Non-clicked impressions: conversion rate = {not_clicked_conv:.3f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 9: Score Mechanics Validation (Priority 9)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: SCORE MECHANICS VALIDATION", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Validate that RANKING = f(QUALITY, FINAL_BID).", f)
        log("", f)

        # Compute expected_rank = rank of (QUALITY * FINAL_BID) within auction
        log("Computing expected rank from QUALITY * FINAL_BID...", f)

        def compute_expected_rank(group):
            group = group.copy()
            group['score'] = group['QUALITY'] * group['FINAL_BID']
            group['expected_rank'] = group['score'].rank(ascending=False, method='first').astype(int)
            return group

        ar_validated = ar.groupby('AUCTION_ID', group_keys=False).apply(compute_expected_rank)

        # Compare to actual RANKING
        ar_validated['rank_match'] = ar_validated['RANKING'] == ar_validated['expected_rank']
        ar_validated['rank_error'] = (ar_validated['RANKING'] - ar_validated['expected_rank']).abs()

        pct_match = ar_validated['rank_match'].mean() * 100
        correlation = ar_validated[['RANKING', 'expected_rank']].corr().iloc[0, 1]
        mae = ar_validated['rank_error'].mean()

        log(f"  % exact match: {pct_match:.2f}%", f)
        log(f"  Correlation: {correlation:.6f}", f)
        log(f"  MAE: {mae:.4f}", f)
        log("", f)

        # Check if QUALITY alone determines rank
        log("Testing if QUALITY alone determines rank:", f)

        def compute_quality_rank(group):
            group = group.copy()
            group['quality_rank'] = group['QUALITY'].rank(ascending=False, method='first').astype(int)
            return group

        ar_quality = ar.groupby('AUCTION_ID', group_keys=False).apply(compute_quality_rank)
        ar_quality['quality_match'] = ar_quality['RANKING'] == ar_quality['quality_rank']

        pct_quality_match = ar_quality['quality_match'].mean() * 100
        quality_correlation = ar_quality[['RANKING', 'quality_rank']].corr().iloc[0, 1]

        log(f"  % exact match (QUALITY only): {pct_quality_match:.2f}%", f)
        log(f"  Correlation (QUALITY only): {quality_correlation:.6f}", f)
        log("", f)

        # Cases where expected_rank differs from actual RANKING
        mismatches = ar_validated[ar_validated['rank_match'] == False]
        log(f"Mismatches (expected != actual): {len(mismatches):,} ({len(mismatches)/len(ar_validated)*100:.2f}%)", f)

        if len(mismatches) > 0:
            log("", f)
            log("Mismatch analysis:", f)
            log(f"  Mean rank error in mismatches: {mismatches['rank_error'].mean():.2f}", f)
            log(f"  Max rank error: {mismatches['rank_error'].max():.0f}", f)

            # Check if mismatches are due to ties
            log("", f)
            log("Checking for tied scores within mismatched auctions:", f)
            mismatch_auctions = mismatches['AUCTION_ID'].unique()
            sample_auctions = mismatch_auctions[:min(100, len(mismatch_auctions))]

            tied_count = 0
            for auction_id in sample_auctions:
                auction_data = ar_validated[ar_validated['AUCTION_ID'] == auction_id]
                if auction_data['score'].duplicated().any():
                    tied_count += 1

            log(f"  Auctions with tied scores (sample of {len(sample_auctions)}): {tied_count} ({tied_count/len(sample_auctions)*100:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 10: Competition as Instrument (Priority 10)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 10: COMPETITION AS INSTRUMENT", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Assess viability of auction competition as instrument.", f)
        log("", f)

        # Distribution of bidders per auction
        log("Distribution of bidders per auction:", f)
        bidder_dist = ar.groupby('AUCTION_ID').size().reset_index(name='n_bidders_check')

        log(f"  Mean: {bidder_dist['n_bidders_check'].mean():.1f}", f)
        log(f"  Median: {bidder_dist['n_bidders_check'].median():.0f}", f)
        log(f"  P25: {bidder_dist['n_bidders_check'].quantile(0.25):.0f}", f)
        log(f"  P75: {bidder_dist['n_bidders_check'].quantile(0.75):.0f}", f)
        log(f"  P95: {bidder_dist['n_bidders_check'].quantile(0.95):.0f}", f)
        log(f"  Max: {bidder_dist['n_bidders_check'].max():.0f}", f)
        log("", f)

        # Correlation: num_bidders -> RANKING for same product
        log("Effect of competition on RANKING for same product:", f)

        # For products appearing in multiple auctions, check if more bidders -> lower rank
        product_auction = ar.groupby(['PRODUCT_ID', 'AUCTION_ID']).agg({
            'RANKING': 'first',
            'n_bidders': 'first'
        }).reset_index()

        products_multi = product_auction.groupby('PRODUCT_ID').filter(lambda x: len(x) >= 5)

        if len(products_multi) > 100:
            # Within-product correlation
            within_corrs = []
            for pid, group in tqdm(products_multi.groupby('PRODUCT_ID'), desc="Computing within-product correlations"):
                if len(group) >= 5:
                    corr = group[['n_bidders', 'RANKING']].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        within_corrs.append(corr)

            if within_corrs:
                log(f"  Within-product correlation (n_bidders vs RANKING):", f)
                log(f"    Mean: {np.mean(within_corrs):.4f}", f)
                log(f"    Median: {np.median(within_corrs):.4f}", f)
                log(f"    % positive (more competition -> lower rank): {(np.array(within_corrs) > 0).mean()*100:.1f}%", f)
        log("", f)

        # First stage: RANKING ~ num_bidders + quality + bid
        log("First stage regression: RANKING ~ num_bidders + quality + bid", f)

        # Prepare data for impressions only
        first_stage_data = imp_with_rank[['RANKING', 'n_bidders', 'QUALITY', 'FINAL_BID']].dropna()

        if len(first_stage_data) > 100:
            X = first_stage_data[['n_bidders', 'QUALITY', 'FINAL_BID']]
            y = first_stage_data['RANKING']

            model = LinearRegression()
            model.fit(X, y)

            log(f"  Coefficients:", f)
            for feat, coef in zip(X.columns, model.coef_):
                log(f"    {feat}: {coef:.4f}", f)
            log(f"  Intercept: {model.intercept_:.4f}", f)
            log(f"  R2: {model.score(X, y):.4f}", f)
            log("", f)

            # F-statistic for num_bidders
            # Run with and without num_bidders
            X_full = first_stage_data[['n_bidders', 'QUALITY', 'FINAL_BID']]
            X_restricted = first_stage_data[['QUALITY', 'FINAL_BID']]

            model_full = LinearRegression()
            model_full.fit(X_full, y)
            ss_res_full = ((y - model_full.predict(X_full)) ** 2).sum()

            model_restricted = LinearRegression()
            model_restricted.fit(X_restricted, y)
            ss_res_restricted = ((y - model_restricted.predict(X_restricted)) ** 2).sum()

            n = len(y)
            p_full = 3
            p_restricted = 2

            f_stat = ((ss_res_restricted - ss_res_full) / (p_full - p_restricted)) / (ss_res_full / (n - p_full))

            log(f"  F-statistic for num_bidders: {f_stat:.2f}", f)
            log(f"  Threshold for strong instrument: F > 10", f)
            log(f"  Status: {'STRONG' if f_stat > 10 else 'WEAK'} instrument", f)
        log("", f)

        # Exclusion restriction check
        log("Exclusion restriction discussion:", f)
        log("  For num_bidders to be valid instrument:", f)
        log("  1. Relevance: num_bidders affects RANKING (tested above)", f)
        log("  2. Exclusion: num_bidders affects CTR only through RANKING", f)
        log("  Potential violations:", f)
        log("    - More bidders may indicate more popular products (selection)", f)
        log("    - More bidders may occur in busier sessions (user engagement)", f)
        log("    - Auction size correlates with placement type", f)
        log("", f)

        # Check correlation of num_bidders with CTR directly
        log("Direct effect of num_bidders on CTR:", f)

        ctr_by_size = imp_with_rank.groupby('n_bidders').agg({
            'clicked': 'mean'
        }).reset_index()
        ctr_by_size.columns = ['n_bidders', 'ctr']

        corr = ctr_by_size[['n_bidders', 'ctr']].corr().iloc[0, 1]
        log(f"  Correlation (num_bidders vs CTR): {corr:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 11: Summary Tables
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 11: SUMMARY TABLES", f)
        log("=" * 80, f)
        log("", f)

        log("KEY NUMBERS COMPILATION", f)
        log("-" * 40, f)
        log("", f)

        log("Data Volume:", f)
        log(f"  Auctions: {ar['AUCTION_ID'].nunique():,}", f)
        log(f"  Bids: {len(ar):,}", f)
        log(f"  Impressions: {len(imp):,}", f)
        log(f"  Clicks: {len(clicks):,}", f)
        log(f"  Purchases: {len(purchases):,}", f)
        log(f"  Users: {au['USER_ID'].nunique():,}", f)
        log("", f)

        log("Rates:", f)
        log(f"  CTR (overall): {imp_with_rank['clicked'].mean()*100:.3f}%", f)
        log(f"  Conversion rate (click->purchase): {clicked_conv:.3f}%", f)
        log(f"  Winner rate: {ar['IS_WINNER'].mean()*100:.2f}%", f)
        log(f"  Impression rate (for winners): {ar[ar['IS_WINNER']==True]['got_impression'].mean()*100:.2f}%", f)
        log("", f)

        log("Position Variation:", f)
        log(f"  Products with 5+ auctions AND range > 3: {len(good_variation):,}", f)
        log(f"  % of impressed products: {len(good_variation)/len(product_variation)*100:.1f}%", f)
        log("", f)

        log("Score Mechanics:", f)
        log(f"  QUALITY*BID predicts RANKING: {pct_match:.1f}% exact match", f)
        log(f"  QUALITY alone predicts RANKING: {pct_quality_match:.1f}% exact match", f)
        log("", f)

        log("RDD Feasibility:", f)
        gap_below_001 = (gaps.abs() < 0.001).mean() * 100
        log(f"  Score gaps < 0.001: {gap_below_001:.1f}%", f)
        log(f"  Marginal winners: {len(marginal_winners):,}", f)
        log(f"  Marginal losers: {len(marginal_losers):,}", f)
        log("", f)

        log("Instrument Strength:", f)
        if 'f_stat' in dir():
            log(f"  F-statistic for num_bidders: {f_stat:.2f}", f)
            log(f"  Status: {'STRONG' if f_stat > 10 else 'WEAK'}", f)
        log("", f)

        log("=" * 80, f)
        log("METHOD VIABILITY ASSESSMENT", f)
        log("=" * 80, f)
        log("", f)

        log(f"{'Method':<30} {'Key Metric':<25} {'Value':<15} {'Viability':<15}", f)
        log(f"{'-'*30} {'-'*25} {'-'*15} {'-'*15}", f)

        # 1. Within-product variation (PBM)
        pct_good_var = len(good_variation) / len(product_variation) * 100 if len(product_variation) > 0 else 0
        pbm_viable = pct_good_var > 5
        log(f"{'1. PBM (within-product)':<30} {'Products w/ variation':<25} {pct_good_var:.1f}%{'':<8} {'YES' if pbm_viable else 'LIMITED':<15}", f)

        # 2. RDD at cutoff
        rdd_viable = len(marginal_winners) > 5000 and len(marginal_losers) > 5000
        log(f"{'2. RDD at winner cutoff':<30} {'Marginal sample':<25} {len(marginal_winners)+len(marginal_losers):,}{'':<5} {'YES' if rdd_viable else 'LIMITED':<15}", f)

        # 3. Score-based RDD
        score_rdd_viable = gap_below_001 > 30
        log(f"{'3. Score-based RDD':<30} {'Gaps < 0.001':<25} {gap_below_001:.1f}%{'':<8} {'YES' if score_rdd_viable else 'LIMITED':<15}", f)

        # 4. IV with competition
        iv_viable = f_stat > 10 if 'f_stat' in dir() else False
        iv_value = f"{f_stat:.1f}" if 'f_stat' in dir() else "N/A"
        log(f"{'4. IV (competition)':<30} {'F-statistic':<25} {iv_value:<15} {'YES' if iv_viable else 'WEAK':<15}", f)

        # 5. Selection model
        selection_viable = pseudo_r2 > 0.1
        log(f"{'5. Selection model':<30} {'Pseudo-R2':<25} {pseudo_r2:.3f}{'':<9} {'YES' if selection_viable else 'LIMITED':<15}", f)

        log("", f)
        log("=" * 80, f)
        log("CAUSAL EDA COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
