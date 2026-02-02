#!/usr/bin/env python3
"""
Position Effects EDA: Viability Assessment for Causal Inference Methods

Assesses feasibility of 5 methodological approaches:
1. PBM/Click Models (same ad at multiple positions)
2. IPW/Doubly Robust (propensity overlap)
3. RDD at Auction Margins (score discontinuity)
4. Survival/Hazard Models (sequential examination)
5. Partial Identification (bounds under minimal assumptions)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "02_position_effects_eda.txt"

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
        log("POSITION EFFECTS EDA: METHOD VIABILITY ASSESSMENT", f)
        log("=" * 80, f)
        log("", f)

        log("OBJECTIVE:", f)
        log("  Assess viability of causal inference methods for estimating position", f)
        log("  effects in sponsored search advertising.", f)
        log("", f)

        log("CORE PROBLEM:", f)
        log("  Organic clicks/impressions are unobserved, creating latent truncation.", f)
        log("  We need to determine which methodologies are feasible with available data.", f)
        log("", f)

        log("METHODS UNDER EVALUATION:", f)
        log("  1. PBM/Click Models - Requires same ad at multiple positions", f)
        log("  2. IPW/Doubly Robust - Requires propensity overlap", f)
        log("  3. RDD at Margins - Requires exploitable score discontinuity", f)
        log("  4. Survival/Hazard - Requires sequential examination assumption", f)
        log("  5. Partial Identification - Bounds under minimal assumptions", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 0: Data Loading and Basic Stats
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 0: DATA LOADING AND BASIC STATS", f)
        log("-" * 40, f)
        log("", f)

        # Check for data files
        ar_path = DATA_DIR / "auctions_results_all.parquet"
        au_path = DATA_DIR / "auctions_users_all.parquet"
        imp_path = DATA_DIR / "impressions_all.parquet"
        clicks_path = DATA_DIR / "clicks_all.parquet"
        catalog_path = DATA_DIR / "catalog_all.parquet"

        files_exist = {
            'auctions_results': ar_path.exists(),
            'auctions_users': au_path.exists(),
            'impressions': imp_path.exists(),
            'clicks': clicks_path.exists(),
            'catalog': catalog_path.exists()
        }

        log("File availability:", f)
        for name, exists in files_exist.items():
            status = "FOUND" if exists else "MISSING"
            log(f"  {name}: {status}", f)
        log("", f)

        # Check minimum required files
        required_files = ['auctions_results', 'auctions_users', 'impressions', 'clicks']
        missing = [k for k in required_files if not files_exist.get(k, False)]
        if missing:
            log(f"ERROR: Required data files not found: {missing}", f)
            log("Please run 01_data_pull.ipynb first.", f)
            return

        # Load data
        log("Loading data files...", f)

        ar = pd.read_parquet(ar_path)
        log(f"  auctions_results: {len(ar):,} rows, {ar.shape[1]} columns", f)

        au = pd.read_parquet(au_path)
        log(f"  auctions_users: {len(au):,} rows, {au.shape[1]} columns", f)

        imp = pd.read_parquet(imp_path)
        log(f"  impressions: {len(imp):,} rows, {imp.shape[1]} columns", f)

        clicks = pd.read_parquet(clicks_path)
        log(f"  clicks: {len(clicks):,} rows, {clicks.shape[1]} columns", f)

        # Load catalog if available
        catalog = None
        if files_exist.get('catalog', False):
            catalog = pd.read_parquet(catalog_path)
            log(f"  catalog: {len(catalog):,} rows, {catalog.shape[1]} columns", f)
        log("", f)

        # Column summaries
        log("AUCTIONS_RESULTS columns:", f)
        for col in ar.columns:
            dtype = ar[col].dtype
            n_unique = ar[col].nunique()
            n_null = ar[col].isna().sum()
            log(f"  {col}: dtype={dtype}, unique={n_unique:,}, null={n_null:,}", f)
        log("", f)

        # Date ranges
        log("Date ranges:", f)
        log(f"  auctions_results: {ar['CREATED_AT'].min()} to {ar['CREATED_AT'].max()}", f)
        log(f"  auctions_users: {au['CREATED_AT'].min()} to {au['CREATED_AT'].max()}", f)
        log(f"  impressions: {imp['OCCURRED_AT'].min()} to {imp['OCCURRED_AT'].max()}", f)
        log(f"  clicks: {clicks['OCCURRED_AT'].min()} to {clicks['OCCURRED_AT'].max()}", f)
        log("", f)

        # Basic counts
        log("Basic counts:", f)
        log(f"  Unique auctions (AR): {ar['AUCTION_ID'].nunique():,}", f)
        log(f"  Unique auctions (AU): {au['AUCTION_ID'].nunique():,}", f)
        log(f"  Unique vendors: {ar['VENDOR_ID'].nunique():,}", f)
        log(f"  Unique products: {ar['PRODUCT_ID'].nunique():,}", f)
        log(f"  Unique users: {au['USER_ID'].nunique():,}", f)
        log("", f)

        # Join validation
        log("Join validation (auction_id overlaps):", f)
        ar_auctions = set(ar['AUCTION_ID'].unique())
        au_auctions = set(au['AUCTION_ID'].unique())
        imp_auctions = set(imp['AUCTION_ID'].unique())
        click_auctions = set(clicks['AUCTION_ID'].unique())

        ar_au_overlap = len(ar_auctions & au_auctions) / len(ar_auctions) * 100
        imp_au_overlap = len(imp_auctions & au_auctions) / len(imp_auctions) * 100 if len(imp_auctions) > 0 else 0
        click_au_overlap = len(click_auctions & au_auctions) / len(click_auctions) * 100 if len(click_auctions) > 0 else 0

        log(f"  AR ∩ AU: {ar_au_overlap:.1f}%", f)
        log(f"  IMP ∩ AU: {imp_au_overlap:.1f}%", f)
        log(f"  CLICK ∩ AU: {click_au_overlap:.1f}%", f)
        log("", f)

        # RANKING distribution
        log("RANKING distribution:", f)
        ranking_stats = ar['RANKING'].describe()
        for stat, val in ranking_stats.items():
            log(f"  {stat}: {val:.2f}", f)
        log("", f)

        # IS_WINNER distribution
        log("IS_WINNER distribution:", f)
        winner_counts = ar['IS_WINNER'].value_counts()
        for val, cnt in winner_counts.items():
            pct = cnt / len(ar) * 100
            log(f"  {val}: {cnt:,} ({pct:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: PBM/Click Model Feasibility
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: PBM/CLICK MODEL FEASIBILITY", f)
        log("-" * 40, f)
        log("", f)

        log("REQUIREMENT: Same ad (VENDOR_ID, PRODUCT_ID) appears at multiple positions.", f)
        log("", f)

        # Focus on winners only (ads that actually appeared)
        winners = ar[ar['IS_WINNER'] == True].copy()
        log(f"Winner bids: {len(winners):,}", f)
        log("", f)

        # Group by (VENDOR_ID, PRODUCT_ID) and count unique rankings
        log("Analyzing position variation for winner ads...", f)

        ad_positions = winners.groupby(['VENDOR_ID', 'PRODUCT_ID']).agg({
            'RANKING': ['nunique', 'min', 'max', 'count'],
            'AUCTION_ID': 'nunique'
        }).reset_index()
        ad_positions.columns = ['VENDOR_ID', 'PRODUCT_ID', 'n_positions', 'min_rank', 'max_rank', 'n_wins', 'n_auctions']
        ad_positions['position_range'] = ad_positions['max_rank'] - ad_positions['min_rank']

        total_ads = len(ad_positions)
        log(f"Total unique (vendor, product) combinations with wins: {total_ads:,}", f)
        log("", f)

        # Diagnostic 1: Ads with 2+ positions
        ads_2plus = (ad_positions['n_positions'] >= 2).sum()
        pct_2plus = ads_2plus / total_ads * 100
        log(f"Ads with 2+ positions: {ads_2plus:,} ({pct_2plus:.1f}%)", f)
        log(f"  Viability threshold: >5%", f)
        log(f"  Status: {'VIABLE' if pct_2plus > 5 else 'NOT VIABLE'}", f)
        log("", f)

        # Diagnostic 2: Ads with 3+ positions
        ads_3plus = (ad_positions['n_positions'] >= 3).sum()
        pct_3plus = ads_3plus / total_ads * 100
        log(f"Ads with 3+ positions: {ads_3plus:,} ({pct_3plus:.1f}%)", f)
        log(f"  Viability threshold: >1%", f)
        log(f"  Status: {'VIABLE' if pct_3plus > 1 else 'NOT VIABLE'}", f)
        log("", f)

        # Diagnostic 3: Position range per ad
        mean_range = ad_positions['position_range'].mean()
        log(f"Position range per ad:", f)
        log(f"  Mean: {mean_range:.2f}", f)
        log(f"  Median: {ad_positions['position_range'].median():.2f}", f)
        log(f"  Max: {ad_positions['position_range'].max():.0f}", f)
        log(f"  Viability threshold: Mean >3", f)
        log(f"  Status: {'VIABLE' if mean_range > 3 else 'NOT VIABLE'}", f)
        log("", f)

        # Diagnostic 4: CTR by position (winners)
        log("CTR by position (winners only):", f)

        # Merge clicks to winners
        click_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
        winners['clicked'] = winners.apply(
            lambda row: (row['AUCTION_ID'], row['PRODUCT_ID']) in click_set, axis=1
        )

        ctr_by_rank = winners.groupby('RANKING').agg({
            'clicked': ['sum', 'count', 'mean']
        }).reset_index()
        ctr_by_rank.columns = ['RANKING', 'clicks', 'impressions', 'CTR']

        # Show top 20 positions
        log(f"  {'Rank':<8} {'Impressions':<15} {'Clicks':<12} {'CTR %':<10}", f)
        log(f"  {'-'*8} {'-'*15} {'-'*12} {'-'*10}", f)

        for _, row in ctr_by_rank.head(20).iterrows():
            log(f"  {int(row['RANKING']):<8} {int(row['impressions']):<15,} {int(row['clicks']):<12,} {row['CTR']*100:<10.3f}", f)

        # Check monotonicity
        ctr_values = ctr_by_rank.head(10)['CTR'].values
        is_monotonic = all(ctr_values[i] >= ctr_values[i+1] for i in range(len(ctr_values)-1))
        log("", f)
        log(f"CTR monotonically decreasing (top 10 positions): {is_monotonic}", f)
        log(f"  Viability: {'EXPECTED PATTERN' if is_monotonic else 'UNEXPECTED - investigate'}", f)
        log("", f)

        # PBM viability summary
        pbm_viable = pct_3plus > 1
        log(f"PBM/CLICK MODEL VIABILITY: {'YES' if pbm_viable else 'NO'}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: IPW/Doubly Robust Feasibility
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: IPW/DOUBLY ROBUST FEASIBILITY", f)
        log("-" * 40, f)
        log("", f)

        log("REQUIREMENT: Overlap in propensity scores between winners and losers.", f)
        log("", f)

        # Prepare data for propensity model
        log("Preparing propensity model data...", f)

        # Filter to rows with required features
        prop_data = ar[['IS_WINNER', 'FINAL_BID', 'QUALITY', 'PACING', 'AUCTION_ID']].dropna()
        log(f"  Rows with complete features: {len(prop_data):,} / {len(ar):,} ({len(prop_data)/len(ar)*100:.1f}%)", f)
        log("", f)

        if len(prop_data) < 1000:
            log("  WARNING: Insufficient data for propensity model.", f)
            log("  IPW/DR VIABILITY: CANNOT ASSESS", f)
            ipw_viable = False
        else:
            # Compute within-auction competitor features
            log("Computing within-auction competitor features...", f)

            auction_stats = prop_data.groupby('AUCTION_ID').agg({
                'FINAL_BID': ['mean', 'max', 'std'],
                'QUALITY': ['mean', 'max'],
                'IS_WINNER': 'sum'
            }).reset_index()
            auction_stats.columns = ['AUCTION_ID', 'auction_mean_bid', 'auction_max_bid', 'auction_std_bid',
                                     'auction_mean_quality', 'auction_max_quality', 'n_winners']

            prop_data = prop_data.merge(auction_stats, on='AUCTION_ID', how='left')

            # Feature: relative bid (own bid / auction mean)
            prop_data['relative_bid'] = prop_data['FINAL_BID'] / prop_data['auction_mean_bid'].clip(lower=1)

            # Feature: relative quality
            prop_data['relative_quality'] = prop_data['QUALITY'] / prop_data['auction_mean_quality'].clip(lower=0.001)

            log(f"  Competitor features computed.", f)
            log("", f)

            # Fit logistic regression
            log("Fitting winner prediction model...", f)

            feature_cols = ['FINAL_BID', 'QUALITY', 'PACING', 'relative_bid', 'relative_quality']
            X = prop_data[feature_cols].fillna(0)
            y = prop_data['IS_WINNER'].astype(int)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled, y)

            # Predict probabilities
            prop_data['propensity'] = model.predict_proba(X_scaled)[:, 1]

            # Compute AUC
            auc = roc_auc_score(y, prop_data['propensity'])
            log(f"  AUC: {auc:.4f}", f)
            log(f"  Viability threshold: >0.6", f)
            log(f"  Status: {'VIABLE' if auc > 0.6 else 'WEAK - proceed with caution'}", f)
            log("", f)

            # Propensity distribution
            log("Propensity score distribution:", f)

            winners_prop = prop_data[prop_data['IS_WINNER'] == True]['propensity']
            losers_prop = prop_data[prop_data['IS_WINNER'] == False]['propensity']

            log(f"  Winners (N={len(winners_prop):,}):", f)
            log(f"    Mean: {winners_prop.mean():.4f}", f)
            log(f"    Median: {winners_prop.median():.4f}", f)
            log(f"    P10: {winners_prop.quantile(0.10):.4f}", f)
            log(f"    P90: {winners_prop.quantile(0.90):.4f}", f)
            log("", f)

            log(f"  Losers (N={len(losers_prop):,}):", f)
            log(f"    Mean: {losers_prop.mean():.4f}", f)
            log(f"    Median: {losers_prop.median():.4f}", f)
            log(f"    P10: {losers_prop.quantile(0.10):.4f}", f)
            log(f"    P90: {losers_prop.quantile(0.90):.4f}", f)
            log("", f)

            # Overlap assessment
            # Common support: proportion of winners with propensity in loser range
            loser_min = losers_prop.quantile(0.05)
            loser_max = losers_prop.quantile(0.95)
            winner_in_loser_range = ((winners_prop >= loser_min) & (winners_prop <= loser_max)).mean()

            log(f"Common support (overlap):", f)
            log(f"  Loser propensity range (P5-P95): [{loser_min:.4f}, {loser_max:.4f}]", f)
            log(f"  Winners in loser range: {winner_in_loser_range*100:.1f}%", f)
            log(f"  Viability threshold: >50%", f)
            log(f"  Status: {'VIABLE' if winner_in_loser_range > 0.5 else 'NOT VIABLE - limited overlap'}", f)
            log("", f)

            # Effective sample size
            # For IPW, ESS = (sum of weights)^2 / sum of weights^2
            # Using inverse propensity weights for losers
            loser_weights = prop_data[prop_data['IS_WINNER'] == False]['propensity'] / (1 - prop_data[prop_data['IS_WINNER'] == False]['propensity'].clip(upper=0.99))
            ess = (loser_weights.sum() ** 2) / (loser_weights ** 2).sum()
            ess_pct = ess / len(losers_prop) * 100

            log(f"Effective sample size (ESS):", f)
            log(f"  ESS: {ess:,.0f}", f)
            log(f"  ESS as % of control N: {ess_pct:.1f}%", f)
            log(f"  Viability threshold: >50%", f)
            log(f"  Status: {'VIABLE' if ess_pct > 50 else 'NOT VIABLE - extreme weights'}", f)
            log("", f)

            ipw_viable = auc > 0.6 and winner_in_loser_range > 0.5

            log(f"IPW/DR VIABILITY: {'YES' if ipw_viable else 'NO'}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: RDD at Auction Margins Feasibility
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: RDD AT AUCTION MARGINS FEASIBILITY", f)
        log("-" * 40, f)
        log("", f)

        log("REQUIREMENT: Score discontinuity at win/lose margin is exploitable.", f)
        log("", f)

        # Score construction
        log("Score construction:", f)
        log("  Testing: score = f(FINAL_BID, QUALITY, PACING)", f)
        log("", f)

        # Use rows with all features
        rdd_data = ar[['RANKING', 'FINAL_BID', 'QUALITY', 'PACING', 'IS_WINNER', 'AUCTION_ID']].dropna()
        log(f"  Rows with complete features: {len(rdd_data):,}", f)
        log("", f)

        if len(rdd_data) < 1000:
            log("  WARNING: Insufficient data for RDD analysis.", f)
            log("  RDD VIABILITY: CANNOT ASSESS", f)
            rdd_viable = False
        else:
            # Compute composite score
            # Try: score = bid * quality
            rdd_data['score_v1'] = rdd_data['FINAL_BID'] * rdd_data['QUALITY']

            # Try: score = bid * quality * pacing
            rdd_data['score_v2'] = rdd_data['FINAL_BID'] * rdd_data['QUALITY'] * rdd_data['PACING']

            # Regress RANKING on scores within auction
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            # Sample auctions for speed
            sample_auctions = rdd_data['AUCTION_ID'].drop_duplicates().sample(min(10000, rdd_data['AUCTION_ID'].nunique()), random_state=42)
            rdd_sample = rdd_data[rdd_data['AUCTION_ID'].isin(sample_auctions)]

            # Model 1: score_v1
            X1 = rdd_sample[['score_v1']]
            y = rdd_sample['RANKING']
            model1 = LinearRegression()
            model1.fit(X1, y)
            r2_v1 = r2_score(y, model1.predict(X1))

            # Model 2: score_v2
            X2 = rdd_sample[['score_v2']]
            model2 = LinearRegression()
            model2.fit(X2, y)
            r2_v2 = r2_score(y, model2.predict(X2))

            # Model 3: all features
            X3 = rdd_sample[['FINAL_BID', 'QUALITY', 'PACING']]
            model3 = LinearRegression()
            model3.fit(X3, y)
            r2_v3 = r2_score(y, model3.predict(X3))

            log("Score-to-Ranking R² (higher = more deterministic):", f)
            log(f"  BID * QUALITY:                {r2_v1:.4f}", f)
            log(f"  BID * QUALITY * PACING:       {r2_v2:.4f}", f)
            log(f"  BID + QUALITY + PACING (OLS): {r2_v3:.4f}", f)
            log("", f)

            # Compute margins within auctions
            log("Computing within-auction margins...", f)

            # For each auction, compute score gap between adjacent ranks
            def compute_margins(group):
                group = group.sort_values('RANKING')
                group['score'] = group['score_v2']
                group['score_next'] = group['score'].shift(-1)
                group['margin'] = group['score'] - group['score_next']
                return group

            margins_data = rdd_sample.groupby('AUCTION_ID', group_keys=False).apply(compute_margins)
            margins_data = margins_data[margins_data['margin'].notna()]

            log(f"  Margin observations: {len(margins_data):,}", f)
            log("", f)

            # Margin distribution
            log("Margin distribution (score_k - score_{k+1}):", f)
            margin_stats = margins_data['margin'].describe()
            for stat, val in margin_stats.items():
                log(f"  {stat}: {val:.4f}", f)
            log("", f)

            # McCrary density test proxy
            # Check if margins cluster around 0 (manipulation) or are smooth
            log("McCrary density test proxy:", f)
            margin_near_zero = (margins_data['margin'].abs() < margins_data['margin'].std() * 0.1).mean()
            margin_positive = (margins_data['margin'] > 0).mean()
            margin_negative = (margins_data['margin'] < 0).mean()

            log(f"  Margins near zero (within 10% of std): {margin_near_zero*100:.1f}%", f)
            log(f"  Positive margins: {margin_positive*100:.1f}%", f)
            log(f"  Negative margins: {margin_negative*100:.1f}%", f)
            log("", f)

            # Check if IS_WINNER is deterministic in RANKING
            log("Sharp RD check:", f)
            # Within each auction, is winner = rank 1?
            rdd_sample['is_rank_1'] = rdd_sample['RANKING'] == 1
            winner_is_rank_1 = (rdd_sample[rdd_sample['IS_WINNER'] == True]['is_rank_1']).mean()
            log(f"  Winners at Rank 1: {winner_is_rank_1*100:.1f}%", f)

            # Actually, IS_WINNER can be true for multiple ranks (slots)
            # Check: what fraction of IS_WINNER=True are in top K ranks?
            top_k = 10
            winner_in_top_k = (rdd_sample[rdd_sample['IS_WINNER'] == True]['RANKING'] <= top_k).mean()
            log(f"  Winners in top {top_k} ranks: {winner_in_top_k*100:.1f}%", f)
            log("", f)

            # Sample in optimal bandwidth
            # Use Imbens-Kalyanaraman optimal bandwidth (simplified: use margins within 1 std)
            optimal_bw = margins_data['margin'].std()
            sample_in_bw = (margins_data['margin'].abs() <= optimal_bw).sum()
            log(f"Sample in bandwidth (margin within 1 std):", f)
            log(f"  Bandwidth: {optimal_bw:.4f}", f)
            log(f"  Sample in bandwidth: {sample_in_bw:,}", f)
            log(f"  Viability threshold: >5000", f)
            log(f"  Status: {'VIABLE' if sample_in_bw > 5000 else 'NOT VIABLE - insufficient local sample'}", f)
            log("", f)

            rdd_viable = r2_v3 > 0.5 and sample_in_bw > 5000

            log(f"RDD VIABILITY: {'YES' if rdd_viable else 'NO (high R² suggests determinism, RDD may not add value)'}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Survival/Hazard Model Feasibility
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: SURVIVAL/HAZARD MODEL FEASIBILITY", f)
        log("-" * 40, f)
        log("", f)

        log("REQUIREMENT: Users examine positions sequentially (cascade model).", f)
        log("", f)

        # Click position distribution
        log("Click position distribution:", f)

        # Merge clicks to winners to get ranking of clicked ads
        winners_with_clicks = winners.merge(
            clicks[['AUCTION_ID', 'PRODUCT_ID']].drop_duplicates(),
            on=['AUCTION_ID', 'PRODUCT_ID'],
            how='inner'
        )

        if len(winners_with_clicks) == 0:
            log("  WARNING: No clicks matched to winners.", f)
            log("  SURVIVAL/HAZARD VIABILITY: CANNOT ASSESS", f)
            survival_viable = False
        else:
            click_rank_dist = winners_with_clicks['RANKING'].value_counts().sort_index()

            log(f"  Total clicked ads: {len(winners_with_clicks):,}", f)
            log("", f)

            log(f"  {'Rank':<8} {'Clicks':<12} {'Cumulative %':<15}", f)
            log(f"  {'-'*8} {'-'*12} {'-'*15}", f)

            cumsum = 0
            total_clicks = len(winners_with_clicks)
            for rank in sorted(click_rank_dist.index[:15]):
                cnt = click_rank_dist[rank]
                cumsum += cnt
                cumpct = cumsum / total_clicks * 100
                log(f"  {rank:<8} {cnt:<12,} {cumpct:<15.1f}", f)

            log("", f)

            # Click concentration ratio
            clicks_top_3 = click_rank_dist.head(3).sum() if len(click_rank_dist) >= 3 else click_rank_dist.sum()
            concentration_ratio = clicks_top_3 / total_clicks

            log(f"Click concentration ratio (ranks 1-3 / all):", f)
            log(f"  Ratio: {concentration_ratio:.3f}", f)
            log(f"  Viability threshold: >0.5", f)
            log(f"  Status: {'VIABLE' if concentration_ratio > 0.5 else 'NOT VIABLE - clicks too dispersed'}", f)
            log("", f)

            # Hazard by position
            log("Hazard rate by position:", f)
            log("  (clicks at position j / impressions reaching position j)", f)
            log("", f)

            # For each position, compute: clicks / (impressions that reached that position)
            # Simplified: assume all users see all positions up to their click (or all if no click)
            # This is a strong assumption but provides a starting point

            # Count impressions at each rank
            imp_by_rank = winners.groupby('RANKING').size().sort_index()

            # Count clicks at each rank
            click_by_rank = winners_with_clicks.groupby('RANKING').size().sort_index()

            # Compute hazard
            hazard_data = pd.DataFrame({
                'impressions': imp_by_rank,
                'clicks': click_by_rank
            }).fillna(0)
            hazard_data['hazard'] = hazard_data['clicks'] / hazard_data['impressions'].clip(lower=1)

            log(f"  {'Rank':<8} {'Impressions':<15} {'Clicks':<12} {'Hazard':<10}", f)
            log(f"  {'-'*8} {'-'*15} {'-'*12} {'-'*10}", f)

            for rank in hazard_data.index[:15]:
                row = hazard_data.loc[rank]
                log(f"  {rank:<8} {int(row['impressions']):<15,} {int(row['clicks']):<12,} {row['hazard']:<10.4f}", f)

            log("", f)

            # Check monotonicity of hazard
            hazard_values = hazard_data.head(10)['hazard'].values
            hazard_monotonic = all(hazard_values[i] >= hazard_values[i+1] for i in range(len(hazard_values)-1) if hazard_values[i+1] > 0)
            log(f"Hazard monotonically decreasing (top 10): {hazard_monotonic}", f)
            log("", f)

            # Multi-click rate
            log("Multi-click rate:", f)
            clicks_per_auction = clicks.groupby('AUCTION_ID').size()
            multi_click_auctions = (clicks_per_auction >= 2).sum()
            multi_click_rate = multi_click_auctions / len(clicks_per_auction) * 100

            log(f"  Auctions with 2+ clicks: {multi_click_auctions:,} ({multi_click_rate:.1f}%)", f)
            log(f"  Viability threshold: <10%", f)
            log(f"  Status: {'VIABLE' if multi_click_rate < 10 else 'QUESTIONABLE - high multi-click rate'}", f)
            log("", f)

            survival_viable = concentration_ratio > 0.5 and multi_click_rate < 10

            log(f"SURVIVAL/HAZARD VIABILITY: {'YES' if survival_viable else 'NO'}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Partial Identification
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: PARTIAL IDENTIFICATION", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Characterize what's learnable under minimal assumptions.", f)
        log("", f)

        # CTR by position (raw)
        log("CTR by position (raw, no adjustment):", f)

        if 'CTR' in ctr_by_rank.columns:
            log(f"  {'Rank':<8} {'CTR %':<10}", f)
            log(f"  {'-'*8} {'-'*10}", f)

            for _, row in ctr_by_rank.head(15).iterrows():
                log(f"  {int(row['RANKING']):<8} {row['CTR']*100:<10.3f}", f)

            log("", f)

            # Bounds width
            # Under minimal assumptions, position effect at rank k could be anywhere between:
            # Lower bound: CTR(k) - CTR(1) (assuming no selection)
            # Upper bound: CTR(k) / CTR(1) (relative effect)

            ctr_1 = ctr_by_rank[ctr_by_rank['RANKING'] == 1]['CTR'].values[0] if 1 in ctr_by_rank['RANKING'].values else ctr_by_rank['CTR'].max()

            log("Position effect bounds (relative to rank 1):", f)
            log(f"  {'Rank':<8} {'CTR %':<10} {'Additive':<12} {'Ratio':<10}", f)
            log(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10}", f)

            bound_widths = []
            for _, row in ctr_by_rank.head(10).iterrows():
                if row['impressions'] > 100:  # Only for meaningful sample sizes
                    additive = (row['CTR'] - ctr_1) * 100
                    ratio = row['CTR'] / ctr_1 if ctr_1 > 0 else 0
                    bound_width = abs(ratio - 1)  # How far from no effect
                    bound_widths.append(bound_width)
                    log(f"  {int(row['RANKING']):<8} {row['CTR']*100:<10.3f} {additive:<12.3f} {ratio:<10.3f}", f)

            log("", f)

            # Bounds informativeness
            avg_bound_width = np.mean(bound_widths) if bound_widths else 0
            log(f"Average bound width (deviation from 1): {avg_bound_width:.3f}", f)
            log(f"  Viability: {'INFORMATIVE' if avg_bound_width < 0.5 else 'WEAK - bounds too wide'}", f)
            log("", f)

            # Monotonicity check
            log("Monotonicity check:", f)
            ctr_values = ctr_by_rank[ctr_by_rank['impressions'] > 100].head(10)['CTR'].values
            is_monotonic = all(ctr_values[i] >= ctr_values[i+1] for i in range(len(ctr_values)-1))
            log(f"  CTR decreasing in rank (top 10 with N>100): {is_monotonic}", f)
            log(f"  Viability: {'EXPECTED - supports cascade model' if is_monotonic else 'UNEXPECTED - investigate'}", f)

            partial_id_informative = avg_bound_width < 0.5
        else:
            log("  CTR data not available.", f)
            partial_id_informative = False

        log("", f)
        log(f"PARTIAL ID INFORMATIVENESS: {'YES' if partial_id_informative else 'NO'}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Summary Table
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: SUMMARY TABLE", f)
        log("=" * 80, f)
        log("", f)

        log("POSITION EFFECT METHOD VIABILITY ASSESSMENT", f)
        log("=" * 80, f)
        log("", f)

        log(f"{'Method':<25} {'Key Diagnostic':<30} {'Value':<15} {'Viability':<15}", f)
        log(f"{'-'*25} {'-'*30} {'-'*15} {'-'*15}", f)

        # Method 1: PBM
        pbm_key = f"{pct_3plus:.1f}%"
        pbm_status = "YES" if pbm_viable else "NO"
        log(f"{'1. PBM/Click Models':<25} {'Ads with 3+ positions':<30} {pbm_key:<15} {pbm_status:<15}", f)

        # Method 2: IPW/DR
        if 'winner_in_loser_range' in dir():
            ipw_key = f"{winner_in_loser_range*100:.1f}%"
            ipw_status = "YES" if ipw_viable else "NO"
        else:
            ipw_key = "N/A"
            ipw_status = "CANNOT ASSESS"
        log(f"{'2. IPW/DR':<25} {'Common support':<30} {ipw_key:<15} {ipw_status:<15}", f)

        # Method 3: RDD
        if 'r2_v3' in dir():
            rdd_key = f"R²={r2_v3:.2f}"
            rdd_status = "YES" if rdd_viable else "NO"
        else:
            rdd_key = "N/A"
            rdd_status = "CANNOT ASSESS"
        log(f"{'3. RD at Margins':<25} {'Score R² for ranking':<30} {rdd_key:<15} {rdd_status:<15}", f)

        # Method 4: Survival
        if 'concentration_ratio' in dir():
            surv_key = f"{concentration_ratio:.2f}"
            surv_status = "YES" if survival_viable else "NO"
        else:
            surv_key = "N/A"
            surv_status = "CANNOT ASSESS"
        log(f"{'4. Survival/Hazard':<25} {'Click concentration':<30} {surv_key:<15} {surv_status:<15}", f)

        # Method 5: Partial ID
        if 'avg_bound_width' in dir():
            partial_key = f"{avg_bound_width:.2f}"
            partial_status = "INFORMATIVE" if partial_id_informative else "WEAK"
        else:
            partial_key = "N/A"
            partial_status = "CANNOT ASSESS"
        log(f"{'5. Partial ID':<25} {'Bound width':<30} {partial_key:<15} {partial_status:<15}", f)

        log("", f)
        log("=" * 80, f)
        log("", f)

        # Recommendations
        log("RECOMMENDATIONS:", f)
        log("-" * 40, f)
        log("", f)

        viable_methods = []
        if pbm_viable:
            viable_methods.append("PBM/Click Models")
        if 'ipw_viable' in dir() and ipw_viable:
            viable_methods.append("IPW/Doubly Robust")
        if 'rdd_viable' in dir() and rdd_viable:
            viable_methods.append("RDD at Margins")
        if 'survival_viable' in dir() and survival_viable:
            viable_methods.append("Survival/Hazard")
        if 'partial_id_informative' in dir() and partial_id_informative:
            viable_methods.append("Partial Identification")

        if viable_methods:
            log("Viable methods for further investigation:", f)
            for i, method in enumerate(viable_methods, 1):
                log(f"  {i}. {method}", f)
        else:
            log("No methods meet all viability thresholds.", f)
            log("Consider:", f)
            log("  - Collecting more data (longer time window)", f)
            log("  - Using methods with weaker assumptions", f)
            log("  - Focusing on bounds rather than point estimates", f)

        log("", f)
        log("=" * 80, f)
        log("POSITION EFFECTS EDA COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
