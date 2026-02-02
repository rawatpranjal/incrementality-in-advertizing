#!/usr/bin/env python3
"""
Position Effects Analysis: Four Causal Inference Methods

Implements:
1. Reduced Form Baseline - Logistic regression with controls and fixed effects
2. RDD at Adjacent Rank Margins - Score discontinuity at rank boundaries
3. PBM via EM - Position-Based Model decomposing CTR into position bias and relevance
4. Survival/Hazard Model - Discrete-time hazard with complementary log-log link

References:
- Craswell et al. (2008) WSDM - Cascade model
- Chapelle & Zhang (2009) WWW - DBN click model
- Joachims et al. (2017) WSDM - IPW for position bias
- Narayanan & Kalyanam (2015) Marketing Science - RDD approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "03_position_effects_analysis.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# ANALYSIS 1: REDUCED FORM BASELINE
# =============================================================================
def analysis_1_reduced_form(winners, f):
    """
    Reduced Form Baseline: Logistic regression with controls

    Unit of analysis: Winner bid (ad shown to user)

    Model equation:
        P(click_ik = 1) = logit^{-1}(β₀ + β₁·log(RANKING) + β₂·log(QUALITY)
                                      + β₃·log(FINAL_BID) + γ_placement + ε)

    Independent variables:
        - log(RANKING): Position in auction (1 = highest)
        - log(QUALITY): Quality score determining rank
        - log(FINAL_BID): Bid amount
        - PLACEMENT: Fixed effects for page location

    Dependent variable:
        - clicked: Binary indicator of click

    Purpose: Establish baseline position gradient after controlling for
    observable quality metrics. β₁ captures position effect holding quality constant.

    Expected signs:
        - β₁ < 0: Lower ranks (higher numbers) have lower CTR
        - β₂ > 0: Higher quality ads have higher CTR

    Error term captures: Unobserved ad-specific relevance, user preferences,
    timing effects, competitive dynamics
    """
    log("=" * 80, f)
    log("ANALYSIS 1: REDUCED FORM BASELINE", f)
    log("=" * 80, f)
    log("", f)

    log("UNIT OF ANALYSIS: Winner bid (ad impression)", f)
    log("SAMPLE SIZE: {:,}".format(len(winners)), f)
    log("", f)

    log("MODEL SPECIFICATION:", f)
    log("  P(click) = logit^-1(β₀ + β₁·log(RANKING) + β₂·log(QUALITY) + β₃·log(FINAL_BID) + γ_placement)", f)
    log("", f)

    log("INTERPRETATION:", f)
    log("  β₁: Position effect - expected to be negative (higher rank number → lower CTR)", f)
    log("  β₂: Quality effect - expected to be positive (higher quality → higher CTR)", f)
    log("  β₃: Bid effect - may be positive if bids signal advertiser confidence", f)
    log("  γ_placement: Placement-specific baseline CTR", f)
    log("", f)

    # Prepare data
    df = winners.copy()
    df['log_ranking'] = np.log(df['RANKING'])
    df['log_quality'] = np.log(df['QUALITY'].clip(lower=1e-10))
    df['log_bid'] = np.log(df['FINAL_BID'].clip(lower=1))

    # Handle placement - create dummies
    df['PLACEMENT'] = df['PLACEMENT'].astype(str)
    placement_dummies = pd.get_dummies(df['PLACEMENT'], prefix='placement', drop_first=True, dtype=float)

    log("-" * 40, f)
    log("MODEL 1A: Base model (no fixed effects)", f)
    log("-" * 40, f)

    X_vars = ['log_ranking', 'log_quality', 'log_bid']
    X = df[X_vars].copy()
    X = sm.add_constant(X)
    y = df['clicked'].astype(int)

    model_1a = Logit(y, X).fit(disp=0)

    log("", f)
    log("Coefficients:", f)
    log(f"  {'Variable':<20} {'Coef':<12} {'Std Err':<12} {'z':<10} {'P>|z|':<10}", f)
    log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", f)

    for var in model_1a.params.index:
        coef = model_1a.params[var]
        se = model_1a.bse[var]
        z = model_1a.tvalues[var]
        p = model_1a.pvalues[var]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        log(f"  {var:<20} {coef:<12.6f} {se:<12.6f} {z:<10.3f} {p:<10.4f} {sig}", f)

    log("", f)
    log(f"Pseudo R²: {model_1a.prsquared:.6f}", f)
    log(f"Log-likelihood: {model_1a.llf:.2f}", f)
    log(f"N observations: {model_1a.nobs:.0f}", f)
    log("", f)

    # Marginal effects at mean
    log("Marginal effects at mean:", f)
    mfx = model_1a.get_margeff(at='mean')
    for i, var in enumerate(X_vars):
        me = mfx.margeff[i]
        se = mfx.margeff_se[i]
        log(f"  {var}: {me:.6f} (SE: {se:.6f})", f)
    log("", f)

    log("-" * 40, f)
    log("MODEL 1B: With placement fixed effects", f)
    log("-" * 40, f)

    X_1b = pd.concat([df[X_vars], placement_dummies], axis=1)
    X_1b = sm.add_constant(X_1b)

    model_1b = Logit(y, X_1b).fit(disp=0)

    log("", f)
    log("Coefficients (main variables only):", f)
    log(f"  {'Variable':<20} {'Coef':<12} {'Std Err':<12} {'z':<10} {'P>|z|':<10}", f)
    log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", f)

    for var in ['const'] + X_vars:
        coef = model_1b.params[var]
        se = model_1b.bse[var]
        z = model_1b.tvalues[var]
        p = model_1b.pvalues[var]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        log(f"  {var:<20} {coef:<12.6f} {se:<12.6f} {z:<10.3f} {p:<10.4f} {sig}", f)

    log("", f)
    log("Placement fixed effects:", f)
    for var in model_1b.params.index:
        if var.startswith('placement_'):
            coef = model_1b.params[var]
            se = model_1b.bse[var]
            log(f"  {var}: {coef:.6f} (SE: {se:.6f})", f)

    log("", f)
    log(f"Pseudo R²: {model_1b.prsquared:.6f}", f)
    log(f"Log-likelihood: {model_1b.llf:.2f}", f)
    log("", f)

    log("-" * 40, f)
    log("MODEL 1C: Stratified by placement", f)
    log("-" * 40, f)

    placements = sorted(df['PLACEMENT'].unique())

    log("", f)
    log(f"  {'Placement':<12} {'N':<12} {'β_log_rank':<12} {'SE':<12} {'P-value':<12}", f)
    log(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

    for placement in placements:
        subset = df[df['PLACEMENT'] == placement]
        if len(subset) < 100 or subset['clicked'].sum() < 10:
            log(f"  {placement:<12} {len(subset):<12,} Insufficient data", f)
            continue

        X_sub = subset[X_vars].copy()
        X_sub = sm.add_constant(X_sub)
        y_sub = subset['clicked'].astype(int)

        try:
            model_sub = Logit(y_sub, X_sub).fit(disp=0)
            coef = model_sub.params['log_ranking']
            se = model_sub.bse['log_ranking']
            p = model_sub.pvalues['log_ranking']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            log(f"  {placement:<12} {len(subset):<12,} {coef:<12.6f} {se:<12.6f} {p:<12.4f} {sig}", f)
        except Exception as e:
            log(f"  {placement:<12} {len(subset):<12,} Model failed: {str(e)[:30]}", f)

    log("", f)

    log("-" * 40, f)
    log("CTR BY RANK (Marginal Effects)", f)
    log("-" * 40, f)

    log("", f)
    log("Using Model 1B coefficients to compute predicted CTR by rank:", f)
    log("", f)

    # Compute predicted CTR at mean values for each rank
    mean_quality = df['log_quality'].mean()
    mean_bid = df['log_bid'].mean()

    log(f"  {'Rank':<8} {'Predicted CTR %':<18} {'Marginal Effect':<18}", f)
    log(f"  {'-'*8} {'-'*18} {'-'*18}", f)

    beta_const = model_1b.params['const']
    beta_rank = model_1b.params['log_ranking']
    beta_quality = model_1b.params['log_quality']
    beta_bid = model_1b.params['log_bid']

    prev_ctr = None
    for rank in range(1, 21):
        log_rank = np.log(rank)
        linear_pred = beta_const + beta_rank * log_rank + beta_quality * mean_quality + beta_bid * mean_bid
        pred_ctr = 1 / (1 + np.exp(-linear_pred))

        if prev_ctr is not None:
            marg_effect = (pred_ctr - prev_ctr) * 100
            log(f"  {rank:<8} {pred_ctr*100:<18.4f} {marg_effect:<18.4f}", f)
        else:
            log(f"  {rank:<8} {pred_ctr*100:<18.4f} {'--':<18}", f)
        prev_ctr = pred_ctr

    log("", f)

    return model_1b


# =============================================================================
# ANALYSIS 2: RDD AT ADJACENT RANK MARGINS
# =============================================================================
def analysis_2_rdd(ar, winners, f):
    """
    Regression Discontinuity at Adjacent Rank Margins

    Unit of analysis: Adjacent rank pairs within auction

    Setup:
        Running variable: score_margin = score_k - score_{k+1}
        Treatment: Being at rank k vs rank k+1
        Outcome: CTR difference

    Identification: At the margin where two ads have nearly identical scores,
    the position assignment is quasi-random. Comparing CTR at the margin
    identifies the local average treatment effect (LATE) of position.

    Bandwidth selection: Imbens-Kalyanaraman optimal, plus 0.5× and 2× for robustness

    Diagnostics:
        - McCrary density test for manipulation
        - Covariate balance at cutoff
    """
    log("=" * 80, f)
    log("ANALYSIS 2: RDD AT ADJACENT RANK MARGINS", f)
    log("=" * 80, f)
    log("", f)

    log("IDENTIFICATION STRATEGY:", f)
    log("  At the margin between adjacent ranks, score differences are small and", f)
    log("  position assignment is quasi-random. Comparing CTR at the margin", f)
    log("  identifies the causal effect of position conditional on being at margin.", f)
    log("", f)

    log("RUNNING VARIABLE: score = QUALITY × FINAL_BID × PACING", f)
    log("MARGIN: score_k - score_{k+1} for adjacent ranks within auction", f)
    log("", f)

    # Compute scores for all bids
    df = ar.copy()
    df['score'] = df['QUALITY'] * df['FINAL_BID'] * df['PACING']

    # Keep only winners (those that got impressions)
    df = df[df['IS_WINNER'] == True].copy()

    log(f"Working with {len(df):,} winner bids", f)
    log("", f)

    # Merge click outcome from winners (which already has clicked column)
    winner_clicks = winners[['AUCTION_ID', 'PRODUCT_ID', 'clicked']].drop_duplicates()
    df = df.merge(winner_clicks, on=['AUCTION_ID', 'PRODUCT_ID'], how='left')
    df['clicked'] = df['clicked'].fillna(0).astype(int)

    log("-" * 40, f)
    log("STEP 1: Compute within-auction margins", f)
    log("-" * 40, f)

    # For each auction, compute score gap between adjacent ranks
    def compute_margins(group):
        group = group.sort_values('RANKING')
        group['score_next'] = group['score'].shift(-1)
        group['rank_next'] = group['RANKING'].shift(-1)
        group['clicked_next'] = group['clicked'].shift(-1)
        group['margin'] = group['score'] - group['score_next']
        return group

    log("Computing margins for each auction...", f)
    margins_data = df.groupby('AUCTION_ID', group_keys=False).apply(compute_margins)
    margins_data = margins_data[margins_data['margin'].notna() & margins_data['clicked_next'].notna()].copy()

    log(f"  Total margin pairs: {len(margins_data):,}", f)
    log("", f)

    # Create binary outcome: did higher-ranked ad get more clicks?
    margins_data['higher_rank_clicked'] = margins_data['clicked'].astype(int)
    margins_data['lower_rank_clicked'] = margins_data['clicked_next'].astype(int)
    margins_data['click_diff'] = margins_data['higher_rank_clicked'] - margins_data['lower_rank_clicked']

    log("-" * 40, f)
    log("STEP 2: Bandwidth selection", f)
    log("-" * 40, f)

    # Compute IK optimal bandwidth (simplified version)
    # Use rule of thumb: h_opt = 1.06 * std * n^(-1/5)
    margin_std = margins_data['margin'].std()
    n = len(margins_data)
    h_ik = 1.06 * margin_std * (n ** (-1/5))

    log(f"  Margin distribution:", f)
    log(f"    Mean: {margins_data['margin'].mean():.6f}", f)
    log(f"    Std: {margin_std:.6f}", f)
    log(f"    Min: {margins_data['margin'].min():.6f}", f)
    log(f"    Max: {margins_data['margin'].max():.6f}", f)
    log("", f)
    log(f"  IK-style optimal bandwidth: {h_ik:.6f}", f)
    log("", f)

    # Test multiple bandwidths
    bandwidths = {
        '0.5× IK': h_ik * 0.5,
        '1× IK': h_ik,
        '2× IK': h_ik * 2,
    }

    log("-" * 40, f)
    log("STEP 3: RDD estimates by bandwidth", f)
    log("-" * 40, f)
    log("", f)

    log(f"  {'Bandwidth':<15} {'h':<12} {'N in band':<12} {'CTR_high':<12} {'CTR_low':<12} {'LATE':<12} {'SE':<12}", f)
    log(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

    rdd_results = {}
    for name, h in bandwidths.items():
        # Filter to observations near margin = 0
        in_band = margins_data[margins_data['margin'].abs() <= h].copy()
        n_in_band = len(in_band)

        if n_in_band < 100:
            log(f"  {name:<15} {h:<12.6f} {n_in_band:<12} Insufficient sample", f)
            continue

        # CTR for higher-ranked ad
        ctr_high = in_band['higher_rank_clicked'].mean()
        ctr_low = in_band['lower_rank_clicked'].mean()

        # LATE = CTR difference
        late = ctr_high - ctr_low

        # Standard error (assuming independence)
        se_high = np.sqrt(ctr_high * (1 - ctr_high) / n_in_band)
        se_low = np.sqrt(ctr_low * (1 - ctr_low) / n_in_band)
        se_late = np.sqrt(se_high**2 + se_low**2)

        rdd_results[name] = {'h': h, 'n': n_in_band, 'late': late, 'se': se_late}

        log(f"  {name:<15} {h:<12.6f} {n_in_band:<12,} {ctr_high*100:<12.4f} {ctr_low*100:<12.4f} {late*100:<12.4f} {se_late*100:<12.4f}", f)

    log("", f)

    log("-" * 40, f)
    log("STEP 4: RDD by rank transition", f)
    log("-" * 40, f)
    log("", f)

    log("Local average treatment effect by rank transition (using 1× IK bandwidth):", f)
    log("", f)

    h = h_ik
    in_band = margins_data[margins_data['margin'].abs() <= h].copy()

    log(f"  {'Transition':<15} {'N':<10} {'CTR_high %':<12} {'CTR_low %':<12} {'LATE pp':<12} {'95% CI':<20}", f)
    log(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*20}", f)

    for rank_from in range(1, 11):
        rank_to = rank_from + 1
        subset = in_band[(in_band['RANKING'] == rank_from) & (in_band['rank_next'] == rank_to)]
        n_subset = len(subset)

        if n_subset < 30:
            continue

        ctr_high = subset['higher_rank_clicked'].mean()
        ctr_low = subset['lower_rank_clicked'].mean()
        late = ctr_high - ctr_low

        se_high = np.sqrt(ctr_high * (1 - ctr_high) / n_subset)
        se_low = np.sqrt(ctr_low * (1 - ctr_low) / n_subset)
        se_late = np.sqrt(se_high**2 + se_low**2)

        ci_low = (late - 1.96 * se_late) * 100
        ci_high = (late + 1.96 * se_late) * 100

        transition = f"{rank_from}→{rank_to}"
        log(f"  {transition:<15} {n_subset:<10,} {ctr_high*100:<12.4f} {ctr_low*100:<12.4f} {late*100:<12.4f} [{ci_low:.3f}, {ci_high:.3f}]", f)

    log("", f)

    log("-" * 40, f)
    log("STEP 5: McCrary density test (manipulation check)", f)
    log("-" * 40, f)
    log("", f)

    # Check for bunching at margin = 0
    # Bin margins and check for discontinuity in density
    bin_width = h_ik / 10
    bins = np.arange(-h_ik, h_ik + bin_width, bin_width)

    margin_counts, bin_edges = np.histogram(margins_data['margin'], bins=bins)

    # Compare density just below and just above zero
    zero_idx = np.searchsorted(bin_edges, 0)
    if zero_idx > 0 and zero_idx < len(margin_counts):
        density_below = margin_counts[:zero_idx].mean()
        density_above = margin_counts[zero_idx:].mean()
        ratio = density_above / density_below if density_below > 0 else np.nan

        log(f"  Density below margin=0: {density_below:.2f}", f)
        log(f"  Density above margin=0: {density_above:.2f}", f)
        log(f"  Ratio (above/below): {ratio:.3f}", f)
        log(f"  Interpretation: Ratio near 1.0 suggests no manipulation", f)
        log(f"  Status: {'PASS' if 0.8 < ratio < 1.2 else 'INVESTIGATE'}", f)

    log("", f)

    log("-" * 40, f)
    log("STEP 6: Covariate balance at cutoff", f)
    log("-" * 40, f)
    log("", f)

    # Compare covariates just above and below cutoff
    in_band = margins_data[margins_data['margin'].abs() <= h_ik].copy()
    above_cutoff = in_band[in_band['margin'] >= 0]
    below_cutoff = in_band[in_band['margin'] < 0]

    covariates = ['QUALITY', 'FINAL_BID', 'PACING']

    log(f"  {'Covariate':<15} {'Mean (above)':<15} {'Mean (below)':<15} {'Diff':<12} {'t-stat':<10}", f)
    log(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*12} {'-'*10}", f)

    for cov in covariates:
        if cov not in in_band.columns:
            continue
        mean_above = above_cutoff[cov].mean()
        mean_below = below_cutoff[cov].mean()
        diff = mean_above - mean_below

        # t-test
        t_stat, p_val = stats.ttest_ind(above_cutoff[cov].dropna(), below_cutoff[cov].dropna())

        log(f"  {cov:<15} {mean_above:<15.4f} {mean_below:<15.4f} {diff:<12.4f} {t_stat:<10.3f}", f)

    log("", f)

    return rdd_results


# =============================================================================
# ANALYSIS 3: PBM VIA EM
# =============================================================================
def analysis_3_pbm_em(winners, f):
    """
    Position-Based Model via Expectation-Maximization

    Model:
        P(click | ad i at position k) = θ_k × α_i

    Where:
        θ_k = examination probability at position k (position bias)
        α_i = relevance/attractiveness of ad i

    Normalization: θ_1 = 1 (position 1 is fully examined)

    EM Algorithm:
        E-step: Compute expected examination
            If clicked: examined = 1
            If not clicked: P(examined | no click) = θ_k(1 - α_i) / (1 - θ_k·α_i)

        M-step: Update parameters
            θ_k = Σ[E(examined) at rank k] / N_k
            α_i = Σ[clicks on i] / Σ[E(examined) for i]

    Sample: User-product pairs with 2+ exposures at different ranks
    """
    log("=" * 80, f)
    log("ANALYSIS 3: POSITION-BASED MODEL VIA EM", f)
    log("=" * 80, f)
    log("", f)

    log("MODEL:", f)
    log("  P(click | ad i, position k) = θ_k × α_i", f)
    log("", f)
    log("  θ_k = examination probability at position k (position bias)", f)
    log("  α_i = intrinsic relevance of ad i", f)
    log("", f)
    log("NORMALIZATION: θ_1 = 1 (position 1 always examined)", f)
    log("", f)

    log("-" * 40, f)
    log("STEP 1: Identify ads with multiple position exposures", f)
    log("-" * 40, f)

    # Sample for computational efficiency (use 10% of data)
    sample_frac = 0.1
    df = winners[['PRODUCT_ID', 'RANKING', 'clicked', 'AUCTION_ID']].sample(frac=sample_frac, random_state=42).copy()
    log(f"  Sampled {sample_frac*100:.0f}% of data for PBM: {len(df):,} observations", f)

    # Count positions per product
    product_positions = df.groupby('PRODUCT_ID').agg({
        'RANKING': ['nunique', 'count'],
        'clicked': 'sum'
    }).reset_index()
    product_positions.columns = ['PRODUCT_ID', 'n_positions', 'n_impressions', 'n_clicks']

    # Filter to products with 2+ positions
    multi_position_products = product_positions[product_positions['n_positions'] >= 2]['PRODUCT_ID']

    log(f"  Total products: {len(product_positions):,}", f)
    log(f"  Products with 2+ positions: {len(multi_position_products):,} ({len(multi_position_products)/len(product_positions)*100:.1f}%)", f)
    log("", f)

    # Filter data to multi-position products
    df_pbm = df[df['PRODUCT_ID'].isin(multi_position_products)].copy()
    log(f"  Observations for PBM: {len(df_pbm):,}", f)
    log("", f)

    if len(df_pbm) < 1000:
        log("  WARNING: Insufficient data for PBM estimation", f)
        log("", f)
        return None

    log("-" * 40, f)
    log("STEP 2: Initialize parameters", f)
    log("-" * 40, f)

    # Get unique products and positions
    products = df_pbm['PRODUCT_ID'].unique()
    max_rank = min(20, df_pbm['RANKING'].max())  # Cap at 20 for stability

    df_pbm = df_pbm[df_pbm['RANKING'] <= max_rank].copy()

    log(f"  Unique products: {len(products):,}", f)
    log(f"  Max rank considered: {max_rank}", f)
    log("", f)

    # Initialize θ_k from raw CTR decay
    ctr_by_rank = df_pbm.groupby('RANKING')['clicked'].mean()
    theta = np.ones(max_rank + 1)
    for k in range(1, max_rank + 1):
        if k in ctr_by_rank.index:
            theta[k] = ctr_by_rank[k] / ctr_by_rank.get(1, ctr_by_rank.iloc[0])
    theta = np.clip(theta, 0.01, 1.0)
    theta[1] = 1.0  # Normalize

    # Initialize α_i from overall product CTR
    product_ctr = df_pbm.groupby('PRODUCT_ID')['clicked'].mean()
    alpha = product_ctr.to_dict()
    for p in products:
        if p not in alpha:
            alpha[p] = 0.01
        alpha[p] = max(0.01, min(0.99, alpha[p]))

    log("  Initial θ (examination probabilities):", f)
    log(f"    {'Rank':<8} {'θ_k':<10}", f)
    log(f"    {'-'*8} {'-'*10}", f)
    for k in range(1, min(11, max_rank + 1)):
        log(f"    {k:<8} {theta[k]:<10.4f}", f)
    log("", f)

    log("-" * 40, f)
    log("STEP 3: EM algorithm (vectorized)", f)
    log("-" * 40, f)

    n_iter = 50
    tol = 1e-6
    prev_ll = -np.inf

    log("", f)
    log("Running EM iterations...", f)

    # Prepare vectorized data
    df_pbm = df_pbm.copy()
    df_pbm['rank_idx'] = df_pbm['RANKING'].astype(int)

    # Create product-to-index mapping for alpha based on products IN df_pbm
    products_in_pbm = df_pbm['PRODUCT_ID'].unique()
    product_list = list(products_in_pbm)
    product_to_idx = {p: i for i, p in enumerate(product_list)}
    df_pbm['product_idx'] = df_pbm['PRODUCT_ID'].map(product_to_idx)

    # Initialize alpha as array
    alpha_arr = np.array([alpha.get(p, 0.01) for p in product_list])

    for iteration in tqdm(range(n_iter), desc="  EM Progress"):
        # E-step: Compute expected examination (vectorized)
        theta_k = theta[df_pbm['rank_idx'].values]
        alpha_p = alpha_arr[df_pbm['product_idx'].values]
        clicked = df_pbm['clicked'].values

        # P(click) = theta * alpha
        prob_click = theta_k * alpha_p
        prob_no_click = 1 - prob_click

        # E[examined | observed]
        e_exam = np.where(
            clicked == 1,
            1.0,
            np.where(prob_no_click > 0, theta_k * (1 - alpha_p) / prob_no_click, 0.5)
        )

        # Log-likelihood
        ll_click = np.where(clicked == 1, np.log(np.clip(prob_click, 1e-10, 1)), 0)
        ll_no_click = np.where(clicked == 0, np.log(np.clip(prob_no_click, 1e-10, 1)), 0)
        log_likelihood = ll_click.sum() + ll_no_click.sum()

        df_pbm['e_exam'] = e_exam

        # M-step: Update θ_k (vectorized)
        new_theta = np.ones(max_rank + 1)
        theta_update = df_pbm.groupby('rank_idx')['e_exam'].mean()
        for k in theta_update.index:
            if k <= max_rank:
                new_theta[k] = theta_update[k]

        # Normalize: θ_1 = 1
        if new_theta[1] > 0:
            new_theta = new_theta / new_theta[1]
        new_theta = np.clip(new_theta, 0.01, 1.0)
        new_theta[1] = 1.0

        # M-step: Update α_i (vectorized)
        alpha_num = df_pbm.groupby('product_idx')['clicked'].sum()
        alpha_denom = df_pbm.groupby('product_idx')['e_exam'].sum()
        new_alpha_arr = np.clip(alpha_num.values / np.clip(alpha_denom.values, 1e-10, None), 0.01, 0.99)

        # Check convergence
        theta_change = np.max(np.abs(new_theta - theta))

        theta = new_theta
        alpha_arr = new_alpha_arr

        if iteration % 10 == 0:
            log(f"    Iteration {iteration}: LL = {log_likelihood:.2f}, max Δθ = {theta_change:.6f}", f)

        if abs(log_likelihood - prev_ll) < tol and theta_change < tol:
            log(f"    Converged at iteration {iteration}", f)
            break

        prev_ll = log_likelihood

    # Convert alpha back to dict for compatibility
    alpha = {product_list[i]: alpha_arr[i] for i in range(len(product_list))}

    log("", f)

    log("-" * 40, f)
    log("STEP 4: Final estimates", f)
    log("-" * 40, f)
    log("", f)

    log("Position bias (θ_k) - Examination probabilities:", f)
    log("", f)
    log(f"  {'Rank':<8} {'θ_k':<12} {'Raw CTR %':<12} {'Ratio θ_k/CTR_k':<15}", f)
    log(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*15}", f)

    for k in range(1, min(21, max_rank + 1)):
        raw_ctr = ctr_by_rank.get(k, np.nan)
        ctr_1 = ctr_by_rank.get(1, ctr_by_rank.iloc[0])
        if pd.notna(raw_ctr) and raw_ctr > 0:
            ratio = theta[k] / (raw_ctr / ctr_1)
        else:
            ratio = np.nan
        raw_ctr_pct = raw_ctr * 100 if pd.notna(raw_ctr) else np.nan
        log(f"  {k:<8} {theta[k]:<12.4f} {raw_ctr_pct:<12.4f} {ratio:<15.4f}", f)

    log("", f)

    log("Relevance (α_i) distribution:", f)
    alpha_values = list(alpha.values())
    log(f"  Mean: {np.mean(alpha_values):.4f}", f)
    log(f"  Std: {np.std(alpha_values):.4f}", f)
    log(f"  Min: {np.min(alpha_values):.4f}", f)
    log(f"  Max: {np.max(alpha_values):.4f}", f)
    log(f"  Median: {np.median(alpha_values):.4f}", f)
    log("", f)

    log("Interpretation:", f)
    log("  θ_k represents the probability a user examines position k", f)
    log("  α_i represents the probability of click given examination", f)
    log("  If θ_k decays faster than raw CTR, products in lower positions", f)
    log("  are more relevant than those in higher positions (quality sorts)", f)
    log("", f)

    return {'theta': theta, 'alpha': alpha}


# =============================================================================
# ANALYSIS 4: SURVIVAL/HAZARD MODEL
# =============================================================================
def analysis_4_survival(winners, f):
    """
    Survival/Hazard Model with Complementary Log-Log Link

    Framework: Position as discrete time, click as terminal event

    Data preparation:
        1. Keep first click per user-auction
        2. Create person-position panel up to click or max rank
        3. Outcome: clicked_here binary

    Model (complementary log-log):
        log(-log(1 - h_ik)) = γ_k + X_i'β + u_i

    Where:
        h_ik = hazard rate (prob of click at position k given survived to k)
        γ_k = baseline hazard by position
        X_i = covariates (quality, bid, etc.)
        u_i = frailty term (optional)
    """
    log("=" * 80, f)
    log("ANALYSIS 4: SURVIVAL/HAZARD MODEL", f)
    log("=" * 80, f)
    log("", f)

    log("FRAMEWORK:", f)
    log("  Treat position as discrete time periods", f)
    log("  Click is the 'failure' event", f)
    log("  Users 'survive' to each position until they click or exit", f)
    log("", f)

    log("MODEL (complementary log-log):", f)
    log("  log(-log(1 - h_ik)) = γ_k + X_i'β", f)
    log("", f)
    log("  h_ik = P(click at position k | survived to position k)", f)
    log("  γ_k = baseline hazard at position k", f)
    log("  X_i = covariates (log quality, log bid)", f)
    log("", f)

    log("-" * 40, f)
    log("STEP 1: Data preparation", f)
    log("-" * 40, f)

    df = winners.copy()

    # Get max rank per auction for censoring
    max_ranks = df.groupby('AUCTION_ID')['RANKING'].max().to_dict()

    # For each auction, keep only first click
    first_clicks = df[df['clicked'] == 1].groupby('AUCTION_ID')['RANKING'].min().to_dict()

    log(f"  Total auctions: {df['AUCTION_ID'].nunique():,}", f)
    log(f"  Auctions with clicks: {len(first_clicks):,}", f)
    log("", f)

    # Create person-period data (vectorized)
    # Each row is (auction, position) with indicator of click
    log("  Creating person-period panel (vectorized)...", f)

    # Sample auctions for speed
    auctions = df['AUCTION_ID'].unique()[:10000]
    df_sample = df[df['AUCTION_ID'].isin(auctions)].copy()

    # Add first click rank
    df_sample['first_click_rank'] = df_sample['AUCTION_ID'].map(first_clicks).fillna(np.inf)

    # Filter to rows at or before first click
    df_sample = df_sample[df_sample['RANKING'] <= df_sample['first_click_rank']].copy()

    # Create panel
    df_sample['clicked_here'] = (df_sample['RANKING'] == df_sample['first_click_rank']).astype(int)
    df_sample['log_quality'] = np.log(df_sample['QUALITY'].clip(lower=1e-10))
    df_sample['log_bid'] = np.log(df_sample['FINAL_BID'].clip(lower=1))
    if 'PLACEMENT' in df_sample.columns:
        df_sample['PLACEMENT'] = df_sample['PLACEMENT'].astype(str)
    else:
        df_sample['PLACEMENT'] = '0'

    panel = df_sample[['AUCTION_ID', 'RANKING', 'clicked_here', 'log_quality', 'log_bid', 'PLACEMENT']].copy()

    log(f"  Panel observations: {len(panel):,}", f)
    log(f"  Clicks in panel: {panel['clicked_here'].sum():,}", f)
    log("", f)

    if len(panel) < 1000 or panel['clicked_here'].sum() < 50:
        log("  WARNING: Insufficient data for hazard model", f)
        log("", f)
        return None

    log("-" * 40, f)
    log("STEP 2: Non-parametric hazard estimates", f)
    log("-" * 40, f)
    log("", f)

    # Compute Kaplan-Meier style hazard by position
    hazard_by_rank = panel.groupby('RANKING').agg({
        'clicked_here': ['sum', 'count']
    }).reset_index()
    hazard_by_rank.columns = ['RANKING', 'clicks', 'at_risk']
    hazard_by_rank['hazard'] = hazard_by_rank['clicks'] / hazard_by_rank['at_risk']
    hazard_by_rank['survival'] = (1 - hazard_by_rank['hazard']).cumprod()

    log(f"  {'Rank':<8} {'At Risk':<12} {'Clicks':<10} {'Hazard':<12} {'Survival':<12}", f)
    log(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*12} {'-'*12}", f)

    for _, row in hazard_by_rank.head(15).iterrows():
        log(f"  {int(row['RANKING']):<8} {int(row['at_risk']):<12,} {int(row['clicks']):<10,} {row['hazard']:<12.5f} {row['survival']:<12.5f}", f)

    log("", f)

    log("-" * 40, f)
    log("STEP 3: Complementary log-log regression", f)
    log("-" * 40, f)
    log("", f)

    # Create rank dummies for baseline hazard
    panel['log_ranking'] = np.log(panel['RANKING'])

    # Prepare design matrix
    X_vars = ['log_ranking', 'log_quality', 'log_bid']
    X = panel[X_vars].copy()
    X = sm.add_constant(X)
    y = panel['clicked_here'].astype(int)

    # Fit logit as approximation (cloglog not directly available in statsmodels)
    # Note: For rare events, logit ≈ cloglog
    try:
        model = Logit(y, X).fit(disp=0)

        log("Coefficients (logit approximation to cloglog):", f)
        log(f"  {'Variable':<20} {'Coef':<12} {'Std Err':<12} {'z':<10} {'P>|z|':<10}", f)
        log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", f)

        for var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            z = model.tvalues[var]
            p = model.pvalues[var]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            log(f"  {var:<20} {coef:<12.6f} {se:<12.6f} {z:<10.3f} {p:<10.4f} {sig}", f)

        log("", f)
        log(f"Pseudo R²: {model.prsquared:.6f}", f)
        log(f"Log-likelihood: {model.llf:.2f}", f)
        log("", f)

        log("Interpretation:", f)
        log(f"  log_ranking coefficient: {model.params['log_ranking']:.4f}", f)
        log(f"    A 1% increase in rank (worse position) is associated with", f)
        log(f"    {model.params['log_ranking']/100:.4f} change in log-odds of click", f)
        log("", f)

    except Exception as e:
        log(f"  Model fitting failed: {str(e)}", f)
        model = None

    log("-" * 40, f)
    log("STEP 4: Baseline hazard γ_k estimates", f)
    log("-" * 40, f)
    log("", f)

    # Estimate position-specific hazards using dummy variables
    panel_subset = panel[panel['RANKING'] <= 15].copy()
    rank_dummies = pd.get_dummies(panel_subset['RANKING'], prefix='rank', drop_first=True, dtype=float)

    X_rank = pd.concat([panel_subset[['log_quality', 'log_bid']].astype(float), rank_dummies], axis=1)
    X_rank = sm.add_constant(X_rank)
    y_rank = panel_subset['clicked_here'].astype(int)

    try:
        model_rank = Logit(y_rank, X_rank).fit(disp=0)

        log("Baseline hazard (γ_k) relative to rank 1:", f)
        log(f"  {'Rank':<8} {'γ_k':<12} {'SE':<12} {'Exp(γ_k)':<12}", f)
        log(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}", f)

        log(f"  {1:<8} {'0 (ref)':<12} {'--':<12} {'1.000':<12}", f)

        for var in model_rank.params.index:
            if var.startswith('rank_'):
                rank = int(var.split('_')[1])
                coef = model_rank.params[var]
                se = model_rank.bse[var]
                exp_coef = np.exp(coef)
                log(f"  {rank:<8} {coef:<12.4f} {se:<12.4f} {exp_coef:<12.4f}", f)

        log("", f)
        log("Interpretation:", f)
        log("  γ_k < 0 means lower hazard (less likely to click) than rank 1", f)
        log("  Exp(γ_k) is the hazard ratio relative to rank 1", f)
        log("", f)

    except Exception as e:
        log(f"  Model with rank dummies failed: {str(e)}", f)

    return hazard_by_rank


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("POSITION EFFECTS ANALYSIS: FOUR CAUSAL INFERENCE METHODS", f)
        log("=" * 80, f)
        log("", f)

        log("METHODS:", f)
        log("  1. Reduced Form Baseline - Logistic regression with controls", f)
        log("  2. RDD at Adjacent Rank Margins - Score discontinuity", f)
        log("  3. PBM via EM - Position-bias and relevance decomposition", f)
        log("  4. Survival/Hazard Model - Discrete-time click hazard", f)
        log("", f)

        log("REFERENCES:", f)
        log("  - Craswell et al. (2008) WSDM - Cascade model", f)
        log("  - Chapelle & Zhang (2009) WWW - DBN click model", f)
        log("  - Joachims et al. (2017) WSDM - IPW for position bias", f)
        log("  - Ai et al. (2018) SIGIR - Dual propensity estimation", f)
        log("  - Narayanan & Kalyanam (2015) Marketing Science - RDD approach", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # Load data
        log("LOADING DATA", f)
        log("-" * 40, f)

        ar_path = DATA_DIR / "auctions_results_all.parquet"
        au_path = DATA_DIR / "auctions_users_all.parquet"
        imp_path = DATA_DIR / "impressions_all.parquet"
        clicks_path = DATA_DIR / "clicks_all.parquet"

        ar = pd.read_parquet(ar_path)
        log(f"  auctions_results: {len(ar):,} rows", f)

        au = pd.read_parquet(au_path)
        log(f"  auctions_users: {len(au):,} rows", f)

        imp = pd.read_parquet(imp_path)
        log(f"  impressions: {len(imp):,} rows", f)

        clicks = pd.read_parquet(clicks_path)
        log(f"  clicks: {len(clicks):,} rows", f)
        log("", f)

        # Prepare winners with click labels
        log("Preparing winner data with click labels...", f)

        winners = ar[ar['IS_WINNER'] == True].copy()

        # Merge placement from auctions_users
        placement_map = au[['AUCTION_ID', 'PLACEMENT']].drop_duplicates()
        winners = winners.merge(placement_map, on='AUCTION_ID', how='left')
        winners['PLACEMENT'] = winners['PLACEMENT'].fillna('0').astype(str)

        # Create click indicator using merge (vectorized)
        clicks_dedup = clicks[['AUCTION_ID', 'PRODUCT_ID']].drop_duplicates()
        clicks_dedup['clicked'] = 1
        winners = winners.merge(clicks_dedup, on=['AUCTION_ID', 'PRODUCT_ID'], how='left')
        winners['clicked'] = winners['clicked'].fillna(0).astype(int)

        log(f"  Winners: {len(winners):,}", f)
        log(f"  Clicked: {winners['clicked'].sum():,} ({winners['clicked'].mean()*100:.3f}%)", f)
        log("", f)

        # =====================================================================
        # Run analyses
        # =====================================================================

        log("", f)
        model_1 = analysis_1_reduced_form(winners, f)

        log("", f)
        rdd_results = analysis_2_rdd(ar, winners, f)

        log("", f)
        pbm_results = analysis_3_pbm_em(winners, f)

        log("", f)
        hazard_results = analysis_4_survival(winners, f)

        # =====================================================================
        # Summary
        # =====================================================================
        log("", f)
        log("=" * 80, f)
        log("SUMMARY: POSITION EFFECT ESTIMATES", f)
        log("=" * 80, f)
        log("", f)

        log("COMPARISON OF POSITION EFFECT ESTIMATES:", f)
        log("", f)

        log("1. REDUCED FORM (Model 1B):", f)
        if model_1 is not None:
            beta_rank = model_1.params.get('log_ranking', np.nan)
            se_rank = model_1.bse.get('log_ranking', np.nan)
            log(f"   β_log_rank = {beta_rank:.4f} (SE: {se_rank:.4f})", f)
            log(f"   Interpretation: 1% increase in rank → {beta_rank/100:.4f} change in log-odds", f)
        log("", f)

        log("2. RDD AT MARGINS:", f)
        if rdd_results:
            for name, res in rdd_results.items():
                log(f"   {name}: LATE = {res['late']*100:.4f} pp (SE: {res['se']*100:.4f}, N={res['n']:,})", f)
        log("", f)

        log("3. PBM EXAMINATION PROBABILITIES:", f)
        if pbm_results:
            theta = pbm_results['theta']
            log(f"   θ_1 = 1.000 (normalized)", f)
            log(f"   θ_5 = {theta[5]:.4f}", f)
            log(f"   θ_10 = {theta[10]:.4f}", f)
            log(f"   Interpretation: Position 10 examined at {theta[10]*100:.1f}% rate of position 1", f)
        log("", f)

        log("4. SURVIVAL MODEL HAZARD:", f)
        if hazard_results is not None:
            hr_1 = hazard_results[hazard_results['RANKING'] == 1]['hazard'].values
            hr_5 = hazard_results[hazard_results['RANKING'] == 5]['hazard'].values
            hr_10 = hazard_results[hazard_results['RANKING'] == 10]['hazard'].values
            if len(hr_1) > 0:
                log(f"   h(1) = {hr_1[0]:.5f}", f)
            if len(hr_5) > 0:
                log(f"   h(5) = {hr_5[0]:.5f}", f)
            if len(hr_10) > 0:
                log(f"   h(10) = {hr_10[0]:.5f}", f)
            if len(hr_1) > 0 and len(hr_10) > 0 and hr_1[0] > 0:
                log(f"   Hazard ratio h(10)/h(1) = {hr_10[0]/hr_1[0]:.4f}", f)
        log("", f)

        log("CONSISTENCY CHECK:", f)
        log("  If methods are identifying the same causal effect, we expect:", f)
        log("  - PBM θ_k should track hazard h(k) pattern", f)
        log("  - RDD LATE should be consistent with reduced form gradient", f)
        log("  - Survival hazard ratio should align with CTR ratios", f)
        log("", f)

        log("=" * 80, f)
        log("POSITION EFFECTS ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
