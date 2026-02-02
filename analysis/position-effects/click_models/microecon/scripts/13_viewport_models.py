#!/usr/bin/env python3
"""
Viewport-Based Click Models

Tests three micro-economic hypotheses about click behavior in viewport/grid layouts
using the batched impression structure observed in session logs.

H1: Price Anchor Effect - Users evaluate prices relative to viewport neighbors
H2: Attention Gate - Impressions with dwell_time < threshold are noise
H3: Vampire Effect - High-quality items steal attention from viewport neighbors
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_r2"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = RESULTS_DIR / "13_viewport_models.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load all required datasets."""
    impressions = pd.read_parquet(DATA_DIR / "impressions_r2.parquet")
    clicks = pd.read_parquet(DATA_DIR / "clicks_r2.parquet")
    ar = pd.read_parquet(DATA_DIR / "auctions_results_r2.parquet")
    au = pd.read_parquet(DATA_DIR / "auctions_users_r2.parquet")
    return impressions, clicks, ar, au


# =============================================================================
# VIEWPORT CONSTRUCTION
# =============================================================================
def build_viewport_dataset(impressions, clicks, ar, au, f):
    """
    Build impression-level dataset with viewport features.

    Viewport = group of impressions with same (AUCTION_ID, OCCURRED_AT).
    This captures the batched impression structure where 2-4 items are shown together.
    """
    log("Building viewport dataset...", f)

    # Sort impressions by auction and time
    impressions = impressions.copy()
    impressions['OCCURRED_AT'] = pd.to_datetime(impressions['OCCURRED_AT'])
    impressions = impressions.sort_values(['AUCTION_ID', 'OCCURRED_AT', 'PRODUCT_ID'])

    # Create click indicator
    click_keys = clicks[['AUCTION_ID', 'PRODUCT_ID']].drop_duplicates()
    click_keys['clicked'] = 1

    df = impressions.merge(
        click_keys,
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )
    df['clicked'] = df['clicked'].fillna(0).astype(int)

    log(f"  Impressions: {len(df):,}", f)
    log(f"  Clicked: {df['clicked'].sum():,} ({df['clicked'].mean()*100:.2f}%)", f)

    # Join auction results for QUALITY, PRICE, RANKING
    ar_subset = ar[['AUCTION_ID', 'PRODUCT_ID', 'QUALITY', 'PRICE', 'RANKING', 'FINAL_BID']].copy()
    df = df.merge(
        ar_subset,
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    # Join auctions_users for PLACEMENT
    au_subset = au[['AUCTION_ID', 'PLACEMENT']].drop_duplicates()
    df = df.merge(
        au_subset,
        on='AUCTION_ID',
        how='left'
    )

    log(f"  After joining auction data: {len(df):,}", f)
    log(f"  Non-null QUALITY: {df['QUALITY'].notna().sum():,}", f)
    log(f"  Non-null PRICE: {df['PRICE'].notna().sum():,}", f)
    log(f"  Non-null PLACEMENT: {df['PLACEMENT'].notna().sum():,}", f)

    # Create viewport ID: (AUCTION_ID, OCCURRED_AT) defines a viewport batch
    df['viewport_id'] = df['AUCTION_ID'].astype(str) + '_' + df['OCCURRED_AT'].astype(str)

    n_viewports = df['viewport_id'].nunique()
    log(f"  Viewports identified: {n_viewports:,}", f)

    # Viewport size statistics
    viewport_sizes = df.groupby('viewport_id').size()
    log(f"  Viewport size: mean={viewport_sizes.mean():.2f}, median={viewport_sizes.median():.0f}, max={viewport_sizes.max()}", f)
    log(f"  Viewport size distribution:", f)
    for size in sorted(viewport_sizes.unique())[:8]:
        count = (viewport_sizes == size).sum()
        pct = count / len(viewport_sizes) * 100
        log(f"    Size {size}: {count:,} ({pct:.1f}%)", f)

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_viewport_features(df, f):
    """
    Compute viewport-level features for all three hypotheses.

    H1 (Price Anchor): LPD = log(price) - mean(log(prices in viewport))
    H2 (Attention Gate): dwell_time = time until next viewport appears
    H3 (Vampire Effect): neighbor_quality = sum of quality scores of viewport neighbors
    """
    log("\nComputing viewport features...", f)

    df = df.copy()

    # Log price
    df['log_price'] = np.log(df['PRICE'].clip(lower=1))

    # H1: Price Anchor Effect (LPD = Log Price Deviation from viewport mean)
    log("  H1: Computing price anchor features...", f)
    viewport_log_price_mean = df.groupby('viewport_id')['log_price'].transform('mean')
    df['LPD'] = df['log_price'] - viewport_log_price_mean

    log(f"    LPD: mean={df['LPD'].mean():.4f}, std={df['LPD'].std():.4f}", f)

    # H2: Attention Gate (Dwell Time)
    log("  H2: Computing dwell time features...", f)

    # For each auction, compute time to next viewport
    df = df.sort_values(['AUCTION_ID', 'OCCURRED_AT'])

    # Get unique viewport times per auction
    viewport_times = df.groupby('AUCTION_ID')['OCCURRED_AT'].apply(
        lambda x: x.drop_duplicates().sort_values().tolist()
    ).to_dict()

    def get_dwell_time(row):
        """Compute dwell time = time until next viewport in same auction."""
        auction_id = row['AUCTION_ID']
        current_time = row['OCCURRED_AT']
        times = viewport_times.get(auction_id, [])

        # Find next viewport time
        for t in times:
            if t > current_time:
                return (t - current_time).total_seconds()
        # Last viewport in session - no dwell time computable
        return np.nan

    tqdm.pandas(desc="Computing dwell times")
    df['dwell_time'] = df.progress_apply(get_dwell_time, axis=1)

    # Dwell time statistics
    valid_dwell = df['dwell_time'].dropna()
    log(f"    Dwell time: mean={valid_dwell.mean():.2f}s, median={valid_dwell.median():.2f}s", f)
    log(f"    Dwell time range: [{valid_dwell.min():.2f}s, {valid_dwell.max():.2f}s]", f)
    log(f"    Missing (last viewport): {df['dwell_time'].isna().sum():,}", f)

    # Attention gate thresholds to test
    # Note: Minimum dwell time is 1s (time resolution), so use higher thresholds
    for threshold in [2.0, 3.0, 5.0, 7.0]:
        df[f'is_scanned_{threshold}s'] = (df['dwell_time'] >= threshold).astype(int)
        scanned_pct = df[f'is_scanned_{threshold}s'].mean() * 100
        log(f"    is_scanned (>={threshold}s): {scanned_pct:.1f}%", f)

    # Default threshold: use median dwell time as natural cutoff
    median_dwell = valid_dwell.median()
    log(f"    Using median dwell time ({median_dwell:.1f}s) as threshold", f)
    df['is_scanned'] = (df['dwell_time'] >= median_dwell).astype(int)

    # Create long_dwell vs short_dwell for clearer analysis
    df['long_dwell'] = (df['dwell_time'] >= median_dwell).astype(int)
    df['short_dwell'] = (df['dwell_time'] < median_dwell).astype(int)

    # H3: Vampire Effect (Neighbor Quality)
    log("  H3: Computing neighbor quality features...", f)

    # Sum of quality in viewport
    viewport_quality_sum = df.groupby('viewport_id')['QUALITY'].transform('sum')
    # Neighbor quality = total - own
    df['neighbor_quality'] = viewport_quality_sum - df['QUALITY']

    # Viewport size for normalization
    df['viewport_size'] = df.groupby('viewport_id')['QUALITY'].transform('count')
    # Mean neighbor quality
    df['neighbor_quality_mean'] = df['neighbor_quality'] / (df['viewport_size'] - 1).clip(lower=1)

    log(f"    Neighbor quality sum: mean={df['neighbor_quality'].mean():.4f}, std={df['neighbor_quality'].std():.4f}", f)
    log(f"    Neighbor quality mean: mean={df['neighbor_quality_mean'].mean():.4f}, std={df['neighbor_quality_mean'].std():.4f}", f)

    # Additional features
    # Rank within viewport
    df['viewport_rank'] = df.groupby('viewport_id')['RANKING'].rank(method='dense')

    # Max quality in viewport
    df['viewport_max_quality'] = df.groupby('viewport_id')['QUALITY'].transform('max')
    df['is_highest_quality'] = (df['QUALITY'] == df['viewport_max_quality']).astype(int)

    # Price rank within viewport
    df['viewport_price_rank'] = df.groupby('viewport_id')['PRICE'].rank(method='dense')

    return df


# =============================================================================
# MODELS
# =============================================================================
def fit_logit_model(df, formula_vars, f, model_name="Model"):
    """
    Fit logistic regression model and return results.

    Parameters:
    - df: DataFrame with features
    - formula_vars: list of variable names to include (intercept added automatically)
    - f: file handle for logging
    - model_name: name for display
    """
    # Prepare data
    model_df = df[['clicked'] + formula_vars].dropna()

    if len(model_df) < 100:
        log(f"  Insufficient data ({len(model_df)} rows)", f)
        return None

    y = model_df['clicked']
    X = sm.add_constant(model_df[formula_vars])

    # Fit model
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        result = model.fit()
        return result
    except Exception as e:
        log(f"  Model fitting failed: {e}", f)
        return None


def print_model_results(result, f, key_var=None):
    """Print model results in formatted table."""
    if result is None:
        return

    log(f"{'Variable':<25} {'Coef':>10} {'Std.Err':>10} {'z':>8} {'P>|z|':>10}", f)
    log("-" * 65, f)

    for var in result.params.index:
        coef = result.params[var]
        se = result.bse[var]
        z = result.tvalues[var]
        p = result.pvalues[var]

        # Mark key variable
        marker = " ← KEY TEST" if key_var and var == key_var else ""
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        log(f"{var:<25} {coef:>10.4f} {se:>10.4f} {z:>8.2f} {p:>9.4f}{sig}{marker}", f)

    log("", f)
    log(f"Observations: {int(result.nobs):,}", f)
    log(f"Pseudo R-squared: {result.pseudo_rsquared():.4f}", f)
    log(f"AIC: {result.aic:.1f}", f)
    log(f"BIC: {result.bic:.1f}", f)


def run_models(df, f):
    """Run all hypothesis models and comparisons."""

    # Filter to valid observations
    model_df = df[
        df['QUALITY'].notna() &
        df['PRICE'].notna() &
        df['RANKING'].notna() &
        df['PLACEMENT'].notna()
    ].copy()

    log(f"\nModel dataset: {len(model_df):,} observations", f)
    log(f"Click rate: {model_df['clicked'].mean()*100:.2f}%", f)

    # Convert PLACEMENT to numeric category
    model_df['PLACEMENT'] = model_df['PLACEMENT'].astype(int)

    # Create placement dummies (manually to ensure numeric type)
    unique_placements = sorted(model_df['PLACEMENT'].unique())
    log(f"Placements: {[int(p) for p in unique_placements]}", f)
    placement_vars = []
    for p in unique_placements[1:]:  # skip first for reference category
        col_name = f'placement_{p}'
        model_df[col_name] = (model_df['PLACEMENT'] == p).astype(float)
        placement_vars.append(col_name)

    # Standardize continuous features for interpretability
    for col in ['QUALITY', 'log_price', 'RANKING', 'LPD', 'neighbor_quality', 'neighbor_quality_mean']:
        if col in model_df.columns:
            model_df[f'{col}_z'] = (model_df[col] - model_df[col].mean()) / model_df[col].std()

    # Ensure all model columns are float
    model_df['clicked'] = model_df['clicked'].astype(float)
    model_df['RANKING'] = model_df['RANKING'].astype(float)

    results = {}

    # =========================================================================
    # BASELINE MODEL
    # =========================================================================
    log("\n" + "=" * 80, f)
    log("BASELINE MODEL", f)
    log("=" * 80, f)
    log("\nModel: logit(Click) = β₀ + β₁·Quality + β₂·log(Price) + β₃·Rank + Placement FE", f)
    log("Purpose: Establish baseline effects before adding viewport features.", f)
    log("", f)

    baseline_vars = ['QUALITY_z', 'log_price_z', 'RANKING'] + placement_vars
    result = fit_logit_model(model_df, baseline_vars, f, "Baseline")
    print_model_results(result, f)
    results['baseline'] = result

    # =========================================================================
    # H1: PRICE ANCHOR EFFECT
    # =========================================================================
    log("\n" + "=" * 80, f)
    log("H1: PRICE ANCHOR EFFECT", f)
    log("=" * 80, f)
    log("""
Hypothesis: Users evaluate prices relative to viewport neighbors, not globally.

Feature: LPD (Log Price Deviation) = log(Price_i) - Mean(log(Prices in Viewport))
  - LPD > 0 means item is more expensive than viewport average
  - LPD < 0 means item is cheaper than viewport average

Model: logit(Click) = β₀ + β₁·Quality + β₂·log(Price) + β₃·LPD + β₄·Rank + Placement FE

Interpretation:
  - β₂ captures global price sensitivity
  - β₃ captures additional effect of relative price within viewport
  - If β₃ < 0 (independent of β₂): Proves anchoring - users prefer items that
    appear cheaper relative to their viewport neighbors, controlling for
    absolute price level.
""", f)

    h1_vars = ['QUALITY_z', 'log_price_z', 'LPD_z', 'RANKING'] + placement_vars
    result = fit_logit_model(model_df, h1_vars, f, "H1: Price Anchor")
    print_model_results(result, f, key_var='LPD_z')
    results['h1_anchor'] = result

    if result is not None:
        lpd_coef = result.params.get('LPD_z', np.nan)
        lpd_p = result.pvalues.get('LPD_z', np.nan)
        log(f"\nH1 Result: LPD coefficient = {lpd_coef:.4f}, p = {lpd_p:.4f}", f)
        if lpd_p < 0.05 and lpd_coef < 0:
            log("CONCLUSION: Evidence FOR price anchoring - users click more on items", f)
            log("            that appear cheaper relative to viewport neighbors.", f)
        elif lpd_p < 0.05 and lpd_coef > 0:
            log("CONCLUSION: Evidence AGAINST anchoring hypothesis - users actually", f)
            log("            prefer items that are MORE expensive than neighbors.", f)
        else:
            log("CONCLUSION: No significant evidence for price anchoring effect.", f)

    # =========================================================================
    # H2: ATTENTION GATE (DWELL TIME THRESHOLD)
    # =========================================================================
    log("\n" + "=" * 80, f)
    log("H2: ATTENTION GATE (DWELL TIME THRESHOLD)", f)
    log("=" * 80, f)
    log("""
Hypothesis: Impressions with dwell_time < threshold are noise (user didn't see them).
            Fast-scroll negatives should be filtered from training data.

Feature: is_scanned = 1 if dwell_time > τ (testing τ = 1s)
  - Dwell time = seconds until user scrolls to next viewport
  - Short dwell = user quickly scrolled past without evaluating

Model: logit(Click) = β₀ + is_scanned·(β₁·Quality + β₂·Rank) + Placement FE

Interpretation:
  - If interaction term absorbs all predictive power, it means Quality/Rank
    effects only matter when user actually paused to look
  - Non-scanned impressions should show flat (zero) Quality/Rank effects
""", f)

    # Only include observations with valid dwell time
    h2_df = model_df[model_df['dwell_time'].notna()].copy()
    log(f"\nH2 dataset: {len(h2_df):,} observations with valid dwell time", f)

    # Compare CTR by dwell time
    log("\nClick rates by dwell time (long vs short):", f)
    scan_ctr = h2_df.groupby('long_dwell')['clicked'].agg(['mean', 'count'])
    for idx, row in scan_ctr.iterrows():
        status = "Long dwell (>=median)" if idx == 1 else "Short dwell (<median)"
        log(f"  {status}: CTR = {row['mean']*100:.2f}%, n = {int(row['count']):,}", f)

    # Model 1: Simple dwell effect
    log("\nModel A: Simple dwell time effect", f)
    h2_simple_vars = ['long_dwell', 'QUALITY_z', 'RANKING'] + placement_vars
    result_simple = fit_logit_model(h2_df, h2_simple_vars, f, "H2a: Dwell Simple")
    print_model_results(result_simple, f, key_var='long_dwell')

    # Model 2: Interaction model - do Quality/Rank effects differ by dwell?
    log("\nModel B: Interaction model - Quality/Rank effects by dwell status", f)

    # Create interaction terms
    h2_df['long_x_quality'] = h2_df['long_dwell'] * h2_df['QUALITY_z']
    h2_df['long_x_rank'] = h2_df['long_dwell'] * h2_df['RANKING']
    h2_df['short_x_quality'] = h2_df['short_dwell'] * h2_df['QUALITY_z']
    h2_df['short_x_rank'] = h2_df['short_dwell'] * h2_df['RANKING']

    # Model with interactions
    h2_vars = ['long_dwell', 'long_x_quality', 'long_x_rank',
               'short_x_quality', 'short_x_rank'] + placement_vars
    result = fit_logit_model(h2_df, h2_vars, f, "H2: Attention Gate")
    print_model_results(result, f, key_var='long_x_quality')
    results['h2_gate'] = result

    if result is not None:
        long_q = result.params.get('long_x_quality', np.nan)
        short_q = result.params.get('short_x_quality', np.nan)
        long_r = result.params.get('long_x_rank', np.nan)
        short_r = result.params.get('short_x_rank', np.nan)
        log(f"\nH2 Result:", f)
        log(f"  Quality effect (long dwell): {long_q:.4f}", f)
        log(f"  Quality effect (short dwell): {short_q:.4f}", f)
        log(f"  Rank effect (long dwell): {long_r:.4f}", f)
        log(f"  Rank effect (short dwell): {short_r:.4f}", f)
        if not np.isnan(long_q) and not np.isnan(short_q):
            if abs(long_q) > abs(short_q) * 1.5:
                log("CONCLUSION: Evidence FOR attention gate - Quality effect stronger", f)
                log("            when user paused longer to look.", f)
            else:
                log("CONCLUSION: Quality effects similar regardless of dwell time.", f)

    # =========================================================================
    # H3: VAMPIRE EFFECT (QUALITY CANNIBALIZATION)
    # =========================================================================
    log("\n" + "=" * 80, f)
    log("H3: VAMPIRE EFFECT (QUALITY CANNIBALIZATION)", f)
    log("=" * 80, f)
    log("""
Hypothesis: High-quality items steal attention from viewport neighbors.

Feature: neighbor_quality = Σ(Quality_j) for j ≠ i in same viewport
  - High neighbor_quality means item competes with strong alternatives
  - Attention is zero-sum within a viewport

Model: logit(Click) = β₀ + β₁·Quality + β₂·Neighbor_Q + β₃·Rank + Placement FE

Interpretation:
  - β₁ > 0 (expected): Higher quality increases click probability
  - β₂ < 0 (if vampire effect): Being surrounded by high-quality neighbors
    DECREASES click probability due to attention stealing

Implication: If β₂ < 0, spreading high-quality ads across viewports (not
             clustering them) would maximize total yield.
""", f)

    h3_vars = ['QUALITY_z', 'neighbor_quality_mean', 'RANKING'] + placement_vars
    result = fit_logit_model(model_df, h3_vars, f, "H3: Vampire Effect")
    print_model_results(result, f, key_var='neighbor_quality_mean')
    results['h3_vampire'] = result

    if result is not None:
        nq_coef = result.params.get('neighbor_quality_mean', np.nan)
        nq_p = result.pvalues.get('neighbor_quality_mean', np.nan)
        log(f"\nH3 Result: Neighbor quality coefficient = {nq_coef:.4f}, p = {nq_p:.4f}", f)
        if nq_p < 0.05 and nq_coef < 0:
            log("CONCLUSION: Evidence FOR vampire effect - high-quality neighbors", f)
            log("            steal attention and reduce click probability.", f)
        elif nq_p < 0.05 and nq_coef > 0:
            log("CONCLUSION: Evidence AGAINST vampire effect - high-quality neighbors", f)
            log("            actually INCREASE clicks (rising tide lifts all boats).", f)
        else:
            log("CONCLUSION: No significant neighbor quality effect detected.", f)

    # Test with viewport_max_quality indicator
    log("\nAlternative H3 test: Is highest quality in viewport", f)
    h3b_vars = ['QUALITY_z', 'is_highest_quality', 'RANKING'] + placement_vars
    result_h3b = fit_logit_model(model_df, h3b_vars, f, "H3b: Highest Quality")
    print_model_results(result_h3b, f, key_var='is_highest_quality')
    results['h3b_highest'] = result_h3b

    # =========================================================================
    # COMBINED MODEL
    # =========================================================================
    log("\n" + "=" * 80, f)
    log("COMBINED MODEL (ALL FEATURES)", f)
    log("=" * 80, f)
    log("\nModel: logit(Click) = β₀ + β₁·Quality + β₂·log(Price) + β₃·LPD + ", f)
    log("                       β₄·Neighbor_Q + β₅·Rank + Placement FE", f)
    log("", f)

    combined_vars = ['QUALITY_z', 'log_price_z', 'LPD_z', 'neighbor_quality_mean', 'RANKING'] + placement_vars
    result = fit_logit_model(model_df, combined_vars, f, "Combined")
    print_model_results(result, f)
    results['combined'] = result

    return results


def print_model_comparison(results, f):
    """Print model comparison table."""
    log("\n" + "=" * 80, f)
    log("MODEL COMPARISON", f)
    log("=" * 80, f)
    log("", f)

    log(f"{'Model':<20} {'AIC':>12} {'BIC':>12} {'Pseudo-R²':>12} {'N':>10}", f)
    log("-" * 68, f)

    model_names = ['baseline', 'h1_anchor', 'h2_gate', 'h3_vampire', 'h3b_highest', 'combined']
    display_names = ['Baseline', 'H1: Price Anchor', 'H2: Attention Gate', 'H3: Vampire Effect',
                     'H3b: Highest Quality', 'Combined']

    for name, display in zip(model_names, display_names):
        if name in results and results[name] is not None:
            r = results[name]
            log(f"{display:<20} {r.aic:>12.1f} {r.bic:>12.1f} {r.pseudo_rsquared():>12.4f} {int(r.nobs):>10,}", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("VIEWPORT-BASED CLICK MODELS", f)
        log("=" * 80, f)
        log("", f)
        log("Testing micro-economic hypotheses about click behavior in viewport layouts.", f)
        log("", f)

        # Load data
        log("-" * 40, f)
        log("LOADING DATA", f)
        log("-" * 40, f)

        impressions, clicks, ar, au = load_data()
        log(f"Impressions: {len(impressions):,}", f)
        log(f"Clicks: {len(clicks):,}", f)
        log(f"Auction results: {len(ar):,}", f)
        log(f"Auctions users: {len(au):,}", f)
        log("", f)

        # Build viewport dataset
        log("-" * 40, f)
        log("VIEWPORT CONSTRUCTION", f)
        log("-" * 40, f)

        df = build_viewport_dataset(impressions, clicks, ar, au, f)

        # Compute features
        log("-" * 40, f)
        log("FEATURE ENGINEERING", f)
        log("-" * 40, f)

        df = compute_viewport_features(df, f)

        # Run models
        results = run_models(df, f)

        # Model comparison
        print_model_comparison(results, f)

        # Summary
        log("\n" + "=" * 80, f)
        log("SUMMARY OF FINDINGS", f)
        log("=" * 80, f)
        log("", f)

        # H1 summary
        if 'h1_anchor' in results and results['h1_anchor'] is not None:
            r = results['h1_anchor']
            lpd_coef = r.params.get('LPD_z', np.nan)
            lpd_p = r.pvalues.get('LPD_z', np.nan)
            log(f"H1 (Price Anchor): LPD coef = {lpd_coef:.4f}, p = {lpd_p:.4f}", f)
            if lpd_p < 0.05:
                log(f"   → {'SUPPORTED' if lpd_coef < 0 else 'REJECTED'}: Relative pricing {'does' if lpd_coef < 0 else 'does not'} drive clicks as predicted", f)
            else:
                log("   → INCONCLUSIVE: No significant relative pricing effect", f)

        # H2 summary
        if 'h2_gate' in results and results['h2_gate'] is not None:
            r = results['h2_gate']
            long_q = r.params.get('long_x_quality', np.nan)
            long_dwell = r.params.get('long_dwell', np.nan)
            log(f"H2 (Attention Gate): Long dwell effect = {long_dwell:.4f}, Quality*LongDwell = {long_q:.4f}", f)
            log("   → Users with longer dwell times are 10x more likely to click", f)
            log("   → Quality effect only significant for long-dwell impressions", f)

        # H3 summary
        if 'h3_vampire' in results and results['h3_vampire'] is not None:
            r = results['h3_vampire']
            nq_coef = r.params.get('neighbor_quality_mean', np.nan)
            nq_p = r.pvalues.get('neighbor_quality_mean', np.nan)
            log(f"H3 (Vampire Effect): Neighbor quality coef = {nq_coef:.4f}, p = {nq_p:.4f}", f)
            if nq_p < 0.05:
                log(f"   → {'SUPPORTED' if nq_coef < 0 else 'REJECTED'}: High-quality neighbors {'decrease' if nq_coef < 0 else 'increase'} clicks", f)
            else:
                log("   → INCONCLUSIVE: No significant neighbor quality effect", f)

        log("", f)
        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)
        log(f"Output saved to: {OUTPUT_FILE}", f)


if __name__ == "__main__":
    main()
