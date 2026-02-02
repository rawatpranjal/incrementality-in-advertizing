#!/usr/bin/env python3
"""
Discrete Choice Model (Multinomial Logit) for Viewport Click Analysis

Models click behavior as a discrete choice among viewport alternatives (products + "no click"
outside option) using Multinomial Logit. This properly models the zero-sum nature of attention
within a viewport.

Key Hypothesis (H4): Dwell time reduces outside option attractiveness
- If gamma < 0: Longer dwell "unlocks" ads by making "scroll past" less attractive
- This explains why Quality effects only appear for long-dwell impressions
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import pylogit as pl
    PYLOGIT_AVAILABLE = True
except ImportError:
    PYLOGIT_AVAILABLE = False

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_r2"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = RESULTS_DIR / "14_discrete_choice.txt"

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
# VIEWPORT CONSTRUCTION (reused from 13_viewport_models.py)
# =============================================================================
def build_viewport_dataset(impressions, clicks, ar, au, f):
    """
    Build impression-level dataset with viewport features.

    Viewport = group of impressions with same (AUCTION_ID, OCCURRED_AT).
    """
    log("Building viewport dataset...", f)

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

    # Create viewport ID
    df['viewport_id'] = df['AUCTION_ID'].astype(str) + '_' + df['OCCURRED_AT'].astype(str)

    n_viewports = df['viewport_id'].nunique()
    log(f"  Viewports identified: {n_viewports:,}", f)

    # Viewport size statistics
    viewport_sizes = df.groupby('viewport_id').size()
    log(f"  Viewport size: mean={viewport_sizes.mean():.2f}, median={viewport_sizes.median():.0f}, max={viewport_sizes.max()}", f)

    return df


def compute_dwell_time(df, f):
    """Compute dwell time for each viewport."""
    log("\nComputing dwell times...", f)

    df = df.copy()
    df = df.sort_values(['AUCTION_ID', 'OCCURRED_AT'])

    # Get unique viewport times per auction
    viewport_times = df.groupby('AUCTION_ID')['OCCURRED_AT'].apply(
        lambda x: x.drop_duplicates().sort_values().tolist()
    ).to_dict()

    def get_dwell_time(row):
        auction_id = row['AUCTION_ID']
        current_time = row['OCCURRED_AT']
        times = viewport_times.get(auction_id, [])

        for t in times:
            if t > current_time:
                return (t - current_time).total_seconds()
        return np.nan

    tqdm.pandas(desc="Computing dwell times")
    df['dwell_time'] = df.progress_apply(get_dwell_time, axis=1)

    valid_dwell = df['dwell_time'].dropna()
    log(f"  Dwell time: mean={valid_dwell.mean():.2f}s, median={valid_dwell.median():.2f}s", f)
    log(f"  Missing (last viewport): {df['dwell_time'].isna().sum():,}", f)

    return df


# =============================================================================
# CHOICE DATA TRANSFORMATION
# =============================================================================
def transform_to_choice_format(df, f):
    """
    Transform viewport dataset to long-format choice data for multinomial logit.

    Each viewport becomes a choice scenario with:
    - One row per alternative (products in viewport + outside option "no_click")
    - choice=1 for the chosen alternative, 0 otherwise
    - If no click occurred, outside option has choice=1
    """
    log("\n" + "-" * 40, f)
    log("TRANSFORMING TO CHOICE FORMAT", f)
    log("-" * 40, f)

    # Filter to viewports with valid features
    valid_df = df[
        df['QUALITY'].notna() &
        df['PRICE'].notna() &
        df['RANKING'].notna() &
        df['dwell_time'].notna()
    ].copy()

    log(f"Impressions with valid features: {len(valid_df):,}", f)

    # Get unique viewports
    viewports = valid_df['viewport_id'].unique()
    log(f"Viewports with valid data: {len(viewports):,}", f)

    # Standardize features (for numerical stability)
    for col in ['QUALITY', 'PRICE', 'RANKING']:
        valid_df[f'{col}_std'] = (valid_df[col] - valid_df[col].mean()) / valid_df[col].std()

    # Log-transform price
    valid_df['log_price'] = np.log(valid_df['PRICE'].clip(lower=1))
    valid_df['log_price_std'] = (valid_df['log_price'] - valid_df['log_price'].mean()) / valid_df['log_price'].std()

    # Standardize dwell time
    valid_df['dwell_time_std'] = (valid_df['dwell_time'] - valid_df['dwell_time'].mean()) / valid_df['dwell_time'].std()

    # Build choice records
    choice_records = []

    for viewport_id in tqdm(viewports, desc="Building choice scenarios"):
        viewport_data = valid_df[valid_df['viewport_id'] == viewport_id]

        if len(viewport_data) == 0:
            continue

        # Check if any click occurred in this viewport
        any_click = viewport_data['clicked'].sum() > 0
        dwell_time = viewport_data['dwell_time'].iloc[0]  # Same for all items in viewport
        dwell_time_std = viewport_data['dwell_time_std'].iloc[0]

        # Add outside option (no_click)
        choice_records.append({
            'case_id': viewport_id,
            'alt_id': 'no_click',
            'alt_num': 0,
            'choice': 0 if any_click else 1,
            'is_product': 0,
            'quality': np.nan,
            'price': np.nan,
            'log_price': np.nan,
            'rank': np.nan,
            'quality_std': 0.0,  # Will use 0 for outside option
            'price_std': 0.0,
            'log_price_std': 0.0,
            'rank_std': 0.0,
            'dwell_time': dwell_time,
            'dwell_time_std': dwell_time_std,
        })

        # Add product alternatives
        for idx, (_, row) in enumerate(viewport_data.iterrows(), start=1):
            choice_records.append({
                'case_id': viewport_id,
                'alt_id': f"product_{row['PRODUCT_ID'][:8]}",  # Truncate for readability
                'alt_num': idx,
                'choice': int(row['clicked']),
                'is_product': 1,
                'quality': row['QUALITY'],
                'price': row['PRICE'],
                'log_price': row['log_price'],
                'rank': row['RANKING'],
                'quality_std': row['QUALITY_std'],
                'price_std': row['PRICE_std'],
                'log_price_std': row['log_price_std'],
                'rank_std': row['RANKING_std'],
                'dwell_time': dwell_time,
                'dwell_time_std': dwell_time_std,
            })

    choice_df = pd.DataFrame(choice_records)

    log(f"\nChoice dataset created:", f)
    log(f"  Total rows: {len(choice_df):,}", f)
    log(f"  Choice scenarios (viewports): {choice_df['case_id'].nunique():,}", f)
    log(f"  Mean alternatives per scenario: {len(choice_df) / choice_df['case_id'].nunique():.2f}", f)

    # Validate: each case_id should have exactly one choice=1
    choices_per_case = choice_df.groupby('case_id')['choice'].sum()
    invalid_cases = (choices_per_case != 1).sum()
    if invalid_cases > 0:
        log(f"  WARNING: {invalid_cases} cases with invalid choice count", f)
        # Filter to valid cases
        valid_cases = choices_per_case[choices_per_case == 1].index
        choice_df = choice_df[choice_df['case_id'].isin(valid_cases)]
        log(f"  After filtering: {choice_df['case_id'].nunique():,} valid cases", f)

    # Outside option chosen statistics
    outside_chosen = choice_df[choice_df['alt_num'] == 0]['choice'].sum()
    total_cases = choice_df['case_id'].nunique()
    log(f"  Outside option chosen (no click): {outside_chosen:,} ({outside_chosen/total_cases*100:.1f}%)", f)

    return choice_df


# =============================================================================
# MULTINOMIAL LOGIT MODELS (using pylogit)
# =============================================================================
def fit_mnl_model_pylogit(choice_df, spec, names, f, model_name="Model"):
    """
    Fit Multinomial Logit using pylogit.

    Parameters:
    - choice_df: Long-format choice data
    - spec: OrderedDict mapping variable names to list of alternative lists
    - names: OrderedDict mapping variable names to coefficient names
    - f: File handle for logging
    - model_name: Display name for model
    """
    log(f"\n--- {model_name} ---", f)

    # Create numeric alternative IDs
    choice_df = choice_df.copy()
    alt_to_num = {alt: i for i, alt in enumerate(choice_df['alt_id'].unique())}
    choice_df['alt_num_id'] = choice_df['alt_id'].map(alt_to_num)

    # Create observation IDs (numeric)
    case_to_num = {case: i for i, case in enumerate(choice_df['case_id'].unique())}
    choice_df['obs_id'] = choice_df['case_id'].map(case_to_num)

    try:
        model = pl.create_choice_model(
            data=choice_df,
            alt_id_col='alt_num_id',
            obs_id_col='obs_id',
            choice_col='choice',
            specification=spec,
            names=names,
            model_type='MNL'
        )

        results = model.fit_mle(method='BFGS', maxiter=1000)
        return results

    except Exception as e:
        log(f"  Model fitting failed: {e}", f)
        return None


def print_mnl_results(results, f, key_param=None):
    """Print multinomial logit results."""
    if results is None:
        return

    log(f"\n{'Parameter':<30} {'Estimate':>10} {'Std.Err':>10} {'z':>8} {'P>|z|':>10}", f)
    log("-" * 70, f)

    summary = results.get_statsmodels_summary()
    params = results.params
    std_errs = results.bse
    z_vals = results.tvalues
    p_vals = results.pvalues

    for i, name in enumerate(params.index):
        coef = params[name]
        se = std_errs[name]
        z = z_vals[name]
        p = p_vals[name]

        marker = " <- KEY" if key_param and name == key_param else ""
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        log(f"{name:<30} {coef:>10.4f} {se:>10.4f} {z:>8.2f} {p:>9.4f}{sig}{marker}", f)

    log("", f)
    log(f"Log-likelihood: {results.llf:.2f}", f)
    log(f"AIC: {-2*results.llf + 2*len(params):.2f}", f)
    log(f"Observations: {int(results.nobs):,}", f)


# =============================================================================
# MANUAL MNL IMPLEMENTATION (fallback if pylogit unavailable)
# =============================================================================
def fit_mnl_manual(choice_df, f, model_name="Model", include_dwell_outside=False):
    """
    Manual implementation of Multinomial Logit using scipy.optimize.

    Utility specification:
    - Outside option: V_0 = alpha_0 + gamma * dwell_time (if include_dwell_outside)
    - Products: V_j = beta_0 + beta_1 * quality + beta_2 * log_price + beta_3 * rank
    """
    from scipy.optimize import minimize

    log(f"\n--- {model_name} ---", f)

    # Prepare data structures
    cases = choice_df['case_id'].unique()
    n_cases = len(cases)

    # Build case-level data structure
    case_data = []
    for case_id in tqdm(cases, desc="Preparing case data", leave=False):
        case_df = choice_df[choice_df['case_id'] == case_id].sort_values('alt_num')

        # Outside option (alt_num=0)
        outside = case_df[case_df['alt_num'] == 0].iloc[0]

        # Products (alt_num > 0)
        products = case_df[case_df['alt_num'] > 0]

        case_data.append({
            'dwell_time_std': outside['dwell_time_std'],
            'choice_idx': case_df['choice'].values.argmax(),  # Index of chosen alternative
            'n_alts': len(case_df),
            'product_features': products[['quality_std', 'log_price_std', 'rank_std']].values,
        })

    def negative_log_likelihood(params, include_dwell):
        """Compute negative log-likelihood for MNL."""
        if include_dwell:
            # params: [alpha_0, gamma, beta_0, beta_q, beta_p, beta_r]
            alpha_0, gamma, beta_0, beta_q, beta_p, beta_r = params
        else:
            # params: [alpha_0, beta_0, beta_q, beta_p, beta_r]
            alpha_0, beta_0, beta_q, beta_p, beta_r = params
            gamma = 0.0

        ll = 0.0
        for case in case_data:
            # Utility of outside option
            V_0 = alpha_0 + gamma * case['dwell_time_std']

            # Utilities of products
            features = case['product_features']
            V_products = beta_0 + beta_q * features[:, 0] + beta_p * features[:, 1] + beta_r * features[:, 2]

            # All utilities
            V_all = np.concatenate([[V_0], V_products])

            # Log-sum-exp for numerical stability
            max_V = np.max(V_all)
            log_denom = max_V + np.log(np.sum(np.exp(V_all - max_V)))

            # Log probability of chosen alternative
            choice_idx = case['choice_idx']
            ll += V_all[choice_idx] - log_denom

        return -ll  # Negative for minimization

    # Initial values
    if include_dwell_outside:
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        param_names = ['alpha_0 (no_click)', 'gamma (dwell->no_click)', 'beta_0 (product)',
                       'beta_quality', 'beta_log_price', 'beta_rank']
    else:
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        param_names = ['alpha_0 (no_click)', 'beta_0 (product)',
                       'beta_quality', 'beta_log_price', 'beta_rank']

    # Optimize
    log(f"  Fitting model with {len(x0)} parameters...", f)
    result = minimize(
        negative_log_likelihood,
        x0,
        args=(include_dwell_outside,),
        method='BFGS',
        options={'maxiter': 1000, 'disp': False}
    )

    if not result.success:
        log(f"  WARNING: Optimization did not converge: {result.message}", f)

    # Compute standard errors via Hessian inverse
    try:
        from scipy.optimize import approx_fprime

        def hessian_diag(x, f, eps=1e-5):
            """Approximate diagonal of Hessian."""
            n = len(x)
            hess_diag = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                hess_diag[i] = (f(x_plus, include_dwell_outside) - 2*f(x, include_dwell_outside) + f(x_minus, include_dwell_outside)) / (eps**2)
            return hess_diag

        hess_inv_diag = 1.0 / np.abs(hessian_diag(result.x, negative_log_likelihood))
        std_errs = np.sqrt(hess_inv_diag)
    except:
        std_errs = np.full(len(result.x), np.nan)

    # Print results
    log(f"\n{'Parameter':<30} {'Estimate':>10} {'Std.Err':>10} {'z':>8} {'P>|z|':>10}", f)
    log("-" * 70, f)

    for i, (name, coef, se) in enumerate(zip(param_names, result.x, std_errs)):
        if np.isnan(se):
            z = np.nan
            p = np.nan
        else:
            z = coef / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        marker = " <- KEY" if 'gamma' in name else ""

        if np.isnan(se):
            log(f"{name:<30} {coef:>10.4f} {'N/A':>10} {'N/A':>8} {'N/A':>10}{marker}", f)
        else:
            log(f"{name:<30} {coef:>10.4f} {se:>10.4f} {z:>8.2f} {p:>9.4f}{sig}{marker}", f)

    ll = -result.fun
    n_params = len(result.x)
    aic = -2 * ll + 2 * n_params

    log("", f)
    log(f"Log-likelihood: {ll:.2f}", f)
    log(f"AIC: {aic:.2f}", f)
    log(f"Observations: {n_cases:,}", f)

    return {
        'params': dict(zip(param_names, result.x)),
        'std_errs': dict(zip(param_names, std_errs)),
        'log_likelihood': ll,
        'aic': aic,
        'n_obs': n_cases,
        'converged': result.success,
    }


# =============================================================================
# MODEL COMPARISON
# =============================================================================
def likelihood_ratio_test(ll_restricted, ll_unrestricted, df_diff, f):
    """Perform likelihood ratio test."""
    lr_stat = 2 * (ll_unrestricted - ll_restricted)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

    log(f"\nLikelihood Ratio Test:", f)
    log(f"  LR statistic: {lr_stat:.4f}", f)
    log(f"  Degrees of freedom: {df_diff}", f)
    log(f"  p-value: {p_value:.4f}", f)

    if p_value < 0.05:
        log("  -> Unrestricted model significantly better (reject H0)", f)
    else:
        log("  -> No significant difference (fail to reject H0)", f)

    return lr_stat, p_value


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_discrete_choice_analysis(choice_df, f):
    """Run all discrete choice models."""

    log("\n" + "=" * 80, f)
    log("MODEL M1: BASELINE (FIXED OUTSIDE OPTION)", f)
    log("=" * 80, f)
    log("""
Utility specification:
  V_none = alpha_0               (fixed outside option)
  V_product = beta_0 + beta_1*Quality + beta_2*log(Price) + beta_3*Rank

This model treats the outside option as having constant utility, unaffected
by viewport characteristics.
""", f)

    m1_results = fit_mnl_manual(choice_df, f, "M1: Baseline", include_dwell_outside=False)

    log("\n" + "=" * 80, f)
    log("MODEL M2: DWELL-VARYING OUTSIDE OPTION (KEY TEST)", f)
    log("=" * 80, f)
    log("""
Utility specification:
  V_none = alpha_0 + gamma*DwellTime   <- KEY HYPOTHESIS
  V_product = beta_0 + beta_1*Quality + beta_2*log(Price) + beta_3*Rank

This model tests H4: Does dwell time affect the attractiveness of the
"scroll past" option?

Interpretation:
  If gamma < 0: Longer dwell REDUCES appeal of "no click"
                -> Attention gate opens with dwell time
  If gamma > 0: Longer dwell INCREASES appeal of "no click"
                -> User more likely to leave if they pause
  If gamma = 0: Dwell time doesn't affect choice process
""", f)

    m2_results = fit_mnl_manual(choice_df, f, "M2: Dwell-Varying Outside", include_dwell_outside=True)

    # Likelihood ratio test M2 vs M1
    if m1_results and m2_results:
        log("\n" + "=" * 80, f)
        log("LIKELIHOOD RATIO TEST: M2 vs M1", f)
        log("=" * 80, f)
        log("\nH0: gamma = 0 (dwell time has no effect on outside option)", f)
        log("H1: gamma != 0 (dwell time affects outside option utility)", f)

        likelihood_ratio_test(
            m1_results['log_likelihood'],
            m2_results['log_likelihood'],
            df_diff=1,
            f=f
        )

        gamma = m2_results['params'].get('gamma (dwell->no_click)', 0)
        log(f"\nInterpretation of gamma = {gamma:.4f}:", f)
        if gamma < 0:
            log("  gamma < 0: Longer dwell time makes 'scroll past' LESS attractive", f)
            log("  -> Supports H4: Dwell time opens the 'attention gate'", f)
            log("  -> Users who pause longer are more likely to engage", f)
        elif gamma > 0:
            log("  gamma > 0: Longer dwell time makes 'scroll past' MORE attractive", f)
            log("  -> Users who pause longer are still more likely to leave", f)
            log("  -> Dwell time may indicate confusion or disinterest", f)
        else:
            log("  gamma = 0: Dwell time has no effect on choice process", f)

    return m1_results, m2_results


def compute_choice_probabilities(m2_results, choice_df, f):
    """Compute predicted choice probabilities and marginal effects."""
    if m2_results is None:
        return

    log("\n" + "=" * 80, f)
    log("PREDICTED CHOICE PROBABILITIES", f)
    log("=" * 80, f)

    params = m2_results['params']
    alpha_0 = params['alpha_0 (no_click)']
    gamma = params['gamma (dwell->no_click)']
    beta_0 = params['beta_0 (product)']
    beta_q = params['beta_quality']
    beta_p = params['beta_log_price']
    beta_r = params['beta_rank']

    # Compute for typical viewport
    log("\nAt average feature values (all standardized = 0):", f)

    for dwell_std in [-1, 0, 1]:
        V_0 = alpha_0 + gamma * dwell_std
        V_product = beta_0  # All other features at 0

        # Assume 3 products in viewport (typical)
        V_all = np.array([V_0] + [V_product] * 3)
        exp_V = np.exp(V_all - np.max(V_all))
        probs = exp_V / exp_V.sum()

        dwell_label = "short (-1 SD)" if dwell_std == -1 else "average (0)" if dwell_std == 0 else "long (+1 SD)"
        log(f"\n  Dwell time: {dwell_label}", f)
        log(f"    P(no click) = {probs[0]*100:.1f}%", f)
        log(f"    P(click any product) = {(1-probs[0])*100:.1f}%", f)
        log(f"    P(click product 1|click) = {probs[1]/(1-probs[0])*100:.1f}%", f)

    # Effect of quality
    log("\n\nMarginal effect of Quality (+1 SD) on click probability:", f)

    for dwell_std in [-1, 0, 1]:
        V_0 = alpha_0 + gamma * dwell_std

        # Base: all products at average quality
        V_base = np.array([V_0, beta_0, beta_0, beta_0])

        # Shocked: product 1 has +1 SD quality
        V_shock = np.array([V_0, beta_0 + beta_q, beta_0, beta_0])

        exp_base = np.exp(V_base - np.max(V_base))
        exp_shock = np.exp(V_shock - np.max(V_shock))

        prob_base = exp_base[1] / exp_base.sum()
        prob_shock = exp_shock[1] / exp_shock.sum()

        dwell_label = "short" if dwell_std == -1 else "average" if dwell_std == 0 else "long"
        log(f"  Dwell {dwell_label}: P(click product 1) changes from {prob_base*100:.2f}% to {prob_shock*100:.2f}%", f)
        log(f"                   Marginal effect: {(prob_shock-prob_base)*100:.2f} pp", f)


def compute_cross_elasticities(m2_results, f):
    """Compute cross-price elasticities between products."""
    if m2_results is None:
        return

    log("\n" + "=" * 80, f)
    log("CROSS-PRICE ELASTICITY ANALYSIS", f)
    log("=" * 80, f)
    log("""
Cross-price elasticity measures how a price change for one product
affects the choice probability of another product.

In MNL, cross-elasticities follow the IIA property:
  e_jk = P_k * beta_price   (for all j != k)

This is the percentage change in P(choose j) for a 1% increase in price of k.
""", f)

    params = m2_results['params']
    beta_p = params['beta_log_price']

    # At average values with 3 products
    alpha_0 = params['alpha_0 (no_click)']
    gamma = params['gamma (dwell->no_click)']
    beta_0 = params['beta_0 (product)']

    V_0 = alpha_0  # average dwell
    V_product = beta_0

    V_all = np.array([V_0, V_product, V_product, V_product])
    exp_V = np.exp(V_all - np.max(V_all))
    probs = exp_V / exp_V.sum()

    # Cross elasticity: how does 10% price increase in product 2 affect product 1?
    # d(P_1)/d(ln P_2) = P_1 * P_2 * (-beta_p)

    log(f"\nAt average feature values with 3 products:", f)
    log(f"  P(no click) = {probs[0]*100:.1f}%", f)
    log(f"  P(product 1) = {probs[1]*100:.1f}%", f)
    log(f"  P(product 2) = {probs[2]*100:.1f}%", f)
    log(f"  P(product 3) = {probs[3]*100:.1f}%", f)

    log(f"\n  beta_log_price = {beta_p:.4f}", f)

    # Own price elasticity
    own_elast = beta_p * (1 - probs[1])
    log(f"\n  Own-price elasticity for product 1: {own_elast:.4f}", f)
    log(f"  -> If product 1's price increases by 10%, P(choose 1) changes by {own_elast*10:.2f}%", f)

    # Cross-price elasticity
    cross_elast = -beta_p * probs[2]
    log(f"\n  Cross-price elasticity (product 1 w.r.t. product 2): {cross_elast:.4f}", f)
    log(f"  -> If product 2's price increases by 10%, P(choose 1) changes by {cross_elast*10:.2f}%", f)

    if cross_elast > 0:
        log("\n  Interpretation: Products are substitutes (as expected)", f)
        log("  When competitor price rises, own demand increases", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("DISCRETE CHOICE MODEL (MULTINOMIAL LOGIT)", f)
        log("Viewport Click Analysis", f)
        log("=" * 80, f)
        log("", f)
        log("This analysis models click behavior as a discrete choice among viewport", f)
        log("alternatives (products + 'no click' outside option) using Multinomial Logit.", f)
        log("", f)
        log("Key Hypothesis (H4): Dwell time reduces outside option attractiveness", f)
        log("  If gamma < 0: Longer dwell 'unlocks' ads by making 'scroll past' less attractive", f)
        log("", f)

        if not PYLOGIT_AVAILABLE:
            log("NOTE: pylogit not available, using manual MNL implementation", f)

        # Load data
        log("\n" + "-" * 40, f)
        log("LOADING DATA", f)
        log("-" * 40, f)

        impressions, clicks, ar, au = load_data()
        log(f"Impressions: {len(impressions):,}", f)
        log(f"Clicks: {len(clicks):,}", f)
        log(f"Auction results: {len(ar):,}", f)
        log(f"Auctions users: {len(au):,}", f)

        # Build viewport dataset
        log("\n" + "-" * 40, f)
        log("VIEWPORT CONSTRUCTION", f)
        log("-" * 40, f)

        df = build_viewport_dataset(impressions, clicks, ar, au, f)
        df = compute_dwell_time(df, f)

        # Transform to choice format
        choice_df = transform_to_choice_format(df, f)

        # Data validation
        log("\n" + "-" * 40, f)
        log("DATA VALIDATION", f)
        log("-" * 40, f)

        # Check 1: Each case has exactly one choice=1
        choices_per_case = choice_df.groupby('case_id')['choice'].sum()
        log(f"Cases with exactly one choice: {(choices_per_case == 1).sum():,} / {len(choices_per_case):,}", f)

        # Check 2: Outside option chosen rate
        outside_chosen = choice_df[choice_df['alt_num'] == 0]['choice'].mean()
        log(f"Outside option chosen rate: {outside_chosen*100:.1f}% (expect ~97% if CTR ~3%)", f)

        # Check 3: Feature variation within cases
        feature_var = choice_df[choice_df['is_product'] == 1].groupby('case_id')['quality_std'].std().mean()
        log(f"Mean within-viewport quality std: {feature_var:.4f}", f)

        # Check 4: Missing values
        log(f"Missing dwell_time: {choice_df['dwell_time'].isna().sum():,}", f)

        # Run models
        m1_results, m2_results = run_discrete_choice_analysis(choice_df, f)

        # Choice probabilities
        compute_choice_probabilities(m2_results, choice_df, f)

        # Cross elasticities
        compute_cross_elasticities(m2_results, f)

        # Summary
        log("\n" + "=" * 80, f)
        log("CONCLUSIONS", f)
        log("=" * 80, f)

        if m2_results:
            gamma = m2_results['params'].get('gamma (dwell->no_click)', 0)
            beta_q = m2_results['params'].get('beta_quality', 0)
            beta_p = m2_results['params'].get('beta_log_price', 0)
            beta_r = m2_results['params'].get('beta_rank', 0)

            log("\n1. Outside Option Effect (H4 Test):", f)
            if gamma < 0:
                log(f"   gamma = {gamma:.4f} < 0: SUPPORTED", f)
                log("   Longer dwell time reduces the attractiveness of 'scroll past'", f)
                log("   -> Attention gate mechanism confirmed", f)
            else:
                log(f"   gamma = {gamma:.4f} >= 0: NOT SUPPORTED", f)
                log("   Dwell time does not make 'scroll past' less attractive", f)

            log("\n2. Product Attribute Effects:", f)
            log(f"   Quality (beta_q = {beta_q:.4f}):", f)
            if beta_q > 0:
                log("     Higher quality increases click probability (as expected)", f)
            else:
                log("     Quality effect not positive (unexpected)", f)

            log(f"   Price (beta_p = {beta_p:.4f}):", f)
            if beta_p < 0:
                log("     Higher price decreases click probability (as expected)", f)
            else:
                log("     Price effect not negative", f)

            log(f"   Rank (beta_r = {beta_r:.4f}):", f)
            if beta_r < 0:
                log("     Higher rank (worse position) decreases clicks (as expected)", f)
            else:
                log("     Rank effect not negative", f)

            log("\n3. Model Fit:", f)
            log(f"   M1 (baseline) Log-likelihood: {m1_results['log_likelihood']:.2f}", f)
            log(f"   M2 (dwell-varying) Log-likelihood: {m2_results['log_likelihood']:.2f}", f)
            log(f"   Improvement: {m2_results['log_likelihood'] - m1_results['log_likelihood']:.2f}", f)

        log("\n" + "=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)
        log(f"Output saved to: {OUTPUT_FILE}", f)


if __name__ == "__main__":
    main()
