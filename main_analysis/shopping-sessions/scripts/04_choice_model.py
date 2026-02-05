#!/usr/bin/env python3
"""
04_choice_model.py - Discrete Choice Model for Product Selection

Unit of Analysis: Product-session (choice occasion)
Dependent Variable: chosen (binary) - whether product was purchased
Choice Set: Products impressed/clicked during session + organic alternatives

Model: Conditional Logit / Mixed Logit
Purpose: Estimate how advertising exposure (impressed, clicked) affects
product choice conditional on purchase decision.

Key Identification:
- Within-session variation in exposure across products
- Products in consideration set that were NOT purchased as controls
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PATHS
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "0_data_pull" / "data"
RESULTS_DIR = BASE_DIR / "results"


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def section(title, fh):
    log("\n" + "="*80, fh)
    log(title, fh)
    log("="*80, fh)


def subsection(title, fh):
    log(f"\n--- {title} ---", fh)


def load_data(f):
    """Load all required data."""
    section("LOADING DATA", f)

    data = {}
    files = ['sessions', 'session_events', 'auctions_results', 'catalog', 'purchases']
    for name in files:
        path = DATA_DIR / f'{name}.parquet'
        if path.exists():
            data[name] = pd.read_parquet(path)
            log(f"  {name}: {len(data[name]):,} rows", f)
        else:
            log(f"  {name}: NOT FOUND", f)
            data[name] = pd.DataFrame()

    return data


def build_consideration_sets(data, f):
    """
    Build consideration set for each session.

    Consideration set includes:
    1. Products with impressions in the session
    2. Products with clicks in the session
    3. Products purchased in the session

    For each product-session, track exposure status.
    """
    section("BUILDING CONSIDERATION SETS", f)

    events = data['session_events']
    sessions = data['sessions']
    purchases = data['purchases']
    ar = data['auctions_results']

    # Filter to purchasing sessions only
    purch_sessions = sessions[sessions['purchased'] == 1]['session_id'].unique()
    log(f"Purchasing sessions: {len(purch_sessions):,}", f)

    # Get events for purchasing sessions
    events_purch = events[events['session_id'].isin(purch_sessions)].copy()

    # Build product-session records
    records = []

    for session_id in tqdm(purch_sessions, desc="Building consideration sets"):
        session_events = events_purch[events_purch['session_id'] == session_id]
        user_id = session_events['user_id'].iloc[0]

        # Get session time bounds
        session_start = session_events['event_time'].min()
        session_end = session_events['event_time'].max()

        # Products impressed
        impressed = session_events[session_events['event_type'] == 'impression']['product_id'].dropna().unique()

        # Products clicked
        clicked = session_events[session_events['event_type'] == 'click']['product_id'].dropna().unique()

        # Products purchased (from events)
        purchased_in_session = session_events[session_events['event_type'] == 'purchase']['product_id'].dropna().unique()

        # Consideration set = union of impressed, clicked, purchased
        consideration_set = set(impressed) | set(clicked) | set(purchased_in_session)

        if len(consideration_set) == 0:
            continue

        for product_id in consideration_set:
            was_impressed = product_id in impressed
            was_clicked = product_id in clicked
            was_purchased = product_id in purchased_in_session

            records.append({
                'session_id': session_id,
                'user_id': user_id,
                'product_id': product_id,
                'impressed': int(was_impressed),
                'clicked': int(was_clicked),
                'chosen': int(was_purchased),
            })

    choice_df = pd.DataFrame(records)
    log(f"\nChoice observations: {len(choice_df):,}", f)
    log(f"Unique sessions: {choice_df['session_id'].nunique():,}", f)
    log(f"Unique products: {choice_df['product_id'].nunique():,}", f)

    # Consideration set size distribution
    cs_size = choice_df.groupby('session_id').size()
    log(f"\nConsideration set size:", f)
    log(f"  Mean: {cs_size.mean():.1f}", f)
    log(f"  Median: {cs_size.median():.0f}", f)
    log(f"  Min: {cs_size.min()}", f)
    log(f"  Max: {cs_size.max()}", f)

    return choice_df


def add_product_attributes(choice_df, data, f):
    """Add product-level attributes from auctions_results and catalog."""
    subsection("Adding Product Attributes", f)

    ar = data['auctions_results']
    catalog = data['catalog']
    events = data['session_events']

    # Get quality scores from auctions_results (median per product)
    if len(ar) > 0:
        # Build aggregation dict dynamically based on available columns
        agg_dict = {}
        rename_cols = ['product_id']
        if 'quality' in ar.columns:
            agg_dict['quality'] = 'median'
            rename_cols.append('quality')
        if 'final_bid' in ar.columns:
            agg_dict['final_bid'] = 'median'
            rename_cols.append('final_bid')
        if 'price' in ar.columns:
            agg_dict['price'] = 'median'
            rename_cols.append('price_ar')
        if 'ranking' in ar.columns:
            agg_dict['ranking'] = 'median'
            rename_cols.append('ranking')

        if agg_dict:
            product_quality = ar.groupby('product_id').agg(agg_dict).reset_index()
            product_quality.columns = rename_cols

            choice_df = choice_df.merge(product_quality, on='product_id', how='left')
            merged_cols = [c for c in rename_cols if c != 'product_id']
            log(f"  Merged from auctions_results: {merged_cols}", f)
        else:
            log(f"  No quality/bid/price columns in auctions_results", f)

    # Get price from catalog
    if len(catalog) > 0:
        catalog_price = catalog[['product_id', 'price']].drop_duplicates()
        catalog_price.columns = ['product_id', 'price_catalog']
        choice_df = choice_df.merge(catalog_price, on='product_id', how='left')

        # Use catalog price if available, else auction price
        choice_df['price'] = choice_df['price_catalog'].fillna(choice_df.get('price_ar', np.nan))
        log(f"  Merged catalog price: {choice_df['price'].notna().sum():,} non-null", f)

    # Compute exposure intensity (number of impressions per product-session)
    if len(events) > 0:
        imp_counts = events[events['event_type'] == 'impression'].groupby(
            ['session_id', 'product_id']).size().reset_index(name='n_impressions_product')
        choice_df = choice_df.merge(imp_counts, on=['session_id', 'product_id'], how='left')
        choice_df['n_impressions_product'] = choice_df['n_impressions_product'].fillna(0)

        clk_counts = events[events['event_type'] == 'click'].groupby(
            ['session_id', 'product_id']).size().reset_index(name='n_clicks_product')
        choice_df = choice_df.merge(clk_counts, on=['session_id', 'product_id'], how='left')
        choice_df['n_clicks_product'] = choice_df['n_clicks_product'].fillna(0)

    # Fill missing values
    for col in ['quality', 'price', 'ranking']:
        if col in choice_df.columns:
            choice_df[col] = choice_df[col].fillna(choice_df[col].median())

    # Log transform price
    if 'price' in choice_df.columns:
        choice_df['log_price'] = np.log1p(choice_df['price'])

    log(f"\nFinal choice dataset: {len(choice_df):,} rows", f)
    log(f"Columns: {list(choice_df.columns)}", f)

    return choice_df


def choice_data_summary(choice_df, f):
    """Summary statistics of choice data."""
    section("CHOICE DATA SUMMARY", f)

    log(f"\nTotal observations: {len(choice_df):,}", f)
    log(f"Sessions: {choice_df['session_id'].nunique():,}", f)
    log(f"Products: {choice_df['product_id'].nunique():,}", f)

    subsection("Exposure Distribution", f)
    log(f"  Impressed: {choice_df['impressed'].sum():,} ({choice_df['impressed'].mean()*100:.1f}%)", f)
    log(f"  Clicked: {choice_df['clicked'].sum():,} ({choice_df['clicked'].mean()*100:.1f}%)", f)
    log(f"  Chosen: {choice_df['chosen'].sum():,} ({choice_df['chosen'].mean()*100:.1f}%)", f)

    subsection("Chosen by Exposure Status", f)
    # Cross-tab
    imp_chosen = choice_df.groupby(['impressed', 'clicked'])['chosen'].agg(['sum', 'count', 'mean'])
    log(f"\n  {'Impressed':>10s} {'Clicked':>10s} {'Chosen':>10s} {'Total':>10s} {'Rate':>10s}", f)
    log(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", f)
    for (imp, clk), row in imp_chosen.iterrows():
        log(f"  {imp:>10d} {clk:>10d} {row['sum']:>10.0f} {row['count']:>10.0f} {row['mean']*100:>10.2f}%", f)

    subsection("Product Attributes", f)
    for col in ['quality', 'log_price', 'ranking', 'n_impressions_product']:
        if col in choice_df.columns:
            vals = choice_df[col].dropna()
            log(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}", f)


def conditional_logit_mle(choice_df, f):
    """
    Estimate conditional logit model via MLE.

    Model:
    P(chosen_ij = 1 | session_i) = exp(V_ij) / sum_k exp(V_ik)

    where V_ij = beta_1 * impressed_ij + beta_2 * clicked_ij + beta_3 * quality_ij + beta_4 * log_price_ij

    Identification: Within-session variation in product attributes.
    """
    section("CONDITIONAL LOGIT MODEL", f)

    log(f"\nModel:", f)
    log(f"  P(chosen | session) = exp(V) / sum(exp(V))", f)
    log(f"  V = beta_1 * impressed + beta_2 * clicked + beta_3 * quality + beta_4 * log_price", f)
    log(f"\nIdentification: Within-session variation across products", f)

    # Prepare data
    features = ['impressed', 'clicked']
    if 'quality' in choice_df.columns and choice_df['quality'].notna().sum() > 0:
        features.append('quality')
    if 'log_price' in choice_df.columns and choice_df['log_price'].notna().sum() > 0:
        features.append('log_price')

    log(f"Features: {features}", f)

    # Filter to complete cases
    df = choice_df[['session_id', 'chosen'] + features].dropna().copy()
    log(f"Complete cases: {len(df):,}", f)

    # Group by session
    sessions = df['session_id'].unique()
    n_sessions = len(sessions)
    log(f"Sessions for estimation: {n_sessions:,}", f)

    # Create session index
    session_map = {s: i for i, s in enumerate(sessions)}
    df['session_idx'] = df['session_id'].map(session_map)

    # Prepare arrays
    X = df[features].values
    y = df['chosen'].values
    session_idx = df['session_idx'].values

    n_features = len(features)

    def neg_log_likelihood(beta):
        """Negative log-likelihood for conditional logit."""
        V = X @ beta
        ll = 0
        for i in range(n_sessions):
            mask = session_idx == i
            V_i = V[mask]
            y_i = y[mask]
            # Log-sum-exp for numerical stability
            max_V = V_i.max()
            log_sum_exp = max_V + np.log(np.sum(np.exp(V_i - max_V)))
            ll += np.sum(y_i * V_i) - np.sum(y_i) * log_sum_exp
        return -ll

    def gradient(beta):
        """Gradient of negative log-likelihood."""
        V = X @ beta
        grad = np.zeros(n_features)
        for i in range(n_sessions):
            mask = session_idx == i
            X_i = X[mask]
            V_i = V[mask]
            y_i = y[mask]
            # Softmax probabilities
            max_V = V_i.max()
            exp_V = np.exp(V_i - max_V)
            prob_i = exp_V / exp_V.sum()
            # Gradient contribution
            grad += X_i.T @ y_i - np.sum(y_i) * (X_i.T @ prob_i)
        return -grad

    # Initial values
    beta_init = np.zeros(n_features)

    # Optimize
    log(f"\nOptimizing...", f)
    result = minimize(neg_log_likelihood, beta_init, method='BFGS', jac=gradient,
                      options={'maxiter': 1000, 'disp': False})

    beta_hat = result.x
    log(f"Converged: {result.success}", f)
    log(f"Final log-likelihood: {-result.fun:.4f}", f)

    # Hessian for standard errors
    from scipy.optimize import approx_fprime
    eps = 1e-5
    hess = np.zeros((n_features, n_features))
    for i in range(n_features):
        def grad_i(beta):
            return gradient(beta)[i]
        hess[i, :] = approx_fprime(beta_hat, grad_i, eps)

    try:
        var_beta = np.linalg.inv(hess)
        se_beta = np.sqrt(np.diag(var_beta))
    except:
        log("  WARNING: Hessian not invertible, using pseudo-inverse", f)
        var_beta = np.linalg.pinv(hess)
        se_beta = np.sqrt(np.abs(np.diag(var_beta)))

    subsection("Results", f)
    log(f"\n{'Variable':20s} {'Coef':>12s} {'SE':>12s} {'z-stat':>10s} {'p-value':>10s} {'Odds Ratio':>12s}", f)
    log(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}", f)
    for i, name in enumerate(features):
        z = beta_hat[i] / se_beta[i] if se_beta[i] > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        odds = np.exp(beta_hat[i])
        stars = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))
        log(f"{name:20s} {beta_hat[i]:12.4f} {se_beta[i]:12.4f} {z:10.2f} {p_val:10.4f} {odds:12.4f} {stars}", f)

    # Model fit
    ll_null = -n_sessions * np.log(df.groupby('session_id').size().mean())  # Approximate
    pseudo_r2 = 1 - (-result.fun) / ll_null if ll_null != 0 else np.nan
    log(f"\nPseudo R-squared: {pseudo_r2:.4f}", f)
    log(f"N observations: {len(df):,}", f)
    log(f"N sessions: {n_sessions:,}", f)

    subsection("Interpretation", f)
    for i, name in enumerate(features):
        odds = np.exp(beta_hat[i])
        if name == 'impressed':
            log(f"  {name}: Being impressed increases odds of choice by {(odds-1)*100:.1f}%", f)
        elif name == 'clicked':
            log(f"  {name}: Being clicked increases odds of choice by {(odds-1)*100:.1f}%", f)
        elif name == 'quality':
            log(f"  {name}: 1 unit increase in quality changes odds by {(odds-1)*100:.2f}%", f)
        elif name == 'log_price':
            log(f"  {name}: 1% increase in price changes odds by {beta_hat[i]:.4f}%", f)

    return {'beta': beta_hat, 'se': se_beta, 'features': features, 'll': -result.fun}


def alternative_specs(choice_df, f):
    """Run alternative model specifications."""
    section("ALTERNATIVE SPECIFICATIONS", f)

    # Spec 1: Impressed only
    subsection("Spec 1: Impressed Only", f)
    df = choice_df[['session_id', 'chosen', 'impressed']].dropna()
    log(f"N = {len(df):,}", f)

    # Simple within-session comparison
    purch_if_impressed = df[df['impressed'] == 1]['chosen'].mean()
    purch_if_not = df[df['impressed'] == 0]['chosen'].mean()
    log(f"  Choice rate if impressed: {purch_if_impressed*100:.2f}%", f)
    log(f"  Choice rate if not impressed: {purch_if_not*100:.2f}%", f)
    log(f"  Difference: {(purch_if_impressed - purch_if_not)*100:.2f} pp", f)

    # Spec 2: Clicked only
    subsection("Spec 2: Clicked Only", f)
    df = choice_df[['session_id', 'chosen', 'clicked']].dropna()
    purch_if_clicked = df[df['clicked'] == 1]['chosen'].mean()
    purch_if_not = df[df['clicked'] == 0]['chosen'].mean()
    log(f"  Choice rate if clicked: {purch_if_clicked*100:.2f}%", f)
    log(f"  Choice rate if not clicked: {purch_if_not*100:.2f}%", f)
    log(f"  Difference: {(purch_if_clicked - purch_if_not)*100:.2f} pp", f)

    # Spec 3: Exposure intensity
    subsection("Spec 3: Exposure Intensity", f)
    if 'n_impressions_product' in choice_df.columns:
        df = choice_df[['session_id', 'chosen', 'n_impressions_product']].dropna()
        corr = df['chosen'].corr(df['n_impressions_product'])
        log(f"  Correlation(chosen, n_impressions): {corr:.4f}", f)

        # Binned analysis
        df['imp_bin'] = pd.cut(df['n_impressions_product'], bins=[0, 1, 2, 5, 100], labels=['1', '2', '3-5', '6+'])
        bin_rates = df.groupby('imp_bin')['chosen'].mean()
        log(f"\n  Impressions bin -> Choice rate:", f)
        for bin_label, rate in bin_rates.items():
            log(f"    {bin_label}: {rate*100:.2f}%", f)


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / '04_choice_model.txt'

    with open(output_file, 'w') as f:
        log("="*80, f)
        log("04_CHOICE_MODEL - Discrete Choice Model for Product Selection", f)
        log("="*80, f)
        log(f"Data directory: {DATA_DIR}", f)

        # Load data
        data = load_data(f)

        # Build consideration sets
        choice_df = build_consideration_sets(data, f)

        if len(choice_df) == 0:
            log("\nERROR: No choice data created. Check if there are purchasing sessions.", f)
        else:
            # Add attributes
            choice_df = add_product_attributes(choice_df, data, f)

            # Summary
            choice_data_summary(choice_df, f)

            # Main model
            result = conditional_logit_mle(choice_df, f)

            # Alternative specs
            alternative_specs(choice_df, f)

        log(f"\n" + "="*80, f)
        log(f"Output saved to: {output_file}", f)
