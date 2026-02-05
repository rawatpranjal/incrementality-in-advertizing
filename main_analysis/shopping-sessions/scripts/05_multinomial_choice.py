#!/usr/bin/env python3
"""
05_multinomial_choice.py - Multinomial Choice Model for Product Selection

Unit of Analysis: Product-session alternative
Dependent Variable: chosen (binary) - whether product was purchased in session
Choice Set: Top products impressed in the session + purchased product

Model: Conditional Logit (custom PyTorch implementation)
    V_sj = beta_clicked * clicked_sj
         + beta_n_imp * n_impressions_sj
         + beta_rank * ranking_sj
         + beta_price * log_price_j
         + week_FE_s

Identification: Within-session variation in product attributes determines choice probabilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PATHS
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "0_data_pull" / "data"
RESULTS_DIR = BASE_DIR / "results"
PAPER_DIR = Path(__file__).resolve().parents[3] / "paper" / "05-sessions"

# CONFIG
SAMPLE_FRACTION = 0.15  # 15% of purchasing sessions (~300 sessions)
TOP_N_PRODUCTS = 100    # Focus on top 100 products
RANDOM_SEED = 42


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
    files = ['sessions', 'session_events', 'auctions_results', 'catalog']
    for name in files:
        path = DATA_DIR / f'{name}.parquet'
        if path.exists():
            data[name] = pd.read_parquet(path)
            log(f"  {name}: {len(data[name]):,} rows", f)
        else:
            log(f"  {name}: NOT FOUND", f)
            data[name] = pd.DataFrame()

    return data


def build_choice_data(data, f):
    """
    Build choice data in long format.

    For each purchasing session:
    - Choice set = all impressed products (filtered to top N) + purchased product
    - Outcome = 1 if product was purchased, 0 otherwise
    """
    section("BUILDING CHOICE DATA", f)

    events = data['session_events']
    sessions = data['sessions']
    catalog = data['catalog']
    ar = data['auctions_results']

    # Get purchasing sessions and sample
    all_purch_sessions = sessions[sessions['purchased'] == 1]['session_id'].unique()
    log(f"Total purchasing sessions: {len(all_purch_sessions):,}", f)

    np.random.seed(RANDOM_SEED)
    n_sample = max(1, int(len(all_purch_sessions) * SAMPLE_FRACTION))
    purch_sessions = np.random.choice(all_purch_sessions, size=n_sample, replace=False)
    log(f"Sampled sessions ({SAMPLE_FRACTION*100:.0f}%): {len(purch_sessions):,}", f)

    # Get all events for purchasing sessions
    events_purch = events[events['session_id'].isin(purch_sessions)].copy()

    # Identify top products by impression frequency
    imp_counts = events_purch[events_purch['event_type'] == 'impression']['product_id'].value_counts()
    top_products = set(imp_counts.nlargest(TOP_N_PRODUCTS).index)
    log(f"Top {TOP_N_PRODUCTS} products selected", f)

    # Add all purchased products to ensure they're in choice sets
    purchased_products = set(events_purch[events_purch['event_type'] == 'purchase']['product_id'].dropna().unique())
    top_products = top_products | purchased_products
    log(f"Top products + purchased: {len(top_products):,}", f)

    # Get catalog prices
    cat_price = catalog[['product_id', 'price']].drop_duplicates().set_index('product_id')['price'].to_dict()

    # Get rankings from auctions_results (median per product)
    product_ranking = ar.groupby('product_id')['ranking'].median().to_dict()

    # Get week from session_start
    sessions['week'] = pd.to_datetime(sessions['session_start']).dt.isocalendar().week
    session_week = sessions.set_index('session_id')['week'].to_dict()

    # Build choice records
    records = []
    valid_sessions = 0

    for session_id in tqdm(purch_sessions, desc="Building choice data"):
        sess_events = events_purch[events_purch['session_id'] == session_id]

        # Get impressed products (filtered to top)
        impressed = set(sess_events[sess_events['event_type'] == 'impression']['product_id'].dropna()) & top_products

        # Get clicked products
        clicked = set(sess_events[sess_events['event_type'] == 'click']['product_id'].dropna())

        # Get purchased products that were ALSO impressed (ad-driven purchases only)
        purchased_all = set(sess_events[sess_events['event_type'] == 'purchase']['product_id'].dropna())
        purchased_impressed = list(purchased_all & impressed & top_products)

        # Skip if no impressed purchase in top products
        if len(purchased_impressed) == 0:
            continue

        # Select first purchased+impressed product as the chosen one
        chosen_product = purchased_impressed[0]

        # Choice set = impressed products only (all were shown ads)
        choice_set = impressed

        # Need at least 2 alternatives
        if len(choice_set) < 2:
            continue

        valid_sessions += 1

        # Count impressions per product in this session
        imp_counts_sess = sess_events[sess_events['event_type'] == 'impression'].groupby('product_id').size().to_dict()

        # Get rankings for this session's auctions
        sess_auctions = sess_events[sess_events['event_type'] == 'impression']['auction_id'].dropna().unique()
        if len(sess_auctions) > 0:
            sess_rankings = ar[
                (ar['auction_id'].isin(sess_auctions)) &
                (ar['product_id'].isin(impressed))
            ].groupby('product_id')['ranking'].min().to_dict()
        else:
            sess_rankings = {}

        # Get week
        week = session_week.get(session_id, 1)

        # Create record for each alternative
        for product_id in choice_set:
            is_clicked = 1 if product_id in clicked else 0
            is_purchased = 1 if product_id == chosen_product else 0
            n_imp = imp_counts_sess.get(product_id, 0)
            rank = sess_rankings.get(product_id, product_ranking.get(product_id, 20.0))
            price = cat_price.get(product_id, 30.0)  # Default price

            records.append({
                'session_id': session_id,
                'product_id': product_id,
                'clicked': is_clicked,
                'n_impressions': n_imp,
                'ranking': rank if not pd.isna(rank) else 20.0,
                'price': price if price > 0 else 30.0,
                'log_price': np.log(price) if price > 0 else np.log(30.0),
                'chosen': is_purchased,
                'week': week
            })

    choice_df = pd.DataFrame(records)
    log(f"\nValid sessions: {valid_sessions:,}", f)
    log(f"Total observations: {len(choice_df):,}", f)
    log(f"Unique products: {choice_df['product_id'].nunique():,}", f)

    # Choice set size distribution
    cs_sizes = choice_df.groupby('session_id').size()
    log(f"\nChoice set size:", f)
    log(f"  Mean: {cs_sizes.mean():.1f}", f)
    log(f"  Median: {cs_sizes.median():.0f}", f)
    log(f"  Min: {cs_sizes.min()}", f)
    log(f"  Max: {cs_sizes.max()}", f)

    return choice_df


def describe_choice_data(choice_df, f):
    """Summary statistics of choice data."""
    section("CHOICE DATA SUMMARY", f)

    log(f"\nTotal observations: {len(choice_df):,}", f)
    log(f"Sessions: {choice_df['session_id'].nunique():,}", f)
    log(f"Products: {choice_df['product_id'].nunique():,}", f)

    subsection("Exposure Distribution", f)
    log(f"  Clicked: {choice_df['clicked'].sum():,} ({choice_df['clicked'].mean()*100:.1f}%)", f)
    log(f"  Chosen: {choice_df['chosen'].sum():,} ({choice_df['chosen'].mean()*100:.1f}%)", f)

    subsection("Chosen by Click Status", f)
    cross = choice_df.groupby('clicked')['chosen'].agg(['sum', 'count', 'mean'])
    log(f"\n  {'Clicked':>10} {'Chosen':>10} {'Total':>10} {'Rate':>10}", f)
    log(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}", f)
    for clk, row in cross.iterrows():
        log(f"  {clk:>10} {row['sum']:>10.0f} {row['count']:>10.0f} {row['mean']*100:>10.2f}%", f)

    subsection("Variable Distributions", f)
    for col in ['n_impressions', 'ranking', 'log_price']:
        vals = choice_df[col].dropna()
        log(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}", f)


class ConditionalLogit(nn.Module):
    """Custom conditional logit model."""

    def __init__(self, n_features, n_weeks=0):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(n_features))
        if n_weeks > 0:
            self.week_fe = nn.Parameter(torch.zeros(n_weeks))
        else:
            self.week_fe = None
        self.n_weeks = n_weeks

    def forward(self, X, week_idx=None, mask=None):
        """
        X: (batch, max_alts, n_features)
        week_idx: (batch,) - week index for session FE
        mask: (batch, max_alts) - True where alternative exists
        """
        # Compute utility
        V = torch.matmul(X, self.beta)  # (batch, max_alts)

        # Add week FE if applicable
        if self.week_fe is not None and week_idx is not None:
            week_effect = self.week_fe[week_idx]  # (batch,)
            V = V + week_effect.unsqueeze(1)  # Broadcast to all alternatives

        # Apply mask (set utility to -inf for non-existent alternatives)
        if mask is not None:
            V = V.masked_fill(~mask, -1e9)

        return V


def estimate_model(choice_df, f, include_week_fe=True):
    """Estimate conditional logit model."""
    subsection(f"Estimating Conditional Logit (Week FE: {include_week_fe})", f)

    # Create session groups
    sessions = choice_df['session_id'].unique()
    n_sessions = len(sessions)
    session_to_idx = {s: i for i, s in enumerate(sessions)}

    log(f"Sessions: {n_sessions:,}", f)

    # Feature columns
    feature_cols = ['clicked', 'n_impressions', 'ranking', 'log_price']
    n_features = len(feature_cols)

    # Week encoding
    if include_week_fe:
        unique_weeks = sorted(choice_df['week'].unique())
        week_to_idx = {w: i for i, w in enumerate(unique_weeks)}
        n_weeks = len(unique_weeks)
        log(f"Weeks: {n_weeks}", f)
    else:
        n_weeks = 0

    # Build tensors - find max choice set size
    cs_sizes = choice_df.groupby('session_id').size()
    max_alts = cs_sizes.max()
    log(f"Max alternatives per session: {max_alts}", f)

    # Initialize tensors
    X = torch.zeros(n_sessions, max_alts, n_features, dtype=torch.float32)
    y = torch.zeros(n_sessions, dtype=torch.long)  # Index of chosen alternative
    mask = torch.zeros(n_sessions, max_alts, dtype=torch.bool)
    week_idx = torch.zeros(n_sessions, dtype=torch.long) if include_week_fe else None

    # Fill tensors
    for session_id in tqdm(sessions, desc="Building tensors"):
        s_idx = session_to_idx[session_id]
        sess_data = choice_df[choice_df['session_id'] == session_id]

        for a_idx, (_, row) in enumerate(sess_data.iterrows()):
            X[s_idx, a_idx, 0] = row['clicked']
            X[s_idx, a_idx, 1] = row['n_impressions']
            X[s_idx, a_idx, 2] = row['ranking']
            X[s_idx, a_idx, 3] = row['log_price']
            mask[s_idx, a_idx] = True

            if row['chosen'] == 1:
                y[s_idx] = a_idx

        if include_week_fe:
            week = sess_data['week'].iloc[0]
            week_idx[s_idx] = week_to_idx[week]

    log(f"X shape: {X.shape}", f)
    log(f"y shape: {y.shape}", f)

    # Initialize model
    model = ConditionalLogit(n_features, n_weeks)

    # Optimizer
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        V = model(X, week_idx, mask)
        log_probs = torch.log_softmax(V, dim=1)
        batch_idx = torch.arange(n_sessions)
        nll = -log_probs[batch_idx, y].mean()
        nll.backward()
        return nll

    log(f"\nFitting model...", f)
    for epoch in range(10):
        loss = optimizer.step(closure)
        log(f"  Epoch {epoch}: loss = {loss.item():.6f}", f)

    # Extract coefficients
    beta = model.beta.detach().numpy()

    # Compute standard errors via Hessian
    model.eval()
    with torch.no_grad():
        V = model(X, week_idx, mask)
        probs = torch.softmax(V, dim=1)  # (S, A)

        # Score and Hessian for beta
        # For conditional logit: score_s = X[s,j,:] - sum_k prob[s,k] * X[s,k,:]
        # Hessian = -sum_s sum_k prob[s,k] * (X[s,k] - X_bar[s]) * (X[s,k] - X_bar[s])'

        # Compute X_bar = expected features
        probs_expanded = probs.unsqueeze(-1)  # (S, A, 1)
        X_bar = (probs_expanded * X).sum(dim=1)  # (S, F)

        # Compute Hessian
        H = torch.zeros(n_features, n_features)
        for s in range(n_sessions):
            for a in range(mask[s].sum().item()):
                p = probs[s, a].item()
                x_diff = X[s, a, :] - X_bar[s]
                H -= p * torch.outer(x_diff, x_diff)

    # Standard errors
    try:
        H_inv = torch.linalg.inv(H.detach())
        se = torch.sqrt(-torch.diag(H_inv)).numpy()
    except:
        log("  WARNING: Hessian not invertible, using pseudo-inverse", f)
        H_inv = torch.linalg.pinv(H.detach())
        se = torch.sqrt(torch.abs(torch.diag(H_inv))).numpy()

    # Results
    subsection("Results", f)
    log(f"\n{'Variable':<20} {'Coef':>12} {'SE':>12} {'z-stat':>10} {'p-value':>10} {'Odds Ratio':>12}", f)
    log(f"{'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}", f)

    results = []
    for i, var in enumerate(feature_cols):
        coef = beta[i]
        std_err = se[i]
        z = coef / std_err if std_err > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        odds = np.exp(coef)
        stars = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))

        log(f"{var:<20} {coef:>12.4f} {std_err:>12.4f} {z:>10.2f} {p_val:>10.4f} {odds:>12.4f} {stars}", f)
        results.append({
            'variable': var,
            'coefficient': coef,
            'std_error': std_err,
            'z_stat': z,
            'p_value': p_val,
            'odds_ratio': odds
        })

    # Model fit
    V_final = model(X, week_idx, mask)
    log_probs_final = torch.log_softmax(V_final, dim=1)
    batch_idx = torch.arange(n_sessions)
    ll = log_probs_final[batch_idx, y].sum().item()

    # Null model (equal probabilities)
    n_alts_per_session = mask.sum(dim=1).float()
    ll_null = -torch.log(n_alts_per_session).sum().item()

    pseudo_r2 = 1 - (ll / ll_null)

    log(f"\nLog-likelihood: {ll:.4f}", f)
    log(f"Null log-likelihood: {ll_null:.4f}", f)
    log(f"Pseudo R-squared: {pseudo_r2:.4f}", f)
    log(f"N sessions: {n_sessions:,}", f)

    subsection("Interpretation", f)
    for r in results:
        var = r['variable']
        coef = r['coefficient']
        odds = r['odds_ratio']
        if var == 'clicked':
            log(f"  {var}: Clicking increases odds of purchase by {(odds-1)*100:.1f}%", f)
        elif var == 'n_impressions':
            log(f"  {var}: Each additional impression changes odds by {(odds-1)*100:.1f}%", f)
        elif var == 'ranking':
            log(f"  {var}: Each position worse changes odds by {(odds-1)*100:.1f}%", f)
        elif var == 'log_price':
            log(f"  {var}: 1% higher price changes odds by {coef:.4f}%", f)

    return {
        'results': results,
        'n_sessions': n_sessions,
        'n_products': choice_df['product_id'].nunique(),
        'include_week_fe': include_week_fe,
        'll': ll,
        'pseudo_r2': pseudo_r2
    }


def generate_latex_table(model_output, f, filename='multinomial_choice_results.tex'):
    """Generate LaTeX table for paper."""
    section("LATEX OUTPUT", f)

    if model_output is None:
        log("No results to export", f)
        return

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    latex_path = PAPER_DIR / filename

    results = model_output['results']
    n_sessions = model_output['n_sessions']
    n_products = model_output['n_products']
    include_week_fe = model_output['include_week_fe']

    latex = r"""\begin{table}[H]
\centering
\caption{Product-Level Choice Model: Conditional Logit}
\label{tab:multinomial_choice}
\begin{tabular}{lcccc}
\toprule
 & Coefficient & Std. Error & z-stat & Odds Ratio \\
\midrule
"""

    var_labels = {
        'clicked': 'Clicked',
        'n_impressions': 'N Impressions',
        'ranking': 'Ranking',
        'log_price': 'Log(Price)'
    }

    for r in results:
        var = r['variable']
        var_fmt = var_labels.get(var, var)
        coef = r['coefficient']
        se = r['std_error']
        z = r['z_stat']
        odds = r['odds_ratio']
        p = r['p_value']

        stars = '^{***}' if p < 0.01 else ('^{**}' if p < 0.05 else ('^{*}' if p < 0.1 else ''))
        latex += f"{var_fmt} & {coef:.4f}{stars} & ({se:.4f}) & {z:.2f} & {odds:.3f} \\\\\n"

    latex += r"""\midrule
"""
    week_fe_str = 'Yes' if include_week_fe else 'No'
    latex += f"\\multicolumn{{5}}{{l}}{{\\footnotesize Week FE: {week_fe_str}. Sessions: {n_sessions:,}. Products: {n_products:,}.}} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(latex_path, 'w') as lf:
        lf.write(latex)

    log(f"\nLaTeX table saved to: {latex_path}", f)
    log("\nLaTeX content:", f)
    log(latex, f)


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / '05_multinomial_choice.txt'

    with open(output_file, 'w') as f:
        log("="*80, f)
        log("05_MULTINOMIAL_CHOICE - Multinomial Choice Model", f)
        log("="*80, f)
        log(f"Data directory: {DATA_DIR}", f)
        log(f"Top products: {TOP_N_PRODUCTS}", f)
        log(f"Random seed: {RANDOM_SEED}", f)

        # Load data
        data = load_data(f)

        # Build choice data
        choice_df = build_choice_data(data, f)

        if len(choice_df) == 0:
            log("\nERROR: No choice data created.", f)
        else:
            # Describe data
            describe_choice_data(choice_df, f)

            # Main model: with week FE
            section("MAIN MODEL: WITH WEEK FE", f)
            model_output_fe = estimate_model(choice_df, f, include_week_fe=True)
            generate_latex_table(model_output_fe, f)

            # Robustness: without week FE
            section("ROBUSTNESS: WITHOUT WEEK FE", f)
            model_output_no_fe = estimate_model(choice_df, f, include_week_fe=False)

            # Comparison
            section("MODEL COMPARISON", f)
            log(f"\n{'Variable':<20} {'With Week FE':>15} {'Without Week FE':>15}", f)
            log(f"{'-'*20} {'-'*15} {'-'*15}", f)
            for r_fe, r_no_fe in zip(model_output_fe['results'], model_output_no_fe['results']):
                var = r_fe['variable']
                coef_fe = r_fe['coefficient']
                coef_no_fe = r_no_fe['coefficient']
                log(f"{var:<20} {coef_fe:>15.4f} {coef_no_fe:>15.4f}", f)

        log(f"\n" + "="*80, f)
        log(f"Output saved to: {output_file}", f)
