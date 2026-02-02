import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import os
import warnings

warnings.filterwarnings('ignore')

RESULTS_DIR = 'analysis/position-effects/results'
DATA_DIR = 'analysis/position-effects/data'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    try:
        auctions = pd.read_parquet(f'{DATA_DIR}/auctions_results_all.parquet')
        impressions = pd.read_parquet(f'{DATA_DIR}/impressions_all.parquet')
        clicks = pd.read_parquet(f'{DATA_DIR}/clicks_all.parquet')
    except:
        base_dir = '/Users/pranjal/Code/topsort-incrementality/analysis/position-effects/data'
        auctions = pd.read_parquet(f'{base_dir}/auctions_results_all.parquet')
        impressions = pd.read_parquet(f'{base_dir}/impressions_all.parquet')
        clicks = pd.read_parquet(f'{base_dir}/clicks_all.parquet')

    auctions.columns = [c.lower() for c in auctions.columns]
    impressions.columns = [c.lower() for c in impressions.columns]
    clicks.columns = [c.lower() for c in clicks.columns]
    
    return auctions, impressions, clicks

def prep_data(auctions, impressions, clicks):
    print("Prepping Data...")
    if 'final_bid' in auctions.columns and 'bid' not in auctions.columns:
        auctions['bid'] = auctions['final_bid']
    if 'quality' in auctions.columns:
        auctions['quality_score'] = auctions['quality']
        
    auctions['score'] = auctions['quality_score'] * auctions['bid']
    
    # Enrichment
    auctions['one'] = 1
    stats = auctions.groupby('auction_id').agg(
        num_bidders=('one', 'sum')
    ).reset_index()
    auctions = auctions.merge(stats, on='auction_id', how='left')

    shown = auctions.merge(
        impressions[['auction_id', 'product_id']].drop_duplicates(),
        on=['auction_id', 'product_id'],
        how='inner'
    )
    
    clicks['is_clicked'] = 1
    clicks_dedup = clicks[['auction_id', 'product_id', 'is_clicked']].drop_duplicates()
    
    df = shown.merge(clicks_dedup, on=['auction_id', 'product_id'], how='left')
    df['is_clicked'] = df['is_clicked'].fillna(0)
    
    df['ranking'] = pd.to_numeric(df['ranking'], errors='coerce')
    df = df.dropna(subset=['ranking', 'score'])
    
    return df

# ==============================================================================
# TEST 1: Coefficient Stability
# ==============================================================================
def run_coefficient_stability(df):
    """
    Oster's Approach Logic:
    Compare Beta(Rank) across models:
    1. Naive: Click ~ Rank
    2. +Bid: Click ~ Rank + Bid
    3. +Quality: Click ~ Rank + Bid + Quality
    
    If Beta goes to 0, Selection Bias dominates.
    """
    print("\n=== TEST 1: COEFFICIENT STABILITY (SELECTION BIAS CHECK) ===")
    
    # Standardize vars for comparable coefficients
    df_std = df.copy()
    for col in ['ranking', 'bid', 'quality_score']:
        df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
        
    formula_1 = "is_clicked ~ ranking"
    formula_2 = "is_clicked ~ ranking + bid"
    formula_3 = "is_clicked ~ ranking + bid + quality_score"
    formula_4 = "is_clicked ~ ranking + bid + quality_score + num_bidders" # + Competition context
    
    models = [
        ("Naive (Rank Only)", formula_1),
        ("+ Bid Control", formula_2),
        ("+ Quality Control", formula_3),
        ("+ Competitors", formula_4)
    ]
    
    print(f"{'Model':<25} | {'Rank Coef (Std)':<15} | {'P-Value':<10} | {'R-Squared':<10}")
    print("-" * 70)
    
    baseline_beta = 0
    
    for name, form in models:
        mod = smf.ols(form, data=df_std).fit()
        beta = mod.params['ranking']
        pval = mod.pvalues['ranking']
        r2 = mod.rsquared
        
        if name == "Naive (Rank Only)":
            baseline_beta = beta
            
        print(f"{name:<25} | {beta:<15.4f} | {pval:<10.4f} | {r2:<10.4f}")
        
    final_beta = beta
    attenuation = 1 - (final_beta / baseline_beta)
    print("\nStability Metrics:")
    print(f"Attenuation of Position Effect: {attenuation*100:.1f}%")
    print("(100% means the effect completely disappeared after controls)")

# ==============================================================================
# TEST 2: PBM "Forced Fit"
# ==============================================================================
def run_pbm_forced_fit(df, max_rank=10):
    """
    Compare Unconstrained PBM vs "Steep decay" PBM.
    """
    print("\n=== TEST 2: PBM MODEL FIT COMPARISON (FORCED CURVE) ===")
    
    df_em = df[df['ranking'] <= max_rank].copy()
    
    # Pre-calculate Theta (naive Item CTR) as fixed for this check?
    # Or optimize Theta for each Gamma? Ideally optimize Theta.
    # To save time, we'll implement a simplified P(C) = Gamma_k * Theta_i estimator 
    # and check MSE/LogLikelihood.
    
    # Gamma Scenarios
    scenarios = {
        'Flat (Gamma=1.0)': lambda k: 1.0,
        'Shallow (1/sqrt(k))': lambda k: 1.0 / np.sqrt(k),
        'Steep (1/k)': lambda k: 1.0 / k,
        'Very Steep (1/k^2)': lambda k: 1.0 / (k**2),
        'Data Driven (From prev)': lambda k: 1.0 if k==1 else (0.47 if k==10 else 0.5) # Approximate logical step
    }
    
    # For "Data Driven", let's use the actual values from prev run or just allow it to be free (proxy via 1/k^0.something)
    
    results = []
    
    for name, func in scenarios.items():
        # Gammas
        gamma_vec = df_em['ranking'].apply(func)
        
        # Optimize Theta given Gamma
        # MLE for Theta_i: Sum(Clicks_i) / Sum(Gamma_i)
        # We can compute this directly.
        df_em['gamma_k'] = gamma_vec
        
        stats = df_em.groupby('product_id').agg(
            clicks=('is_clicked', 'sum'),
            exposure=('gamma_k', 'sum')
        )
        stats['theta_hat'] = stats['clicks'] / (stats['exposure'] + 1e-9)
        stats['theta_hat'] = stats['theta_hat'].clip(0, 1)
        
        # Predict Clicks
        df_em['theta_i'] = df_em['product_id'].map(stats['theta_hat'])
        df_em['pred_ctr'] = df_em['gamma_k'] * df_em['theta_i']
        
        # Calculate Log Likelihood (Bernoulli)
        # LL = y*log(p) + (1-y)*log(1-p)
        p = df_em['pred_ctr'].clip(1e-9, 1-1e-9)
        y = df_em['is_clicked']
        ll = (y * np.log(p) + (1-y) * np.log(1-p)).sum()
        
        results.append({
            'Scenario': name,
            'LogLikelihood': ll,
            'Avg_Pred_CTR': p.mean()
        })
        
    res_df = pd.DataFrame(results).sort_values('LogLikelihood', ascending=False)
    print(res_df)
    
    print("\nInterpretation:")
    print("Higher LogLikelihood (less negative) is better fit.")
    best = res_df.iloc[0]['Scenario']
    print(f"Best fitting shape: {best}")

# ==============================================================================
# TEST 3: Heterogeneity
# ==============================================================================
def run_heterogeneity(df):
    """
    Does Position Effect exist in specific slices?
    """
    print("\n=== TEST 3: HETEROGENEITY ANALYSIS ===")
    
    # Slices
    df['quality_quartile'] = pd.qcut(df['quality_score'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    df['comp_quartile'] = pd.qcut(df['num_bidders'], 3, labels=['Low Comp', 'Med Comp', 'High Comp'])
    
    segments = [
        ('All', df),
        ('Quality: Low', df[df['quality_quartile'] == 'Q1 (Low)']),
        ('Quality: High', df[df['quality_quartile'] == 'Q4 (High)']),
        ('Competition: Low', df[df['comp_quartile'] == 'Low Comp']),
        ('Competition: High', df[df['comp_quartile'] == 'High Comp']),
    ]
    
    print(f"{'Segment':<20} | {'Rank Coef':<12} | {'P-Value':<8} | {'N':<8}")
    print("-" * 60)
    
    for name, subset in segments:
        if len(subset) < 1000:
            continue
            
        # Run controlled regression: Click ~ Rank + Quality + Bid
        # Note: Rank is raw here (1, 2, 3)
        res = smf.ols("is_clicked ~ ranking + quality_score + bid", data=subset).fit()
        
        coef = res.params['ranking']
        pval = res.pvalues['ranking']
        
        print(f"{name:<20} | {coef:<12.5f} | {pval:<8.3f} | {len(subset):<8}")

def main():
    original_stdout = sys.stdout
    with open(f'{RESULTS_DIR}/07_sensitivity_analysis.txt', 'w') as f:
        sys.stdout = f
        
        auctions, impressions, clicks = load_data()
        df = prep_data(auctions, impressions, clicks)
        
        run_coefficient_stability(df)
        run_pbm_forced_fit(df)
        run_heterogeneity(df)
        
    sys.stdout = original_stdout
    print(f"Done. Results saved to {RESULTS_DIR}/07_sensitivity_analysis.txt")

if __name__ == "__main__":
    main()
