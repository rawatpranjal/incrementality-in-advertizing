import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
import sys
import os
import warnings
from scipy.optimize import minimize_scalar

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
        # Absolute path fallback
        base_dir = '/Users/pranjal/Code/topsort-incrementality/analysis/position-effects/data'
        auctions = pd.read_parquet(f'{base_dir}/auctions_results_all.parquet')
        impressions = pd.read_parquet(f'{base_dir}/impressions_all.parquet')
        clicks = pd.read_parquet(f'{base_dir}/clicks_all.parquet')

    # Lowercase cols
    auctions.columns = [c.lower() for c in auctions.columns]
    impressions.columns = [c.lower() for c in impressions.columns]
    clicks.columns = [c.lower() for c in clicks.columns]
    
    return auctions, impressions, clicks

def prep_data(auctions, impressions, clicks):
    print("Prepping Analysis DataFrame...")
    
    # Standardize Score inputs
    if 'final_bid' in auctions.columns and 'bid' not in auctions.columns:
        auctions['bid'] = auctions['final_bid']
    if 'quality' in auctions.columns:
        auctions['quality_score'] = auctions['quality']
        
    auctions['score'] = auctions['quality_score'] * auctions['bid']
    
    # Enrich Auctions with "Competitor Context"
    # Num Bidders per Auction
    auctions['one'] = 1
    auction_stats = auctions.groupby('auction_id').agg(
        num_bidders=('one', 'sum'),
        mean_score=('score', 'mean'),
        max_score=('score', 'max')
    ).reset_index()
    
    auctions = auctions.merge(auction_stats, on='auction_id', how='left')
    
    # Filter to WINNERS & SHOWN
    shown = auctions.merge(
        impressions[['auction_id', 'product_id']].drop_duplicates(),
        on=['auction_id', 'product_id'],
        how='inner'
    )
    
    # Join Clicks
    clicks['is_clicked'] = 1
    clicks_dedup = clicks[['auction_id', 'product_id', 'is_clicked']].drop_duplicates()
    
    df = shown.merge(clicks_dedup, on=['auction_id', 'product_id'], how='left')
    df['is_clicked'] = df['is_clicked'].fillna(0)
    
    df['ranking'] = pd.to_numeric(df['ranking'], errors='coerce')
    df = df.dropna(subset=['ranking', 'score'])
    
    return df

# ==============================================================================
# METHOD 1: PBM via EM Algorithm
# ==============================================================================
def run_pbm_em(df, max_rank=10, max_iter=20, tol=1e-4):
    """
    Position-Based Model: P(Click) = P(Examine|Rank) * P(Relevance|Item)
    gamma_k = P(Examine|Rank k)
    theta_i = P(Relevance|Item i)
    Constraint: gamma_1 = 1
    """
    print("\n=== METHOD 1: PBM VIA EM ALGORITHM ===")
    
    # Filter to top ranks
    df_em = df[df['ranking'] <= max_rank].copy()
    items = df_em['product_id'].unique()
    ranks = sorted(df_em['ranking'].unique())
    
    print(f"Items: {len(items)}, Ranks: {len(ranks)}, Obs: {len(df_em)}")
    
    # Initialization
    # theta_i = global CTR of item i (naive)
    item_stats = df_em.groupby('product_id')['is_clicked'].mean()
    theta = item_stats.to_dict()
    
    # gamma_k = naive CTR at rank k / global CTR estimate usually, but init to 1, 0.9, ...
    gamma = {k: 1.0/k for k in ranks} # Decay initialization
    gamma[1] = 1.0 # Identification constraint
    
    # Add indices for fast mapping
    df_em['theta_init'] = df_em['product_id'].map(theta)
    
    # EM Loop
    for it in range(max_iter):
        # E-Step: Calculate P(Examine=1 | Click, Rank, Item)
        # If Click=1 => E=1 (Assumed)
        # If Click=0 => P(E=1 | C=0) = P(C=0|E=1)P(E=1) / P(C=0) 
        #           = (1-theta)*gamma / (1 - gamma*theta)
        
        # Vectorized ops
        g = df_em['ranking'].map(gamma)
        t = df_em['product_id'].map(theta)
        c = df_em['is_clicked']
        
        # Posterior Prob of Examination
        p_examine = np.where(c == 1, 1.0, (g * (1-t)) / (1 - g*t + 1e-9))
        
        # M-Step: Update gamma (avg examination by rank) and theta (avg relevance by item)
        df_em['p_examine'] = p_examine
        
        # Update Gamma: Average p_examine for each rank
        # Constraint: gamma_1 fixed at 1 usually? Or normalize?
        # Usually gamma_k = Sum(p_examine_k) / N_k
        new_gamma_series = df_em.groupby('ranking')['p_examine'].mean()
        # Normalize so Rank 1 = 1.0
        scale = new_gamma_series[1] if 1 in new_gamma_series else 1.0
        new_gamma = (new_gamma_series / scale).to_dict()
        new_gamma[1] = 1.0 
        
        # Update Theta: Sum(Clicks) / Sum(P_examine) for each item
        # MLE for binomial: k/n where n is expected examinations
        item_sums = df_em.groupby('product_id').agg(
            clicks=('is_clicked', 'sum'),
            exp_exam=('p_examine', 'sum')
        )
        new_theta_series = item_sums['clicks'] / (item_sums['exp_exam'] + 1e-9)
        new_theta_series = new_theta_series.clip(0.001, 0.999) # prevent 0/1
        new_theta = new_theta_series.to_dict()
        
        # Check convergence (on gamma)
        diff = sum(abs(new_gamma.get(k,0) - gamma.get(k,0)) for k in ranks)
        if diff < tol:
            print(f"Converged at iter {it}")
            gamma = new_gamma
            theta = new_theta
            break
            
        gamma = new_gamma
        theta = new_theta
        if it % 5 == 0:
            print(f"Iter {it}: Diff {diff:.4f}")
            
    # Output Gammas
    print("\nEstimated Examination Probabilities (Gamma):")
    print("Rank | Gamma (PBM)")
    print("-----|------------")
    for k in sorted(gamma.keys()):
        print(f"{k:4} | {gamma[k]:.4f}")
        
    return gamma

# ==============================================================================
# METHOD 2: IV (Competitor Count)
# ==============================================================================
def run_iv_model(df):
    """
    Instrument Rank with Num_Bidders.
    IV Logic: More bidders -> Lower Rank (Higher Rank Number). Exogenous to item quality?
    Maybe. High quality items might attract competition? Assuming Auction-level #bidders is exogenous to Item i.
    Control for quality_score.
    """
    print("\n=== METHOD 2: IV REGRESSION (COMPETITOR COUNT) ===")
    
    # Data
    iv_df = df[['is_clicked', 'ranking', 'num_bidders', 'quality_score', 'bid']].copy().dropna()
    
    # 2SLS Scheme
    # Stage 1: Rank ~ Num_Bidders + Quality + Bid
    # Stage 2: Click ~ Rank_Hat + Quality + Bid
    
    # Check Instrument Strength
    stage1 = smf.ols("ranking ~ num_bidders + quality_score + bid", data=iv_df).fit()
    print("Stage 1 (Rank ~ Instrument):")
    print(stage1.summary().tables[1])
    f_stat = stage1.fvalue
    print(f"Stage 1 F-Stat: {f_stat:.2f}")
    
    if f_stat < 10:
        print("Weak Instrument! Skipping 2SLS.")
        return
        
    iv_df['rank_hat'] = stage1.predict()
    
    # Stage 2
    stage2 = smf.ols("is_clicked ~ rank_hat + quality_score + bid", data=iv_df).fit()
    print("\nStage 2 (Click ~ Rank_Hat):")
    print(stage2.summary().tables[1])
    
    coef = stage2.params['rank_hat']
    print(f"\nIV Estimate of Rank Position Effect (Linear): {coef:.5f} (p={stage2.pvalues['rank_hat']:.3f})")
    print("Interpretation: Change in CTR for 1 unit increase in Rank (worse position).")

# ==============================================================================
# METHOD 3: Within-Auction FE
# ==============================================================================
def run_within_auction_fe(df):
    """
    Click ~ C(Rank) + Auction_FE + Quality
    Controls for "Auction Quality" (user intent).
    Does not perfectly control for "Item Quality" selection within auction, 
    but with explicit 'quality_score' control it's better.
    """
    print("\n=== METHOD 3: WITHIN-AUCTION FIXED EFFECTS ===")
    
    # We use de-meaning for Auction FE (High Cardinality)
    cols = ['is_clicked', 'quality_score', 'bid']
    # Filter to auctions with variance in clicks or just generally valid
    fe_df = df[['auction_id', 'ranking'] + cols].copy().dropna()
    
    # Only auctions with >1 item
    cts = fe_df['auction_id'].value_counts()
    fe_df = fe_df[fe_df['auction_id'].isin(cts[cts > 1].index)]
    
    # Demean by Auction
    means = fe_df.groupby('auction_id')[cols].transform('mean')
    for c in cols:
        fe_df[f'{c}_demeaned'] = fe_df[c] - means[c]
        
    # Rank Dummies
    # We can't demean categorical Rank easily without expanding.
    # Expand Rank 1..10
    fe_df = fe_df[fe_df['ranking'] <= 10]
    rank_dummies = pd.get_dummies(fe_df['ranking'], prefix='rank', drop_first=True).astype(float)
    fe_df = pd.concat([fe_df, rank_dummies], axis=1)
    
    dummy_cols = rank_dummies.columns.tolist()
    dummy_means = fe_df.groupby('auction_id')[dummy_cols].transform('mean')
    
    X = fe_df[dummy_cols] - dummy_means
    X['quality_demeaned'] = fe_df['quality_score_demeaned']
    y = fe_df['is_clicked_demeaned']
    
    mod = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HC1')
    print(mod.summary())

# ==============================================================================
# METHOD 4: Multi-Click Analysis
# ==============================================================================
def run_multi_click_analysis(df):
    """
    Analyze relative rank of clicked items in multi-click sessions.
    If Position bias is strong, users click Rank 1 then Rank 2.
    If Selection bias is strong, users search for Quality regardless of rank.
    """
    print("\n=== METHOD 4: MULTI-CLICK ANALYSIS ===")
    
    # Find multi-click auctions
    # Clicks here are per product-auction.
    clicks_only = df[df['is_clicked'] == 1]
    click_counts = clicks_only['auction_id'].value_counts()
    multi_click_auctions = click_counts[click_counts >= 2].index
    
    print(f"Multi-click auctions: {len(multi_click_auctions)}")
    
    if len(multi_click_auctions) == 0:
        return

    mc_df = clicks_only[clicks_only['auction_id'].isin(multi_click_auctions)].copy()
    
    # In a multi-click set, what is the avg rank distance?
    # Or: P(Click Rank k | Click Rank 1 also happened)
    
    # Let's count pairs of clicks (k1, k2) where k1 < k2
    pair_counts = {}
    
    # Group by auction, get list of clicked ranks
    clicked_ranks = mc_df.groupby('auction_id')['ranking'].apply(list)
    
    total_pairs = 0
    rank_sum_diff = 0
    
    for ranks in clicked_ranks:
        ranks = sorted(ranks)
        for i in range(len(ranks)):
            for j in range(i+1, len(ranks)):
                r1, r2 = ranks[i], ranks[j]
                if r2 > r1:
                    pair_counts[(r1, r2)] = pair_counts.get((r1, r2), 0) + 1
                    total_pairs += 1
                    rank_sum_diff += (r2 - r1)
                    
    print("Top Co-Click Pairs (Rank A, Rank B):")
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for p, c in sorted_pairs:
        print(f"  Ranks {p}: {c} times")
        
    print(f"Avg Rank Distance in Co-Clicks: {rank_sum_diff / total_pairs:.2f}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    original_stdout = sys.stdout
    with open(f'{RESULTS_DIR}/06_comprehensive_analysis.txt', 'w') as f:
        sys.stdout = f
        
        auctions, impressions, clicks = load_data()
        df = prep_data(auctions, impressions, clicks)
        
        run_pbm_em(df)
        run_iv_model(df)
        run_within_auction_fe(df)
        run_multi_click_analysis(df)
        
    sys.stdout = original_stdout
    print(f"Done. Results saved to {RESULTS_DIR}/06_comprehensive_analysis.txt")

if __name__ == "__main__":
    main()
