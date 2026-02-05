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
    except Exception as e:
        print(f"Error loading local data: {e}")
        # Fallback for absolute path
        base_dir = '/Users/pranjal/Code/topsort-incrementality/analysis/position-effects/data'
        auctions = pd.read_parquet(f'{base_dir}/auctions_results_all.parquet')
        impressions = pd.read_parquet(f'{base_dir}/impressions_all.parquet')
        clicks = pd.read_parquet(f'{base_dir}/clicks_all.parquet')

    # Lowercase cols
    auctions.columns = [c.lower() for c in auctions.columns]
    impressions.columns = [c.lower() for c in impressions.columns]
    clicks.columns = [c.lower() for c in clicks.columns]
    
    return auctions, impressions, clicks

def prep_pairwise_data(auctions, impressions, clicks):
    """
    Constructs the dataset of Adjacent Impressed Pairs.
    Filters to auctions with at least 2 impressed ads.
    Computes Delta(Click) and Delta(Score).
    """
    print("Prepping pairwise data...")
    
    # Standardize Score
    # Check for 'final_bid' vs 'bid'
    if 'final_bid' in auctions.columns and 'bid' not in auctions.columns:
        auctions['bid'] = auctions['final_bid']
    
    # Calculate Score
    if 'quality' in auctions.columns:
        auctions['quality_score'] = auctions['quality']
    
    auctions['score'] = auctions['quality_score'] * auctions['bid']
    auctions = auctions.dropna(subset=['score', 'ranking'])
    
    # Filter to WINNERS
    # Actually, we only care about IMPRESSED ads for Position Effect
    # Join with Impressions
    shown = auctions.merge(
        impressions[['auction_id', 'product_id']].drop_duplicates(),
        on=['auction_id', 'product_id'],
        how='inner'
    )
    
    # Helper: display position inferring (if needed) - but assuming 'ranking' is display rank for now
    # EDA 4 showed high corr (0.85). We'll use 'ranking' as the treatment variable.
    
    # Join Clicks
    clicks['is_clicked'] = 1
    clicks_dedup = clicks[['auction_id', 'product_id', 'is_clicked']].drop_duplicates()
    
    df = shown.merge(clicks_dedup, on=['auction_id', 'product_id'], how='left')
    df['is_clicked'] = df['is_clicked'].fillna(0)
    
    # Filter auctions with >= 2 shown ads
    cts = df['auction_id'].value_counts()
    valid_auctions = cts[cts >= 2].index
    df = df[df['auction_id'].isin(valid_auctions)]
    
    # Sort by Auction, Rank
    df = df.sort_values(['auction_id', 'ranking'])
    
    # SHIFT to create pairs
    # grouping by auction_id is necessary
    # shift(-1) gets the NEXT rank
    
    df['next_ranking'] = df.groupby('auction_id')['ranking'].shift(-1)
    df['next_score'] = df.groupby('auction_id')['score'].shift(-1)
    df['next_clicked'] = df.groupby('auction_id')['is_clicked'].shift(-1)
    df['next_bid'] = df.groupby('auction_id')['bid'].shift(-1)
    df['next_quality'] = df.groupby('auction_id')['quality_score'].shift(-1)
    
    # Filter where valid next rank exists
    pairs = df.dropna(subset=['next_ranking']).copy()
    
    # Calculate Deltas (Current - Next)
    # Note: Current Rank < Next Rank (e.g. 1 vs 2).
    # Since Score_1 > Score_2 usually, Delta Score > 0.
    
    pairs['delta_score'] = pairs['score'] - pairs['next_score']
    pairs['delta_click'] = pairs['is_clicked'] - pairs['next_clicked']
    pairs['delta_bid'] = pairs['bid'] - pairs['next_bid']
    pairs['delta_quality'] = pairs['quality_score'] - pairs['next_quality']
    
    # Identifier for the Rank Transition (e.g. "1_2")
    pairs['pair_id'] = pairs['ranking'].astype(int).astype(str) + "_" + pairs['next_ranking'].astype(int).astype(str)
    
    print(f"Constructed {len(pairs)} adjacent pairs.")
    print("Top Transitions:")
    print(pairs['pair_id'].value_counts().head(5))
    
    return pairs

def run_rdd(pairs):
    """
    Runs Pairwise RDD (Limit Regression).
    Delta_Click ~ Intercept + Delta_Score
    Intercept is the LATE of Position (Current vs Next) at Tie.
    """
    print("\n=== RDD: PAIRWISE RANK MARGINS ===")
    
    # Focus on top transitions (dense data)
    transitions = ['1_2', '2_3', '3_4', '4_5', '5_6']
    
    results = []
    
    for trans in transitions:
        subset = pairs[pairs['pair_id'] == trans].copy()
        if len(subset) < 100:
            continue
            
        # Bandwidths: Full, P50, P25
        # Note: Delta Score is naturally positive (sorted).
        # We look at behavior as Delta Score -> 0
        
        subset = subset.sort_values('delta_score')
        
        # Define bandwidths
        bw_full = subset['delta_score'].max()
        bw_med = subset['delta_score'].median()
        bw_small = subset['delta_score'].quantile(0.25)
        bw_tiny = subset['delta_score'].quantile(0.10)
        
        bws = {
            'All': bw_full,
            'Median': bw_med,
            'Small (P25)': bw_small,
            'Tiny (P10)': bw_tiny
        }
        
        print(f"\n--- Transition {trans} (N={len(subset)}) ---")
        
        for name, bw in bws.items():
            # Local linear regression near 0
            # Data: delta_score strictly >= 0 (mostly).
            # Sometimes negative if 'ranking' != 'score' order (noise).
            # We regress on delta_score. The intercept is Value at 0.
            
            data_reg = subset[subset['delta_score'] <= bw]
            if len(data_reg) < 50:
                continue
                
            # Model: delta_click ~ delta_score
            mod = smf.ols("delta_click ~ delta_score", data=data_reg).fit(cov_type='HC1')
            
            # Intercept is the 'Position Effect' (Click rate gain from being Rank k vs k+1, holding score equal)
            intercept = mod.params['Intercept']
            pval = mod.pvalues['Intercept']
            n_obs = int(mod.nobs)
            
            # Store
            results.append({
                'Transition': trans,
                'Bandwidth_Name': name,
                'Bandwidth_Val': bw,
                'N': n_obs,
                'Intercept (Pos Effect)': intercept,
                'P-Value': pval,
                'Conf_Low': mod.conf_int().iloc[0,0],
                'Conf_High': mod.conf_int().iloc[0,1]
            })
            
            if name in ['All', 'Tiny (P10)']:
                print(f"  [{name}] BW={bw:.4f}, N={n_obs}: Effect={intercept:.4f} (p={pval:.3f})")

    # Summary Table
    res_df = pd.DataFrame(results)
    return res_df

def main():
    original_stdout = sys.stdout
    with open(f'{RESULTS_DIR}/05_rdd_results.txt', 'w') as f:
        sys.stdout = f
        
        auctions, impressions, clicks = load_data()
        pairs = prep_pairwise_data(auctions, impressions, clicks)
        
        res_df = run_rdd(pairs)
        
        print("\n\n=== FINAL SUMMARY TABLE ===")
        # Pivot for readability
        pivot = res_df[res_df['Bandwidth_Name'].isin(['All', 'Tiny (P10)'])]
        print(pivot[['Transition', 'Bandwidth_Name', 'Intercept (Pos Effect)', 'P-Value', 'N']].to_string(index=False))
        
        print("\nInterpretation:")
        print("Intercept > 0 implies estimates imply Higher Rank gets more clicks, even at zero score gap.")
        
    sys.stdout = original_stdout
    print(f"Done. Results saved to {RESULTS_DIR}/05_rdd_results.txt")

if __name__ == "__main__":
    main()
