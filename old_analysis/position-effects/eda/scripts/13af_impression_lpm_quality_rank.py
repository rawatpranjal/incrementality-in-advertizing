#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

def get_paths(round_name: str) -> dict:
    if round_name == 'round1':
        return {
            'auctions_results': DATA_DIR / 'round1/auctions_results_all.parquet',
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
        }
    raise ValueError(round_name)

def safe_log_arr(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(float)
    return np.log(np.clip(x, eps, None))

def fit_ols(y: np.ndarray, X: pd.DataFrame, cov: str = 'HC1'):
    Xc = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, Xc)
    res = model.fit(cov_type=cov)
    return res

def fwl_residualize(x: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Zc = sm.add_constant(Z, has_constant='add')
    beta = np.linalg.pinv(Zc.T @ Zc) @ (Zc.T @ x)
    return x - Zc @ beta

def main():
    ap = argparse.ArgumentParser(description='Impression-level LPM: click ~ quality + rank variants (rank, log(rank), rank^2) + FWL tests')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    # Base: impressions (each row is a served bid)
    imp = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    clk = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','RANKING','QUALITY'])

    # Click flag per impression via exact key join
    clk['clicked'] = 1
    clicks_key = clk[['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','clicked']].drop_duplicates()
    df = imp.merge(clicks_key, on=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'], how='left')
    df['clicked'] = df['clicked'].fillna(0).astype(int)

    # Join bid features for those impressions
    # Prefer strict join on AUCTION_ID, PRODUCT_ID, VENDOR_ID
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']),
                  on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.dropna(subset=['RANKING','QUALITY'])
    df = df[(df['RANKING'] > 0) & (df['QUALITY'] > 0)]

    # Features
    df['rank'] = df['RANKING'].astype(float)
    df['log_rank'] = safe_log_arr(df['rank'].values)
    df['rank_sq'] = df['rank'] ** 2
    df['quality'] = df['QUALITY'].astype(float)
    y = df['clicked'].values.astype(float)

    # Report base stats
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13af_impression_lpm_quality_rank_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s) + "\n"); fh.flush(); print(s)
        wprint(f"Impression-level LPM on {args.round}")
        wprint(f"Rows (impressions) = {len(df):,}")
        wprint(f"Unique auctions = {df['AUCTION_ID'].nunique():,}")
        wprint(f"Unique vendors = {df['VENDOR_ID'].nunique():,}")
        wprint(f"CTR = {df['clicked'].mean()*100:.3f}%")

        # Model 1: y ~ quality + rank
        X1 = df[['quality','rank']]
        r1 = fit_ols(y, X1)
        wprint("\nModel 1: y ~ quality + rank  (HC1 SE)")
        wprint(r1.summary())

        # Model 2: y ~ quality + log_rank
        X2 = df[['quality','log_rank']]
        r2 = fit_ols(y, X2)
        wprint("\nModel 2: y ~ quality + log_rank  (HC1 SE)")
        wprint(r2.summary())

        # Model 3: y ~ quality + rank + rank_sq
        X3 = df[['quality','rank','rank_sq']]
        r3 = fit_ols(y, X3)
        wprint("\nModel 3: y ~ quality + rank + rank^2  (HC1 SE)")
        wprint(r3.summary())

        # FWL: residualize rank on quality, regress y on residual
        wprint("\nFWL tests (residualizing on quality):")
        # rank residual
        r_rank = fwl_residualize(df['rank'].values, df[['quality']].values)
        r_logrank = fwl_residualize(df['log_rank'].values, df[['quality']].values)
        r_rank_sq = fwl_residualize(df['rank_sq'].values, df[['quality']].values)

        r_rank_res = fit_ols(y, pd.DataFrame({'rank_resid': r_rank}))
        wprint("FWL: y ~ (rank | quality)")
        wprint(r_rank_res.summary())

        r_logrank_res = fit_ols(y, pd.DataFrame({'log_rank_resid': r_logrank}))
        wprint("FWL: y ~ (log_rank | quality)")
        wprint(r_logrank_res.summary())

        r_ranksq_res = fit_ols(y, pd.DataFrame({'rank_sq_resid': r_rank_sq}))
        wprint("FWL: y ~ (rank^2 | quality)")
        wprint(r_ranksq_res.summary())

        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()

