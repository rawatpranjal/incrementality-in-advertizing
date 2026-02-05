#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'

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

def main():
    ap = argparse.ArgumentParser(description='Test rank effect with QUALITY fixed: logit(click) = offset + a*log(rank)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--offset', choices=['logit_quality','log_quality'], default='logit_quality', help='Form of QUALITY offset')
    ap.add_argument('--cov', choices=['HC1','none'], default='HC1')
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','QUALITY','RANKING'])
    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID'])

    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('clks').reset_index()
    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID']), on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df = df.dropna(subset=['QUALITY','RANKING'])
    # Keep valid values
    df = df[(df['imps']>0) & (df['RANKING']>0) & (df['QUALITY']>0)]

    # Build offset
    q = df['QUALITY'].values.astype(float)
    if args.offset == 'logit_quality':
        q_clip = np.clip(q, 1e-6, 1-1e-6)
        offset = np.log(q_clip/(1-q_clip))
    else:
        offset = safe_log_arr(q)

    # Design: intercept + log(rank)
    X = np.column_stack([np.ones(len(df)), safe_log_arr(df['RANKING'].values)])
    exog = pd.DataFrame(X, columns=['intercept','log_rank']).astype(float)
    endog = np.column_stack([df['clks'].values.astype(float), (df['imps'].values - df['clks'].values).astype(float)])

    glm = sm.GLM(endog, exog, family=sm.families.Binomial(), offset=offset)
    if args.cov == 'HC1':
        res = glm.fit(cov_type='HC1')
    else:
        res = glm.fit()

    # Fit offset-only (intercept only) for deviance comparison
    exog0 = pd.DataFrame({'intercept': np.ones(len(df))}, dtype=float)
    glm0 = sm.GLM(endog, exog0, family=sm.families.Binomial(), offset=offset)
    res0 = glm0.fit()
    dev = 2*(res.llf - res0.llf)

    # Report
    se = np.sqrt(np.diag(res.cov_params()))
    a = res.params['log_rank']
    a_se = se[list(res.params.index).index('log_rank')]
    z = a / a_se if a_se>0 else np.nan
    from scipy.stats import norm
    p = 2*(1-norm.cdf(abs(z))) if np.isfinite(z) else np.nan

    print(f"\n=== {args.round.upper()} rank effect given QUALITY (offset={args.offset}) ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print(f"intercept={res.params['intercept']:.4f} (se={se[0]:.4f})")
    print(f"a (log_rank)={a:.4f}  (se={a_se:.4f})  z={z:.2f}  p={p:.3e}")
    print(f"Deviance improvement over offset-only: {dev:.1f} (df=1)")

if __name__ == '__main__':
    main()

