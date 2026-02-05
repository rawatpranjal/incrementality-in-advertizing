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
            'auctions_users': DATA_DIR / 'round1/auctions_users_all.parquet',
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
        }
    raise ValueError(round_name)

def auc_from_aggregated(scores: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    order = np.argsort(scores)
    s = scores[order]
    p = clks[order].astype(float)
    n = (imps[order] - clks[order]).astype(float)
    uniq, idx_start = np.unique(s, return_index=True)
    p_sum = []
    n_sum = []
    for i in range(len(uniq)):
        start = idx_start[i]
        end = idx_start[i+1] if i+1 < len(idx_start) else len(s)
        p_sum.append(p[start:end].sum())
        n_sum.append(n[start:end].sum())
    p_sum = np.array(p_sum)
    n_sum = np.array(n_sum)
    cum_n_below = np.cumsum(np.concatenate([[0.0], n_sum[:-1]]))
    U = np.sum(p_sum * (cum_n_below + 0.5 * n_sum))
    P = float(p_sum.sum())
    N = float(n_sum.sum())
    if P <= 0 or N <= 0:
        return np.nan
    return U / (P * N)

def safe_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x.astype(float), eps))

def main():
    ap = argparse.ArgumentParser(description='GLM Binomial with QUALITY as offset; features: log bid/price/cvr/pacing/rank + placement FEs')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--cluster', choices=['none','auction','placement'], default='none')
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar_cols = ['AUCTION_ID','PRODUCT_ID','QUALITY','FINAL_BID','PRICE','CONVERSION_RATE','RANKING','PACING']
    ar = pd.read_parquet(paths['auctions_results'], columns=ar_cols)
    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()

    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('clks').reset_index()
    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID']), on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df = df.merge(au, on='AUCTION_ID', how='left')
    df = df.dropna(subset=['QUALITY','FINAL_BID','PRICE','RANKING','PACING','PLACEMENT'])

    # Offset from QUALITY
    q = np.clip(df['QUALITY'].values.astype(float), 1e-6, 1-1e-6)
    offset = np.log(q / (1 - q))

    # Features
    df['log_b'] = safe_log(df['FINAL_BID'])
    df['log_p'] = safe_log(df['PRICE'])
    df['log_cvr'] = safe_log(df['CONVERSION_RATE']) if 'CONVERSION_RATE' in df.columns else 0.0
    df['log_pac'] = safe_log(df['PACING'])
    df['log_rank'] = safe_log(df['RANKING'])

    # Placement FE (drop first)
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    plc = plc.iloc[:, 1:] if plc.shape[1] > 0 else plc
    exog_cols = ['log_b','log_p','log_cvr','log_pac','log_rank']
    exog = pd.concat([pd.Series(1.0, index=df.index, name='intercept'), df[exog_cols], plc], axis=1).astype(float)

    # Endog as successes/failures for Binomial
    endog = np.column_stack([df['clks'].values.astype(float), (df['imps'] - df['clks']).values.astype(float)])

    # GLM fit with robust SEs
    glm = sm.GLM(endog, exog, family=sm.families.Binomial(), offset=offset)
    if args.cluster == 'auction':
        res = glm.fit(cov_type='cluster', cov_kwds={'groups': df['AUCTION_ID'].values})
    elif args.cluster == 'placement':
        res = glm.fit(cov_type='cluster', cov_kwds={'groups': df['PLACEMENT'].values})
    else:
        res = glm.fit(cov_type='HC1')

    # AUCs: quality only vs full linear score
    lin_off = offset
    lin_full = offset + np.asarray(exog) @ res.params
    auc_q = auc_from_aggregated(lin_off, df['imps'].values, df['clks'].values)
    auc_full = auc_from_aggregated(lin_full, df['imps'].values, df['clks'].values)

    print(f"\n=== {args.round.upper()} GLM(click) with QUALITY offset (statsmodels) ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print(f"Covariance: {res.cov_type}")
    coefs = pd.DataFrame({'term': exog.columns, 'beta': res.params, 'se': np.sqrt(np.diag(res.cov_params()))})
    for _, r in coefs.iterrows():
        print(f"  {r['term']:>12}: {r['beta']: .4f}  (se={r['se']:.4f})")
    print(f"AUC(offset only)={auc_q:.4f}  |  AUC(offset + features)={auc_full:.4f}")
    print(f"Deviance={res.deviance:.1f}  AIC={res.aic:.1f}")

if __name__ == '__main__':
    main()
