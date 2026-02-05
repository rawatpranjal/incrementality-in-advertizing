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

def auc_from_aggregated(scores: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    order = np.argsort(scores)
    s = scores[order]
    p = clks[order].astype(float)
    n = (imps[order] - clks[order]).astype(float)
    uniq, idx_start = np.unique(s, return_index=True)
    p_sum, n_sum = [], []
    for i in range(len(uniq)):
        start = idx_start[i]
        end = idx_start[i+1] if i+1 < len(idx_start) else len(s)
        p_sum.append(p[start:end].sum())
        n_sum.append(n[start:end].sum())
    p_sum = np.array(p_sum)
    n_sum = np.array(n_sum)
    P = float(p_sum.sum())
    N = float(n_sum.sum())
    if P <= 0 or N <= 0:
        return np.nan
    cum_n_below = np.cumsum(np.concatenate([[0.0], n_sum[:-1]]))
    U = np.sum(p_sum * (cum_n_below + 0.5 * n_sum))
    return U / (P * N)

def main():
    ap = argparse.ArgumentParser(description='Residualize log(rank) on log(quality); test residual in GLM with QUALITY offset')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--per_placement', action='store_true', help='residualize within placement if available')
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar_cols = ['AUCTION_ID','PRODUCT_ID','QUALITY','RANKING']
    ar = pd.read_parquet(paths['auctions_results'], columns=ar_cols)
    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID'])

    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('clks').reset_index()
    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID']), on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df = df.dropna(subset=['QUALITY','RANKING'])
    df = df[(df['imps']>0) & (df['RANKING']>0) & (df['QUALITY']>0)]

    # Residualize log(rank) ~ [1, log(quality)]
    lr = safe_log_arr(df['RANKING'].values)
    lq = safe_log_arr(df['QUALITY'].values)
    X = np.column_stack([np.ones(len(df)), lq])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ lr)
    lr_hat = (X @ beta)
    r = lr - lr_hat

    # GLM with QUALITY offset and residual as regressor
    q_clip = np.clip(df['QUALITY'].values.astype(float), 1e-6, 1-1e-6)
    offset = np.log(q_clip/(1-q_clip))
    exog = pd.DataFrame({'intercept': 1.0, 'rank_resid': r}).astype(float)
    endog = np.column_stack([df['clks'].values.astype(float), (df['imps'].values - df['clks'].values).astype(float)])

    glm = sm.GLM(endog, exog, family=sm.families.Binomial(), offset=offset)
    res = glm.fit(cov_type='HC1')
    se = np.sqrt(np.diag(res.cov_params()))
    b = float(res.params['rank_resid'])
    b_se = float(se[list(res.params.index).index('rank_resid')])
    z = b / b_se if b_se>0 else np.nan
    from scipy.stats import norm
    p = 2*(1-norm.cdf(abs(z))) if np.isfinite(z) else np.nan

    # AUC of offset only vs full linear score
    lin_off = offset
    lin_full = offset + np.asarray(exog) @ res.params
    auc_off = auc_from_aggregated(lin_off, df['imps'].values, df['clks'].values)
    auc_full = auc_from_aggregated(lin_full, df['imps'].values, df['clks'].values)

    print(f"\n=== {args.round.upper()} Residualized rank beyond quality ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print(f"First-stage OLS: log(rank) = {beta[0]:.4f} + {beta[1]:.4f}*log(quality)")
    print(f"Second-stage GLM: coef(rank_resid)={b:.4f}  (se={b_se:.4f})  z={z:.2f}  p={p:.3e}")
    print(f"ROC AUC: offset-only={auc_off:.4f}  ->  offset+rank_resid={auc_full:.4f}")

if __name__ == '__main__':
    main()

