#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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

def auc_from_aggregated(scores: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    # Mann-Whitney U with ties using aggregated counts by score
    # Sort by score ascending
    order = np.argsort(scores)
    s = scores[order]
    p = clks[order].astype(float)
    n = (imps[order] - clks[order]).astype(float)
    # Group by unique scores to handle ties
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
    # Cumulative negatives below each tie block
    cum_n_below = np.cumsum(np.concatenate([[0.0], n_sum[:-1]]))
    U = np.sum(p_sum * (cum_n_below + 0.5 * n_sum))
    P = float(p_sum.sum())
    N = float(n_sum.sum())
    if P <= 0 or N <= 0:
        return np.nan
    return U / (P * N)

def fit_logistic_irls(X: np.ndarray, y: np.ndarray, w: np.ndarray, max_iter: int = 25, tol: float = 1e-6):
    # Weighted logistic regression for binomial counts aggregated: y = clicks/imps, weights = imps
    n, k = X.shape
    beta = np.zeros(k)
    for it in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-6, 1-1e-6)
        # Working response and weights
        z = eta + (y - mu) / (mu * (1 - mu))
        W = w * (mu * (1 - mu))
        # Solve (X' W X) beta = X' W z
        WX = X * W[:, None]
        XtWX = X.T @ WX
        XtWz = X.T @ (W * z)
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # Approximate covariance
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-6, 1-1e-6)
    W = w * (mu * (1 - mu))
    WX = X * W[:, None]
    XtWX = X.T @ WX
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se

def main():
    ap = argparse.ArgumentParser(description='QUALITY calibration, AUC, and logistic click model with log(QUALITY), log(FINAL_BID)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','QUALITY','FINAL_BID'])
    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID'])

    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('clks').reset_index()
    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID']), on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df = df.dropna(subset=['QUALITY','FINAL_BID'])
    df = df[(df['QUALITY']>0) & (df['FINAL_BID']>0)]
    df['ctr'] = df['clks'] / df['imps']

    # Calibration by QUALITY decile
    df['q_decile'] = pd.qcut(df['QUALITY'], 10, labels=False, duplicates='drop')
    calib = df.groupby('q_decile').apply(lambda g: pd.Series({
        'mean_quality': g['QUALITY'].mean(),
        'ctr': g['clks'].sum() / g['imps'].sum(),
        'n_pairs': len(g),
        'imps': g['imps'].sum(),
    })).reset_index()

    # AUC using QUALITY as score
    auc_q = auc_from_aggregated(df['QUALITY'].values, df['imps'].values, df['clks'].values)

    # Logistic model: intercept + log(QUALITY) + log(FINAL_BID)
    df['log_q'] = np.log(df['QUALITY'].values)
    df['log_b'] = np.log(df['FINAL_BID'].values)
    X = np.column_stack([np.ones(len(df)), df['log_q'].values, df['log_b'].values])
    y = df['ctr'].values
    w = df['imps'].values.astype(float)
    beta, se = fit_logistic_irls(X, y, w)

    # AUC of fitted linear predictor
    lin_score = X @ beta
    auc_lin = auc_from_aggregated(lin_score, df['imps'].values, df['clks'].values)

    print(f"\n=== {args.round.upper()} QUALITY calibration and click model ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print("Calibration by QUALITY decile (mean QUALITY vs empirical CTR):")
    for _, r in calib.iterrows():
        print(f"  decile {int(r['q_decile'])}: meanQ={r['mean_quality']:.6f}  CTR={r['ctr']*100:.3f}%  n_pairs={int(r['n_pairs'])}  imps={int(r['imps'])}")
    print(f"AUC using QUALITY as score: {auc_q:.4f}")
    print("Weighted logistic regression: click ~ log(QUALITY) + log(FINAL_BID)")
    print(f"  intercept={beta[0]:.4f} (se={se[0]:.4f})  logQ={beta[1]:.4f} (se={se[1]:.4f})  logB={beta[2]:.4f} (se={se[2]:.4f})")
    print(f"AUC using fitted linear predictor: {auc_lin:.4f}")

if __name__ == '__main__':
    main()

