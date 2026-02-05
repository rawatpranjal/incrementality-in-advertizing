#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'

def get_path(round_name: str) -> Path:
    if round_name == 'round1':
        return DATA_DIR / 'round1/auctions_results_all.parquet'
    if round_name == 'round2':
        return DATA_DIR / 'round2/auctions_results_r2.parquet'
    raise ValueError(round_name)

def within_transform(df: pd.DataFrame, y_col: str, x_cols: list, g_col: str):
    use = [y_col] + x_cols + [g_col]
    d = df[use].copy()
    gm = d.groupby(g_col).transform('mean')
    y_til = (d[y_col] - gm[y_col]).values
    X_til = (d[x_cols] - gm[x_cols]).values
    groups = d[g_col].values
    return y_til, X_til, groups

def ols_fit(y: np.ndarray, X: np.ndarray):
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.pinv(XtX) @ Xty
    yhat = X @ beta
    resid = y - yhat
    return beta, resid, XtX

def se_homoskedastic(resid: np.ndarray, XtX: np.ndarray, n: int, k: int):
    sigma2 = float(resid.T @ resid) / max(n - k, 1)
    cov = sigma2 * np.linalg.pinv(XtX)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return se

def se_cluster(groups: np.ndarray, X: np.ndarray, resid: np.ndarray, XtX: np.ndarray):
    # Cluster-robust by group using sandwich estimator with finite-sample correction
    invXtX = np.linalg.pinv(XtX)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    df['resid'] = resid
    df['g'] = groups
    # Compute S_g = X_g' u_g via group sums of x_j * resid
    agg = df.groupby('g').apply(lambda d: (d.drop(columns=['resid','g']).values * d['resid'].values[:,None]).sum(axis=0))
    S = np.vstack(agg.values)  # G x k
    meat = S.T @ S
    # Finite-sample correction
    N = X.shape[0]
    K = X.shape[1]
    G = S.shape[0]
    scale = (G/(G-1)) * ((N-1)/(N-K)) if G > 1 and N > K else 1.0
    cov_cr = scale * (invXtX @ meat @ invXtX)
    se_cr = np.sqrt(np.clip(np.diag(cov_cr), 0, None))
    df_denom = max(G - 1, 1)
    return se_cr, df_denom

def main():
    ap = argparse.ArgumentParser(description='Vendor FE log model with log(price), log(cvr), log(pacing) + significance')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','PRICE','CONVERSION_RATE','PACING']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0) & (df['PACING']>0)].copy()
    df['log_bid'] = np.log(df['FINAL_BID'])
    df['log_price'] = np.log(df['PRICE'])
    df['log_cvr'] = np.log(df['CONVERSION_RATE'])
    df['log_pacing'] = np.log(df['PACING'])

    y_til, X_til, groups = within_transform(df, 'log_bid', ['log_price','log_cvr','log_pacing'], 'VENDOR_ID')
    n, k = X_til.shape
    beta, resid, XtX = ols_fit(y_til, X_til)

    # Homoskedastic SE
    se = se_homoskedastic(resid, XtX, n, k)
    t_stats = beta / se
    pvals = 2 * student_t.sf(np.abs(t_stats), df=max(n-k,1))

    # Cluster-robust SE by vendor
    se_cr, df_cr = se_cluster(groups, X_til, resid, XtX)
    t_cr = beta / se_cr
    p_cr = 2 * student_t.sf(np.abs(t_cr), df=df_cr)

    # Report
    names = ['log_price','log_cvr','log_pacing']
    print(f"\n=== {args.round.upper()} Vendor FE with pacing (significance) ===")
    print(f"n={n:,} vendors={df['VENDOR_ID'].nunique():,} k={k}")
    for i, nm in enumerate(names):
        print(f"{nm:>12}: beta={beta[i]: .6f} | SE={se[i]: .6e} t={t_stats[i]: .2f} p={pvals[i]:.3e} | SE_clust={se_cr[i]: .6e} t_clust={t_cr[i]: .2f} p_clust={p_cr[i]:.3e}")

if __name__ == '__main__':
    main()

