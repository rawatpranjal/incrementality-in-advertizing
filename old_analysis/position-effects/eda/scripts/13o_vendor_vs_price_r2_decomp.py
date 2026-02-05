#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'

def get_path(round_name: str) -> Path:
    if round_name == 'round1':
        return DATA_DIR / 'round1/auctions_results_all.parquet'
    if round_name == 'round2':
        return DATA_DIR / 'round2/auctions_results_r2.parquet'
    raise ValueError(round_name)

def fit_fe_plus_X(y: np.ndarray, X: np.ndarray, g: np.ndarray):
    # y = alpha_g + X beta + e; return beta, alpha_g, yhat, SSE
    df = pd.DataFrame({'y': y, 'g': g})
    X_df = pd.DataFrame(X)
    df = pd.concat([df, X_df], axis=1)
    gm = df.groupby('g').transform('mean')
    y_til = (df['y'] - gm['y']).values.reshape(-1,1)
    X_til = (X_df - gm[X_df.columns]).values
    if X_til.ndim == 1:
        X_til = X_til.reshape(-1,1)
    XtX = X_til.T @ X_til
    Xty = X_til.T @ y_til
    beta = (np.linalg.pinv(XtX) @ Xty).flatten() if X_til.shape[1] > 0 else np.zeros(0)
    # Recover FE
    means = df.groupby('g').mean()
    xbar = means[X_df.columns].values if X_df.shape[1] > 0 else np.zeros((means.shape[0],0))
    ybar = means['y'].values
    alpha = ybar - (xbar @ beta) if X_df.shape[1] > 0 else ybar.copy()
    alpha_series = pd.Series(alpha, index=means.index)
    yhat = alpha_series.loc[df['g']].values + (X_df.values @ beta if X_df.shape[1] > 0 else 0)
    resid = df['y'].values - yhat
    sse = float(np.sum(resid**2))
    return beta, alpha_series, yhat, sse

def main():
    ap = argparse.ArgumentParser(description='Decompose R^2 of log(FINAL_BID) into vendor FE vs log(PRICE) parts')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','PRICE','CONVERSION_RATE']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0)].copy()
    y = np.log(df['FINAL_BID'].values.astype(float))
    lp = np.log(df['PRICE'].values.astype(float))
    lc = np.log(df['CONVERSION_RATE'].values.astype(float))
    g = df['VENDOR_ID'].values
    sst = float(np.sum((y - y.mean())**2))

    # FE only
    _, alpha_fe, yhat_fe, sse_fe = fit_fe_plus_X(y, X=np.zeros((len(y),0)), g=g)
    r2_fe = 1.0 - sse_fe/sst

    # FE + log price
    beta_p, _, yhat_fep, sse_fep = fit_fe_plus_X(y, X=lp.reshape(-1,1), g=g)
    r2_fep = 1.0 - sse_fep/sst
    r2_inc_price = (sse_fe - sse_fep)/sst

    # FE + log price + log cvr
    beta_pc, _, yhat_fepc, sse_fepc = fit_fe_plus_X(y, X=np.column_stack([lp, lc]), g=g)
    r2_fepc = 1.0 - sse_fepc/sst
    r2_inc_cvr_given_price = (sse_fep - sse_fepc)/sst

    print(f"\n=== {args.round.upper()} Decomposition (log scale) ===")
    print(f"n={len(df):,} vendors={df['VENDOR_ID'].nunique():,}")
    print(f"Total R^2 FE+log(price)+log(cvr): {r2_fepc:.4f}")
    print(f"  FE only R^2:                         {r2_fe:.4f}")
    print(f"  + log(price) incremental R^2:        {r2_inc_price:.4f}  (beta_price={beta_p[0]:.4f})")
    print(f"  + log(cvr) incremental R^2 (given P): {r2_inc_cvr_given_price:.6f}  (beta_cvr={beta_pc[1]:.4f})")
    print(f"Check: FE + inc(price) ≈ {r2_fe + r2_inc_price:.4f}")

if __name__ == '__main__':
    main()

