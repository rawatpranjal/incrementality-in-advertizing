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

def within_transform(df: pd.DataFrame, y_col: str, x_cols: list, g_col: str):
    # De-mean by group (vendor FE)
    use = [y_col] + x_cols + [g_col]
    d = df[use].copy()
    gm = d.groupby(g_col).transform('mean')
    y_til = (d[y_col] - gm[y_col]).values
    X_til = (d[x_cols] - gm[x_cols]).values
    return y_til, X_til

def ols_sse(y: np.ndarray, X: np.ndarray) -> float:
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if X.shape[1] == 0:
        # No regressors; SSE is SST around 0 for within data (mean zero)
        return float(np.sum(y**2))
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    resid = y - (X @ beta)
    return float(np.sum(resid**2))

def main():
    ap = argparse.ArgumentParser(description='Partial R^2 for log(PRICE) and log(CVR) in vendor FE log model')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','PRICE','CONVERSION_RATE']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0)].copy()
    df['log_bid'] = np.log(df['FINAL_BID'])
    df['log_price'] = np.log(df['PRICE'])
    df['log_cvr'] = np.log(df['CONVERSION_RATE'])

    # Within transform to remove vendor FE
    y_til, X_til = within_transform(df, 'log_bid', ['log_price','log_cvr'], 'VENDOR_ID')
    x_price = X_til[:,0]
    x_cvr = X_til[:,1]

    # Full model SSE and R^2 (within)
    sse_full = ols_sse(y_til, X_til)
    sst = float(np.sum(y_til**2))  # mean of y_til is 0 by construction
    r2_full = 1.0 - sse_full/sst if sst>0 else np.nan

    # Reduced models SSE
    sse_red_price_only = ols_sse(y_til, x_price)
    sse_red_cvr_only = ols_sse(y_til, x_cvr)

    # Partial R^2
    # Contribution of log_price controlling for log_cvr: compare reduced (cvr only) vs full
    pr2_price = 1.0 - (sse_full / sse_red_cvr_only) if sse_red_cvr_only>0 else np.nan
    # Contribution of log_cvr controlling for log_price: compare reduced (price only) vs full
    pr2_cvr = 1.0 - (sse_full / sse_red_price_only) if sse_red_price_only>0 else np.nan

    print(f"\n=== {args.round.upper()} (Vendor FE, within) ===")
    print(f"n={len(df):,} vendors={df['VENDOR_ID'].nunique():,}")
    print(f"Within R^2 (full: log_price + log_cvr) = {r2_full:.4f}")
    print(f"Partial R^2 | log_price (unique, controlling log_cvr) = {pr2_price:.4f}")
    print(f"Partial R^2 | log_cvr   (unique, controlling log_price) = {pr2_cvr:.6f}")

if __name__ == '__main__':
    main()

