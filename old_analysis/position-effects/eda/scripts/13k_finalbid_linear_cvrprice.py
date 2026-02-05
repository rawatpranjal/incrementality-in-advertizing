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

def ols_intercept(y: np.ndarray, x: np.ndarray):
    # Fit y = a + b*x by closed-form OLS
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat
    sst = float(np.sum((y - y.mean())**2))
    sse = float(np.sum(resid**2))
    r2 = 1.0 - sse/sst if sst > 0 else np.nan
    return beta[0], beta[1], r2

def fe_within(y: np.ndarray, x: np.ndarray, g):
    # One-regressor vendor FE: y = alpha_g + b*x + e (within transformation)
    df = pd.DataFrame({'y': y, 'x': x, 'g': g})
    gm = df.groupby('g').transform('mean')
    y_til = (df['y'] - gm['y']).values
    x_til = (df['x'] - gm['x']).values
    denom = float(np.dot(x_til, x_til))
    b = float(np.dot(x_til, y_til) / denom) if denom > 0 else np.nan
    means = df.groupby('g').mean()
    alpha = means['y'].values - b * means['x'].values
    alpha_series = pd.Series(alpha, index=means.index, name='alpha_vendor')
    yhat = alpha_series.loc[df['g']].values + b * df['x'].values
    resid = df['y'].values - yhat
    sst = float(np.sum((df['y'].values - df['y'].values.mean())**2))
    sse = float(np.sum(resid**2))
    r2 = 1.0 - sse/sst if sst > 0 else np.nan
    return b, alpha_series, r2

def main():
    ap = argparse.ArgumentParser(description='Compare FINAL_BID ~ a + CVR*PRICE and vendor-FE variant')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','CONVERSION_RATE','PRICE']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['CONVERSION_RATE']>0) & (df['PRICE']>0)].copy()
    df['x'] = df['CONVERSION_RATE'] * df['PRICE']

    # Plain OLS in levels
    a, b, r2_plain = ols_intercept(df['FINAL_BID'].values, df['x'].values)

    # Vendor FE in levels (same dependent scale)
    b_fe, alpha_vendor, r2_fe = fe_within(df['FINAL_BID'].values, df['x'].values, df['VENDOR_ID'].values)

    print(f"\n=== {args.round.upper()} ===")
    print(f"n={len(df):,} vendors={alpha_vendor.size:,}")
    print(f"Plain OLS: FINAL_BID = {a:.6f} + {b:.6f} * (CVR*PRICE)   (R^2={r2_plain:.3f})")
    print(f"Vendor FE: FINAL_BID = alpha_vendor + {b_fe:.6f} * (CVR*PRICE)   (R^2={r2_fe:.3f})")

if __name__ == '__main__':
    main()

