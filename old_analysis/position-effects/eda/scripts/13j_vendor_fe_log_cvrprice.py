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

def fit_within_single(y: np.ndarray, x: np.ndarray, groups) -> tuple:
    # Within-transformation for one regressor with vendor FE
    df = pd.DataFrame({'y': y, 'x': x, 'g': groups})
    gm = df.groupby('g').transform('mean')
    y_til = (df['y'] - gm['y']).values
    x_til = (df['x'] - gm['x']).values
    # OLS beta = (x'x)^{-1} x'y
    denom = float(np.dot(x_til, x_til))
    beta = float(np.dot(x_til, y_til) / denom) if denom > 0 else np.nan
    # FE alpha_g = ybar_g - beta * xbar_g
    means = df.groupby('g').mean()
    alpha = means['y'].values - beta * means['x'].values
    alpha_series = pd.Series(alpha, index=means.index, name='alpha_vendor')
    # Fitted and R^2
    yhat = alpha_series.loc[df['g']].values + beta * df['x'].values
    resid = df['y'].values - yhat
    sst = np.sum((df['y'].values - df['y'].values.mean())**2)
    sse = np.sum(resid**2)
    r2 = 1.0 - sse/sst if sst>0 else np.nan
    return beta, alpha_series, r2

def main():
    ap = argparse.ArgumentParser(description='log(BID) = vendor FE + beta*(log(CVR)+log(PRICE))')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','CONVERSION_RATE','PRICE']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['CONVERSION_RATE']>0) & (df['PRICE']>0)].copy()
    df['log_bid'] = np.log(df['FINAL_BID'])
    df['log_cp'] = np.log(df['CONVERSION_RATE']) + np.log(df['PRICE'])

    beta, alpha_vendor, r2 = fit_within_single(df['log_bid'].values, df['log_cp'].values, df['VENDOR_ID'].values)
    print(f"\n=== {args.round.upper()} ===")
    print(f"n={len(df):,} vendors={alpha_vendor.size:,}")
    print(f"log(BID) = alpha_vendor + {beta:.4f} * (log(CVR)+log(PRICE))   (R^2={r2:.3f})")
    implied_troas = np.exp(-alpha_vendor)  # if beta≈1, this is the vendor tROAS
    qs = implied_troas.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(4)
    print("implied vendor tROAS summary (if beta≈1):")
    print(qs)

if __name__ == '__main__':
    main()

