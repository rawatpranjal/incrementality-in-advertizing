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

def fit_within(y, X, groups):
    # Within-transformation for one-way fixed effects by groups
    df = pd.DataFrame({'y': y, 'g': groups})
    X_df = pd.DataFrame(X, columns=['log_price','log_cvr'])
    df = pd.concat([df, X_df], axis=1)
    # group means
    gm = df.groupby('g').transform('mean')
    y_til = (df['y'] - gm['y']).values.reshape(-1,1)
    X_til = (X_df - gm[['log_price','log_cvr']]).values
    # OLS coefficients (no intercept): beta = (X'X)^{-1} X'y
    XtX = X_til.T @ X_til
    Xty = X_til.T @ y_til
    beta = np.linalg.pinv(XtX) @ Xty
    beta = beta.flatten()
    # Recover FE alpha_g = ybar_g - beta * xbar_g
    means = df.groupby('g').mean()
    xbar = means[['log_price','log_cvr']].values
    ybar = means['y'].values
    alpha = ybar - xbar @ beta
    alpha_series = pd.Series(alpha, index=means.index, name='alpha_vendor')
    # Fitted values and R^2
    yhat = alpha_series.loc[df['g']].values + (X_df.values @ beta)
    resid = df['y'].values - yhat
    sst = np.sum((df['y'].values - df['y'].values.mean())**2)
    sse = np.sum(resid**2)
    r2 = 1.0 - sse/sst if sst>0 else np.nan
    return beta, alpha_series, r2

def main():
    ap = argparse.ArgumentParser(description='log(FINAL_BID) = vendor FE + beta*log(PRICE) + gamma*log(CVR)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','PRICE','CONVERSION_RATE']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0)].copy()
    df['log_bid'] = np.log(df['FINAL_BID'])
    df['log_price'] = np.log(df['PRICE'])
    df['log_cvr'] = np.log(df['CONVERSION_RATE'])
    y = df['log_bid'].values
    X = df[['log_price','log_cvr']].values
    g = df['VENDOR_ID'].values

    beta, alpha_vendor, r2 = fit_within(y, X, g)
    b_price, b_cvr = beta[0], beta[1]
    print(f"\n=== {args.round.upper()} ===")
    print(f"n={len(df):,} vendors={alpha_vendor.size:,}")
    print(f"log(FINAL_BID) = alpha_vendor + {b_price:.4f}*log(PRICE) + {b_cvr:.4f}*log(CVR)  (R^2={r2:.3f})")
    # Summaries of alpha and implied vendor target roas
    implied_troas = np.exp(-alpha_vendor)
    qs = implied_troas.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(4)
    print("implied vendor tROAS (exp(-alpha_vendor)) summary:")
    print(qs)

if __name__ == '__main__':
    main()

