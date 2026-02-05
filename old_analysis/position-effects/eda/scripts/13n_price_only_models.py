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

def fe_within_single(y: np.ndarray, x: np.ndarray, g: np.ndarray):
    # One-regressor FE: y = alpha_g + b*x + e, return b, yhat, within R^2
    df = pd.DataFrame({'y': y, 'x': x, 'g': g})
    gm = df.groupby('g').transform('mean')
    y_til = (df['y'] - gm['y']).values
    x_til = (df['x'] - gm['x']).values
    denom = float(np.dot(x_til, x_til))
    b = float(np.dot(x_til, y_til) / denom) if denom > 0 else np.nan
    means = df.groupby('g').mean()
    alpha = means['y'].values - b * means['x'].values
    alpha_series = pd.Series(alpha, index=means.index)
    yhat = alpha_series.loc[df['g']].values + b * df['x'].values
    # within R^2
    sse = float(np.sum((y_til - b * x_til)**2))
    sst = float(np.sum(y_til**2))
    r2_within = 1.0 - sse/sst if sst>0 else np.nan
    return b, yhat, r2_within

def ols_loglog(y_log: np.ndarray, x_log: np.ndarray):
    X = np.column_stack([np.ones_like(x_log), x_log])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y_log)
    yhat_log = X @ beta
    resid = y_log - yhat_log
    sst = float(np.sum((y_log - y_log.mean())**2))
    sse = float(np.sum(resid**2))
    r2 = 1.0 - sse/sst if sst>0 else np.nan
    return float(beta[0]), float(beta[1]), yhat_log, r2

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return float(np.sqrt(np.mean(e*e)))

def main():
    ap = argparse.ArgumentParser(description='Price-only models: (1) FE log(BID)=alpha+beta*log(PRICE); (2) OLS log-log')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','PRICE']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0)].copy()

    bid = df['FINAL_BID'].values.astype(float)
    log_bid = np.log(bid)
    log_price = np.log(df['PRICE'].values.astype(float))
    vendor = df['VENDOR_ID'].values

    # (1) Vendor FE: log(BID) = alpha_vendor + beta*log(PRICE)
    beta_fe, yhat_log_fe, r2_within = fe_within_single(log_bid, log_price, vendor)
    resid_log_fe = log_bid - yhat_log_fe
    smear_fe = float(np.mean(np.exp(resid_log_fe)))
    yhat_fe_bid = np.exp(yhat_log_fe) * smear_fe
    rmse_fe = rmse(bid, yhat_fe_bid)

    # (2) Plain OLS: log(BID) = a + beta*log(PRICE)
    a_ols, b_ols, yhat_log_ols, r2_ols = ols_loglog(log_bid, log_price)
    resid_log_ols = log_bid - yhat_log_ols
    smear_ols = float(np.mean(np.exp(resid_log_ols)))
    yhat_ols_bid = np.exp(yhat_log_ols) * smear_ols
    rmse_ols = rmse(bid, yhat_ols_bid)

    print(f"\n=== {args.round.upper()} Price-only Models ===")
    print(f"n={len(df):,} vendors={df['VENDOR_ID'].nunique():,}")
    print(f"FE log:     log(BID)=alpha_vendor + {beta_fe:.4f}*log(PRICE)  within R^2={r2_within:.4f}  RMSE(BID)={rmse_fe:.6f}  (smear={smear_fe:.5f})")
    print(f"OLS loglog: log(BID)={a_ols:.4f} + {b_ols:.4f}*log(PRICE)    R^2={r2_ols:.4f}       RMSE(BID)={rmse_ols:.6f}  (smear={smear_ols:.5f})")

if __name__ == '__main__':
    main()

