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

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return float(np.sqrt(np.mean(e * e)))

def ols_intercept(y: np.ndarray, x: np.ndarray):
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    yhat = X @ beta
    return float(beta[0]), float(beta[1]), yhat

def fe_within_coeff(y: np.ndarray, x: np.ndarray, g: np.ndarray):
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
    return b, yhat

def fe_within_coeff_multi(y: np.ndarray, X: np.ndarray, g: np.ndarray):
    # X with 2 columns: log_price, log_cvr
    df = pd.DataFrame({'y': y, 'g': g})
    X_df = pd.DataFrame(X, columns=['x1','x2'])
    df = pd.concat([df, X_df], axis=1)
    gm = df.groupby('g').transform('mean')
    y_til = (df['y'] - gm['y']).values.reshape(-1,1)
    X_til = (X_df - gm[['x1','x2']]).values
    XtX = X_til.T @ X_til
    Xty = X_til.T @ y_til
    beta = (np.linalg.pinv(XtX) @ Xty).flatten()
    means = df.groupby('g').mean()
    xbar = means[['x1','x2']].values
    ybar = means['y'].values
    alpha = ybar - xbar @ beta
    alpha_series = pd.Series(alpha, index=means.index)
    yhat = alpha_series.loc[df['g']].values + (X_df.values @ beta)
    return beta, yhat

def main():
    ap = argparse.ArgumentParser(description='Compare RMSE on FINAL_BID across models')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','CONVERSION_RATE','PRICE']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['CONVERSION_RATE']>0) & (df['PRICE']>0)].copy()

    # Precompute variables
    bid = df['FINAL_BID'].values.astype(float)
    vendor = df['VENDOR_ID'].values
    cp = (df['CONVERSION_RATE'] * df['PRICE']).values.astype(float)
    log_bid = np.log(bid)
    log_price = np.log(df['PRICE'].values.astype(float))
    log_cvr = np.log(df['CONVERSION_RATE'].values.astype(float))
    log_cp = log_price + log_cvr

    # 1) Plain levels: y = a + b*CP
    a, b, yhat_plain = ols_intercept(bid, cp)
    rmse_plain = rmse(bid, yhat_plain)

    # 2) Vendor FE levels: y = alpha_g + b*CP
    b_fe_levels, yhat_fe_levels = fe_within_coeff(bid, cp, vendor)
    rmse_fe_levels = rmse(bid, yhat_fe_levels)

    # 3) Vendor FE log separate: log y = alpha_g + beta*log_price + gamma*log_cvr
    beta_vec, yhat_log_sep = fe_within_coeff_multi(log_bid, np.column_stack([log_price, log_cvr]), vendor)
    resid_log_sep = log_bid - yhat_log_sep
    smear_sep = float(np.mean(np.exp(resid_log_sep)))
    yhat_sep = np.exp(yhat_log_sep) * smear_sep
    rmse_log_sep = rmse(bid, yhat_sep)

    # 4) Vendor FE log single: log y = alpha_g + delta*(log_price+log_cvr)
    d_fe_single, yhat_log_single = fe_within_coeff(log_bid, log_cp, vendor)
    resid_log_single = log_bid - yhat_log_single
    smear_single = float(np.mean(np.exp(resid_log_single)))
    yhat_single = np.exp(yhat_log_single) * smear_single
    rmse_log_single = rmse(bid, yhat_single)

    print(f"\n=== {args.round.upper()} RMSE on FINAL_BID ===")
    print(f"n={len(df):,} vendors={df['VENDOR_ID'].nunique():,}")
    print(f"Plain levels:     FINAL_BID = a + b*(CVR*PRICE)                 RMSE={rmse_plain:.6f}")
    print(f"Vendor FE levels: FINAL_BID = alpha_vendor + b*(CVR*PRICE)      RMSE={rmse_fe_levels:.6f}")
    print(f"Vendor FE log:    log(BID) = alpha + beta*log(P) + gamma*log(C)  RMSE={rmse_log_sep:.6f}  (smear={smear_sep:.5f})")
    print(f"Vendor FE log1:   log(BID) = alpha + delta*(log(P)+log(C))       RMSE={rmse_log_single:.6f}  (smear={smear_single:.5f})")

if __name__ == '__main__':
    main()

