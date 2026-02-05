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
    use = [y_col] + x_cols + [g_col]
    d = df[use].copy()
    gm = d.groupby(g_col).transform('mean')
    y_til = (d[y_col] - gm[y_col]).values
    X_til = (d[x_cols] - gm[x_cols]).values
    return y_til, X_til

def ols_beta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1,1)
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.pinv(XtX) @ Xty
    return beta

def sse(y: np.ndarray, X: np.ndarray) -> float:
    if X.ndim == 1:
        X = X.reshape(-1,1)
    beta = ols_beta(y, X)
    resid = y - (X @ beta)
    return float(np.sum(resid**2))

def fit_vendor_fe_with_pacing(round_name: str):
    path = get_path(round_name)
    cols = ['VENDOR_ID','FINAL_BID','PRICE','CONVERSION_RATE','PACING']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0) & (df['PACING']>0)].copy()
    df['log_bid'] = np.log(df['FINAL_BID'])
    df['log_price'] = np.log(df['PRICE'])
    df['log_cvr'] = np.log(df['CONVERSION_RATE'])
    df['log_pacing'] = np.log(df['PACING'])

    y_til, X_til = within_transform(df, 'log_bid', ['log_price','log_cvr','log_pacing'], 'VENDOR_ID')
    sst = float(np.sum(y_til**2))

    # Full model
    beta_full = ols_beta(y_til, X_til).flatten()
    sse_full = sse(y_til, X_til)
    r2_within = 1.0 - sse_full/sst if sst>0 else np.nan

    # Reduced models for partial R^2
    # Drop log_price -> keep [log_cvr, log_pacing]
    sse_drop_price = sse(y_til, X_til[:,[1,2]])
    pr2_price = 1.0 - (sse_full / sse_drop_price) if sse_drop_price>0 else np.nan
    # Drop log_cvr -> keep [log_price, log_pacing]
    sse_drop_cvr = sse(y_til, X_til[:,[0,2]])
    pr2_cvr = 1.0 - (sse_full / sse_drop_cvr) if sse_drop_cvr>0 else np.nan
    # Drop log_pacing -> keep [log_price, log_cvr]
    sse_drop_pacing = sse(y_til, X_til[:,[0,1]])
    pr2_pacing = 1.0 - (sse_full / sse_drop_pacing) if sse_drop_pacing>0 else np.nan

    return df, beta_full, r2_within, pr2_price, pr2_cvr, pr2_pacing

def main():
    ap = argparse.ArgumentParser(description='Vendor FE log model with log(price), log(cvr), log(pacing)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    df, beta, r2w, pr2_p, pr2_c, pr2_pa = fit_vendor_fe_with_pacing(args.round)
    b_price, b_cvr, b_pacing = beta.tolist()
    print(f"\n=== {args.round.upper()} Vendor FE with pacing ===")
    print(f"n={len(df):,} vendors={df['VENDOR_ID'].nunique():,}")
    print(f"log(BID) = alpha_vendor + {b_price:.4f}*log(PRICE) + {b_cvr:.4f}*log(CVR) + {b_pacing:.4f}*log(PACING)")
    print(f"Within R^2 (FE removed): {r2w:.4f}")
    print("Partial R^2 contributions (within):")
    print(f"  | log(price):  {pr2_p:.4f}")
    print(f"  | log(cvr):    {pr2_c:.6f}")
    print(f"  | log(pacing): {pr2_pa:.6f}")

if __name__ == '__main__':
    main()

