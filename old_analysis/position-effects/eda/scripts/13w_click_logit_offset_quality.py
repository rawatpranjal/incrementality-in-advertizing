#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'

def get_paths(round_name: str) -> dict:
    if round_name == 'round1':
        return {
            'auctions_results': DATA_DIR / 'round1/auctions_results_all.parquet',
            'auctions_users': DATA_DIR / 'round1/auctions_users_all.parquet',
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
        }
    raise ValueError(round_name)

def auc_from_aggregated(scores: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    order = np.argsort(scores)
    s = scores[order]
    p = clks[order].astype(float)
    n = (imps[order] - clks[order]).astype(float)
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
    cum_n_below = np.cumsum(np.concatenate([[0.0], n_sum[:-1]]))
    U = np.sum(p_sum * (cum_n_below + 0.5 * n_sum))
    P = float(p_sum.sum())
    N = float(n_sum.sum())
    if P <= 0 or N <= 0:
        return np.nan
    return U / (P * N)

def fit_logistic_irls_offset(X: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray, max_iter: int = 30, tol: float = 1e-6):
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = offset + X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-6, 1-1e-6)
        z = eta + (y - mu) / (mu * (1 - mu))
        W = w * (mu * (1 - mu))
        WX = X * W[:, None]
        XtWX = X.T @ WX
        XtWz = X.T @ (W * (z - offset))  # subtract offset for normal equations
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # Final SEs
    eta = offset + X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-6, 1-1e-6)
    W = w * (mu * (1 - mu))
    WX = X * W[:, None]
    XtWX = X.T @ WX
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    # Log-likelihood (up to constant)
    p = y * w
    n = w - p
    ll = float(np.sum(p * np.log(mu) + n * np.log(1 - mu)))
    return beta, se, ll

def safe_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x.astype(float), eps))

def main():
    ap = argparse.ArgumentParser(description='Click logistic with QUALITY as offset; test added predictors')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar_cols = ['AUCTION_ID','PRODUCT_ID','QUALITY','FINAL_BID','PRICE','CONVERSION_RATE','RANKING','PACING']
    ar = pd.read_parquet(paths['auctions_results'], columns=ar_cols)
    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()

    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('clks').reset_index()
    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID']), on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df = df.merge(au, on='AUCTION_ID', how='left')
    df = df.dropna(subset=['QUALITY','FINAL_BID','PRICE','RANKING','PACING','PLACEMENT'])

    # Construct offset from QUALITY: use logit(QUALITY)
    q = np.clip(df['QUALITY'].values.astype(float), 1e-6, 1-1e-6)
    offset = np.log(q / (1 - q))

    # Features beyond QUALITY
    df['log_b'] = safe_log(df['FINAL_BID'])
    df['log_p'] = safe_log(df['PRICE'])
    df['log_cvr'] = safe_log(df['CONVERSION_RATE']) if 'CONVERSION_RATE' in df.columns else 0.0
    df['log_pac'] = safe_log(df['PACING'])
    df['log_rank'] = safe_log(df['RANKING'])

    plc_dum = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc_dum.shape[1] > 0:
        plc_dum = plc_dum.iloc[:, 1:]  # drop first level
        df = pd.concat([df, plc_dum], axis=1)

    base_cols = ['log_b','log_p','log_cvr','log_pac','log_rank']
    cols_present = [c for c in base_cols if df[c].notna().all()]

    X_list = [np.ones(len(df))]  # intercept allowed with offset
    names = ['intercept']
    for c in cols_present:
        X_list.append(df[c].values)
        names.append(c)
    for c in plc_dum.columns if plc_dum.shape[1] > 0 else []:
        X_list.append(df[c].values)
        names.append(c)
    X = np.column_stack(X_list)
    y = (df['clks'] / df['imps']).values
    w = df['imps'].values.astype(float)

    # Offset-only model (intercept only)
    X0 = np.ones((len(df), 1))
    beta0, se0, ll0 = fit_logistic_irls_offset(X0, y, w, offset)

    # Full model (intercept + features)
    beta, se, ll = fit_logistic_irls_offset(X, y, w, offset)

    # LRT
    df_diff = X.shape[1] - X0.shape[1]
    dev = 2 * (ll - ll0)
    p_lrt = 1 - chi2.cdf(dev, df_diff) if df_diff > 0 else np.nan

    # AUCs
    auc_q = auc_from_aggregated(offset, df['imps'].values, df['clks'].values)  # monotone in QUALITY
    auc_lin = auc_from_aggregated(offset + X @ beta, df['imps'].values, df['clks'].values)

    print(f"\n=== {args.round.upper()} Click logit with QUALITY as offset ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print(f"Design columns (beyond offset): {', '.join(names)}")
    for nm, b, s in zip(names, beta, se):
        print(f"  {nm:>12}: {b: .4f}  (se={s:.4f})")
    print(f"LogLik offset-only={ll0:.1f}  |  full={ll:.1f}  |  LRT dev={dev:.1f}  df={df_diff}  p={p_lrt:.3e}")
    print(f"AUC(offset only)={auc_q:.4f}  |  AUC(offset + features)={auc_lin:.4f}")

if __name__ == '__main__':
    main()

