#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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

def fit_logistic_irls(X: np.ndarray, y: np.ndarray, w: np.ndarray, max_iter: int = 30, tol: float = 1e-6):
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-6, 1-1e-6)
        z = eta + (y - mu) / (mu * (1 - mu))
        W = w * (mu * (1 - mu))
        WX = X * W[:, None]
        XtWX = X.T @ WX
        XtWz = X.T @ (W * z)
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-6, 1-1e-6)
    W = w * (mu * (1 - mu))
    WX = X * W[:, None]
    XtWX = X.T @ WX
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se

def safe_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x.astype(float), eps))

def main():
    ap = argparse.ArgumentParser(description='Click logistic: logit(click) ~ log quality, log price, log bid, log cvr, log pacing, log rank, placement FEs')
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

    # Features
    df['log_q'] = safe_log(df['QUALITY'])
    df['log_b'] = safe_log(df['FINAL_BID'])
    df['log_p'] = safe_log(df['PRICE'])
    if 'CONVERSION_RATE' in df.columns:
        df['log_cvr'] = safe_log(df['CONVERSION_RATE'])
    else:
        df['log_cvr'] = np.nan
    df['log_pac'] = safe_log(df['PACING'])
    df['log_rank'] = safe_log(df['RANKING'])

    # Placement dummies (drop first to avoid collinearity)
    plc_dum = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc_dum.shape[1] > 0:
        plc_dum = plc_dum.iloc[:, 1:]
        df = pd.concat([df, plc_dum], axis=1)

    # Build design matrix
    base_cols = ['log_q','log_b','log_p','log_cvr','log_pac','log_rank']
    cols_present = [c for c in base_cols if df[c].notna().all()]
    X_list = [np.ones(len(df))]
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

    beta, se = fit_logistic_irls(X, y, w)

    # AUCs
    auc_q = auc_from_aggregated(df['QUALITY'].values, df['imps'].values, df['clks'].values)
    lin_score = X @ beta
    auc_lin = auc_from_aggregated(lin_score, df['imps'].values, df['clks'].values)

    print(f"\n=== {args.round.upper()} Click logistic with rich features ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print(f"Design columns: {', '.join(names)}")
    for nm, b, s in zip(names, beta, se):
        print(f"  {nm:>12}: {b: .4f}  (se={s:.4f})")
    print(f"AUC(QUALITY only)={auc_q:.4f}  |  AUC(full linear score)={auc_lin:.4f}")

if __name__ == '__main__':
    main()

