#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

def get_paths(round_name: str) -> dict:
    if round_name == 'round1':
        return {
            'auctions_results': DATA_DIR / 'round1/auctions_results_all.parquet',
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
            'auctions_users': DATA_DIR / 'round1/auctions_users_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
        }
    raise ValueError(round_name)

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def felogit_vendor_IRLS(df: pd.DataFrame, feature_cols: list, max_iter: int = 25, tol: float = 1e-6):
    g = df['VENDOR_ID'].values
    vendors, inv = np.unique(g, return_inverse=True)
    G = vendors.size
    y = df['clicked'].values.astype(float)
    X = df[feature_cols].values.astype(float)
    k = X.shape[1]
    beta = np.zeros(k)
    alpha = np.zeros(G)
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        eta = alpha[inv] + X @ beta
        mu = logistic(eta)
        mu = np.clip(mu, 1e-6, 1-1e-6)
        W = (mu * (1 - mu))
        z = eta + (y - mu) / (mu * (1 - mu))
        idx = pd.Series(inv)
        Wser = pd.Series(W)
        X_til_cols = []
        for j in range(k):
            col = pd.Series(X[:, j])
            col_mean = (col * Wser).groupby(idx).sum() / Wser.groupby(idx).sum()
            X_til_cols.append(X[:, j] - col_mean.iloc[inv].values)
        X_til = np.column_stack(X_til_cols)
        z_mean = (pd.Series(z) * Wser).groupby(idx).sum() / Wser.groupby(idx).sum()
        z_til = z - z_mean.iloc[inv].values
        WX = X_til * W[:, None]
        XtWX = X_til.T @ WX
        XtWz = X_til.T @ (W * z_til)
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        res = z - (X @ beta_new)
        num = (pd.Series(res) * Wser).groupby(idx).sum()
        den = Wser.groupby(idx).sum()
        alpha_new = (num / den).fillna(0.0).values
        if max(np.max(np.abs(beta_new - beta)), np.max(np.abs(alpha_new - alpha))) < tol:
            beta, alpha = beta_new, alpha_new
            break
        beta, alpha = beta_new, alpha_new
    eta = alpha[inv] + X @ beta
    mu = logistic(eta)
    mu = np.clip(mu, 1e-9, 1-1e-9)
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    return beta, alpha, vendors, inv, mu, ll, iters

def build_impression_df(paths: dict) -> pd.DataFrame:
    imp = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    clk = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','RANKING','QUALITY','PRICE','CONVERSION_RATE'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    clk = clk.assign(clicked=1)
    clicks_key = clk[['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','clicked']].drop_duplicates()
    df = imp.merge(clicks_key, on=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'], how='left')
    df['clicked'] = df['clicked'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']), on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.merge(au, on='AUCTION_ID', how='left')
    df = df.dropna(subset=['RANKING','QUALITY','PRICE','CONVERSION_RATE'])
    df = df[(df['RANKING']>0) & (df['QUALITY']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0)]
    df = df.rename(columns={'RANKING':'rank','QUALITY':'quality','PRICE':'price','CONVERSION_RATE':'cvr'})
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]
    df = pd.concat([df, plc], axis=1)
    return df, list(plc.columns)

def main():
    ap = argparse.ArgumentParser(description='Vendor FE overfitting check: restrict to vendors with sufficient impressions')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--min_impressions', type=int, default=200)
    args = ap.parse_args()

    paths = get_paths(args.round)
    df, plc_cols = build_impression_df(paths)
    base_rows = len(df)
    base_ctr = df['clicked'].mean()
    vc = df['VENDOR_ID'].value_counts()
    q = vc.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(2)

    # Restrict vendors with enough impressions
    keep_vendors = set(vc[vc >= args.min_impressions].index)
    df_r = df[df['VENDOR_ID'].isin(keep_vendors)].copy()
    rows_r = len(df_r)
    ctr_r = df_r['clicked'].mean()

    feature_cols = ['quality','rank','price','cvr'] + plc_cols

    # Quality-only scores for ROC/PR
    y = df_r['clicked'].values.astype(int)
    s_q = df_r['quality'].values.astype(float)

    beta, alpha, vendors, inv, p_full, ll, iters = felogit_vendor_IRLS(df_r, feature_cols)
    roc_q = roc_auc_score(y, s_q)
    pr_q = average_precision_score(y, s_q)
    roc_full = roc_auc_score(y, p_full)
    pr_full = average_precision_score(y, p_full)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13ak_vendor_fe_restricted_{args.round}_min{args.min_impressions}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Vendor FE restricted run — {args.round}")
        wprint(f"Restriction: min_impressions_per_vendor = {args.min_impressions}")
        wprint(f"All vendors: rows={base_rows:,} CTR={base_ctr*100:.3f}% vendors={df['VENDOR_ID'].nunique():,}")
        wprint(f"  vendor impressions summary:\n{q}")
        wprint(f"Kept vendors: rows={rows_r:,} CTR={ctr_r*100:.3f}% vendors={df_r['VENDOR_ID'].nunique():,}")
        wprint(f"Spec: quality + rank + price + cvr + placement_FE (vendor FE absorbed)")
        wprint(f"ROC_AUC: quality={roc_q:.4f}  full={roc_full:.4f}")
        wprint(f"PR_AUC:  quality={pr_q:.4f}  full={pr_full:.4f}")
        wprint(f"logLik(full)={ll:.1f}  iters={iters}")
        wprint(f"Output saved to: {out}")

if __name__ == '__main__':
    main()

