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

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x.astype(float), eps, None))

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def felogit_vendor_IRLS(df: pd.DataFrame, feature_cols: list, max_iter: int = 25, tol: float = 1e-6):
    g = df['VENDOR_ID'].values
    vendors, inv = np.unique(g, return_inverse=True)
    G = vendors.size
    n = len(df)
    y = df['clicked'].values.astype(float)
    X = df[feature_cols].values.astype(float)
    k = X.shape[1]
    beta = np.zeros(k)
    alpha = np.zeros(G)
    for it in range(max_iter):
        eta = alpha[inv] + X @ beta
        mu = logistic(eta)
        mu = np.clip(mu, 1e-6, 1-1e-6)
        W = (mu * (1 - mu))
        z = eta + (y - mu) / (mu * (1 - mu))
        # weighted group means
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
        max_delta = max(np.max(np.abs(beta_new - beta)), np.max(np.abs(alpha_new - alpha)))
        beta, alpha = beta_new, alpha_new
        if max_delta < tol:
            break
    return beta, alpha, vendors, inv, it+1

def main():
    ap = argparse.ArgumentParser(description='AUC/PR: QUALITY vs full FE-logit (vendor FE + placement FE) on impressions')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
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

    # features
    df['quality'] = df['QUALITY'].astype(float)
    df['rank'] = df['RANKING'].astype(float)
    df['price'] = df['PRICE'].astype(float)
    df['cvr'] = df['CONVERSION_RATE'].astype(float)
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]
    df = pd.concat([df, plc], axis=1)
    feature_cols = ['quality','rank','price','cvr'] + list(plc.columns)

    # Baseline scores: quality only (ranking metrics are order-invariant to monotone transforms)
    s_quality = df['quality'].values.astype(float)

    # Fit FE-logit and compute full-model predicted probabilities
    beta, alpha, vendors, inv, iters = felogit_vendor_IRLS(df, feature_cols)
    eta_full = alpha[inv] + df[feature_cols].values.astype(float) @ beta
    p_full = logistic(eta_full)

    y = df['clicked'].values.astype(int)
    # AUC ROC and PR
    roc_q = roc_auc_score(y, s_quality)
    pr_q = average_precision_score(y, s_quality)
    roc_full = roc_auc_score(y, p_full)
    pr_full = average_precision_score(y, p_full)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13ah_auc_quality_vs_full_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"AUC/PR on impressions ({args.round}) — QUALITY vs Full FE-logit")
        wprint(f"Rows={len(df):,} CTR={df['clicked'].mean()*100:.3f}% Vendors={df['VENDOR_ID'].nunique():,} Iters={iters}")
        wprint(f"QUALITY only: ROC_AUC={roc_q:.4f}  PR_AUC={pr_q:.4f}")
        wprint(f"Full model (vendor FE + placement FE): ROC_AUC={roc_full:.4f}  PR_AUC={pr_full:.4f}")
        wprint(f"Output saved to: {out}")

if __name__ == '__main__':
    main()
