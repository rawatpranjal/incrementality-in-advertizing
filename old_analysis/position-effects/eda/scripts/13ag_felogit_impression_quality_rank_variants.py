#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

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
    # df must have columns: 'VENDOR_ID', 'clicked' (0/1), feature_cols
    g = df['VENDOR_ID'].values
    vendors, inv = np.unique(g, return_inverse=True)
    G = vendors.size
    n = len(df)

    y = df['clicked'].values.astype(float)
    m = np.ones(n, dtype=float)  # impression-level weights
    X = df[feature_cols].values.astype(float)

    k = X.shape[1]
    beta = np.zeros(k)
    alpha = np.zeros(G)

    for it in range(max_iter):
        eta = alpha[inv] + X @ beta
        mu = logistic(eta)
        mu = np.clip(mu, 1e-6, 1-1e-6)
        W = m * (mu * (1 - mu))
        z = eta + (y - mu) / (mu * (1 - mu))

        W_series = pd.Series(W)
        idx = pd.Series(inv)
        # Weighted group means for each column
        X_til_cols = []
        for j in range(k):
            col = pd.Series(X[:, j])
            col_mean = (col * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
            X_til_cols.append(X[:, j] - col_mean.iloc[inv].values)
        X_til = np.column_stack(X_til_cols)
        z_mean = (pd.Series(z) * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
        z_til = z - z_mean.iloc[inv].values

        WX = X_til * W[:, None]
        XtWX = X_til.T @ WX
        XtWz = X_til.T @ (W * z_til)
        beta_new = np.linalg.pinv(XtWX) @ XtWz

        res = z - (X @ beta_new)
        num = (pd.Series(res) * W_series).groupby(idx).sum()
        den = W_series.groupby(idx).sum()
        alpha_new = (num / den).fillna(0.0).values

        max_delta = max(np.max(np.abs(beta_new - beta)), np.max(np.abs(alpha_new - alpha)))
        beta, alpha = beta_new, alpha_new
        if max_delta < tol:
            break

    # SE for beta (Fisher) on demeaned design
    eta = alpha[inv] + X @ beta
    mu = logistic(eta)
    mu = np.clip(mu, 1e-6, 1-1e-6)
    W = m * (mu * (1 - mu))
    W_series = pd.Series(W)
    idx = pd.Series(inv)
    X_til_cols = []
    for j in range(k):
        col = pd.Series(X[:, j])
        col_mean = (col * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
        X_til_cols.append(X[:, j] - col_mean.iloc[inv].values)
    X_til = np.column_stack(X_til_cols)
    WX = X_til * W[:, None]
    XtWX = X_til.T @ WX
    cov_beta = np.linalg.pinv(XtWX)
    se_beta = np.sqrt(np.clip(np.diag(cov_beta), 0, None))

    ll = float(np.sum(m * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))
    # FE-only baseline
    # Compute vendor-level mean logit quickly
    grp = pd.DataFrame({'y': y, 'm': m, 'g': inv}).groupby('g').agg({'y':'sum','m':'sum'})
    p_g = np.clip((grp['y']/grp['m']).values, 1e-6, 1-1e-6)
    alpha0 = np.log(p_g/(1 - p_g))
    mu0 = logistic(alpha0[inv])
    ll0 = float(np.sum(m * (y * np.log(mu0) + (1 - y) * np.log(1 - mu0))))
    pseudo_r2 = 1.0 - ll/ll0 if ll0 != 0 else np.nan

    return beta, se_beta, alpha, vendors, ll, ll0, pseudo_r2, it+1

def main():
    ap = argparse.ArgumentParser(description='Impression-level FE logit: click ~ quality + rank variants, vendor FE')
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
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']),
                  on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    # Add placement via AUCTION_ID
    df = df.merge(au, on='AUCTION_ID', how='left')
    df = df.dropna(subset=['RANKING','QUALITY','PRICE','CONVERSION_RATE'])
    df = df[(df['RANKING']>0) & (df['QUALITY']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0)]

    df['quality'] = df['QUALITY'].astype(float)
    df['rank'] = df['RANKING'].astype(float)
    df['log_rank'] = safe_log(df['rank'].values)
    df['rank_sq'] = df['rank']**2
    df['price'] = df['PRICE'].astype(float)
    df['cvr'] = df['CONVERSION_RATE'].astype(float)
    # Placement dummies (drop first level to avoid collinearity with vendor FE absorption)
    plc_dum = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc_dum.shape[1] > 0:
        plc_dum = plc_dum.iloc[:, 1:]
    df = pd.concat([df, plc_dum], axis=1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13ag_felogit_impression_quality_rank_variants_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Impression-level FE-logit (vendor FE) on {args.round}")
        wprint(f"Rows={len(df):,} impressions CTR={df['clicked'].mean()*100:.3f}% vendors={df['VENDOR_ID'].nunique():,}")

        base_specs = [
            (['quality','rank','price','cvr'], 'quality + rank + price + cvr'),
        ]
        # Append placement FE columns to each spec if present
        plc_cols = list(plc_dum.columns)
        specs = []
        for cols, name in base_specs:
            if plc_cols:
                specs.append((cols + plc_cols, name + ' + placement_FE'))
            else:
                specs.append((cols, name))
        for cols, name in specs:
            wprint(f"\nSpec: {name}")
            beta, se, alpha, vendors, ll, ll0, pr2, iters = felogit_vendor_IRLS(df, cols)
            for c, b, s in zip(cols, beta, se):
                z = b/s if s>0 else np.nan
                wprint(f"  {c:>10}: {b: .6f} (se={s:.6f}) z={z:.2f}")
            wprint(f"  logLik={ll:.1f}  logLik(FE-only)={ll0:.1f}  pseudoR2={pr2:.5f}  iters={iters}")
        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()
