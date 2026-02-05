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
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
        }
    raise ValueError(round_name)

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x.astype(float), eps, None))

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def felogit_vendor_IRLS(df: pd.DataFrame, max_iter: int = 25, tol: float = 1e-6):
    # df columns: vendor, imps, clks, logit_q, log_rank
    g = df['VENDOR_ID'].values
    vendors, inv = np.unique(g, return_inverse=True)
    G = vendors.size
    n = len(df)

    y = (df['clks'].values / np.clip(df['imps'].values, 1e-12, None)).astype(float)
    m = df['imps'].values.astype(float)
    X = np.column_stack([
        df['logit_q'].values.astype(float),
        df['log_rank'].values.astype(float),
    ])

    # Initialize
    beta = np.zeros(2)
    alpha = np.zeros(G)

    for it in range(max_iter):
        eta = alpha[inv] + X @ beta
        mu = logistic(eta)
        mu = np.clip(mu, 1e-6, 1-1e-6)
        W = m * (mu * (1 - mu))
        z = eta + (y - mu) / (mu * (1 - mu))

        # Weighted demeaning within vendor for X and z
        # Compute group weighted means
        W_series = pd.Series(W)
        idx = pd.Series(inv)
        z_mean = (pd.Series(z) * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
        X0 = X[:, 0]
        X1 = X[:, 1]
        X0_mean = (pd.Series(X0) * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
        X1_mean = (pd.Series(X1) * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
        z_til = z - z_mean.iloc[inv].values
        X_til0 = X0 - X0_mean.iloc[inv].values
        X_til1 = X1 - X1_mean.iloc[inv].values
        X_til = np.column_stack([X_til0, X_til1])

        # Solve WLS for beta on demeaned data (no intercept)
        WX = X_til * W[:, None]
        XtWX = X_til.T @ WX
        XtWz = X_til.T @ (W * z_til)
        beta_new = np.linalg.pinv(XtWX) @ XtWz

        # Update alpha as weighted mean of (z - X beta)
        res = z - (X @ beta_new)
        num = (pd.Series(res) * W_series).groupby(idx).sum()
        den = W_series.groupby(idx).sum()
        alpha_new = (num / den).fillna(0.0).values

        max_delta = max(np.max(np.abs(beta_new - beta)), np.max(np.abs(alpha_new - alpha)))
        beta, alpha = beta_new, alpha_new
        if max_delta < tol:
            break

    # Compute naive (Fisher) SE for beta using final W and X_til
    eta = alpha[inv] + X @ beta
    mu = logistic(eta)
    mu = np.clip(mu, 1e-6, 1-1e-6)
    W = m * (mu * (1 - mu))
    # Recompute demeaning matrices at convergence
    W_series = pd.Series(W)
    idx = pd.Series(inv)
    X0_mean = (pd.Series(X[:,0]) * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
    X1_mean = (pd.Series(X[:,1]) * W_series).groupby(idx).sum() / W_series.groupby(idx).sum()
    X_til0 = X[:,0] - X0_mean.iloc[inv].values
    X_til1 = X[:,1] - X1_mean.iloc[inv].values
    X_til = np.column_stack([X_til0, X_til1])
    WX = X_til * W[:, None]
    XtWX = X_til.T @ WX
    cov_beta = np.linalg.pinv(XtWX)
    se_beta = np.sqrt(np.clip(np.diag(cov_beta), 0, None))

    # Log-likelihood and pseudo R^2
    ll = float(np.sum(m * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))
    # Null with vendor FE only (no X): solve alpha via weighted mean of logit(y)
    # Use one IRLS step with X=0
    # Initialize alpha0 with logit of group rates
    y_counts = pd.DataFrame({'y': y, 'm': m, 'g': inv}).groupby('g').apply(lambda d: (d['y']*d['m']).sum() / d['m'].sum()).values
    y_counts = np.clip(y_counts, 1e-6, 1-1e-6)
    alpha0 = np.log(y_counts/(1 - y_counts))
    mu0 = logistic(alpha0[inv])
    ll0 = float(np.sum(m * (y * np.log(mu0) + (1 - y) * np.log(1 - mu0))))
    pseudo_r2 = 1.0 - ll/ll0 if ll0 != 0 else np.nan

    return {
        'beta': beta,
        'se_beta': se_beta,
        'alpha': alpha,
        'vendors': vendors,
        'loglik': ll,
        'loglik_fe_only': ll0,
        'pseudo_r2': pseudo_r2,
        'iters': it+1,
    }

def main():
    ap = argparse.ArgumentParser(description='Vendor FE logistic: click ~ logit(QUALITY) + log(RANKING) + vendor FE (IRLS with absorption)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    # Build (AUCTION_ID, PRODUCT_ID, VENDOR_ID) level data
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','QUALITY','RANKING'])
    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'])

    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID','VENDOR_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID','VENDOR_ID']).size().rename('clks').reset_index()
    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']), on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.dropna(subset=['QUALITY','RANKING'])
    df = df[(df['imps']>0) & (df['RANKING']>0) & (df['QUALITY']>0)]

    # Features
    q = np.clip(df['QUALITY'].values.astype(float), 1e-6, 1-1e-6)
    df['logit_q'] = np.log(q/(1-q))
    df['log_rank'] = safe_log(df['RANKING'].values)

    res = felogit_vendor_IRLS(df)
    bq, br = res['beta'][0], res['beta'][1]
    sq, sr = res['se_beta'][0], res['se_beta'][1]
    zq = bq/sq if sq>0 else np.nan
    zr = br/sr if sr>0 else np.nan

    print(f"\n=== {args.round.upper()} Vendor FE logistic: click ~ logit(QUALITY) + log(RANK) + vendor FE ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,} vendors={df['VENDOR_ID'].nunique():,}")
    print(f"Converged in {res['iters']} IRLS iterations")
    print(f"beta_logitQ = {bq:.4f}  (se={sq:.4f})  z={zq:.2f}")
    print(f"beta_logRank = {br:.4f}  (se={sr:.4f})  z={zr:.2f}")
    print(f"logLik = {res['loglik']:.1f}  logLik(FE-only) = {res['loglik_fe_only']:.1f}  pseudoR2={res['pseudo_r2']:.4f}")

if __name__ == '__main__':
    main()

