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
        max_delta = max(np.max(np.abs(beta_new - beta)), np.max(np.abs(alpha_new - alpha)))
        beta, alpha = beta_new, alpha_new
        if max_delta < tol:
            break
    # Log-likelihood
    eta = alpha[inv] + X @ beta
    mu = logistic(eta)
    mu = np.clip(mu, 1e-9, 1-1e-9)
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    return ll, iters

def build_impression_frame(paths: dict) -> pd.DataFrame:
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
    # Placement dummies (drop first)
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]
    df = pd.concat([df, plc], axis=1)
    return df, list(plc.columns)

def main():
    ap = argparse.ArgumentParser(description='Decompose FE-logit contributions via drop-one Δdeviance, baseline = vendor FE only')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    df, plc_cols = build_impression_frame(paths)

    blocks = {
        'quality': ['quality'],
        'rank': ['rank'],
        'price': ['price'],
        'cvr': ['cvr'],
        'placement': plc_cols,
    }
    full_cols = blocks['quality'] + blocks['rank'] + blocks['price'] + blocks['cvr'] + blocks['placement']

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13ai_contrib_decomposition_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Contribution decomposition (drop-one Δdeviance) — {args.round}")
        wprint(f"Rows={len(df):,} CTR={df['clicked'].mean()*100:.3f}% Vendors={df['VENDOR_ID'].nunique():,}")

        # Baseline: vendor FE only (no X) — approximate LL via per-vendor mean
        grp = df.groupby('VENDOR_ID')['clicked'].agg(['sum','count']).rename(columns={'sum':'clks','count':'n'})
        p_g = np.clip((grp['clks']/grp['n']).values, 1e-6, 1-1e-6)
        p0 = pd.Series(p_g, index=grp.index)
        mu0 = p0.loc[df['VENDOR_ID']].values
        ll0 = float(np.sum(df['clicked'].values * np.log(mu0) + (1 - df['clicked'].values) * np.log(1 - mu0)))
        dev0 = -2 * ll0
        wprint(f"Baseline (vendor FE only): logLik={ll0:.1f}  deviance={dev0:.1f}")

        # Full model
        ll_full, it_full = felogit_vendor_IRLS(df, full_cols)
        dev_full = -2 * ll_full
        gain_total = dev0 - dev_full
        wprint(f"Full model cols={full_cols}")
        wprint(f"Full: logLik={ll_full:.1f} dev={dev_full:.1f}  gain_vs_baseline={gain_total:.1f}  iters={it_full}")

        # Drop-one models
        wprint("\nDrop-one Δdeviance contributions (unique):")
        for name, cols in blocks.items():
            cols_minus = [c for c in full_cols if c not in cols]
            ll_minus, it_m = felogit_vendor_IRLS(df, cols_minus)
            dev_minus = -2 * ll_minus
            contrib = dev_minus - dev_full  # how much worse without block
            share = contrib / gain_total if gain_total > 0 else np.nan
            wprint(f"  -{name:10s}: Δdev={contrib:.1f}  share={share:.3f}  iters={it_m}")
        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()

