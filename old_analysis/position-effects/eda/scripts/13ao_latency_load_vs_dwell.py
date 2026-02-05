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
        # demean X and z within vendor (weighted)
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

def main():
    ap = argparse.ArgumentParser(description='Load vs Dwell latency: click ~ quality + rank + price + log_load + log_dwell + interactions, vendor FE')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--placement', type=str, default=None)
    args = ap.parse_args()

    paths = get_paths(args.round)
    imp = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','OCCURRED_AT'])
    clk = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','RANKING','QUALITY','PRICE'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT','CREATED_AT']).drop_duplicates()

    clk = clk.assign(clicked=1)
    df = imp.merge(clk[['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','clicked']].drop_duplicates(), on=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'], how='left')
    df['clicked'] = df['clicked'].fillna(0).astype(int)
    df = df.merge(au, on='AUCTION_ID', how='left')
    # Ensure datetime
    if not np.issubdtype(df['OCCURRED_AT'].dtype, np.datetime64):
        df['OCCURRED_AT'] = pd.to_datetime(df['OCCURRED_AT'])
    if not np.issubdtype(df['CREATED_AT'].dtype, np.datetime64):
        df['CREATED_AT'] = pd.to_datetime(df['CREATED_AT'])
    # First impression per auction
    first_occ = df.groupby('AUCTION_ID')['OCCURRED_AT'].min().rename('first_occ')
    df = df.merge(first_occ, on='AUCTION_ID', how='left')
    df['is_first'] = (df['OCCURRED_AT'] == df['first_occ']).astype(int)
    # Load and dwell times
    df['load_s'] = (df['first_occ'] - df['CREATED_AT']).dt.total_seconds()
    df['dwell_s'] = (df['OCCURRED_AT'] - df['first_occ']).dt.total_seconds()
    # Replace non-positive with NaN for logs
    df.loc[df['load_s'] <= 0, 'load_s'] = np.nan
    df.loc[df['dwell_s'] <= 0, 'dwell_s'] = np.nan
    # Join bid features
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']), on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.rename(columns={'RANKING':'rank','QUALITY':'quality','PRICE':'price'})
    # Placement filter
    if args.placement is not None:
        df = df[df['PLACEMENT'].astype(str) == str(args.placement)]
    # Filter valid
    df = df.dropna(subset=['rank','quality','price','PLACEMENT','load_s'])
    df['log_load'] = np.log(df['load_s'])
    # Dwell term only for non-first; set to 0 for first and drop NaNs on non-first
    df['log_dwell'] = 0.0
    mask_nonfirst = (df['is_first'] == 0)
    df.loc[mask_nonfirst & (df['dwell_s']>0), 'log_dwell'] = np.log(df.loc[mask_nonfirst & (df['dwell_s']>0), 'dwell_s'])
    # Normalize dwell by average for that placement×rank (cap rank for sparsity)
    df['rank_capped'] = df['rank'].clip(upper=50).astype(int)
    base_dwell = df.loc[mask_nonfirst].groupby(['PLACEMENT','rank_capped'])['log_dwell'].mean().rename('baseline_log_dwell').reset_index()
    df = df.merge(base_dwell, on=['PLACEMENT','rank_capped'], how='left')
    df['dwell_resid'] = 0.0
    # Align indices when assigning residual
    logd = df['log_dwell'].to_numpy()
    based = df['baseline_log_dwell'].to_numpy()
    m = mask_nonfirst.to_numpy()
    resid_vals = logd[m] - based[m]
    df.loc[mask_nonfirst.to_numpy(), 'dwell_resid'] = resid_vals
    # Basic features
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]
        df = pd.concat([df, plc], axis=1)
    # Interactions for dwell with placement
    dwell_inter_cols = []
    for c in plc.columns:
        inter = f'dwell_x_{c}'
        df[inter] = df['log_dwell'] * df[c]
        dwell_inter_cols.append(inter)

    # Squared terms for both load and dwell (residualized)
    df['log_load_sq'] = df['log_load']**2
    df['dwell_resid_sq'] = df['dwell_resid']**2

    feature_cols = ['quality','rank','price','log_load','log_load_sq','dwell_resid','dwell_resid_sq'] + list(plc.columns) + dwell_inter_cols
    df = df[(df['rank']>0) & (df['quality']>0) & (df['price']>0)]
    df = df.reset_index(drop=True)

    # Fit FE-logit
    beta, alpha, vendors, inv, p, ll, iters = felogit_vendor_IRLS(df, feature_cols)

    # APEs with quadratics
    # Load APE: apply only to first impressions (is_first=1); slope = beta_load + 2*beta_load2*log_load
    beta_map = dict(zip(feature_cols, beta))
    b_load = beta_map.get('log_load', 0.0)
    b_load2 = beta_map.get('log_load_sq', 0.0)
    mask_first = (df['is_first']==1)
    slope_load = b_load + 2.0*b_load2*df['log_load'].values
    ape_load = float(np.mean(slope_load[mask_first] * p[mask_first] * (1 - p[mask_first]))) if mask_first.any() else np.nan

    # Dwell slope varies by placement via interactions, plus quadratic term (using residualized dwell)
    # Base slope component for dwell residual
    slope_dwell = np.full(len(df), beta_map.get('dwell_resid', 0.0), dtype=float)
    for c in plc.columns:
        slope_dwell += beta_map.get(f'dwell_x_{c}', 0.0) * df[c].values
    # Add quadratic contribution 2*beta_dwell2*dwell_resid
    b_dwell2 = beta_map.get('dwell_resid_sq', 0.0)
    slope_dwell += 2.0*b_dwell2*df['dwell_resid'].values

    mask_nonfirst = (df['is_first']==0)
    ape_dwell = float(np.mean(slope_dwell[mask_nonfirst] * p[mask_nonfirst] * (1 - p[mask_nonfirst]))) if mask_nonfirst.any() else np.nan
    dlog = np.log(1.1)

    # APE by placement for dwell
    ape_dwell_by_plc = {}
    effect10_dwell_by_plc = {}
    for plc_name, g in df[df['is_first']==0].groupby('PLACEMENT'):
        idx = g.index.values
        slope_g = slope_dwell[idx]
        pg = p[idx]
        ape_g = float(np.mean(slope_g * pg * (1 - pg)))
        ape_dwell_by_plc[plc_name] = ape_g
        effect10_dwell_by_plc[plc_name] = ape_g * dlog

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13ao_latency_load_vs_dwell_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Load vs Dwell latency FE-logit — {args.round}")
        wprint(f"Rows={len(df):,} CTR={df['clicked'].mean()*100:.3f}% Vendors={df['VENDOR_ID'].nunique():,} iters={iters}")
        if args.placement is not None:
            wprint(f"Restricted to PLACEMENT={args.placement}")
    wprint(f"Features: {feature_cols}")
    wprint("Coefficients (beta):")
    for c, b in zip(feature_cols, beta):
        wprint(f"  {c:>16}: {float(b): .6f}")
        wprint(f"logLik={ll:.1f}")
        wprint("\nAPE for log_load (first impressions only):")
        wprint(f"  APE={ape_load:.6f}; +10% effect={ape_load*dlog if not np.isnan(ape_load) else np.nan:.6f}")
        wprint("\nAPE for log_dwell (subsequent impressions):")
        wprint(f"  Overall APE={ape_dwell:.6f}; +10% effect={ape_dwell*dlog if not np.isnan(ape_dwell) else np.nan:.6f}")
        if ape_dwell_by_plc:
            wprint("  By placement:")
            for k in sorted(ape_dwell_by_plc.keys()):
                wprint(f"    {k}: APE={ape_dwell_by_plc[k]:.6f}; +10%Δ={effect10_dwell_by_plc[k]:.6f}")
        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()
