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
    ap = argparse.ArgumentParser(description='Latency penalty: click ~ quality + rank + price + log(latency) [+ log(latency)^2] + placement FE + vendor FE')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--placement', type=str, default=None, help='Restrict to a single placement value (e.g., "1")')
    ap.add_argument('--first_only', action='store_true', help='Keep only the first impression per AUCTION_ID (earliest OCCURRED_AT)')
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
    # Compute latency
    # Ensure datetime
    if not np.issubdtype(df['OCCURRED_AT'].dtype, np.datetime64):
        df['OCCURRED_AT'] = pd.to_datetime(df['OCCURRED_AT'])
    if not np.issubdtype(df['CREATED_AT'].dtype, np.datetime64):
        df['CREATED_AT'] = pd.to_datetime(df['CREATED_AT'])
    df['latency_s'] = (df['OCCURRED_AT'] - df['CREATED_AT']).dt.total_seconds()
    # If initial-only requested: keep earliest impression per auction
    if args.first_only:
        # ensure OCCURRED_AT is datetime
        if not np.issubdtype(df['OCCURRED_AT'].dtype, np.datetime64):
            df['OCCURRED_AT'] = pd.to_datetime(df['OCCURRED_AT'])
        idx = df.groupby('AUCTION_ID')['OCCURRED_AT'].idxmin()
        df = df.loc[idx].copy().reset_index(drop=True)

    # Join bid features for the impressed bid
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']), on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.rename(columns={'RANKING':'rank','QUALITY':'quality','PRICE':'price'})
    # Filter valid
    df = df.dropna(subset=['rank','quality','price','PLACEMENT','latency_s'])
    if args.placement is not None:
        df = df[df['PLACEMENT'].astype(str) == str(args.placement)]
    df = df[(df['rank']>0) & (df['quality']>0) & (df['price']>0) & (df['latency_s']>0)]
    df = df.reset_index(drop=True)
    # Build features
    df['log_latency'] = np.log(df['latency_s'])
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]  # drop first to avoid collinearity
    df = pd.concat([df, plc], axis=1)
    df['log_latency_sq'] = df['log_latency']**2
    feature_cols = ['quality','rank','price','log_latency','log_latency_sq'] + list(plc.columns)

    # Fit FE-logit with vendor FE absorption
    beta, alpha, vendors, inv, p, ll, iters = felogit_vendor_IRLS(df, feature_cols)
    # Extract latency coefficient index
    idx_lat = feature_cols.index('log_latency')
    idx_lat2 = feature_cols.index('log_latency_sq')
    beta_lat = float(beta[idx_lat])
    beta_lat2 = float(beta[idx_lat2])
    # APE overall with quadratic: mean( (beta1 + 2*beta2*log_latency) * p(1-p) )
    slope_overall = beta_lat + 2.0*beta_lat2*df['log_latency'].values
    ape = float(np.mean(slope_overall * p * (1 - p)))
    # Convert to effect of +10% latency (Δlog = log(1.1))
    tenpct = float(ape * np.log(1.1))

    # APE by placement
    by_plc = []
    for plc_name, g in df.groupby('PLACEMENT'):
        idx = g.index.values
        pg = p[idx]
        slope_g = beta_lat + 2.0*beta_lat2*g['log_latency'].values
        ape_g = float(np.mean(slope_g * pg * (1 - pg)))
        tenpct_g = float(ape_g * np.log(1.1))
        by_plc.append((plc_name, ape_g, tenpct_g, len(g), float(g['clicked'].mean())))

    # Latency deciles vs CTR (overall and by placement)
    def decile_table(frame: pd.DataFrame, label: str):
        q = np.linspace(0, 1, 11)
        cuts = frame['latency_s'].quantile(q)
        bins = np.unique(cuts.values)
        if bins.size < 3:
            return pd.DataFrame()
        # Ensure strictly increasing
        bins = np.unique(bins)
        # Add a tiny jitter to ensure bin edges are unique
        bins[0] = max(1e-6, bins[0])
        db = pd.cut(frame['latency_s'], bins=bins, include_lowest=True)
        tab = frame.groupby(db).agg(n=('clicked','size'), ctr=('clicked','mean'), mean_latency=('latency_s','mean')).reset_index()
        tab.insert(0, 'segment', label)
        return tab

    overall_dec = decile_table(df, 'overall')
    plc_dec = []
    for plc_name, g in df.groupby('PLACEMENT'):
        t = decile_table(g, f'plc={plc_name}')
        if not t.empty:
            plc_dec.append(t)
    dec_all = pd.concat([overall_dec] + plc_dec, ignore_index=True) if overall_dec is not None else pd.DataFrame()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13am_latency_penalty_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Latency penalty FE-logit — {args.round}")
        wprint(f"Rows={len(df):,} CTR={df['clicked'].mean()*100:.3f}% Vendors={df['VENDOR_ID'].nunique():,} iters={iters}")
        if args.placement is not None:
            wprint(f"Restricted to PLACEMENT={args.placement}")
        if args.first_only:
            wprint("Restricted to first impression per AUCTION_ID (earliest OCCURRED_AT)")
        wprint(f"Features: {feature_cols}")
        wprint("Coefficients (beta):")
        for c, b in zip(feature_cols, beta):
            wprint(f"  {c:>12}: {float(b): .6f}")
        wprint(f"logLik={ll:.1f}")
        wprint("\nAverage Partial Effect (APE) for log_latency (per unit Δlog latency):")
        wprint(f"  Overall APE: {ape:.6f} (probability points per unit log latency)")
        wprint(f"  Effect of +10% latency: {tenpct:.6f} (absolute Δp)")
        wprint("\nAPE by placement:")
        for plc_name, ape_g, tenpct_g, n_g, ctr_g in by_plc:
            wprint(f"  {plc_name}: APE={ape_g:.6f}  +10%Δ={tenpct_g:.6f}  n={n_g:,}  CTR={ctr_g*100:.2f}%")
        if not dec_all.empty:
            wprint("\nLatency deciles vs CTR:")
            wprint(dec_all.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()
