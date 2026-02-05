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
    ap = argparse.ArgumentParser(description='Scroll latency: residual log-latency vs placement×rank baseline, with interactions')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
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
    if not np.issubdtype(df['OCCURRED_AT'].dtype, np.datetime64):
        df['OCCURRED_AT'] = pd.to_datetime(df['OCCURRED_AT'])
    if not np.issubdtype(df['CREATED_AT'].dtype, np.datetime64):
        df['CREATED_AT'] = pd.to_datetime(df['CREATED_AT'])
    df['latency_s'] = (df['OCCURRED_AT'] - df['CREATED_AT']).dt.total_seconds()
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']), on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.rename(columns={'RANKING':'rank','QUALITY':'quality','PRICE':'price'})
    # Filter and index reset
    df = df.dropna(subset=['rank','quality','price','PLACEMENT','latency_s'])
    df = df[(df['rank']>0) & (df['quality']>0) & (df['price']>0) & (df['latency_s']>0)]
    df = df.reset_index(drop=True)

    # Build baseline log-latency by placement×rank (cap rank to avoid sparse tails)
    df['log_latency'] = np.log(df['latency_s'])
    df['rank_capped'] = df['rank'].clip(upper=50).astype(int)
    baseline = df.groupby(['PLACEMENT','rank_capped'])['log_latency'].mean().rename('baseline_log_latency').reset_index()
    df = df.merge(baseline, on=['PLACEMENT','rank_capped'], how='left')
    # Scroll residual: excess log-latency vs typical for this placement×rank
    df['scroll_log_resid'] = df['log_latency'] - df['baseline_log_latency']

    # Placement FE and interactions with scroll residual
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    base_plc_cols = []
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]
        base_plc_cols = list(plc.columns)
    # attach placement dummies to df
    if plc.shape[1] > 0:
        df = pd.concat([df, plc], axis=1)
    for c in base_plc_cols:
        df[f'scroll_x_{c}'] = df['scroll_log_resid'] * df[c]

    feature_cols = ['quality','rank','price','scroll_log_resid'] + base_plc_cols + [f'scroll_x_{c}' for c in base_plc_cols]

    # Fit FE-logit with vendor FE
    beta, alpha, vendors, inv, p, ll, iters = felogit_vendor_IRLS(df, feature_cols)

    # APE overall for scroll residual: mean(beta_eff * p(1-p)), where beta_eff depends on placement
    # Compute placement-specific slopes
    beta_map = dict(zip(feature_cols, beta))
    slopes = {}
    # Baseline placement (first level): slope = beta(scroll_log_resid)
    base_slope = beta_map.get('scroll_log_resid', 0.0)
    slopes['BASE'] = base_slope
    for c in base_plc_cols:
        slopes[c] = base_slope + beta_map.get(f'scroll_x_{c}', 0.0)

    # APE overall and by placement using pointwise slopes
    ape_overall = float(np.mean([slopes.get('BASE',0.0) * pi * (1-pi) for pi in p]))
    # Compute per-row slope based on placement dummy
    plc_names = df['PLACEMENT'].values
    per_row_slope = np.full(len(df), base_slope, dtype=float)
    for c in base_plc_cols:
        mask = (df[c].values == 1)
        per_row_slope[mask] = slopes[c]
    ape_by_plc = df.groupby('PLACEMENT').apply(lambda g: float(np.mean(per_row_slope[g.index] * p[g.index] * (1 - p[g.index])))).to_dict()

    # +10% latency effect (Δlog=ln 1.1)
    dlog = np.log(1.1)
    effect10_overall = ape_overall * dlog
    effect10_by_plc = {k: v * dlog for k, v in ape_by_plc.items()}

    # Deciles of scroll residual vs CTR
    def decile_table(series: pd.Series, y: pd.Series, label: str) -> pd.DataFrame:
        q = np.linspace(0,1,11)
        cuts = series.quantile(q)
        bins = np.unique(cuts.values)
        if bins.size < 3:
            return pd.DataFrame()
        db = pd.cut(series, bins=bins, include_lowest=True)
        tab = pd.DataFrame({'bin': db, 'y': y, 'x': series}).groupby('bin').agg(n=('y','size'), ctr=('y','mean'), mean_x=('x','mean')).reset_index()
        tab.insert(0, 'segment', label)
        return tab

    overall_dec = decile_table(df['scroll_log_resid'], df['clicked'], 'overall')
    plc_dec = []
    for plc_name, g in df.groupby('PLACEMENT'):
        t = decile_table(g['scroll_log_resid'], g['clicked'], f'plc={plc_name}')
        if not t.empty:
            plc_dec.append(t)
    dec_all = pd.concat([overall_dec] + plc_dec, ignore_index=True) if overall_dec is not None else pd.DataFrame()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13an_scroll_latency_penalty_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Scroll latency FE-logit — {args.round}")
        wprint(f"Rows={len(df):,} CTR={df['clicked'].mean()*100:.3f}% Vendors={df['VENDOR_ID'].nunique():,} iters={iters}")
        wprint(f"Features: {feature_cols}")
        wprint("Coefficients (beta):")
        for c, b in zip(feature_cols, beta):
            wprint(f"  {c:>18}: {float(b): .6f}")
        wprint(f"logLik={ll:.1f}")
        wprint("\nScroll residual APE (per unit Δlog latency residual):")
        wprint(f"  Overall APE: {ape_overall:.6f}; +10% latency effect: {effect10_overall:.6f}")
        wprint("  By placement:")
        for plc_name, val in ape_by_plc.items():
            wprint(f"    {plc_name}: APE={val:.6f}; +10%Δ={effect10_by_plc[plc_name]:.6f}")
        if not dec_all.empty:
            wprint("\nScroll residual deciles vs CTR:")
            wprint(dec_all.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()
