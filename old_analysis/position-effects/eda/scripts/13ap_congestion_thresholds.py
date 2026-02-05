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
            'auctions_users': DATA_DIR / 'round1/auctions_users_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
        }
    raise ValueError(round_name)

def ols_with_const(y: np.ndarray, x: np.ndarray):
    X = np.column_stack([np.ones_like(x, dtype=float), x.astype(float)])
    y = y.astype(float)
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.pinv(XtX) @ Xty
    yhat = X @ beta
    resid = y - yhat
    sst = float(np.sum((y - y.mean())**2))
    sse = float(np.sum(resid**2))
    r2 = 1.0 - sse/sst if sst > 0 else np.nan
    # Classic homoskedastic SE
    n, k = X.shape
    sigma2 = sse / max(n - k, 1)
    cov = sigma2 * np.linalg.pinv(XtX)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, r2, n

def summarize(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(float)
    if s.empty:
        idx = ['count','mean','std','min','p10','p25','p50','p75','p90','max']
        return pd.Series({k: np.nan for k in idx})
    return pd.Series({
        'count': int(s.size),
        'mean': s.mean(),
        'std': s.std(ddof=0),
        'min': s.min(),
        'p10': s.quantile(0.10),
        'p25': s.quantile(0.25),
        'p50': s.quantile(0.50),
        'p75': s.quantile(0.75),
        'p90': s.quantile(0.90),
        'max': s.max(),
    })

def main():
    ap = argparse.ArgumentParser(description='Auction congestion: winner quality and runner-up threshold vs bidder count (elasticities + tables)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','VENDOR_ID','PRODUCT_ID','RANKING','QUALITY','FINAL_BID'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()

    # Clean and score
    ar = ar.dropna(subset=['AUCTION_ID','RANKING','QUALITY','FINAL_BID'])
    ar = ar[(ar['QUALITY']>0) & (ar['FINAL_BID']>0)]
    ar['score'] = ar['QUALITY'] * ar['FINAL_BID']
    # Depth
    depth = ar.groupby('AUCTION_ID').size().rename('bidder_count')
    # Top-2 by score
    ars = ar.sort_values(['AUCTION_ID','score'], ascending=[True, False])
    ars['pos'] = ars.groupby('AUCTION_ID').cumcount() + 1
    win = ars[ars['pos']==1][['AUCTION_ID','QUALITY','FINAL_BID','score']].rename(columns={'QUALITY':'Q1','FINAL_BID':'B1','score':'S1'})
    run = ars[ars['pos']==2][['AUCTION_ID','QUALITY','FINAL_BID','score']].rename(columns={'QUALITY':'Q2','FINAL_BID':'B2','score':'S2'})
    top2 = win.merge(run, on='AUCTION_ID', how='inner')
    top2 = top2.merge(depth, on='AUCTION_ID', how='left').merge(au, on='AUCTION_ID', how='left')
    # Thresholds
    top2 = top2[(top2['S2']>0) & (top2['Q1']>0)]
    top2['threshold_bid'] = top2['S2'] / top2['Q1']
    # Log-transforms
    top2['log_count'] = np.log(top2['bidder_count'].astype(float))
    top2['log_Q1'] = np.log(top2['Q1'].astype(float))
    top2['log_S2'] = np.log(top2['S2'].astype(float))
    top2['log_thr_bid'] = np.log(top2['threshold_bid'].astype(float))

    # Depth bins
    bins = pd.IntervalIndex.from_tuples([(2,5),(5,10),(10,20),(20,50),(50,999999)], closed='right')
    top2['depth_bin'] = pd.cut(top2['bidder_count'], bins)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13ap_congestion_thresholds_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Auction congestion and thresholds — {args.round}")
        wprint(f"Auctions with >=2 valid bids: {top2['AUCTION_ID'].nunique():,}")
        wprint("\nOverall summaries:")
        wprint("bidder_count:" )
        wprint(summarize(top2['bidder_count']).round(4).to_string())
        wprint("Q1 (winner quality):")
        wprint(summarize(top2['Q1']).round(6).to_string())
        wprint("S2 (runner-up score):")
        wprint(summarize(top2['S2']).round(6).to_string())
        wprint("threshold_bid = S2/Q1:")
        wprint(summarize(top2['threshold_bid']).round(6).to_string())

        # Regressions by placement
        wprint("\nElasticities (log-log OLS) by placement:")
        for plc, g in top2.groupby('PLACEMENT', dropna=False):
            if len(g) < 50:
                continue
            bQ, seQ, r2Q, nQ = ols_with_const(g['log_Q1'].values, g['log_count'].values)
            bS, seS, r2S, nS = ols_with_const(g['log_S2'].values, g['log_count'].values)
            bT, seT, r2T, nT = ols_with_const(g['log_thr_bid'].values, g['log_count'].values)
            wprint(f"PLACEMENT={plc}")
            wprint(f"  log(Q1) ~ log(count): beta={bQ[1]:.4f} (se={seQ[1]:.4f}) R^2={r2Q:.3f} n={nQ}")
            wprint(f"  log(S2) ~ log(count): beta={bS[1]:.4f} (se={seS[1]:.4f}) R^2={r2S:.3f} n={nS}")
            wprint(f"  log(thr_bid) ~ log(count): beta={bT[1]:.4f} (se={seT[1]:.4f}) R^2={r2T:.3f} n={nT}")

        # Depth-bin tables by placement
        wprint("\nDepth-bin tables (median by bin) by placement:")
        for plc, g in top2.groupby('PLACEMENT', dropna=False):
            wprint(f"PLACEMENT={plc}")
            tab = g.groupby('depth_bin').agg(
                n=('AUCTION_ID','nunique'),
                med_count=('bidder_count','median'),
                med_Q1=('Q1','median'),
                med_S2=('S2','median'),
                med_thr_bid=('threshold_bid','median'),
            ).reset_index()
            wprint(tab.to_string(index=False))

        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()

