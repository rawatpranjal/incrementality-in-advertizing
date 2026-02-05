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
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
        }
    raise ValueError(round_name)

def compute_deciles(df: pd.DataFrame) -> pd.DataFrame:
    # vectorized within-auction score and rank deciles
    df['_ord_s'] = df.groupby('AUCTION_ID')['score'].rank(method='first')
    n = df.groupby('AUCTION_ID')['score'].transform('size')
    denom = (n - 1).clip(lower=1)
    df['sdec'] = np.floor((df['_ord_s'] - 1) / denom * 10).astype(int)
    df.loc[df['sdec']>9, 'sdec'] = 9

    df['_ord_r'] = df.groupby('AUCTION_ID')['RANKING'].rank(method='first')
    df['rdec'] = np.floor((df['_ord_r'] - 1) / denom * 10).astype(int)
    df.loc[df['rdec']>9, 'rdec'] = 9
    df['au_size'] = n
    return df

def per_auction_corr(sub: pd.DataFrame):
    # Compute Pearson and Spearman between score-decile and inverted rank-decile within auction
    # Spearman by ranking the deciles again
    s = sub['sdec'].values.astype(float)
    r = (9 - sub['rdec'].values).astype(float)  # higher is better
    if np.std(s) == 0 or np.std(r) == 0:
        return np.nan, np.nan
    pear = np.corrcoef(s, r)[0,1]
    sr = pd.Series(s).rank(method='average').values
    rr = pd.Series(r).rank(method='average').values
    if np.std(sr) == 0 or np.std(rr) == 0:
        spear = np.nan
    else:
        spear = np.corrcoef(sr, rr)[0,1]
    return pear, spear

def summarize(corrs: pd.Series) -> dict:
    vals = corrs.dropna().values
    if len(vals) == 0:
        return {}
    qs = [0.1,0.25,0.5,0.75,0.9,0.99]
    pct = np.quantile(vals, qs)
    return {
        'count': len(vals),
        'mean': float(np.mean(vals)),
        'p10': float(pct[0]),
        'p25': float(pct[1]),
        'p50': float(pct[2]),
        'p75': float(pct[3]),
        'p90': float(pct[4]),
        'p99': float(pct[5]),
    }

def main():
    parser = argparse.ArgumentParser(description='Auction-specific score vs rank decile correlation (by placement)')
    parser.add_argument('--round', choices=['round1','round2'], required=True)
    parser.add_argument('--min_size', type=int, default=10, help='min bidders per auction')
    args = parser.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','RANKING','FINAL_BID','QUALITY'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    df = ar.merge(au, on='AUCTION_ID', how='left').dropna(subset=['PLACEMENT'])
    df['score'] = df['FINAL_BID'] * df['QUALITY']
    df = compute_deciles(df)
    # filter auctions with enough bidders
    df = df[df['au_size'] >= args.min_size]

    rows = []
    for plc, gpl in df.groupby('PLACEMENT'):
        # compute correlations per auction
        pa = gpl.groupby('AUCTION_ID').apply(lambda d: pd.Series(per_auction_corr(d), index=['pearson','spearman']))
        pstats = summarize(pa['pearson'])
        sstats = summarize(pa['spearman'])
        print(f"\nPlacement {plc} ({len(pa)} auctions with n>={args.min_size}):")
        if pstats:
            print(f"  Pearson  mean={pstats['mean']:.3f}  p10={pstats['p10']:.3f}  p25={pstats['p25']:.3f}  median={pstats['p50']:.3f}  p75={pstats['p75']:.3f}  p90={pstats['p90']:.3f}")
        if sstats:
            print(f"  Spearman mean={sstats['mean']:.3f}  p10={sstats['p10']:.3f}  p25={sstats['p25']:.3f}  median={sstats['p50']:.3f}  p75={sstats['p75']:.3f}  p90={sstats['p90']:.3f}")

if __name__ == '__main__':
    main()

