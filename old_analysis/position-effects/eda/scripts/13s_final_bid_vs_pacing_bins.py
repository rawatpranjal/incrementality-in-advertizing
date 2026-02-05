#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'

def get_path(round_name: str) -> Path:
    if round_name == 'round1':
        return DATA_DIR / 'round1/auctions_results_all.parquet'
    if round_name == 'round2':
        return DATA_DIR / 'round2/auctions_results_r2.parquet'
    raise ValueError(round_name)

def summarize(y: pd.Series) -> str:
    y = y.dropna()
    if y.empty:
        return "n=0"
    return (
        f"n={len(y):,} median={y.quantile(0.5):.4f} p10={y.quantile(0.1):.4f} "
        f"p90={y.quantile(0.9):.4f} mean={y.mean():.4f}"
    )

def main():
    ap = argparse.ArgumentParser(description='FINAL_BID distribution across pacing bins and zeros check')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['FINAL_BID','PACING']
    df = pd.read_parquet(path, columns=cols)
    rows = len(df)
    zero_bids = int((df['FINAL_BID'] <= 0).sum())
    zero_pacing = int((df['PACING'] == 0).sum())

    bins = pd.IntervalIndex.from_tuples([
        (0.0, 0.5),
        (0.5, 0.8),
        (0.8, 0.95),
        (0.95, 0.9999),
        (0.9999, 1.0),
    ], closed='right')
    df = df[(df['PACING']>0) & (df['FINAL_BID'].notna())]
    df['p_bin'] = pd.cut(df['PACING'], bins)

    print(f"\n=== {args.round.upper()} FINAL_BID vs PACING bins ===")
    print(f"rows={rows:,} zero_bids={zero_bids:,} pacing_zero_rows={zero_pacing:,}")
    for b, g in df.groupby('p_bin', observed=True):
        if pd.isna(b):
            continue
        share = len(g) / len(df) if len(df) else np.nan
        print(f"pacing in {b}: share={share:6.2%} | {summarize(g['FINAL_BID'])}")

if __name__ == '__main__':
    main()
