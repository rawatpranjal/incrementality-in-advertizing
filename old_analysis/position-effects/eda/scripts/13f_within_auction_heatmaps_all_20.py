#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def compute_bins(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    # vectorized within-auction score and rank bins (k-tiles)
    df['_ord_s'] = df.groupby('AUCTION_ID')['score'].rank(method='first')
    n = df.groupby('AUCTION_ID')['score'].transform('size')
    denom = (n - 1).clip(lower=1)
    df['sbin'] = np.floor((df['_ord_s'] - 1) / denom * k).astype(int)
    df.loc[df['sbin']>=k, 'sbin'] = k-1
    df['_ord_r'] = df.groupby('AUCTION_ID')['RANKING'].rank(method='first')
    df['rbin'] = np.floor((df['_ord_r'] - 1) / denom * k).astype(int)
    df.loc[df['rbin']>=k, 'rbin'] = k-1
    return df

def pivot_pct(sub: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    piv = sub.pivot_table(index='sbin', columns='rbin', values='AUCTION_ID', aggfunc='size', fill_value=0)
    piv = piv.reindex(index=range(k), columns=range(k), fill_value=0)
    row_sums = piv.sum(axis=1).replace(0,1)
    return piv.div(row_sums, axis=0) * 100.0

def main():
    parser = argparse.ArgumentParser(description='One figure: within-auction heatmaps (20-tiles) for all placements (no pacing)')
    parser.add_argument('--round', choices=['round1','round2'], required=True)
    args = parser.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','RANKING','FINAL_BID','QUALITY'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    df = ar.merge(au, on='AUCTION_ID', how='left').dropna(subset=['PLACEMENT'])
    df['score'] = df['FINAL_BID'] * df['QUALITY']
    df = compute_bins(df, k=20)

    placements = sorted(df['PLACEMENT'].dropna().unique())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12,10), constrained_layout=True)
    axes = axes.ravel()
    for idx, plc in enumerate(placements[:4]):
        mat = pivot_pct(df[df['PLACEMENT']==plc], k=20)
        ax = axes[idx]
        im = ax.imshow(mat.values, origin='upper', cmap='Blues', vmin=0, vmax=100)
        ax.set_title(f'Placement {plc}')
        ax.set_xlabel('rank 20-tiles (0=best)')
        ax.set_ylabel('score 20-tiles (0=low)')
        ax.set_xticks(range(0,20,2)); ax.set_yticks(range(0,20,2))
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85)
    cbar.set_label('% of rows in rank bin')
    fig.suptitle(f'Within-auction: score vs rank (20-tiles), round={args.round}, score=bid*quality', fontsize=12)
    out = RESULTS_DIR / f"13f_within_auction_heatmaps20_all_{args.round}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved figure: {out}')

if __name__ == '__main__':
    main()

