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

def compute_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # score & rank deciles within auction
    g = df.groupby('AUCTION_ID', group_keys=False)
    df = g.apply(lambda d: d.assign(_ord_s=d['score'].rank(method='first'), _n=len(d)))
    df['sdec'] = np.floor((df['_ord_s']-1)/np.maximum(df['_n']-1,1)*10).astype(int)
    df.loc[df['sdec']>9,'sdec']=9
    df = df.groupby('AUCTION_ID', group_keys=False).apply(lambda d: d.assign(_ord_r=d['RANKING'].rank(method='first')))
    df['rdec'] = np.floor((df['_ord_r']-1)/np.maximum(df['_n']-1,1)*10).astype(int)
    df.loc[df['rdec']>9,'rdec']=9
    piv = df.pivot_table(index='sdec', columns='rdec', values='AUCTION_ID', aggfunc='size', fill_value=0)
    row_sums = piv.sum(axis=1).replace(0,1)
    piv_pct = (piv.div(row_sums, axis=0)*100)
    return piv_pct

def plot_heatmap(mat: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(mat.values, origin='upper', cmap='Blues', vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel('rank decile within auction (0=best)')
    ax.set_ylabel('score decile within auction (0=low)')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% of rows in rank decile')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Within-auction score vs rank decile heatmaps by placement')
    parser.add_argument('--round', choices=['round1','round2'], required=True)
    args = parser.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','RANKING','FINAL_BID','QUALITY'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    df = ar.merge(au, on='AUCTION_ID', how='left').dropna(subset=['PLACEMENT'])
    df['score'] = df['FINAL_BID'] * df['QUALITY']

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for plc in sorted(df['PLACEMENT'].unique()):
        sub = df[df['PLACEMENT']==plc].copy()
        if len(sub)==0:
            continue
        mat = compute_matrix(sub)
        out = RESULTS_DIR / f"13c_within_auction_heatmap_{args.round}_P{plc}.png"
        plot_heatmap(mat, f'Placement {plc} — {args.round}', out)
        print(f'Saved heatmap: {out}')

if __name__ == '__main__':
    main()

