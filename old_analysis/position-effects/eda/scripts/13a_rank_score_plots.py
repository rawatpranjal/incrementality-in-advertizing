#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "0_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

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

def compute_deciles(df: pd.DataFrame, score_name: str, score: pd.Series):
    dec = pd.qcut(score.rank(method='first'), 10, labels=False)
    tmp = pd.DataFrame({'decile': dec, 'rank': df['RANKING'], 'score': score})
    agg = tmp.groupby('decile').agg(
        rank_median=('rank','median'),
        top10_share=('rank', lambda x: (x<=10).mean()),
        top20_share=('rank', lambda x: (x<=20).mean()),
        score_median=('score','median')
    ).reset_index().sort_values('decile')
    agg['score_name'] = score_name
    return agg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', choices=['round1','round2'], required=True)
    args = parser.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','RANKING','FINAL_BID','QUALITY','PACING'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    df = ar.merge(au, on='AUCTION_ID', how='left')

    placements = sorted(df['PLACEMENT'].dropna().unique())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for plc in placements:
        sub = df[df['PLACEMENT']==plc].copy()
        if len(sub)==0:
            continue
        scores = {
            'bid*quality': sub['FINAL_BID']*sub['QUALITY'],
            'pacing*bid*quality': sub['FINAL_BID']*sub['QUALITY']*sub['PACING'],
        }
        aggs = [compute_deciles(sub, k, v) for k,v in scores.items()]
        data = pd.concat(aggs, ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
        for name, g in data.groupby('score_name'):
            axes[0].plot(g['decile'], g['rank_median'], marker='o', label=name)
        axes[0].invert_yaxis()
        axes[0].set_title(f'Median rank vs score decile (placement {plc})')
        axes[0].set_xlabel('Score decile (0=low, 9=high)')
        axes[0].set_ylabel('Median rank (lower is better)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        for name, g in data.groupby('score_name'):
            axes[1].plot(g['decile'], g['top10_share']*100, marker='o', label=name)
        axes[1].set_title(f'Top-10 share vs score decile (placement {plc})')
        axes[1].set_xlabel('Score decile (0=low, 9=high)')
        axes[1].set_ylabel('% of rows with rank ≤ 10')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        out_png = RESULTS_DIR / f"13a_rank_vs_score_{args.round}_P{plc}.png"
        fig.suptitle(f"Rank vs Score Deciles ({args.round}) - Placement {plc}", fontsize=12)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {out_png}")

if __name__ == '__main__':
    main()
