#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

def main():
    parser = argparse.ArgumentParser(description='Scatter: rank vs (bid*quality) per placement')
    parser.add_argument('--round', choices=['round1','round2'], required=True)
    parser.add_argument('--sample', type=int, default=50000, help='max points per placement')
    args = parser.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','RANKING','FINAL_BID','QUALITY'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    df = ar.merge(au, on='AUCTION_ID', how='left').dropna(subset=['PLACEMENT'])
    df['score'] = df['FINAL_BID'] * df['QUALITY']

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for plc in sorted(df['PLACEMENT'].unique()):
        sub = df[df['PLACEMENT']==plc]
        if len(sub) == 0:
            continue
        # sample for plotting
        if len(sub) > args.sample:
            plot_df = sub.sample(n=args.sample, random_state=42)
        else:
            plot_df = sub
        X = plot_df['score'].values.reshape(-1,1)
        y = plot_df['RANKING'].values

        # Fit linear regression (rank ~ score)
        lin = LinearRegression().fit(X, y)
        r2 = lin.score(X, y)

        # Create scatter
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(plot_df['score'], plot_df['RANKING'], s=4, alpha=0.2, edgecolors='none')
        # Regression line over the middle score range
        xs = np.linspace(np.percentile(plot_df['score'], 1), np.percentile(plot_df['score'], 99), 100)
        ys = lin.predict(xs.reshape(-1,1))
        ax.plot(xs, ys, color='tomato', linewidth=2, label=f'lin fit (R^2={r2:.3f})')
        ax.set_title(f'Placement {plc}: Rank vs (bid*quality) [{args.round}]')
        ax.set_xlabel('score = FINAL_BID * QUALITY')
        ax.set_ylabel('RANKING (lower is better)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend()

        out = RESULTS_DIR / f"13b_rank_score_scatter_{args.round}_P{plc}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'Saved plot: {out}')

if __name__ == '__main__':
    main()

