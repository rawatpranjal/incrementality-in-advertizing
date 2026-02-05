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

def r2_from_group_means(y: np.ndarray, yhat: np.ndarray) -> float:
    sst = np.sum((y - y.mean())**2)
    sse = np.sum((y - yhat)**2)
    return 1.0 - sse/sst if sst > 0 else np.nan

def main():
    ap = argparse.ArgumentParser(description='Vendor fixed effects per placement: y = log(bid) - log(price)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--min_rows', type=int, default=1000, help='min vendor rows per placement to list extremes')
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','VENDOR_ID','FINAL_BID','PRICE'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    df = ar.merge(au, on='AUCTION_ID', how='left').dropna(subset=['PLACEMENT'])
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0)].copy()
    df['y'] = np.log(df['FINAL_BID']) - np.log(df['PRICE'])

    for plc, g in df.groupby('PLACEMENT'):
        vend_mean = g.groupby('VENDOR_ID')['y'].mean()
        yhat = g['VENDOR_ID'].map(vend_mean).values
        r2 = r2_from_group_means(g['y'].values, yhat)
        fe = vend_mean.rename('alpha_vendor')
        implied_t = np.exp(-fe)
        desc_fe = fe.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(4)
        desc_t = implied_t.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(4)
        print(f"\nPlacement {plc} — rows={len(g):,}, vendors={len(fe):,}, R^2(group means)={r2:.3f}")
        print("alpha_vendor (≈ -log(tROAS_vendor|placement)):")
        print(desc_fe)
        print("implied tROAS per vendor (placement):")
        print(desc_t)

        counts = g['VENDOR_ID'].value_counts()
        fe_df = pd.DataFrame({'alpha_vendor': fe, 'n_rows': counts}).dropna()
        fe_df['implied_troas'] = np.exp(-fe_df['alpha_vendor'])
        fe_df = fe_df[fe_df['n_rows'] >= args.min_rows]
        if len(fe_df) > 0:
            top = fe_df.nsmallest(5, 'implied_troas')
            bot = fe_df.nlargest(5, 'implied_troas')
            print("  Most aggressive (lowest implied tROAS):")
            print(top[['implied_troas','n_rows']])
            print("  Most conservative (highest implied tROAS):")
            print(bot[['implied_troas','n_rows']])

if __name__ == '__main__':
    main()

