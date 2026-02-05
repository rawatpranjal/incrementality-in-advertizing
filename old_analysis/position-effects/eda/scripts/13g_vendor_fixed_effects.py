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

def r2_from_group_means(y: np.ndarray, yhat: np.ndarray) -> float:
    sst = np.sum((y - y.mean())**2)
    sse = np.sum((y - yhat)**2)
    return 1.0 - sse/sst if sst > 0 else np.nan

def main():
    ap = argparse.ArgumentParser(description='Vendor fixed effects: y = log(bid) - log(price)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['VENDOR_ID','FINAL_BID','PRICE','AUCTION_ID','RANKING']
    df = pd.read_parquet(path, columns=cols)
    df = df[(df['FINAL_BID']>0) & (df['PRICE']>0)].copy()
    df['y'] = np.log(df['FINAL_BID']) - np.log(df['PRICE'])  # ~ -log(tROAS_vendor)

    # Overall vendor FE as group means
    vend_mean = df.groupby('VENDOR_ID')['y'].mean()
    yhat = df['VENDOR_ID'].map(vend_mean).values
    r2 = r2_from_group_means(df['y'].values, yhat)

    # Distribution of FE and implied target ROAS per vendor
    fe = vend_mean.rename('alpha_vendor')
    implied_troas = np.exp(-fe)  # since y ≈ -log(troas)
    desc_fe = fe.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(4)
    desc_t = implied_troas.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).round(4)

    print(f"\n=== Vendor fixed effects on y=log(bid)-log(price) ({args.round}) ===")
    print(f"rows={len(df):,}  vendors={fe.size:,}  R^2(group means)={r2:.3f}")
    print("\nalpha_vendor = mean_y per vendor (approx -log(tROAS_vendor))")
    print(desc_fe)
    print("\nimplied tROAS per vendor = exp(-alpha_vendor)")
    print(desc_t)

    # Show top/bottom vendors by implied tROAS (with min support)
    counts = df['VENDOR_ID'].value_counts()
    fe_df = pd.DataFrame({'alpha_vendor': fe, 'n_rows': counts}).dropna()
    fe_df['implied_troas'] = np.exp(-fe_df['alpha_vendor'])
    fe_df = fe_df[fe_df['n_rows'] >= 1000]  # minimum support
    top = fe_df.nsmallest(10, 'implied_troas')  # lowest target roas (highest bids for given pcvr*price)
    bot = fe_df.nlargest(10, 'implied_troas')   # highest target roas (lowest bids)
    print("\nTop-10 vendors by aggressiveness (lowest implied tROAS), n>=1000:")
    print(top[['implied_troas','n_rows']])
    print("\nTop-10 vendors by conservativeness (highest implied tROAS), n>=1000:")
    print(bot[['implied_troas','n_rows']])

if __name__ == '__main__':
    main()

