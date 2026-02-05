#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'

def get_path(round_name: str) -> Path:
    if round_name == 'round1':
        return DATA_DIR / 'round1/auctions_results_all.parquet'
    if round_name == 'round2':
        return DATA_DIR / 'round2/auctions_results_r2.parquet'
    raise ValueError(round_name)

def sp_group(x: pd.DataFrame, col: str):
    if len(x) < 2:
        return np.nan
    a = x['RANKING'].values
    b = (-x[col].values)
    # If constant input, spearmanr returns nan; that's fine
    return spearmanr(a, b).correlation

def summarize(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "n=0"
    return (
        f"n={s.size} median={s.median():.4f} p10={s.quantile(0.1):.4f} "
        f"p90={s.quantile(0.9):.4f} perfect(=1)={(np.isclose(s,1.0)).mean():.4f}"
    )

def main():
    ap = argparse.ArgumentParser(description='Check auctions with PACING==0 and ranking alignment with/without pacing')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['AUCTION_ID','RANKING','QUALITY','FINAL_BID','PACING']
    df = pd.read_parquet(path, columns=cols)
    df = df.dropna(subset=cols)
    df = df[(df['RANKING']>=1) & (df['FINAL_BID']>0) & (df['QUALITY']>0)]

    total_rows = len(df)
    total_auctions = df['AUCTION_ID'].nunique()
    zero_rows = int((df['PACING'] == 0).sum())

    # Mark auctions by zero-pacing presence
    by_auc = df.groupby('AUCTION_ID')['PACING']
    any_zero_auc = by_auc.apply(lambda s: (s == 0).any())
    all_zero_auc = by_auc.apply(lambda s: (s == 0).all())
    auc_any_zero = set(any_zero_auc[any_zero_auc].index)
    auc_all_zero = set(all_zero_auc[all_zero_auc].index)

    n_auc_any = len(auc_any_zero)
    n_auc_all = len(auc_all_zero)

    df['score_nop'] = df['QUALITY'] * df['FINAL_BID']
    df['score_withp'] = df['QUALITY'] * df['FINAL_BID'] * df['PACING']

    # Restrict to auctions with any zero pacing
    df_any = df[df['AUCTION_ID'].isin(auc_any_zero)]
    g_any = df_any.groupby('AUCTION_ID', observed=True)
    idx_rank1 = g_any['RANKING'].idxmin()
    idx_s1 = g_any['score_nop'].idxmax()
    idx_s2 = g_any['score_withp'].idxmax()
    top1_nop = (idx_rank1 == idx_s1).mean() if len(idx_rank1)>0 else np.nan
    top1_withp = (idx_rank1 == idx_s2).mean() if len(idx_rank1)>0 else np.nan
    rho_nop = g_any.apply(lambda x: sp_group(x, 'score_nop'))
    rho_withp = g_any.apply(lambda x: sp_group(x, 'score_withp'))

    # For auctions where all rows have pacing==0
    df_all = df[df['AUCTION_ID'].isin(auc_all_zero)]
    g_all = df_all.groupby('AUCTION_ID', observed=True)
    idx_rank1_all = g_all['RANKING'].idxmin()
    idx_s1_all = g_all['score_nop'].idxmax()
    idx_s2_all = g_all['score_withp'].idxmax()  # ties resolved by first occurrence
    top1_nop_all = (idx_rank1_all == idx_s1_all).mean() if len(idx_rank1_all)>0 else np.nan
    top1_withp_all = (idx_rank1_all == idx_s2_all).mean() if len(idx_rank1_all)>0 else np.nan
    rho_nop_all = g_all.apply(lambda x: sp_group(x, 'score_nop'))
    rho_withp_all = g_all.apply(lambda x: sp_group(x, 'score_withp'))

    print(f"\n=== {args.round.upper()} PACING==0 analysis ===")
    print(f"rows={total_rows:,} auctions={total_auctions:,} zero_pacing_rows={zero_rows:,}")
    print(f"auctions with any pacing==0: {n_auc_any:,} | all rows pacing==0: {n_auc_all:,}")
    print("Auctions with any zero pacing:")
    print(f"  Top-1 acc: score=Q*BID {top1_nop:.4f} | score=Q*BID*PACING {top1_withp:.4f}")
    print(f"  Spearman rho: no pacing -> {summarize(rho_nop)}")
    print(f"                 with pacing -> {summarize(rho_withp)}")
    print("Auctions with all rows zero pacing:")
    print(f"  count={len(auc_all_zero)}")
    print(f"  Top-1 acc: score=Q*BID {top1_nop_all:.4f} | score=Q*BID*PACING {top1_withp_all:.4f}")
    print(f"  Spearman rho: no pacing -> {summarize(rho_nop_all)}")
    print(f"                 with pacing -> {summarize(rho_withp_all)}  (expect undefined/low due to all-zero scores)")

if __name__ == '__main__':
    main()

