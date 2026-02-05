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
    ap = argparse.ArgumentParser(description='Infer payment rule via top-2 score ratio S1/S2 (winner vs runner-up)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','VENDOR_ID','PRODUCT_ID','RANKING','QUALITY','FINAL_BID'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()

    # Clean and compute score per bid
    ar = ar.dropna(subset=['AUCTION_ID','RANKING','QUALITY','FINAL_BID'])
    ar = ar[(ar['QUALITY']>0) & (ar['FINAL_BID']>0)]
    ar['score'] = ar['QUALITY'] * ar['FINAL_BID']

    # Bidder count per auction
    bidders = ar.groupby('AUCTION_ID').size().rename('bidder_count')
    # Top-2 by RANKING
    top2_rank = ar[ar['RANKING'].between(1,2)].copy()
    # Top-2 by score (descending)
    ar_sorted = ar.sort_values(['AUCTION_ID','score'], ascending=[True, False])
    ar_sorted['score_pos'] = ar_sorted.groupby('AUCTION_ID').cumcount() + 1
    top2_score = ar_sorted[ar_sorted['score_pos']<=2].copy()

    # Build winner/runner-up frames by score order
    win_s = top2_score[top2_score['score_pos']==1][['AUCTION_ID','VENDOR_ID','PRODUCT_ID','QUALITY','FINAL_BID','score']]
    win_s = win_s.rename(columns={'QUALITY':'Q1','FINAL_BID':'B1','score':'S1','VENDOR_ID':'V1','PRODUCT_ID':'P1'})
    run_s = top2_score[top2_score['score_pos']==2][['AUCTION_ID','VENDOR_ID','PRODUCT_ID','QUALITY','FINAL_BID','score']]
    run_s = run_s.rename(columns={'QUALITY':'Q2','FINAL_BID':'B2','score':'S2','VENDOR_ID':'V2','PRODUCT_ID':'P2'})
    pair_s = win_s.merge(run_s, on='AUCTION_ID', how='inner')

    # Build winner/runner-up frames by provided RANKING
    win_r = top2_rank[top2_rank['RANKING']==1][['AUCTION_ID','VENDOR_ID','PRODUCT_ID','QUALITY','FINAL_BID','score']]
    win_r = win_r.rename(columns={'QUALITY':'Q1r','FINAL_BID':'B1r','score':'S1r','VENDOR_ID':'V1r','PRODUCT_ID':'P1r'})
    run_r = top2_rank[top2_rank['RANKING']==2][['AUCTION_ID','VENDOR_ID','PRODUCT_ID','QUALITY','FINAL_BID','score']]
    run_r = run_r.rename(columns={'QUALITY':'Q2r','FINAL_BID':'B2r','score':'S2r','VENDOR_ID':'V2r','PRODUCT_ID':'P2r'})
    pair_r = win_r.merge(run_r, on='AUCTION_ID', how='inner')

    # Merge both definitions to assess alignment
    pair = pair_s.merge(pair_r, on='AUCTION_ID', how='left')
    pair = pair.merge(bidders, on='AUCTION_ID', how='left')
    pair = pair.merge(au, on='AUCTION_ID', how='left')

    # Alignment: top-2 by score equals provided ranking top-2 (compare vendor+product)
    pair['aligned_top1'] = (pair['V1'] == pair['V1r']) & (pair['P1'] == pair['P1r'])
    pair['aligned_top2'] = (pair['V2'] == pair['V2r']) & (pair['P2'] == pair['P2r'])
    pair['aligned'] = pair['aligned_top1'] & pair['aligned_top2']

    # Ratios using score order (robust): ratio = S1/S2; overhang = ratio - 1
    pair = pair[pair['S1'].notna() & pair['S2'].notna()]
    pair = pair[pair['S2'] > 0]
    pair['ratio'] = pair['S1'] / pair['S2']
    pair['overhang'] = pair['ratio'] - 1.0

    # Define bidder-count bins
    bins = pd.IntervalIndex.from_tuples([(2,5),(5,10),(10,20),(20,999999)], closed='right')
    pair['bidder_bin'] = pd.cut(pair['bidder_count'], bins)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13al_payment_rule_inference_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Payment Rule Inference via Top-2 Score Ratio (S1/S2) — {args.round}")
        wprint(f"Bids rows used: {len(ar):,}")
        wprint(f"Auctions with >=2 valid bids (score-order pairs): {pair['AUCTION_ID'].nunique():,}")
        # Alignment coverage
        aligned_share = pair['aligned'].mean()
        wprint(f"Alignment (top-2 by score equals provided rank top-2): {aligned_share*100:.2f}%")

        # Overall stats (all)
        wprint("\nOverall (all auctions with >=2 bids): ratio=S1/S2, overhang=ratio-1")
        wprint(summarize(pair['ratio']).round(4).to_string())
        wprint(summarize(pair['overhang']).round(4).to_string())
        wprint("Share ratio<=1.05: {:.2f}%".format(100* (pair['ratio'] <= 1.05).mean()))
        wprint("Share ratio<=1.10: {:.2f}%".format(100* (pair['ratio'] <= 1.10).mean()))

        # Aligned only
        aligned = pair[pair['aligned']]
        wprint("\nAligned only (top-2 score order matches provided ranking):")
        wprint(summarize(aligned['ratio']).round(4).to_string())
        wprint(summarize(aligned['overhang']).round(4).to_string())
        wprint("Share ratio<=1.05: {:.2f}%".format(100* (aligned['ratio'] <= 1.05).mean()))
        wprint("Share ratio<=1.10: {:.2f}%".format(100* (aligned['ratio'] <= 1.10).mean()))

        # By placement (aligned)
        wprint("\nBy placement (aligned): ratio stats")
        for plc, g in aligned.groupby('PLACEMENT', dropna=False):
            wprint(f"PLACEMENT={plc}")
            wprint(summarize(g['ratio']).round(4).to_string())
            wprint("Share ratio<=1.05: {:.2f}%".format(100* (g['ratio'] <= 1.05).mean()))
            wprint("Share ratio<=1.10: {:.2f}%".format(100* (g['ratio'] <= 1.10).mean()))

        # By bidder-count bins (aligned)
        wprint("\nBy bidder-count bin (aligned): ratio stats")
        for bb, g in aligned.groupby('bidder_bin', dropna=False):
            wprint(f"bidder_bin={bb}")
            wprint(summarize(g['ratio']).round(4).to_string())
            wprint("Share ratio<=1.05: {:.2f}%".format(100* (g['ratio'] <= 1.05).mean()))
            wprint("Share ratio<=1.10: {:.2f}%".format(100* (g['ratio'] <= 1.10).mean()))

        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()

