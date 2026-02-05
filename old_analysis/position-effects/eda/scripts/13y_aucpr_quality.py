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
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
        }
    raise ValueError(round_name)

def average_precision_from_aggregated(score: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    # Compute AP using a step-wise precision-recall curve at unique score thresholds (desc)
    # Following sklearn-style AP: AP = sum((R_i - R_{i-1}) * P_enveloped_i)
    order = np.argsort(-score)  # descending by score
    s = score[order]
    p = clks[order].astype(float)
    n = (imps[order] - clks[order]).astype(float)
    # group by unique scores
    uniq, idx_start = np.unique(s, return_index=True)
    p_sum = []
    n_sum = []
    for i in range(len(uniq)):
        start = idx_start[i]
        end = idx_start[i+1] if i+1 < len(idx_start) else len(s)
        p_sum.append(p[start:end].sum())
        n_sum.append(n[start:end].sum())
    p_sum = np.array(p_sum)
    n_sum = np.array(n_sum)
    P = float(p_sum.sum())
    if P <= 0:
        return float('nan')
    # cumulative at each threshold (after including the block)
    tp = np.cumsum(p_sum)
    fp = np.cumsum(n_sum)
    recall = tp / P
    precision = tp / np.clip(tp + fp, 1e-12, None)
    # precision envelope (monotone non-increasing)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    # integrate
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - recall_prev) * precision_envelope))
    return ap

def main():
    ap = argparse.ArgumentParser(description='Compute PR AUC (Average Precision) for QUALITY→clicks')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','QUALITY'])
    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID'])

    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('clks').reset_index()
    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID']), on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df = df.dropna(subset=['QUALITY'])

    score = df['QUALITY'].values.astype(float)
    imps_arr = df['imps'].values.astype(float)
    clks_arr = df['clks'].values.astype(float)
    ap_val = average_precision_from_aggregated(score, imps_arr, clks_arr)

    print(f"\n=== {args.round.upper()} Average Precision (QUALITY→clicks) ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print(f"AP (AUC-PR) for QUALITY score: {ap_val:.4f}")

if __name__ == '__main__':
    main()

