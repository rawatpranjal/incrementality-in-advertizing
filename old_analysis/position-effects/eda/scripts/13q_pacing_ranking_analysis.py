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

def within_fit(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    # De-mean by group, OLS without intercept
    df = pd.DataFrame({'y': y, 'g': groups})
    X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    df = pd.concat([df, X_df], axis=1)
    gm = df.groupby('g').transform('mean')
    y_til = (df['y'] - gm['y']).values
    X_til = (X_df - gm[X_df.columns]).values
    XtX = X_til.T @ X_til
    Xty = X_til.T @ y_til
    beta = np.linalg.pinv(XtX) @ Xty
    # R^2 on within-transformed y
    yhat_til = X_til @ beta
    sst = float(np.sum(y_til**2))
    sse = float(np.sum((y_til - yhat_til)**2))
    r2w = 1.0 - sse/sst if sst>0 else np.nan
    return beta.flatten(), r2w

def main():
    ap = argparse.ArgumentParser(description='Pacing vs FinalBid correlation and ranking effects, plus within-auction log-rank regression')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    path = get_path(args.round)
    cols = ['AUCTION_ID','RANKING','QUALITY','FINAL_BID','PACING']
    df = pd.read_parquet(path, columns=cols)
    df = df.dropna(subset=['AUCTION_ID','RANKING','QUALITY','FINAL_BID','PACING']).copy()
    df = df[(df['RANKING']>=1) & (df['FINAL_BID']>0) & (df['QUALITY']>0) & (df['PACING']>0)]

    # 1) Correlation between FINAL_BID and PACING (levels and logs)
    corr_levels = df['FINAL_BID'].corr(df['PACING'])
    corr_logs = np.log(df['FINAL_BID']).corr(np.log(df['PACING']))

    # Within-vendor correlation (median across vendors) if vendor id exists
    if 'VENDOR_ID' in df.columns:
        by_vendor = df.groupby('VENDOR_ID').apply(lambda x: np.log(x['FINAL_BID']).corr(np.log(x['PACING']))).dropna()
        corr_vendor_med = float(by_vendor.median()) if not by_vendor.empty else np.nan
    else:
        corr_vendor_med = np.nan

    # 2) Ranking alignment with and without PACING
    df['score_nop'] = df['QUALITY'] * df['FINAL_BID']
    df['score_withp'] = df['QUALITY'] * df['FINAL_BID'] * df['PACING']

    # Top1 accuracy per auction
    g = df.groupby('AUCTION_ID', observed=True)
    idx_rank1 = g['RANKING'].idxmin()
    idx_s1 = g['score_nop'].idxmax()
    idx_s2 = g['score_withp'].idxmax()
    top1_acc_nop = (idx_rank1 == idx_s1).mean()
    top1_acc_withp = (idx_rank1 == idx_s2).mean()

    # Spearman rho per auction
    def sp_for_group(x: pd.DataFrame, col: str) -> float:
        # Spearman between RANKING (ascending) and score col (descending)
        if len(x) < 2:
            return np.nan
        return spearmanr(x['RANKING'].values, (-x[col].values)).correlation

    rho_nop = g.apply(lambda x: sp_for_group(x, 'score_nop'))
    rho_withp = g.apply(lambda x: sp_for_group(x, 'score_withp'))
    # Summaries
    def summarize(s: pd.Series) -> dict:
        s = s.dropna()
        if s.empty:
            return {'n':0,'median':np.nan,'p10':np.nan,'p90':np.nan,'perfect':np.nan}
        return {
            'n': int(s.size),
            'median': float(s.median()),
            'p10': float(s.quantile(0.1)),
            'p90': float(s.quantile(0.9)),
            'perfect': float((np.isclose(s, 1.0)).mean())
        }

    summ_nop = summarize(rho_nop)
    summ_withp = summarize(rho_withp)

    # 3) Within-auction regression: log(RANKING) ~ a_auction + b1*log(FINAL_BID) + b2*log(QUALITY) + b3*log(PACING)
    y = np.log(df['RANKING'].values.astype(float))
    X = np.column_stack([
        np.log(df['FINAL_BID'].values.astype(float)),
        np.log(df['QUALITY'].values.astype(float)),
        np.log(df['PACING'].values.astype(float)),
    ])
    beta_all, r2w_all = within_fit(y, X, df['AUCTION_ID'].values)
    b_bid, b_qual, b_pac = beta_all.tolist()

    print(f"\n=== {args.round.upper()} pacing vs ranking analysis ===")
    print(f"n rows={len(df):,} auctions={df['AUCTION_ID'].nunique():,}")
    print(f"Corr(FINAL_BID, PACING): levels={corr_levels:.4f}, logs={corr_logs:.4f}; within-vendor(log) median={corr_vendor_med:.4f}")
    print(f"Top-1 accuracy: score=Q*BID: {top1_acc_nop:.4f} | score=Q*BID*PACING: {top1_acc_withp:.4f}")
    print("Spearman rho across auctions (RANK vs score):")
    print(f"  No pacing: median={summ_nop['median']:.4f}, p10={summ_nop['p10']:.4f}, p90={summ_nop['p90']:.4f}, perfect(=1)={summ_nop['perfect']:.4f}, n={summ_nop['n']}")
    print(f"  With pacing: median={summ_withp['median']:.4f}, p10={summ_withp['p10']:.4f}, p90={summ_withp['p90']:.4f}, perfect(=1)={summ_withp['perfect']:.4f}, n={summ_withp['n']}")
    print("Within-auction log-rank regression:")
    print(f"  log(RANK) = a_auction + {b_bid:.4f}*log(BID) + {b_qual:.4f}*log(QUALITY) + {b_pac:.4f}*log(PACING);   R^2_within={r2w_all:.4f}")

if __name__ == '__main__':
    main()

