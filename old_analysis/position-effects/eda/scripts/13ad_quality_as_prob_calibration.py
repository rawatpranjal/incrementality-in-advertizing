#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

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

def auc_roc_aggregated(scores: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    order = np.argsort(scores)
    s = scores[order]
    p = clks[order].astype(float)
    n = (imps[order] - clks[order]).astype(float)
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
    N = float(n_sum.sum())
    if P <= 0 or N <= 0:
        return np.nan
    cum_n_below = np.cumsum(np.concatenate([[0.0], n_sum[:-1]]))
    U = np.sum(p_sum * (cum_n_below + 0.5 * n_sum))
    return U / (P * N)

def auc_pr_aggregated(scores: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    order = np.argsort(-scores)
    s = scores[order]
    p = clks[order].astype(float)
    n = (imps[order] - clks[order]).astype(float)
    uniq, idx_start = np.unique(s, return_index=True)
    p_sum, n_sum = [], []
    for i in range(len(uniq)):
        start = idx_start[i]
        end = idx_start[i+1] if i+1 < len(idx_start) else len(s)
        p_sum.append(p[start:end].sum())
        n_sum.append(n[start:end].sum())
    p_sum = np.array(p_sum)
    n_sum = np.array(n_sum)
    P = float(p_sum.sum())
    if P <= 0:
        return np.nan
    tp = np.cumsum(p_sum)
    fp = np.cumsum(n_sum)
    recall = tp / P
    precision = tp / np.clip(tp + fp, 1e-12, None)
    precision_env = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - recall_prev) * precision_env))
    return ap

def brier(q: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    y = clks / np.clip(imps, 1e-12, None)
    w = imps.astype(float)
    return float(np.sum(w * (q - y) ** 2) / np.sum(w))

def logloss(q: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    q = np.clip(q, 1e-9, 1 - 1e-9)
    y = clks / np.clip(imps, 1e-12, None)
    w = imps.astype(float)
    return float(np.sum(w * ( -y * np.log(q) - (1 - y) * np.log(1 - q) )) / np.sum(w))

def calibration_bins(q: np.ndarray, imps: np.ndarray, clks: np.ndarray, n_bins: int = 20):
    # Quantile bins by q
    ranks = pd.Series(q).rank(pct=True).values
    bins = np.clip((ranks * n_bins).astype(int), 1, n_bins)
    out = []
    for b in range(1, n_bins + 1):
        m = bins == b
        if not np.any(m):
            out.append((b, 0, np.nan, np.nan))
            continue
        w = imps[m].sum()
        p_avg = float(np.average(q[m], weights=imps[m]))
        y_avg = float(clks[m].sum() / np.clip(imps[m].sum(), 1e-12, None))
        out.append((b, int(w), p_avg, y_avg))
    return out

def ece(q: np.ndarray, imps: np.ndarray, clks: np.ndarray, n_bins: int = 20) -> float:
    bins = calibration_bins(q, imps, clks, n_bins)
    total = float(imps.sum())
    err = 0.0
    for _, w, p_avg, y_avg in bins:
        if w == 0 or np.isnan(p_avg) or np.isnan(y_avg):
            continue
        err += (w / total) * abs(p_avg - y_avg)
    return float(err)

def main():
    ap = argparse.ArgumentParser(description='Treat QUALITY as probability: calibration metrics and calibrated mappings')
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

    q = df['QUALITY'].values.astype(float)
    imps_arr = df['imps'].values.astype(float)
    clks_arr = df['clks'].values.astype(float)
    base_rate = float(clks_arr.sum()/imps_arr.sum())

    # Raw QUALITY metrics
    roc = auc_roc_aggregated(q, imps_arr, clks_arr)
    pr = auc_pr_aggregated(q, imps_arr, clks_arr)
    bs = brier(q, imps_arr, clks_arr)
    ll = logloss(q, imps_arr, clks_arr)
    ece20 = ece(q, imps_arr, clks_arr, 20)

    # Logistic calibration: logit(click) ~ a + b*logit(q)
    q_clip = np.clip(q, 1e-6, 1 - 1e-6)
    logit_q = np.log(q_clip/(1 - q_clip))
    exog = sm.add_constant(logit_q)
    endog = np.column_stack([clks_arr, (imps_arr - clks_arr)])
    glm = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = glm.fit()
    p_logit = 1/(1 + np.exp(-(exog @ res.params)))

    # Isotonic calibration: p_iso = g(q)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
    p_iso = iso.fit_transform(q, clks_arr / np.clip(imps_arr, 1e-12, None), sample_weight=imps_arr)

    # Metrics for calibrated probabilities
    metrics = {
        'raw':   (q, roc, pr, bs, ll, ece20),
        'logit': (p_logit, auc_roc_aggregated(p_logit, imps_arr, clks_arr), auc_pr_aggregated(p_logit, imps_arr, clks_arr), brier(p_logit, imps_arr, clks_arr), logloss(p_logit, imps_arr, clks_arr), ece(p_logit, imps_arr, clks_arr, 20)),
        'iso':   (p_iso, auc_roc_aggregated(p_iso, imps_arr, clks_arr), auc_pr_aggregated(p_iso, imps_arr, clks_arr), brier(p_iso, imps_arr, clks_arr), logloss(p_iso, imps_arr, clks_arr), ece(p_iso, imps_arr, clks_arr, 20)),
    }

    print(f"\n=== {args.round.upper()} QUALITY as probability: calibration summary ===")
    print(f"rows={len(df):,} impressions={int(imps_arr.sum()):,} clicks={int(clks_arr.sum()):,} base_CTR={base_rate:.4f}")
    for name, (p, roc2, pr2, bs2, ll2, ece2) in metrics.items():
        print(f"{name:>6} -> ROC_AUC={roc2:.4f}  PR_AUC={pr2:.4f}  Brier={bs2:.6f}  LogLoss={ll2:.6f}  ECE@20={ece2:.4f}")
    # Show 10-bin calibration table for raw quality
    print("Calibration bins (raw QUALITY): bin, imps, mean_pred, empirical_CTR")
    for b, w, p_avg, y_avg in calibration_bins(q, imps_arr, clks_arr, 10):
        print(f"  {b:2d}  imps={w:7d}  meanQ={p_avg:.6f}  CTR={y_avg:.6f}")

if __name__ == '__main__':
    main()

