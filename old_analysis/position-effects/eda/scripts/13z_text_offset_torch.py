#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'

def get_paths(round_name: str) -> dict:
    if round_name == 'round1':
        return {
            'auctions_results': DATA_DIR / 'round1/auctions_results_all.parquet',
            'auctions_users': DATA_DIR / 'round1/auctions_users_all.parquet',
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
            'catalog': DATA_DIR / 'round1/catalog_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
            'catalog': DATA_DIR / 'round2/catalog_r2.parquet',
        }
    raise ValueError(round_name)

def auc_from_aggregated(scores: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
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
    cum_n_below = np.cumsum(np.concatenate([[0.0], n_sum[:-1]]))
    U = np.sum(p_sum * (cum_n_below + 0.5 * n_sum))
    P = float(p_sum.sum())
    N = float(n_sum.sum())
    if P <= 0 or N <= 0:
        return np.nan
    return U / (P * N)

def average_precision_from_aggregated(score: np.ndarray, imps: np.ndarray, clks: np.ndarray) -> float:
    order = np.argsort(-score)
    s = score[order]
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
    if P <= 0:
        return float('nan')
    tp = np.cumsum(p_sum)
    fp = np.cumsum(n_sum)
    recall = tp / P
    precision = tp / np.clip(tp + fp, 1e-12, None)
    precision_env = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - recall_prev) * precision_env))
    return ap

def safe_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x.astype(float), eps))

def tokenize(text: str) -> list:
    return [t for t in text.lower().split() if t]

def hash_embed(tokens: list, n_dim: int = 256) -> np.ndarray:
    v = np.zeros(n_dim, dtype=np.float32)
    for t in tokens:
        idx = hash(t) % n_dim
        v[idx] += 1.0
    # l2 normalize to reduce length bias
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v

class OffsetLogistic(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.w = nn.Linear(d_in, 1, bias=True)
    def forward(self, X, offset):
        logits = self.w(X).squeeze(1) + offset
        return logits

def main():
    ap = argparse.ArgumentParser(description='Offset logistic with QUALITY offset; numeric + hashed text features (PyTorch)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--text_dim', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.1)
    args = ap.parse_args()

    paths = get_paths(args.round)
    ar_cols = ['AUCTION_ID','PRODUCT_ID','QUALITY','FINAL_BID','PRICE','CONVERSION_RATE','RANKING','PACING']
    ar = pd.read_parquet(paths['auctions_results'], columns=ar_cols)
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()
    cat = pd.read_parquet(paths['catalog'])
    # build a simple text field
    text_cols = [c for c in ['NAME','CATEGORIES','DESCRIPTION'] if c in cat.columns]
    cat['TEXT'] = cat[text_cols].astype(str).agg(' '.join, axis=1) if text_cols else ''
    cat = cat[['PRODUCT_ID','TEXT']]

    imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID'])
    clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID'])
    imp_counts = imps.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('imps').reset_index()
    clk_counts = clks.groupby(['AUCTION_ID','PRODUCT_ID']).size().rename('clks').reset_index()

    df = imp_counts.merge(clk_counts, on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df['clks'] = df['clks'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID']), on=['AUCTION_ID','PRODUCT_ID'], how='left')
    df = df.merge(au, on='AUCTION_ID', how='left')
    df = df.merge(cat, on='PRODUCT_ID', how='left')
    df = df.dropna(subset=['QUALITY','FINAL_BID','PRICE','RANKING','PACING','PLACEMENT'])

    # Offset = logit(QUALITY)
    q = np.clip(df['QUALITY'].values.astype(float), 1e-6, 1-1e-6)
    offset = np.log(q / (1 - q)).astype(np.float32)

    # Numeric features
    num_cols = ['FINAL_BID','PRICE','CONVERSION_RATE','PACING','RANKING']
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
    X_num = np.column_stack([
        safe_log(df['FINAL_BID']).values,
        safe_log(df['PRICE']).values,
        safe_log(df['CONVERSION_RATE']).values if 'CONVERSION_RATE' in df.columns else np.zeros(len(df)),
        safe_log(df['PACING']).values,
        safe_log(df['RANKING']).values,
    ]).astype(np.float32)

    # Placement one-hots (drop first)
    plc = pd.get_dummies(df['PLACEMENT'])
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]
    X_plc = plc.values.astype(np.float32) if plc.shape[1] > 0 else np.zeros((len(df),0), dtype=np.float32)

    # Text hashed embeddings
    texts = (df['TEXT'].fillna('').astype(str)).tolist()
    X_txt = np.vstack([hash_embed(tokenize(t), n_dim=args.text_dim) for t in texts]).astype(np.float32)

    X = np.concatenate([X_num, X_plc, X_txt], axis=1)
    y = (df['clks'] / df['imps']).values.astype(np.float32)
    w = df['imps'].values.astype(np.float32)

    # Baselines AP/AUC using offset only
    auc_off = auc_from_aggregated(offset, df['imps'].values, df['clks'].values)
    ap_off = average_precision_from_aggregated(offset, df['imps'].values, df['clks'].values)

    # Train model
    device = torch.device('cpu')
    model = OffsetLogistic(d_in=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X_t = torch.from_numpy(X)
    off_t = torch.from_numpy(offset)
    # Use clicks/imps targets via sample-wise BCE with weights=imps
    y_t = torch.from_numpy(y)
    w_t = torch.from_numpy(w)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    bs = 8192
    n = X.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb = X_t[idx]
            ofb = off_t[idx]
            yb = y_t[idx]
            wb = w_t[idx]
            logits = model(xb, ofb)
            loss = (bce(logits, yb) * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        # simple epoch print (optional)
        # print(f"epoch {epoch+1}: loss {total:.3f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_t, off_t).cpu().numpy().astype(np.float64)
    auc_full = auc_from_aggregated(logits, df['imps'].values, df['clks'].values)
    ap_full = average_precision_from_aggregated(logits, df['imps'].values, df['clks'].values)

    base_rate = float(df['clks'].sum()/df['imps'].sum())
    print(f"\n=== {args.round.upper()} Offset logistic (QUALITY + features incl. text) ===")
    print(f"rows={len(df):,} impressions={int(df['imps'].sum()):,} clicks={int(df['clks'].sum()):,}")
    print(f"Base CTR (random/AP baseline): {base_rate:.4f}")
    print(f"ROC AUC: offset-only={auc_off:.4f}  ->  features+text={auc_full:.4f}")
    print(f"PR AUC:  offset-only={ap_off:.4f}   ->  features+text={ap_full:.4f}")

if __name__ == '__main__':
    main()

