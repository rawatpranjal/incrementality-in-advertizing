#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import sparse

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def get_paths(round_name: str) -> dict:
    if round_name == 'round1':
        return {
            'auctions_results': DATA_DIR / 'round1/auctions_results_all.parquet',
            'impressions': DATA_DIR / 'round1/impressions_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
            'auctions_users': DATA_DIR / 'round1/auctions_users_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_results': DATA_DIR / 'round2/auctions_results_r2.parquet',
            'impressions': DATA_DIR / 'round2/impressions_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
        }
    raise ValueError(round_name)


def build_impression_frame(paths: dict, placement: str | None = None) -> pd.DataFrame:
    imp = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    clk = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','RANKING','QUALITY','PRICE','CONVERSION_RATE'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()

    clk = clk.assign(clicked=1)
    df = imp.merge(clk[['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','clicked']].drop_duplicates(),
                   on=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'], how='left')
    df['clicked'] = df['clicked'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']),
                  on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.merge(au, on='AUCTION_ID', how='left')
    # Filters
    df = df.dropna(subset=['RANKING','QUALITY','PRICE','CONVERSION_RATE','PLACEMENT'])
    df = df[(df['RANKING']>0) & (df['QUALITY']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0)]
    if placement is not None:
        df = df[df['PLACEMENT'].astype(str) == str(placement)]
    df = df.rename(columns={'RANKING':'rank','QUALITY':'quality','PRICE':'price','CONVERSION_RATE':'cvr'})
    df = df.reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description='FE-logit via scikit-learn (sparse OHE): click ~ quality + rank + price + cvr + placement FE + vendor FE')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    ap.add_argument('--placement', type=str, default=None, help='Optional: restrict to a placement id (e.g., 1)')
    args = ap.parse_args()

    paths = get_paths(args.round)
    df = build_impression_frame(paths, placement=args.placement)

    # Define variables explicitly
    # Dependent: clicked (binary)
    # Continuous regressors: quality (predicted CTR score), rank (1=best), price (AOV), cvr (predicted conversion rate)
    # Fixed effects: vendor_id (advertiser), placement (surface)
    y = df['clicked'].values.astype(int)
    X_num = df[['quality','rank','price','cvr']].astype(float)
    cat_df = df[['VENDOR_ID','PLACEMENT']].astype(str)

    # Build sparse design: numeric + OHE(cat) with drop='first' to avoid collinearity
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True)
    ct = ColumnTransformer([
        ('num', 'passthrough', ['quality','rank','price','cvr']),
        ('cat', ohe, ['VENDOR_ID','PLACEMENT'])
    ], sparse_threshold=1.0)
    X = ct.fit_transform(pd.concat([X_num, cat_df], axis=1))
    # Ensure CSR
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    # Fit FE-logit approximation: L2-penalized logit with very large C to minimize shrinkage
    clf = LogisticRegression(
        penalty='l2',
        C=1000.0,
        solver='saga',
        max_iter=200,
        n_jobs=1,
        verbose=0,
    )
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)

    # Extract numeric coefficients (first 4 columns in the transformed order)
    # ColumnTransformer keeps num first, then cat (with drop='first')
    beta = clf.coef_.ravel()
    beta_quality, beta_rank, beta_price, beta_cvr = beta[0], beta[1], beta[2], beta[3]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13aq_felogit_sklearn_{args.round}{('_pl'+args.placement) if args.placement else ''}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"FE-logit (sparse OHE) — {args.round}")
        if args.placement:
            wprint(f"Restricted to PLACEMENT={args.placement}")
        wprint("Variables (precise definitions):")
        wprint("  clicked: binary indicator for whether the impression received a click")
        wprint("  quality: QUALITY score from AUCTIONS_RESULTS (platform pCTR proxy)")
        wprint("  rank: bid rank (1 = best) from AUCTIONS_RESULTS")
        wprint("  price: PRICE from AUCTIONS_RESULTS (AOV proxy)")
        wprint("  cvr: CONVERSION_RATE from AUCTIONS_RESULTS (platform pCVR proxy)")
        wprint("  placement FE: categorical surface identifier from AUCTIONS_USERS (one-hot, drop='first')")
        wprint("  vendor FE: advertiser identifier from AUCTIONS_RESULTS (one-hot, drop='first')")
        wprint(f"Rows={len(df):,} CTR={df['clicked'].mean()*100:.3f}% Vendors={df['VENDOR_ID'].nunique():,} Placements={df['PLACEMENT'].nunique():,}")
        wprint(f"ROC_AUC={roc:.4f} PR_AUC={pr:.4f}")
        wprint("Numeric coefficients (log-odds):")
        wprint(f"  quality: {beta_quality:.6f}")
        wprint(f"  rank   : {beta_rank:.6f}")
        wprint(f"  price  : {beta_price:.6f}")
        wprint(f"  cvr    : {beta_cvr:.6f}")
        wprint(f"Output saved to: {out}")

if __name__ == '__main__':
    main()
