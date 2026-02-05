#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score

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

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def fit_glm_binom(y: np.ndarray, X_df: pd.DataFrame):
    X = sm.add_constant(X_df.astype(float), has_constant='add')
    endog = np.column_stack([y.astype(float), (1 - y).astype(float)])
    glm = sm.GLM(endog, X, family=sm.families.Binomial())
    res = glm.fit()
    ll = float(res.llf)
    p = res.predict()  # fitted probabilities
    return ll, p, res

def felogit_vendor_IRLS(df: pd.DataFrame, feature_cols: list, max_iter: int = 25, tol: float = 1e-6):
    # Vendor FE absorption with IRLS; returns log-likelihood and fitted probabilities
    g = df['VENDOR_ID'].values
    vendors, inv = np.unique(g, return_inverse=True)
    G = vendors.size
    y = df['clicked'].values.astype(float)
    X = df[feature_cols].values.astype(float)
    k = X.shape[1]
    beta = np.zeros(k)
    alpha = np.zeros(G)
    for _ in range(max_iter):
        eta = alpha[inv] + X @ beta
        mu = logistic(eta)
        mu = np.clip(mu, 1e-6, 1-1e-6)
        W = (mu * (1 - mu))
        z = eta + (y - mu) / (mu * (1 - mu))
        idx = pd.Series(inv)
        Wser = pd.Series(W)
        # demean X and z within vendor (weighted)
        X_til_cols = []
        for j in range(k):
            col = pd.Series(X[:, j])
            col_mean = (col * Wser).groupby(idx).sum() / Wser.groupby(idx).sum()
            X_til_cols.append(X[:, j] - col_mean.iloc[inv].values)
        X_til = np.column_stack(X_til_cols)
        z_mean = (pd.Series(z) * Wser).groupby(idx).sum() / Wser.groupby(idx).sum()
        z_til = z - z_mean.iloc[inv].values
        WX = X_til * W[:, None]
        XtWX = X_til.T @ WX
        XtWz = X_til.T @ (W * z_til)
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        res = z - (X @ beta_new)
        num = (pd.Series(res) * Wser).groupby(idx).sum()
        den = Wser.groupby(idx).sum()
        alpha_new = (num / den).fillna(0.0).values
        if max(np.max(np.abs(beta_new - beta)), np.max(np.abs(alpha_new - alpha))) < tol:
            beta, alpha = beta_new, alpha_new
            break
        beta, alpha = beta_new, alpha_new
    # likelihood and proba
    eta = alpha[inv] + X @ beta
    mu = logistic(eta)
    mu = np.clip(mu, 1e-9, 1-1e-9)
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    return ll, mu

def main():
    ap = argparse.ArgumentParser(description='Stepwise from QUALITY: add rank, price, cvr, placement FE, vendor FE (impressions)')
    ap.add_argument('--round', choices=['round1','round2'], required=True)
    args = ap.parse_args()

    paths = get_paths(args.round)
    # Build impression-level frame
    imp = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    clk = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'])
    ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','RANKING','QUALITY','PRICE','CONVERSION_RATE'])
    au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID','PLACEMENT']).drop_duplicates()

    clk = clk.assign(clicked=1)
    clicks_key = clk[['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','clicked']].drop_duplicates()
    df = imp.merge(clicks_key, on=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID'], how='left')
    df['clicked'] = df['clicked'].fillna(0).astype(int)
    df = df.merge(ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID']), on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
    df = df.merge(au, on='AUCTION_ID', how='left')
    df = df.dropna(subset=['RANKING','QUALITY','PRICE','CONVERSION_RATE'])
    df = df[(df['RANKING']>0) & (df['QUALITY']>0) & (df['PRICE']>0) & (df['CONVERSION_RATE']>0)]
    df = df.rename(columns={'RANKING':'rank','QUALITY':'quality','PRICE':'price','CONVERSION_RATE':'cvr'})
    # Placement FE
    plc = pd.get_dummies(df['PLACEMENT'], prefix='plc')
    if plc.shape[1] > 0:
        plc = plc.iloc[:,1:]  # drop first
    df = pd.concat([df, plc], axis=1)
    plc_cols = list(plc.columns)

    y = df['clicked'].values.astype(int)
    rows = len(df)
    ctr = df['clicked'].mean()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13aj_stepwise_quality_additions_{args.round}.txt"
    with open(out, 'w') as fh:
        def wprint(s): fh.write(str(s)+"\n"); fh.flush(); print(s)
        wprint(f"Stepwise additions starting from QUALITY — {args.round}")
        wprint(f"Rows={rows:,} CTR={ctr*100:.3f}% Vendors={df['VENDOR_ID'].nunique():,}")

        steps = []
        # 0) QUALITY only (no FE)
        ll0, p0, res0 = fit_glm_binom(y, df[['quality']])
        roc0 = roc_auc_score(y, p0)
        pr0 = average_precision_score(y, p0)
        steps.append(('quality', ll0, roc0, pr0))
        wprint(f"quality: logLik={ll0:.1f} ROC_AUC={roc0:.4f} PR_AUC={pr0:.4f}")

        # 1) + rank
        ll1, p1, _ = fit_glm_binom(y, df[['quality','rank']])
        roc1 = roc_auc_score(y, p1)
        pr1 = average_precision_score(y, p1)
        steps.append(('+rank', ll1, roc1, pr1))
        wprint(f"+rank:   logLik={ll1:.1f} ΔLL={ll1-ll0:.1f} ROC_AUC={roc1:.4f} PR_AUC={pr1:.4f}")

        # 2) + price
        ll2, p2, _ = fit_glm_binom(y, df[['quality','rank','price']])
        roc2 = roc_auc_score(y, p2)
        pr2 = average_precision_score(y, p2)
        steps.append(('+price', ll2, roc2, pr2))
        wprint(f"+price:  logLik={ll2:.1f} ΔLL={ll2-ll1:.1f} ROC_AUC={roc2:.4f} PR_AUC={pr2:.4f}")

        # 3) + cvr
        ll3, p3, _ = fit_glm_binom(y, df[['quality','rank','price','cvr']])
        roc3 = roc_auc_score(y, p3)
        pr3 = average_precision_score(y, p3)
        steps.append(('+cvr', ll3, roc3, pr3))
        wprint(f"+cvr:    logLik={ll3:.1f} ΔLL={ll3-ll2:.1f} ROC_AUC={roc3:.4f} PR_AUC={pr3:.4f}")

        # 4) + placement FE (no vendor FE yet)
        cols4 = ['quality','rank','price','cvr'] + plc_cols
        ll4, p4, _ = fit_glm_binom(y, df[cols4])
        roc4 = roc_auc_score(y, p4)
        pr4 = average_precision_score(y, p4)
        steps.append(('+placementFE', ll4, roc4, pr4))
        wprint(f"+plcFE:  logLik={ll4:.1f} ΔLL={ll4-ll3:.1f} ROC_AUC={roc4:.4f} PR_AUC={pr4:.4f}")

        # 5) + vendor FE (absorption)
        ll5, p5 = felogit_vendor_IRLS(df, cols4)
        roc5 = roc_auc_score(y, p5)
        pr5 = average_precision_score(y, p5)
        steps.append(('+vendorFE', ll5, roc5, pr5))
        wprint(f"+venFE:  logLik={ll5:.1f} ΔLL={ll5-ll4:.1f} ROC_AUC={roc5:.4f} PR_AUC={pr5:.4f}")

        wprint(f"\nOutput saved to: {out}")

if __name__ == '__main__':
    main()

