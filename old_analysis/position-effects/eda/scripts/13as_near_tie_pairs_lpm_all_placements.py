#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / 'results'
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / '0_data'


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


def fit_lpm_ols(y: np.ndarray, X: np.ndarray, cluster: np.ndarray = None):
    """Fit OLS and return coef, se, t, p for all columns.
    Uses statsmodels if available; else falls back to numpy OLS with homoskedastic SE.
    Cluster-robust by 'cluster' if provided and statsmodels is available.
    """
    try:
        import statsmodels.api as sm
        model = sm.OLS(y, X)
        if cluster is not None:
            res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster})
        else:
            res = model.fit(cov_type='HC1')
        coefs = res.params
        ses = res.bse
        tvals = res.tvalues
        pvals = res.pvalues
        return coefs, ses, tvals, pvals
    except Exception:
        # Fallback: numpy OLS
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (X.T @ y)
        resid = y - X @ beta
        n, k = X.shape
        s2 = float(resid.T @ resid) / max(n - k, 1)
        var = s2 * XtX_inv
        se = np.sqrt(np.diag(var))
        tvals = beta / np.where(se == 0, np.nan, se)
        from scipy.stats import t as student_t
        pvals = 2 * (1 - student_t.cdf(np.abs(tvals), df=max(n - k, 1)))
        return beta, se, tvals, pvals


def main():
    ap = argparse.ArgumentParser(description='Near-tie pairs across placements; LPM: click ~ lucky + score + score^2. Boundaries 2v3,4v5,6v7.')
    ap.add_argument('--round', required=True, choices=['round1','round2'])
    ap.add_argument('--window_minutes', type=int, default=600)
    ap.add_argument('--tau', type=float, default=0.01, help='Near-tie relative gap threshold (<= tau).')
    ap.add_argument('--boundaries', type=str, default='2,4,6', help='Comma-separated even starts, e.g., 2,4,6 for (2v3),(4v5),(6v7).')
    args = ap.parse_args()

    paths = get_paths(args.round)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13as_near_tie_pairs_lpm_all_placements_{args.round}.txt"

    with open(out, 'w') as fh:
        def w(s): fh.write(str(s)+"\n"); fh.flush(); print(s)

        w(f"Near-tie pairs LPM across placements — {args.round}")
        w(f"Window: last {args.window_minutes} minutes; near-tie rel_gap <= {args.tau}; boundaries: {args.boundaries}")
        w("Outcome: clicked (ever clicked in auction); LPM: click ~ 1 + lucky + score + score^2. Both-impressed pairs only.\n")

        # Load minimal columns
        imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','OCCURRED_AT'])
        clks = pd.read_parquet(paths['clicks'],      columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','OCCURRED_AT'])
        ar   = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','RANKING','QUALITY','FINAL_BID'])
        au   = pd.read_parquet(paths['auctions_users'],   columns=['AUCTION_ID','PLACEMENT','CREATED_AT']).drop_duplicates()

        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'], utc=True, errors='coerce')
        end_time = au['CREATED_AT'].max()
        start_time = end_time - pd.Timedelta(minutes=args.window_minutes)
        au = au[(au['CREATED_AT'] >= start_time) & (au['CREATED_AT'] <= end_time)]

        placements = sorted(au['PLACEMENT'].astype(str).unique())
        boundaries = [int(x.strip()) for x in args.boundaries.split(',') if x.strip()]

        # Precompute first impression and click flags
        imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')
        first_imp = imps.groupby(['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], as_index=False)['OCCURRED_AT'].min().rename(columns={'OCCURRED_AT':'FIRST_IMP_AT'})
        clk_flag = clks[['AUCTION_ID','PRODUCT_ID','VENDOR_ID']].drop_duplicates().assign(clicked=1)

        # Score prep
        ar = ar.dropna(subset=['QUALITY','FINAL_BID','RANKING'])
        ar = ar[(ar['QUALITY']>0) & (ar['FINAL_BID']>0) & (ar['RANKING']>=1)]
        ar['score'] = ar['QUALITY'].astype(float) * ar['FINAL_BID'].astype(float)

        for pl in placements:
            w(f"=== Placement {pl} ===")
            auc_ids = set(au[au['PLACEMENT'].astype(str)==pl]['AUCTION_ID'].astype(str))
            if not auc_ids:
                w("(no auctions)")
                continue

            ar_pl = ar[ar['AUCTION_ID'].astype(str).isin(auc_ids)].copy()
            if ar_pl.empty:
                w("(no bids)")
                continue

            # pos_by_score per auction
            ar_pl['pos_by_score'] = ar_pl.groupby('AUCTION_ID')['score'].rank(ascending=False, method='first')

            all_pairs = []
            for b in boundaries:
                want = {b, b+1}
                sub = ar_pl[ar_pl['pos_by_score'].isin(want)].copy()
                if sub.empty:
                    continue
                sub['boundary'] = b
                g = sub.groupby(['AUCTION_ID','boundary'])
                picked = g.apply(lambda df: df.sort_values('score', ascending=False).head(2))
                if isinstance(picked, pd.DataFrame):
                    all_pairs.append(picked.reset_index(drop=True))
            if not all_pairs:
                w("(no pairs)")
                continue
            pairs = pd.concat(all_pairs, ignore_index=True)
            pairs['pair_id'] = pairs.groupby(['AUCTION_ID','boundary']).ngroup()
            pairs = pairs.sort_values(['pair_id','score'], ascending=[True,False])
            pairs['score_rank'] = pairs.groupby('pair_id').cumcount() + 1
            agg = pairs.groupby('pair_id')['score'].agg(['max','min']).rename(columns={'max':'score_hi','min':'score_lo'})
            pairs = pairs.merge(agg, on='pair_id', how='left')
            pairs['rel_gap'] = (pairs['score_hi'] - pairs['score_lo']) / pairs['score_hi']

            # Attach impressions and clicks
            pairs = pairs.merge(first_imp, on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
            pairs = pairs.merge(clk_flag, on=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'], how='left')
            pairs['clicked'] = pairs['clicked'].fillna(0).astype(int)
            # Keep both impressed
            keep_ids = pairs.groupby('pair_id')['FIRST_IMP_AT'].apply(lambda s: s.notna().sum())
            keep_ids = set(keep_ids[keep_ids==2].index)
            pairs = pairs[pairs['pair_id'].isin(keep_ids)].copy()
            # Near-ties
            pairs = pairs[pairs['rel_gap'] <= args.tau].copy()
            if pairs.empty:
                w("(no near-tie both-impressed pairs)")
                continue

            # lucky: lower RANKING within pair
            pairs['lucky'] = (pairs['RANKING'] == pairs.groupby('pair_id')['RANKING'].transform('min')).astype(int)

            # LPM: click ~ 1 + lucky + score + score^2
            y = pairs['clicked'].values.astype(float)
            score = pairs['score'].values.astype(float)
            X = np.column_stack([np.ones(len(pairs)), pairs['lucky'].values.astype(float), score, score**2])
            groups = pairs['pair_id'].values.astype(int)

            coefs, ses, tvals, pvals = fit_lpm_ols(y, X, cluster=groups)
            names = ['const','lucky','score','score_sq']
            w(f"Rows={len(pairs):,} Pairs={pairs['pair_id'].nunique():,}")
            w("coefficients (clustered by pair):")
            for i, nm in enumerate(names):
                w(f"  {nm}: {coefs[i]:.6f}  (se={ses[i]:.6f}, t={tvals[i]:.2f}, p={pvals[i]:.4f})")

            # Boundary-specific fits
            for b in boundaries:
                sub = pairs[pairs['boundary']==b]
                if sub.empty:
                    continue
                yb = sub['clicked'].values.astype(float)
                sb = sub['score'].values.astype(float)
                Xb = np.column_stack([np.ones(len(sub)), sub['lucky'].values.astype(float), sb, sb**2])
                gb = sub['pair_id'].values.astype(int)
                cb, seb, tb, pb = fit_lpm_ols(yb, Xb, cluster=gb)
                w(f"  boundary {b} vs {b+1}: rows={len(sub):,} pairs={sub['pair_id'].nunique():,}")
                w(f"    lucky: {cb[1]:.6f} (se={seb[1]:.6f}, t={tb[1]:.2f}, p={pb[1]:.4f})")

        w(f"\nOutput saved to: {out}")


if __name__ == '__main__':
    main()

