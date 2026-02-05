#!/usr/bin/env python3
"""
Near-tie RDD analysis with unconditional outcomes following Narayanan & Kalyanam (2015)
and Cohen et al. (2016) methodology.

Primary specification: Local linear regression with forcing variable trends on BOTH sides.
    Y_{ai} = alpha + beta*Lucky_{ai} + gamma1*z_{ai} + gamma2*z_{ai}*Lucky_{ai} + epsilon_{ai}

Outcomes:
    - clicked (unconditional) - PRIMARY: total position effect
    - impressed (unconditional) - visibility mechanism
    - clicked | both impressed - conditional CTR (mechanism decomposition)

Standard errors clustered at pair level. Reports SEs and p-values for all estimates.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results'
# Use the larger data from the recent pull
DATA_DIR = Path(__file__).resolve().parents[3] / 'analysis' / 'position-effects' / '1_data_pull' / 'data'


def get_paths() -> dict:
    return {
        'auctions_results': DATA_DIR / 'auctions_results_all.parquet',
        'impressions': DATA_DIR / 'impressions_all.parquet',
        'clicks': DATA_DIR / 'clicks_all.parquet',
        'auctions_users': DATA_DIR / 'auctions_users_all.parquet',
    }


def fit_lpm_local_linear(y: np.ndarray, lucky: np.ndarray, z: np.ndarray, cluster: np.ndarray = None):
    """
    Fit local linear regression following Narayanan & Kalyanam (2015) Equation 8:
        Y = alpha + beta*Lucky + gamma1*z + gamma2*z*Lucky + epsilon

    beta is the position effect of interest.
    gamma1, gamma2 control for linear trends in forcing variable on BOTH sides.

    Returns dict with coef, se, t, p for 'lucky' coefficient.
    """
    try:
        import statsmodels.api as sm
        z_lucky = z * lucky
        X = np.column_stack([np.ones(len(y)), lucky, z, z_lucky])
        model = sm.OLS(y, X)
        if cluster is not None and len(np.unique(cluster)) > 1:
            res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster})
        else:
            res = model.fit(cov_type='HC1')

        names = ['const', 'lucky', 'z', 'z_lucky']
        return {
            'coef': dict(zip(names, res.params)),
            'se': dict(zip(names, res.bse)),
            't': dict(zip(names, res.tvalues)),
            'p': dict(zip(names, res.pvalues)),
            'nobs': int(res.nobs),
            'r2': res.rsquared
        }
    except Exception as e:
        return {'error': str(e)}


def fit_simple_diff(y: np.ndarray, lucky: np.ndarray, cluster: np.ndarray = None):
    """
    Simple difference in means (for comparison with local linear).
        Y = alpha + beta*Lucky + epsilon
    """
    try:
        import statsmodels.api as sm
        X = np.column_stack([np.ones(len(y)), lucky])
        model = sm.OLS(y, X)
        if cluster is not None and len(np.unique(cluster)) > 1:
            res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster})
        else:
            res = model.fit(cov_type='HC1')
        return {
            'coef': res.params[1],
            'se': res.bse[1],
            't': res.tvalues[1],
            'p': res.pvalues[1],
            'nobs': int(res.nobs)
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description='Near-tie RDD with unconditional outcomes (Narayanan/Cohen methodology)')
    ap.add_argument('--window_minutes', type=int, default=600)
    ap.add_argument('--boundaries', type=str, default='2,4,6,7', help='Comma-separated rank starts, e.g., 2,4,6,7 for (2v3),(4v5),(6v7),(7v8)')
    args = ap.parse_args()

    paths = get_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "01_near_tie_rdd.txt"

    with open(out, 'w') as fh:
        def w(s): fh.write(str(s) + "\n"); fh.flush(); print(s)

        w("=" * 80)
        w("Near-tie RDD Analysis: Unconditional Outcomes")
        w("Following Narayanan & Kalyanam (2015) and Cohen et al. (2016)")
        w("=" * 80)
        w(f"\nWindow: last {args.window_minutes} minutes")
        w(f"Boundaries: {args.boundaries}")
        w("")
        w("REGRESSION SPECIFICATION (Equation 8 from Narayanan & Kalyanam 2015):")
        w("  Y_{ai} = alpha + beta*Lucky_{ai} + gamma1*z_{ai} + gamma2*z_{ai}*Lucky_{ai} + epsilon_{ai}")
        w("")
        w("where:")
        w("  Y = outcome (clicked, impressed)")
        w("  Lucky = 1 for higher-scoring ad in near-tie pair")
        w("  z = normalized score gap (s_r - s_{r+1})/s_r")
        w("  beta = position effect of interest")
        w("  gamma1, gamma2 = local linear trends on each side of threshold")
        w("  Standard errors clustered at pair level")
        w("")

        # Load data
        w("Loading data...")
        imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID', 'PRODUCT_ID', 'USER_ID', 'VENDOR_ID', 'OCCURRED_AT'])
        clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID', 'PRODUCT_ID', 'USER_ID', 'VENDOR_ID', 'OCCURRED_AT'])
        ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'RANKING', 'QUALITY', 'FINAL_BID'])
        au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID', 'PLACEMENT', 'CREATED_AT']).drop_duplicates()

        w(f"  Impressions: {len(imps):,}")
        w(f"  Clicks: {len(clks):,}")
        w(f"  Auction results: {len(ar):,}")
        w(f"  Auctions users: {len(au):,}")

        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'], utc=True, errors='coerce')
        end_time = au['CREATED_AT'].max()
        start_time = end_time - pd.Timedelta(minutes=args.window_minutes)
        au = au[(au['CREATED_AT'] >= start_time) & (au['CREATED_AT'] <= end_time)]
        w(f"  After time filter ({args.window_minutes} min window): {len(au):,} auctions")

        placements = sorted(au['PLACEMENT'].astype(str).unique())
        boundaries = [int(x.strip()) for x in args.boundaries.split(',') if x.strip()]

        # Precompute impression and click flags
        imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')
        first_imp = imps.groupby(['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], as_index=False)['OCCURRED_AT'].min()
        first_imp = first_imp.rename(columns={'OCCURRED_AT': 'FIRST_IMP_AT'})
        clk_flag = clks[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates().assign(clicked=1)

        # Score prep
        ar = ar.dropna(subset=['QUALITY', 'FINAL_BID', 'RANKING'])
        ar = ar[(ar['QUALITY'] > 0) & (ar['FINAL_BID'] > 0) & (ar['RANKING'] >= 1)]
        ar['score'] = ar['QUALITY'].astype(float) * ar['FINAL_BID'].astype(float)
        w(f"  After score filter: {len(ar):,} bids with valid quality*final_bid")

        # Bandwidths to test
        taus = [0.005, 0.01, 0.02, 0.05]

        # Results storage
        all_results = []

        for pl in tqdm(placements, desc="Processing placements"):
            w(f"\n{'=' * 80}")
            w(f"PLACEMENT {pl}")
            w("=" * 80)

            auc_ids = set(au[au['PLACEMENT'].astype(str) == pl]['AUCTION_ID'].astype(str))
            if not auc_ids:
                w("(no auctions)")
                continue

            ar_pl = ar[ar['AUCTION_ID'].astype(str).isin(auc_ids)].copy()
            if ar_pl.empty:
                w("(no bids)")
                continue

            w(f"  Auctions in placement: {len(auc_ids):,}")
            w(f"  Bids in placement: {len(ar_pl):,}")

            # pos_by_score per auction
            ar_pl['pos_by_score'] = ar_pl.groupby('AUCTION_ID')['score'].rank(ascending=False, method='first')

            # Build pairs for all boundaries
            all_pairs = []
            for b in boundaries:
                want = {b, b + 1}
                sub = ar_pl[ar_pl['pos_by_score'].isin(want)].copy()
                if sub.empty:
                    continue
                sub['boundary'] = b
                g = sub.groupby(['AUCTION_ID', 'boundary'])
                picked = g.apply(lambda df: df.sort_values('score', ascending=False).head(2))
                if isinstance(picked, pd.DataFrame):
                    all_pairs.append(picked.reset_index(drop=True))

            if not all_pairs:
                w("(no pairs)")
                continue

            pairs = pd.concat(all_pairs, ignore_index=True)
            pairs['pair_id'] = pairs.groupby(['AUCTION_ID', 'boundary']).ngroup()
            pairs = pairs.sort_values(['pair_id', 'score'], ascending=[True, False])
            pairs['score_rank'] = pairs.groupby('pair_id').cumcount() + 1

            # Score gap calculation
            agg = pairs.groupby('pair_id')['score'].agg(['max', 'min']).rename(columns={'max': 'score_hi', 'min': 'score_lo'})
            pairs = pairs.merge(agg, on='pair_id', how='left')
            pairs['rel_gap'] = (pairs['score_hi'] - pairs['score_lo']) / pairs['score_hi']
            pairs['z'] = pairs['rel_gap']  # forcing variable

            # Attach impressions and clicks
            pairs = pairs.merge(first_imp, on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], how='left')
            pairs = pairs.merge(clk_flag, on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], how='left')
            pairs['clicked'] = pairs['clicked'].fillna(0).astype(int)
            pairs['impressed'] = pairs['FIRST_IMP_AT'].notna().astype(int)

            # lucky: higher score within pair (lower RANKING)
            pairs['lucky'] = (pairs['RANKING'] == pairs.groupby('pair_id')['RANKING'].transform('min')).astype(int)

            for b in boundaries:
                w(f"\n--- Boundary {b} vs {b + 1} ---")
                sub_b = pairs[pairs['boundary'] == b].copy()
                if sub_b.empty:
                    w("  (no pairs)")
                    continue

                for tau in taus:
                    near_ties = sub_b[sub_b['rel_gap'] <= tau].copy()
                    if len(near_ties) < 20:
                        continue

                    n_pairs = near_ties['pair_id'].nunique()
                    groups = near_ties['pair_id'].values.astype(int)

                    # OUTCOME 1: Clicked (UNCONDITIONAL) - PRIMARY
                    y_click = near_ties['clicked'].values.astype(float)
                    lucky = near_ties['lucky'].values.astype(float)
                    z = near_ties['z'].values.astype(float)

                    res_click = fit_lpm_local_linear(y_click, lucky, z, cluster=groups)

                    # OUTCOME 2: Impressed (UNCONDITIONAL) - visibility mechanism
                    y_imp = near_ties['impressed'].values.astype(float)
                    res_imp = fit_lpm_local_linear(y_imp, lucky, z, cluster=groups)

                    # OUTCOME 3: Clicked | Both Impressed - conditional CTR (mechanism)
                    keep_ids = near_ties.groupby('pair_id')['impressed'].sum()
                    keep_ids = set(keep_ids[keep_ids == 2].index)
                    both_imp = near_ties[near_ties['pair_id'].isin(keep_ids)].copy()

                    res_ctr = None
                    n_both_imp = 0
                    if len(both_imp) >= 20:
                        n_both_imp = both_imp['pair_id'].nunique()
                        y_ctr = both_imp['clicked'].values.astype(float)
                        lucky_ctr = both_imp['lucky'].values.astype(float)
                        z_ctr = both_imp['z'].values.astype(float)
                        groups_ctr = both_imp['pair_id'].values.astype(int)
                        res_ctr = fit_lpm_local_linear(y_ctr, lucky_ctr, z_ctr, cluster=groups_ctr)

                    # Balance check: compare quality and bid between lucky/unlucky
                    lucky_sub = near_ties[near_ties['lucky'] == 1]
                    unlucky_sub = near_ties[near_ties['lucky'] == 0]
                    qual_diff = lucky_sub['QUALITY'].mean() - unlucky_sub['QUALITY'].mean()
                    qual_pool_sd = near_ties['QUALITY'].std()
                    std_diff_qual = qual_diff / qual_pool_sd if qual_pool_sd > 0 else 0
                    bid_diff = lucky_sub['FINAL_BID'].mean() - unlucky_sub['FINAL_BID'].mean()
                    bid_pool_sd = near_ties['FINAL_BID'].std()
                    std_diff_bid = bid_diff / bid_pool_sd if bid_pool_sd > 0 else 0
                    balance = 'Pass' if abs(std_diff_qual) < 0.1 and abs(std_diff_bid) < 0.1 else 'Fail'

                    w(f"\n  tau = {tau}  |  N_pairs = {n_pairs}  |  N_both_impressed = {n_both_imp}  |  Balance: {balance}")
                    w(f"    Std.diff(quality) = {std_diff_qual:.4f}, Std.diff(bid) = {std_diff_bid:.4f}")

                    if 'error' not in res_click:
                        beta = res_click['coef']['lucky']
                        se = res_click['se']['lucky']
                        t = res_click['t']['lucky']
                        p = res_click['p']['lucky']
                        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                        w(f"    CLICKED (unconditional):  beta = {beta:.4f} (SE={se:.4f}, t={t:.2f}, p={p:.4f}){sig}")
                        all_results.append({
                            'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                            'outcome': 'clicked', 'conditioning': 'none',
                            'beta': beta, 'se': se, 't': t, 'p': p,
                            'n_pairs': n_pairs, 'balance': balance
                        })

                    if 'error' not in res_imp:
                        beta = res_imp['coef']['lucky']
                        se = res_imp['se']['lucky']
                        t = res_imp['t']['lucky']
                        p = res_imp['p']['lucky']
                        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                        w(f"    IMPRESSED (unconditional): beta = {beta:.4f} (SE={se:.4f}, t={t:.2f}, p={p:.4f}){sig}")
                        all_results.append({
                            'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                            'outcome': 'impressed', 'conditioning': 'none',
                            'beta': beta, 'se': se, 't': t, 'p': p,
                            'n_pairs': n_pairs, 'balance': balance
                        })

                    if res_ctr and 'error' not in res_ctr:
                        beta = res_ctr['coef']['lucky']
                        se = res_ctr['se']['lucky']
                        t = res_ctr['t']['lucky']
                        p = res_ctr['p']['lucky']
                        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                        w(f"    CLICKED|both_imp (cond):   beta = {beta:.4f} (SE={se:.4f}, t={t:.2f}, p={p:.4f}){sig}")
                        all_results.append({
                            'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                            'outcome': 'clicked', 'conditioning': 'both_impressed',
                            'beta': beta, 'se': se, 't': t, 'p': p,
                            'n_pairs': n_both_imp, 'balance': balance
                        })

        # Summary tables
        w("\n" + "=" * 80)
        w("SUMMARY TABLES (for paper)")
        w("=" * 80)

        if all_results:
            df = pd.DataFrame(all_results)

            # Table 1: Main results at tau = 0.02 (unconditional clicked)
            w("\n--- Table 1: Exposure and Click Decomposition by Rank Boundary (tau = 0.02) ---")
            w("(Unconditional outcomes as primary specification)")
            w("")
            w(f"{'Boundary':<12} {'Delta_exp':<12} {'(SE)':<10} {'Delta_click':<12} {'(SE)':<10} {'N pairs':<10}")
            w("-" * 66)

            for pl in ['1']:  # Focus on placement 1 (search)
                pl_df = df[(df['placement'] == pl) & (df['tau'] == 0.02)]
                for bdry in ['2v3', '4v5', '6v7', '7v8']:
                    imp_row = pl_df[(pl_df['boundary'] == bdry) & (pl_df['outcome'] == 'impressed') & (pl_df['conditioning'] == 'none')]
                    click_row = pl_df[(pl_df['boundary'] == bdry) & (pl_df['outcome'] == 'clicked') & (pl_df['conditioning'] == 'none')]

                    if not imp_row.empty and not click_row.empty:
                        d_exp = imp_row['beta'].values[0]
                        se_exp = imp_row['se'].values[0]
                        d_click = click_row['beta'].values[0]
                        se_click = click_row['se'].values[0]
                        n = int(click_row['n_pairs'].values[0])
                        w(f"{bdry:<12} {d_exp:>8.3f}    ({se_exp:.3f})    {d_click:>8.3f}    ({se_click:.3f})    {n:>6}")

            # Table 2: Bandwidth sensitivity
            w("\n--- Table 2: Bandwidth Sensitivity at Fold (2 vs 3), Placement 1 ---")
            w("")
            w(f"{'tau':<10} {'Delta_exp':<12} {'(SE)':<10} {'p-value':<10} {'N pairs':<10} {'Balance':<10}")
            w("-" * 62)

            pl_df = df[(df['placement'] == '1') & (df['boundary'] == '2v3') & (df['outcome'] == 'impressed') & (df['conditioning'] == 'none')]
            for tau in taus:
                row = pl_df[pl_df['tau'] == tau]
                if not row.empty:
                    beta = row['beta'].values[0]
                    se = row['se'].values[0]
                    p = row['p'].values[0]
                    n = int(row['n_pairs'].values[0])
                    bal = row['balance'].values[0]
                    w(f"{tau:<10.3f} {beta:>8.3f}    ({se:.3f})    {p:>6.4f}      {n:>6}      {bal:<8}")

            # Save full results
            csv_out = RESULTS_DIR / "01_near_tie_rdd_full.csv"
            df.to_csv(csv_out, index=False)
            w(f"\nFull results saved to: {csv_out}")

        w(f"\nOutput saved to: {out}")


if __name__ == '__main__':
    main()
