#!/usr/bin/env python3
"""
Complete Near-tie RDD analysis for paper tables.

Generates all tables for the Fold Effects section:
- Table 1: Exposure and click effects by rank boundary (tau = 0.02)
- Table 2: Bandwidth sensitivity at the fold boundary (2 vs 3)
- Table 3: Device heterogeneity at the fold boundary (tau = 0.02)
- Table 4: Placement heterogeneity at the fold boundary (tau = 0.02)

Following Narayanan & Kalyanam (2015) and Cohen et al. (2016) methodology.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results'
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


def detect_device_by_auction(imps: pd.DataFrame) -> pd.DataFrame:
    """
    Detect device type per auction based on impression batch sizes.
    Batch = impressions with same (AUCTION_ID, OCCURRED_AT).
    mobile: max_batch_size <= 2
    desktop: max_batch_size >= 3
    """
    imps = imps.copy()
    imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')
    batch_sizes = imps.groupby(['AUCTION_ID', 'OCCURRED_AT']).size().reset_index(name='batch_size')
    auction_max_batch = batch_sizes.groupby('AUCTION_ID')['batch_size'].max().reset_index()
    auction_max_batch.columns = ['AUCTION_ID', 'max_batch_size']
    auction_max_batch['device'] = np.where(auction_max_batch['max_batch_size'] <= 2, 'mobile', 'desktop')
    return auction_max_batch


def run_rdd_analysis(pairs: pd.DataFrame, tau: float):
    """Run RDD analysis for a given tau bandwidth."""
    near_ties = pairs[pairs['rel_gap'] <= tau].copy()
    if len(near_ties) < 20:
        return None

    n_pairs = near_ties['pair_id'].nunique()
    groups = near_ties['pair_id'].values.astype(int)

    y_click = near_ties['clicked'].values.astype(float)
    lucky = near_ties['lucky'].values.astype(float)
    z = near_ties['z'].values.astype(float)

    res_click = fit_lpm_local_linear(y_click, lucky, z, cluster=groups)

    y_imp = near_ties['impressed'].values.astype(float)
    res_imp = fit_lpm_local_linear(y_imp, lucky, z, cluster=groups)

    # Conditional CTR (both impressed)
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

    # Balance check
    lucky_sub = near_ties[near_ties['lucky'] == 1]
    unlucky_sub = near_ties[near_ties['lucky'] == 0]
    qual_diff = lucky_sub['QUALITY'].mean() - unlucky_sub['QUALITY'].mean()
    qual_pool_sd = near_ties['QUALITY'].std()
    std_diff_qual = qual_diff / qual_pool_sd if qual_pool_sd > 0 else 0
    bid_diff = lucky_sub['FINAL_BID'].mean() - unlucky_sub['FINAL_BID'].mean()
    bid_pool_sd = near_ties['FINAL_BID'].std()
    std_diff_bid = bid_diff / bid_pool_sd if bid_pool_sd > 0 else 0
    balance = 'Pass' if abs(std_diff_qual) < 0.1 and abs(std_diff_bid) < 0.1 else 'Fail'

    return {
        'n_pairs': n_pairs,
        'n_both_imp': n_both_imp,
        'balance': balance,
        'std_diff_qual': std_diff_qual,
        'std_diff_bid': std_diff_bid,
        'res_imp': res_imp,
        'res_click': res_click,
        'res_ctr': res_ctr
    }


def main():
    ap = argparse.ArgumentParser(description='Complete RDD analysis for paper tables')
    ap.add_argument('--window_minutes', type=int, default=60)
    args = ap.parse_args()

    paths = get_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "02_full_rdd_analysis.txt"

    with open(out, 'w') as fh:
        def w(s): fh.write(str(s) + "\n"); fh.flush(); print(s)

        w("=" * 80)
        w("COMPLETE NEAR-TIE RDD ANALYSIS FOR PAPER")
        w("Following Narayanan & Kalyanam (2015) and Cohen et al. (2016)")
        w("=" * 80)
        w(f"\nWindow: last {args.window_minutes} minutes")
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

        # Detect devices
        w("\nDetecting device types...")
        device_map = detect_device_by_auction(imps)
        n_mobile = (device_map['device'] == 'mobile').sum()
        n_desktop = (device_map['device'] == 'desktop').sum()
        w(f"  Mobile auctions: {n_mobile:,} ({n_mobile / len(device_map) * 100:.1f}%)")
        w(f"  Desktop auctions: {n_desktop:,} ({n_desktop / len(device_map) * 100:.1f}%)")

        # Precompute impression and click flags
        imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')
        first_imp = imps.groupby(['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], as_index=False)['OCCURRED_AT'].min()
        first_imp = first_imp.rename(columns={'OCCURRED_AT': 'FIRST_IMP_AT'})
        clk_flag = clks[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates().assign(clicked=1)

        # Score prep
        ar = ar.dropna(subset=['QUALITY', 'FINAL_BID', 'RANKING'])
        ar = ar[(ar['QUALITY'] > 0) & (ar['FINAL_BID'] > 0) & (ar['RANKING'] >= 1)]
        ar['score'] = ar['QUALITY'].astype(float) * ar['FINAL_BID'].astype(float)
        w(f"  After score filter: {len(ar):,} bids")

        boundaries = [2, 4, 6, 7]
        taus = [0.005, 0.01, 0.02, 0.05]
        placements = ['1', '2', '3', '5']
        placement_names = {'1': 'Search', '2': 'Brand', '3': 'Product', '5': 'Category'}

        all_results = []

        # Build pairs for all placements
        w("\nBuilding near-tie pairs for all placements...")
        pairs_by_placement = {}

        for pl in tqdm(placements, desc="Placements"):
            auc_ids = set(au[au['PLACEMENT'].astype(str) == pl]['AUCTION_ID'].astype(str))
            if not auc_ids:
                continue

            ar_pl = ar[ar['AUCTION_ID'].astype(str).isin(auc_ids)].copy()
            if ar_pl.empty:
                continue

            ar_pl['pos_by_score'] = ar_pl.groupby('AUCTION_ID')['score'].rank(ascending=False, method='first')

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
                continue

            pairs = pd.concat(all_pairs, ignore_index=True)
            pairs['pair_id'] = pairs.groupby(['AUCTION_ID', 'boundary']).ngroup()
            pairs = pairs.sort_values(['pair_id', 'score'], ascending=[True, False])

            agg = pairs.groupby('pair_id')['score'].agg(['max', 'min']).rename(columns={'max': 'score_hi', 'min': 'score_lo'})
            pairs = pairs.merge(agg, on='pair_id', how='left')
            pairs['rel_gap'] = (pairs['score_hi'] - pairs['score_lo']) / pairs['score_hi']
            pairs['z'] = pairs['rel_gap']

            pairs = pairs.merge(first_imp, on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], how='left')
            pairs = pairs.merge(clk_flag, on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], how='left')
            pairs['clicked'] = pairs['clicked'].fillna(0).astype(int)
            pairs['impressed'] = pairs['FIRST_IMP_AT'].notna().astype(int)
            pairs['lucky'] = (pairs['RANKING'] == pairs.groupby('pair_id')['RANKING'].transform('min')).astype(int)
            pairs = pairs.merge(device_map[['AUCTION_ID', 'device']], on='AUCTION_ID', how='left')

            pairs_by_placement[pl] = pairs

        # Run all analyses
        w("\nRunning RDD analyses...")

        for pl, pairs in tqdm(pairs_by_placement.items(), desc="Analyzing placements"):
            for b in boundaries:
                sub_b = pairs[pairs['boundary'] == b].copy()
                if sub_b.empty:
                    continue

                for tau in taus:
                    # Pooled
                    res = run_rdd_analysis(sub_b, tau)
                    if res:
                        if 'error' not in res['res_imp']:
                            all_results.append({
                                'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                                'device': 'pooled', 'outcome': 'impressed', 'conditioning': 'none',
                                'beta': res['res_imp']['coef']['lucky'],
                                'se': res['res_imp']['se']['lucky'],
                                'p': res['res_imp']['p']['lucky'],
                                'n_pairs': res['n_pairs'], 'balance': res['balance']
                            })
                        if 'error' not in res['res_click']:
                            all_results.append({
                                'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                                'device': 'pooled', 'outcome': 'clicked', 'conditioning': 'none',
                                'beta': res['res_click']['coef']['lucky'],
                                'se': res['res_click']['se']['lucky'],
                                'p': res['res_click']['p']['lucky'],
                                'n_pairs': res['n_pairs'], 'balance': res['balance']
                            })
                        if res['res_ctr'] and 'error' not in res['res_ctr']:
                            all_results.append({
                                'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                                'device': 'pooled', 'outcome': 'clicked', 'conditioning': 'both_impressed',
                                'beta': res['res_ctr']['coef']['lucky'],
                                'se': res['res_ctr']['se']['lucky'],
                                'p': res['res_ctr']['p']['lucky'],
                                'n_pairs': res['n_both_imp'], 'balance': res['balance']
                            })

                    # Mobile only (for P1 at fold)
                    if pl == '1' and b == 2:
                        for device in ['mobile', 'desktop']:
                            sub_device = sub_b[sub_b['device'] == device].copy()
                            if sub_device.empty:
                                continue
                            # Renumber pair_ids for device subset
                            sub_device['pair_id'] = sub_device.groupby(['AUCTION_ID', 'boundary']).ngroup()
                            res = run_rdd_analysis(sub_device, tau)
                            if res:
                                if 'error' not in res['res_imp']:
                                    all_results.append({
                                        'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                                        'device': device, 'outcome': 'impressed', 'conditioning': 'none',
                                        'beta': res['res_imp']['coef']['lucky'],
                                        'se': res['res_imp']['se']['lucky'],
                                        'p': res['res_imp']['p']['lucky'],
                                        'n_pairs': res['n_pairs'], 'balance': res['balance']
                                    })
                                if res['res_ctr'] and 'error' not in res['res_ctr']:
                                    all_results.append({
                                        'placement': pl, 'boundary': f"{b}v{b+1}", 'tau': tau,
                                        'device': device, 'outcome': 'clicked', 'conditioning': 'both_impressed',
                                        'beta': res['res_ctr']['coef']['lucky'],
                                        'se': res['res_ctr']['se']['lucky'],
                                        'p': res['res_ctr']['p']['lucky'],
                                        'n_pairs': res['n_both_imp'], 'balance': res['balance']
                                    })

        df = pd.DataFrame(all_results)

        # Generate Paper Tables
        w("\n" + "=" * 80)
        w("PAPER TABLES")
        w("=" * 80)

        def sig_stars(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''

        # Table 1: Exposure and click effects by rank boundary (tau = 0.02, Placement 1)
        w("\n" + "-" * 80)
        w("TABLE 1: Exposure and click effects by rank boundary at tau = 0.02")
        w("-" * 80)
        w(f"{'Boundary':<15} {'beta_exp':<12} {'(SE)':<10} {'beta_ctr':<12} {'(SE)':<10} {'N':<10}")
        w("-" * 69)

        for bdry in ['2v3', '4v5', '6v7', '7v8']:
            imp_row = df[(df['placement'] == '1') & (df['boundary'] == bdry) &
                         (df['outcome'] == 'impressed') & (df['conditioning'] == 'none') &
                         (df['device'] == 'pooled') & (df['tau'] == 0.02)]
            ctr_row = df[(df['placement'] == '1') & (df['boundary'] == bdry) &
                         (df['outcome'] == 'clicked') & (df['conditioning'] == 'both_impressed') &
                         (df['device'] == 'pooled') & (df['tau'] == 0.02)]

            if not imp_row.empty:
                b_exp = imp_row['beta'].values[0]
                se_exp = imp_row['se'].values[0]
                p_exp = imp_row['p'].values[0]
                n = int(imp_row['n_pairs'].values[0])
                sig_exp = sig_stars(p_exp)

                if not ctr_row.empty:
                    b_ctr = ctr_row['beta'].values[0]
                    se_ctr = ctr_row['se'].values[0]
                    p_ctr = ctr_row['p'].values[0]
                    sig_ctr = sig_stars(p_ctr)
                    w(f"{bdry:<15} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {b_ctr:>7.3f}{sig_ctr:<4} ({se_ctr:.3f})    {n:>6}")
                else:
                    w(f"{bdry:<15} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {'---':>7}     {'---':>10}    {n:>6}")

        # Table 2: Bandwidth sensitivity at the fold boundary (2 vs 3)
        w("\n" + "-" * 80)
        w("TABLE 2: Bandwidth sensitivity at the fold boundary (2 vs 3)")
        w("-" * 80)
        w(f"{'tau':<10} {'beta_exp':<12} {'(SE)':<10} {'beta_ctr':<12} {'(SE)':<10} {'N pairs':<10}")
        w("-" * 64)

        for tau in taus:
            imp_row = df[(df['placement'] == '1') & (df['boundary'] == '2v3') &
                         (df['outcome'] == 'impressed') & (df['conditioning'] == 'none') &
                         (df['device'] == 'pooled') & (df['tau'] == tau)]
            ctr_row = df[(df['placement'] == '1') & (df['boundary'] == '2v3') &
                         (df['outcome'] == 'clicked') & (df['conditioning'] == 'both_impressed') &
                         (df['device'] == 'pooled') & (df['tau'] == tau)]

            if not imp_row.empty:
                b_exp = imp_row['beta'].values[0]
                se_exp = imp_row['se'].values[0]
                p_exp = imp_row['p'].values[0]
                n = int(imp_row['n_pairs'].values[0])
                sig_exp = sig_stars(p_exp)

                if not ctr_row.empty:
                    b_ctr = ctr_row['beta'].values[0]
                    se_ctr = ctr_row['se'].values[0]
                    sig_ctr = sig_stars(ctr_row['p'].values[0])
                    w(f"{tau:<10.3f} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {b_ctr:>7.3f}{sig_ctr:<4} ({se_ctr:.3f})    {n:>6}")
                else:
                    w(f"{tau:<10.3f} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {'---':>7}     {'---':>10}    {n:>6}")

        # Table 3: Device heterogeneity at the fold boundary (tau = 0.02)
        w("\n" + "-" * 80)
        w("TABLE 3: Device heterogeneity at the fold boundary (tau = 0.02)")
        w("-" * 80)
        w(f"{'Device':<12} {'beta_exp':<12} {'(SE)':<10} {'beta_ctr':<12} {'(SE)':<10} {'N pairs':<10}")
        w("-" * 64)

        for device in ['pooled', 'mobile', 'desktop']:
            device_label = 'Pooled' if device == 'pooled' else device.capitalize()
            imp_row = df[(df['placement'] == '1') & (df['boundary'] == '2v3') &
                         (df['outcome'] == 'impressed') & (df['conditioning'] == 'none') &
                         (df['device'] == device) & (df['tau'] == 0.02)]
            ctr_row = df[(df['placement'] == '1') & (df['boundary'] == '2v3') &
                         (df['outcome'] == 'clicked') & (df['conditioning'] == 'both_impressed') &
                         (df['device'] == device) & (df['tau'] == 0.02)]

            if not imp_row.empty:
                b_exp = imp_row['beta'].values[0]
                se_exp = imp_row['se'].values[0]
                p_exp = imp_row['p'].values[0]
                n = int(imp_row['n_pairs'].values[0])
                sig_exp = sig_stars(p_exp)

                if not ctr_row.empty:
                    b_ctr = ctr_row['beta'].values[0]
                    se_ctr = ctr_row['se'].values[0]
                    sig_ctr = sig_stars(ctr_row['p'].values[0])
                    w(f"{device_label:<12} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {b_ctr:>7.3f}{sig_ctr:<4} ({se_ctr:.3f})    {n:>6}")
                else:
                    w(f"{device_label:<12} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {'---':>7}     {'---':>10}    {n:>6}")

        # Table 4: Placement heterogeneity at the fold boundary (tau = 0.02)
        w("\n" + "-" * 80)
        w("TABLE 4: Placement heterogeneity at the fold boundary (tau = 0.02)")
        w("-" * 80)
        w(f"{'Placement':<15} {'beta_exp':<12} {'(SE)':<10} {'beta_ctr':<12} {'(SE)':<10} {'N pairs':<10}")
        w("-" * 69)

        for pl in placements:
            pl_name = f"{placement_names[pl]} (P{pl})"
            imp_row = df[(df['placement'] == pl) & (df['boundary'] == '2v3') &
                         (df['outcome'] == 'impressed') & (df['conditioning'] == 'none') &
                         (df['device'] == 'pooled') & (df['tau'] == 0.02)]
            ctr_row = df[(df['placement'] == pl) & (df['boundary'] == '2v3') &
                         (df['outcome'] == 'clicked') & (df['conditioning'] == 'both_impressed') &
                         (df['device'] == 'pooled') & (df['tau'] == 0.02)]

            if not imp_row.empty:
                b_exp = imp_row['beta'].values[0]
                se_exp = imp_row['se'].values[0]
                p_exp = imp_row['p'].values[0]
                n = int(imp_row['n_pairs'].values[0])
                sig_exp = sig_stars(p_exp)

                if not ctr_row.empty:
                    b_ctr = ctr_row['beta'].values[0]
                    se_ctr = ctr_row['se'].values[0]
                    sig_ctr = sig_stars(ctr_row['p'].values[0])
                    w(f"{pl_name:<15} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {b_ctr:>7.3f}{sig_ctr:<4} ({se_ctr:.3f})    {n:>6}")
                else:
                    w(f"{pl_name:<15} {b_exp:>7.3f}{sig_exp:<4} ({se_exp:.3f})    {'---':>7}     {'---':>10}    {n:>6}")

        # Save full results
        csv_out = RESULTS_DIR / "02_full_rdd_analysis.csv"
        df.to_csv(csv_out, index=False)
        w(f"\nFull results saved to: {csv_out}")
        w(f"Output saved to: {out}")


if __name__ == '__main__':
    main()
