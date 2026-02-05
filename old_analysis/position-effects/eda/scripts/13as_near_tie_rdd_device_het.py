#!/usr/bin/env python3
"""
Near-tie RDD analysis with device heterogeneity following Narayanan & Kalyanam (2015)
and Cohen et al. (2016) methodology.

Extends 13as_near_tie_rdd_unconditional.py with device stratification based on
batch size heuristics from 14_device_detection_heuristics.py.

Device detection:
  - Batch = impressions with same (AUCTION_ID, OCCURRED_AT)
  - mobile: max_batch_size <= 2 for an auction
  - desktop: max_batch_size >= 3 for an auction

Primary specification: Local linear regression with forcing variable trends on BOTH sides.
    Y_{ai} = α + β·Lucky_{ai} + γ₁·z_{ai} + γ₂·z_{ai}·Lucky_{ai} + ε_{ai}

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


def fit_lpm_local_linear(y: np.ndarray, lucky: np.ndarray, z: np.ndarray, cluster: np.ndarray = None):
    """
    Fit local linear regression following Narayanan & Kalyanam (2015) Equation 8:
        Y = α + β·Lucky + γ₁·z + γ₂·z·Lucky + ε

    β is the position effect of interest.
    γ₁, γ₂ control for linear trends in forcing variable on BOTH sides.

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


def detect_device_by_auction(imps: pd.DataFrame) -> pd.DataFrame:
    """
    Detect device type per auction based on impression batch sizes.

    Batch = impressions with same (AUCTION_ID, OCCURRED_AT).
    Device logic:
      - mobile: max_batch_size <= 2
      - desktop: max_batch_size >= 3

    Returns DataFrame with columns [AUCTION_ID, device, max_batch_size]
    """
    imps = imps.copy()
    imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')

    batch_sizes = imps.groupby(['AUCTION_ID', 'OCCURRED_AT']).size().reset_index(name='batch_size')
    auction_max_batch = batch_sizes.groupby('AUCTION_ID')['batch_size'].max().reset_index()
    auction_max_batch.columns = ['AUCTION_ID', 'max_batch_size']
    auction_max_batch['device'] = np.where(auction_max_batch['max_batch_size'] <= 2, 'mobile', 'desktop')

    return auction_max_batch


def run_rdd_for_subset(pairs: pd.DataFrame, boundaries: list, taus: list, label: str, w):
    """Run RDD analysis for a subset of pairs (e.g., mobile or desktop)."""
    results = []

    for b in boundaries:
        w(f"\n--- Boundary {b} vs {b + 1} ({label}) ---")
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

            y_click = near_ties['clicked'].values.astype(float)
            lucky = near_ties['lucky'].values.astype(float)
            z = near_ties['z'].values.astype(float)

            res_click = fit_lpm_local_linear(y_click, lucky, z, cluster=groups)

            y_imp = near_ties['impressed'].values.astype(float)
            res_imp = fit_lpm_local_linear(y_imp, lucky, z, cluster=groups)

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

            lucky_sub = near_ties[near_ties['lucky'] == 1]
            unlucky_sub = near_ties[near_ties['lucky'] == 0]
            qual_diff = lucky_sub['QUALITY'].mean() - unlucky_sub['QUALITY'].mean()
            qual_pool_sd = near_ties['QUALITY'].std()
            std_diff_qual = qual_diff / qual_pool_sd if qual_pool_sd > 0 else 0
            bid_diff = lucky_sub['FINAL_BID'].mean() - unlucky_sub['FINAL_BID'].mean()
            bid_pool_sd = near_ties['FINAL_BID'].std()
            std_diff_bid = bid_diff / bid_pool_sd if bid_pool_sd > 0 else 0
            balance = 'Pass' if abs(std_diff_qual) < 0.1 and abs(std_diff_bid) < 0.1 else 'Fail'

            w(f"\n  τ = {tau}  |  N_pairs = {n_pairs}  |  N_both_impressed = {n_both_imp}  |  Balance: {balance}")
            w(f"    Std.diff(quality) = {std_diff_qual:.4f}, Std.diff(bid) = {std_diff_bid:.4f}")

            if 'error' not in res_click:
                beta = res_click['coef']['lucky']
                se = res_click['se']['lucky']
                t = res_click['t']['lucky']
                p = res_click['p']['lucky']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                w(f"    CLICKED (unconditional):  β = {beta:.4f} (SE={se:.4f}, t={t:.2f}, p={p:.4f}){sig}")
                results.append({
                    'device': label, 'boundary': f"{b}v{b+1}", 'tau': tau,
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
                w(f"    IMPRESSED (unconditional): β = {beta:.4f} (SE={se:.4f}, t={t:.2f}, p={p:.4f}){sig}")
                results.append({
                    'device': label, 'boundary': f"{b}v{b+1}", 'tau': tau,
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
                w(f"    CLICKED|both_imp (cond):   β = {beta:.4f} (SE={se:.4f}, t={t:.2f}, p={p:.4f}){sig}")
                results.append({
                    'device': label, 'boundary': f"{b}v{b+1}", 'tau': tau,
                    'outcome': 'clicked', 'conditioning': 'both_impressed',
                    'beta': beta, 'se': se, 't': t, 'p': p,
                    'n_pairs': n_both_imp, 'balance': balance
                })

    return results


def main():
    ap = argparse.ArgumentParser(description='Near-tie RDD with device heterogeneity')
    ap.add_argument('--round', required=True, choices=['round1', 'round2'])
    ap.add_argument('--window_minutes', type=int, default=600)
    ap.add_argument('--boundaries', type=str, default='2,4,6,7', help='Comma-separated rank starts, e.g., 2,4,6,7 for (2v3),(4v5),(6v7),(7v8)')
    args = ap.parse_args()

    paths = get_paths(args.round)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13as_near_tie_rdd_device_het_{args.round}.txt"

    with open(out, 'w') as fh:
        def w(s): fh.write(str(s) + "\n"); fh.flush(); print(s)

        w("=" * 80)
        w("Near-tie RDD Analysis: Device Heterogeneity")
        w("Following Narayanan & Kalyanam (2015) and Cohen et al. (2016)")
        w("=" * 80)
        w(f"\nData round: {args.round}")
        w(f"Window: last {args.window_minutes} minutes")
        w(f"Boundaries: {args.boundaries}")
        w("")
        w("DEVICE DETECTION HEURISTIC:")
        w("  Batch = impressions with same (AUCTION_ID, OCCURRED_AT)")
        w("  mobile: max_batch_size <= 2")
        w("  desktop: max_batch_size >= 3")
        w("")
        w("REGRESSION SPECIFICATION (Equation 8 from Narayanan & Kalyanam 2015):")
        w("  Y_{ai} = α + β·Lucky_{ai} + γ₁·z_{ai} + γ₂·z_{ai}·Lucky_{ai} + ε_{ai}")
        w("")
        w("where:")
        w("  Y = outcome (clicked, impressed)")
        w("  Lucky = 1 for higher-scoring ad in near-tie pair")
        w("  z = normalized score gap (s_r - s_{r+1})/s_r")
        w("  β = position effect of interest")
        w("  γ₁, γ₂ = local linear trends on each side of threshold")
        w("  Standard errors clustered at pair level")
        w("")

        w("Loading data...")
        imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID', 'PRODUCT_ID', 'USER_ID', 'VENDOR_ID', 'OCCURRED_AT'])
        clks = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID', 'PRODUCT_ID', 'USER_ID', 'VENDOR_ID', 'OCCURRED_AT'])
        ar = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'RANKING', 'QUALITY', 'FINAL_BID'])
        au = pd.read_parquet(paths['auctions_users'], columns=['AUCTION_ID', 'PLACEMENT', 'CREATED_AT']).drop_duplicates()

        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'], utc=True, errors='coerce')
        end_time = au['CREATED_AT'].max()
        start_time = end_time - pd.Timedelta(minutes=args.window_minutes)
        au = au[(au['CREATED_AT'] >= start_time) & (au['CREATED_AT'] <= end_time)]

        w(f"Auctions in window: {len(au):,}")
        w(f"Impressions: {len(imps):,}")
        w(f"Clicks: {len(clks):,}")
        w(f"Bids: {len(ar):,}")

        boundaries = [int(x.strip()) for x in args.boundaries.split(',') if x.strip()]
        taus = [0.005, 0.01, 0.02, 0.05]

        w("\nDetecting device type by auction...")
        device_map = detect_device_by_auction(imps)

        n_mobile = (device_map['device'] == 'mobile').sum()
        n_desktop = (device_map['device'] == 'desktop').sum()
        w(f"  Mobile auctions: {n_mobile:,} ({n_mobile / len(device_map) * 100:.1f}%)")
        w(f"  Desktop auctions: {n_desktop:,} ({n_desktop / len(device_map) * 100:.1f}%)")

        imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')
        first_imp = imps.groupby(['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], as_index=False)['OCCURRED_AT'].min()
        first_imp = first_imp.rename(columns={'OCCURRED_AT': 'FIRST_IMP_AT'})
        clk_flag = clks[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates().assign(clicked=1)

        ar = ar.dropna(subset=['QUALITY', 'FINAL_BID', 'RANKING'])
        ar = ar[(ar['QUALITY'] > 0) & (ar['FINAL_BID'] > 0) & (ar['RANKING'] >= 1)]
        ar['score'] = ar['QUALITY'].astype(float) * ar['FINAL_BID'].astype(float)

        all_results = []

        for pl in ['1']:
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

            ar_pl['pos_by_score'] = ar_pl.groupby('AUCTION_ID')['score'].rank(ascending=False, method='first')

            all_pairs = []
            for b in tqdm(boundaries, desc="Building pairs"):
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

            w(f"\nTotal pairs: {pairs['pair_id'].nunique():,}")
            pairs_mobile = pairs[pairs['device'] == 'mobile']
            pairs_desktop = pairs[pairs['device'] == 'desktop']
            w(f"Mobile pairs: {pairs_mobile['pair_id'].nunique():,}")
            w(f"Desktop pairs: {pairs_desktop['pair_id'].nunique():,}")

            w("\n" + "=" * 40)
            w("POOLED ANALYSIS (all devices)")
            w("=" * 40)
            results_pooled = run_rdd_for_subset(pairs, boundaries, taus, 'pooled', w)
            all_results.extend([{**r, 'placement': pl} for r in results_pooled])

            w("\n" + "=" * 40)
            w("MOBILE ANALYSIS (batch size <= 2)")
            w("=" * 40)
            if pairs_mobile['pair_id'].nunique() > 0:
                results_mobile = run_rdd_for_subset(pairs_mobile, boundaries, taus, 'mobile', w)
                all_results.extend([{**r, 'placement': pl} for r in results_mobile])
            else:
                w("(insufficient mobile pairs)")

            w("\n" + "=" * 40)
            w("DESKTOP ANALYSIS (batch size >= 3)")
            w("=" * 40)
            if pairs_desktop['pair_id'].nunique() > 0:
                results_desktop = run_rdd_for_subset(pairs_desktop, boundaries, taus, 'desktop', w)
                all_results.extend([{**r, 'placement': pl} for r in results_desktop])
            else:
                w("(insufficient desktop pairs)")

        w("\n" + "=" * 80)
        w("SUMMARY TABLES (for paper)")
        w("=" * 80)

        if all_results:
            df = pd.DataFrame(all_results)

            w("\n--- Table 1: Bandwidth Sensitivity at Fold (2v3), Placement 1 ---")
            w("Unconditional exposure effect (β_exp)")
            w("")
            w(f"{'τ':<10} {'β_exp':<12} {'(SE)':<10} {'p-value':<10} {'N pairs':<10} {'Balance':<10}")
            w("-" * 62)

            pl_df = df[(df['placement'] == '1') & (df['boundary'] == '2v3') &
                       (df['outcome'] == 'impressed') & (df['conditioning'] == 'none') &
                       (df['device'] == 'pooled')]
            for tau in taus:
                row = pl_df[pl_df['tau'] == tau]
                if not row.empty:
                    beta = row['beta'].values[0]
                    se = row['se'].values[0]
                    p = row['p'].values[0]
                    n = int(row['n_pairs'].values[0])
                    bal = row['balance'].values[0]
                    w(f"{tau:<10.3f} {beta:>8.4f}    ({se:.4f})    {p:>6.4f}      {n:>6}      {bal:<8}")

            w("\n--- Table 2: Bandwidth Sensitivity at Fold (2v3), Placement 1 ---")
            w("Conditional click effect (β_ctr | both impressed)")
            w("")
            w(f"{'τ':<10} {'β_ctr':<12} {'(SE)':<10} {'p-value':<10} {'N pairs':<10} {'Balance':<10}")
            w("-" * 62)

            pl_df = df[(df['placement'] == '1') & (df['boundary'] == '2v3') &
                       (df['outcome'] == 'clicked') & (df['conditioning'] == 'both_impressed') &
                       (df['device'] == 'pooled')]
            for tau in taus:
                row = pl_df[pl_df['tau'] == tau]
                if not row.empty:
                    beta = row['beta'].values[0]
                    se = row['se'].values[0]
                    p = row['p'].values[0]
                    n = int(row['n_pairs'].values[0])
                    bal = row['balance'].values[0]
                    w(f"{tau:<10.3f} {beta:>8.4f}    ({se:.4f})    {p:>6.4f}      {n:>6}      {bal:<8}")

            w("\n--- Table 3: Device Heterogeneity at Fold (2v3), τ=0.02 ---")
            w("")
            w(f"{'Device':<12} {'Outcome':<20} {'β':<12} {'(SE)':<10} {'p-value':<10} {'N pairs':<10}")
            w("-" * 74)

            for device in ['pooled', 'mobile', 'desktop']:
                for outcome, cond, label in [('impressed', 'none', 'Exposure (β_exp)'),
                                              ('clicked', 'both_impressed', 'CTR|seen (β_ctr)')]:
                    row = df[(df['placement'] == '1') & (df['boundary'] == '2v3') &
                             (df['tau'] == 0.02) & (df['device'] == device) &
                             (df['outcome'] == outcome) & (df['conditioning'] == cond)]
                    if not row.empty:
                        beta = row['beta'].values[0]
                        se = row['se'].values[0]
                        p = row['p'].values[0]
                        n = int(row['n_pairs'].values[0])
                        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                        w(f"{device:<12} {label:<20} {beta:>8.4f}{sig:<3} ({se:.4f})    {p:>6.4f}      {n:>6}")

            w("\n--- Table 4: Effect Decomposition by Boundary (τ=0.02, Pooled) ---")
            w("")
            w(f"{'Boundary':<12} {'β_exp':<12} {'(SE)':<10} {'β_ctr':<12} {'(SE)':<10} {'N pairs':<10}")
            w("-" * 66)

            for bdry in ['2v3', '4v5', '6v7', '7v8']:
                imp_row = df[(df['placement'] == '1') & (df['boundary'] == bdry) &
                             (df['outcome'] == 'impressed') & (df['conditioning'] == 'none') &
                             (df['device'] == 'pooled') & (df['tau'] == 0.02)]
                ctr_row = df[(df['placement'] == '1') & (df['boundary'] == bdry) &
                             (df['outcome'] == 'clicked') & (df['conditioning'] == 'both_impressed') &
                             (df['device'] == 'pooled') & (df['tau'] == 0.02)]

                if not imp_row.empty:
                    d_exp = imp_row['beta'].values[0]
                    se_exp = imp_row['se'].values[0]
                    n = int(imp_row['n_pairs'].values[0])

                    if not ctr_row.empty:
                        d_ctr = ctr_row['beta'].values[0]
                        se_ctr = ctr_row['se'].values[0]
                        w(f"{bdry:<12} {d_exp:>8.4f}    ({se_exp:.4f})    {d_ctr:>8.4f}    ({se_ctr:.4f})    {n:>6}")
                    else:
                        w(f"{bdry:<12} {d_exp:>8.4f}    ({se_exp:.4f})    {'--':>8}    {'--':>10}    {n:>6}")

            csv_out = RESULTS_DIR / f"13as_near_tie_rdd_device_het_{args.round}_full.csv"
            df.to_csv(csv_out, index=False)
            w(f"\nFull results saved to: {csv_out}")

        w(f"\nOutput saved to: {out}")


if __name__ == '__main__':
    main()
