#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def summarize(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce')
    x = x.dropna()
    if x.empty:
        return pd.Series({
            'count': 0, 'mean': np.nan, 'std': np.nan, 'min': np.nan,
            'p25': np.nan, 'median': np.nan, 'p75': np.nan, 'max': np.nan
        })
    return pd.Series({
        'count': int(x.shape[0]),
        'mean': float(x.mean()),
        'std': float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
        'min': float(x.min()),
        'p25': float(x.quantile(0.25)),
        'median': float(x.quantile(0.50)),
        'p75': float(x.quantile(0.75)),
        'max': float(x.max())
    })


def main():
    ap = argparse.ArgumentParser(description='EDA: Near-tie score pairs at the fold (rank2 vs rank3) for PLACEMENT=1; no models.')
    ap.add_argument('--round', required=True, choices=['round1', 'round2'])
    ap.add_argument('--window_minutes', type=int, default=600, help='Slice based on AUCTIONS_USERS.CREATED_AT (last minutes).')
    ap.add_argument('--tau', type=float, default=0.01, help='Near-tie threshold: relative gap (S2-S3)/S2 <= tau.')
    ap.add_argument('--tau2', type=float, default=0.02, help='Secondary threshold for sensitivity.')
    args = ap.parse_args()

    paths = get_paths(args.round)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13ar_near_tie_rank2_vs_rank3_eda_{args.round}.txt"

    with open(out, 'w') as f:
        def log(s: str):
            f.write(str(s) + "\n"); f.flush(); print(s)

        log(f"Near-tie Rank-2 vs Rank-3 EDA — {args.round}")
        log("Design: score = QUALITY × FINAL_BID (ignore pacing); placement = 1 only.")
        log(f"Window: last {args.window_minutes} minutes based on AUCTIONS_USERS.CREATED_AT\n")

        # Load minimal columns
        log("Loading data (minimal columns)...")
        imps = pd.read_parquet(paths['impressions'], columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','OCCURRED_AT'])
        clks = pd.read_parquet(paths['clicks'],      columns=['AUCTION_ID','PRODUCT_ID','USER_ID','VENDOR_ID','OCCURRED_AT'])
        ar   = pd.read_parquet(paths['auctions_results'], columns=['AUCTION_ID','PRODUCT_ID','VENDOR_ID','RANKING','QUALITY','FINAL_BID'])
        au   = pd.read_parquet(paths['auctions_users'],   columns=['AUCTION_ID','PLACEMENT','CREATED_AT']).drop_duplicates()

        # Placement filter (P1 only) and window
        log("Filtering to PLACEMENT=1 and time window...")
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'], utc=True, errors='coerce')
        au = au[au['PLACEMENT'].astype(str) == '1']
        if not au['CREATED_AT'].empty:
            end_time = au['CREATED_AT'].max()
            start_time = end_time - pd.Timedelta(minutes=args.window_minutes)
            au = au[(au['CREATED_AT'] >= start_time) & (au['CREATED_AT'] <= end_time)]
        kept_auctions = set(au['AUCTION_ID'].astype(str))
        ar = ar[ar['AUCTION_ID'].astype(str).isin(kept_auctions)]
        imps = imps[imps['AUCTION_ID'].astype(str).isin(kept_auctions)]
        clks = clks[clks['AUCTION_ID'].astype(str).isin(kept_auctions)]

        # Prepare AR with score and dedupe
        log("Computing score and preparing auction-level ordering...")
        ar = ar.dropna(subset=['QUALITY','FINAL_BID','RANKING'])
        ar = ar[(ar['QUALITY'] > 0) & (ar['FINAL_BID'] > 0) & (ar['RANKING'] >= 1)]
        ar['score'] = ar['QUALITY'].astype(float) * ar['FINAL_BID'].astype(float)
        ar_small = ar.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'])

        # Within-auction top-3 by score; focus on 2nd vs 3rd by score
        log("Building top-3 by score per auction (score=Q×BID)...")
        def top3(df):
            sdf = df.sort_values('score', ascending=False)
            return sdf.head(3)
        top3_df = ar_small.groupby('AUCTION_ID', sort=False, group_keys=False).apply(top3)
        # Keep only auctions with >=3
        counts = top3_df.groupby('AUCTION_ID').size()
        valid_auctions = set(counts[counts >= 3].index.astype(str))
        top3_df = top3_df[top3_df['AUCTION_ID'].astype(str).isin(valid_auctions)]

        # Extract 2nd and 3rd by score rows into a single frame
        log("Extracting 2nd and 3rd by score and constructing near-tie indicators...")
        def pick_2_3(sdf):
            sdf2 = sdf.sort_values('score', ascending=False).iloc[1:3].copy()
            # label rank-by-score positions
            sdf2 = sdf2.reset_index(drop=True)
            sdf2.loc[:, 'score_pos'] = [2, 3]
            return sdf2
        pair_df = top3_df.groupby('AUCTION_ID', sort=False, group_keys=False).apply(pick_2_3)
        # Pivot to have wide format (2 vs 3)
        def to_wide(df):
            out = {}
            for _, row in df.iterrows():
                pos = int(row['score_pos'])
                out[f'P{pos}_PRODUCT_ID'] = row['PRODUCT_ID']
                out[f'P{pos}_VENDOR_ID'] = row['VENDOR_ID']
                out[f'P{pos}_RANKING'] = row['RANKING']
                out[f'P{pos}_QUALITY'] = float(row['QUALITY'])
                out[f'P{pos}_FINAL_BID'] = float(row['FINAL_BID'])
                out[f'P{pos}_SCORE'] = float(row['score'])
            return pd.Series(out)
        pairs = pair_df.groupby('AUCTION_ID').apply(to_wide).reset_index()
        pairs = pairs.merge(au[['AUCTION_ID','CREATED_AT']], on='AUCTION_ID', how='left')
        pairs['CREATED_AT'] = pd.to_datetime(pairs['CREATED_AT'], utc=True, errors='coerce')

        # Near-tie metrics
        pairs['rel_gap'] = (pairs['P2_SCORE'] - pairs['P3_SCORE']) / pairs['P2_SCORE']
        pairs['ratio'] = pairs['P2_SCORE'] / pairs['P3_SCORE']
        pairs['abs_gap'] = pairs['P2_SCORE'] - pairs['P3_SCORE']
        # Alignment with provided rankings
        pairs['rnk_align_23'] = (pairs['P2_RANKING'] < pairs['P3_RANKING']).astype(int)

        # Attach impression times (first impression per product in the auction)
        log("Attaching first impression timestamps for each item (within auction)...")
        imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')
        # Pre-aggregate first impression per (AUCTION_ID, PRODUCT_ID, VENDOR_ID)
        grp_cols = ['AUCTION_ID','PRODUCT_ID','VENDOR_ID']
        first_imp = imps.groupby(grp_cols, as_index=False)['OCCURRED_AT'].min().rename(columns={'OCCURRED_AT':'FIRST_IMP_AT'})

        # Map for P2 and P3
        def lookup_first_imp(row, pos):
            key = (row['AUCTION_ID'], row[f'P{pos}_PRODUCT_ID'], row[f'P{pos}_VENDOR_ID'])
            hit = first_imp[(first_imp['AUCTION_ID']==key[0]) & (first_imp['PRODUCT_ID']==key[1]) & (first_imp['VENDOR_ID']==key[2])]
            if hit.empty:
                return pd.NaT
            return pd.to_datetime(hit['FIRST_IMP_AT'].iloc[0], utc=True)

        # Compute first impression overall for the auction
        first_auc_imp = imps.groupby('AUCTION_ID', as_index=False)['OCCURRED_AT'].min().rename(columns={'OCCURRED_AT':'FIRST_AUC_IMP_AT'})
        pairs = pairs.merge(first_auc_imp, on='AUCTION_ID', how='left')

        log("Computing time deltas (load and dwell)...")
        p2_first_times = []
        p3_first_times = []
        for i in tqdm(range(len(pairs)), desc='Lookup first impressions'):
            row = pairs.iloc[i]
            p2_first_times.append(lookup_first_imp(row, 2))
            p3_first_times.append(lookup_first_imp(row, 3))
        pairs['P2_FIRST_IMP_AT'] = pd.to_datetime(p2_first_times)
        pairs['P3_FIRST_IMP_AT'] = pd.to_datetime(p3_first_times)

        # Deltas
        pairs['load_time_s'] = (pairs['FIRST_AUC_IMP_AT'] - pairs['CREATED_AT']).dt.total_seconds()
        pairs['P2_time_to_imp_s'] = (pairs['P2_FIRST_IMP_AT'] - pairs['CREATED_AT']).dt.total_seconds()
        pairs['P3_time_to_imp_s'] = (pairs['P3_FIRST_IMP_AT'] - pairs['CREATED_AT']).dt.total_seconds()
        pairs['P2_dwell_s'] = (pairs['P2_FIRST_IMP_AT'] - pairs['FIRST_AUC_IMP_AT']).dt.total_seconds()
        pairs['P3_dwell_s'] = (pairs['P3_FIRST_IMP_AT'] - pairs['FIRST_AUC_IMP_AT']).dt.total_seconds()
        pairs['both_impressed'] = (~pairs['P2_FIRST_IMP_AT'].isna()) & (~pairs['P3_FIRST_IMP_AT'].isna())
        pairs['p2_only'] = (~pairs['P2_FIRST_IMP_AT'].isna()) & (pairs['P3_FIRST_IMP_AT'].isna())
        pairs['p3_only'] = (pairs['P2_FIRST_IMP_AT'].isna()) & (~pairs['P3_FIRST_IMP_AT'].isna())
        pairs['none_impressed'] = pairs['P2_FIRST_IMP_AT'].isna() & pairs['P3_FIRST_IMP_AT'].isna()

        # Click flags (ever clicked within auction for that product)
        clks_flag = clks.copy()
        clks_flag['clicked'] = 1
        clks_flag = clks_flag.drop_duplicates(subset=['AUCTION_ID','PRODUCT_ID','VENDOR_ID'])
        def has_click(row, pos):
            key = (row['AUCTION_ID'], row[f'P{pos}_PRODUCT_ID'], row[f'P{pos}_VENDOR_ID'])
            hit = clks_flag[(clks_flag['AUCTION_ID']==key[0]) & (clks_flag['PRODUCT_ID']==key[1]) & (clks_flag['VENDOR_ID']==key[2])]
            return 1 if not hit.empty else 0
        p2_clicked = []
        p3_clicked = []
        for i in tqdm(range(len(pairs)), desc='Lookup clicks'):
            row = pairs.iloc[i]
            p2_clicked.append(has_click(row, 2))
            p3_clicked.append(has_click(row, 3))
        pairs['P2_clicked'] = np.array(p2_clicked, dtype=int)
        pairs['P3_clicked'] = np.array(p3_clicked, dtype=int)

        # Basic overview
        log("\n=== Universe and near-tie selection ===")
        log(f"Auctions in PLACEMENT=1 window: {len(kept_auctions):,}")
        log(f"Auctions with >=3 valid bids (by score construction): {len(valid_auctions):,}")
        log("\nDistribution of near-tie closeness for 2nd vs 3rd (score order):")
        log(summarize(pairs['rel_gap']).round(6).to_string())
        n_tau = int((pairs['rel_gap'] <= args.tau).sum())
        n_tau2 = int((pairs['rel_gap'] <= args.tau2).sum())
        log(f"\nNear-ties count (rel_gap <= {args.tau:.3f}): {n_tau:,}")
        log(f"Near-ties count (rel_gap <= {args.tau2:.3f}): {n_tau2:,}")

        # Alignment with provided ranking
        log("\nRank alignment (expect RANKING(2) < RANKING(3)):")
        overall_align = pairs['rnk_align_23'].mean() if len(pairs) else np.nan
        align_tau = pairs.loc[pairs['rel_gap'] <= args.tau, 'rnk_align_23'].mean() if n_tau else np.nan
        align_tau2 = pairs.loc[pairs['rel_gap'] <= args.tau2, 'rnk_align_23'].mean() if n_tau2 else np.nan
        log(f"Overall alignment share: {overall_align:.4f}")
        log(f"Alignment share (<= {args.tau:.3f}): {align_tau:.4f}")
        log(f"Alignment share (<= {args.tau2:.3f}): {align_tau2:.4f}")

        # Impression coverage
        log("\nImpression coverage for near-ties:")
        sub_tau = pairs[pairs['rel_gap'] <= args.tau].copy()
        def coverage_stats(df):
            tot = len(df)
            if tot == 0:
                return "(no rows)"
            b = int(df['both_impressed'].sum())
            p2o = int(df['p2_only'].sum())
            p3o = int(df['p3_only'].sum())
            none = int(df['none_impressed'].sum())
            return f"N={tot:,} | both={b:,} ({b/tot:.3f}) | p2_only={p2o:,} ({p2o/tot:.3f}) | p3_only={p3o:,} ({p3o/tot:.3f}) | none={none:,} ({none/tot:.3f})"
        log(f"rel_gap <= {args.tau:.3f}: {coverage_stats(sub_tau)}")
        sub_tau2 = pairs[pairs['rel_gap'] <= args.tau2].copy()
        log(f"rel_gap <= {args.tau2:.3f}: {coverage_stats(sub_tau2)}")

        # Timing evidence (dwell and time-to-impression) among both-impressed near-ties
        def timing_block(df, label):
            log(f"\nTiming among both-impressed near-ties — {label}")
            df2 = df[df['both_impressed']].copy()
            if df2.empty:
                log("(no rows)")
                return
            log("P2 dwell_s (first impression minus auction first impression):")
            log(summarize(df2['P2_dwell_s']).round(6).to_string())
            log("P3 dwell_s:")
            log(summarize(df2['P3_dwell_s']).round(6).to_string())
            share_p3_after = (df2['P3_dwell_s'] > 0).mean()
            share_p3_after_1s = (df2['P3_dwell_s'] > 1.0).mean()
            share_p3_after_5s = (df2['P3_dwell_s'] > 5.0).mean()
            log(f"Share P3 appears after first view: {share_p3_after:.3f} ( >1s: {share_p3_after_1s:.3f}, >5s: {share_p3_after_5s:.3f} )")
            log("\nTime to impression (s) from request:")
            log("P2 time_to_imp_s:")
            log(summarize(df2['P2_time_to_imp_s']).round(6).to_string())
            log("P3 time_to_imp_s:")
            log(summarize(df2['P3_time_to_imp_s']).round(6).to_string())

        timing_block(sub_tau, f"rel_gap <= {args.tau:.3f}")
        timing_block(sub_tau2, f"rel_gap <= {args.tau2:.3f}")

        # Click evidence (ever-clicked flags)
        def click_block(df, label):
            log(f"\nClick flags (ever clicked within auction) — {label}")
            if df.empty:
                log("(no rows)")
                return
            log(f"P2 clicked rate: {df['P2_clicked'].mean():.4f}")
            log(f"P3 clicked rate: {df['P3_clicked'].mean():.4f}")
            # Restrict to both impressed for fair comparison
            dff = df[df['both_impressed']]
            if not dff.empty:
                log(f"(Both impressed) P2 clicked rate: {dff['P2_clicked'].mean():.4f}")
                log(f"(Both impressed) P3 clicked rate: {dff['P3_clicked'].mean():.4f}")

        click_block(sub_tau, f"rel_gap <= {args.tau:.3f}")
        click_block(sub_tau2, f"rel_gap <= {args.tau2:.3f}")

        # Additional sanity tables
        log("\n=== Additional sanity checks ===")
        log("Score component summaries for pairs (second vs third):")
        comp = pd.DataFrame({
            'P2_QUALITY': pairs['P2_QUALITY'],
            'P3_QUALITY': pairs['P3_QUALITY'],
            'P2_FINAL_BID': pairs['P2_FINAL_BID'],
            'P3_FINAL_BID': pairs['P3_FINAL_BID'],
            'P2_SCORE': pairs['P2_SCORE'],
            'P3_SCORE': pairs['P3_SCORE'],
        })
        log("P2_QUALITY:")
        log(summarize(comp['P2_QUALITY']).round(6).to_string())
        log("P3_QUALITY:")
        log(summarize(comp['P3_QUALITY']).round(6).to_string())
        log("P2_FINAL_BID:")
        log(summarize(comp['P2_FINAL_BID']).round(6).to_string())
        log("P3_FINAL_BID:")
        log(summarize(comp['P3_FINAL_BID']).round(6).to_string())

        log(f"\nOutput saved to: {out}")


if __name__ == '__main__':
    main()

