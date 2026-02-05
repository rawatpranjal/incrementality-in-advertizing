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
            'auctions_users': DATA_DIR / 'round1/auctions_users_all.parquet',
            'clicks': DATA_DIR / 'round1/clicks_all.parquet',
        }
    if round_name == 'round2':
        return {
            'auctions_users': DATA_DIR / 'round2/auctions_users_r2.parquet',
            'clicks': DATA_DIR / 'round2/clicks_r2.parquet',
        }
    raise ValueError(round_name)


def summarize(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce').dropna()
    if x.empty:
        return pd.Series({'count': 0, 'mean': np.nan, 'std': np.nan, 'min': np.nan,
                          'p25': np.nan, 'median': np.nan, 'p75': np.nan, 'max': np.nan})
    return pd.Series({'count': int(x.shape[0]), 'mean': float(x.mean()),
                      'std': float(x.std(ddof=1)) if x.size > 1 else 0.0,
                      'min': float(x.min()), 'p25': float(x.quantile(0.25)),
                      'median': float(x.quantile(0.50)), 'p75': float(x.quantile(0.75)),
                      'max': float(x.max())})


def main():
    ap = argparse.ArgumentParser(description='Infer organic product page arrivals using P1 (search) and P3 (product) auctions and ad clicks.')
    ap.add_argument('--round', required=True, choices=['round1','round2'])
    ap.add_argument('--window_minutes', type=int, default=600, help='Time slice based on AUCTIONS_USERS.CREATED_AT (last minutes).')
    ap.add_argument('--lookback_seconds', type=str, default='15,30,60', help='Comma-separated windows to attribute P3 to prior ad click or P1.')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13at_infer_organic_clicks_from_p3_{args.round}.txt"
    wins = [int(x.strip()) for x in args.lookback_seconds.split(',') if x.strip()]

    paths = get_paths(args.round)

    with open(out, 'w') as fh:
        def w(s):
            fh.write(str(s)+"\n"); fh.flush(); print(s)

        w(f"Infer Organic Clicks from Product Page Arrivals — {args.round}")
        w(f"Window: last {args.window_minutes} minutes; lookbacks (seconds): {wins}")
        w("Assumptions: P3 auction indicates a product page view. If a P3 view is not preceded by a sponsored ad click within a short window, and follows a P1 auction closely, treat it as a likely organic arrival from search. This is a proxy; we do not observe organic click events.")

        # Load minimal columns
        au = pd.read_parquet(paths['auctions_users']).drop_duplicates()
        # Normalize user id column name across rounds
        if 'OPAQUE_USER_ID' in au.columns:
            au = au.rename(columns={'OPAQUE_USER_ID':'USER_ID'})
        elif 'USER_ID' not in au.columns:
            raise ValueError('Expected USER_ID or OPAQUE_USER_ID in auctions_users parquet')
        au = au[['AUCTION_ID','USER_ID','PLACEMENT','CREATED_AT']]
        cl = pd.read_parquet(paths['clicks'], columns=['AUCTION_ID','USER_ID','OCCURRED_AT']).drop_duplicates()

        # Time slice and placements 1 (search) and 3 (product)
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'], utc=True, errors='coerce')
        end_time = au['CREATED_AT'].max()
        start_time = end_time - pd.Timedelta(minutes=args.window_minutes)
        au = au[(au['CREATED_AT'] >= start_time) & (au['CREATED_AT'] <= end_time)]
        au['PLACEMENT'] = au['PLACEMENT'].astype(str)
        p1 = au[au['PLACEMENT'] == '1'][['AUCTION_ID','USER_ID','CREATED_AT']]
        p3 = au[au['PLACEMENT'] == '3'][['AUCTION_ID','USER_ID','CREATED_AT']]

        # Normalize times
        cl['OCCURRED_AT'] = pd.to_datetime(cl['OCCURRED_AT'], utc=True, errors='coerce')

        w(f"Counts: P1 auctions={len(p1):,}, P3 auctions={len(p3):,}, clicks={len(cl):,}")

        # Build compact event streams per user: only timestamps and types
        p1_evt = p1.assign(EVT='P1_AUC', TS=p1['CREATED_AT'])[ ['USER_ID','TS','EVT','AUCTION_ID'] ]
        p3_evt = p3.assign(EVT='P3_AUC', TS=p3['CREATED_AT'])[ ['USER_ID','TS','EVT','AUCTION_ID'] ]
        cl_evt = cl.assign(EVT='AD_CLK', TS=cl['OCCURRED_AT'])[ ['USER_ID','TS','EVT','AUCTION_ID'] ]
        stream = pd.concat([p1_evt, p3_evt, cl_evt], ignore_index=True)
        stream = stream.dropna(subset=['USER_ID','TS']).sort_values(['USER_ID','TS']).reset_index(drop=True)

        # For each P3, compute time since last AD_CLK and since last P1
        w("Indexing last-event times per user...")
        stream['last_adclk_ts'] = pd.NaT
        stream['last_p1_ts'] = pd.NaT
        stream['last_ev'] = None
        # Rolling last times per user
        last_ad = {}
        last_p1 = {}
        user = None
        for i, row in tqdm(stream.iterrows(), total=len(stream)):
            uid = row['USER_ID']
            if uid != user:
                last_ad = {}
                last_p1 = {}
                user = uid
                lad = None
                lp1 = None
            # write current lasts
            stream.at[i, 'last_adclk_ts'] = lad
            stream.at[i, 'last_p1_ts'] = lp1
            # update
            if row['EVT'] == 'AD_CLK':
                lad = row['TS']
            elif row['EVT'] == 'P1_AUC':
                lp1 = row['TS']

        p3_only = stream[stream['EVT'] == 'P3_AUC'].copy()
        # Ensure last_ts columns are datetime with NaT for missing
        p3_only['last_adclk_ts'] = pd.to_datetime(p3_only['last_adclk_ts'], utc=True, errors='coerce')
        p3_only['last_p1_ts'] = pd.to_datetime(p3_only['last_p1_ts'], utc=True, errors='coerce')
        p3_only['dt_last_adclk_s'] = (p3_only['TS'] - p3_only['last_adclk_ts']).dt.total_seconds()
        p3_only['dt_last_p1_s'] = (p3_only['TS'] - p3_only['last_p1_ts']).dt.total_seconds()

        # Classify by lookbacks
        for L in wins:
            # Ad-attributed if an ad click within L seconds
            p3_only[f'ad_attr_{L}s'] = (p3_only['dt_last_adclk_s'].notna()) & (p3_only['dt_last_adclk_s'] >= 0) & (p3_only['dt_last_adclk_s'] <= L)
            # Likely organic from search if a P1 within L seconds and no ad click between
            p3_only[f'org_from_p1_{L}s'] = (p3_only['dt_last_p1_s'].notna()) & (p3_only['dt_last_p1_s'] >= 0) & (p3_only['dt_last_p1_s'] <= L) & (~p3_only[f'ad_attr_{L}s'])

        # Summaries
        w("\n=== P3 arrivals: attribution vs likely organic (shares) ===")
        total_p3 = len(p3_only)
        w(f"Total P3: {total_p3:,}")
        for L in wins:
            ad_share = p3_only[f'ad_attr_{L}s'].mean()
            org_share = p3_only[f'org_from_p1_{L}s'].mean()
            w(f"L={L:>3}s  ad_attributed={ad_share:.4f}  likely_organic_from_P1={org_share:.4f}")

        # Time-gap distributions for likely organic cases
        w("\n=== Time gaps (P1 -> P3) for likely organic cases ===")
        for L in wins:
            sub = p3_only[p3_only[f'org_from_p1_{L}s']]
            w(f"L={L:>3}s  N={len(sub):,}")
            if not sub.empty:
                w(summarize(sub['dt_last_p1_s']).round(3).to_string())

        # P1-to-P3 organic proxy rates: for each P1, did a P3 occur within L seconds with no earlier ad click?
        w("\n=== P1->P3 organic proxy rates (no intervening ad click) ===")
        # Forward-looking next-event times per user
        stream = stream.sort_values(['USER_ID','TS']).reset_index(drop=True)
        stream['next_p3_ts'] = pd.NaT
        stream['next_ad_ts'] = pd.NaT
        for uid, grp in stream.groupby('USER_ID', sort=False):
            next_p3_time = pd.NaT
            next_ad_time = pd.NaT
            for idx in reversed(grp.index.tolist()):
                stream.at[idx, 'next_p3_ts'] = next_p3_time
                stream.at[idx, 'next_ad_ts'] = next_ad_time
                ev = stream.at[idx, 'EVT']
                ts = stream.at[idx, 'TS']
                if ev == 'P3_AUC':
                    next_p3_time = ts
                elif ev == 'AD_CLK':
                    next_ad_time = ts
        # Ensure datetime dtype
        stream['next_p3_ts'] = pd.to_datetime(stream['next_p3_ts'], utc=True, errors='coerce')
        stream['next_ad_ts'] = pd.to_datetime(stream['next_ad_ts'], utc=True, errors='coerce')
        p1_rows = stream[stream['EVT'] == 'P1_AUC'].copy()
        dt_p1_p3 = (p1_rows['next_p3_ts'] - p1_rows['TS']).dt.total_seconds()
        dt_p1_ad = (p1_rows['next_ad_ts'] - p1_rows['TS']).dt.total_seconds()
        for L in wins:
            cond_p3 = (dt_p1_p3.notna()) & (dt_p1_p3 >= 0) & (dt_p1_p3 <= L)
            cond_ad_before = (dt_p1_ad.notna()) & (dt_p1_ad >= 0) & (dt_p1_ad <= dt_p1_p3)
            rate = (cond_p3 & (~cond_ad_before)).mean()
            w(f"L={L:>3}s  P1->P3 organic proxy rate: {rate:.4f}  (N={len(p1_rows):,})")

        w(f"\nOutput saved to: {out}")


if __name__ == '__main__':
    main()
