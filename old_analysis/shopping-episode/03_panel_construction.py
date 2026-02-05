#!/usr/bin/env python3
"""
03_panel_construction.py
Builds estimation panels for regression analysis:
1. Panel A: (u, t, v) - User × Week × Vendor
2. Panel B: (s, t, v) - Session-Week × Vendor (for each gap threshold)

Variables:
- C = sponsored click count
- Y = spend (promoted-linked only)
- I = impression count
- Controls: avg_rank, share_rank1, avg_pacing, avg_quality, avg_final_bid
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = Path("/Users/pranjal/Code/marketplace-incrementality/eda/data")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "03_panel_construction.txt"

SESSION_GAPS = [1, 2, 3, 5, 7]

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("03_PANEL_CONSTRUCTION", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script constructs the regression-ready panels at two levels of aggregation.", f)
        log("Panel A aggregates to (user, week, vendor) where the estimand is the effect of", f)
        log("an additional click on weekly vendor spend, controlling for user heterogeneity,", f)
        log("time shocks, and vendor popularity via fixed effects. Panel B aggregates to", f)
        log("(session, week, vendor) where session-level fixed effects can absorb shopping", f)
        log("intent that is constant within a session but varies across sessions. The session", f)
        log("gap threshold (1/2/3/5/7 days) determines how much purchase history is grouped", f)
        log("together, testing whether results are sensitive to session definition.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD CANONICAL TABLES
        # ============================================================
        log("Loading canonical tables...", f)

        promoted_events = pd.read_parquet(DATA_DIR / 'promoted_events.parquet')
        purchases_mapped = pd.read_parquet(DATA_DIR / 'purchases_mapped.parquet')
        events_with_sessions = pd.read_parquet(DATA_DIR / 'events_with_sessions.parquet')

        log(f"Promoted events: {len(promoted_events):,}", f)
        log(f"Purchases mapped: {len(purchases_mapped):,}", f)
        log(f"Events with sessions: {len(events_with_sessions):,}", f)

        # Add week info to promoted_events
        promoted_events['week'] = pd.to_datetime(promoted_events['click_time']).dt.isocalendar().week
        promoted_events['year'] = pd.to_datetime(promoted_events['click_time']).dt.year
        promoted_events['year_week'] = promoted_events['year'].astype(str) + '_W' + promoted_events['week'].astype(str).str.zfill(2)

        # Add week info to purchases
        purchases_valid = purchases_mapped[purchases_mapped['is_post_click']].copy()
        purchases_valid['week'] = pd.to_datetime(purchases_valid['purchase_time']).dt.isocalendar().week
        purchases_valid['year'] = pd.to_datetime(purchases_valid['purchase_time']).dt.year
        purchases_valid['year_week'] = purchases_valid['year'].astype(str) + '_W' + purchases_valid['year'].astype(str).str.zfill(2)

        log(f"Valid purchases: {len(purchases_valid):,}", f)
        log(f"Total valid spend: ${purchases_valid['spend'].sum():,.2f}", f)

        # ============================================================
        # 2. PANEL A: USER × WEEK × VENDOR
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("PANEL A: USER × WEEK × VENDOR", f)
        log("=" * 80, f)

        # Aggregate clicks to (user, week, vendor)
        clicks_utv = promoted_events.groupby(['user_id', 'year_week', 'vendor_id']).agg({
            'click_id': 'count',
            'ranking': ['mean', 'min'],
            'is_winner': 'mean',
            'final_bid': 'mean',
            'quality': 'mean',
            'pacing': 'mean',
            'conversion_rate': 'mean',
            'price': 'mean'
        }).reset_index()

        clicks_utv.columns = ['user_id', 'year_week', 'vendor_id',
                              'C', 'avg_rank', 'min_rank', 'share_winner',
                              'avg_final_bid', 'avg_quality', 'avg_pacing',
                              'avg_conversion_rate', 'avg_price']

        # Share rank=1
        rank1_counts = promoted_events[promoted_events['ranking'] == 1].groupby(
            ['user_id', 'year_week', 'vendor_id']
        ).size().reset_index(name='rank1_clicks')

        clicks_utv = clicks_utv.merge(rank1_counts, on=['user_id', 'year_week', 'vendor_id'], how='left')
        clicks_utv['rank1_clicks'] = clicks_utv['rank1_clicks'].fillna(0)
        clicks_utv['share_rank1'] = clicks_utv['rank1_clicks'] / clicks_utv['C']

        log(f"Click aggregates: {len(clicks_utv):,} (u,t,v) observations", f)

        # Aggregate spend to (user, week, vendor)
        spend_utv = purchases_valid.groupby(['user_id', 'year_week', 'click_vendor_id']).agg({
            'spend': 'sum',
            'purchase_id': 'count'
        }).reset_index()
        spend_utv.columns = ['user_id', 'year_week', 'vendor_id', 'Y', 'n_purchases']

        log(f"Spend aggregates: {len(spend_utv):,} (u,t,v) observations", f)

        # Merge clicks and spend
        panel_utv = clicks_utv.merge(
            spend_utv,
            on=['user_id', 'year_week', 'vendor_id'],
            how='outer'
        )

        panel_utv['C'] = panel_utv['C'].fillna(0).astype(int)
        panel_utv['Y'] = panel_utv['Y'].fillna(0)
        panel_utv['n_purchases'] = panel_utv['n_purchases'].fillna(0).astype(int)
        panel_utv['D'] = (panel_utv['Y'] > 0).astype(int)
        panel_utv['log_Y'] = np.log1p(panel_utv['Y'])

        log(f"\nPanel A dimensions: {len(panel_utv):,} observations", f)
        log(f"Unique users: {panel_utv['user_id'].nunique():,}", f)
        log(f"Unique weeks: {panel_utv['year_week'].nunique()}", f)
        log(f"Unique vendors: {panel_utv['vendor_id'].nunique():,}", f)

        log("\n--- Panel A Summary Statistics ---", f)
        log(f"\nClick distribution (C):", f)
        log(str(panel_utv['C'].describe()), f)
        log(f"Zero clicks: {(panel_utv['C'] == 0).mean()*100:.1f}%", f)

        log(f"\nSpend distribution (Y):", f)
        log(str(panel_utv['Y'].describe()), f)
        log(f"Zero spend: {(panel_utv['Y'] == 0).mean()*100:.1f}%", f)

        log(f"\nConversion rate: {panel_utv['D'].mean()*100:.2f}%", f)

        panel_utv.to_parquet(DATA_DIR / 'panel_utv.parquet', index=False)
        log(f"\nSaved Panel A to {DATA_DIR / 'panel_utv.parquet'}", f)

        # ============================================================
        # 3. PANEL B: SESSION-WEEK × VENDOR
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("PANEL B: SESSION-WEEK × VENDOR", f)
        log("=" * 80, f)

        panels_stv = {}

        for gap_days in tqdm(SESSION_GAPS, desc="Building session panels"):
            session_col = f'session_id_{gap_days}d'

            clicks_events = events_with_sessions[events_with_sessions['event_type'] == 'click'].copy()

            clicks_stv = clicks_events.groupby([session_col, 'year_week', 'vendor_id']).agg({
                'user_id': 'first',
                'product_id': 'count'
            }).reset_index()
            clicks_stv.columns = ['session_id', 'year_week', 'vendor_id', 'user_id', 'C']

            purchase_events = events_with_sessions[events_with_sessions['event_type'] == 'purchase'].copy()

            spend_stv = purchase_events.groupby([session_col, 'year_week', 'vendor_id']).agg({
                'spend': 'sum'
            }).reset_index()
            spend_stv.columns = ['session_id', 'year_week', 'vendor_id', 'Y']

            panel_stv = clicks_stv.merge(
                spend_stv,
                on=['session_id', 'year_week', 'vendor_id'],
                how='outer'
            )

            panel_stv['C'] = panel_stv['C'].fillna(0).astype(int)
            panel_stv['Y'] = panel_stv['Y'].fillna(0)
            panel_stv['D'] = (panel_stv['Y'] > 0).astype(int)
            panel_stv['log_Y'] = np.log1p(panel_stv['Y'])

            if panel_stv['user_id'].isna().any():
                panel_stv['user_id'] = panel_stv['user_id'].fillna(
                    panel_stv['session_id'].str.rsplit('_S', n=1).str[0]
                )

            panels_stv[gap_days] = panel_stv

            log(f"\n{gap_days}-day gap panel: {len(panel_stv):,} (s,t,v) observations", f)
            log(f"  Sessions: {panel_stv['session_id'].nunique():,}", f)
            log(f"  Users: {panel_stv['user_id'].nunique():,}", f)
            log(f"  Zero spend: {(panel_stv['Y'] == 0).mean()*100:.1f}%", f)

        for gap_days, panel in panels_stv.items():
            filename = f'panel_stv_{gap_days}d.parquet'
            panel.to_parquet(DATA_DIR / filename, index=False)
            log(f"Saved {filename}: {len(panel):,} rows", f)

        # ============================================================
        # 4. ADD IMPRESSIONS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("ADDING IMPRESSIONS", f)
        log("=" * 80, f)

        impressions = pd.read_parquet(SOURCE_DIR / 'impressions_365d.parquet')
        impressions['impression_time'] = pd.to_datetime(impressions['OCCURRED_AT'])
        impressions['week'] = impressions['impression_time'].dt.isocalendar().week
        impressions['year'] = impressions['impression_time'].dt.year
        impressions['year_week'] = impressions['year'].astype(str) + '_W' + impressions['week'].astype(str).str.zfill(2)

        log(f"Total impressions: {len(impressions):,}", f)

        impressions_utv = impressions.groupby(['USER_ID', 'year_week', 'VENDOR_ID']).size().reset_index(name='I')
        impressions_utv.columns = ['user_id', 'year_week', 'vendor_id', 'I']

        log(f"Impression aggregates: {len(impressions_utv):,} (u,t,v) observations", f)

        panel_utv = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
        panel_utv = panel_utv.merge(
            impressions_utv,
            on=['user_id', 'year_week', 'vendor_id'],
            how='left'
        )
        panel_utv['I'] = panel_utv['I'].fillna(0).astype(int)

        panel_utv.to_parquet(DATA_DIR / 'panel_utv.parquet', index=False)
        log(f"Updated Panel A with impressions: {len(panel_utv):,} rows", f)

        # ============================================================
        # 5. CREATE FIXED EFFECT INDICES
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("CREATING FIXED EFFECT INDICES", f)
        log("=" * 80, f)

        panel_utv = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
        panel_utv['user_fe'] = pd.Categorical(panel_utv['user_id']).codes
        panel_utv['week_fe'] = pd.Categorical(panel_utv['year_week']).codes
        panel_utv['vendor_fe'] = pd.Categorical(panel_utv['vendor_id']).codes

        log(f"User FE levels: {panel_utv['user_fe'].nunique()}", f)
        log(f"Week FE levels: {panel_utv['week_fe'].nunique()}", f)
        log(f"Vendor FE levels: {panel_utv['vendor_fe'].nunique()}", f)

        panel_utv.to_parquet(DATA_DIR / 'panel_utv.parquet', index=False)

        for gap_days in SESSION_GAPS:
            filename = f'panel_stv_{gap_days}d.parquet'
            panel = pd.read_parquet(DATA_DIR / filename)

            panel['session_fe'] = pd.Categorical(panel['session_id']).codes
            panel['week_fe'] = pd.Categorical(panel['year_week']).codes
            panel['vendor_fe'] = pd.Categorical(panel['vendor_id']).codes
            panel['user_fe'] = pd.Categorical(panel['user_id']).codes

            panel.to_parquet(DATA_DIR / filename, index=False)
            log(f"{filename}: session_fe={panel['session_fe'].nunique()}, week_fe={panel['week_fe'].nunique()}, vendor_fe={panel['vendor_fe'].nunique()}", f)

        # ============================================================
        # 6. SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("PANEL CONSTRUCTION COMPLETE", f)
        log("=" * 80, f)

        log("\n--- Panel A: (u, t, v) ---", f)
        panel_utv = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
        log(f"Observations: {len(panel_utv):,}", f)
        log(f"Columns: {list(panel_utv.columns)}", f)
        log(f"C range: [{panel_utv['C'].min()}, {panel_utv['C'].max()}], mean={panel_utv['C'].mean():.2f}", f)
        log(f"Y range: [${panel_utv['Y'].min():.2f}, ${panel_utv['Y'].max():.2f}], mean=${panel_utv['Y'].mean():.2f}", f)
        log(f"I range: [{panel_utv['I'].min()}, {panel_utv['I'].max()}], mean={panel_utv['I'].mean():.2f}", f)

        log("\n--- Panel B: (s, t, v) by gap threshold ---", f)
        for gap_days in SESSION_GAPS:
            panel = pd.read_parquet(DATA_DIR / f'panel_stv_{gap_days}d.parquet')
            log(f"\n{gap_days}-day gap:", f)
            log(f"  Observations: {len(panel):,}", f)
            log(f"  Sessions: {panel['session_id'].nunique():,}", f)
            log(f"  Conversion rate: {panel['D'].mean()*100:.2f}%", f)

        log("\n--- Output Files ---", f)
        for fp in sorted(DATA_DIR.glob('panel_*.parquet')):
            size_mb = fp.stat().st_size / 1e6
            log(f"  {fp.name}: {size_mb:.1f} MB", f)

if __name__ == "__main__":
    main()
