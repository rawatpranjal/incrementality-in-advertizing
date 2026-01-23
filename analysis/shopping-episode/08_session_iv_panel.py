#!/usr/bin/env python3
"""
08_session_iv_panel.py
Aggregates auction-level panel to session-vendor level for IV estimation.
Creates instruments: MarginalWin (rank=K), MarginalLoss (rank=K+1), Close (both).
Merges with clicks and spend from events_with_sessions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "08_session_iv_panel.txt"

SESSION_GAPS = [1, 2, 3, 5, 7]

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("08_SESSION_IV_PANEL", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script aggregates auction-level marginal indicators to the session-vendor", f)
        log("level to create instruments for IV estimation. The key instrument is", f)
        log("MarginalWin_usv = count of auctions where vendor v barely won an impression", f)
        log("slot (rank = K) for user u in session s. We also create MarginalLoss_usv", f)
        log("(rank = K+1) as a placebo instrument and Eligible_usv as the total count of", f)
        log("auctions where vendor got an impression. The identifying assumption is that", f)
        log("conditional on being near the margin, winning vs losing is quasi-random.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD DATA
        # ============================================================
        log("LOADING DATA", f)
        log("-" * 40, f)

        log("\nLoading auction_panel...", f)
        auction_panel = pd.read_parquet(DATA_DIR / 'auction_panel.parquet')
        log(f"  Rows: {len(auction_panel):,}", f)

        log("\nLoading events_with_sessions...", f)
        events = pd.read_parquet(DATA_DIR / 'events_with_sessions.parquet')
        log(f"  Rows: {len(events):,}", f)

        log("\nLoading purchases_mapped...", f)
        purchases = pd.read_parquet(DATA_DIR / 'purchases_mapped.parquet')
        valid_purchases = purchases[purchases['is_post_click']].copy()
        log(f"  Valid purchases: {len(valid_purchases):,}", f)

        # ============================================================
        # 2. CREATE SESSION-AUCTION MAPPING
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("CREATING SESSION-AUCTION MAPPING", f)
        log("=" * 80, f)

        # Get click events with session IDs
        clicks = events[events['event_type'] == 'click'].copy()
        log(f"\nClick events: {len(clicks):,}", f)

        # We need to link auctions to sessions via the clicks that occurred
        # Get (user, auction, session) mapping from promoted_events
        promoted_events = pd.read_parquet(DATA_DIR / 'promoted_events.parquet')
        promoted_events['click_time'] = pd.to_datetime(promoted_events['click_time'])

        # For each gap threshold, assign session IDs to auctions via clicks
        # An auction belongs to a session if a click on that auction occurred in that session

        for gap_days in tqdm(SESSION_GAPS, desc="Processing session gaps"):
            log("", f)
            log("=" * 80, f)
            log(f"SESSION GAP: {gap_days} DAYS", f)
            log("=" * 80, f)

            session_col = f'session_id_{gap_days}d'

            # Get session IDs for clicks
            click_sessions = clicks[['user_id', 'vendor_id', 'timestamp', session_col, 'year_week']].copy()
            click_sessions = click_sessions.rename(columns={session_col: 'session_id'})

            # Merge promoted_events with session info
            # Match by user, vendor, and timestamp (within tolerance)
            promoted_events['click_date'] = promoted_events['click_time'].dt.date

            # Get unique (user, vendor, session) from click events
            user_vendor_sessions = click_sessions.groupby(['user_id', 'vendor_id', 'session_id', 'year_week']).size().reset_index(name='n_clicks')
            log(f"\nUnique (user, vendor, session): {len(user_vendor_sessions):,}", f)

            # ============================================================
            # Aggregate auction panel to (user, vendor, year_week)
            # ============================================================
            log("\nAggregating auction panel to (user, vendor, year_week)...", f)

            auction_agg = auction_panel.groupby(['user_id', 'vendor_id', 'year_week']).agg({
                'eligible': 'sum',
                'marginal_win': 'sum',
                'marginal_loss': 'sum',
                'close': 'sum',
                'clicked': 'sum',
                'auction_id': 'count'  # Total auctions participated in
            }).reset_index()

            auction_agg.columns = ['user_id', 'vendor_id', 'year_week',
                                   'Eligible', 'MarginalWin', 'MarginalLoss', 'Close',
                                   'Clicked_auctions', 'TotalAuctions']

            log(f"  Auction aggregates: {len(auction_agg):,} (user, vendor, week)", f)

            # ============================================================
            # Merge with session-level clicks and spend
            # ============================================================
            log("\nMerging with session-level clicks...", f)

            # Get clicks per (user, vendor, session, week)
            clicks_stv = click_sessions.groupby(['user_id', 'vendor_id', 'session_id', 'year_week']).size().reset_index(name='C')
            log(f"  Click aggregates: {len(clicks_stv):,}", f)

            # Get purchases per (user, vendor, session, week)
            purchase_events = events[events['event_type'] == 'purchase'].copy()
            purchase_events = purchase_events.rename(columns={session_col: 'session_id'})

            spend_stv = purchase_events.groupby(['user_id', 'vendor_id', 'session_id', 'year_week']).agg({
                'spend': 'sum'
            }).reset_index()
            spend_stv.columns = ['user_id', 'vendor_id', 'session_id', 'year_week', 'Y']
            log(f"  Spend aggregates: {len(spend_stv):,}", f)

            # ============================================================
            # Build IV panel
            # ============================================================
            log("\nBuilding IV panel...", f)

            # Start with clicks (defines which sessions have activity)
            panel_iv = clicks_stv.copy()

            # Merge with spend
            panel_iv = panel_iv.merge(
                spend_stv,
                on=['user_id', 'vendor_id', 'session_id', 'year_week'],
                how='left'
            )
            panel_iv['Y'] = panel_iv['Y'].fillna(0)

            # Merge with auction aggregates (by user, vendor, week)
            panel_iv = panel_iv.merge(
                auction_agg,
                on=['user_id', 'vendor_id', 'year_week'],
                how='left'
            )

            # Fill missing auction counts with 0
            for col in ['Eligible', 'MarginalWin', 'MarginalLoss', 'Close', 'Clicked_auctions', 'TotalAuctions']:
                panel_iv[col] = panel_iv[col].fillna(0).astype(int)

            # Create derived variables
            panel_iv['D'] = (panel_iv['Y'] > 0).astype(int)
            panel_iv['log_Y'] = np.log1p(panel_iv['Y'])

            # Create FE indices
            panel_iv['user_fe'] = pd.Categorical(panel_iv['user_id']).codes
            panel_iv['session_fe'] = pd.Categorical(panel_iv['session_id']).codes
            panel_iv['vendor_fe'] = pd.Categorical(panel_iv['vendor_id']).codes
            panel_iv['week_fe'] = pd.Categorical(panel_iv['year_week']).codes

            # ============================================================
            # Panel Summary
            # ============================================================
            log("\n--- Panel Summary ---", f)
            log(f"Observations: {len(panel_iv):,}", f)
            log(f"Unique users: {panel_iv['user_id'].nunique():,}", f)
            log(f"Unique sessions: {panel_iv['session_id'].nunique():,}", f)
            log(f"Unique vendors: {panel_iv['vendor_id'].nunique():,}", f)

            log("\n--- Instrument Summary ---", f)
            log(f"MarginalWin: sum={panel_iv['MarginalWin'].sum():,}, mean={panel_iv['MarginalWin'].mean():.3f}", f)
            log(f"MarginalLoss: sum={panel_iv['MarginalLoss'].sum():,}, mean={panel_iv['MarginalLoss'].mean():.3f}", f)
            log(f"Close: sum={panel_iv['Close'].sum():,}, mean={panel_iv['Close'].mean():.3f}", f)
            log(f"Eligible: sum={panel_iv['Eligible'].sum():,}, mean={panel_iv['Eligible'].mean():.3f}", f)

            log("\n--- Outcome Summary ---", f)
            log(f"C (clicks): mean={panel_iv['C'].mean():.3f}, max={panel_iv['C'].max()}", f)
            log(f"Y (spend): mean=${panel_iv['Y'].mean():.2f}, sum=${panel_iv['Y'].sum():,.2f}", f)
            log(f"D (converted): {panel_iv['D'].sum():,} ({panel_iv['D'].mean()*100:.2f}%)", f)

            log("\n--- Correlation Matrix (key variables) ---", f)
            corr_vars = ['C', 'Y', 'MarginalWin', 'MarginalLoss', 'Eligible']
            corr_matrix = panel_iv[corr_vars].corr()
            log(str(corr_matrix.round(3)), f)

            # ============================================================
            # First Stage Check: Does MarginalWin predict C?
            # ============================================================
            log("\n--- First Stage Preview ---", f)

            # Simple correlation between instrument and endogenous variable
            corr_z_c = panel_iv['MarginalWin'].corr(panel_iv['C'])
            corr_z_y = panel_iv['MarginalWin'].corr(panel_iv['Y'])
            log(f"Corr(MarginalWin, C): {corr_z_c:.4f}", f)
            log(f"Corr(MarginalWin, Y): {corr_z_y:.4f}", f)

            # Conditional means
            z_high = panel_iv[panel_iv['MarginalWin'] > 0]
            z_zero = panel_iv[panel_iv['MarginalWin'] == 0]
            log(f"\nC | MarginalWin > 0: mean={z_high['C'].mean():.3f} (n={len(z_high):,})", f)
            log(f"C | MarginalWin = 0: mean={z_zero['C'].mean():.3f} (n={len(z_zero):,})", f)
            log(f"Y | MarginalWin > 0: mean=${z_high['Y'].mean():.2f}", f)
            log(f"Y | MarginalWin = 0: mean=${z_zero['Y'].mean():.2f}", f)

            # ============================================================
            # Save Panel
            # ============================================================
            output_file = DATA_DIR / f'panel_iv_{gap_days}d.parquet'
            panel_iv.to_parquet(output_file, index=False)
            log(f"\nSaved to {output_file}", f)
            log(f"Columns: {list(panel_iv.columns)}", f)

        # ============================================================
        # OVERALL SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("SESSION IV PANEL CONSTRUCTION COMPLETE", f)
        log("=" * 80, f)

        log("\n--- Output Files ---", f)
        for fp in sorted(DATA_DIR.glob('panel_iv_*.parquet')):
            size_mb = fp.stat().st_size / 1e6
            panel = pd.read_parquet(fp)
            log(f"  {fp.name}: {len(panel):,} rows, {size_mb:.2f} MB", f)

        log("\n--- Variable Definitions ---", f)
        log("C: Clicks on vendor v by user u in session s", f)
        log("Y: Spend on vendor v by user u in session s (post-click)", f)
        log("D: Conversion indicator (1 if Y > 0)", f)
        log("MarginalWin: Count of auctions at rank = K (instrument)", f)
        log("MarginalLoss: Count of auctions at rank = K+1 (placebo)", f)
        log("Close: MarginalWin + MarginalLoss (bandwidth check)", f)
        log("Eligible: Count of auctions where vendor got impression (rank <= K)", f)

        log("\n--- Identification Strategy ---", f)
        log("Instrument: Z = MarginalWin", f)
        log("First Stage: C ~ Z | FE", f)
        log("Reduced Form: Y ~ Z | FE", f)
        log("2SLS: Y ~ C_hat | FE, where C_hat instrumented by Z", f)
        log("", f)
        log("Exclusion Restriction: MarginalWin affects Y only through C", f)
        log("(vendors at rank K get impression -> can be clicked -> may buy)", f)

        log("", f)
        log("=" * 80, f)
        log("08_SESSION_IV_PANEL COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
