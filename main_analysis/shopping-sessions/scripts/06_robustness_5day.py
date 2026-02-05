#!/usr/bin/env python3
"""
06_robustness_5day.py - Re-sessionize with 5-day Gap for Robustness

This script re-creates sessions using a 5-day (120-hour) inactivity gap
instead of the baseline 3-day (72-hour) gap. This provides a robustness
check on the session definition.

Unit of Analysis: User-session (5-day definition)
Output:
- data/sessions_5day.parquet: Sessions with 5-day gap
- data/session_events_5day.parquet: Events with 5-day session IDs
- results/06_robustness_5day.txt: Comparison statistics
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PATHS
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "0_data_pull" / "data"
OUTPUT_DIR = BASE_DIR / "0_data_pull" / "data"
RESULTS_DIR = BASE_DIR / "results"

SESSION_GAP_HOURS_5DAY = 120  # 5 days


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def load_session_events(f):
    """Load the session events from the original sessionization."""
    log("\nLOADING DATA", f)

    # Load session events (already has user_id and event_time)
    session_events = pd.read_parquet(DATA_DIR / 'session_events.parquet')
    log(f"  session_events: {len(session_events):,} rows", f)

    # Also load original sessions for comparison
    sessions_3day = pd.read_parquet(DATA_DIR / 'sessions.parquet')
    log(f"  sessions (3-day): {len(sessions_3day):,} rows", f)

    return session_events, sessions_3day


def reassign_sessions(events, gap_hours, f):
    """Re-assign session IDs based on new inactivity gap."""
    log(f"\nREASSIGNING SESSIONS (gap={gap_hours}h)", f)

    events = events.copy()
    events['event_time'] = pd.to_datetime(events['event_time'])
    events = events.sort_values(['user_id', 'event_time'])

    events['time_diff'] = events.groupby('user_id')['event_time'].diff()
    events['time_diff_hours'] = events['time_diff'].dt.total_seconds() / 3600
    events['new_session'] = (events['time_diff_hours'] > gap_hours) | (events['time_diff'].isna())
    events['session_num'] = events.groupby('user_id')['new_session'].cumsum()
    events['session_id_5day'] = events['user_id'] + '_5d_' + events['session_num'].astype(str)

    n_sessions = events['session_id_5day'].nunique()
    n_users = events['user_id'].nunique()
    log(f"  Users: {n_users:,}", f)
    log(f"  Sessions (5-day): {n_sessions:,}", f)
    log(f"  Sessions per user: {n_sessions/n_users:.2f}", f)

    return events


def aggregate_sessions(events, f):
    """Create session-level aggregates."""
    log("\nAGGREGATING SESSIONS (5-day)", f)

    # Basic aggregations
    sessions = events.groupby('session_id_5day').agg(
        user_id=('user_id', 'first'),
        session_start=('event_time', 'min'),
        session_end=('event_time', 'max'),
    ).reset_index()

    sessions = sessions.rename(columns={'session_id_5day': 'session_id'})

    # Duration
    sessions['session_duration_hours'] = (
        sessions['session_end'] - sessions['session_start']
    ).dt.total_seconds() / 3600

    # Event counts by type
    event_counts = events.groupby(['session_id_5day', 'event_type']).size().unstack(fill_value=0).reset_index()
    event_counts = event_counts.rename(columns={'session_id_5day': 'session_id'})
    for col in ['auction', 'impression', 'click', 'purchase']:
        if col not in event_counts.columns:
            event_counts[col] = 0
    event_counts = event_counts.rename(columns={
        'auction': 'n_auctions',
        'impression': 'n_impressions',
        'click': 'n_clicks',
        'purchase': 'n_purchases'
    })
    sessions = sessions.merge(event_counts[['session_id', 'n_auctions', 'n_impressions', 'n_clicks', 'n_purchases']],
                               on='session_id', how='left')

    # Spend from purchases
    purchases = events[events['event_type'] == 'purchase']
    spend_agg = purchases.groupby('session_id_5day')['amount'].sum().reset_index()
    spend_agg.columns = ['session_id', 'total_spend']
    sessions = sessions.merge(spend_agg, on='session_id', how='left')
    sessions['total_spend'] = sessions['total_spend'].fillna(0)
    sessions['purchased'] = (sessions['n_purchases'] > 0).astype(int)

    # Product diversity
    impressions = events[events['event_type'] == 'impression']
    imp_diversity = impressions.groupby('session_id_5day').agg(
        n_products_impressed=('product_id', 'nunique'),
        n_vendors_impressed=('vendor_id', 'nunique')
    ).reset_index()
    imp_diversity.columns = ['session_id', 'n_products_impressed', 'n_vendors_impressed']
    sessions = sessions.merge(imp_diversity, on='session_id', how='left')

    for col in ['n_products_impressed', 'n_vendors_impressed']:
        sessions[col] = sessions[col].fillna(0).astype(int)

    log(f"  Created {len(sessions):,} session records", f)

    return sessions


def compare_sessions(sessions_3day, sessions_5day, f):
    """Compare 3-day and 5-day sessionizations."""
    log("\n" + "="*80, f)
    log("SESSION DEFINITION COMPARISON: 3-DAY vs 5-DAY", f)
    log("="*80, f)

    n_users_3d = sessions_3day['user_id'].nunique()
    n_users_5d = sessions_5day['user_id'].nunique()

    comparison = {
        'Metric': [
            'Total Sessions',
            'Unique Users',
            'Sessions per User',
            'Mean Duration (hours)',
            'Median Duration (hours)',
            'P95 Duration (hours)',
            'Mean Impressions',
            'Mean Clicks',
            'Purchase Rate',
            'Mean Spend (cents)',
            'Sessions with Purchases'
        ],
        '3-Day Gap': [
            len(sessions_3day),
            n_users_3d,
            len(sessions_3day) / n_users_3d,
            sessions_3day['session_duration_hours'].mean(),
            sessions_3day['session_duration_hours'].median(),
            sessions_3day['session_duration_hours'].quantile(0.95),
            sessions_3day['n_impressions'].mean(),
            sessions_3day['n_clicks'].mean(),
            sessions_3day['purchased'].mean(),
            sessions_3day['total_spend'].mean(),
            sessions_3day['purchased'].sum()
        ],
        '5-Day Gap': [
            len(sessions_5day),
            n_users_5d,
            len(sessions_5day) / n_users_5d,
            sessions_5day['session_duration_hours'].mean(),
            sessions_5day['session_duration_hours'].median(),
            sessions_5day['session_duration_hours'].quantile(0.95),
            sessions_5day['n_impressions'].mean(),
            sessions_5day['n_clicks'].mean(),
            sessions_5day['purchased'].mean(),
            sessions_5day['total_spend'].mean(),
            sessions_5day['purchased'].sum()
        ]
    }

    df_compare = pd.DataFrame(comparison)
    df_compare['Change (%)'] = (
        (df_compare['5-Day Gap'].astype(float) - df_compare['3-Day Gap'].astype(float))
        / df_compare['3-Day Gap'].astype(float) * 100
    )

    log("\n", f)
    log(f"{'Metric':<30} {'3-Day Gap':>15} {'5-Day Gap':>15} {'Change (%)':>12}", f)
    log(f"{'-'*30} {'-'*15} {'-'*15} {'-'*12}", f)

    for _, row in df_compare.iterrows():
        val_3d = row['3-Day Gap']
        val_5d = row['5-Day Gap']
        change = row['Change (%)']

        if isinstance(val_3d, float):
            if val_3d > 1000:
                val_3d_str = f"{val_3d:,.0f}"
                val_5d_str = f"{val_5d:,.0f}"
            elif val_3d < 1:
                val_3d_str = f"{val_3d:.4f}"
                val_5d_str = f"{val_5d:.4f}"
            else:
                val_3d_str = f"{val_3d:.2f}"
                val_5d_str = f"{val_5d:.2f}"
        else:
            val_3d_str = f"{val_3d:,}"
            val_5d_str = f"{val_5d:,}"

        log(f"{row['Metric']:<30} {val_3d_str:>15} {val_5d_str:>15} {change:>11.1f}%", f)

    return df_compare


def analyze_session_merging(events, f):
    """Analyze how sessions merge when using 5-day gap."""
    log("\n" + "="*80, f)
    log("SESSION MERGING ANALYSIS", f)
    log("="*80, f)

    # Count 3-day sessions per 5-day session
    session_mapping = events.groupby('session_id_5day')['session_id'].nunique().reset_index()
    session_mapping.columns = ['session_id_5day', 'n_3day_sessions_merged']

    log("\nNumber of 3-day sessions merged into each 5-day session:", f)
    log(f"  1 (no merging): {(session_mapping['n_3day_sessions_merged'] == 1).sum():,} "
        f"({(session_mapping['n_3day_sessions_merged'] == 1).mean()*100:.1f}%)", f)
    log(f"  2: {(session_mapping['n_3day_sessions_merged'] == 2).sum():,} "
        f"({(session_mapping['n_3day_sessions_merged'] == 2).mean()*100:.1f}%)", f)
    log(f"  3+: {(session_mapping['n_3day_sessions_merged'] >= 3).sum():,} "
        f"({(session_mapping['n_3day_sessions_merged'] >= 3).mean()*100:.1f}%)", f)

    log(f"\n  Mean 3-day sessions per 5-day session: {session_mapping['n_3day_sessions_merged'].mean():.2f}", f)
    log(f"  Max 3-day sessions merged: {session_mapping['n_3day_sessions_merged'].max()}", f)

    return session_mapping


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / '06_robustness_5day.txt'

    with open(output_file, 'w') as f:
        log("="*80, f)
        log("06_ROBUSTNESS_5DAY - Re-sessionize with 5-Day Gap", f)
        log("="*80, f)
        log(f"Session gap threshold: {SESSION_GAP_HOURS_5DAY} hours (5 days)", f)
        log(f"Data directory: {DATA_DIR}", f)
        log(f"Output directory: {OUTPUT_DIR}", f)

        # Load data
        session_events, sessions_3day = load_session_events(f)

        # Re-assign sessions with 5-day gap
        events_5day = reassign_sessions(session_events, SESSION_GAP_HOURS_5DAY, f)

        # Aggregate to session level
        sessions_5day = aggregate_sessions(events_5day, f)

        # Compare sessionizations
        comparison = compare_sessions(sessions_3day, sessions_5day, f)

        # Analyze merging
        merging = analyze_session_merging(events_5day, f)

        # Save outputs
        log("\n" + "="*80, f)
        log("SAVING OUTPUT", f)
        log("="*80, f)

        # Save 5-day sessions
        sessions_5day_path = OUTPUT_DIR / 'sessions_5day.parquet'
        sessions_5day.to_parquet(sessions_5day_path, index=False)
        log(f"  {sessions_5day_path}: {len(sessions_5day):,} rows", f)

        # Save session events with 5-day IDs
        events_5day_path = OUTPUT_DIR / 'session_events_5day.parquet'
        events_5day.to_parquet(events_5day_path, index=False)
        log(f"  {events_5day_path}: {len(events_5day):,} rows", f)

        log(f"\nOutput saved to: {output_file}", f)
