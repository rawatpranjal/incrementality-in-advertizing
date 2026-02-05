#!/usr/bin/env python3
"""
07_robustness_7day.py - Re-sessionize with 7-day Gap for Robustness

This script re-creates sessions using a 7-day (168-hour) inactivity gap
in parallel to the 5-day robustness script.

Outputs:
- data/sessions_7day.parquet: Sessions with 7-day gap
- data/session_events_7day.parquet: Events with 7-day session IDs
- results/07_robustness_7day.txt: Comparison statistics
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "0_data_pull" / "data"
OUTPUT_DIR = DATA_DIR
RESULTS_DIR = BASE_DIR / "results"

SESSION_GAP_HOURS = 168  # 7 days


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    of = RESULTS_DIR / '07_robustness_7day.txt'
    with open(of, 'w') as f:
        log("=" * 80, f)
        log("07_ROBUSTNESS_7DAY - Re-sessionize with 7-Day Gap", f)
        log("=" * 80, f)
        log(f"Session gap threshold: {SESSION_GAP_HOURS} hours (7 days)", f)
        log(f"Data directory: {DATA_DIR}", f)

        # Load events and 3-day sessions
        session_events = pd.read_parquet(DATA_DIR / 'session_events.parquet')
        sessions_3day = pd.read_parquet(DATA_DIR / 'sessions.parquet')
        log(f"session_events: {len(session_events):,}", f)
        log(f"sessions (3-day): {len(sessions_3day):,}", f)

        # Reassign sessions
        log(f"\nREASSIGNING SESSIONS (gap={SESSION_GAP_HOURS}h)", f)
        ev = session_events.copy()
        ev['event_time'] = pd.to_datetime(ev['event_time'])
        ev = ev.sort_values(['user_id', 'event_time'])
        ev['time_diff'] = ev.groupby('user_id')['event_time'].diff()
        ev['time_diff_hours'] = ev['time_diff'].dt.total_seconds() / 3600
        ev['new_session'] = (ev['time_diff_hours'] > SESSION_GAP_HOURS) | (ev['time_diff'].isna())
        ev['session_num'] = ev.groupby('user_id')['new_session'].cumsum()
        ev['session_id_7day'] = ev['user_id'] + '_7d_' + ev['session_num'].astype(str)

        n_users = ev['user_id'].nunique()
        n_sessions = ev['session_id_7day'].nunique()
        log(f"Users: {n_users:,}", f)
        log(f"Sessions (7-day): {n_sessions:,}", f)

        # Aggregate to session-level
        log("\nAGGREGATING SESSIONS (7-day)", f)
        sessions = ev.groupby('session_id_7day').agg(
            user_id=('user_id', 'first'),
            session_start=('event_time', 'min'),
            session_end=('event_time', 'max'),
        ).reset_index().rename(columns={'session_id_7day': 'session_id'})

        # Duration
        sessions['session_duration_hours'] = (sessions['session_end'] - sessions['session_start']).dt.total_seconds() / 3600

        # Event counts
        counts = ev.groupby(['session_id_7day', 'event_type']).size().unstack(fill_value=0).reset_index()
        counts = counts.rename(columns={'session_id_7day': 'session_id'})
        for c in ['auction', 'impression', 'click', 'purchase']:
            if c not in counts.columns:
                counts[c] = 0
        counts = counts.rename(columns={'auction': 'n_auctions', 'impression': 'n_impressions', 'click': 'n_clicks', 'purchase': 'n_purchases'})
        sessions = sessions.merge(counts[['session_id', 'n_auctions', 'n_impressions', 'n_clicks', 'n_purchases']], on='session_id', how='left')

        # Spend
        purchases = ev[ev['event_type'] == 'purchase']
        spend = purchases.groupby('session_id_7day')['amount'].sum().reset_index().rename(columns={'session_id_7day': 'session_id', 'amount': 'total_spend'})
        sessions = sessions.merge(spend, on='session_id', how='left')
        sessions['total_spend'] = sessions['total_spend'].fillna(0)
        sessions['purchased'] = (sessions['n_purchases'] > 0).astype(int)

        # Diversity
        imps = ev[ev['event_type'] == 'impression']
        div = imps.groupby('session_id_7day').agg(n_products_impressed=('product_id', 'nunique'), n_vendors_impressed=('vendor_id', 'nunique')).reset_index()
        div = div.rename(columns={'session_id_7day': 'session_id'})
        sessions = sessions.merge(div, on='session_id', how='left')
        for c in ['n_products_impressed', 'n_vendors_impressed']:
            sessions[c] = sessions[c].fillna(0).astype(int)

        log(f"Created {len(sessions):,} 7-day session rows", f)

        # Save outputs
        sessions_7day_path = OUTPUT_DIR / 'sessions_7day.parquet'
        events_7day_path = OUTPUT_DIR / 'session_events_7day.parquet'
        sessions.to_parquet(sessions_7day_path, index=False)
        ev_out = ev.copy()
        if 'session_id' in ev_out.columns:
            ev_out = ev_out.drop(columns=['session_id'])
        ev_out = ev_out.rename(columns={'session_id_7day': 'session_id'})
        ev_out.to_parquet(events_7day_path, index=False)
        log(f"Saved: {sessions_7day_path}", f)
        log(f"Saved: {events_7day_path}", f)

        log(f"\nOutput saved to: {of}", f)


if __name__ == '__main__':
    main()
