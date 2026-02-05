#!/usr/bin/env python3
"""
01_sessionize.py - Create shopping sessions from raw event data

Session Definition:
- 72-hour (3-day) inactivity gap triggers new session
- Session = all user activity between gaps
- Events: auctions, impressions, clicks, purchases

Unit of Analysis: User-session
Output:
- data/sessions.parquet: Session-level aggregates
- data/session_events.parquet: Event-level with session_id
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

SESSION_GAP_HOURS = 72  # 3 days


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def load_data(f):
    """Load all parquet files."""
    log("\nLOADING DATA", f)
    data = {}
    files = ['auctions_users', 'auctions_results', 'impressions', 'clicks', 'purchases', 'catalog']
    for name in files:
        path = DATA_DIR / f'{name}.parquet'
        if path.exists():
            data[name] = pd.read_parquet(path)
            log(f"  {name}: {len(data[name]):,} rows, {data[name].shape[1]} cols", f)
        else:
            log(f"  {name}: FILE NOT FOUND", f)
            data[name] = pd.DataFrame()
    return data


def create_unified_event_stream(data, f):
    """Combine all events into single timeline per user."""
    log("\n" + "="*80, f)
    log("CREATING UNIFIED EVENT STREAM", f)
    log("="*80, f)

    events = []

    # Auctions
    if len(data['auctions_users']) > 0:
        au_cols = ['user_id', 'auction_id', 'auction_time']
        if 'placement' in data['auctions_users'].columns:
            au_cols.append('placement')
        au = data['auctions_users'][au_cols].copy()
        au['event_type'] = 'auction'
        au['event_time'] = pd.to_datetime(au['auction_time'])
        au['product_id'] = None
        au['vendor_id'] = None
        au['amount'] = None
        if 'placement' not in au.columns:
            au['placement'] = None
        events.append(au[['user_id', 'event_time', 'event_type', 'auction_id', 'product_id', 'vendor_id', 'placement', 'amount']])
        log(f"  Auctions: {len(au):,}", f)

    # Impressions
    if len(data['impressions']) > 0:
        imp = data['impressions'][['user_id', 'impression_time', 'auction_id', 'product_id', 'vendor_id']].copy()
        imp['event_type'] = 'impression'
        imp['event_time'] = pd.to_datetime(imp['impression_time'])
        imp['placement'] = None
        imp['amount'] = None
        events.append(imp[['user_id', 'event_time', 'event_type', 'auction_id', 'product_id', 'vendor_id', 'placement', 'amount']])
        log(f"  Impressions: {len(imp):,}", f)

    # Clicks
    if len(data['clicks']) > 0:
        clk = data['clicks'][['user_id', 'click_time', 'auction_id', 'product_id', 'vendor_id']].copy()
        clk['event_type'] = 'click'
        clk['event_time'] = pd.to_datetime(clk['click_time'])
        clk['placement'] = None
        clk['amount'] = None
        events.append(clk[['user_id', 'event_time', 'event_type', 'auction_id', 'product_id', 'vendor_id', 'placement', 'amount']])
        log(f"  Clicks: {len(clk):,}", f)

    # Purchases
    if len(data['purchases']) > 0:
        pur = data['purchases'][['user_id', 'purchase_time', 'product_id', 'quantity', 'unit_price']].copy()
        pur['event_type'] = 'purchase'
        pur['event_time'] = pd.to_datetime(pur['purchase_time'])
        pur['auction_id'] = None
        pur['vendor_id'] = None
        pur['placement'] = None
        pur['amount'] = pur['quantity'] * pur['unit_price']
        events.append(pur[['user_id', 'event_time', 'event_type', 'auction_id', 'product_id', 'vendor_id', 'placement', 'amount']])
        log(f"  Purchases: {len(pur):,}", f)

    if not events:
        log("  ERROR: No events found", f)
        return pd.DataFrame()

    all_events = pd.concat(events, ignore_index=True)
    all_events = all_events.sort_values(['user_id', 'event_time']).reset_index(drop=True)
    log(f"  Total events: {len(all_events):,}", f)
    log(f"  Unique users: {all_events['user_id'].nunique():,}", f)
    log(f"  Date range: {all_events['event_time'].min()} to {all_events['event_time'].max()}", f)

    return all_events


def assign_sessions(events, gap_hours, f):
    """Assign session IDs based on inactivity gaps."""
    log("\n" + "="*80, f)
    log(f"ASSIGNING SESSIONS (gap={gap_hours}h)", f)
    log("="*80, f)

    events = events.sort_values(['user_id', 'event_time']).copy()
    events['time_diff'] = events.groupby('user_id')['event_time'].diff()
    events['time_diff_hours'] = events['time_diff'].dt.total_seconds() / 3600
    events['new_session'] = (events['time_diff_hours'] > gap_hours) | (events['time_diff'].isna())
    events['session_num'] = events.groupby('user_id')['new_session'].cumsum()
    events['session_id'] = events['user_id'] + '_' + events['session_num'].astype(str)

    n_sessions = events['session_id'].nunique()
    n_users = events['user_id'].nunique()
    log(f"  Users: {n_users:,}", f)
    log(f"  Sessions: {n_sessions:,}", f)
    log(f"  Sessions per user: {n_sessions/n_users:.2f}", f)

    # Distribution of sessions per user
    sessions_per_user = events.groupby('user_id')['session_id'].nunique()
    log(f"\n  Sessions per user distribution:", f)
    log(f"    Min: {sessions_per_user.min()}", f)
    log(f"    P25: {sessions_per_user.quantile(0.25):.0f}", f)
    log(f"    P50: {sessions_per_user.median():.0f}", f)
    log(f"    P75: {sessions_per_user.quantile(0.75):.0f}", f)
    log(f"    P95: {sessions_per_user.quantile(0.95):.0f}", f)
    log(f"    Max: {sessions_per_user.max()}", f)

    return events


def detect_device(session_impressions):
    """
    Detect mobile vs desktop from impression batch pattern.
    Mobile shows 2 ads at a time, desktop shows 4+.
    """
    if len(session_impressions) < 2:
        return 'unknown'

    times = pd.to_datetime(session_impressions['event_time']).sort_values()
    diffs = times.diff().dt.total_seconds()

    # Count impressions within 1 second (same batch)
    batch_sizes = []
    current_batch = 1
    for d in diffs.iloc[1:]:
        if pd.notna(d) and d < 1.0:
            current_batch += 1
        else:
            batch_sizes.append(current_batch)
            current_batch = 1
    batch_sizes.append(current_batch)

    if not batch_sizes:
        return 'unknown'

    avg_batch = np.mean(batch_sizes)

    # Mobile shows 2 ads at a time, desktop shows 4+
    if avg_batch <= 2.5:
        return 'mobile'
    else:
        return 'desktop'


def aggregate_sessions(events, data, f):
    """Create session-level aggregates using vectorized operations."""
    log("\n" + "="*80, f)
    log("AGGREGATING SESSIONS", f)
    log("="*80, f)

    # Prepare quality lookup from auctions_results
    ar_cols = ['auction_id', 'product_id']
    has_quality = 'quality' in data['auctions_results'].columns
    has_ranking = 'ranking' in data['auctions_results'].columns
    has_price = 'price' in data['auctions_results'].columns
    for col in ['quality', 'final_bid', 'price', 'ranking']:
        if col in data['auctions_results'].columns:
            ar_cols.append(col)
    ar = data['auctions_results'][ar_cols].copy()
    ar = ar.drop_duplicates(subset=['auction_id', 'product_id'])

    # Merge quality scores into events
    events_with_quality = events.merge(
        ar,
        on=['auction_id', 'product_id'],
        how='left'
    )

    log("  Using vectorized aggregation...", f)

    # Basic aggregations using groupby
    sessions_base = events.groupby('session_id').agg(
        user_id=('user_id', 'first'),
        session_start=('event_time', 'min'),
        session_end=('event_time', 'max'),
    ).reset_index()

    # Duration
    sessions_base['session_duration_hours'] = (
        sessions_base['session_end'] - sessions_base['session_start']
    ).dt.total_seconds() / 3600

    # Event counts by type
    event_counts = events.groupby(['session_id', 'event_type']).size().unstack(fill_value=0).reset_index()
    for col in ['auction', 'impression', 'click', 'purchase']:
        if col not in event_counts.columns:
            event_counts[col] = 0
    event_counts = event_counts.rename(columns={
        'auction': 'n_auctions',
        'impression': 'n_impressions',
        'click': 'n_clicks',
        'purchase': 'n_purchases'
    })
    sessions_base = sessions_base.merge(event_counts[['session_id', 'n_auctions', 'n_impressions', 'n_clicks', 'n_purchases']], on='session_id', how='left')

    # Spend from purchases
    purchases = events[events['event_type'] == 'purchase']
    spend_agg = purchases.groupby('session_id')['amount'].sum().reset_index()
    spend_agg.columns = ['session_id', 'total_spend']
    sessions_base = sessions_base.merge(spend_agg, on='session_id', how='left')
    sessions_base['total_spend'] = sessions_base['total_spend'].fillna(0)
    sessions_base['purchased'] = (sessions_base['n_purchases'] > 0).astype(int)

    # Product/vendor diversity for impressions
    impressions = events[events['event_type'] == 'impression']
    imp_diversity = impressions.groupby('session_id').agg(
        n_products_impressed=('product_id', 'nunique'),
        n_vendors_impressed=('vendor_id', 'nunique')
    ).reset_index()
    sessions_base = sessions_base.merge(imp_diversity, on='session_id', how='left')

    # Product/vendor diversity for clicks
    clicks = events[events['event_type'] == 'click']
    clk_diversity = clicks.groupby('session_id').agg(
        n_products_clicked=('product_id', 'nunique'),
        n_vendors_clicked=('vendor_id', 'nunique')
    ).reset_index()
    sessions_base = sessions_base.merge(clk_diversity, on='session_id', how='left')

    # Fill NAs for diversity metrics
    for col in ['n_products_impressed', 'n_vendors_impressed', 'n_products_clicked', 'n_vendors_clicked']:
        if col in sessions_base.columns:
            sessions_base[col] = sessions_base[col].fillna(0).astype(int)

    # Placement counts from auctions
    auctions = events[events['event_type'] == 'auction']
    placement_agg = auctions.groupby('session_id')['placement'].agg(
        lambda x: len(x.dropna().unique())
    ).reset_index()
    placement_agg.columns = ['session_id', 'n_placements']
    sessions_base = sessions_base.merge(placement_agg, on='session_id', how='left')
    sessions_base['n_placements'] = sessions_base['n_placements'].fillna(0).astype(int)

    # Placement list
    placement_list_agg = auctions.groupby('session_id')['placement'].agg(
        lambda x: ','.join(sorted([str(p) for p in x.dropna().unique()]))
    ).reset_index()
    placement_list_agg.columns = ['session_id', 'placements']
    sessions_base = sessions_base.merge(placement_list_agg, on='session_id', how='left')
    sessions_base['placements'] = sessions_base['placements'].fillna('')

    # Quality/ranking/price for impressions
    if has_quality or has_ranking or has_price:
        imp_q = events_with_quality[events_with_quality['event_type'] == 'impression']
        clk_q = events_with_quality[events_with_quality['event_type'] == 'click']

        agg_cols = {}
        if has_quality:
            agg_cols['avg_quality_impressed'] = ('quality', 'mean')
        if has_ranking:
            agg_cols['avg_ranking_impressed'] = ('ranking', 'mean')
        if has_price:
            agg_cols['avg_price_impressed'] = ('price', 'mean')

        if agg_cols:
            imp_quality_agg = imp_q.groupby('session_id').agg(**agg_cols).reset_index()
            sessions_base = sessions_base.merge(imp_quality_agg, on='session_id', how='left')

            clk_agg_cols = {}
            if has_quality:
                clk_agg_cols['avg_quality_clicked'] = ('quality', 'mean')
            if has_ranking:
                clk_agg_cols['avg_ranking_clicked'] = ('ranking', 'mean')
            if has_price:
                clk_agg_cols['avg_price_clicked'] = ('price', 'mean')

            clk_quality_agg = clk_q.groupby('session_id').agg(**clk_agg_cols).reset_index()
            sessions_base = sessions_base.merge(clk_quality_agg, on='session_id', how='left')
    else:
        sessions_base['avg_quality_impressed'] = np.nan
        sessions_base['avg_quality_clicked'] = np.nan
        sessions_base['avg_ranking_impressed'] = np.nan
        sessions_base['avg_ranking_clicked'] = np.nan
        sessions_base['avg_price_impressed'] = np.nan
        sessions_base['avg_price_clicked'] = np.nan

    # Device detection (simplified - compute for each session with impressions)
    log("  Detecting devices...", f)
    device_results = []
    for session_id in tqdm(sessions_base['session_id'].unique(), desc="Detecting devices"):
        session_impressions = impressions[impressions['session_id'] == session_id]
        device = detect_device(session_impressions)
        device_results.append({'session_id': session_id, 'device': device})
    device_df = pd.DataFrame(device_results)
    sessions_base = sessions_base.merge(device_df, on='session_id', how='left')

    # CTR and purchase rate
    sessions_base['ctr'] = sessions_base['n_clicks'] / sessions_base['n_impressions'].replace(0, np.nan)
    sessions_base['purchase_rate'] = sessions_base['n_purchases'] / sessions_base['n_clicks'].replace(0, np.nan)

    sessions_df = sessions_base
    log(f"  Created {len(sessions_df):,} session records", f)

    return sessions_df


def print_summary(sessions, f):
    """Print session summary statistics."""
    log("\n" + "="*80, f)
    log("SESSION SUMMARY STATISTICS", f)
    log("="*80, f)

    log(f"\nTotal sessions: {len(sessions):,}", f)
    log(f"Total users: {sessions['user_id'].nunique():,}", f)
    log(f"Sessions per user: {len(sessions)/sessions['user_id'].nunique():.2f}", f)

    log(f"\n--- Session Duration (hours) ---", f)
    log(f"  Mean: {sessions['session_duration_hours'].mean():.2f}", f)
    log(f"  Median: {sessions['session_duration_hours'].median():.2f}", f)
    log(f"  Std: {sessions['session_duration_hours'].std():.2f}", f)
    log(f"  Min: {sessions['session_duration_hours'].min():.2f}", f)
    log(f"  P25: {sessions['session_duration_hours'].quantile(0.25):.2f}", f)
    log(f"  P75: {sessions['session_duration_hours'].quantile(0.75):.2f}", f)
    log(f"  P95: {sessions['session_duration_hours'].quantile(0.95):.2f}", f)
    log(f"  Max: {sessions['session_duration_hours'].max():.2f}", f)

    log(f"\n--- Events per Session ---", f)
    for col in ['n_auctions', 'n_impressions', 'n_clicks', 'n_purchases']:
        log(f"  {col}:", f)
        log(f"    Mean: {sessions[col].mean():.2f}, Median: {sessions[col].median():.0f}, Max: {sessions[col].max():.0f}", f)

    log(f"\n--- Purchase Outcomes ---", f)
    log(f"  Sessions with purchase: {sessions['purchased'].sum():,} ({sessions['purchased'].mean()*100:.3f}%)", f)
    log(f"  Total spend (all sessions): {sessions['total_spend'].sum():,.0f} cents", f)
    log(f"  Mean spend (purchasing sessions): {sessions[sessions['purchased']==1]['total_spend'].mean():,.0f} cents", f)

    log(f"\n--- Device Distribution ---", f)
    for device, count in sessions['device'].value_counts().items():
        log(f"  {device}: {count:,} ({count/len(sessions)*100:.1f}%)", f)

    log(f"\n--- Placement Distribution ---", f)
    placement_counts = sessions['placements'].value_counts().head(10)
    for placement, count in placement_counts.items():
        log(f"  '{placement}': {count:,} ({count/len(sessions)*100:.1f}%)", f)

    log(f"\n--- Advertising Exposure ---", f)
    log(f"  Products impressed per session: Mean={sessions['n_products_impressed'].mean():.1f}, Median={sessions['n_products_impressed'].median():.0f}", f)
    log(f"  Products clicked per session: Mean={sessions['n_products_clicked'].mean():.2f}, Median={sessions['n_products_clicked'].median():.0f}", f)
    log(f"  Vendors impressed per session: Mean={sessions['n_vendors_impressed'].mean():.1f}, Median={sessions['n_vendors_impressed'].median():.0f}", f)

    log(f"\n--- Quality Scores ---", f)
    if 'avg_quality_impressed' in sessions.columns:
        log(f"  Avg quality (impressed): Mean={sessions['avg_quality_impressed'].mean():.4f}, Std={sessions['avg_quality_impressed'].std():.4f}", f)
        log(f"  Avg quality (clicked): Mean={sessions['avg_quality_clicked'].mean():.4f}, Std={sessions['avg_quality_clicked'].std():.4f}", f)
    else:
        log(f"  Quality scores not available in data", f)

    log(f"\n--- Click-Through Rate (per session) ---", f)
    valid_ctr = sessions['ctr'].dropna()
    log(f"  Sessions with CTR data: {len(valid_ctr):,}", f)
    log(f"  Mean CTR: {valid_ctr.mean()*100:.3f}%", f)
    log(f"  Median CTR: {valid_ctr.median()*100:.3f}%", f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create shopping sessions from event data')
    parser.add_argument('--gap-hours', type=int, default=SESSION_GAP_HOURS,
                        help=f'Session gap threshold in hours (default: {SESSION_GAP_HOURS})')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / '01_sessionize.txt'

    with open(output_file, 'w') as f:
        log("="*80, f)
        log("01_SESSIONIZE - Shopping Session Construction", f)
        log("="*80, f)
        log(f"Session gap threshold: {args.gap_hours} hours", f)
        log(f"Data directory: {DATA_DIR}", f)
        log(f"Output directory: {OUTPUT_DIR}", f)

        # Load data
        data = load_data(f)

        # Check if data exists
        if all(len(data[k]) == 0 for k in ['auctions_users', 'impressions', 'clicks', 'purchases']):
            log("\nERROR: No data found. Please run the data pull notebook first.", f)
            exit(1)

        # Create event stream
        events = create_unified_event_stream(data, f)

        if len(events) == 0:
            log("\nERROR: No events created.", f)
            exit(1)

        # Assign sessions
        events = assign_sessions(events, args.gap_hours, f)

        # Aggregate
        sessions = aggregate_sessions(events, data, f)

        # Summary
        print_summary(sessions, f)

        # Save
        log("\n" + "="*80, f)
        log("SAVING OUTPUT", f)
        log("="*80, f)
        events.to_parquet(OUTPUT_DIR / 'session_events.parquet', index=False)
        sessions.to_parquet(OUTPUT_DIR / 'sessions.parquet', index=False)
        log(f"  {OUTPUT_DIR / 'session_events.parquet'}: {len(events):,} rows", f)
        log(f"  {OUTPUT_DIR / 'sessions.parquet'}: {len(sessions):,} rows", f)

        log(f"\nOutput saved to: {output_file}", f)
