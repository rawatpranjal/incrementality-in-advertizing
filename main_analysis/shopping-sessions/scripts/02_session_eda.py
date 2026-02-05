#!/usr/bin/env python3
"""
02_session_eda.py - Exploratory Data Analysis for Shopping Sessions

Unit of Analysis: Session-level
Purpose: Comprehensive EDA of session data before modeling.
All tables printed to stdout, no opinions, raw dump of facts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# PATHS
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "0_data_pull" / "data"
RESULTS_DIR = BASE_DIR / "results"


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def section(title, fh):
    log("\n" + "="*80, fh)
    log(title, fh)
    log("="*80, fh)


def subsection(title, fh):
    log(f"\n--- {title} ---", fh)


def load_data(f):
    """Load sessions and events data."""
    section("LOADING DATA", f)
    sessions = pd.read_parquet(DATA_DIR / 'sessions.parquet')
    events = pd.read_parquet(DATA_DIR / 'session_events.parquet')
    log(f"Sessions: {len(sessions):,} rows, {sessions.shape[1]} cols", f)
    log(f"Events: {len(events):,} rows, {events.shape[1]} cols", f)
    return sessions, events


def sample_overview(sessions, events, f):
    """Basic sample statistics."""
    section("SAMPLE OVERVIEW", f)

    log(f"\nSession count: {len(sessions):,}", f)
    log(f"User count: {sessions['user_id'].nunique():,}", f)
    log(f"Event count: {len(events):,}", f)

    log(f"\nDate range:", f)
    log(f"  Start: {sessions['session_start'].min()}", f)
    log(f"  End: {sessions['session_end'].max()}", f)

    log(f"\nSessions per user:", f)
    spu = sessions.groupby('user_id').size()
    log(f"  Mean: {spu.mean():.2f}", f)
    log(f"  Median: {spu.median():.0f}", f)
    log(f"  Std: {spu.std():.2f}", f)
    log(f"  Min: {spu.min()}", f)
    log(f"  Max: {spu.max()}", f)

    log(f"\nSession columns:", f)
    for col in sessions.columns:
        dtype = sessions[col].dtype
        n_null = sessions[col].isna().sum()
        n_unique = sessions[col].nunique()
        log(f"  {col}: dtype={dtype}, null={n_null}, unique={n_unique}", f)


def session_duration_analysis(sessions, f):
    """Analyze session duration distribution."""
    section("SESSION DURATION ANALYSIS", f)

    dur = sessions['session_duration_hours']

    subsection("Duration (hours) - Full Distribution", f)
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        val = dur.quantile(p/100)
        log(f"  P{p:3d}: {val:10.2f} hours", f)

    log(f"\n  Mean: {dur.mean():.2f}", f)
    log(f"  Std: {dur.std():.2f}", f)
    log(f"  Skewness: {dur.skew():.2f}", f)
    log(f"  Kurtosis: {dur.kurtosis():.2f}", f)

    subsection("Duration Buckets", f)
    bins = [0, 0.083, 0.5, 1, 6, 24, 72, float('inf')]  # 5min, 30min, 1h, 6h, 24h, 72h
    labels = ['<5min', '5-30min', '30min-1h', '1-6h', '6-24h', '24-72h', '>72h']
    sessions['dur_bucket'] = pd.cut(dur, bins=bins, labels=labels, include_lowest=True)
    for bucket in labels:
        count = (sessions['dur_bucket'] == bucket).sum()
        pct = count / len(sessions) * 100
        log(f"  {bucket:12s}: {count:8,} ({pct:5.1f}%)", f)

    subsection("Zero-Duration Sessions", f)
    zero_dur = (dur == 0).sum()
    log(f"  Count: {zero_dur:,} ({zero_dur/len(sessions)*100:.1f}%)", f)


def event_counts_analysis(sessions, f):
    """Analyze event counts per session."""
    section("EVENT COUNTS PER SESSION", f)

    event_cols = ['n_auctions', 'n_impressions', 'n_clicks', 'n_purchases']

    for col in event_cols:
        subsection(col, f)
        vals = sessions[col]
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        for p in percentiles:
            val = vals.quantile(p/100)
            log(f"  P{p:3d}: {val:10.0f}", f)
        log(f"  Mean: {vals.mean():.2f}", f)
        log(f"  Std: {vals.std():.2f}", f)
        zeros = (vals == 0).sum()
        log(f"  Zero count: {zeros:,} ({zeros/len(sessions)*100:.1f}%)", f)

    subsection("Event Ratios", f)
    imp_per_auction = sessions['n_impressions'] / sessions['n_auctions'].replace(0, np.nan)
    clk_per_imp = sessions['n_clicks'] / sessions['n_impressions'].replace(0, np.nan)
    pur_per_clk = sessions['n_purchases'] / sessions['n_clicks'].replace(0, np.nan)

    log(f"\n  Impressions per auction:", f)
    log(f"    Mean: {imp_per_auction.mean():.2f}, Median: {imp_per_auction.median():.2f}", f)
    log(f"\n  Clicks per impression (CTR):", f)
    log(f"    Mean: {clk_per_imp.mean()*100:.3f}%, Median: {clk_per_imp.median()*100:.3f}%", f)
    log(f"\n  Purchases per click:", f)
    log(f"    Mean: {pur_per_clk.mean()*100:.3f}%, Median: {pur_per_clk.median()*100:.3f}%", f)


def conversion_analysis(sessions, f):
    """Analyze conversion funnel."""
    section("CONVERSION FUNNEL ANALYSIS", f)

    total = len(sessions)

    # Funnel stages
    with_auctions = (sessions['n_auctions'] > 0).sum()
    with_impressions = (sessions['n_impressions'] > 0).sum()
    with_clicks = (sessions['n_clicks'] > 0).sum()
    with_purchases = (sessions['n_purchases'] > 0).sum()

    subsection("Session-Level Funnel", f)
    log(f"  Total sessions:      {total:8,} (100.0%)", f)
    log(f"  With auctions:       {with_auctions:8,} ({with_auctions/total*100:.1f}%)", f)
    log(f"  With impressions:    {with_impressions:8,} ({with_impressions/total*100:.1f}%)", f)
    log(f"  With clicks:         {with_clicks:8,} ({with_clicks/total*100:.1f}%)", f)
    log(f"  With purchases:      {with_purchases:8,} ({with_purchases/total*100:.1f}%)", f)

    subsection("Conversion Rates", f)
    log(f"  Impression → Click rate: {with_clicks/with_impressions*100:.3f}% (of sessions with impressions)", f)
    log(f"  Click → Purchase rate:   {with_purchases/with_clicks*100:.3f}% (of sessions with clicks)", f)
    log(f"  Session → Purchase rate: {with_purchases/total*100:.3f}% (overall)", f)

    subsection("Purchase Distribution", f)
    purchasing = sessions[sessions['purchased'] == 1]
    log(f"  Purchasing sessions: {len(purchasing):,}", f)
    log(f"  Non-purchasing sessions: {len(sessions) - len(purchasing):,}", f)
    log(f"  Purchases per purchasing session:", f)
    log(f"    Mean: {purchasing['n_purchases'].mean():.2f}", f)
    log(f"    Median: {purchasing['n_purchases'].median():.0f}", f)
    log(f"    Max: {purchasing['n_purchases'].max():.0f}", f)


def spend_analysis(sessions, f):
    """Analyze spending patterns."""
    section("SPEND ANALYSIS", f)

    subsection("All Sessions", f)
    spend = sessions['total_spend']
    log(f"  Total spend: {spend.sum():,.0f} cents (${spend.sum()/100:,.2f})", f)
    log(f"  Mean spend: {spend.mean():.2f} cents", f)
    log(f"  Sessions with spend > 0: {(spend > 0).sum():,}", f)

    subsection("Purchasing Sessions Only", f)
    purch_sessions = sessions[sessions['purchased'] == 1]
    purch_spend = purch_sessions['total_spend']
    if len(purch_sessions) > 0:
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        for p in percentiles:
            val = purch_spend.quantile(p/100)
            log(f"  P{p:3d}: {val:10,.0f} cents (${val/100:,.2f})", f)
        log(f"\n  Mean: {purch_spend.mean():,.0f} cents", f)
        log(f"  Std: {purch_spend.std():,.0f} cents", f)


def advertising_exposure_analysis(sessions, f):
    """Analyze advertising exposure metrics."""
    section("ADVERTISING EXPOSURE ANALYSIS", f)

    subsection("Products Impressed", f)
    col = sessions['n_products_impressed']
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        val = col.quantile(p/100)
        log(f"  P{p:3d}: {val:10.0f}", f)

    subsection("Products Clicked", f)
    col = sessions['n_products_clicked']
    for p in percentiles:
        val = col.quantile(p/100)
        log(f"  P{p:3d}: {val:10.0f}", f)

    subsection("Vendors Impressed", f)
    col = sessions['n_vendors_impressed']
    for p in percentiles:
        val = col.quantile(p/100)
        log(f"  P{p:3d}: {val:10.0f}", f)

    subsection("Vendors Clicked", f)
    col = sessions['n_vendors_clicked']
    for p in percentiles:
        val = col.quantile(p/100)
        log(f"  P{p:3d}: {val:10.0f}", f)


def quality_score_analysis(sessions, f):
    """Analyze quality scores."""
    section("QUALITY SCORE ANALYSIS", f)

    if 'avg_quality_impressed' not in sessions.columns:
        log("  Quality scores not available in data", f)
        return

    subsection("Quality of Impressed Products", f)
    col = sessions['avg_quality_impressed'].dropna()
    log(f"  N (non-null): {len(col):,}", f)
    if len(col) > 0:
        log(f"  Mean: {col.mean():.6f}", f)
        log(f"  Std: {col.std():.6f}", f)
        log(f"  Min: {col.min():.6f}", f)
        log(f"  Max: {col.max():.6f}", f)
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        for p in percentiles:
            val = col.quantile(p/100)
            log(f"  P{p:3d}: {val:.6f}", f)

    subsection("Quality of Clicked Products", f)
    if 'avg_quality_clicked' in sessions.columns:
        col = sessions['avg_quality_clicked'].dropna()
        log(f"  N (non-null): {len(col):,}", f)
        if len(col) > 0:
            log(f"  Mean: {col.mean():.6f}", f)
            log(f"  Std: {col.std():.6f}", f)
            log(f"  Min: {col.min():.6f}", f)
            log(f"  Max: {col.max():.6f}", f)
            for p in percentiles:
                val = col.quantile(p/100)
                log(f"  P{p:3d}: {val:.6f}", f)

    subsection("Quality Comparison: Clicked vs Impressed", f)
    if 'avg_quality_impressed' in sessions.columns and 'avg_quality_clicked' in sessions.columns:
        both = sessions[sessions['avg_quality_impressed'].notna() & sessions['avg_quality_clicked'].notna()]
        if len(both) > 0:
            diff = both['avg_quality_clicked'] - both['avg_quality_impressed']
            log(f"  Sessions with both: {len(both):,}", f)
            log(f"  Mean difference (clicked - impressed): {diff.mean():.6f}", f)
            log(f"  Std of difference: {diff.std():.6f}", f)
            log(f"  % where clicked > impressed: {(diff > 0).mean()*100:.1f}%", f)


def ranking_analysis(sessions, f):
    """Analyze ranking positions."""
    section("RANKING ANALYSIS", f)

    subsection("Ranking of Impressed Products", f)
    col = sessions['avg_ranking_impressed'].dropna()
    log(f"  N (non-null): {len(col):,}", f)
    if len(col) > 0:
        log(f"  Mean: {col.mean():.2f}", f)
        log(f"  Std: {col.std():.2f}", f)
        log(f"  Min: {col.min():.0f}", f)
        log(f"  Max: {col.max():.0f}", f)
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            val = col.quantile(p/100)
            log(f"  P{p:3d}: {val:.1f}", f)

    subsection("Ranking of Clicked Products", f)
    col = sessions['avg_ranking_clicked'].dropna()
    log(f"  N (non-null): {len(col):,}", f)
    if len(col) > 0:
        log(f"  Mean: {col.mean():.2f}", f)
        log(f"  Std: {col.std():.2f}", f)
        log(f"  Min: {col.min():.0f}", f)
        log(f"  Max: {col.max():.0f}", f)
        for p in percentiles:
            val = col.quantile(p/100)
            log(f"  P{p:3d}: {val:.1f}", f)


def device_analysis(sessions, f):
    """Analyze device distribution."""
    section("DEVICE ANALYSIS", f)

    subsection("Device Distribution", f)
    for device, count in sessions['device'].value_counts().items():
        pct = count / len(sessions) * 100
        log(f"  {device:12s}: {count:8,} ({pct:5.1f}%)", f)

    subsection("Metrics by Device", f)
    device_stats = sessions.groupby('device').agg({
        'n_impressions': 'mean',
        'n_clicks': 'mean',
        'purchased': 'mean',
        'total_spend': 'mean',
        'session_duration_hours': 'mean',
    }).round(4)

    log(f"\n  {'Device':<12} {'Imp':<10} {'Clicks':<10} {'Purch%':<10} {'Spend':<12} {'Duration':<10}", f)
    log(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}", f)
    for device in device_stats.index:
        row = device_stats.loc[device]
        log(f"  {device:<12} {row['n_impressions']:<10.2f} {row['n_clicks']:<10.2f} {row['purchased']*100:<10.2f} {row['total_spend']:<12.2f} {row['session_duration_hours']:<10.2f}", f)


def placement_analysis(sessions, f):
    """Analyze placement patterns."""
    section("PLACEMENT ANALYSIS", f)

    subsection("Number of Placements per Session", f)
    col = sessions['n_placements']
    for val in sorted(col.unique())[:10]:
        count = (col == val).sum()
        pct = count / len(sessions) * 100
        log(f"  {val} placements: {count:,} ({pct:.1f}%)", f)

    subsection("Top Placement Combinations", f)
    placement_counts = sessions['placements'].value_counts().head(15)
    for placement, count in placement_counts.items():
        pct = count / len(sessions) * 100
        log(f"  '{placement}': {count:,} ({pct:.1f}%)", f)


def correlation_analysis(sessions, f):
    """Compute correlation matrix."""
    section("CORRELATION ANALYSIS", f)

    numeric_cols = [
        'session_duration_hours', 'n_auctions', 'n_impressions', 'n_clicks',
        'n_purchases', 'total_spend', 'n_products_impressed', 'n_products_clicked',
        'avg_quality_impressed', 'avg_quality_clicked', 'purchased'
    ]

    existing_cols = [c for c in numeric_cols if c in sessions.columns]
    corr = sessions[existing_cols].corr()

    subsection("Correlation Matrix", f)
    log(f"\n{'':25s}", f)
    header = "".join([f"{c[:8]:>10s}" for c in existing_cols])
    log(f"{'':25s}{header}", f)

    for row in existing_cols:
        row_vals = "".join([f"{corr.loc[row, col]:10.3f}" for col in existing_cols])
        log(f"{row[:25]:25s}{row_vals}", f)

    subsection("Key Correlations with Purchase", f)
    if 'purchased' in existing_cols:
        purch_corr = corr['purchased'].drop('purchased').sort_values(ascending=False)
        for var, val in purch_corr.items():
            log(f"  {var:30s}: {val:+.4f}", f)


def purchasing_vs_nonpurchasing(sessions, f):
    """Compare purchasing vs non-purchasing sessions."""
    section("PURCHASING VS NON-PURCHASING SESSIONS", f)

    purch = sessions[sessions['purchased'] == 1]
    non_purch = sessions[sessions['purchased'] == 0]

    log(f"\nPurchasing sessions: {len(purch):,}", f)
    log(f"Non-purchasing sessions: {len(non_purch):,}", f)

    compare_cols = [
        'session_duration_hours', 'n_auctions', 'n_impressions', 'n_clicks',
        'n_products_impressed', 'n_products_clicked', 'avg_quality_impressed',
        'avg_quality_clicked', 'avg_ranking_impressed', 'avg_ranking_clicked'
    ]

    log(f"\n{'Variable':30s} {'Purch Mean':>12s} {'Non-P Mean':>12s} {'Diff':>10s} {'T-stat':>10s} {'P-value':>10s}", f)
    log(f"{'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)

    for col in compare_cols:
        if col not in sessions.columns:
            continue
        p_vals = purch[col].dropna()
        np_vals = non_purch[col].dropna()
        if len(p_vals) < 2 or len(np_vals) < 2:
            continue
        p_mean = p_vals.mean()
        np_mean = np_vals.mean()
        diff = p_mean - np_mean

        # T-test
        t_stat, p_val = stats.ttest_ind(p_vals, np_vals, equal_var=False)

        log(f"{col:30s} {p_mean:12.4f} {np_mean:12.4f} {diff:+10.4f} {t_stat:10.2f} {p_val:10.4f}", f)


def event_sequence_analysis(events, f):
    """Analyze event sequences within sessions."""
    section("EVENT SEQUENCE ANALYSIS", f)

    subsection("Event Type Distribution", f)
    event_counts = events['event_type'].value_counts()
    for etype, count in event_counts.items():
        pct = count / len(events) * 100
        log(f"  {etype:15s}: {count:10,} ({pct:5.1f}%)", f)

    subsection("Events per Session", f)
    events_per_session = events.groupby('session_id').size()
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        val = events_per_session.quantile(p/100)
        log(f"  P{p:3d}: {val:10.0f}", f)

    subsection("Time Between Events (within session)", f)
    events_sorted = events.sort_values(['session_id', 'event_time'])
    events_sorted['time_to_next'] = events_sorted.groupby('session_id')['event_time'].diff().shift(-1)
    time_diffs = events_sorted['time_to_next'].dropna().dt.total_seconds()
    log(f"  Mean: {time_diffs.mean():.1f} seconds", f)
    log(f"  Median: {time_diffs.median():.1f} seconds", f)
    log(f"  P95: {time_diffs.quantile(0.95):.1f} seconds", f)


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / '02_session_eda.txt'

    with open(output_file, 'w') as f:
        log("="*80, f)
        log("02_SESSION_EDA - Shopping Sessions Exploratory Data Analysis", f)
        log("="*80, f)
        log(f"Data directory: {DATA_DIR}", f)

        # Load data
        sessions, events = load_data(f)

        # Run all analyses
        sample_overview(sessions, events, f)
        session_duration_analysis(sessions, f)
        event_counts_analysis(sessions, f)
        conversion_analysis(sessions, f)
        spend_analysis(sessions, f)
        advertising_exposure_analysis(sessions, f)
        quality_score_analysis(sessions, f)
        ranking_analysis(sessions, f)
        device_analysis(sessions, f)
        placement_analysis(sessions, f)
        correlation_analysis(sessions, f)
        purchasing_vs_nonpurchasing(sessions, f)
        event_sequence_analysis(events, f)

        log(f"\n" + "="*80, f)
        log(f"Output saved to: {output_file}", f)
