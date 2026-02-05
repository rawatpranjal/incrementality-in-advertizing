#!/usr/bin/env python3
"""
07_session_features.py - Extended Session Features from Literature

New features based on:
1. "Between Click and Purchase" - repeated views, category breadth, exploration depth
2. "Flash Sales AEIO" - session position, relative timing, event distribution

Unit of Analysis: Session
Output:
- data/sessions_extended.parquet: Sessions with new features
- results/07_session_features.txt: Feature statistics

Features Added:
- n_repeated_product_views: Products viewed more than once
- pct_repeated_views: Fraction of repeat impressions
- n_unique_categories: Category breadth (from catalog)
- products_per_auction: Exploration depth
- session_position: Normalized position in user's session history
- relative_session_time: Session start as fraction of user's timeline
- early_session_indicator: First 25% of sessions
- late_session_indicator: Last 25% of sessions
- events_in_first_half: Count of events in first half of session
- time_to_first_click: Hours from session start to first click
- time_to_first_purchase: Hours from session start to first purchase
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


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def load_data(f):
    """Load required data files."""
    log("\nLOADING DATA", f)

    sessions = pd.read_parquet(DATA_DIR / 'sessions.parquet')
    log(f"  sessions: {len(sessions):,} rows", f)

    session_events = pd.read_parquet(DATA_DIR / 'session_events.parquet')
    log(f"  session_events: {len(session_events):,} rows", f)

    catalog = None
    catalog_path = DATA_DIR / 'catalog.parquet'
    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
        log(f"  catalog: {len(catalog):,} rows", f)
    else:
        log(f"  catalog: NOT FOUND", f)

    return sessions, session_events, catalog


def compute_repeated_views(session_events, f):
    """Compute repeated product view features per session."""
    log("\n" + "="*80, f)
    log("COMPUTING REPEATED VIEW FEATURES", f)
    log("="*80, f)

    impressions = session_events[session_events['event_type'] == 'impression'].copy()
    log(f"  Total impressions: {len(impressions):,}", f)

    # Count impressions per product per session
    product_counts = impressions.groupby(['session_id', 'product_id']).size().reset_index(name='view_count')

    # Aggregate to session level
    session_stats = product_counts.groupby('session_id').agg(
        n_unique_products=('product_id', 'nunique'),
        n_total_impressions=('view_count', 'sum'),
        n_repeated_products=('view_count', lambda x: (x > 1).sum()),
        n_repeated_views=('view_count', lambda x: (x - 1).clip(lower=0).sum())
    ).reset_index()

    session_stats['pct_repeated_views'] = session_stats['n_repeated_views'] / session_stats['n_total_impressions']
    session_stats['pct_repeated_products'] = session_stats['n_repeated_products'] / session_stats['n_unique_products']

    log(f"  Sessions with repeated views: {(session_stats['n_repeated_products'] > 0).sum():,}", f)
    log(f"  Mean repeated products per session: {session_stats['n_repeated_products'].mean():.2f}", f)
    log(f"  Mean pct repeated views: {session_stats['pct_repeated_views'].mean()*100:.2f}%", f)

    return session_stats[['session_id', 'n_repeated_products', 'n_repeated_views', 'pct_repeated_views', 'pct_repeated_products']]


def compute_category_features(session_events, catalog, f):
    """Compute category breadth features per session."""
    log("\n" + "="*80, f)
    log("COMPUTING CATEGORY FEATURES", f)
    log("="*80, f)

    if catalog is None or 'categories' not in catalog.columns:
        log("  Skipping: catalog not available or missing categories column", f)
        return None

    impressions = session_events[session_events['event_type'] == 'impression'].copy()

    # Parse categories from catalog (stored as array/list)
    catalog_clean = catalog[['product_id', 'categories']].copy()

    # Explode categories if they are lists
    if catalog_clean['categories'].dtype == 'object':
        try:
            # Try to parse as lists
            catalog_clean['categories'] = catalog_clean['categories'].apply(
                lambda x: x if isinstance(x, list) else (eval(x) if pd.notna(x) and x else [])
            )
        except Exception as e:
            log(f"  Warning: Could not parse categories: {e}", f)
            return None

    # Explode to one row per category
    catalog_exploded = catalog_clean.explode('categories')
    catalog_exploded = catalog_exploded[catalog_exploded['categories'].notna()]
    log(f"  Catalog with categories: {len(catalog_exploded):,} product-category pairs", f)

    # Join to impressions
    imp_with_cat = impressions.merge(catalog_exploded, on='product_id', how='left')

    # Count unique categories per session
    cat_counts = imp_with_cat.groupby('session_id').agg(
        n_unique_categories=('categories', 'nunique'),
        n_categorized_impressions=('categories', lambda x: x.notna().sum())
    ).reset_index()

    log(f"  Sessions with category data: {len(cat_counts):,}", f)
    log(f"  Mean unique categories per session: {cat_counts['n_unique_categories'].mean():.2f}", f)

    return cat_counts


def compute_exploration_depth(session_events, f):
    """Compute exploration depth features per session."""
    log("\n" + "="*80, f)
    log("COMPUTING EXPLORATION DEPTH FEATURES", f)
    log("="*80, f)

    impressions = session_events[session_events['event_type'] == 'impression'].copy()
    auctions = session_events[session_events['event_type'] == 'auction'].copy()

    # Products per auction
    imp_per_auction = impressions.groupby(['session_id', 'auction_id']).agg(
        products_in_auction=('product_id', 'nunique')
    ).reset_index()

    # Aggregate to session level
    exploration = imp_per_auction.groupby('session_id').agg(
        mean_products_per_auction=('products_in_auction', 'mean'),
        max_products_per_auction=('products_in_auction', 'max'),
        n_auctions_with_impressions=('auction_id', 'nunique')
    ).reset_index()

    log(f"  Sessions with auction data: {len(exploration):,}", f)
    log(f"  Mean products per auction: {exploration['mean_products_per_auction'].mean():.2f}", f)

    return exploration


def compute_session_position(sessions, f):
    """Compute session position features within user history."""
    log("\n" + "="*80, f)
    log("COMPUTING SESSION POSITION FEATURES", f)
    log("="*80, f)

    sessions = sessions.copy()
    sessions['session_start'] = pd.to_datetime(sessions['session_start'])

    # Sort by user and time
    sessions = sessions.sort_values(['user_id', 'session_start'])

    # Compute session number within user
    sessions['session_num_within_user'] = sessions.groupby('user_id').cumcount() + 1
    sessions['total_user_sessions'] = sessions.groupby('user_id')['session_id'].transform('count')

    # Normalized position (0 = first session, 1 = last session)
    sessions['session_position'] = (sessions['session_num_within_user'] - 1) / (sessions['total_user_sessions'] - 1).clip(lower=1)

    # User timeline position
    user_timeline = sessions.groupby('user_id').agg(
        user_first_session=('session_start', 'min'),
        user_last_session=('session_start', 'max')
    ).reset_index()

    sessions = sessions.merge(user_timeline, on='user_id', how='left')
    user_span = (sessions['user_last_session'] - sessions['user_first_session']).dt.total_seconds()
    session_offset = (sessions['session_start'] - sessions['user_first_session']).dt.total_seconds()
    sessions['relative_session_time'] = session_offset / user_span.clip(lower=1)

    # Early/late indicators (first/last 25% of sessions)
    sessions['early_session_indicator'] = (sessions['session_position'] <= 0.25).astype(int)
    sessions['late_session_indicator'] = (sessions['session_position'] >= 0.75).astype(int)

    log(f"  Sessions with position data: {len(sessions):,}", f)
    log(f"  Mean session position: {sessions['session_position'].mean():.3f}", f)
    log(f"  Early sessions (first 25%): {sessions['early_session_indicator'].sum():,}", f)
    log(f"  Late sessions (last 25%): {sessions['late_session_indicator'].sum():,}", f)

    return sessions[['session_id', 'session_num_within_user', 'total_user_sessions',
                     'session_position', 'relative_session_time',
                     'early_session_indicator', 'late_session_indicator']]


def compute_event_timing(session_events, sessions, f):
    """Compute event timing features within sessions."""
    log("\n" + "="*80, f)
    log("COMPUTING EVENT TIMING FEATURES", f)
    log("="*80, f)

    session_events = session_events.copy()
    session_events['event_time'] = pd.to_datetime(session_events['event_time'])

    # Get session start/end times
    session_times = sessions[['session_id', 'session_start', 'session_end']].copy()
    session_times['session_start'] = pd.to_datetime(session_times['session_start'])
    session_times['session_end'] = pd.to_datetime(session_times['session_end'])
    session_times['session_duration_secs'] = (session_times['session_end'] - session_times['session_start']).dt.total_seconds()

    # Merge with events
    events = session_events.merge(session_times, on='session_id', how='left')

    # Compute relative time within session
    events['time_from_start'] = (events['event_time'] - events['session_start']).dt.total_seconds()
    events['relative_time'] = events['time_from_start'] / events['session_duration_secs'].clip(lower=1)

    # Events in first half
    events['in_first_half'] = (events['relative_time'] <= 0.5).astype(int)
    first_half_counts = events.groupby('session_id')['in_first_half'].sum().reset_index()
    first_half_counts.columns = ['session_id', 'events_in_first_half']

    # Time to first click
    clicks = events[events['event_type'] == 'click']
    first_click = clicks.groupby('session_id')['time_from_start'].min().reset_index()
    first_click.columns = ['session_id', 'time_to_first_click_secs']
    first_click['time_to_first_click_hours'] = first_click['time_to_first_click_secs'] / 3600

    # Time to first purchase
    purchases = events[events['event_type'] == 'purchase']
    first_purchase = purchases.groupby('session_id')['time_from_start'].min().reset_index()
    first_purchase.columns = ['session_id', 'time_to_first_purchase_secs']
    first_purchase['time_to_first_purchase_hours'] = first_purchase['time_to_first_purchase_secs'] / 3600

    # Combine
    timing = first_half_counts.merge(first_click[['session_id', 'time_to_first_click_hours']], on='session_id', how='left')
    timing = timing.merge(first_purchase[['session_id', 'time_to_first_purchase_hours']], on='session_id', how='left')

    log(f"  Sessions with timing data: {len(timing):,}", f)
    log(f"  Sessions with clicks: {timing['time_to_first_click_hours'].notna().sum():,}", f)
    log(f"  Sessions with purchases: {timing['time_to_first_purchase_hours'].notna().sum():,}", f)
    log(f"  Mean time to first click: {timing['time_to_first_click_hours'].mean():.2f} hours", f)

    return timing


def print_feature_summary(sessions_extended, f):
    """Print summary statistics for all new features."""
    log("\n" + "="*80, f)
    log("FEATURE SUMMARY STATISTICS", f)
    log("="*80, f)

    new_features = [
        'n_repeated_products', 'n_repeated_views', 'pct_repeated_views', 'pct_repeated_products',
        'n_unique_categories', 'mean_products_per_auction', 'max_products_per_auction',
        'session_position', 'relative_session_time', 'early_session_indicator', 'late_session_indicator',
        'events_in_first_half', 'time_to_first_click_hours', 'time_to_first_purchase_hours'
    ]

    for feat in new_features:
        if feat in sessions_extended.columns:
            col = sessions_extended[feat]
            valid = col.notna()
            log(f"\n{feat}:", f)
            log(f"  Valid: {valid.sum():,} ({valid.mean()*100:.1f}%)", f)
            if valid.sum() > 0:
                log(f"  Mean: {col.mean():.4f}", f)
                log(f"  Std: {col.std():.4f}", f)
                log(f"  Min: {col.min():.4f}", f)
                log(f"  P25: {col.quantile(0.25):.4f}", f)
                log(f"  P50: {col.median():.4f}", f)
                log(f"  P75: {col.quantile(0.75):.4f}", f)
                log(f"  P95: {col.quantile(0.95):.4f}", f)
                log(f"  Max: {col.max():.4f}", f)


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / '07_session_features.txt'

    with open(output_file, 'w') as f:
        log("="*80, f)
        log("07_SESSION_FEATURES - Extended Session Features from Literature", f)
        log("="*80, f)
        log(f"Data directory: {DATA_DIR}", f)
        log(f"Output directory: {OUTPUT_DIR}", f)

        # Load data
        sessions, session_events, catalog = load_data(f)

        # Start with original sessions
        sessions_extended = sessions.copy()

        # Compute repeated view features
        repeated_views = compute_repeated_views(session_events, f)
        sessions_extended = sessions_extended.merge(repeated_views, on='session_id', how='left')

        # Compute category features
        cat_features = compute_category_features(session_events, catalog, f)
        if cat_features is not None:
            sessions_extended = sessions_extended.merge(cat_features, on='session_id', how='left')

        # Compute exploration depth
        exploration = compute_exploration_depth(session_events, f)
        sessions_extended = sessions_extended.merge(exploration, on='session_id', how='left')

        # Compute session position
        position = compute_session_position(sessions, f)
        sessions_extended = sessions_extended.merge(position, on='session_id', how='left')

        # Compute event timing
        timing = compute_event_timing(session_events, sessions, f)
        sessions_extended = sessions_extended.merge(timing, on='session_id', how='left')

        # Summary
        print_feature_summary(sessions_extended, f)

        # Save
        log("\n" + "="*80, f)
        log("SAVING OUTPUT", f)
        log("="*80, f)

        output_path = OUTPUT_DIR / 'sessions_extended.parquet'
        sessions_extended.to_parquet(output_path, index=False)
        log(f"  {output_path}: {len(sessions_extended):,} rows, {sessions_extended.shape[1]} columns", f)

        log(f"\nNew columns added:", f)
        new_cols = [c for c in sessions_extended.columns if c not in sessions.columns]
        for col in new_cols:
            log(f"  - {col}", f)

        log(f"\nOutput saved to: {output_file}", f)
