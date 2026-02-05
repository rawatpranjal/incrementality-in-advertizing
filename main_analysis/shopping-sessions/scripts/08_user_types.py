#!/usr/bin/env python3
"""
08_user_types.py - User Type Classification

Classifies users into behavioral types based on session characteristics.
Based on Moe (2003) shopping modes and platform-specific patterns.

User Type Definitions:
1. Focused: Short duration, few products, high CTR, quick purchases
2. Meandering: Long duration, many products, lots of browsing
3. Exploratory: High vendor/category diversity
4. Default/Mixed: Not fitting above patterns

Classification Method:
- Compute user-level averages across all sessions
- Apply percentile-based thresholds
- Assign type based on dominant characteristic

Unit of Analysis: User
Output:
- data/user_types.parquet: User-level type classification
- data/sessions_with_user_type.parquet: Sessions merged with user type
- results/08_user_types.txt: Classification statistics
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
    """Load sessions data."""
    log("\nLOADING DATA", f)

    # Try extended sessions first, fall back to regular
    extended_path = DATA_DIR / 'sessions_extended.parquet'
    regular_path = DATA_DIR / 'sessions.parquet'

    if extended_path.exists():
        sessions = pd.read_parquet(extended_path)
        log(f"  sessions_extended: {len(sessions):,} rows, {sessions.shape[1]} columns", f)
    else:
        sessions = pd.read_parquet(regular_path)
        log(f"  sessions: {len(sessions):,} rows, {sessions.shape[1]} columns", f)

    return sessions


def compute_user_level_stats(sessions, f):
    """Compute user-level aggregates for classification."""
    log("\n" + "="*80, f)
    log("COMPUTING USER-LEVEL STATISTICS", f)
    log("="*80, f)

    # Define metrics to aggregate
    agg_dict = {
        'session_id': 'count',
        'session_duration_hours': ['mean', 'median', 'std'],
        'n_impressions': ['mean', 'median', 'sum'],
        'n_clicks': ['mean', 'median', 'sum'],
        'n_purchases': ['mean', 'sum'],
        'n_products_impressed': ['mean', 'median'],
        'purchased': ['mean', 'sum'],
        'total_spend': ['mean', 'sum'],
    }

    # Add optional columns if they exist
    if 'n_vendors_impressed' in sessions.columns:
        agg_dict['n_vendors_impressed'] = ['mean', 'median']
    if 'ctr' in sessions.columns:
        agg_dict['ctr'] = ['mean', 'median']
    if 'n_unique_categories' in sessions.columns:
        agg_dict['n_unique_categories'] = ['mean', 'median']

    user_stats = sessions.groupby('user_id').agg(agg_dict)

    # Flatten column names
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
    user_stats = user_stats.reset_index()

    # Rename for clarity
    user_stats = user_stats.rename(columns={
        'session_id_count': 'n_sessions',
        'purchased_mean': 'purchase_rate',
        'purchased_sum': 'total_purchases'
    })

    log(f"  Computed stats for {len(user_stats):,} users", f)

    # Print distributions
    log("\n  User-level distributions:", f)
    for col in ['n_sessions', 'session_duration_hours_mean', 'n_impressions_mean',
                'n_clicks_mean', 'purchase_rate', 'n_products_impressed_mean']:
        if col in user_stats.columns:
            log(f"    {col}: mean={user_stats[col].mean():.2f}, median={user_stats[col].median():.2f}", f)

    return user_stats


def classify_users(user_stats, f):
    """Classify users into behavioral types."""
    log("\n" + "="*80, f)
    log("CLASSIFYING USERS", f)
    log("="*80, f)

    user_stats = user_stats.copy()

    # Compute percentiles for thresholds
    duration_p25 = user_stats['session_duration_hours_mean'].quantile(0.25)
    duration_p75 = user_stats['session_duration_hours_mean'].quantile(0.75)

    products_p25 = user_stats['n_products_impressed_mean'].quantile(0.25)
    products_p75 = user_stats['n_products_impressed_mean'].quantile(0.75)

    clicks_p75 = user_stats['n_clicks_mean'].quantile(0.75)

    purchase_rate_p75 = user_stats['purchase_rate'].quantile(0.75)

    # CTR if available
    ctr_col = 'ctr_mean' if 'ctr_mean' in user_stats.columns else None
    if ctr_col:
        ctr_p75 = user_stats[ctr_col].quantile(0.75)

    # Vendor diversity if available
    vendor_col = 'n_vendors_impressed_mean' if 'n_vendors_impressed_mean' in user_stats.columns else None
    if vendor_col:
        vendor_p75 = user_stats[vendor_col].quantile(0.75)

    log(f"\n  Classification thresholds:", f)
    log(f"    Duration P25: {duration_p25:.2f} hours", f)
    log(f"    Duration P75: {duration_p75:.2f} hours", f)
    log(f"    Products P25: {products_p25:.1f}", f)
    log(f"    Products P75: {products_p75:.1f}", f)
    log(f"    Clicks P75: {clicks_p75:.1f}", f)
    log(f"    Purchase rate P75: {purchase_rate_p75:.3f}", f)
    if ctr_col:
        log(f"    CTR P75: {ctr_p75:.4f}", f)
    if vendor_col:
        log(f"    Vendors P75: {vendor_p75:.1f}", f)

    # Initialize user type
    user_stats['user_type'] = 'mixed'

    # Focused: Short sessions, few products, high CTR or purchase rate
    focused_mask = (
        (user_stats['session_duration_hours_mean'] <= duration_p25) &
        (user_stats['n_products_impressed_mean'] <= products_p25) &
        (user_stats['purchase_rate'] >= purchase_rate_p75)
    )
    user_stats.loc[focused_mask, 'user_type'] = 'focused'

    # Meandering: Long sessions, many products, lots of browsing
    meandering_mask = (
        (user_stats['session_duration_hours_mean'] >= duration_p75) &
        (user_stats['n_products_impressed_mean'] >= products_p75) &
        (user_stats['n_clicks_mean'] >= clicks_p75)
    )
    user_stats.loc[meandering_mask, 'user_type'] = 'meandering'

    # Exploratory: High vendor diversity (if available)
    if vendor_col:
        exploratory_mask = (
            (user_stats[vendor_col] >= vendor_p75) &
            (user_stats['n_products_impressed_mean'] >= products_p75) &
            (user_stats['user_type'] == 'mixed')  # Only classify if not already assigned
        )
        user_stats.loc[exploratory_mask, 'user_type'] = 'exploratory'

    # Print distribution
    log(f"\n  User type distribution:", f)
    for user_type, count in user_stats['user_type'].value_counts().items():
        pct = count / len(user_stats) * 100
        log(f"    {user_type}: {count:,} ({pct:.1f}%)", f)

    return user_stats


def analyze_user_types(user_stats, sessions, f):
    """Analyze characteristics by user type."""
    log("\n" + "="*80, f)
    log("USER TYPE ANALYSIS", f)
    log("="*80, f)

    # Merge user types to sessions for analysis
    sessions_typed = sessions.merge(user_stats[['user_id', 'user_type']], on='user_id', how='left')

    # Analyze by type
    type_analysis = sessions_typed.groupby('user_type').agg({
        'session_id': 'count',
        'session_duration_hours': 'mean',
        'n_impressions': 'mean',
        'n_clicks': 'mean',
        'purchased': 'mean',
        'total_spend': 'mean',
        'n_products_impressed': 'mean'
    }).reset_index()

    type_analysis = type_analysis.rename(columns={
        'session_id': 'n_sessions',
        'purchased': 'purchase_rate',
        'total_spend': 'mean_spend'
    })

    log("\n  Session characteristics by user type:", f)
    log(f"\n  {'Type':<12} {'Sessions':>10} {'Duration':>10} {'Imps':>8} {'Clicks':>8} {'PurchRate':>10} {'AvgSpend':>10}", f)
    log(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}", f)

    for _, row in type_analysis.iterrows():
        log(f"  {row['user_type']:<12} {row['n_sessions']:>10,.0f} {row['session_duration_hours']:>10.1f} "
            f"{row['n_impressions']:>8.1f} {row['n_clicks']:>8.1f} {row['purchase_rate']:>10.3f} {row['mean_spend']:>10.0f}", f)

    return sessions_typed


def compute_type_scores(user_stats, f):
    """Compute continuous scores for each user type dimension."""
    log("\n" + "="*80, f)
    log("COMPUTING TYPE SCORES", f)
    log("="*80, f)

    user_stats = user_stats.copy()

    # Normalize key metrics to 0-1 scale (using percentile rank)
    user_stats['duration_percentile'] = user_stats['session_duration_hours_mean'].rank(pct=True)
    user_stats['products_percentile'] = user_stats['n_products_impressed_mean'].rank(pct=True)
    user_stats['clicks_percentile'] = user_stats['n_clicks_mean'].rank(pct=True)
    user_stats['purchase_rate_percentile'] = user_stats['purchase_rate'].rank(pct=True)

    # Focused score: high when duration low, products low, purchase rate high
    user_stats['focused_score'] = (
        (1 - user_stats['duration_percentile']) +
        (1 - user_stats['products_percentile']) +
        user_stats['purchase_rate_percentile']
    ) / 3

    # Meandering score: high when duration high, products high, clicks high
    user_stats['meandering_score'] = (
        user_stats['duration_percentile'] +
        user_stats['products_percentile'] +
        user_stats['clicks_percentile']
    ) / 3

    # Exploratory score: use vendor diversity if available
    if 'n_vendors_impressed_mean' in user_stats.columns:
        user_stats['vendor_percentile'] = user_stats['n_vendors_impressed_mean'].rank(pct=True)
        user_stats['exploratory_score'] = (
            user_stats['vendor_percentile'] +
            user_stats['products_percentile']
        ) / 2
    else:
        user_stats['exploratory_score'] = user_stats['products_percentile']

    log("\n  Score distributions:", f)
    for score_col in ['focused_score', 'meandering_score', 'exploratory_score']:
        if score_col in user_stats.columns:
            log(f"    {score_col}: mean={user_stats[score_col].mean():.3f}, "
                f"std={user_stats[score_col].std():.3f}", f)

    return user_stats


if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / '08_user_types.txt'

    with open(output_file, 'w') as f:
        log("="*80, f)
        log("08_USER_TYPES - User Type Classification", f)
        log("="*80, f)
        log(f"Data directory: {DATA_DIR}", f)
        log(f"Output directory: {OUTPUT_DIR}", f)

        # Load data
        sessions = load_data(f)

        # Compute user-level stats
        user_stats = compute_user_level_stats(sessions, f)

        # Classify users
        user_stats = classify_users(user_stats, f)

        # Compute type scores
        user_stats = compute_type_scores(user_stats, f)

        # Analyze by type
        sessions_typed = analyze_user_types(user_stats, sessions, f)

        # Save outputs
        log("\n" + "="*80, f)
        log("SAVING OUTPUT", f)
        log("="*80, f)

        # Save user types
        user_types_path = OUTPUT_DIR / 'user_types.parquet'
        user_stats.to_parquet(user_types_path, index=False)
        log(f"  {user_types_path}: {len(user_stats):,} users, {user_stats.shape[1]} columns", f)

        # Save sessions with user type
        sessions_typed_path = OUTPUT_DIR / 'sessions_with_user_type.parquet'
        sessions_typed.to_parquet(sessions_typed_path, index=False)
        log(f"  {sessions_typed_path}: {len(sessions_typed):,} sessions", f)

        # Print column summary
        log(f"\nUser-level columns:", f)
        for col in user_stats.columns:
            log(f"  - {col}", f)

        log(f"\nOutput saved to: {output_file}", f)
