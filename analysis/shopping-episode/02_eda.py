#!/usr/bin/env python3
"""
02_eda.py
Exploratory Data Analysis to validate episode definition and check modeling assumptions.

RESEARCH HYPOTHESES:
- 48-hour gap creates coherent shopping journeys (not too fragmented, not monolithic)
- Outcome sparsity determines need for Hurdle vs OLS models
- Vendor competition (>1 vendor per episode) enables MNLogit
- Winner/Loser overlap validates counterfactual analysis
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "02_eda.txt"

# Thresholds for Go/No-Go decisions
MAX_EPISODE_DURATION_DAYS = 7  # If median > 7 days, gap is too loose
MIN_PURCHASE_RATE = 0.01  # If < 1% episodes have purchases, data is too sparse
MIN_VENDORS_PER_EPISODE = 1.5  # If mean < 1.5, MNLogit is not viable
MIN_LOSS_RATIO = 0.05  # If < 5% of bids are losses, no counterfactuals


def log(msg: str, file=None):
    """Print and optionally write to file."""
    print(msg)
    if file:
        file.write(msg + "\n")


def compute_gini(values: pd.Series) -> float:
    """Compute Gini coefficient for a series of values."""
    values = values.dropna().values
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def validate_episode_definition(df_events: pd.DataFrame, df_episodes: pd.DataFrame, f):
    """Check if 48-hour gap creates sensible episodes."""
    log("\n" + "=" * 80, f)
    log("1. EPISODE DEFINITION VALIDATION", f)
    log("=" * 80, f)

    # Inter-event time distribution
    log("\n--- Inter-Event Time Distribution ---", f)
    time_diffs = df_events['TIME_DIFF_HOURS'].dropna()
    log(f"  Count: {len(time_diffs):,}", f)
    log(f"  Mean: {time_diffs.mean():.2f} hours", f)
    log(f"  Median: {time_diffs.median():.2f} hours", f)
    log(f"  Std: {time_diffs.std():.2f} hours", f)
    log(f"  Min: {time_diffs.min():.2f} hours", f)
    log(f"  Max: {time_diffs.max():.2f} hours", f)

    # Distribution buckets
    log("\n  Distribution by bucket:", f)
    buckets = [0, 1, 6, 12, 24, 48, 72, 168, float('inf')]
    labels = ['<1h', '1-6h', '6-12h', '12-24h', '24-48h', '48-72h', '72h-1wk', '>1wk']
    for i in range(len(buckets)-1):
        mask = (time_diffs >= buckets[i]) & (time_diffs < buckets[i+1])
        pct = mask.sum() / len(time_diffs) * 100
        log(f"    {labels[i]}: {mask.sum():,} ({pct:.1f}%)", f)

    # Episode duration stats
    log("\n--- Episode Duration ---", f)
    log(f"  Episodes: {len(df_episodes):,}", f)
    log(f"  Mean: {df_episodes['DURATION_HOURS'].mean():.2f} hours", f)
    log(f"  Median: {df_episodes['DURATION_HOURS'].median():.2f} hours", f)
    log(f"  Std: {df_episodes['DURATION_HOURS'].std():.2f} hours", f)
    log(f"  5th percentile: {df_episodes['DURATION_HOURS'].quantile(0.05):.2f} hours", f)
    log(f"  25th percentile: {df_episodes['DURATION_HOURS'].quantile(0.25):.2f} hours", f)
    log(f"  75th percentile: {df_episodes['DURATION_HOURS'].quantile(0.75):.2f} hours", f)
    log(f"  95th percentile: {df_episodes['DURATION_HOURS'].quantile(0.95):.2f} hours", f)
    log(f"  Max: {df_episodes['DURATION_HOURS'].max():.2f} hours", f)

    median_days = df_episodes['DURATION_HOURS'].median() / 24
    log(f"\n  Median duration: {median_days:.2f} days", f)

    # GO/NO-GO check
    if median_days > MAX_EPISODE_DURATION_DAYS:
        log(f"\n  [WARNING] Median episode > {MAX_EPISODE_DURATION_DAYS} days. Consider reducing gap threshold.", f)
    else:
        log(f"\n  [OK] Episode duration within expected range.", f)

    # Event density per episode
    log("\n--- Event Density per Episode ---", f)
    log(f"  Mean events: {df_episodes['EVENT_COUNT'].mean():.2f}", f)
    log(f"  Median events: {df_episodes['EVENT_COUNT'].median():.0f}", f)
    log(f"  Mean impressions: {df_episodes['IMPRESSION_COUNT'].mean():.2f}", f)
    log(f"  Median impressions: {df_episodes['IMPRESSION_COUNT'].median():.0f}", f)


def analyze_outcome_sparsity(df_episodes: pd.DataFrame, df_purchases: pd.DataFrame,
                              df_impressions: pd.DataFrame, df_events: pd.DataFrame, f):
    """Determine if zero-inflated models are needed."""
    log("\n" + "=" * 80, f)
    log("2. OUTCOME SPARSITY ANALYSIS", f)
    log("=" * 80, f)

    # Map purchases to episodes
    df_events_pur = df_events[df_events['EVENT_TYPE'] == 'purchase']

    # GMV per episode
    df_pur_with_ep = df_purchases.merge(
        df_events_pur[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
        left_on=['USER_ID', 'PURCHASED_AT'],
        right_on=['USER_ID', 'OCCURRED_AT'],
        how='left'
    )

    episode_gmv = df_pur_with_ep.groupby('EPISODE_ID')['SPEND'].sum().reset_index()
    episode_gmv.columns = ['EPISODE_ID', 'TOTAL_GMV']

    # Merge with all episodes
    df_ep_gmv = df_episodes.merge(episode_gmv, on='EPISODE_ID', how='left')
    df_ep_gmv['TOTAL_GMV'] = df_ep_gmv['TOTAL_GMV'].fillna(0)
    df_ep_gmv['HAS_PURCHASE'] = df_ep_gmv['TOTAL_GMV'] > 0

    log("\n--- Browser vs Buyer ---", f)
    n_total = len(df_ep_gmv)
    n_buyers = df_ep_gmv['HAS_PURCHASE'].sum()
    n_browsers = n_total - n_buyers
    browser_ratio = n_browsers / n_total * 100

    log(f"  Total episodes: {n_total:,}", f)
    log(f"  Buyers (GMV > 0): {n_buyers:,} ({n_buyers/n_total*100:.1f}%)", f)
    log(f"  Browsers (GMV = 0): {n_browsers:,} ({browser_ratio:.1f}%)", f)

    # GO/NO-GO check
    purchase_rate = n_buyers / n_total
    if purchase_rate < MIN_PURCHASE_RATE:
        log(f"\n  [WARNING] Purchase rate < {MIN_PURCHASE_RATE*100}%. Consider Hurdle model.", f)
    elif browser_ratio > 90:
        log(f"\n  [WARNING] Browser ratio > 90%. Consider Hurdle model.", f)
    else:
        log(f"\n  [OK] Purchase rate sufficient for OLS.", f)

    # GMV distribution for buyers
    log("\n--- GMV Distribution (buyers only) ---", f)
    buyers_gmv = df_ep_gmv[df_ep_gmv['HAS_PURCHASE']]['TOTAL_GMV']
    if len(buyers_gmv) > 0:
        log(f"  Count: {len(buyers_gmv):,}", f)
        log(f"  Mean: ${buyers_gmv.mean():.2f}", f)
        log(f"  Median: ${buyers_gmv.median():.2f}", f)
        log(f"  Std: ${buyers_gmv.std():.2f}", f)
        log(f"  Min: ${buyers_gmv.min():.2f}", f)
        log(f"  25th: ${buyers_gmv.quantile(0.25):.2f}", f)
        log(f"  75th: ${buyers_gmv.quantile(0.75):.2f}", f)
        log(f"  95th: ${buyers_gmv.quantile(0.95):.2f}", f)
        log(f"  Max: ${buyers_gmv.max():.2f}", f)

        # Log-normality check
        log_gmv = np.log(buyers_gmv + 1)
        log(f"\n  Log(GMV+1) mean: {log_gmv.mean():.2f}", f)
        log(f"  Log(GMV+1) std: {log_gmv.std():.2f}", f)

    # Organic gap analysis
    log("\n--- Organic Gap Analysis ---", f)

    # Purchases that were impressed/clicked (promoted) vs not (organic)
    if not df_impressions.empty and not df_purchases.empty:
        promoted_products = set(df_impressions['PRODUCT_ID'].dropna().unique())
        df_purchases['IS_PROMOTED'] = df_purchases['PRODUCT_ID'].isin(promoted_products)

        n_promoted = df_purchases['IS_PROMOTED'].sum()
        n_organic = len(df_purchases) - n_promoted
        spend_promoted = df_purchases[df_purchases['IS_PROMOTED']]['SPEND'].sum()
        spend_organic = df_purchases[~df_purchases['IS_PROMOTED']]['SPEND'].sum()

        log(f"  Total purchases: {len(df_purchases):,}", f)
        log(f"  Promoted (product in impressions): {n_promoted:,} ({n_promoted/len(df_purchases)*100:.1f}%)", f)
        log(f"  Organic (product not impressed): {n_organic:,} ({n_organic/len(df_purchases)*100:.1f}%)", f)
        log(f"  Spend promoted: ${spend_promoted:,.2f} ({spend_promoted/(spend_promoted+spend_organic)*100:.1f}%)", f)
        log(f"  Spend organic: ${spend_organic:,.2f} ({spend_organic/(spend_promoted+spend_organic)*100:.1f}%)", f)

    return df_ep_gmv


def analyze_vendor_competition(df_impressions: pd.DataFrame, df_events: pd.DataFrame, f):
    """Check if vendor competition is sufficient for MNLogit."""
    log("\n" + "=" * 80, f)
    log("3. VENDOR COMPETITION ANALYSIS", f)
    log("=" * 80, f)

    if df_impressions.empty:
        log("  No impressions data available", f)
        return

    # Map impressions to episodes
    df_imp_ep = df_impressions.merge(
        df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
        left_on=['USER_ID', 'OCCURRED_AT'],
        right_on=['USER_ID', 'OCCURRED_AT'],
        how='left'
    )

    # Unique vendors per episode
    vendors_per_ep = df_imp_ep.groupby('EPISODE_ID')['VENDOR_ID'].nunique().reset_index()
    vendors_per_ep.columns = ['EPISODE_ID', 'UNIQUE_VENDORS']

    log("\n--- Vendors per Episode ---", f)
    log(f"  Episodes with impressions: {len(vendors_per_ep):,}", f)
    log(f"  Mean vendors: {vendors_per_ep['UNIQUE_VENDORS'].mean():.2f}", f)
    log(f"  Median vendors: {vendors_per_ep['UNIQUE_VENDORS'].median():.0f}", f)
    log(f"  Std: {vendors_per_ep['UNIQUE_VENDORS'].std():.2f}", f)
    log(f"  Min: {vendors_per_ep['UNIQUE_VENDORS'].min()}", f)
    log(f"  Max: {vendors_per_ep['UNIQUE_VENDORS'].max()}", f)

    # Distribution
    log("\n  Distribution:", f)
    for n in [1, 2, 3, 5, 10]:
        count = (vendors_per_ep['UNIQUE_VENDORS'] == n).sum() if n < 10 else (vendors_per_ep['UNIQUE_VENDORS'] >= n).sum()
        pct = count / len(vendors_per_ep) * 100
        label = f"={n}" if n < 10 else f">={n}"
        log(f"    {label} vendors: {count:,} ({pct:.1f}%)", f)

    # GO/NO-GO check
    mean_vendors = vendors_per_ep['UNIQUE_VENDORS'].mean()
    if mean_vendors < MIN_VENDORS_PER_EPISODE:
        log(f"\n  [WARNING] Mean vendors < {MIN_VENDORS_PER_EPISODE}. MNLogit may not be viable.", f)
    else:
        log(f"\n  [OK] Sufficient vendor competition for MNLogit.", f)

    # Share of Voice concentration (Gini within episode)
    log("\n--- Share of Voice Concentration ---", f)

    imp_counts = df_imp_ep.groupby(['EPISODE_ID', 'VENDOR_ID']).size().reset_index(name='IMP_COUNT')
    sov_gini = imp_counts.groupby('EPISODE_ID')['IMP_COUNT'].apply(compute_gini)

    log(f"  Mean Gini (within episode): {sov_gini.mean():.3f}", f)
    log(f"  Median Gini: {sov_gini.median():.3f}", f)
    log(f"  High Gini (>0.8) episodes: {(sov_gini > 0.8).sum():,} ({(sov_gini > 0.8).mean()*100:.1f}%)", f)

    if sov_gini.mean() > 0.8:
        log("\n  [WARNING] High concentration (Gini > 0.8). One vendor dominates most episodes.", f)
    else:
        log("\n  [OK] Reasonable competition within episodes.", f)


def analyze_counterfactual_validity(df_bids: pd.DataFrame, f):
    """Check if winner/loser comparison is valid."""
    log("\n" + "=" * 80, f)
    log("4. COUNTERFACTUAL VALIDITY", f)
    log("=" * 80, f)

    if df_bids.empty:
        log("  No bids data available", f)
        return

    # Winner vs loser counts
    log("\n--- Winner/Loser Distribution ---", f)
    n_total = len(df_bids)
    n_winners = df_bids['IS_WINNER'].sum()
    n_losers = n_total - n_winners
    loss_ratio = n_losers / n_total

    log(f"  Total bids: {n_total:,}", f)
    log(f"  Winners: {n_winners:,} ({n_winners/n_total*100:.1f}%)", f)
    log(f"  Losers: {n_losers:,} ({n_losers/n_total*100:.1f}%)", f)

    # GO/NO-GO check
    if loss_ratio < MIN_LOSS_RATIO:
        log(f"\n  [WARNING] Loss ratio < {MIN_LOSS_RATIO*100}%. Insufficient counterfactuals.", f)
    else:
        log(f"\n  [OK] Sufficient losers for counterfactual analysis.", f)

    # Vendor-level win/loss patterns
    log("\n--- Vendor Win Rates ---", f)
    vendor_stats = df_bids.groupby('VENDOR_ID').agg({
        'IS_WINNER': ['sum', 'count']
    }).reset_index()
    vendor_stats.columns = ['VENDOR_ID', 'WINS', 'BIDS']
    vendor_stats['WIN_RATE'] = vendor_stats['WINS'] / vendor_stats['BIDS']

    log(f"  Unique vendors: {len(vendor_stats):,}", f)
    log(f"  Mean win rate: {vendor_stats['WIN_RATE'].mean():.2%}", f)
    log(f"  Median win rate: {vendor_stats['WIN_RATE'].median():.2%}", f)

    # Vendors with both wins and losses
    mixed_vendors = vendor_stats[(vendor_stats['WINS'] > 0) & (vendor_stats['WINS'] < vendor_stats['BIDS'])]
    log(f"  Vendors with both wins and losses: {len(mixed_vendors):,} ({len(mixed_vendors)/len(vendor_stats)*100:.1f}%)", f)

    # Rank distribution
    log("\n--- Rank Distribution ---", f)
    for rank in range(1, 11):
        count = (df_bids['RANKING'] == rank).sum()
        pct = count / n_total * 100
        log(f"    Rank {rank}: {count:,} ({pct:.1f}%)", f)


def generate_go_no_go_summary(df_episodes: pd.DataFrame, df_ep_gmv: pd.DataFrame,
                               df_impressions: pd.DataFrame, df_events: pd.DataFrame,
                               df_bids: pd.DataFrame, f):
    """Generate final Go/No-Go summary."""
    log("\n" + "=" * 80, f)
    log("5. GO/NO-GO SUMMARY", f)
    log("=" * 80, f)

    decisions = []

    # 1. Episode duration
    median_days = df_episodes['DURATION_HOURS'].median() / 24
    if median_days > MAX_EPISODE_DURATION_DAYS:
        decisions.append(("Episode Duration", "NO-GO", f"Median {median_days:.1f} days > {MAX_EPISODE_DURATION_DAYS} days. Reduce gap."))
    else:
        decisions.append(("Episode Duration", "GO", f"Median {median_days:.1f} days"))

    # 2. Purchase rate
    if df_ep_gmv is not None:
        purchase_rate = df_ep_gmv['HAS_PURCHASE'].mean()
        if purchase_rate < MIN_PURCHASE_RATE:
            decisions.append(("Purchase Rate", "WARN", f"{purchase_rate*100:.1f}% < {MIN_PURCHASE_RATE*100}%. Use Hurdle model."))
        else:
            decisions.append(("Purchase Rate", "GO", f"{purchase_rate*100:.1f}%"))

    # 3. Vendor competition
    if not df_impressions.empty:
        df_imp_ep = df_impressions.merge(
            df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
            on=['USER_ID', 'OCCURRED_AT'],
            how='left'
        )
        vendors_per_ep = df_imp_ep.groupby('EPISODE_ID')['VENDOR_ID'].nunique()
        mean_vendors = vendors_per_ep.mean()
        if mean_vendors < MIN_VENDORS_PER_EPISODE:
            decisions.append(("Vendor Competition", "NO-GO", f"Mean {mean_vendors:.2f} < {MIN_VENDORS_PER_EPISODE}. Binary Logit only."))
        else:
            decisions.append(("Vendor Competition", "GO", f"Mean {mean_vendors:.2f} vendors/episode"))

    # 4. Counterfactual validity
    if not df_bids.empty:
        loss_ratio = (~df_bids['IS_WINNER']).mean()
        if loss_ratio < MIN_LOSS_RATIO:
            decisions.append(("Counterfactuals", "NO-GO", f"Loss ratio {loss_ratio*100:.1f}% < {MIN_LOSS_RATIO*100}%"))
        else:
            decisions.append(("Counterfactuals", "GO", f"Loss ratio {loss_ratio*100:.1f}%"))

    # Print summary table
    log("\n| Check | Status | Detail |", f)
    log("|-------|--------|--------|", f)
    for check, status, detail in decisions:
        log(f"| {check} | {status} | {detail} |", f)

    # Overall decision
    no_go_count = sum(1 for _, status, _ in decisions if status == "NO-GO")
    warn_count = sum(1 for _, status, _ in decisions if status == "WARN")

    log("\n--- OVERALL ---", f)
    if no_go_count > 0:
        log(f"  [NO-GO] {no_go_count} critical issue(s). Address before modeling.", f)
    elif warn_count > 0:
        log(f"  [PROCEED WITH CAUTION] {warn_count} warning(s). Consider adjustments.", f)
    else:
        log(f"  [GO] All checks passed. Proceed to modeling.", f)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("02_EDA (Episode Validation)", f)
        log("=" * 80, f)
        log(f"\nTimestamp: {datetime.now()}", f)

        # Load data
        log("\n--- LOADING DATA ---", f)

        try:
            df_events = pd.read_parquet(DATA_DIR / "events.parquet")
            log(f"  events.parquet: {len(df_events):,} rows", f)
        except FileNotFoundError:
            log("  ERROR: events.parquet not found. Run 01_data_pull.py first.", f)
            return

        try:
            df_episodes = pd.read_parquet(DATA_DIR / "episodes.parquet")
            log(f"  episodes.parquet: {len(df_episodes):,} rows", f)
        except FileNotFoundError:
            log("  ERROR: episodes.parquet not found. Run 01_data_pull.py first.", f)
            return

        try:
            df_purchases = pd.read_parquet(DATA_DIR / "purchases.parquet")
            log(f"  purchases.parquet: {len(df_purchases):,} rows", f)
        except FileNotFoundError:
            df_purchases = pd.DataFrame()
            log("  purchases.parquet: Not found (empty)", f)

        try:
            df_impressions = pd.read_parquet(DATA_DIR / "impressions.parquet")
            log(f"  impressions.parquet: {len(df_impressions):,} rows", f)
        except FileNotFoundError:
            df_impressions = pd.DataFrame()
            log("  impressions.parquet: Not found (empty)", f)

        try:
            df_bids = pd.read_parquet(DATA_DIR / "auctions_results.parquet")
            log(f"  auctions_results.parquet: {len(df_bids):,} rows", f)
        except FileNotFoundError:
            df_bids = pd.DataFrame()
            log("  auctions_results.parquet: Not found (empty)", f)

        # Run analyses
        validate_episode_definition(df_events, df_episodes, f)

        df_ep_gmv = analyze_outcome_sparsity(df_episodes, df_purchases, df_impressions, df_events, f)

        analyze_vendor_competition(df_impressions, df_events, f)

        analyze_counterfactual_validity(df_bids, f)

        generate_go_no_go_summary(df_episodes, df_ep_gmv, df_impressions, df_events, df_bids, f)

        log("\n" + "=" * 80, f)
        log("02_EDA COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
