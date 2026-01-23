#!/usr/bin/env python3
"""
Synthetic Data Generator for Multi-Day Shopping Session Pipeline (OPTIMIZED)

Generates 6 parquet files matching the real data schema.
Includes ground-truth treatment effect (beta=5.0) for regression validation.

Usage:
    python 00_generate_synthetic_data.py --scale medium
"""

import argparse
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ScaleConfig:
    """Scale parameters for data generation."""
    n_users: int
    n_vendors: int
    n_products: int
    n_weeks: int
    auctions_per_user_per_week: float

    @property
    def n_campaigns(self) -> int:
        return self.n_vendors * 3

SCALE_CONFIGS = {
    'small': ScaleConfig(1_000, 200, 5_000, 12, 3.0),
    'medium': ScaleConfig(10_000, 1_000, 50_000, 26, 3.0),
    'large': ScaleConfig(50_000, 5_000, 100_000, 52, 3.0)
}

# Ground-truth treatment effect
TRUE_BETA = 5.0

# Funnel conversion rates
IMPRESSION_RATE = 0.13
CLICK_RATE = 0.03
PURCHASE_RATE = 0.15

# Fixed effect distributions
USER_BASELINE_MEAN = 50.0
USER_BASELINE_STD = 20.0
WEEK_SHOCK_STD = 5.0
VENDOR_EFFECT_STD = 10.0
NOISE_STD = 15.0
SELECTION_GAMMA = 0.02

# =============================================================================
# VECTORIZED ID GENERATORS
# =============================================================================

def generate_hex_ids(n: int) -> np.ndarray:
    """Generate n hex string IDs."""
    return np.array([uuid.uuid4().hex for _ in range(n)])

def generate_uuid_strs(n: int) -> np.ndarray:
    """Generate n UUID strings."""
    return np.array([str(uuid.uuid4()) for _ in range(n)])

def generate_user_ids(n: int) -> np.ndarray:
    """Generate n user IDs in ext1:uuid format."""
    return np.array([f"ext1:{uuid.uuid4()}" for _ in range(n)])

# =============================================================================
# ENTITY POOL GENERATION
# =============================================================================

def generate_entity_pools(config: ScaleConfig, rng: np.random.Generator) -> Dict:
    """Generate pools of users, vendors, products, campaigns with fixed effects."""
    print("Generating entity pools...")

    # Users
    print(f"  Creating {config.n_users:,} users...")
    user_ids = generate_user_ids(config.n_users)
    user_baselines = rng.normal(USER_BASELINE_MEAN, USER_BASELINE_STD, config.n_users)
    user_intents = user_baselines / USER_BASELINE_MEAN
    users_df = pd.DataFrame({
        'user_id': user_ids,
        'user_idx': np.arange(config.n_users),
        'baseline': user_baselines,
        'intent': user_intents
    })

    # Vendors
    print(f"  Creating {config.n_vendors:,} vendors...")
    vendor_ids = generate_hex_ids(config.n_vendors)
    vendor_effects = rng.normal(0, VENDOR_EFFECT_STD, config.n_vendors)
    vendors_df = pd.DataFrame({
        'vendor_id': vendor_ids,
        'vendor_idx': np.arange(config.n_vendors),
        'effect': vendor_effects
    })

    # Campaigns (3 per vendor)
    print(f"  Creating {config.n_campaigns:,} campaigns...")
    campaign_ids = generate_hex_ids(config.n_campaigns)
    campaign_vendors = np.repeat(vendor_ids, 3)
    campaigns_df = pd.DataFrame({
        'campaign_id': campaign_ids,
        'vendor_id': campaign_vendors
    })

    # Products
    print(f"  Creating {config.n_products:,} products...")
    product_ids = generate_hex_ids(config.n_products)
    product_prices = np.exp(rng.normal(3, 1, config.n_products)).round(2)
    products_df = pd.DataFrame({
        'product_id': product_ids,
        'product_idx': np.arange(config.n_products),
        'price': product_prices,
        'active': rng.random(config.n_products) < 0.95,
        'is_deleted': rng.random(config.n_products) < 0.05
    })

    # Weeks
    print(f"  Creating {config.n_weeks} weeks...")
    start_date = datetime(2024, 1, 1)
    week_starts = [start_date + timedelta(weeks=w) for w in range(config.n_weeks)]
    week_shocks = rng.normal(0, WEEK_SHOCK_STD, config.n_weeks)
    year_weeks = [f"{w.year}_W{w.isocalendar()[1]:02d}" for w in week_starts]
    weeks_df = pd.DataFrame({
        'week_idx': np.arange(config.n_weeks),
        'year_week': year_weeks,
        'week_start': week_starts,
        'shock': week_shocks
    })

    return {
        'users': users_df,
        'vendors': vendors_df,
        'campaigns': campaigns_df,
        'products': products_df,
        'weeks': weeks_df
    }

# =============================================================================
# CATALOG GENERATION
# =============================================================================

def generate_catalog(pools: Dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate CATALOG table."""
    print("\nGenerating CATALOG...")
    products_df = pools['products']
    vendors_df = pools['vendors']
    vendor_ids = vendors_df['vendor_id'].values

    n_products = len(products_df)

    # Assign 1-3 vendors to each product
    n_vendors_per_product = rng.integers(1, 4, n_products)
    vendor_lists = []
    for i in range(n_products):
        n_v = n_vendors_per_product[i]
        assigned = rng.choice(vendor_ids, size=n_v, replace=False).tolist()
        vendor_lists.append(str(assigned))

    catalog = pd.DataFrame({
        'PRODUCT_ID': products_df['product_id'].values,
        'NAME': [f"Product_{i}" for i in range(n_products)],
        'ACTIVE': products_df['active'].values,
        'CATEGORIES': [f'["category-{rng.integers(1, 20)}"]' for _ in range(n_products)],
        'DESCRIPTION': [f"Description for Product_{i}" for i in range(n_products)],
        'PRICE': products_df['price'].values,
        'VENDORS': vendor_lists,
        'IS_DELETED': products_df['is_deleted'].values
    })

    print(f"  Catalog rows: {len(catalog):,}")
    return catalog

# =============================================================================
# VECTORIZED AUCTIONS GENERATION
# =============================================================================

def generate_auctions(pools: Dict, config: ScaleConfig, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate AUCTIONS_USERS and AUCTIONS_RESULTS tables (vectorized)."""
    print("\nGenerating AUCTIONS (vectorized)...")

    users_df = pools['users']
    vendors_df = pools['vendors']
    campaigns_df = pools['campaigns']
    products_df = pools['products']
    weeks_df = pools['weeks']

    # Pre-compute total expected auctions
    # Each user has Poisson(lambda) auctions per week, lambda scaled by intent
    # Expected total = n_users * n_weeks * mean_lambda
    mean_lambda = config.auctions_per_user_per_week
    expected_auctions = int(config.n_users * config.n_weeks * mean_lambda * 1.5)  # buffer

    print(f"  Pre-allocating for ~{expected_auctions:,} auctions...")

    # Generate auction counts per (user, week)
    user_intents = users_df['intent'].values
    auction_lambdas = mean_lambda * (1 + SELECTION_GAMMA * user_intents)

    # Generate all auction counts at once
    auction_counts = rng.poisson(
        np.outer(auction_lambdas, np.ones(config.n_weeks)).flatten()
    ).reshape(config.n_users, config.n_weeks)

    total_auctions = auction_counts.sum()
    print(f"  Total auctions: {total_auctions:,}")

    # Generate auction IDs
    print("  Generating auction IDs...")
    auction_ids = generate_hex_ids(total_auctions)

    # Build auctions_users table
    placements = ['search', 'browse', 'pdp']
    user_ids = users_df['user_id'].values
    week_starts = weeks_df['week_start'].values

    auctions_users_records = []
    auction_idx = 0

    for week_idx in tqdm(range(config.n_weeks), desc="Building auctions"):
        week_start = pd.Timestamp(week_starts[week_idx]).to_pydatetime()

        for user_idx in range(config.n_users):
            n_auctions = auction_counts[user_idx, week_idx]
            if n_auctions == 0:
                continue

            for _ in range(n_auctions):
                auction_time = week_start + timedelta(
                    days=int(rng.integers(0, 7)),
                    hours=int(rng.integers(0, 24)),
                    minutes=int(rng.integers(0, 60))
                )
                auctions_users_records.append({
                    'AUCTION_ID': auction_ids[auction_idx],
                    'OPAQUE_USER_ID': user_ids[user_idx],
                    'CREATED_AT': auction_time,
                    'PLACEMENT': placements[rng.integers(0, 3)],
                    'week_idx': week_idx,
                    'user_idx': user_idx
                })
                auction_idx += 1

    auctions_users_df = pd.DataFrame(auctions_users_records)
    print(f"  Auctions: {len(auctions_users_df):,}")

    # Generate bids for each auction (vectorized per batch)
    print("  Generating bids...")
    vendor_ids = vendors_df['vendor_id'].values
    product_ids = products_df['product_id'].values
    product_prices = products_df['price'].values

    # Map campaigns to vendors
    campaign_vendor_map = campaigns_df.groupby('vendor_id')['campaign_id'].apply(list).to_dict()

    auctions_results_records = []
    batch_size = 10000

    for batch_start in tqdm(range(0, len(auctions_users_df), batch_size), desc="Bid batches"):
        batch = auctions_users_df.iloc[batch_start:batch_start + batch_size]

        for _, auction in batch.iterrows():
            n_bids = rng.integers(5, 30)
            bid_vendor_indices = rng.choice(len(vendor_ids), size=min(n_bids, len(vendor_ids)), replace=False)

            for rank, v_idx in enumerate(bid_vendor_indices, 1):
                vendor_id = vendor_ids[v_idx]
                campaigns = campaign_vendor_map.get(vendor_id, [])
                campaign_id = rng.choice(campaigns) if campaigns else generate_hex_ids(1)[0]

                p_idx = rng.integers(0, len(product_ids))

                auctions_results_records.append({
                    'AUCTION_ID': auction['AUCTION_ID'],
                    'VENDOR_ID': vendor_id,
                    'CAMPAIGN_ID': campaign_id,
                    'PRODUCT_ID': product_ids[p_idx],
                    'RANKING': rank,
                    'IS_WINNER': rank <= 5,
                    'CREATED_AT': auction['CREATED_AT'],
                    'QUALITY': rng.beta(2, 5),
                    'FINAL_BID': np.exp(rng.normal(-3, 1)),
                    'PRICE': product_prices[p_idx],
                    'CONVERSION_RATE': rng.beta(1, 20),
                    'PACING': rng.beta(8, 2)
                })

    auctions_results_df = pd.DataFrame(auctions_results_records)
    print(f"  Bids: {len(auctions_results_df):,}")
    print(f"  Winners: {auctions_results_df['IS_WINNER'].sum():,}")

    # Clean up helper columns
    auctions_users_df = auctions_users_df.drop(columns=['week_idx', 'user_idx'])

    return auctions_users_df, auctions_results_df

# =============================================================================
# IMPRESSIONS & CLICKS GENERATION
# =============================================================================

def generate_impressions_clicks(
    auctions_users_df: pd.DataFrame,
    auctions_results_df: pd.DataFrame,
    pools: Dict,
    rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate IMPRESSIONS and CLICKS tables."""
    print("\nGenerating IMPRESSIONS and CLICKS...")

    users_df = pools['users']
    user_intent_map = users_df.set_index('user_id')['intent'].to_dict()

    # Only winning bids
    winners = auctions_results_df[auctions_results_df['IS_WINNER']].copy()

    # Join with auctions to get user info
    winners = winners.merge(
        auctions_users_df[['AUCTION_ID', 'OPAQUE_USER_ID', 'CREATED_AT']],
        on='AUCTION_ID',
        suffixes=('', '_auction')
    )

    print(f"  Processing {len(winners):,} winning bids...")

    # Vectorized impression generation
    imp_probs = rng.random(len(winners))
    imp_mask = imp_probs < IMPRESSION_RATE

    impressions_data = winners[imp_mask].copy()
    impressions_data['INTERACTION_ID'] = generate_uuid_strs(len(impressions_data))
    impressions_data['OCCURRED_AT'] = impressions_data['CREATED_AT_auction'] + pd.to_timedelta(
        rng.integers(1, 10, len(impressions_data)), unit='s'
    )
    impressions_data['USER_ID'] = impressions_data['OPAQUE_USER_ID']

    impressions_df = impressions_data[['INTERACTION_ID', 'AUCTION_ID', 'PRODUCT_ID',
                                        'USER_ID', 'CAMPAIGN_ID', 'VENDOR_ID', 'OCCURRED_AT']].copy()

    print(f"  Impressions: {len(impressions_df):,}")

    # Vectorized click generation
    click_probs = rng.random(len(impressions_df))
    intents = impressions_df['USER_ID'].map(user_intent_map).fillna(1.0).values
    click_thresholds = CLICK_RATE * (1 + SELECTION_GAMMA * intents)
    click_mask = click_probs < click_thresholds

    clicks_data = impressions_df[click_mask].copy()
    clicks_data['INTERACTION_ID'] = generate_uuid_strs(len(clicks_data))
    clicks_data['OCCURRED_AT'] = clicks_data['OCCURRED_AT'] + pd.to_timedelta(
        rng.integers(1, 30, len(clicks_data)), unit='s'
    )

    clicks_df = clicks_data[['INTERACTION_ID', 'AUCTION_ID', 'PRODUCT_ID',
                              'USER_ID', 'CAMPAIGN_ID', 'VENDOR_ID', 'OCCURRED_AT']].copy()

    print(f"  Clicks: {len(clicks_df):,}")
    if len(impressions_df) > 0:
        print(f"  CTR: {len(clicks_df)/len(impressions_df)*100:.2f}%")

    return impressions_df, clicks_df

# =============================================================================
# PURCHASES GENERATION (WITH GROUND-TRUTH TREATMENT EFFECT)
# =============================================================================

def generate_purchases(
    clicks_df: pd.DataFrame,
    pools: Dict,
    config: ScaleConfig,
    rng: np.random.Generator
) -> pd.DataFrame:
    """Generate PURCHASES with ground-truth treatment effect β."""
    print(f"\nGenerating PURCHASES with ground-truth β = {TRUE_BETA}...")

    if len(clicks_df) == 0:
        print("  No clicks - no purchases")
        return pd.DataFrame(columns=['PURCHASE_ID', 'PURCHASED_AT', 'PRODUCT_ID',
                                      'QUANTITY', 'UNIT_PRICE', 'USER_ID', 'PURCHASE_LINE'])

    users_df = pools['users']
    vendors_df = pools['vendors']
    weeks_df = pools['weeks']

    user_baseline_map = users_df.set_index('user_id')['baseline'].to_dict()
    vendor_effect_map = vendors_df.set_index('vendor_id')['effect'].to_dict()
    week_shock_map = weeks_df.set_index('week_idx')['shock'].to_dict()

    # Add week_idx to clicks
    clicks_df = clicks_df.copy()
    clicks_df['week_idx'] = ((clicks_df['OCCURRED_AT'] - datetime(2024, 1, 1)).dt.days // 7).clip(0, config.n_weeks - 1)

    # Aggregate clicks by (user, week, vendor)
    click_agg = clicks_df.groupby(['USER_ID', 'week_idx', 'VENDOR_ID']).agg({
        'PRODUCT_ID': 'first',
        'OCCURRED_AT': 'max'
    }).reset_index()
    click_agg['n_clicks'] = clicks_df.groupby(['USER_ID', 'week_idx', 'VENDOR_ID']).size().values

    print(f"  Click aggregates: {len(click_agg):,}")

    # Apply DGP
    alpha_u = click_agg['USER_ID'].map(user_baseline_map).fillna(USER_BASELINE_MEAN).values
    lambda_t = click_agg['week_idx'].map(week_shock_map).fillna(0).values
    phi_v = click_agg['VENDOR_ID'].map(vendor_effect_map).fillna(0).values
    n_clicks = click_agg['n_clicks'].values
    epsilon = rng.normal(0, NOISE_STD, len(click_agg))

    spend = alpha_u + lambda_t + phi_v + TRUE_BETA * n_clicks + epsilon
    spend = np.maximum(0, spend)

    # Conversion probability
    purchase_probs = rng.random(len(click_agg))
    purchase_mask = (purchase_probs < PURCHASE_RATE) & (spend > 0)

    purchases_data = click_agg[purchase_mask].copy()
    purchases_data['spend'] = spend[purchase_mask]

    # Generate purchase records
    purchases_data['PURCHASE_ID'] = generate_uuid_strs(len(purchases_data))
    # Keep purchases in same week as clicks (small lag: avg 2 hours)
    purchases_data['PURCHASED_AT'] = purchases_data['OCCURRED_AT'] + pd.to_timedelta(
        rng.exponential(2, len(purchases_data)), unit='h'
    )
    purchases_data['QUANTITY'] = np.where(rng.random(len(purchases_data)) < 0.9, 1, rng.integers(2, 4, len(purchases_data)))
    purchases_data['UNIT_PRICE'] = (purchases_data['spend'] / purchases_data['QUANTITY']).round(2)
    purchases_data['PURCHASE_LINE'] = 1

    purchases_df = purchases_data[['PURCHASE_ID', 'PURCHASED_AT', 'PRODUCT_ID',
                                    'QUANTITY', 'UNIT_PRICE', 'USER_ID', 'PURCHASE_LINE']].copy()
    purchases_df = purchases_df.rename(columns={'USER_ID': 'USER_ID'})

    print(f"  Purchases: {len(purchases_df):,}")
    print(f"  Conversion rate: {len(purchases_df)/len(clicks_df)*100:.2f}%")

    return purchases_df

# =============================================================================
# VALIDATION
# =============================================================================

def validate_data(
    auctions_users_df: pd.DataFrame,
    auctions_results_df: pd.DataFrame,
    impressions_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    purchases_df: pd.DataFrame,
    catalog_df: pd.DataFrame
) -> None:
    """Validate generated data."""
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print("\n1. Composite Key Joins:")
    if len(clicks_df) > 0 and len(impressions_df) > 0:
        merged = clicks_df.merge(
            impressions_df[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID']],
            on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID'],
            how='inner'
        )
        print(f"   Clicks → Impressions: {len(merged)/len(clicks_df)*100:.1f}% match")

    if len(impressions_df) > 0:
        winners = auctions_results_df[auctions_results_df['IS_WINNER']]
        merged = impressions_df.merge(
            winners[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID']],
            on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID', 'CAMPAIGN_ID'],
            how='inner'
        )
        print(f"   Impressions → Winners: {len(merged)/len(impressions_df)*100:.1f}% match")

    print("\n2. Funnel Rates:")
    print(f"   Auctions: {len(auctions_users_df):,}")
    print(f"   Bids: {len(auctions_results_df):,}")
    print(f"   Winners: {auctions_results_df['IS_WINNER'].sum():,}")
    print(f"   Impressions: {len(impressions_df):,}")
    print(f"   Clicks: {len(clicks_df):,}")
    print(f"   Purchases: {len(purchases_df):,}")

    if len(impressions_df) > 0:
        print(f"   CTR: {len(clicks_df)/len(impressions_df)*100:.2f}%")
    if len(clicks_df) > 0:
        print(f"   Click→Purchase: {len(purchases_df)/len(clicks_df)*100:.2f}%")

    print("\n3. Distributions:")
    if len(auctions_results_df) > 0:
        print(f"   Final Bid: median=${auctions_results_df['FINAL_BID'].median():.3f}")
        print(f"   Quality: mean={auctions_results_df['QUALITY'].mean():.3f}")
        print(f"   Pacing: mean={auctions_results_df['PACING'].mean():.3f}")
    if len(purchases_df) > 0:
        print(f"   Purchase Value: median=${purchases_df['UNIT_PRICE'].median():.2f}")

    print("\n4. Ground Truth:")
    print(f"   TRUE_BETA = {TRUE_BETA}")
    print(f"   Expected: User+Week+Vendor FE should recover β ≈ {TRUE_BETA}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for pipeline testing')
    parser.add_argument('--scale', choices=['small', 'medium', 'large'], default='medium')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='data')
    args = parser.parse_args()

    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR (OPTIMIZED)")
    print("=" * 80)

    config = SCALE_CONFIGS[args.scale]
    print(f"\nScale: {args.scale}")
    print(f"  Users: {config.n_users:,}")
    print(f"  Vendors: {config.n_vendors:,}")
    print(f"  Products: {config.n_products:,}")
    print(f"  Weeks: {config.n_weeks}")
    print(f"  Campaigns: {config.n_campaigns:,}")
    print(f"\nGround Truth β = {TRUE_BETA}")

    rng = np.random.default_rng(args.seed)

    pools = generate_entity_pools(config, rng)
    catalog_df = generate_catalog(pools, rng)
    auctions_users_df, auctions_results_df = generate_auctions(pools, config, rng)
    impressions_df, clicks_df = generate_impressions_clicks(
        auctions_users_df, auctions_results_df, pools, rng
    )
    purchases_df = generate_purchases(clicks_df, pools, config, rng)

    validate_data(
        auctions_users_df, auctions_results_df,
        impressions_df, clicks_df,
        purchases_df, catalog_df
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING TO PARQUET")
    print("=" * 80)

    files = {
        'auctions_users_synth.parquet': auctions_users_df,
        'auctions_results_synth.parquet': auctions_results_df,
        'impressions_synth.parquet': impressions_df,
        'clicks_synth.parquet': clicks_df,
        'purchases_synth.parquet': purchases_df,
        'catalog_synth.parquet': catalog_df
    }

    for filename, df in files.items():
        filepath = output_dir / filename
        df.to_parquet(filepath, index=False)
        size_mb = filepath.stat().st_size / 1e6
        print(f"  {filename}: {len(df):,} rows, {size_mb:.1f} MB")

    # Save metadata
    import json
    metadata = {
        'scale': args.scale,
        'seed': args.seed,
        'true_beta': TRUE_BETA,
        'n_users': config.n_users,
        'n_vendors': config.n_vendors,
        'n_products': config.n_products,
        'n_weeks': config.n_weeks
    }

    with open(output_dir / 'synthetic_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_dir.resolve()}")
    print(f"TRUE_BETA = {TRUE_BETA} (regressions should recover this)")

if __name__ == '__main__':
    main()
