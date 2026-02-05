#!/usr/bin/env python3
"""
Placement Product Characteristics

Product diversity, brand/category patterns by placement.
Documents what kinds of products appear in each placement.

Usage:
    python 07_placement_product_characteristics.py --round round1
    python 07_placement_product_characteristics.py --round round2
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_BASE = BASE_DIR / "0_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_data_paths(round_name):
    """Return data paths for specified round."""
    if round_name == "round1":
        return {
            'auctions_results': DATA_BASE / "round1/auctions_results_all.parquet",
            'auctions_users': DATA_BASE / "round1/auctions_users_all.parquet",
            'catalog': DATA_BASE / "round1/catalog_all.parquet",
        }
    elif round_name == "round2":
        return {
            'auctions_results': DATA_BASE / "round2/auctions_results_r2.parquet",
            'auctions_users': DATA_BASE / "round2/auctions_users_r2.parquet",
            'catalog': DATA_BASE / "round2/catalog_r2.parquet",
        }
    else:
        raise ValueError(f"Unknown round: {round_name}")


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def product_diversity_by_placement(ar_with_placement, f):
    """Analyze product diversity by placement."""
    log(f"\n{'='*80}", f)
    log(f"PRODUCT DIVERSITY BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None or 'PLACEMENT' not in ar_with_placement.columns:
        log("PLACEMENT column not available", f)
        return

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    log(f"\n--- Unique Products by Placement ---", f)
    log(f"{'Placement':<12} {'Total Bids':>15} {'Unique Products':>18} {'Products/1K Bids':>18}", f)
    log(f"{'-'*12} {'-'*15} {'-'*18} {'-'*18}", f)

    for p in placements:
        subset = ar_with_placement[ar_with_placement['PLACEMENT'] == p]
        n_bids = len(subset)
        n_products = subset['PRODUCT_ID'].nunique()
        products_per_1k = n_products / (n_bids / 1000) if n_bids > 0 else 0
        log(f"{str(p):<12} {n_bids:>15,} {n_products:>18,} {products_per_1k:>18.1f}", f)

    log(f"\n--- Products per Auction by Placement ---", f)
    log(f"{'Placement':<12} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}", f)

    for p in placements:
        subset = ar_with_placement[ar_with_placement['PLACEMENT'] == p]
        products_per_auction = subset.groupby('AUCTION_ID')['PRODUCT_ID'].nunique()
        if len(products_per_auction) > 0:
            log(f"{str(p):<12} {products_per_auction.mean():>10.1f} {products_per_auction.median():>10.0f} {products_per_auction.min():>8} {products_per_auction.max():>8}", f)


def product_concentration_by_placement(ar_with_placement, f):
    """Analyze product concentration (how often same products appear)."""
    log(f"\n{'='*80}", f)
    log(f"PRODUCT CONCENTRATION BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None or 'PLACEMENT' not in ar_with_placement.columns:
        log("PLACEMENT column not available", f)
        return

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    log(f"\n--- Top 10 Product Share by Placement ---", f)
    log(f"{'Placement':<12} {'Top 10 Share %':>18} {'Top 100 Share %':>18}", f)
    log(f"{'-'*12} {'-'*18} {'-'*18}", f)

    for p in placements:
        subset = ar_with_placement[ar_with_placement['PLACEMENT'] == p]
        product_counts = subset['PRODUCT_ID'].value_counts()

        if len(product_counts) > 0:
            top10_share = product_counts.head(10).sum() / len(subset) * 100
            top100_share = product_counts.head(100).sum() / len(subset) * 100 if len(product_counts) >= 100 else 100
            log(f"{str(p):<12} {top10_share:>17.1f}% {top100_share:>17.1f}%", f)

    log(f"\n--- Top 5 Products per Placement ---", f)

    for p in placements:
        log(f"\nPlacement {p}:", f)
        subset = ar_with_placement[ar_with_placement['PLACEMENT'] == p]
        product_counts = subset['PRODUCT_ID'].value_counts().head(5)

        log(f"  {'Product ID (truncated)':<35} {'Count':>10} {'%':>8}", f)
        log(f"  {'-'*35} {'-'*10} {'-'*8}", f)

        for prod_id, count in product_counts.items():
            pct = count / len(subset) * 100
            log(f"  {str(prod_id)[:35]:<35} {count:>10,} {pct:>7.2f}%", f)


def price_distribution_by_placement(ar_with_placement, catalog, f):
    """Analyze price distribution by placement."""
    log(f"\n{'='*80}", f)
    log(f"PRICE DISTRIBUTION BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None or catalog is None:
        log("Required data not available", f)
        return

    if 'PLACEMENT' not in ar_with_placement.columns:
        log("PLACEMENT column not available", f)
        return

    price_col = None
    for col in ['CATALOG_PRICE', 'PRICE']:
        if col in catalog.columns:
            price_col = col
            break

    if price_col is None:
        log("Price column not found in catalog", f)
        return

    winners = ar_with_placement[ar_with_placement['IS_WINNER'] == True].copy()
    winners_with_price = winners.merge(catalog[['PRODUCT_ID', price_col]], on='PRODUCT_ID', how='left')

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    log(f"\n--- Price Statistics by Placement (Winners Only) ---", f)
    log(f"{'Placement':<12} {'Mean':>12} {'Median':>12} {'P25':>10} {'P75':>10} {'P95':>12}", f)
    log(f"{'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}", f)

    for p in placements:
        prices = winners_with_price[winners_with_price['PLACEMENT'] == p][price_col].dropna()
        if len(prices) > 0:
            log(f"{str(p):<12} ${prices.mean():>10.2f} ${prices.median():>10.2f} ${prices.quantile(0.25):>8.2f} ${prices.quantile(0.75):>8.2f} ${prices.quantile(0.95):>10.2f}", f)

    log(f"\n--- Price Range Distribution by Placement ---", f)
    log(f"{'Placement':<12} {'<$20':>10} {'$20-50':>10} {'$50-100':>10} {'$100-500':>12} {'$500+':>10}", f)
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}", f)

    for p in placements:
        prices = winners_with_price[winners_with_price['PLACEMENT'] == p][price_col].dropna()
        total = len(prices)
        if total > 0:
            pct_under_20 = (prices < 20).sum() / total * 100
            pct_20_50 = ((prices >= 20) & (prices < 50)).sum() / total * 100
            pct_50_100 = ((prices >= 50) & (prices < 100)).sum() / total * 100
            pct_100_500 = ((prices >= 100) & (prices < 500)).sum() / total * 100
            pct_500plus = (prices >= 500).sum() / total * 100
            log(f"{str(p):<12} {pct_under_20:>9.1f}% {pct_20_50:>9.1f}% {pct_50_100:>9.1f}% {pct_100_500:>11.1f}% {pct_500plus:>9.1f}%", f)


def category_analysis_by_placement(ar_with_placement, catalog, f):
    """Analyze category patterns by placement."""
    log(f"\n{'='*80}", f)
    log(f"CATEGORY ANALYSIS BY PLACEMENT", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None or catalog is None:
        log("Required data not available", f)
        return

    if 'CATEGORIES' not in catalog.columns:
        log("CATEGORIES column not found in catalog", f)
        return

    if 'PLACEMENT' not in ar_with_placement.columns:
        log("PLACEMENT column not available", f)
        return

    winners = ar_with_placement[ar_with_placement['IS_WINNER'] == True].copy()
    winners_with_cat = winners.merge(catalog[['PRODUCT_ID', 'CATEGORIES']], on='PRODUCT_ID', how='left')

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    log(f"\n--- Category Presence by Placement ---", f)
    log(f"{'Placement':<12} {'Winners':>12} {'With Categories':>18} {'%':>10}", f)
    log(f"{'-'*12} {'-'*12} {'-'*18} {'-'*10}", f)

    for p in placements:
        subset = winners_with_cat[winners_with_cat['PLACEMENT'] == p]
        has_categories = subset['CATEGORIES'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)
        n_with_cats = has_categories.sum()
        pct = n_with_cats / len(subset) * 100 if len(subset) > 0 else 0
        log(f"{str(p):<12} {len(subset):>12,} {n_with_cats:>18,} {pct:>9.1f}%", f)

    log(f"\n--- Top Categories by Placement ---", f)

    for p in placements:
        log(f"\nPlacement {p}:", f)
        subset = winners_with_cat[winners_with_cat['PLACEMENT'] == p]

        all_cats = []
        for cats in subset['CATEGORIES'].dropna():
            if isinstance(cats, (list, np.ndarray)):
                all_cats.extend(cats)

        if all_cats:
            cat_counts = Counter(all_cats)
            top_cats = cat_counts.most_common(10)

            log(f"  {'Category (truncated)':<40} {'Count':>10}", f)
            log(f"  {'-'*40} {'-'*10}", f)

            for cat, count in top_cats:
                log(f"  {str(cat)[:40]:<40} {count:>10,}", f)
        else:
            log(f"  No categories found", f)


def within_auction_diversity(ar_with_placement, catalog, f):
    """Analyze product diversity within auctions by placement."""
    log(f"\n{'='*80}", f)
    log(f"WITHIN-AUCTION PRODUCT DIVERSITY", f)
    log(f"{'='*80}", f)

    if ar_with_placement is None or catalog is None:
        log("Required data not available", f)
        return

    if 'PLACEMENT' not in ar_with_placement.columns:
        log("PLACEMENT column not available", f)
        return

    if 'CATEGORIES' not in catalog.columns:
        log("CATEGORIES column not found in catalog", f)
        return

    winners = ar_with_placement[ar_with_placement['IS_WINNER'] == True].copy()
    winners_with_cat = winners.merge(catalog[['PRODUCT_ID', 'CATEGORIES']], on='PRODUCT_ID', how='left')

    placements = sorted(ar_with_placement['PLACEMENT'].dropna().unique())

    log("Analyzing category homogeneity within auctions (do similar items appear together?)...", f)

    for p in placements:
        log(f"\n--- Placement {p} ---", f)

        subset = winners_with_cat[winners_with_cat['PLACEMENT'] == p]

        sample_auctions = subset['AUCTION_ID'].drop_duplicates().sample(min(500, subset['AUCTION_ID'].nunique()), random_state=42)
        sample = subset[subset['AUCTION_ID'].isin(sample_auctions)]

        overlaps = []
        for auction_id, group in tqdm(sample.groupby('AUCTION_ID'), desc=f"P{p}", leave=False):
            categories_list = group['CATEGORIES'].dropna().tolist()
            if len(categories_list) < 2:
                continue

            all_cats = []
            for cats in categories_list:
                if isinstance(cats, (list, np.ndarray)):
                    all_cats.extend(cats)

            if not all_cats:
                continue

            cat_counts = Counter(all_cats)
            if cat_counts:
                most_common_freq = cat_counts.most_common(1)[0][1]
                overlap_rate = most_common_freq / len(categories_list)
                overlaps.append(overlap_rate)

        if overlaps:
            log(f"Auctions analyzed: {len(overlaps):,}", f)
            log(f"Mean category overlap: {np.mean(overlaps):.3f}", f)
            log(f"Median category overlap: {np.median(overlaps):.3f}", f)
        else:
            log(f"Could not compute category overlap", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Placement product characteristics EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"07_placement_product_characteristics_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("PLACEMENT PRODUCT CHARACTERISTICS", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)

        paths = get_data_paths(args.round)

        log(f"\n--- Loading Data ---", f)
        ar = pd.read_parquet(paths['auctions_results'])
        log(f"Auctions results: {len(ar):,} rows", f)

        au = pd.read_parquet(paths['auctions_users'])
        log(f"Auctions users: {len(au):,} rows", f)

        catalog = None
        if paths['catalog'].exists():
            catalog = pd.read_parquet(paths['catalog'])
            log(f"Catalog: {len(catalog):,} rows", f)

        ar_with_placement = ar.merge(au[['AUCTION_ID', 'PLACEMENT']], on='AUCTION_ID', how='left')

        product_diversity_by_placement(ar_with_placement, f)
        product_concentration_by_placement(ar_with_placement, f)
        price_distribution_by_placement(ar_with_placement, catalog, f)
        category_analysis_by_placement(ar_with_placement, catalog, f)
        within_auction_diversity(ar_with_placement, catalog, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
