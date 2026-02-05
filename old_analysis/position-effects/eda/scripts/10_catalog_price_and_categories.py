#!/usr/bin/env python3
"""
Catalog Price and Categories

Price distributions, category coverage analysis.
Documents product catalog characteristics.

Usage:
    python 10_catalog_price_and_categories.py --round round1
    python 10_catalog_price_and_categories.py --round round2
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
            'catalog': DATA_BASE / "round1/catalog_all.parquet",
        }
    elif round_name == "round2":
        return {
            'auctions_results': DATA_BASE / "round2/auctions_results_r2.parquet",
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
def catalog_overview(catalog, f):
    """Basic catalog statistics."""
    log(f"\n{'='*80}", f)
    log(f"CATALOG OVERVIEW", f)
    log(f"{'='*80}", f)

    log(f"\nTotal products in catalog: {len(catalog):,}", f)
    log(f"Columns: {list(catalog.columns)}", f)

    log(f"\n--- Field Coverage ---", f)
    log(f"{'Field':<25} {'Non-null':>15} {'%':>10}", f)
    log(f"{'-'*25} {'-'*15} {'-'*10}", f)

    for col in catalog.columns:
        non_null = catalog[col].notna().sum()
        pct = non_null / len(catalog) * 100
        log(f"{col:<25} {non_null:>15,} {pct:>9.1f}%", f)


def price_distribution(catalog, f):
    """Analyze price distribution."""
    log(f"\n{'='*80}", f)
    log(f"PRICE DISTRIBUTION", f)
    log(f"{'='*80}", f)

    price_col = None
    for col in ['CATALOG_PRICE', 'PRICE']:
        if col in catalog.columns:
            price_col = col
            break

    if price_col is None:
        log("Price column not found in catalog", f)
        return

    prices = catalog[price_col].dropna()
    log(f"\nProducts with price: {len(prices):,} ({len(prices)/len(catalog)*100:.1f}%)", f)

    log(f"\n--- Price Statistics ---", f)
    log(f"Mean: ${prices.mean():.2f}", f)
    log(f"Median: ${prices.median():.2f}", f)
    log(f"Std: ${prices.std():.2f}", f)
    log(f"Min: ${prices.min():.2f}", f)
    log(f"Max: ${prices.max():.2f}", f)

    log(f"\n--- Percentiles ---", f)
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = prices.quantile(pct/100)
        log(f"P{pct}: ${val:.2f}", f)

    log(f"\n--- Price Range Distribution ---", f)
    log(f"{'Range':>20} {'Count':>15} {'%':>10}", f)
    log(f"{'-'*20} {'-'*15} {'-'*10}", f)

    price_bins = [
        (0, 10, "$0-10"),
        (10, 20, "$10-20"),
        (20, 50, "$20-50"),
        (50, 100, "$50-100"),
        (100, 200, "$100-200"),
        (200, 500, "$200-500"),
        (500, 1000, "$500-1000"),
        (1000, float('inf'), "$1000+"),
    ]

    for low, high, label in price_bins:
        count = ((prices >= low) & (prices < high)).sum()
        pct = count / len(prices) * 100
        log(f"{label:>20} {count:>15,} {pct:>9.1f}%", f)


def category_analysis(catalog, f):
    """Analyze category coverage and distribution."""
    log(f"\n{'='*80}", f)
    log(f"CATEGORY ANALYSIS", f)
    log(f"{'='*80}", f)

    if 'CATEGORIES' not in catalog.columns:
        log("CATEGORIES column not found", f)
        return

    has_categories = catalog['CATEGORIES'].apply(
        lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False
    )
    log(f"\nProducts with categories: {has_categories.sum():,} ({has_categories.sum()/len(catalog)*100:.1f}%)", f)

    categories_per_product = catalog['CATEGORIES'].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )
    log(f"\n--- Categories per Product ---", f)
    log(f"Mean: {categories_per_product.mean():.2f}", f)
    log(f"Median: {categories_per_product.median():.0f}", f)
    log(f"Max: {categories_per_product.max()}", f)

    log(f"\n--- Categories per Product Distribution ---", f)
    log(f"{'N Categories':>15} {'Products':>15} {'%':>10}", f)
    log(f"{'-'*15} {'-'*15} {'-'*10}", f)

    cat_dist = categories_per_product.value_counts().sort_index()
    for n_cats in sorted(cat_dist.index[:15]):
        count = cat_dist[n_cats]
        pct = count / len(catalog) * 100
        log(f"{n_cats:>15} {count:>15,} {pct:>9.1f}%", f)

    all_categories = []
    for cats in catalog['CATEGORIES'].dropna():
        if isinstance(cats, (list, np.ndarray)):
            all_categories.extend(cats)

    if all_categories:
        log(f"\n--- Top 30 Categories ---", f)
        cat_counts = Counter(all_categories)
        total_cats = sum(cat_counts.values())

        log(f"\nTotal unique categories: {len(cat_counts):,}", f)
        log(f"Total category assignments: {total_cats:,}", f)

        log(f"\n{'Category (truncated)':<50} {'Count':>12} {'%':>10}", f)
        log(f"{'-'*50} {'-'*12} {'-'*10}", f)

        for cat, count in cat_counts.most_common(30):
            pct = count / total_cats * 100
            log(f"{str(cat)[:50]:<50} {count:>12,} {pct:>9.2f}%", f)


def name_and_description_coverage(catalog, f):
    """Analyze name and description field coverage."""
    log(f"\n{'='*80}", f)
    log(f"NAME AND DESCRIPTION COVERAGE", f)
    log(f"{'='*80}", f)

    if 'NAME' in catalog.columns:
        has_name = catalog['NAME'].notna() & (catalog['NAME'] != '')
        log(f"\nProducts with NAME: {has_name.sum():,} ({has_name.sum()/len(catalog)*100:.1f}%)", f)

        name_lengths = catalog['NAME'].dropna().apply(len)
        log(f"Mean name length: {name_lengths.mean():.1f} characters", f)
        log(f"Median name length: {name_lengths.median():.0f} characters", f)

    if 'DESCRIPTION' in catalog.columns:
        has_desc = catalog['DESCRIPTION'].notna() & (catalog['DESCRIPTION'] != '')
        log(f"\nProducts with DESCRIPTION: {has_desc.sum():,} ({has_desc.sum()/len(catalog)*100:.1f}%)", f)

        desc_lengths = catalog[catalog['DESCRIPTION'].notna()]['DESCRIPTION'].apply(len)
        if len(desc_lengths) > 0:
            log(f"Mean description length: {desc_lengths.mean():.1f} characters", f)
            log(f"Median description length: {desc_lengths.median():.0f} characters", f)


def catalog_vs_auction_coverage(catalog, ar, f):
    """Compare catalog coverage with auction products."""
    log(f"\n{'='*80}", f)
    log(f"CATALOG vs AUCTION COVERAGE", f)
    log(f"{'='*80}", f)

    auction_products = set(ar['PRODUCT_ID'].unique())
    catalog_products = set(catalog['PRODUCT_ID'].unique())

    in_both = auction_products & catalog_products
    auction_only = auction_products - catalog_products
    catalog_only = catalog_products - auction_products

    log(f"\nProducts in auctions: {len(auction_products):,}", f)
    log(f"Products in catalog: {len(catalog_products):,}", f)
    log(f"\nIn both: {len(in_both):,} ({len(in_both)/len(auction_products)*100:.1f}% of auction products)", f)
    log(f"In auctions only: {len(auction_only):,} ({len(auction_only)/len(auction_products)*100:.1f}%)", f)
    log(f"In catalog only: {len(catalog_only):,}", f)

    price_col = None
    for col in ['CATALOG_PRICE', 'PRICE']:
        if col in catalog.columns:
            price_col = col
            break

    if price_col:
        log(f"\n--- Price Coverage for Auction Products ---", f)

        auction_catalog = catalog[catalog['PRODUCT_ID'].isin(auction_products)]
        has_price = auction_catalog[price_col].notna().sum()
        log(f"Auction products with price: {has_price:,} ({has_price/len(auction_products)*100:.1f}%)", f)


def winner_price_analysis(catalog, ar, f):
    """Analyze prices for winning bids."""
    log(f"\n{'='*80}", f)
    log(f"WINNER PRICE ANALYSIS", f)
    log(f"{'='*80}", f)

    price_col = None
    for col in ['CATALOG_PRICE', 'PRICE']:
        if col in catalog.columns:
            price_col = col
            break

    if price_col is None:
        log("Price column not found", f)
        return

    winners = ar[ar['IS_WINNER'] == True].copy()
    winners_with_price = winners.merge(catalog[['PRODUCT_ID', price_col]], on='PRODUCT_ID', how='left')

    log(f"\nTotal winners: {len(winners):,}", f)
    log(f"Winners with catalog price: {winners_with_price[price_col].notna().sum():,} ({winners_with_price[price_col].notna().sum()/len(winners)*100:.1f}%)", f)

    prices = winners_with_price[price_col].dropna()
    if len(prices) > 0:
        log(f"\n--- Winner Price Statistics ---", f)
        log(f"Mean: ${prices.mean():.2f}", f)
        log(f"Median: ${prices.median():.2f}", f)
        log(f"Std: ${prices.std():.2f}", f)

        log(f"\n--- Winner Price Distribution ---", f)
        log(f"{'Range':>20} {'Count':>15} {'%':>10}", f)
        log(f"{'-'*20} {'-'*15} {'-'*10}", f)

        price_bins = [
            (0, 10, "$0-10"),
            (10, 20, "$10-20"),
            (20, 50, "$20-50"),
            (50, 100, "$50-100"),
            (100, 200, "$100-200"),
            (200, 500, "$200-500"),
            (500, float('inf'), "$500+"),
        ]

        for low, high, label in price_bins:
            count = ((prices >= low) & (prices < high)).sum()
            pct = count / len(prices) * 100
            log(f"{label:>20} {count:>15,} {pct:>9.1f}%", f)


def vendors_analysis(catalog, f):
    """Analyze vendor coverage."""
    log(f"\n{'='*80}", f)
    log(f"VENDORS ANALYSIS", f)
    log(f"{'='*80}", f)

    if 'VENDORS' not in catalog.columns:
        log("VENDORS column not found", f)
        return

    has_vendors = catalog['VENDORS'].apply(
        lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False
    )
    log(f"\nProducts with vendors: {has_vendors.sum():,} ({has_vendors.sum()/len(catalog)*100:.1f}%)", f)

    vendors_per_product = catalog['VENDORS'].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )
    log(f"\n--- Vendors per Product ---", f)
    log(f"Mean: {vendors_per_product.mean():.2f}", f)
    log(f"Median: {vendors_per_product.median():.0f}", f)
    log(f"Max: {vendors_per_product.max()}", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Catalog price and categories EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"10_catalog_price_and_categories_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("CATALOG PRICE AND CATEGORIES", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)

        paths = get_data_paths(args.round)

        log(f"\n--- Loading Data ---", f)

        ar = None
        if paths['auctions_results'].exists():
            ar = pd.read_parquet(paths['auctions_results'])
            log(f"Auctions results: {len(ar):,} rows", f)

        catalog = None
        if paths['catalog'].exists():
            catalog = pd.read_parquet(paths['catalog'])
            log(f"Catalog: {len(catalog):,} rows", f)
        else:
            log("Catalog file not found", f)
            return

        catalog_overview(catalog, f)
        price_distribution(catalog, f)
        category_analysis(catalog, f)
        name_and_description_coverage(catalog, f)

        if ar is not None:
            catalog_vs_auction_coverage(catalog, ar, f)
            winner_price_analysis(catalog, ar, f)

        vendors_analysis(catalog, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
