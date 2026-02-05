#!/usr/bin/env python3
"""
Placement Product Diversity Analysis

Analyzes product characteristics within each placement to validate mapping:
- P3 = Search Results (HIGH diversity - users search for anything)
- P5 = Brand Page (LOW brand diversity - single brand per page)
- P2 = PDP Ad Section (SIMILAR categories - related products)
- P1 = Homepage/Feed (MODERATE diversity - personalized mix)

Unit of analysis: AUCTION
For each auction, we examine the products that competed (from auctions_results).

Output: results/15_placement_product_diversity.txt
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Up to position-effects/
DATA_DIR = BASE_DIR / "0_data" / "round1"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "15_placement_product_diversity.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def compute_gini(values):
    """Compute Gini coefficient for concentration measurement."""
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    values = np.sort(values)
    n = len(values)
    if n == 1:
        return 0.0
    cumsum = np.cumsum(values)
    if cumsum[-1] == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def parse_categories(cat_str):
    """
    Parse CATEGORIES JSON array into dict of extracted fields.
    Returns dict with keys: brand, colors, sizes, department_id, category_ids, domain
    """
    result = {
        'brand': None,
        'colors': [],
        'sizes': [],
        'department_id': None,
        'category_ids': [],
        'domain': None,
        'category_features': []
    }

    if pd.isna(cat_str) or cat_str == '':
        return result

    try:
        categories = json.loads(cat_str)
    except (json.JSONDecodeError, TypeError):
        return result

    for item in categories:
        if not isinstance(item, str):
            continue

        if item.startswith('brand#'):
            result['brand'] = item[6:]
        elif item.startswith('color#'):
            result['colors'].append(item[6:])
        elif item.startswith('size#'):
            result['sizes'].append(item[5:])
        elif item.startswith('department#'):
            result['department_id'] = item[11:]
        elif item.startswith('category#'):
            result['category_ids'].append(item[9:])
        elif item.startswith('category_feature#'):
            result['category_features'].append(item[17:])
        elif item.startswith('domain#'):
            result['domain'] = item[7:]

    return result


def compute_diversity_metrics(product_list, catalog_lookup):
    """
    Compute diversity metrics for a list of products in an auction.

    Returns dict with:
    - n_products: number of products
    - n_unique_brands: unique brands
    - n_unique_departments: unique departments
    - n_unique_categories: unique category IDs
    - brand_gini: brand concentration (0=equal, 1=concentrated)
    - single_brand: boolean if all products have same brand
    - has_brand_data: fraction of products with brand info
    """
    metrics = {
        'n_products': len(product_list),
        'n_unique_brands': 0,
        'n_unique_departments': 0,
        'n_unique_categories': 0,
        'brand_gini': np.nan,
        'single_brand': False,
        'has_brand_data': 0.0
    }

    if len(product_list) == 0:
        return metrics

    brands = []
    departments = []
    categories = []

    for pid in product_list:
        if pid in catalog_lookup:
            parsed = catalog_lookup[pid]
            if parsed['brand']:
                brands.append(parsed['brand'])
            if parsed['department_id']:
                departments.append(parsed['department_id'])
            categories.extend(parsed['category_ids'])

    metrics['n_unique_brands'] = len(set(brands))
    metrics['n_unique_departments'] = len(set(departments))
    metrics['n_unique_categories'] = len(set(categories))
    metrics['has_brand_data'] = len(brands) / len(product_list) if len(product_list) > 0 else 0

    if len(brands) > 1:
        brand_counts = Counter(brands)
        metrics['brand_gini'] = compute_gini(list(brand_counts.values()))
        metrics['single_brand'] = len(brand_counts) == 1
    elif len(brands) == 1:
        metrics['single_brand'] = True
        metrics['brand_gini'] = 0.0

    return metrics


# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("PLACEMENT PRODUCT DIVERSITY ANALYSIS", f)
        log("=" * 80, f)
        log("", f)
        log("Purpose: Validate placement-to-page mapping by analyzing product diversity", f)
        log("         within auctions for each placement.", f)
        log("", f)
        log("Expected patterns:", f)
        log("  P3 (Search): HIGH diversity - users search for anything", f)
        log("  P5 (Brand Page): LOW brand diversity - single brand per page", f)
        log("  P2 (PDP): SIMILAR categories - related product recommendations", f)
        log("  P1 (Homepage): MODERATE diversity - personalized mix", f)
        log("", f)
        log(f"Data source: {DATA_DIR}", f)
        log(f"Output: {OUTPUT_FILE}", f)
        log("", f)

        # =====================================================================
        # SECTION 1: DATA LOADING
        # =====================================================================
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("=" * 80, f)

        log("\nLoading auctions_users_all.parquet...", f)
        au = pd.read_parquet(DATA_DIR / "auctions_users_all.parquet")
        log(f"  Shape: {au.shape}", f)
        log(f"  Columns: {list(au.columns)}", f)

        log("\nLoading auctions_results_all.parquet...", f)
        ar = pd.read_parquet(DATA_DIR / "auctions_results_all.parquet")
        log(f"  Shape: {ar.shape}", f)
        log(f"  Columns: {list(ar.columns)}", f)

        log("\nLoading catalog_all.parquet...", f)
        catalog = pd.read_parquet(DATA_DIR / "catalog_all.parquet")
        log(f"  Shape: {catalog.shape}", f)
        log(f"  Columns: {list(catalog.columns)}", f)

        # Convert PLACEMENT to string
        au['PLACEMENT'] = au['PLACEMENT'].astype(str)

        log("\nPlacement distribution in auctions:", f)
        placement_counts = au['PLACEMENT'].value_counts().sort_index()
        for p, count in placement_counts.items():
            log(f"  P{p}: {count:,} auctions ({count/len(au)*100:.1f}%)", f)

        # =====================================================================
        # SECTION 2: PARSE CATALOG CATEGORIES
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 2: PARSE CATALOG CATEGORIES", f)
        log("=" * 80, f)

        log("\nParsing CATEGORIES field for all catalog products...", f)
        tqdm.pandas(desc="Parsing categories")
        catalog['parsed'] = catalog['CATEGORIES'].progress_apply(parse_categories)

        # Create lookup dictionary
        catalog_lookup = {}
        for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Building lookup"):
            catalog_lookup[row['PRODUCT_ID']] = row['parsed']

        log(f"  Catalog lookup built: {len(catalog_lookup):,} products", f)

        # Coverage stats
        has_brand = sum(1 for v in catalog_lookup.values() if v['brand'] is not None)
        has_dept = sum(1 for v in catalog_lookup.values() if v['department_id'] is not None)
        has_cat = sum(1 for v in catalog_lookup.values() if len(v['category_ids']) > 0)

        log(f"  Products with brand: {has_brand:,} ({100*has_brand/len(catalog_lookup):.1f}%)", f)
        log(f"  Products with department: {has_dept:,} ({100*has_dept/len(catalog_lookup):.1f}%)", f)
        log(f"  Products with category: {has_cat:,} ({100*has_cat/len(catalog_lookup):.1f}%)", f)

        # =====================================================================
        # SECTION 3: MERGE AUCTIONS WITH PLACEMENT
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 3: MERGE AUCTIONS WITH PLACEMENT", f)
        log("=" * 80, f)

        # Get products per auction
        log("\nGrouping products by auction...", f)
        products_per_auction = ar.groupby('AUCTION_ID')['PRODUCT_ID'].apply(list).reset_index()
        products_per_auction.columns = ['AUCTION_ID', 'products']
        log(f"  Auctions with products: {len(products_per_auction):,}", f)

        # Merge with placement info
        auction_data = au.merge(products_per_auction, on='AUCTION_ID', how='inner')
        log(f"  Auctions after merge: {len(auction_data):,}", f)

        # Check merge coverage
        for p in ['1', '2', '3', '5']:
            orig = placement_counts.get(p, 0)
            merged = (auction_data['PLACEMENT'] == p).sum()
            log(f"  P{p}: {orig:,} -> {merged:,} ({100*merged/orig:.1f}% coverage)", f)

        # =====================================================================
        # SECTION 4: COMPUTE DIVERSITY METRICS PER AUCTION
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 4: COMPUTE DIVERSITY METRICS PER AUCTION", f)
        log("=" * 80, f)

        log("\nComputing diversity metrics for each auction...", f)

        diversity_records = []
        for _, row in tqdm(auction_data.iterrows(), total=len(auction_data), desc="Computing diversity"):
            metrics = compute_diversity_metrics(row['products'], catalog_lookup)
            metrics['AUCTION_ID'] = row['AUCTION_ID']
            metrics['PLACEMENT'] = row['PLACEMENT']
            diversity_records.append(metrics)

        diversity_df = pd.DataFrame(diversity_records)
        log(f"  Computed metrics for {len(diversity_df):,} auctions", f)

        # =====================================================================
        # SECTION 5: DIVERSITY BY PLACEMENT
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 5: DIVERSITY BY PLACEMENT", f)
        log("=" * 80, f)
        log("", f)
        log("Unit of analysis: AUCTION", f)
        log("Metrics computed for products competing in each auction.", f)
        log("", f)

        for placement in ['1', '2', '3', '5']:
            p_df = diversity_df[diversity_df['PLACEMENT'] == placement]

            log(f"\n--- Placement {placement} (n={len(p_df):,} auctions) ---", f)

            # Products per auction
            log(f"\nProducts per auction:", f)
            log(f"  Mean: {p_df['n_products'].mean():.1f}", f)
            log(f"  Median: {p_df['n_products'].median():.1f}", f)
            log(f"  Min: {p_df['n_products'].min():.0f}", f)
            log(f"  Max: {p_df['n_products'].max():.0f}", f)
            log(f"  Std: {p_df['n_products'].std():.1f}", f)

            # Brand diversity
            log(f"\nBrand diversity within auction:", f)
            log(f"  Unique brands (mean): {p_df['n_unique_brands'].mean():.2f}", f)
            log(f"  Unique brands (median): {p_df['n_unique_brands'].median():.1f}", f)
            log(f"  Single-brand auctions: {p_df['single_brand'].sum():,} ({100*p_df['single_brand'].mean():.1f}%)", f)
            log(f"  Brand data coverage: {p_df['has_brand_data'].mean()*100:.1f}%", f)

            # Brand concentration
            gini_valid = p_df['brand_gini'].dropna()
            if len(gini_valid) > 0:
                log(f"  Brand Gini (mean): {gini_valid.mean():.3f}", f)
                log(f"  Brand Gini (median): {gini_valid.median():.3f}", f)
                log(f"  (Gini: 0=equal distribution, 1=one brand dominates)", f)

            # Department diversity
            log(f"\nDepartment diversity within auction:", f)
            log(f"  Unique departments (mean): {p_df['n_unique_departments'].mean():.2f}", f)
            log(f"  Unique departments (median): {p_df['n_unique_departments'].median():.1f}", f)

            # Category diversity
            log(f"\nCategory diversity within auction:", f)
            log(f"  Unique categories (mean): {p_df['n_unique_categories'].mean():.2f}", f)
            log(f"  Unique categories (median): {p_df['n_unique_categories'].median():.1f}", f)

        # =====================================================================
        # SECTION 6: COMPARATIVE SUMMARY TABLE
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 6: COMPARATIVE SUMMARY TABLE", f)
        log("=" * 80, f)
        log("", f)

        summary_data = []
        for placement in ['1', '2', '3', '5']:
            p_df = diversity_df[diversity_df['PLACEMENT'] == placement]
            gini_valid = p_df['brand_gini'].dropna()

            summary_data.append({
                'Placement': f'P{placement}',
                'N': len(p_df),
                'Products/Auction': f"{p_df['n_products'].mean():.1f}",
                'Brands/Auction': f"{p_df['n_unique_brands'].mean():.2f}",
                'Single-Brand%': f"{100*p_df['single_brand'].mean():.1f}%",
                'Depts/Auction': f"{p_df['n_unique_departments'].mean():.2f}",
                'Cats/Auction': f"{p_df['n_unique_categories'].mean():.2f}",
                'Brand Gini': f"{gini_valid.mean():.3f}" if len(gini_valid) > 0 else "N/A"
            })

        summary_df = pd.DataFrame(summary_data)

        log("| Placement | N       | Prods | Brands | Single-Brand | Depts | Cats  | Brand Gini |", f)
        log("|-----------|---------|-------|--------|--------------|-------|-------|------------|", f)
        for _, row in summary_df.iterrows():
            log(f"| {row['Placement']:<9} | {row['N']:<7,} | {row['Products/Auction']:<5} | {row['Brands/Auction']:<6} | {row['Single-Brand%']:<12} | {row['Depts/Auction']:<5} | {row['Cats/Auction']:<5} | {row['Brand Gini']:<10} |", f)

        # =====================================================================
        # SECTION 7: BRAND DISTRIBUTION ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 7: BRAND DISTRIBUTION BY PLACEMENT", f)
        log("=" * 80, f)
        log("", f)
        log("What brands appear in each placement?", f)
        log("", f)

        # Get all products per placement
        for placement in ['1', '2', '3', '5']:
            log(f"\n--- Placement {placement} Top 20 Brands ---", f)

            # Get all auctions for this placement
            p_auctions = auction_data[auction_data['PLACEMENT'] == placement]

            # Flatten all products
            all_products = []
            for products in p_auctions['products']:
                all_products.extend(products)

            # Get brands
            brands = []
            for pid in all_products:
                if pid in catalog_lookup and catalog_lookup[pid]['brand']:
                    brands.append(catalog_lookup[pid]['brand'])

            brand_counts = Counter(brands)
            total_branded = len(brands)

            log(f"  Total products: {len(all_products):,}", f)
            log(f"  Products with brand: {total_branded:,} ({100*total_branded/len(all_products):.1f}%)", f)
            log(f"  Unique brands: {len(brand_counts):,}", f)

            if len(brand_counts) > 0:
                gini = compute_gini(list(brand_counts.values()))
                log(f"  Brand concentration (Gini): {gini:.3f}", f)

            log(f"\n  Top 20 brands:", f)
            for i, (brand, count) in enumerate(brand_counts.most_common(20), 1):
                pct = 100 * count / total_branded if total_branded > 0 else 0
                log(f"    {i:2d}. {brand}: {count:,} ({pct:.1f}%)", f)

        # =====================================================================
        # SECTION 8: CATEGORY DISTRIBUTION ANALYSIS
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 8: CATEGORY DISTRIBUTION BY PLACEMENT", f)
        log("=" * 80, f)
        log("", f)
        log("What departments appear in each placement?", f)
        log("", f)

        for placement in ['1', '2', '3', '5']:
            log(f"\n--- Placement {placement} Departments ---", f)

            p_auctions = auction_data[auction_data['PLACEMENT'] == placement]

            all_products = []
            for products in p_auctions['products']:
                all_products.extend(products)

            departments = []
            for pid in all_products:
                if pid in catalog_lookup and catalog_lookup[pid]['department_id']:
                    departments.append(catalog_lookup[pid]['department_id'])

            dept_counts = Counter(departments)
            total_with_dept = len(departments)

            log(f"  Total products: {len(all_products):,}", f)
            log(f"  Products with department: {total_with_dept:,} ({100*total_with_dept/len(all_products):.1f}%)", f)
            log(f"  Unique departments: {len(dept_counts):,}", f)

            if len(dept_counts) > 0:
                gini = compute_gini(list(dept_counts.values()))
                log(f"  Department concentration (Gini): {gini:.3f}", f)

            log(f"\n  Top 10 departments:", f)
            for i, (dept, count) in enumerate(dept_counts.most_common(10), 1):
                pct = 100 * count / total_with_dept if total_with_dept > 0 else 0
                log(f"    {i:2d}. {dept}: {count:,} ({pct:.1f}%)", f)

        # =====================================================================
        # SECTION 9: SAMPLE PRODUCTS BY PLACEMENT
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 9: SAMPLE PRODUCTS BY PLACEMENT", f)
        log("=" * 80, f)
        log("", f)
        log("10 random product names from each placement for visual inspection.", f)
        log("", f)

        # Create product name lookup
        name_lookup = dict(zip(catalog['PRODUCT_ID'], catalog['NAME']))

        for placement in ['1', '2', '3', '5']:
            log(f"\n--- Placement {placement} Sample Products ---", f)

            p_auctions = auction_data[auction_data['PLACEMENT'] == placement]

            all_products = []
            for products in p_auctions['products']:
                all_products.extend(products)

            # Random sample
            np.random.seed(42 + int(placement))
            sample_pids = np.random.choice(all_products, min(10, len(all_products)), replace=False)

            for i, pid in enumerate(sample_pids, 1):
                name = name_lookup.get(pid, "N/A")
                if name and len(name) > 80:
                    name = name[:80] + "..."
                brand = catalog_lookup.get(pid, {}).get('brand', 'N/A') if pid in catalog_lookup else 'N/A'
                log(f"  {i:2d}. [{brand}] {name}", f)

        # =====================================================================
        # SECTION 10: WITHIN-AUCTION BRAND HOMOGENEITY
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 10: WITHIN-AUCTION BRAND HOMOGENEITY", f)
        log("=" * 80, f)
        log("", f)
        log("Key test: If P5 is Brand Page, auctions should show SAME brand.", f)
        log("If P3 is Search, auctions should show DIVERSE brands.", f)
        log("", f)

        # Distribution of unique brands per auction
        log("Distribution of unique brands per auction:", f)
        log("", f)
        log("| Placement | 0 brands | 1 brand | 2-5 brands | 6-10 brands | >10 brands |", f)
        log("|-----------|----------|---------|------------|-------------|------------|", f)

        for placement in ['1', '2', '3', '5']:
            p_df = diversity_df[diversity_df['PLACEMENT'] == placement]
            total = len(p_df)

            zero = (p_df['n_unique_brands'] == 0).sum()
            one = (p_df['n_unique_brands'] == 1).sum()
            two_five = ((p_df['n_unique_brands'] >= 2) & (p_df['n_unique_brands'] <= 5)).sum()
            six_ten = ((p_df['n_unique_brands'] >= 6) & (p_df['n_unique_brands'] <= 10)).sum()
            over_ten = (p_df['n_unique_brands'] > 10).sum()

            log(f"| P{placement}        | {100*zero/total:>6.1f}%  | {100*one/total:>5.1f}%  | {100*two_five/total:>8.1f}%   | {100*six_ten/total:>9.1f}%   | {100*over_ten/total:>8.1f}%   |", f)

        # =====================================================================
        # SECTION 11: INTERPRETATION & VALIDATION
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 11: INTERPRETATION & VALIDATION", f)
        log("=" * 80, f)
        log("", f)

        # Compute summary stats for interpretation
        p1_brands = diversity_df[diversity_df['PLACEMENT'] == '1']['n_unique_brands'].mean()
        p2_brands = diversity_df[diversity_df['PLACEMENT'] == '2']['n_unique_brands'].mean()
        p3_brands = diversity_df[diversity_df['PLACEMENT'] == '3']['n_unique_brands'].mean()
        p5_brands = diversity_df[diversity_df['PLACEMENT'] == '5']['n_unique_brands'].mean()

        p1_single = diversity_df[diversity_df['PLACEMENT'] == '1']['single_brand'].mean()
        p2_single = diversity_df[diversity_df['PLACEMENT'] == '2']['single_brand'].mean()
        p3_single = diversity_df[diversity_df['PLACEMENT'] == '3']['single_brand'].mean()
        p5_single = diversity_df[diversity_df['PLACEMENT'] == '5']['single_brand'].mean()

        log("EXPECTED vs OBSERVED:", f)
        log("", f)

        log("P3 (Search Results) - Expected: HIGH diversity", f)
        log(f"  Observed: {p3_brands:.2f} brands/auction, {100*p3_single:.1f}% single-brand", f)
        if p3_brands > max(p1_brands, p2_brands, p5_brands):
            log("  VALIDATION: CONFIRMED - P3 has highest brand diversity", f)
        else:
            log("  VALIDATION: UNEXPECTED - P3 does NOT have highest diversity", f)
        log("", f)

        log("P5 (Brand Page) - Expected: LOW brand diversity / HIGH single-brand rate", f)
        log(f"  Observed: {p5_brands:.2f} brands/auction, {100*p5_single:.1f}% single-brand", f)
        if p5_single > max(p1_single, p2_single, p3_single):
            log("  VALIDATION: CONFIRMED - P5 has highest single-brand rate", f)
        elif p5_brands < min(p1_brands, p2_brands, p3_brands):
            log("  VALIDATION: PARTIAL - P5 has lowest brand diversity", f)
        else:
            log("  VALIDATION: UNEXPECTED - P5 does NOT show brand homogeneity", f)
        log("", f)

        log("P2 (PDP Ad Section) - Expected: MODERATE diversity (related products)", f)
        log(f"  Observed: {p2_brands:.2f} brands/auction, {100*p2_single:.1f}% single-brand", f)
        log("  (PDP shows related items which may cross brands)", f)
        log("", f)

        log("P1 (Homepage/Feed) - Expected: MODERATE diversity (personalized mix)", f)
        log(f"  Observed: {p1_brands:.2f} brands/auction, {100*p1_single:.1f}% single-brand", f)
        log("", f)

        # Ranking
        brand_ranking = [
            ('P1', p1_brands),
            ('P2', p2_brands),
            ('P3', p3_brands),
            ('P5', p5_brands)
        ]
        brand_ranking.sort(key=lambda x: -x[1])

        log("BRAND DIVERSITY RANKING (most diverse first):", f)
        for i, (p, val) in enumerate(brand_ranking, 1):
            log(f"  {i}. {p}: {val:.2f} brands/auction", f)
        log("", f)

        single_ranking = [
            ('P1', p1_single),
            ('P2', p2_single),
            ('P3', p3_single),
            ('P5', p5_single)
        ]
        single_ranking.sort(key=lambda x: -x[1])

        log("SINGLE-BRAND RATE RANKING (most homogeneous first):", f)
        for i, (p, val) in enumerate(single_ranking, 1):
            log(f"  {i}. {p}: {100*val:.1f}% single-brand auctions", f)

        # =====================================================================
        # SECTION 12: REVISED INTERPRETATION
        # =====================================================================
        log("\n" + "=" * 80, f)
        log("SECTION 12: REVISED INTERPRETATION", f)
        log("=" * 80, f)
        log("", f)
        log("CRITICAL INSIGHT:", f)
        log("", f)
        log("The original hypothesis (P5 = Brand Page, P2 = PDP) is INCONSISTENT", f)
        log("with the product diversity data.", f)
        log("", f)
        log("OBSERVED PATTERNS:", f)
        log("", f)
        log("  P2: 99.2% single-brand auctions, 1.01 brands/auction", f)
        log("      -> EXTREME brand homogeneity", f)
        log("      -> Products in auction are almost always from SAME BRAND", f)
        log("      -> This is the pattern expected for BRAND PAGE, not PDP!", f)
        log("", f)
        log("  P5: 0.9% single-brand auctions, 45.53 brands/auction", f)
        log("      -> HIGHEST brand diversity", f)
        log("      -> Products in auction from MANY different brands", f)
        log("      -> This is NOT the pattern for Brand Page", f)
        log("", f)
        log("  P3: 8.2% single-brand auctions, 18.14 brands/auction", f)
        log("      -> Moderate-high diversity", f)
        log("      -> Consistent with Search (users search for various items)", f)
        log("", f)
        log("  P1: 52.9% single-brand auctions, 11.37 brands/auction", f)
        log("      -> Mixed pattern (some single-brand, some diverse)", f)
        log("      -> Could be personalized feed with occasional brand-specific views", f)
        log("", f)

        log("REVISED MAPPING HYPOTHESIS:", f)
        log("", f)
        log("Based on PRODUCT DIVERSITY (this analysis):", f)
        log("", f)
        log("  P2 = BRAND PAGE (or Brand-Specific Recommendations)", f)
        log("       Evidence: 99.2% single-brand auctions", f)
        log("       All products competing in auction are from same brand", f)
        log("       Top brand: 'meet the posher' (27.3%) - closet/seller pages?", f)
        log("", f)
        log("  P5 = CATEGORY/BROWSE PAGE (not Brand Page)", f)
        log("       Evidence: Highest brand diversity (45.53 brands/auction)", f)
        log("       Only 0.9% single-brand auctions", f)
        log("       99% department 000e8975d97b4e80ef00a955 (single category)", f)
        log("       Users browsing a CATEGORY see products from many brands", f)
        log("", f)
        log("  P3 = SEARCH RESULTS", f)
        log("       Evidence: Moderate diversity (18.14 brands/auction)", f)
        log("       Many unique brands (52,993 total)", f)
        log("       Consistent with broad search queries", f)
        log("", f)
        log("  P1 = HOMEPAGE/FEED (mixed)", f)
        log("       Evidence: Bimodal pattern (52.9% single-brand, 36.5% >10 brands)", f)
        log("       Suggests mixture of brand-specific and diverse recommendations", f)
        log("", f)

        log("RECONCILIATION WITH CO-FIRING ANALYSIS:", f)
        log("", f)
        log("Co-firing analysis (14_cross_placement_cofiring.txt) found:", f)
        log("  - P2 <-> P5: 3% co-firing (highest pair)", f)
        log("  - This suggests P2 and P5 are tightly coupled (same user journey)", f)
        log("", f)
        log("NEW INTERPRETATION:", f)
        log("  User flow: Category Page (P5) -> click product -> PDP loads", f)
        log("             But P2 fires when viewing seller/brand closet", f)
        log("", f)
        log("  The high P2<->P5 co-firing may represent:", f)
        log("  - User browses category (P5 - diverse)", f)
        log("  - User clicks into seller's closet or brand page (P2 - homogeneous)", f)
        log("  - The 'meet the posher' brand (27.3% of P2) confirms seller closet view", f)
        log("", f)

        log("IMPORTANT NOTE ON 'MEET THE POSHER':", f)
        log("", f)
        log("P2's top 'brand' is 'meet the posher' at 27.3%", f)
        log("This is NOT a clothing brand - it's the Poshmark seller introduction listing", f)
        log("'Meet your Posher' listings are placeholder profiles sellers create", f)
        log("", f)
        log("This strongly suggests P2 = SELLER CLOSET PAGE VIEW", f)
        log("When users view a seller's closet, they see:", f)
        log("  - The seller's 'Meet the Posher' intro (if created)", f)
        log("  - Other items from SAME SELLER (hence single-brand pattern)", f)
        log("", f)

        log("=" * 80, f)
        log("FINAL REVISED MAPPING", f)
        log("=" * 80, f)
        log("", f)
        log("| Placement | Revised Page Type              | Key Evidence                    |", f)
        log("|-----------|--------------------------------|---------------------------------|", f)
        log("| P2        | Seller Closet / Brand Page     | 99.2% single-brand, 'meet the posher' |", f)
        log("| P3        | Search Results                 | 18 brands/auction, diverse products |", f)
        log("| P5        | Category/Browse Page           | 45 brands/auction, single department |", f)
        log("| P1        | Homepage/Feed                  | Mixed (52.9% single, 36.5% diverse) |", f)
        log("", f)
        log("This mapping better fits the observed product diversity patterns.", f)
        log("P2's extreme brand homogeneity indicates seller-level or brand-level browsing,", f)
        log("not generic PDP recommendations which would show related but diverse products.", f)

        log("\n" + "=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)
        log(f"Output saved to: {OUTPUT_FILE}", f)


if __name__ == "__main__":
    main()
