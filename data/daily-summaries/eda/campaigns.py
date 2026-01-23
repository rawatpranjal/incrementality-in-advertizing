#!/usr/bin/env python3
"""
Campaign Structure EDA for Academic Paper

Section: "The Structure of Marketplace Advertising"
Purpose: Foundational descriptive analysis of advertising ecosystem
Period: May 2025 (31 contiguous days) for comprehensive campaign lifecycle analysis
Unit of Analysis: Campaign-level, Vendor-level, Product-level aggregations

Research Questions:
1. How concentrated is the advertising market? (Vendor concentration)
2. What are the primary segments of vendor behavior? (Vendor typology)
3. What does a typical campaign look like? (Campaign archetypes)
4. What is the typical lifecycle of a campaign? (Temporal dynamics)
5. Is advertising focused on popular products? (Product concentration)
6. How often are products featured across campaigns? (Product reuse patterns)

Statistical Methods: Gini coefficients, concentration ratios, distribution analysis
Expected Outputs: Academic-quality tables for paper inclusion
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def calculate_gini_coefficient(x):
    """Calculate Gini coefficient for concentration measurement."""
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def print_concentration_analysis(series, name, top_percentiles=[1, 5, 20]):
    """Print concentration analysis with Gini coefficient and top percentiles."""
    print(f"\n{name.upper()} CONCENTRATION ANALYSIS:")

    # Basic stats
    print(f"Total count: {len(series):,}")
    print(f"Total sum: {series.sum():,.0f}")
    print(f"Mean: {series.mean():.2f}")
    print(f"Median: {series.median():.2f}")
    print(f"Std: {series.std():.2f}")

    # Gini coefficient
    gini = calculate_gini_coefficient(series)
    print(f"Gini coefficient: {gini:.3f}")

    # Concentration ratios
    sorted_series = series.sort_values(ascending=False)
    total = sorted_series.sum()

    print(f"\nCONCENTRATION RATIOS:")
    for pct in top_percentiles:
        n_top = max(1, int(len(sorted_series) * pct / 100))
        top_share = sorted_series.head(n_top).sum() / total * 100
        print(f"Top {pct}% ({n_top:,} entities): {top_share:.1f}% of total {name.lower()}")

    # Distribution
    print(f"\nDISTRIBUTION PERCENTILES:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = series.quantile(p/100)
        print(f"P{p:02d}: {val:.2f}")

def load_may_2025_sample():
    """Load comprehensive May 2025 sample across all datasets."""
    print("=== 1. LOADING MAY 2025 COMPREHENSIVE SAMPLE ===")

    BASE_PATH = Path('../data')

    # Generate May 2025 date range
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 5, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    date_strings = [d.strftime('%Y-%m-%d') for d in date_range]

    print(f"Loading data for {len(date_strings)} days: {date_strings[0]} to {date_strings[-1]}")

    datasets = {}

    # Load impressions data (core campaign data)
    print("\nLoading impressions data...")
    impressions_data = []
    impressions_dir = BASE_PATH / 'product_daily_impressions_dataset'

    for date_str in tqdm(date_strings, desc="Loading impressions"):
        file_path = impressions_dir / f'data_{date_str}.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['DATE'] = pd.to_datetime(date_str)
            impressions_data.append(df)

    if impressions_data:
        datasets['impressions'] = pd.concat(impressions_data, ignore_index=True)
        print(f"Impressions: {len(datasets['impressions']):,} records")

    # Load clicks data
    print("\nLoading clicks data...")
    clicks_data = []
    clicks_dir = BASE_PATH / 'product_daily_clicks_dataset'

    for date_str in tqdm(date_strings, desc="Loading clicks"):
        file_path = clicks_dir / f'data_{date_str}.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['DATE'] = pd.to_datetime(date_str)
            clicks_data.append(df)

    if clicks_data:
        datasets['clicks'] = pd.concat(clicks_data, ignore_index=True)
        print(f"Clicks: {len(datasets['clicks']):,} records")

    # Load purchases data
    print("\nLoading purchases data...")
    purchases_data = []
    purchases_dir = BASE_PATH / 'product_daily_purchases_dataset'

    for date_str in tqdm(date_strings, desc="Loading purchases"):
        file_path = purchases_dir / f'data_{date_str}.parquet'
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['DATE'] = pd.to_datetime(date_str)
            purchases_data.append(df)

    if purchases_data:
        datasets['purchases'] = pd.concat(purchases_data, ignore_index=True)
        print(f"Purchases: {len(datasets['purchases']):,} records")

    # Load product catalog (use middle of month snapshot)
    print("\nLoading product catalog...")
    catalog_dir = BASE_PATH / 'product_catalog_processed'
    catalog_file = catalog_dir / 'catalog_processed_2025-05-15.parquet'

    if catalog_file.exists():
        datasets['catalog'] = pd.read_parquet(catalog_file)
        print(f"Catalog: {len(datasets['catalog']):,} products")

    return datasets

def create_campaign_summary(datasets):
    """Create comprehensive campaign summary table."""
    print("\n=== 2. CREATING CAMPAIGN SUMMARY TABLE ===")

    if 'impressions' not in datasets:
        print("No impressions data available")
        return None

    df_imp = datasets['impressions']
    df_clicks = datasets.get('clicks', pd.DataFrame())

    # Campaign-level aggregation from impressions
    print("Aggregating campaign metrics...")
    campaign_summary = df_imp.groupby('CAMPAIGN_ID').agg({
        'VENDOR_ID': 'first',
        'PRODUCT_ID': 'nunique',
        'DATE': ['min', 'max', 'nunique'],
        'IMPRESSIONS': 'sum',
        'DISTINCT_USERS_IMPRESSED': 'sum'
    }).reset_index()

    campaign_summary.columns = ['CAMPAIGN_ID', 'VENDOR_ID', 'unique_products', 'start_date', 'end_date', 'active_days', 'total_impressions', 'total_users_impressed']
    campaign_summary['duration_days'] = (campaign_summary['end_date'] - campaign_summary['start_date']).dt.days + 1

    # Add clicks data if available
    if not df_clicks.empty:
        clicks_summary = df_clicks.groupby('CAMPAIGN_ID').agg({
            'CLICKS': 'sum',
            'DISTINCT_USERS_CLICKED': 'sum'
        }).reset_index()

        campaign_summary = campaign_summary.merge(clicks_summary, on='CAMPAIGN_ID', how='left')
        campaign_summary['CLICKS'] = campaign_summary['CLICKS'].fillna(0)
        campaign_summary['DISTINCT_USERS_CLICKED'] = campaign_summary['DISTINCT_USERS_CLICKED'].fillna(0)
        campaign_summary['ctr'] = campaign_summary['CLICKS'] / campaign_summary['total_impressions']
        campaign_summary['ctr'] = campaign_summary['ctr'].fillna(0)

    # Campaign archetypes
    campaign_summary['campaign_type'] = pd.cut(
        campaign_summary['unique_products'],
        bins=[0, 1, 10, float('inf')],
        labels=['Hero', 'Focused', 'Catalog'],
        right=True
    )

    # Duration types
    campaign_summary['duration_type'] = pd.cut(
        campaign_summary['duration_days'],
        bins=[0, 7, 30, float('inf')],
        labels=['Short', 'Medium', 'Long'],
        right=True
    )

    print(f"Campaign summary created: {len(campaign_summary):,} unique campaigns")
    return campaign_summary

def analyze_vendor_concentration(campaign_summary):
    """Analyze vendor-level market concentration."""
    print("\n=== 3. VENDOR CONCENTRATION ANALYSIS ===")

    # Vendor-level aggregation
    vendor_stats = campaign_summary.groupby('VENDOR_ID').agg({
        'CAMPAIGN_ID': 'count',
        'unique_products': 'sum',
        'total_impressions': 'sum',
        'CLICKS': 'sum' if 'CLICKS' in campaign_summary.columns else lambda x: 0,
        'duration_days': 'mean',
        'start_date': 'min',
        'end_date': 'max'
    }).reset_index()

    vendor_stats.columns = ['VENDOR_ID', 'total_campaigns', 'total_products', 'total_impressions', 'total_clicks', 'avg_duration', 'first_campaign', 'last_campaign']
    vendor_stats['vendor_active_days'] = (vendor_stats['last_campaign'] - vendor_stats['first_campaign']).dt.days + 1

    # Vendor segments
    vendor_stats['vendor_segment'] = pd.cut(
        vendor_stats['total_campaigns'],
        bins=[0, 1, 5, float('inf')],
        labels=['Small', 'Medium', 'Large'],
        right=True
    )

    print("VENDOR CAMPAIGN COUNT CONCENTRATION:")
    print_concentration_analysis(vendor_stats['total_campaigns'], 'campaigns')

    print("\nVENDOR IMPRESSION CONCENTRATION:")
    print_concentration_analysis(vendor_stats['total_impressions'], 'impressions')

    print("\nVENDOR PRODUCT PORTFOLIO CONCENTRATION:")
    print_concentration_analysis(vendor_stats['total_products'], 'products')

    # Vendor segments analysis
    print(f"\n=== VENDOR SEGMENTS ===")
    segment_analysis = vendor_stats.groupby('vendor_segment').agg({
        'VENDOR_ID': 'count',
        'total_campaigns': ['median', 'mean'],
        'total_impressions': ['median', 'mean'],
        'total_products': ['median', 'mean']
    }).round(2)

    print("Vendor segment summary:")
    for segment in ['Small', 'Medium', 'Large']:
        if segment in segment_analysis.index:
            count = segment_analysis.loc[segment, ('VENDOR_ID', 'count')]
            pct = count / len(vendor_stats) * 100
            med_campaigns = segment_analysis.loc[segment, ('total_campaigns', 'median')]
            med_impressions = segment_analysis.loc[segment, ('total_impressions', 'median')]
            print(f"{segment} vendors: {count:,} ({pct:.1f}%) - Median: {med_campaigns} campaigns, {med_impressions:,.0f} impressions")

    return vendor_stats

def analyze_campaign_archetypes(campaign_summary):
    """Analyze campaign archetypes and performance patterns."""
    print("\n=== 4. CAMPAIGN ARCHETYPE ANALYSIS ===")

    # Campaign type analysis
    print("CAMPAIGN TYPE DISTRIBUTION:")
    type_counts = campaign_summary['campaign_type'].value_counts()
    for campaign_type, count in type_counts.items():
        pct = count / len(campaign_summary) * 100
        print(f"{campaign_type}: {count:,} campaigns ({pct:.1f}%)")

    # Campaign archetype characteristics
    print(f"\n=== CAMPAIGN ARCHETYPE CHARACTERISTICS ===")
    archetype_stats = campaign_summary.groupby('campaign_type').agg({
        'CAMPAIGN_ID': 'count',
        'unique_products': ['median', 'mean'],
        'total_impressions': ['median', 'mean'],
        'duration_days': ['median', 'mean'],
        'ctr': ['median', 'mean'] if 'ctr' in campaign_summary.columns else lambda x: np.nan
    }).round(3)

    for campaign_type in ['Hero', 'Focused', 'Catalog']:
        if campaign_type in archetype_stats.index:
            count = archetype_stats.loc[campaign_type, ('CAMPAIGN_ID', 'count')]
            med_products = archetype_stats.loc[campaign_type, ('unique_products', 'median')]
            med_impressions = archetype_stats.loc[campaign_type, ('total_impressions', 'median')]
            med_duration = archetype_stats.loc[campaign_type, ('duration_days', 'median')]

            print(f"\n{campaign_type.upper()} CAMPAIGNS ({count:,} campaigns):")
            print(f"  Median products: {med_products}")
            print(f"  Median impressions: {med_impressions:,.0f}")
            print(f"  Median duration: {med_duration} days")

            if 'ctr' in campaign_summary.columns:
                med_ctr = archetype_stats.loc[campaign_type, ('ctr', 'median')]
                print(f"  Median CTR: {med_ctr:.3f}")

    # Duration analysis
    print(f"\n=== CAMPAIGN DURATION ANALYSIS ===")
    duration_counts = campaign_summary['duration_type'].value_counts()
    for duration_type, count in duration_counts.items():
        pct = count / len(campaign_summary) * 100
        print(f"{duration_type}: {count:,} campaigns ({pct:.1f}%)")

    # Cross-tabulation: Type × Duration
    if 'ctr' in campaign_summary.columns:
        print(f"\n=== CAMPAIGN PERFORMANCE MATRIX (CTR) ===")
        performance_matrix = campaign_summary.pivot_table(
            values='ctr',
            index='campaign_type',
            columns='duration_type',
            aggfunc='median'
        ).round(3)
        print(performance_matrix)

    return campaign_summary

def analyze_product_dynamics(datasets, campaign_summary):
    """Analyze product-level advertising dynamics with catalog integration."""
    print("\n=== 5. PRODUCT-LEVEL ADVERTISING DYNAMICS ===")

    if 'impressions' not in datasets:
        return None

    df_imp = datasets['impressions']
    df_catalog = datasets.get('catalog', pd.DataFrame())

    # Product-level aggregation
    product_stats = df_imp.groupby('PRODUCT_ID').agg({
        'VENDOR_ID': 'nunique',
        'CAMPAIGN_ID': 'nunique',
        'DATE': 'nunique',
        'IMPRESSIONS': 'sum',
        'DISTINCT_USERS_IMPRESSED': 'sum'
    }).reset_index()

    product_stats.columns = ['PRODUCT_ID', 'vendor_count', 'campaign_count', 'active_days', 'total_impressions', 'total_users']

    # Product concentration analysis
    print("PRODUCT IMPRESSION CONCENTRATION:")
    print_concentration_analysis(product_stats['total_impressions'], 'impressions')

    print("\nPRODUCT CAMPAIGN PARTICIPATION:")
    print_concentration_analysis(product_stats['campaign_count'], 'campaigns')

    # Product reuse patterns
    print(f"\n=== PRODUCT REUSE PATTERNS ===")
    single_campaign = (product_stats['campaign_count'] == 1).sum()
    multi_campaign = (product_stats['campaign_count'] > 1).sum()
    total_products = len(product_stats)

    print(f"Single-campaign products: {single_campaign:,} ({single_campaign/total_products*100:.1f}%)")
    print(f"Multi-campaign products: {multi_campaign:,} ({multi_campaign/total_products*100:.1f}%)")

    # Catalog integration if available
    if not df_catalog.empty:
        print(f"\n=== PRODUCT CATALOG INTEGRATION ===")

        # Merge with catalog
        product_catalog = product_stats.merge(
            df_catalog[['PRODUCT_ID', 'PRICE', 'BRAND', 'DEPARTMENT_ID', 'POOR_DESCRIPTION', 'PRICE_OUTLIER']],
            on='PRODUCT_ID',
            how='left'
        )

        # Price analysis
        valid_prices = product_catalog.dropna(subset=['PRICE'])
        if len(valid_prices) > 0:
            print("PRICE DISTRIBUTION OF ADVERTISED PRODUCTS:")
            print_concentration_analysis(valid_prices['PRICE'], 'price')

            # Price correlation with impressions
            price_corr = valid_prices['PRICE'].corr(valid_prices['total_impressions'])
            print(f"Price-Impression correlation: {price_corr:.3f}")

        # Brand diversity
        if 'BRAND' in product_catalog.columns:
            brand_diversity = df_imp.groupby('CAMPAIGN_ID')['PRODUCT_ID'].apply(
                lambda products: df_catalog[df_catalog['PRODUCT_ID'].isin(products)]['BRAND'].nunique()
            ).reset_index()
            brand_diversity.columns = ['CAMPAIGN_ID', 'unique_brands']

            print(f"\nBRAND DIVERSITY IN CAMPAIGNS:")
            single_brand = (brand_diversity['unique_brands'] == 1).sum()
            multi_brand = (brand_diversity['unique_brands'] > 1).sum()
            total_campaigns = len(brand_diversity)

            print(f"Single-brand campaigns: {single_brand:,} ({single_brand/total_campaigns*100:.1f}%)")
            print(f"Multi-brand campaigns: {multi_brand:,} ({multi_brand/total_campaigns*100:.1f}%)")

        # Data quality impact
        if 'POOR_DESCRIPTION' in product_catalog.columns:
            poor_desc_count = product_catalog['POOR_DESCRIPTION'].sum()
            print(f"\nDATA QUALITY ANALYSIS:")
            print(f"Products with poor descriptions: {poor_desc_count:,} ({poor_desc_count/len(product_catalog)*100:.1f}%)")

        if 'PRICE_OUTLIER' in product_catalog.columns:
            price_outlier_count = product_catalog['PRICE_OUTLIER'].sum()
            print(f"Products with outlier prices: {price_outlier_count:,} ({price_outlier_count/len(product_catalog)*100:.1f}%)")

    return product_stats

def analyze_funnel_performance(datasets, campaign_summary):
    """Analyze campaign funnel performance from impressions to purchases."""
    print("\n=== 6. CAMPAIGN FUNNEL PERFORMANCE ===")

    # Impression to click conversion
    if 'ctr' in campaign_summary.columns:
        print("CLICK-THROUGH RATE ANALYSIS:")
        print_concentration_analysis(campaign_summary['ctr'], 'CTR')

        # CTR by campaign type
        print(f"\n=== CTR BY CAMPAIGN TYPE ===")
        ctr_by_type = campaign_summary.groupby('campaign_type')['ctr'].agg(['count', 'median', 'mean']).round(3)
        for campaign_type in ctr_by_type.index:
            count = ctr_by_type.loc[campaign_type, 'count']
            median_ctr = ctr_by_type.loc[campaign_type, 'median']
            mean_ctr = ctr_by_type.loc[campaign_type, 'mean']
            print(f"{campaign_type}: {count:,} campaigns, median CTR: {median_ctr:.3f}, mean CTR: {mean_ctr:.3f}")

    # Purchase analysis if available
    if 'purchases' in datasets:
        df_purchases = datasets['purchases']
        df_imp = datasets['impressions']

        # Get promoted products
        promoted_products = set(df_imp['PRODUCT_ID'].unique())

        # Separate promoted vs organic purchases
        df_purchases['promoted'] = df_purchases['PRODUCT_ID'].isin(promoted_products)

        promoted_purchases = df_purchases[df_purchases['promoted']]
        organic_purchases = df_purchases[~df_purchases['promoted']]

        total_promoted_revenue = promoted_purchases['REVENUE_CENTS'].sum() / 100
        total_organic_revenue = organic_purchases['REVENUE_CENTS'].sum() / 100
        total_revenue = total_promoted_revenue + total_organic_revenue

        print(f"\n=== PROMOTED VS ORGANIC PURCHASES ===")
        print(f"Total promoted revenue: ${total_promoted_revenue:,.2f} ({total_promoted_revenue/total_revenue*100:.1f}%)")
        print(f"Total organic revenue: ${total_organic_revenue:,.2f} ({total_organic_revenue/total_revenue*100:.1f}%)")
        print(f"Promoted products: {len(promoted_products):,}")
        print(f"Total products with purchases: {df_purchases['PRODUCT_ID'].nunique():,}")

def generate_academic_tables(campaign_summary, vendor_stats, product_stats):
    """Generate academic-quality summary tables for paper inclusion."""
    print("\n=== 7. ACADEMIC SUMMARY TABLES ===")

    # Table A: Vendor Advertising Concentration
    print("\nTABLE A: VENDOR ADVERTISING CONCENTRATION (MAY 2025)")
    print("=" * 60)

    vendor_sorted = vendor_stats.sort_values('total_impressions', ascending=False)
    total_vendors = len(vendor_sorted)
    total_impressions = vendor_sorted['total_impressions'].sum()
    total_campaigns = vendor_sorted['total_campaigns'].sum()
    total_products = vendor_sorted['total_products'].sum()

    percentile_groups = [
        ("Top 1%", int(total_vendors * 0.01)),
        ("Top 5%", int(total_vendors * 0.05)),
        ("Top 20%", int(total_vendors * 0.20)),
        ("Bottom 50%", total_vendors - int(total_vendors * 0.50))
    ]

    for label, n_vendors in percentile_groups:
        if label == "Bottom 50%":
            group_data = vendor_sorted.tail(n_vendors)
        else:
            group_data = vendor_sorted.head(n_vendors)

        campaigns_pct = group_data['total_campaigns'].sum() / total_campaigns * 100
        products_pct = group_data['total_products'].sum() / total_products * 100
        impressions_pct = group_data['total_impressions'].sum() / total_impressions * 100

        print(f"{label:12} ({n_vendors:,} vendors): {campaigns_pct:5.1f}% campaigns, {products_pct:5.1f}% products, {impressions_pct:5.1f}% impressions")

    # Table B: Campaign Archetype Summary
    print(f"\nTABLE B: CAMPAIGN ARCHETYPE SUMMARY (MAY 2025)")
    print("=" * 60)

    archetype_summary = campaign_summary.groupby('campaign_type').agg({
        'CAMPAIGN_ID': 'count',
        'unique_products': 'median',
        'total_impressions': 'median',
        'duration_days': 'median',
        'ctr': 'median' if 'ctr' in campaign_summary.columns else lambda x: np.nan
    }).round(2)

    total_campaigns = len(campaign_summary)

    print(f"{'Type':12} {'Count':>8} {'Pct':>6} {'Med Products':>12} {'Med Impressions':>15} {'Med Duration':>12} {'Med CTR':>10}")
    print("-" * 80)

    for campaign_type in ['Hero', 'Focused', 'Catalog']:
        if campaign_type in archetype_summary.index:
            count = int(archetype_summary.loc[campaign_type, 'CAMPAIGN_ID'])
            pct = count / total_campaigns * 100
            med_products = archetype_summary.loc[campaign_type, 'unique_products']
            med_impressions = int(archetype_summary.loc[campaign_type, 'total_impressions'])
            med_duration = archetype_summary.loc[campaign_type, 'duration_days']
            med_ctr = archetype_summary.loc[campaign_type, 'ctr'] if 'ctr' in campaign_summary.columns else 0

            print(f"{campaign_type:12} {count:8,} {pct:5.1f}% {med_products:12.0f} {med_impressions:15,} {med_duration:12.0f} {med_ctr:10.3f}")

    # Table C: Product-Level Advertising Dynamics
    print(f"\nTABLE C: PRODUCT-LEVEL ADVERTISING DYNAMICS (MAY 2025)")
    print("=" * 60)

    product_sorted = product_stats.sort_values('total_impressions', ascending=False)
    total_product_impressions = product_sorted['total_impressions'].sum()

    # Product concentration
    top1_products = int(len(product_sorted) * 0.01)
    top5_products = int(len(product_sorted) * 0.05)
    top20_products = int(len(product_sorted) * 0.20)

    top1_impressions_pct = product_sorted.head(top1_products)['total_impressions'].sum() / total_product_impressions * 100
    top5_impressions_pct = product_sorted.head(top5_products)['total_impressions'].sum() / total_product_impressions * 100
    top20_impressions_pct = product_sorted.head(top20_products)['total_impressions'].sum() / total_product_impressions * 100

    multi_campaign_products = (product_stats['campaign_count'] > 1).sum()
    multi_campaign_pct = multi_campaign_products / len(product_stats) * 100

    print(f"Top 1% of products ({top1_products:,}): {top1_impressions_pct:.1f}% of impressions")
    print(f"Top 5% of products ({top5_products:,}): {top5_impressions_pct:.1f}% of impressions")
    print(f"Top 20% of products ({top20_products:,}): {top20_impressions_pct:.1f}% of impressions")
    print(f"Products in multiple campaigns: {multi_campaign_products:,} ({multi_campaign_pct:.1f}%)")

def print_executive_summary(datasets, campaign_summary, vendor_stats):
    """Print executive summary of key findings."""
    print("\n=== 8. EXECUTIVE SUMMARY ===")
    print("=" * 60)

    # Dataset overview
    print("DATASET OVERVIEW (MAY 2025):")
    for name, df in datasets.items():
        if name == 'catalog':
            print(f"  {name.capitalize()}: {len(df):,} products")
        else:
            print(f"  {name.capitalize()}: {len(df):,} records")

    # Key metrics
    print(f"\nKEY MARKETPLACE METRICS:")
    print(f"  Unique vendors: {len(vendor_stats):,}")
    print(f"  Unique campaigns: {len(campaign_summary):,}")
    print(f"  Unique products advertised: {datasets['impressions']['PRODUCT_ID'].nunique():,}")
    print(f"  Total impressions: {datasets['impressions']['IMPRESSIONS'].sum():,}")
    if 'clicks' in datasets:
        print(f"  Total clicks: {datasets['clicks']['CLICKS'].sum():,}")

    # Market structure insights
    gini_impressions = calculate_gini_coefficient(vendor_stats['total_impressions'])
    gini_campaigns = calculate_gini_coefficient(vendor_stats['total_campaigns'])

    print(f"\nMARKET CONCENTRATION:")
    print(f"  Vendor impression Gini: {gini_impressions:.3f}")
    print(f"  Vendor campaign Gini: {gini_campaigns:.3f}")

    # Campaign characteristics
    print(f"\nCAMPAIGN CHARACTERISTICS:")
    print(f"  Median products per campaign: {campaign_summary['unique_products'].median():.0f}")
    print(f"  Median campaign duration: {campaign_summary['duration_days'].median():.0f} days")
    if 'ctr' in campaign_summary.columns:
        print(f"  Median CTR: {campaign_summary['ctr'].median():.3f}")

    # Vendor behavior
    print(f"\nVENDOR BEHAVIOR:")
    print(f"  Median campaigns per vendor: {vendor_stats['total_campaigns'].median():.0f}")
    small_vendors = (vendor_stats['total_campaigns'] == 1).sum()
    print(f"  Single-campaign vendors: {small_vendors:,} ({small_vendors/len(vendor_stats)*100:.1f}%)")

def main():
    """Execute comprehensive campaign structure EDA for academic paper."""

    print("CAMPAIGN STRUCTURE EDA FOR ACADEMIC PAPER")
    print("=" * 70)
    print("Section: 'The Structure of Marketplace Advertising'")
    print("Period: May 2025 (31 contiguous days)")
    print("Purpose: Foundational descriptive analysis before causal inference")
    print("=" * 70)

    # Load comprehensive May 2025 sample
    datasets = load_may_2025_sample()

    if not datasets or 'impressions' not in datasets:
        print("Insufficient data available! Analysis terminated.")
        return

    # Create campaign summary table
    campaign_summary = create_campaign_summary(datasets)

    if campaign_summary is None:
        print("Failed to create campaign summary! Analysis terminated.")
        return

    # Run comprehensive analyses
    vendor_stats = analyze_vendor_concentration(campaign_summary)
    campaign_summary_enhanced = analyze_campaign_archetypes(campaign_summary)
    product_stats = analyze_product_dynamics(datasets, campaign_summary)
    analyze_funnel_performance(datasets, campaign_summary)

    # Generate academic tables
    generate_academic_tables(campaign_summary, vendor_stats, product_stats)

    # Executive summary
    print_executive_summary(datasets, campaign_summary, vendor_stats)

    print(f"\n{'=' * 70}")
    print("CAMPAIGN STRUCTURE EDA COMPLETE")
    print("Ready for academic paper inclusion")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()