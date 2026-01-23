#!/usr/bin/env python3
"""
Vendor-Week Panel Construction

Purpose: Create vendor-week level panel for causal analysis of advertising effects
Period: Full 178 days (2025-03-14 to 2025-09-07) aggregated to ~25 weeks
Unit of Analysis: Vendor-Week (VENDOR_ID × Week)

Panel Structure:
- Outcomes: Revenue, Sales (total_revenue, total_sales, total_units_sold)
- Treatments: Impressions, Clicks, Auctions (total_impressions, total_clicks, total_auctions)
- Controls: Campaign activity, promotion intensity, temporal patterns

Research Purpose: Enable fixed effects estimation of advertising impact on vendor performance
Expected Output: Balanced panel suitable for causal identification with vendor and time FE
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def create_date_range():
    """Create complete date range for analysis."""
    start_date = datetime(2025, 3, 14)
    end_date = datetime(2025, 9, 7)
    date_range = pd.date_range(start_date, end_date, freq='D')
    return [d.strftime('%Y-%m-%d') for d in date_range]

def sample_vendors():
    """Select top vendors by activity for comprehensive analysis."""
    print("=== 1. TOP VENDOR SELECTION FOR PANEL CONSTRUCTION ===")

    BASE_PATH = Path('../data')

    # Load impressions and purchases data for ranking vendors
    impressions_dir = BASE_PATH / 'product_daily_impressions_dataset'
    purchases_dir = BASE_PATH / 'product_daily_purchases_dataset'

    sample_files = [
        impressions_dir / 'data_2025-03-14.parquet',
        impressions_dir / 'data_2025-04-01.parquet',
        impressions_dir / 'data_2025-05-01.parquet',
        impressions_dir / 'data_2025-06-01.parquet',
        impressions_dir / 'data_2025-07-01.parquet',
        impressions_dir / 'data_2025-08-01.parquet'
    ]

    print("Loading sample data to rank vendors by activity...")

    all_impressions = []
    for file_path in tqdm(sample_files, desc="Loading impression samples"):
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_impressions.append(df)

    if not all_impressions:
        print("No impression files found for vendor ranking!")
        return None, None

    df_impressions = pd.concat(all_impressions, ignore_index=True)

    # Rank vendors by impression activity
    vendor_impressions = df_impressions.groupby('VENDOR_ID').agg({
        'IMPRESSIONS': 'sum',
        'TOTAL_IMPRESSIONS': 'sum',
        'DISTINCT_USERS_IMPRESSED': 'sum',
        'CAMPAIGN_ID': 'nunique',
        'PRODUCT_ID': 'nunique'
    }).reset_index()

    vendor_impressions['impression_score'] = (
        vendor_impressions['TOTAL_IMPRESSIONS'] * 0.4 +
        vendor_impressions['DISTINCT_USERS_IMPRESSED'] * 0.3 +
        vendor_impressions['CAMPAIGN_ID'] * 0.2 +
        vendor_impressions['PRODUCT_ID'] * 0.1
    )

    # Load purchases data for additional ranking
    purchase_files = [
        purchases_dir / 'data_2025-03-14.parquet',
        purchases_dir / 'data_2025-04-01.parquet',
        purchases_dir / 'data_2025-05-01.parquet',
        purchases_dir / 'data_2025-06-01.parquet',
        purchases_dir / 'data_2025-07-01.parquet',
        purchases_dir / 'data_2025-08-01.parquet'
    ]

    all_purchases = []
    for file_path in tqdm(purchase_files, desc="Loading purchase samples"):
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_purchases.append(df)

    vendor_revenue = pd.DataFrame()
    if all_purchases:
        df_purchases = pd.concat(all_purchases, ignore_index=True)
        vendor_revenue = df_purchases.groupby('PRODUCT_ID').agg({
            'REVENUE_CENTS': 'sum',
            'PURCHASES': 'sum',
            'UNITS_SOLD': 'sum'
        }).reset_index()

        # Map to vendors via impressions
        vendor_product_map = df_impressions[['VENDOR_ID', 'PRODUCT_ID']].drop_duplicates()
        vendor_revenue = vendor_revenue.merge(vendor_product_map, on='PRODUCT_ID', how='left')
        vendor_revenue = vendor_revenue[vendor_revenue['VENDOR_ID'].notna()]

        vendor_revenue = vendor_revenue.groupby('VENDOR_ID').agg({
            'REVENUE_CENTS': 'sum',
            'PURCHASES': 'sum',
            'UNITS_SOLD': 'sum'
        }).reset_index()

        vendor_revenue['revenue_score'] = (
            vendor_revenue['REVENUE_CENTS'] * 0.6 +
            vendor_revenue['PURCHASES'] * 0.3 +
            vendor_revenue['UNITS_SOLD'] * 0.1
        )

    # Combine scores
    vendor_ranking = vendor_impressions.merge(vendor_revenue, on='VENDOR_ID', how='left')
    vendor_ranking['revenue_score'] = vendor_ranking['revenue_score'].fillna(0)

    vendor_ranking['combined_score'] = (
        vendor_ranking['impression_score'] * 0.6 +
        vendor_ranking['revenue_score'] * 0.4
    )

    # Select top vendors
    vendor_ranking = vendor_ranking.sort_values('combined_score', ascending=False)
    n_vendor_sample = min(2000, len(vendor_ranking))
    top_vendors = vendor_ranking.head(n_vendor_sample)
    sampled_vendors = top_vendors['VENDOR_ID'].values

    print(f"Total vendors analyzed: {len(vendor_ranking):,}")
    print(f"Top vendors selected: {len(sampled_vendors):,}")
    print(f"Top vendor stats:")
    print(f"  Mean impressions: {top_vendors['TOTAL_IMPRESSIONS'].mean():,.0f}")
    print(f"  Mean campaigns: {top_vendors['CAMPAIGN_ID'].mean():.1f}")
    print(f"  Mean products: {top_vendors['PRODUCT_ID'].mean():.1f}")
    if 'REVENUE_CENTS' in top_vendors.columns:
        print(f"  Mean revenue: ${top_vendors['REVENUE_CENTS'].mean()/100:,.2f}")

    # Get ALL products for selected top vendors
    print("Collecting all products for top vendors...")

    # Products from impressions (promoted products)
    promoted_products = df_impressions[df_impressions['VENDOR_ID'].isin(sampled_vendors)]['PRODUCT_ID'].unique()

    # Products from purchases (all products sold by these vendors)
    all_vendor_products = set(promoted_products)

    if all_purchases:
        purchase_products = df_purchases['PRODUCT_ID'].unique()
        all_vendor_products.update(purchase_products)
        print(f"Total unique products in purchases: {len(purchase_products):,}")

    sampled_products = list(all_vendor_products)

    print(f"Products for top vendors:")
    print(f"  - Promoted products: {len(promoted_products):,}")
    print(f"  - Total products: {len(sampled_products):,}")

    return sampled_vendors, sampled_products

def load_daily_data(sampled_vendors, sampled_products):
    """Load 178-day daily data for all datasets."""
    print("\n=== 2. LOADING DAILY DATA FOR PANEL CONSTRUCTION ===")

    BASE_PATH = Path('../data')
    date_strings = create_date_range()

    print(f"Loading {len(date_strings)} days of data...")

    # Initialize datasets
    all_data = {}

    # Dataset configurations
    datasets = {
        'purchases': {
            'dir': 'product_daily_purchases_dataset',
            'vendor_filter': False,  # Purchases don't have VENDOR_ID
            'product_filter': True
        },
        'impressions': {
            'dir': 'product_daily_impressions_dataset',
            'vendor_filter': True,
            'product_filter': True
        },
        'clicks': {
            'dir': 'product_daily_clicks_dataset',
            'vendor_filter': True,
            'product_filter': True
        },
        'auctions': {
            'dir': 'product_daily_auctions_dataset',
            'vendor_filter': True,
            'product_filter': True
        }
    }

    for dataset_name, config in datasets.items():
        print(f"\nLoading {dataset_name} data...")
        dataset_panels = []

        dataset_dir = BASE_PATH / config['dir']

        for date_str in tqdm(date_strings, desc=f"Loading {dataset_name}"):
            file_path = dataset_dir / f'data_{date_str}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)

                # Apply filters
                if config['vendor_filter'] and config['product_filter']:
                    # Filter by both vendor and product
                    df_filtered = df[
                        (df['VENDOR_ID'].isin(sampled_vendors)) &
                        (df['PRODUCT_ID'].isin(sampled_products))
                    ].copy()
                elif config['product_filter']:
                    # Filter by product only (for purchases)
                    df_filtered = df[df['PRODUCT_ID'].isin(sampled_products)].copy()
                else:
                    df_filtered = df.copy()

                df_filtered['DATE'] = pd.to_datetime(date_str)
                dataset_panels.append(df_filtered)

        if dataset_panels:
            all_data[dataset_name] = pd.concat(dataset_panels, ignore_index=True)
            print(f"{dataset_name.capitalize()} data: {len(all_data[dataset_name]):,} records")

    return all_data

def create_week_identifiers(all_data):
    """Add week identifiers to all datasets."""
    print("\n=== 3. CREATING WEEK IDENTIFIERS ===")

    for dataset_name, df in all_data.items():
        print(f"Adding week identifiers to {dataset_name}...")

        # Convert DATE to datetime if not already
        df['DATE'] = pd.to_datetime(df['DATE'])

        # Create week identifiers
        df['year'] = df['DATE'].dt.year
        df['week_of_year'] = df['DATE'].dt.isocalendar().week
        df['year_week'] = df['year'].astype(str) + '-W' + df['week_of_year'].astype(str).str.zfill(2)

        # Create sequential week number starting from 1
        min_date = df['DATE'].min()
        df['week_number'] = ((df['DATE'] - min_date).dt.days // 7) + 1

        all_data[dataset_name] = df

    # Print week summary
    sample_df = list(all_data.values())[0]
    week_summary = sample_df.groupby(['year_week', 'week_number'])['DATE'].agg(['min', 'max', 'count']).reset_index()
    print(f"\nWeek structure:")
    print(f"Total weeks identified: {len(week_summary)}")
    print(f"Week range: {week_summary['year_week'].iloc[0]} to {week_summary['year_week'].iloc[-1]}")
    print(f"Sequential week numbers: {week_summary['week_number'].min()} to {week_summary['week_number'].max()}")

    return all_data

def create_vendor_product_mapping(all_data):
    """Create vendor-product mapping for purchases data aggregation."""
    print("\n=== 4. CREATING VENDOR-PRODUCT MAPPING ===")

    # Extract vendor-product mapping from impressions data
    if 'impressions' in all_data:
        vendor_product_map = all_data['impressions'][['VENDOR_ID', 'PRODUCT_ID']].drop_duplicates()
        print(f"Vendor-product mappings from impressions: {len(vendor_product_map):,}")

        # Add vendor info to purchases data
        purchases_with_vendor = all_data['purchases'].merge(
            vendor_product_map,
            on='PRODUCT_ID',
            how='left'
        )

        # Check coverage
        total_purchases = len(all_data['purchases'])
        mapped_purchases = purchases_with_vendor['VENDOR_ID'].notna().sum()
        coverage = mapped_purchases / total_purchases * 100

        print(f"Purchase records with vendor mapping: {mapped_purchases:,} / {total_purchases:,} ({coverage:.1f}%)")

        # Update purchases data
        all_data['purchases'] = purchases_with_vendor

        return vendor_product_map
    else:
        print("Warning: No impressions data available for vendor mapping")
        return None

def aggregate_outcomes(all_data):
    """Aggregate outcome variables to vendor-week level."""
    print("\n=== 5. AGGREGATING OUTCOME VARIABLES ===")

    if 'purchases' not in all_data:
        print("No purchases data available for outcome aggregation")
        return None

    purchases_df = all_data['purchases']

    # Filter to records with vendor mapping
    vendor_purchases = purchases_df[purchases_df['VENDOR_ID'].notna()].copy()
    print(f"Purchase records with vendor info: {len(vendor_purchases):,}")

    # Aggregate to vendor-week level
    outcome_aggregation = {
        'REVENUE_CENTS': 'sum',
        'PURCHASES': 'sum',
        'UNITS_SOLD': 'sum',
        'LINES_SOLD': 'sum',
        'PRODUCT_ID': 'nunique',  # Active products per week
        'DISTINCT_USERS_PURCHASED': 'sum'  # Total distinct customers
    }

    vendor_week_outcomes = vendor_purchases.groupby(['VENDOR_ID', 'week_number', 'year_week']).agg(
        outcome_aggregation
    ).reset_index()

    # Rename columns for clarity
    vendor_week_outcomes.rename(columns={
        'REVENUE_CENTS': 'total_revenue_cents',
        'PURCHASES': 'total_purchases',
        'UNITS_SOLD': 'total_units_sold',
        'LINES_SOLD': 'total_lines_sold',
        'PRODUCT_ID': 'active_products',
        'DISTINCT_USERS_PURCHASED': 'total_customers'
    }, inplace=True)

    print(f"Vendor-week outcome records: {len(vendor_week_outcomes):,}")
    print(f"Unique vendors in outcomes: {vendor_week_outcomes['VENDOR_ID'].nunique():,}")
    print(f"Week range in outcomes: {vendor_week_outcomes['week_number'].min()} to {vendor_week_outcomes['week_number'].max()}")

    return vendor_week_outcomes

def aggregate_treatments(all_data):
    """Aggregate treatment variables to vendor-week level."""
    print("\n=== 6. AGGREGATING TREATMENT VARIABLES ===")

    treatment_datasets = ['impressions', 'clicks', 'auctions']
    vendor_week_treatments = None

    for dataset_name in treatment_datasets:
        if dataset_name not in all_data:
            print(f"Warning: {dataset_name} data not available")
            continue

        df = all_data[dataset_name]
        print(f"\nAggregating {dataset_name} to vendor-week level...")

        if dataset_name == 'impressions':
            agg_dict = {
                'IMPRESSIONS': 'sum',
                'TOTAL_IMPRESSIONS': 'sum',
                'DISTINCT_USERS_IMPRESSED': 'sum',
                'CAMPAIGN_ID': 'nunique',
                'PRODUCT_ID': 'nunique'
            }
            rename_dict = {
                'IMPRESSIONS': 'total_impressions',
                'TOTAL_IMPRESSIONS': 'total_impressions_all',
                'DISTINCT_USERS_IMPRESSED': 'total_users_impressed',
                'CAMPAIGN_ID': 'active_campaigns',
                'PRODUCT_ID': 'promoted_products'
            }

        elif dataset_name == 'clicks':
            agg_dict = {
                'CLICKS': 'sum',
                'TOTAL_CLICKS': 'sum',
                'DISTINCT_USERS_CLICKED': 'sum'
            }
            rename_dict = {
                'CLICKS': 'total_clicks',
                'TOTAL_CLICKS': 'total_clicks_all',
                'DISTINCT_USERS_CLICKED': 'total_users_clicked'
            }

        elif dataset_name == 'auctions':
            agg_dict = {
                'PRODUCT_AUCTIONS_COUNT': 'sum',
                'TOTAL_BIDS_FOR_PRODUCT': 'sum',
                'TOTAL_WINS_FOR_PRODUCT': 'sum',
                'AVG_BID_RANK_FOR_PRODUCT': 'mean',
                'DISTINCT_BIDDERS_FOR_PRODUCT': 'sum'
            }
            rename_dict = {
                'PRODUCT_AUCTIONS_COUNT': 'total_auctions',
                'TOTAL_BIDS_FOR_PRODUCT': 'total_bids',
                'TOTAL_WINS_FOR_PRODUCT': 'total_wins',
                'AVG_BID_RANK_FOR_PRODUCT': 'avg_bid_rank',
                'DISTINCT_BIDDERS_FOR_PRODUCT': 'total_bidders'
            }

        # Aggregate to vendor-week
        vendor_week_data = df.groupby(['VENDOR_ID', 'week_number', 'year_week']).agg(agg_dict).reset_index()
        vendor_week_data.rename(columns=rename_dict, inplace=True)

        print(f"{dataset_name.capitalize()} vendor-week records: {len(vendor_week_data):,}")

        # Merge with existing treatment data
        if vendor_week_treatments is None:
            vendor_week_treatments = vendor_week_data
        else:
            vendor_week_treatments = vendor_week_treatments.merge(
                vendor_week_data,
                on=['VENDOR_ID', 'week_number', 'year_week'],
                how='outer'
            )

    if vendor_week_treatments is not None:
        print(f"\nCombined treatment data:")
        print(f"Total vendor-week records: {len(vendor_week_treatments):,}")
        print(f"Unique vendors: {vendor_week_treatments['VENDOR_ID'].nunique():,}")

    return vendor_week_treatments

def create_balanced_panel(vendor_week_outcomes, vendor_week_treatments, sampled_vendors):
    """Create balanced panel with all vendor-week combinations."""
    print("\n=== 7. CREATING BALANCED PANEL ===")

    # Get complete week range
    if vendor_week_outcomes is not None:
        all_weeks = vendor_week_outcomes[['week_number', 'year_week']].drop_duplicates()
    elif vendor_week_treatments is not None:
        all_weeks = vendor_week_treatments[['week_number', 'year_week']].drop_duplicates()
    else:
        print("Error: No data available for panel construction")
        return None

    print(f"Week range: {all_weeks['week_number'].min()} to {all_weeks['week_number'].max()}")
    print(f"Total weeks: {len(all_weeks)}")

    # Create complete vendor-week grid
    vendor_df = pd.DataFrame({'VENDOR_ID': sampled_vendors})
    complete_panel = vendor_df.assign(key=1).merge(all_weeks.assign(key=1), on='key').drop('key', axis=1)

    print(f"Complete panel dimensions: {len(complete_panel):,} vendor-week observations")
    print(f"Vendors: {complete_panel['VENDOR_ID'].nunique():,}")
    print(f"Weeks per vendor: {len(complete_panel) // complete_panel['VENDOR_ID'].nunique()}")

    # Merge with outcomes
    if vendor_week_outcomes is not None:
        complete_panel = complete_panel.merge(
            vendor_week_outcomes,
            on=['VENDOR_ID', 'week_number', 'year_week'],
            how='left'
        )
        print(f"Panel after outcomes merge: {len(complete_panel):,} records")

    # Merge with treatments
    if vendor_week_treatments is not None:
        complete_panel = complete_panel.merge(
            vendor_week_treatments,
            on=['VENDOR_ID', 'week_number', 'year_week'],
            how='left'
        )
        print(f"Panel after treatments merge: {len(complete_panel):,} records")

    return complete_panel

def add_control_variables(complete_panel):
    """Add control variables to the panel."""
    print("\n=== 8. ADDING CONTROL VARIABLES ===")

    # Fill missing values with zeros (inactive periods)
    numeric_columns = complete_panel.select_dtypes(include=[np.number]).columns
    complete_panel[numeric_columns] = complete_panel[numeric_columns].fillna(0)

    # Create derived control variables
    print("Creating derived control variables...")

    # Treatment indicators
    complete_panel['is_promoted_week'] = (complete_panel.get('total_impressions', 0) > 0).astype(int)
    complete_panel['has_clicks'] = (complete_panel.get('total_clicks', 0) > 0).astype(int)
    complete_panel['has_auctions'] = (complete_panel.get('total_auctions', 0) > 0).astype(int)

    # Intensity measures
    if 'total_clicks' in complete_panel.columns and 'total_impressions' in complete_panel.columns:
        complete_panel['click_through_rate'] = np.where(
            complete_panel['total_impressions'] > 0,
            complete_panel['total_clicks'] / complete_panel['total_impressions'],
            0
        )

    if 'total_wins' in complete_panel.columns and 'total_bids' in complete_panel.columns:
        complete_panel['auction_win_rate'] = np.where(
            complete_panel['total_bids'] > 0,
            complete_panel['total_wins'] / complete_panel['total_bids'],
            0
        )

    if 'promoted_products' in complete_panel.columns and 'active_products' in complete_panel.columns:
        complete_panel['promotion_intensity'] = np.where(
            complete_panel['active_products'] > 0,
            complete_panel['promoted_products'] / complete_panel['active_products'],
            0
        )

    if 'active_campaigns' in complete_panel.columns and 'promoted_products' in complete_panel.columns:
        complete_panel['campaign_concentration'] = np.where(
            complete_panel['promoted_products'] > 0,
            complete_panel['active_campaigns'] / complete_panel['promoted_products'],
            0
        )

    # Temporal controls
    complete_panel['week_of_year'] = complete_panel['year_week'].str.extract(r'W(\d+)').astype(int)

    # Lag indicators (previous week activity)
    complete_panel_sorted = complete_panel.sort_values(['VENDOR_ID', 'week_number'])
    complete_panel_sorted['prev_week_promoted'] = complete_panel_sorted.groupby('VENDOR_ID')['is_promoted_week'].shift(1).fillna(0)
    complete_panel_sorted['prev_week_revenue'] = complete_panel_sorted.groupby('VENDOR_ID')['total_revenue_cents'].shift(1).fillna(0)

    print(f"Panel with control variables: {len(complete_panel_sorted):,} records")
    print(f"Control variables added: {len([col for col in complete_panel_sorted.columns if col not in ['VENDOR_ID', 'week_number', 'year_week']])} variables")

    return complete_panel_sorted

def validate_panel(final_panel):
    """Validate panel structure and data quality."""
    print("\n=== 9. PANEL VALIDATION ===")

    # Basic structure validation
    print("PANEL STRUCTURE:")
    print(f"Total observations: {len(final_panel):,}")
    print(f"Unique vendors: {final_panel['VENDOR_ID'].nunique():,}")
    print(f"Unique weeks: {final_panel['week_number'].nunique()}")
    print(f"Expected observations: {final_panel['VENDOR_ID'].nunique() * final_panel['week_number'].nunique():,}")

    # Check balance
    vendor_week_counts = final_panel.groupby('VENDOR_ID').size()
    print(f"Observations per vendor - Min: {vendor_week_counts.min()}, Max: {vendor_week_counts.max()}")

    if vendor_week_counts.min() == vendor_week_counts.max():
        print("✓ Panel is perfectly balanced")
    else:
        print("⚠ Panel is unbalanced")

    # Missing data analysis
    print("\nMISSING DATA ANALYSIS:")
    missing_summary = final_panel.isnull().sum()
    missing_pct = (missing_summary / len(final_panel)) * 100

    for col in missing_summary.index:
        if missing_summary[col] > 0:
            print(f"{col}: {missing_summary[col]:,} ({missing_pct[col]:.1f}%)")

    # Outcome variable distributions
    print("\nOUTCOME VARIABLE DISTRIBUTIONS:")
    outcome_vars = ['total_revenue_cents', 'total_purchases', 'total_units_sold']

    for var in outcome_vars:
        if var in final_panel.columns:
            series = final_panel[var]
            print(f"\n{var}:")
            print(f"  Non-zero observations: {(series > 0).sum():,} ({(series > 0).mean()*100:.1f}%)")
            print(f"  Mean: {series.mean():.2f}")
            print(f"  Median: {series.median():.2f}")
            print(f"  P95: {series.quantile(0.95):.2f}")
            print(f"  Max: {series.max():.2f}")

    # Treatment variable distributions
    print("\nTREATMENT VARIABLE DISTRIBUTIONS:")
    treatment_vars = ['total_impressions', 'total_clicks', 'total_auctions']

    for var in treatment_vars:
        if var in final_panel.columns:
            series = final_panel[var]
            print(f"\n{var}:")
            print(f"  Non-zero observations: {(series > 0).sum():,} ({(series > 0).mean()*100:.1f}%)")
            print(f"  Mean: {series.mean():.2f}")
            print(f"  Median: {series.median():.2f}")
            print(f"  P95: {series.quantile(0.95):.2f}")

    return True

def generate_panel_summary(final_panel):
    """Generate comprehensive panel summary for academic output."""
    print("\n=== 10. ACADEMIC PANEL SUMMARY ===")

    print("VENDOR-WEEK PANEL CONSTRUCTION SUMMARY")
    print("=" * 60)

    # Panel dimensions
    print(f"Panel Dimensions:")
    print(f"  Vendors: {final_panel['VENDOR_ID'].nunique():,}")
    print(f"  Time periods: {final_panel['week_number'].nunique()} weeks")
    print(f"  Total observations: {len(final_panel):,}")
    print(f"  Panel type: {'Balanced' if final_panel.groupby('VENDOR_ID').size().nunique() == 1 else 'Unbalanced'}")

    # Time coverage
    print(f"\nTemporal Coverage:")
    print(f"  Week range: {final_panel['week_number'].min()} to {final_panel['week_number'].max()}")
    print(f"  Year-week range: {final_panel['year_week'].min()} to {final_panel['year_week'].max()}")

    # Treatment variation
    print(f"\nTreatment Variation:")
    if 'is_promoted_week' in final_panel.columns:
        promoted_obs = final_panel['is_promoted_week'].sum()
        total_obs = len(final_panel)
        print(f"  Promoted vendor-weeks: {promoted_obs:,} ({promoted_obs/total_obs*100:.1f}%)")
        print(f"  Non-promoted vendor-weeks: {total_obs - promoted_obs:,} ({(total_obs - promoted_obs)/total_obs*100:.1f}%)")

        # Vendor-level treatment distribution
        vendor_promotion = final_panel.groupby('VENDOR_ID')['is_promoted_week'].agg(['sum', 'count', 'mean'])
        always_promoted = (vendor_promotion['mean'] == 1).sum()
        never_promoted = (vendor_promotion['mean'] == 0).sum()
        sometimes_promoted = len(vendor_promotion) - always_promoted - never_promoted

        print(f"  Always-promoted vendors: {always_promoted:,}")
        print(f"  Never-promoted vendors: {never_promoted:,}")
        print(f"  Sometimes-promoted vendors: {sometimes_promoted:,}")

    # Variable summary
    print(f"\nVariable Summary:")
    numeric_vars = final_panel.select_dtypes(include=[np.number]).columns
    print(f"  Total variables: {len(final_panel.columns)}")
    print(f"  Numeric variables: {len(numeric_vars)}")

    # Key variable categories
    outcome_vars = [col for col in final_panel.columns if any(x in col for x in ['revenue', 'purchases', 'units', 'customers'])]
    treatment_vars = [col for col in final_panel.columns if any(x in col for x in ['impressions', 'clicks', 'auctions', 'campaigns'])]
    control_vars = [col for col in final_panel.columns if any(x in col for x in ['week', 'rate', 'intensity', 'promoted'])]

    print(f"  Outcome variables: {len(outcome_vars)}")
    print(f"  Treatment variables: {len(treatment_vars)}")
    print(f"  Control variables: {len(control_vars)}")

    print(f"\nData Quality:")
    total_cells = len(final_panel) * len(final_panel.columns)
    missing_cells = final_panel.isnull().sum().sum()
    print(f"  Missing data: {missing_cells:,} / {total_cells:,} ({missing_cells/total_cells*100:.2f}%)")

    # Revenue distribution
    if 'total_revenue_cents' in final_panel.columns:
        revenue_stats = final_panel['total_revenue_cents'].describe()
        print(f"\nRevenue Distribution (cents):")
        print(f"  Mean: ${revenue_stats['mean']/100:,.2f}")
        print(f"  Median: ${revenue_stats['50%']/100:,.2f}")
        print(f"  P90: ${revenue_stats.quantile(0.9)/100:,.2f}")
        print(f"  Max: ${revenue_stats['max']/100:,.2f}")

    print("\n" + "=" * 60)
    print("PANEL READY FOR ECONOMETRIC ANALYSIS")
    print("Suitable for vendor and time fixed effects estimation")
    print("=" * 60)

def main():
    """Execute vendor-week panel construction."""

    print("VENDOR-WEEK PANEL CONSTRUCTION")
    print("=" * 70)
    print("Purpose: Create vendor-week panel for advertising effect analysis")
    print("Unit of Analysis: Vendor-Week")
    print("Time Period: 178 days (2025-03-14 to 2025-09-07)")
    print("=" * 70)

    # Step 1: Sample vendors
    sampled_vendors, sampled_products = sample_vendors()
    if sampled_vendors is None:
        print("Failed to sample vendors! Analysis terminated.")
        return

    # Step 2: Load daily data
    all_data = load_daily_data(sampled_vendors, sampled_products)
    if not all_data:
        print("No data loaded! Analysis terminated.")
        return

    # Step 3: Add week identifiers
    all_data = create_week_identifiers(all_data)

    # Step 4: Create vendor-product mapping
    vendor_product_map = create_vendor_product_mapping(all_data)

    # Step 5: Aggregate outcomes
    vendor_week_outcomes = aggregate_outcomes(all_data)

    # Step 6: Aggregate treatments
    vendor_week_treatments = aggregate_treatments(all_data)

    # Step 7: Create balanced panel
    complete_panel = create_balanced_panel(vendor_week_outcomes, vendor_week_treatments, sampled_vendors)
    if complete_panel is None:
        print("Failed to create panel! Analysis terminated.")
        return

    # Step 8: Add control variables
    final_panel = add_control_variables(complete_panel)

    # Step 9: Validate panel
    validate_panel(final_panel)

    # Step 10: Generate summary
    generate_panel_summary(final_panel)

    print(f"\n{'=' * 70}")
    print("VENDOR-WEEK PANEL CONSTRUCTION COMPLETE")
    print(f"Final panel: {len(final_panel):,} vendor-week observations")
    print(f"Ready for fixed effects analysis")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()