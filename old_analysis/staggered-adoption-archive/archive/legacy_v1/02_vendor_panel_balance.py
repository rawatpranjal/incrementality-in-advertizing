#!/usr/bin/env python3
"""
EDA Q2: How balanced is the Vendor Panel?

How many distinct VENDOR_IDs appear in AUCTIONS_RESULTS (bidders) vs. CATALOG (sellers)?
We need to define the "universe" of vendors. Are there vendors in the catalog who never bid?
If we include them as controls, are they actually active sellers or dormant accounts (zeros)?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # staggered-adoption/
EDA_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / ".." / "shopping-sessions" / "data"
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "02_vendor_panel_balance.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("EDA Q2: VENDOR PANEL BALANCE", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  How many distinct VENDOR_IDs appear in AUCTIONS_RESULTS vs CATALOG?", f)
        log("  Are there vendors who never bid? Are there bidders not in catalog?", f)
        log("", f)

        log("PURPOSE:", f)
        log("  Define the universe of vendors for the staggered DiD analysis.", f)
        log("  Identify potential selection issues in control group definition.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load data
        # -----------------------------------------------------------------
        log("LOADING DATA", f)
        log("-" * 40, f)

        # Auctions results (bidders)
        ar_path = RAW_DATA_DIR / "raw_sample_auctions_results.parquet"
        if not ar_path.exists():
            log(f"  [ERROR] File not found: {ar_path}", f)
            return

        log("  Loading auctions_results...", f)
        ar = pd.read_parquet(ar_path)
        log(f"    Loaded {len(ar):,} bid rows", f)

        # Catalog (sellers)
        catalog_path = RAW_DATA_DIR / "processed_sample_catalog.parquet"
        if not catalog_path.exists():
            log(f"  [ERROR] File not found: {catalog_path}", f)
            return

        log("  Loading catalog...", f)
        catalog = pd.read_parquet(catalog_path)
        log(f"    Loaded {len(catalog):,} catalog rows", f)

        # Panel data (for activity check)
        panel_path = DATA_DIR / "panel_vendor_week.parquet"
        panel = None
        if panel_path.exists():
            log("  Loading panel_vendor_week...", f)
            panel = pd.read_parquet(panel_path)
            log(f"    Loaded {len(panel):,} panel rows", f)

        log("", f)

        # -----------------------------------------------------------------
        # Extract vendor IDs
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR ID EXTRACTION", f)
        log("-" * 40, f)
        log("", f)

        # From auctions (bidders)
        ar['VENDOR_ID_CLEAN'] = ar['VENDOR_ID'].astype(str).str.lower().str.strip()
        bidder_vendors = set(ar['VENDOR_ID_CLEAN'].unique())
        log(f"  Unique vendors in AUCTIONS_RESULTS (bidders): {len(bidder_vendors):,}", f)

        # From catalog (sellers)
        # Catalog has VENDORS as an array column
        if 'VENDORS' in catalog.columns:
            # Explode vendors array
            log("  Extracting vendors from CATALOG.VENDORS array...", f)

            def extract_vendors(vendors_val):
                if vendors_val is None:
                    return []
                if isinstance(vendors_val, list):
                    return [str(v).lower().strip() for v in vendors_val if v]
                if isinstance(vendors_val, str):
                    # Try to parse as list string
                    import ast
                    try:
                        parsed = ast.literal_eval(vendors_val)
                        if isinstance(parsed, list):
                            return [str(v).lower().strip() for v in parsed if v]
                    except:
                        pass
                    return [vendors_val.lower().strip()]
                return []

            catalog_vendors_list = []
            for vendors in tqdm(catalog['VENDORS'], desc="Extracting vendors"):
                catalog_vendors_list.extend(extract_vendors(vendors))

            catalog_vendors = set(catalog_vendors_list)
            log(f"  Unique vendors in CATALOG: {len(catalog_vendors):,}", f)
        else:
            log("  [WARNING] No VENDORS column found in catalog", f)
            catalog_vendors = set()

        log("", f)

        # -----------------------------------------------------------------
        # Vendor overlap analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR OVERLAP ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        both = bidder_vendors & catalog_vendors
        only_bidders = bidder_vendors - catalog_vendors
        only_catalog = catalog_vendors - bidder_vendors

        log(f"  Vendors in BOTH auctions AND catalog: {len(both):,}", f)
        log(f"  Vendors ONLY in auctions (not in catalog): {len(only_bidders):,}", f)
        log(f"  Vendors ONLY in catalog (never bid): {len(only_catalog):,}", f)
        log("", f)

        if len(bidder_vendors) > 0:
            bidder_coverage = len(both) / len(bidder_vendors) * 100
            log(f"  Bidder coverage in catalog: {bidder_coverage:.2f}%", f)

        if len(catalog_vendors) > 0:
            catalog_coverage = len(both) / len(catalog_vendors) * 100
            log(f"  Catalog coverage in bidders: {catalog_coverage:.2f}%", f)

        log("", f)

        # -----------------------------------------------------------------
        # Activity analysis from panel
        # -----------------------------------------------------------------
        if panel is not None:
            log("=" * 80, f)
            log("VENDOR ACTIVITY ANALYSIS (FROM PANEL)", f)
            log("-" * 40, f)
            log("", f)

            panel['VENDOR_ID_CLEAN'] = panel['VENDOR_ID'].astype(str).str.lower().str.strip()
            panel_vendors = set(panel['VENDOR_ID_CLEAN'].unique())

            log(f"  Unique vendors in panel: {len(panel_vendors):,}", f)
            log("", f)

            # Activity metrics per vendor
            vendor_activity = panel.groupby('VENDOR_ID_CLEAN').agg({
                'wins': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'promoted_gmv': 'sum',
                'week': 'nunique'
            }).reset_index()

            vendor_activity.columns = ['VENDOR_ID_CLEAN', 'total_wins', 'total_impressions',
                                        'total_clicks', 'total_promoted_gmv', 'active_weeks']

            # Categorize vendors
            vendor_activity['has_any_wins'] = vendor_activity['total_wins'] > 0
            vendor_activity['has_any_gmv'] = vendor_activity['total_promoted_gmv'] > 0
            vendor_activity['has_any_impressions'] = vendor_activity['total_impressions'] > 0

            log("  VENDOR ACTIVITY BREAKDOWN:", f)
            log(f"    Vendors with any auction wins: {vendor_activity['has_any_wins'].sum():,} ({vendor_activity['has_any_wins'].mean()*100:.1f}%)", f)
            log(f"    Vendors with any impressions: {vendor_activity['has_any_impressions'].sum():,} ({vendor_activity['has_any_impressions'].mean()*100:.1f}%)", f)
            log(f"    Vendors with any promoted GMV: {vendor_activity['has_any_gmv'].sum():,} ({vendor_activity['has_any_gmv'].mean()*100:.1f}%)", f)
            log("", f)

            # Activity distribution
            log("  WINS DISTRIBUTION:", f)
            wins_quantiles = vendor_activity['total_wins'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
            for q, val in wins_quantiles.items():
                log(f"    P{int(q*100):02d}: {val:,.0f} wins", f)
            log("", f)

            # Active weeks distribution
            log("  ACTIVE WEEKS DISTRIBUTION:", f)
            weeks_dist = vendor_activity['active_weeks'].value_counts().sort_index()
            total_weeks = panel['week'].nunique()
            log(f"    Total weeks in panel: {total_weeks}", f)
            log(f"    Vendors active all {total_weeks} weeks: {(vendor_activity['active_weeks'] == total_weeks).sum():,}", f)
            log(f"    Vendors active only 1 week: {(vendor_activity['active_weeks'] == 1).sum():,}", f)
            log("", f)

            # Zero activity vendors (potential "control" candidates)
            zero_wins = (vendor_activity['total_wins'] == 0).sum()
            zero_impressions = (vendor_activity['total_impressions'] == 0).sum()
            log("  DORMANT VENDORS (potential controls):", f)
            log(f"    Zero wins: {zero_wins:,} ({zero_wins/len(vendor_activity)*100:.1f}%)", f)
            log(f"    Zero impressions: {zero_impressions:,} ({zero_impressions/len(vendor_activity)*100:.1f}%)", f)
            log("", f)

        # -----------------------------------------------------------------
        # Bidder activity distribution
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("BIDDER ACTIVITY DISTRIBUTION", f)
        log("-" * 40, f)
        log("", f)

        # Bids per vendor
        bids_per_vendor = ar.groupby('VENDOR_ID_CLEAN').size().reset_index(name='n_bids')

        log("  BIDS PER VENDOR:", f)
        log(f"    Mean: {bids_per_vendor['n_bids'].mean():,.1f}", f)
        log(f"    Median: {bids_per_vendor['n_bids'].median():,.0f}", f)
        log(f"    Max: {bids_per_vendor['n_bids'].max():,}", f)
        log("", f)

        bid_quantiles = bids_per_vendor['n_bids'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        for q, val in bid_quantiles.items():
            log(f"    P{int(q*100):02d}: {val:,.0f} bids", f)
        log("", f)

        # Win rate per vendor
        ar['is_winner'] = ar['IS_WINNER'].astype(bool)
        wins_per_vendor = ar.groupby('VENDOR_ID_CLEAN').agg({
            'is_winner': ['sum', 'mean', 'count']
        }).reset_index()
        wins_per_vendor.columns = ['VENDOR_ID_CLEAN', 'n_wins', 'win_rate', 'n_bids']

        log("  WIN RATE PER VENDOR:", f)
        log(f"    Mean win rate: {wins_per_vendor['win_rate'].mean()*100:.2f}%", f)
        log(f"    Median win rate: {wins_per_vendor['win_rate'].median()*100:.2f}%", f)
        log("", f)

        winrate_quantiles = wins_per_vendor['win_rate'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        for q, val in winrate_quantiles.items():
            log(f"    P{int(q*100):02d}: {val*100:.2f}% win rate", f)
        log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        log("  KEY FINDINGS:", f)
        log(f"    1. {len(bidder_vendors):,} vendors placed bids in the auction system", f)
        log(f"    2. {len(catalog_vendors):,} vendors have products in the catalog", f)
        log(f"    3. {len(both):,} vendors appear in both (our core analysis sample)", f)
        log("", f)

        if len(only_catalog) > 0:
            log(f"  [NOTE] {len(only_catalog):,} catalog vendors never bid.", f)
            log("    These could serve as 'never-treated' control group if they have sales.", f)
            log("    However, they may be dormant accounts with zero activity.", f)
            log("", f)

        if len(only_bidders) > 0:
            log(f"  [NOTE] {len(only_bidders):,} bidders have no catalog products.", f)
            log("    These may represent vendors who bid but have no active listings.", f)
            log("", f)

        log("  RECOMMENDATION:", f)
        log("    Define analysis universe as vendors who:", f)
        log("      - Have at least 1 auction participation, OR", f)
        log("      - Have at least 1 purchase (to capture 'never-treated' controls)", f)
        log("    Exclude truly dormant vendors with zero activity.", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
