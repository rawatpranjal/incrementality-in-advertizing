#!/usr/bin/env python3
"""
Table Shapes and Field Coverage

Row counts, column dtypes, null rates, date ranges for all tables.
Single source of truth for data availability.

Usage:
    python 01_table_shapes_and_field_coverage.py --round round1
    python 01_table_shapes_and_field_coverage.py --round round2
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
            'impressions': DATA_BASE / "round1/impressions_all.parquet",
            'clicks': DATA_BASE / "round1/clicks_all.parquet",
            'catalog': DATA_BASE / "round1/catalog_all.parquet",
        }
    elif round_name == "round2":
        return {
            'auctions_results': DATA_BASE / "round2/auctions_results_r2.parquet",
            'auctions_users': DATA_BASE / "round2/auctions_users_r2.parquet",
            'impressions': DATA_BASE / "round2/impressions_r2.parquet",
            'clicks': DATA_BASE / "round2/clicks_r2.parquet",
            'catalog': DATA_BASE / "round2/catalog_r2.parquet",
            'purchases': DATA_BASE / "round2/purchases_r2.parquet",
        }
    else:
        raise ValueError(f"Unknown round: {round_name}")


# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_field_coverage(df, table_name, f):
    """Analyze coverage rates and statistics for each field."""
    log(f"\n{'='*80}", f)
    log(f"TABLE: {table_name}", f)
    log(f"{'='*80}", f)
    log(f"Rows: {len(df):,}", f)
    log(f"Columns: {len(df.columns)}", f)

    log(f"\n{'Field':<30} {'Type':<15} {'Non-null %':<12} {'Unique':<12} {'Sample Values'}", f)
    log(f"{'-'*30} {'-'*15} {'-'*12} {'-'*12} {'-'*40}", f)

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_pct = (df[col].notna().sum() / len(df)) * 100 if len(df) > 0 else 0

        try:
            n_unique = df[col].nunique()
        except (TypeError, ValueError):
            n_unique = -1

        try:
            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = str(sample_vals)[:40] + "..." if len(str(sample_vals)) > 40 else str(sample_vals)
        except:
            sample_str = "[array/list data]"

        unique_str = "ARRAY" if n_unique == -1 else f"{n_unique:,}"
        log(f"{col:<30} {dtype:<15} {non_null_pct:>10.1f}% {unique_str:>10} {sample_str}", f)

    return {
        'rows': len(df),
        'columns': len(df.columns),
    }


def analyze_numeric_stats(df, table_name, f):
    """Detailed stats for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return

    log(f"\n--- Numeric Field Statistics ---", f)
    log(f"{'Field':<25} {'Min':>15} {'Max':>15} {'Mean':>15} {'Std':>15} {'Zeros %':>10}", f)
    log(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*10}", f)

    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        zero_pct = (vals == 0).sum() / len(vals) * 100
        log(f"{col:<25} {vals.min():>15.4g} {vals.max():>15.4g} {vals.mean():>15.4g} {vals.std():>15.4g} {zero_pct:>9.1f}%", f)


def analyze_temporal_range(df, table_name, f):
    """Analyze timestamp columns."""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) == 0:
        return

    log(f"\n--- Temporal Range ---", f)
    for col in datetime_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        log(f"{col}: {vals.min()} to {vals.max()} (span: {vals.max() - vals.min()})", f)


def funnel_summary(data, f):
    """Compute funnel-level summary statistics."""
    log(f"\n{'='*80}", f)
    log(f"FUNNEL SUMMARY", f)
    log(f"{'='*80}", f)

    ar = data.get('auctions_results')
    au = data.get('auctions_users')
    imp = data.get('impressions')
    clicks = data.get('clicks')

    if ar is None or au is None:
        log("ERROR: Required auction data not available", f)
        return

    n_auctions = au['AUCTION_ID'].nunique() if 'AUCTION_ID' in au.columns else len(au)
    n_bids = len(ar)
    n_winners = ar['IS_WINNER'].sum() if 'IS_WINNER' in ar.columns else 0
    n_impressions = len(imp) if imp is not None else 0
    n_clicks = len(clicks) if clicks is not None else 0

    log(f"\n--- Volume Counts ---", f)
    log(f"{'Stage':<35} {'Count':>20}", f)
    log(f"{'-'*35} {'-'*20}", f)
    log(f"{'Auctions':<35} {n_auctions:>20,}", f)
    log(f"{'Bids':<35} {n_bids:>20,}", f)
    log(f"{'Winners':<35} {n_winners:>20,}", f)
    log(f"{'Impressions':<35} {n_impressions:>20,}", f)
    log(f"{'Clicks':<35} {n_clicks:>20,}", f)

    log(f"\n--- Per-Auction Rates ---", f)
    log(f"{'Metric':<40} {'Value':>15}", f)
    log(f"{'-'*40} {'-'*15}", f)
    log(f"{'Bids per auction':<40} {n_bids/n_auctions:>15.1f}", f)
    log(f"{'Winners per auction':<40} {n_winners/n_auctions:>15.1f}", f)
    log(f"{'Impressions per auction':<40} {n_impressions/n_auctions:>15.2f}", f)
    log(f"{'Clicks per auction':<40} {n_clicks/n_auctions:>15.3f}", f)

    log(f"\n--- Funnel Conversion Rates ---", f)
    log(f"{'Transition':<40} {'Rate':>15}", f)
    log(f"{'-'*40} {'-'*15}", f)

    if n_bids > 0:
        log(f"{'Bid -> Winner':<40} {(n_winners/n_bids)*100:>14.1f}%", f)
    if n_winners > 0:
        log(f"{'Winner -> Impression':<40} {(n_impressions/n_winners)*100:>14.1f}%", f)
    if n_impressions > 0:
        log(f"{'Impression -> Click':<40} {(n_clicks/n_impressions)*100:>14.1f}%", f)


def placement_distribution(data, f):
    """Distribution across placements if available."""
    au = data.get('auctions_users')
    if au is None or 'PLACEMENT' not in au.columns:
        return

    log(f"\n{'='*80}", f)
    log(f"PLACEMENT DISTRIBUTION", f)
    log(f"{'='*80}", f)

    placement_counts = au['PLACEMENT'].value_counts()
    placement_pcts = au['PLACEMENT'].value_counts(normalize=True) * 100

    log(f"\n{'Placement':<15} {'Count':>15} {'Percentage':>15}", f)
    log(f"{'-'*15} {'-'*15} {'-'*15}", f)

    for placement in sorted(placement_counts.index):
        count = placement_counts[placement]
        pct = placement_pcts[placement]
        log(f"{str(placement):<15} {count:>15,} {pct:>14.1f}%", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Table shapes and field coverage EDA')
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze (round1 or round2)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"01_table_shapes_and_field_coverage_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log("TABLE SHAPES AND FIELD COVERAGE", f)
        log("=" * 80, f)
        log(f"Round: {args.round}", f)
        log(f"Data source: {DATA_BASE / args.round}", f)

        # Get paths
        paths = get_data_paths(args.round)

        # Load data
        log(f"\n{'='*80}", f)
        log(f"LOADING DATASETS", f)
        log(f"{'='*80}", f)

        data = {}
        for name, filepath in tqdm(paths.items(), desc="Loading parquet files"):
            if filepath.exists():
                data[name] = pd.read_parquet(filepath)
                log(f"Loaded {name}: {len(data[name]):,} rows", f)
            else:
                data[name] = None
                log(f"Missing: {filepath.name}", f)

        # Field-level analysis for each table
        log(f"\n{'='*80}", f)
        log(f"FIELD-LEVEL INVENTORY", f)
        log(f"{'='*80}", f)

        for name, df in tqdm(data.items(), desc="Analyzing tables"):
            if df is not None:
                analyze_field_coverage(df, name.upper(), f)
                analyze_numeric_stats(df, name.upper(), f)
                analyze_temporal_range(df, name.upper(), f)

        # Funnel summary
        funnel_summary(data, f)

        # Placement distribution
        placement_distribution(data, f)

        log(f"\n{'='*80}", f)
        log(f"ANALYSIS COMPLETE", f)
        log(f"{'='*80}", f)
        log(f"Output written to: {output_file}", f)


if __name__ == "__main__":
    main()
