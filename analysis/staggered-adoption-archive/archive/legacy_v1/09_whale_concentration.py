#!/usr/bin/env python3
"""
EDA Q9: Is there "Whale" Concentration?

What percentage of Total GMV and Total Ad Spend is driven by the top 1% of vendors?
If the platform is dominated by a few power sellers, Average Treatment Effects (ATE)
will be skewed. We may need to trim outliers or run weighted regressions.
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
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "09_whale_concentration.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def gini_coefficient(x):
    """Calculate Gini coefficient for a distribution."""
    x = np.array(x)
    x = x[x > 0]  # Exclude zeros for Gini
    if len(x) == 0:
        return 0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n+1) * x)) - (n + 1) * cumx[-1]) / (n * cumx[-1])

def lorenz_curve(x, percentiles=[1, 5, 10, 25, 50]):
    """Calculate Lorenz curve values at given percentiles."""
    x = np.array(x)
    x = np.sort(x)[::-1]  # Sort descending
    cumsum = np.cumsum(x)
    total = cumsum[-1]

    results = {}
    n = len(x)
    for p in percentiles:
        top_n = int(np.ceil(n * p / 100))
        top_share = cumsum[top_n-1] / total * 100 if total > 0 else 0
        results[f'top_{p}pct'] = top_share

    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("EDA Q9: WHALE CONCENTRATION ANALYSIS", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  Is the platform dominated by a few 'whale' vendors?", f)
        log("  If so, ATE may be skewed by outliers.", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. Compute share of GMV/Spend from top 1%, 5%, 10% vendors", f)
        log("  2. Calculate Gini coefficient for concentration", f)
        log("  3. Assess need for trimming or weighted regression", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel data
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_vendor_week.parquet"
        if not panel_path.exists():
            log(f"  [ERROR] File not found: {panel_path}", f)
            return

        panel = pd.read_parquet(panel_path)
        log(f"  Loaded {len(panel):,} vendor-week observations", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Aggregate to vendor level
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR-LEVEL AGGREGATION", f)
        log("-" * 40, f)
        log("", f)

        vendor_totals = panel.groupby('VENDOR_ID').agg({
            'promoted_gmv': 'sum',
            'wins': 'sum',  # Using wins as proxy for spend
            'total_spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum'
        }).reset_index()

        vendor_totals.columns = ['VENDOR_ID', 'total_gmv', 'total_wins',
                                  'total_spend', 'total_impressions', 'total_clicks']

        n_vendors = len(vendor_totals)
        log(f"  Aggregated {n_vendors:,} unique vendors", f)
        log("", f)

        # Overall totals
        overall_gmv = vendor_totals['total_gmv'].sum()
        overall_wins = vendor_totals['total_wins'].sum()
        overall_spend = vendor_totals['total_spend'].sum()
        overall_impressions = vendor_totals['total_impressions'].sum()

        log(f"  OVERALL TOTALS:", f)
        log(f"    Total GMV: ${overall_gmv:,.2f}", f)
        log(f"    Total Wins: {overall_wins:,}", f)
        log(f"    Total Spend: {overall_spend:,.2f}", f)
        log(f"    Total Impressions: {overall_impressions:,.0f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Concentration analysis - GMV
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("GMV CONCENTRATION", f)
        log("-" * 40, f)
        log("", f)

        gmv_lorenz = lorenz_curve(vendor_totals['total_gmv'].values)

        log("  Share of Total GMV by top vendors:", f)
        for key, value in gmv_lorenz.items():
            pct = key.replace('top_', '').replace('pct', '')
            log(f"    Top {pct}%: {value:.2f}% of GMV", f)
        log("", f)

        # Top vendors by GMV
        top_gmv = vendor_totals.nlargest(10, 'total_gmv')
        log("  TOP 10 VENDORS BY GMV:", f)
        log(f"  {'Rank':<6} {'Vendor ID':<36} {'GMV':>15} {'% of Total':>12}", f)
        log(f"  {'-'*6} {'-'*36} {'-'*15} {'-'*12}", f)

        for i, (_, row) in enumerate(top_gmv.iterrows(), 1):
            pct_total = row['total_gmv'] / overall_gmv * 100 if overall_gmv > 0 else 0
            log(f"  {i:<6} {row['VENDOR_ID'][:36]:<36} ${row['total_gmv']:>13,.2f} {pct_total:>11.2f}%", f)

        log("", f)

        # Gini coefficient for GMV
        gmv_gini = gini_coefficient(vendor_totals['total_gmv'].values)
        log(f"  GMV Gini coefficient: {gmv_gini:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Concentration analysis - Spend (Wins)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SPEND (WINS) CONCENTRATION", f)
        log("-" * 40, f)
        log("", f)

        wins_lorenz = lorenz_curve(vendor_totals['total_wins'].values)

        log("  Share of Total Wins by top vendors:", f)
        for key, value in wins_lorenz.items():
            pct = key.replace('top_', '').replace('pct', '')
            log(f"    Top {pct}%: {value:.2f}% of Wins", f)
        log("", f)

        # Top vendors by Wins
        top_wins = vendor_totals.nlargest(10, 'total_wins')
        log("  TOP 10 VENDORS BY WINS:", f)
        log(f"  {'Rank':<6} {'Vendor ID':<36} {'Wins':>12} {'% of Total':>12}", f)
        log(f"  {'-'*6} {'-'*36} {'-'*12} {'-'*12}", f)

        for i, (_, row) in enumerate(top_wins.iterrows(), 1):
            pct_total = row['total_wins'] / overall_wins * 100 if overall_wins > 0 else 0
            log(f"  {i:<6} {row['VENDOR_ID'][:36]:<36} {row['total_wins']:>12,} {pct_total:>11.2f}%", f)

        log("", f)

        # Gini coefficient for Wins
        wins_gini = gini_coefficient(vendor_totals['total_wins'].values)
        log(f"  Wins Gini coefficient: {wins_gini:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Concentration analysis - Impressions
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("IMPRESSIONS CONCENTRATION", f)
        log("-" * 40, f)
        log("", f)

        imp_lorenz = lorenz_curve(vendor_totals['total_impressions'].values)

        log("  Share of Total Impressions by top vendors:", f)
        for key, value in imp_lorenz.items():
            pct = key.replace('top_', '').replace('pct', '')
            log(f"    Top {pct}%: {value:.2f}% of Impressions", f)
        log("", f)

        imp_gini = gini_coefficient(vendor_totals['total_impressions'].values)
        log(f"  Impressions Gini coefficient: {imp_gini:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Distribution statistics
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("DISTRIBUTION STATISTICS", f)
        log("-" * 40, f)
        log("", f)

        for col, label in [('total_gmv', 'GMV'), ('total_wins', 'Wins'), ('total_impressions', 'Impressions')]:
            data = vendor_totals[col]
            nonzero = data[data > 0]

            log(f"  {label}:", f)
            log(f"    Mean: {data.mean():,.2f}", f)
            log(f"    Median: {data.median():,.2f}", f)
            log(f"    Std: {data.std():,.2f}", f)
            log(f"    Mean/Median ratio: {data.mean()/data.median() if data.median() > 0 else 0:.2f}", f)
            log(f"    Skewness: {data.skew():.2f}", f)
            log(f"    % with zero: {(data == 0).sum()/len(data)*100:.1f}%", f)
            log("", f)

        # -----------------------------------------------------------------
        # Vendor size distribution
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR SIZE TIERS", f)
        log("-" * 40, f)
        log("", f)

        # Define tiers based on GMV
        vendor_totals['gmv_tier'] = pd.cut(
            vendor_totals['total_gmv'],
            bins=[-1, 0, 100, 1000, 10000, float('inf')],
            labels=['Zero', 'Micro ($1-100)', 'Small ($100-1K)', 'Medium ($1K-10K)', 'Large ($10K+)']
        )

        tier_summary = vendor_totals.groupby('gmv_tier', observed=True).agg({
            'VENDOR_ID': 'count',
            'total_gmv': 'sum',
            'total_wins': 'sum'
        }).reset_index()
        tier_summary.columns = ['Tier', 'N Vendors', 'Total GMV', 'Total Wins']

        log(f"  {'Tier':<20} {'N Vendors':>12} {'% Vendors':>10} {'GMV':>15} {'% GMV':>10}", f)
        log(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*15} {'-'*10}", f)

        for _, row in tier_summary.iterrows():
            pct_vendors = row['N Vendors'] / n_vendors * 100
            pct_gmv = row['Total GMV'] / overall_gmv * 100 if overall_gmv > 0 else 0
            log(f"  {str(row['Tier']):<20} {row['N Vendors']:>12,} {pct_vendors:>9.1f}% ${row['Total GMV']:>13,.0f} {pct_gmv:>9.1f}%", f)

        log("", f)

        # -----------------------------------------------------------------
        # Pareto analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("PARETO ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # What % of vendors drive 80% of GMV?
        gmv_sorted = vendor_totals.sort_values('total_gmv', ascending=False)
        gmv_cumsum = gmv_sorted['total_gmv'].cumsum()
        threshold_80 = overall_gmv * 0.80
        vendors_for_80 = (gmv_cumsum <= threshold_80).sum() + 1
        pct_vendors_for_80 = vendors_for_80 / n_vendors * 100

        log(f"  80% of GMV comes from top {vendors_for_80:,} vendors ({pct_vendors_for_80:.2f}%)", f)

        threshold_50 = overall_gmv * 0.50
        vendors_for_50 = (gmv_cumsum <= threshold_50).sum() + 1
        pct_vendors_for_50 = vendors_for_50 / n_vendors * 100

        log(f"  50% of GMV comes from top {vendors_for_50:,} vendors ({pct_vendors_for_50:.2f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        top_1_gmv = gmv_lorenz['top_1pct']
        top_10_gmv = gmv_lorenz['top_10pct']

        log("  CONCENTRATION SUMMARY:", f)
        log(f"    Top 1% of vendors account for {top_1_gmv:.1f}% of GMV", f)
        log(f"    Top 10% of vendors account for {top_10_gmv:.1f}% of GMV", f)
        log(f"    GMV Gini: {gmv_gini:.3f}", f)
        log("", f)

        if top_1_gmv > 50:
            log("  [CRITICAL] Extreme concentration - top 1% > 50% of GMV.", f)
            log("  ATE will be dominated by a handful of 'whales'.", f)
            log("  RECOMMENDATIONS:", f)
            log("    1. Trim top 1% or use winsorized outcomes", f)
            log("    2. Report heterogeneous effects by vendor size", f)
            log("    3. Use weighted regression (inverse size weights)", f)
        elif top_10_gmv > 80:
            log("  [WARNING] High concentration - top 10% > 80% of GMV.", f)
            log("  Consider trimming or size-stratified analysis.", f)
        else:
            log("  [OK] Moderate concentration.", f)
            log("  Standard ATE estimation should be representative.", f)

        log("", f)

        log("  GINI INTERPRETATION:", f)
        log(f"    0.00 = Perfect equality", f)
        log(f"    0.50 = Moderate inequality", f)
        log(f"    0.80+ = High inequality (common in business metrics)", f)
        log(f"    Observed: {gmv_gini:.3f}", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
