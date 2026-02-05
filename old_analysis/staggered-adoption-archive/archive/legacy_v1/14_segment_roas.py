#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Segment-Level ROAS
Computes ROAS and incremental ROAS by vendor segment.
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
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_FILE = RESULTS_DIR / "14_segment_roas.txt"

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
        log("STAGGERED ADOPTION: SEGMENT-LEVEL ROAS ANALYSIS", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Compute ROAS (Return on Ad Spend) by vendor segment.", f)
        log("  Answer: For which segments is advertising profitable?", f)
        log("", f)

        log("METRICS:", f)
        log("  Observed ROAS = promoted_gmv / total_spend", f)
        log("  Incremental ROAS = ATT * avg_baseline_gmv / avg_spend", f)
        log("  Profitability Threshold: ROAS > 1 (or iROAS > 0)", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load data
        # -----------------------------------------------------------------
        log("LOADING DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_segments.parquet"
        hte_path = DATA_DIR / "hte_by_segment.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])

        log(f"  Panel: {panel.shape}", f)

        if hte_path.exists():
            hte_results = pd.read_parquet(hte_path)
            log(f"  HTE results: {hte_results.shape}", f)
        else:
            hte_results = pd.DataFrame()
            log("  HTE results not found", f)

        log("", f)

        # -----------------------------------------------------------------
        # Overall ROAS
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("OVERALL ROAS", f)
        log("-" * 40, f)
        log("", f)

        # Filter to treated periods
        treated_panel = panel[panel['treated'] == 1]

        if len(treated_panel) > 0:
            total_gmv = treated_panel['promoted_gmv'].sum()
            total_spend = treated_panel['total_spend'].sum()

            if total_spend > 0:
                overall_roas = total_gmv / total_spend
                log(f"  Total promoted GMV (treated): {total_gmv:,.2f}", f)
                log(f"  Total ad spend (treated): {total_spend:,.2f}", f)
                log(f"  Overall ROAS: {overall_roas:.4f}", f)
                log("", f)

                if overall_roas > 1:
                    log(f"  INTERPRETATION: Profitable (${overall_roas:.2f} return per $1 spent)", f)
                else:
                    log(f"  INTERPRETATION: Not profitable ({overall_roas*100:.1f}% return on spend)", f)
            else:
                log("  No spend data available", f)
                overall_roas = np.nan
        else:
            log("  No treated observations", f)
            overall_roas = np.nan

        log("", f)

        # -----------------------------------------------------------------
        # ROAS by Persona
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ROAS BY PERSONA", f)
        log("-" * 40, f)
        log("", f)

        roas_results = []

        if 'persona' in panel.columns:
            log(f"  {'Persona':<25} {'GMV':<15} {'Spend':<15} {'ROAS':<10} {'Profitable?':<12}", f)
            log(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10} {'-'*12}", f)

            for persona in panel['persona'].unique():
                seg_panel = treated_panel[treated_panel['persona'] == persona]

                if len(seg_panel) == 0:
                    continue

                gmv = seg_panel['promoted_gmv'].sum()
                spend = seg_panel['total_spend'].sum()

                if spend > 0:
                    roas = gmv / spend
                    profitable = "Yes" if roas > 1 else "No"
                else:
                    roas = np.nan
                    profitable = "N/A"

                log(f"  {persona:<25} {gmv:<15,.2f} {spend:<15,.2f} {roas:<10.4f} {profitable:<12}", f)

                roas_results.append({
                    'segment': persona,
                    'segment_type': 'persona',
                    'promoted_gmv': gmv,
                    'total_spend': spend,
                    'roas': roas,
                    'profitable': roas > 1 if not np.isnan(roas) else None
                })

        log("", f)

        # -----------------------------------------------------------------
        # ROAS by Activity Quartile
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ROAS BY ACTIVITY QUARTILE", f)
        log("-" * 40, f)
        log("", f)

        if 'activity_quartile' in panel.columns:
            log(f"  {'Quartile':<25} {'GMV':<15} {'Spend':<15} {'ROAS':<10} {'Profitable?':<12}", f)
            log(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10} {'-'*12}", f)

            for quartile in sorted(panel['activity_quartile'].unique()):
                seg_panel = treated_panel[treated_panel['activity_quartile'] == quartile]

                if len(seg_panel) == 0:
                    continue

                gmv = seg_panel['promoted_gmv'].sum()
                spend = seg_panel['total_spend'].sum()

                if spend > 0:
                    roas = gmv / spend
                    profitable = "Yes" if roas > 1 else "No"
                else:
                    roas = np.nan
                    profitable = "N/A"

                log(f"  {quartile:<25} {gmv:<15,.2f} {spend:<15,.2f} {roas:<10.4f} {profitable:<12}", f)

                roas_results.append({
                    'segment': quartile,
                    'segment_type': 'activity_quartile',
                    'promoted_gmv': gmv,
                    'total_spend': spend,
                    'roas': roas,
                    'profitable': roas > 1 if not np.isnan(roas) else None
                })

        log("", f)

        # -----------------------------------------------------------------
        # ROAS by Price Quartile
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ROAS BY PRICE QUARTILE", f)
        log("-" * 40, f)
        log("", f)

        if 'price_quartile' in panel.columns:
            log(f"  {'Quartile':<25} {'GMV':<15} {'Spend':<15} {'ROAS':<10} {'Profitable?':<12}", f)
            log(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10} {'-'*12}", f)

            for quartile in sorted(panel['price_quartile'].unique()):
                seg_panel = treated_panel[treated_panel['price_quartile'] == quartile]

                if len(seg_panel) == 0:
                    continue

                gmv = seg_panel['promoted_gmv'].sum()
                spend = seg_panel['total_spend'].sum()

                if spend > 0:
                    roas = gmv / spend
                    profitable = "Yes" if roas > 1 else "No"
                else:
                    roas = np.nan
                    profitable = "N/A"

                log(f"  {quartile:<25} {gmv:<15,.2f} {spend:<15,.2f} {roas:<10.4f} {profitable:<12}", f)

                roas_results.append({
                    'segment': quartile,
                    'segment_type': 'price_quartile',
                    'promoted_gmv': gmv,
                    'total_spend': spend,
                    'roas': roas,
                    'profitable': roas > 1 if not np.isnan(roas) else None
                })

        log("", f)

        # -----------------------------------------------------------------
        # Incremental ROAS (using HTE estimates)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INCREMENTAL ROAS (iROAS)", f)
        log("-" * 40, f)
        log("", f)

        log("  Definition:", f)
        log("    iROAS = (ATT * avg_gmv) / avg_spend", f)
        log("    ATT from segment-level HTE analysis", f)
        log("    Measures INCREMENTAL return, not total observed return", f)
        log("", f)

        if len(hte_results) > 0:
            # Compute average GMV and spend by segment
            iroas_results = []

            for _, hte_row in hte_results.iterrows():
                segment = hte_row['segment']
                seg_type = hte_row['segment_type']
                att = hte_row['att']

                if np.isnan(att):
                    continue

                # Get segment data
                if seg_type == 'persona':
                    seg_panel = treated_panel[treated_panel['persona'] == segment]
                elif seg_type == 'activity_quartile':
                    seg_panel = treated_panel[treated_panel['activity_quartile'] == segment]
                elif seg_type == 'price_quartile':
                    seg_panel = treated_panel[treated_panel['price_quartile'] == segment]
                else:
                    continue

                if len(seg_panel) == 0:
                    continue

                # Average values
                avg_gmv = seg_panel['promoted_gmv'].mean()
                avg_spend = seg_panel['total_spend'].mean()

                if avg_spend > 0:
                    # ATT is in log units, convert to level effect
                    # If ATT = d log(GMV), then % change = exp(ATT) - 1
                    pct_effect = np.exp(att) - 1
                    incremental_gmv = avg_gmv * pct_effect
                    iroas = incremental_gmv / avg_spend

                    iroas_results.append({
                        'segment': segment,
                        'segment_type': seg_type,
                        'att': att,
                        'pct_effect': pct_effect * 100,
                        'avg_gmv': avg_gmv,
                        'avg_spend': avg_spend,
                        'incremental_gmv': incremental_gmv,
                        'iroas': iroas
                    })

            if len(iroas_results) > 0:
                log(f"  {'Segment':<25} {'ATT':<10} {'% Effect':<10} {'Incr GMV':<12} {'iROAS':<10}", f)
                log(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*10}", f)

                for result in sorted(iroas_results, key=lambda x: x['iroas'], reverse=True):
                    log(f"  {result['segment']:<25} {result['att']:<10.6f} {result['pct_effect']:<10.2f} {result['incremental_gmv']:<12.4f} {result['iroas']:<10.4f}", f)

                log("", f)

                # Summary
                profitable_segments = [r for r in iroas_results if r['iroas'] > 0]
                log(f"  Segments with positive iROAS: {len(profitable_segments)}/{len(iroas_results)}", f)

                if len(profitable_segments) > 0:
                    best = max(iroas_results, key=lambda x: x['iroas'])
                    log(f"  Best segment: {best['segment']} (iROAS = {best['iroas']:.4f})", f)

        else:
            log("  HTE results not available for iROAS calculation", f)

        log("", f)

        # -----------------------------------------------------------------
        # Vendor-level ROAS distribution
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR-LEVEL ROAS DISTRIBUTION", f)
        log("-" * 40, f)
        log("", f)

        # Compute ROAS per vendor (treated periods only)
        vendor_roas = treated_panel.groupby('VENDOR_ID').agg({
            'promoted_gmv': 'sum',
            'total_spend': 'sum',
            'persona': 'first'
        }).reset_index()

        vendor_roas['roas'] = np.where(
            vendor_roas['total_spend'] > 0,
            vendor_roas['promoted_gmv'] / vendor_roas['total_spend'],
            np.nan
        )

        vendor_roas = vendor_roas[vendor_roas['roas'].notna()]

        if len(vendor_roas) > 0:
            log(f"  Vendors with ROAS data: {len(vendor_roas):,}", f)
            log("", f)

            log("  ROAS Distribution:", f)
            quantiles = vendor_roas['roas'].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
            for q, val in quantiles.items():
                log(f"    P{int(q*100):02d}: {val:.4f}", f)
            log("", f)

            # Profitability breakdown
            profitable = (vendor_roas['roas'] > 1).sum()
            unprofitable = (vendor_roas['roas'] <= 1).sum()

            log(f"  Profitable vendors (ROAS > 1): {profitable:,} ({profitable/len(vendor_roas)*100:.1f}%)", f)
            log(f"  Unprofitable vendors (ROAS <= 1): {unprofitable:,} ({unprofitable/len(vendor_roas)*100:.1f}%)", f)
            log("", f)

            # By persona
            if 'persona' in vendor_roas.columns:
                log("  Profitability by Persona:", f)
                log(f"  {'Persona':<25} {'N Vendors':<12} {'Profitable %':<15} {'Median ROAS':<12}", f)
                log(f"  {'-'*25} {'-'*12} {'-'*15} {'-'*12}", f)

                for persona in vendor_roas['persona'].unique():
                    persona_vendors = vendor_roas[vendor_roas['persona'] == persona]
                    n_vendors = len(persona_vendors)
                    pct_profitable = (persona_vendors['roas'] > 1).mean() * 100
                    median_roas = persona_vendors['roas'].median()

                    log(f"  {persona:<25} {n_vendors:<12,} {pct_profitable:<15.1f} {median_roas:<12.4f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Summary and recommendations
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SUMMARY AND RECOMMENDATIONS", f)
        log("-" * 40, f)
        log("", f)

        roas_df = pd.DataFrame(roas_results)

        if len(roas_df) > 0:
            profitable_segments = roas_df[roas_df['profitable'] == True]
            unprofitable_segments = roas_df[roas_df['profitable'] == False]

            log("  PROFITABLE SEGMENTS (ROAS > 1):", f)
            if len(profitable_segments) > 0:
                for _, row in profitable_segments.sort_values('roas', ascending=False).iterrows():
                    log(f"    - {row['segment']}: ROAS = {row['roas']:.4f}", f)
            else:
                log("    None", f)

            log("", f)

            log("  UNPROFITABLE SEGMENTS (ROAS <= 1):", f)
            if len(unprofitable_segments) > 0:
                for _, row in unprofitable_segments.sort_values('roas', ascending=False).iterrows():
                    log(f"    - {row['segment']}: ROAS = {row['roas']:.4f}", f)
            else:
                log("    None", f)

            log("", f)

            log("  RECOMMENDATIONS:", f)
            log("    1. Focus advertising budget on segments with ROAS > 1", f)
            log("    2. Consider reducing or optimizing spend on unprofitable segments", f)
            log("    3. Note: Observed ROAS includes selection effects;", f)
            log("       incremental ROAS (iROAS) is the true causal metric", f)

        # Save results
        roas_df.to_parquet(DATA_DIR / "segment_roas.parquet", index=False)
        log("", f)
        log(f"  Saved ROAS results to: {DATA_DIR / 'segment_roas.parquet'}", f)

        log("", f)
        log("=" * 80, f)
        log("SEGMENT ROAS ANALYSIS COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
